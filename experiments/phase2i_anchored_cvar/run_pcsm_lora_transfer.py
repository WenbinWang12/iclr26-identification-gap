"""Zero target-task recipe/hyperparameter/checkpoint tuning transfer.

This runner applies the exact BoolQA-derived PCSM v1 recipe to QQP or MultiRC.
It has a distinct artifact format and makes no claim that target-task
development data were used to tune learning rate, schedule, preprocessing,
loss, or gates.  Freshness histories are supplied by CLI and content-addressed
inside every durable lock.  An optional task protocol registry can later bind a
reviewed task to an approved freshness SHA set without changing this runner.

The primary endpoint is raw official free-generation natural-prior EM gain.
Identically calibrated base/post decisions and margin ROC-AUC remain secondary
guards.  Confirm is sealed until both internal and development gates pass and
is then available through an exclusive one-shot confirmation entry point.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

try:
    from . import run_acquisition_rescue as rescue
    from . import run_bap_lora_rescue as bap
    from . import run_pcsm_lora_rescue as pcsm
    from .order4_data import OfficialExample, TASK_BY_NAME, load_official_examples
except ImportError:  # pragma: no cover - direct script execution.
    import run_acquisition_rescue as rescue  # type: ignore
    import run_bap_lora_rescue as bap  # type: ignore
    import run_pcsm_lora_rescue as pcsm  # type: ignore
    from order4_data import (  # type: ignore
        OfficialExample,
        TASK_BY_NAME,
        load_official_examples,
    )


FORMAT_VERSION = "phase2i.binary-pcsm-lora-transfer.v1"
TASK_PROTOCOL_REGISTRY_FORMAT = "phase2i.pcsm-transfer-task-registry.v1"
TRANSFER_TASKS = ("QQP", "MultiRC")
BINARY_LABELS = pcsm.BINARY_LABELS
OBJECTIVE = pcsm.OBJECTIVE

FROZEN_PCSM_SHA256 = "eef0666c5627254cb6a5d835ed0ab75bfbdd1202bc6bfa1b79d7cbcc22eb2e9b"
FROZEN_BAP_SHA256 = "4d1e4fa70954682d7bdb7ad01bd5d4bc592139a12af7da86b0c8af9de235a5b5"
FROZEN_SHARED_SHA256 = "8d1250948b6a4c5941124fe955aa01d50222cbb2c846f789e21b0a021b9de95e"
FROZEN_ORDER4_DATA_SHA256 = "8ca0bc833626528ce0bd00d3068d085005f6388d2238617f59ab50a67c72c571"
FROZEN_OBJECTIVES_SHA256 = "7fce8c71066c64801fb319c9d37728f5c512f44c40f42cdf84dc9005989eb770"
FROZEN_T5_RANK_BANK_SHA256 = "cd1596c1292a1aa6465a408e4582be755a5be3413bdfa4e8f89a05ea6d5be59d"

FORMAL_LEARNING_RATE = 3e-4
FORMAL_FIT_PER_CLASS = 384
FORMAL_CALIBRATION_PER_CLASS = 128
FORMAL_TUNE_PER_CLASS = 128
FORMAL_CONFIRM_PER_CLASS = 32
FORMAL_OPTIMIZER_STEPS = 384
FORMAL_BATCH_SIZE = 4
FORMAL_SEED = 43
FORMAL_MAX_SOURCE_LENGTH = 512
FORMAL_MAX_TARGET_LENGTH = 50
FORMAL_TRUNCATION_MODE = "field_aware"


@dataclass(frozen=True)
class TransferConfig:
    data_root: str
    model_path: str
    output_root: str
    task_name: str
    learning_rate: float = FORMAL_LEARNING_RATE
    fit_per_class: int = FORMAL_FIT_PER_CLASS
    calibration_per_class: int = FORMAL_CALIBRATION_PER_CLASS
    tune_audit_per_class: int = FORMAL_TUNE_PER_CLASS
    confirm_audit_per_class: int = FORMAL_CONFIRM_PER_CLASS
    optimizer_steps: int = FORMAL_OPTIMIZER_STEPS
    seed: int = FORMAL_SEED
    batch_size: int = FORMAL_BATCH_SIZE
    max_source_length: int = FORMAL_MAX_SOURCE_LENGTH
    max_target_length: int = FORMAL_MAX_TARGET_LENGTH
    truncation_mode: str = FORMAL_TRUNCATION_MODE
    min_raw_gain: float = 0.05
    min_valid_label_rate: float = 0.99
    max_class_drop: float = 0.05
    tiny_random_model: bool = False
    dry_run: bool = False
    confirm_selection: str | None = None
    device: str = "cpu"
    fresh_confirm_against_manifests: tuple[str, ...] = ()
    task_protocol_registry: str | None = None

    def __post_init__(self) -> None:
        if self.task_name not in TRANSFER_TASKS:
            raise ValueError(
                f"PCSM transfer task must be one of {TRANSFER_TASKS}, got {self.task_name!r}"
            )
        labels = TASK_BY_NAME[self.task_name].labels
        if len(labels) != 2 or set(labels) != set(BINARY_LABELS):
            raise ValueError("PCSM transfer v1 requires binary True/False labels")
        integers = {
            "fit_per_class": self.fit_per_class,
            "calibration_per_class": self.calibration_per_class,
            "tune_audit_per_class": self.tune_audit_per_class,
            "confirm_audit_per_class": self.confirm_audit_per_class,
            "optimizer_steps": self.optimizer_steps,
            "batch_size": self.batch_size,
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integers.values()
        ):
            raise ValueError(f"positive integer transfer config required: {integers}")
        if self.batch_size % 2:
            raise ValueError("paired transfer batches require an even batch_size")
        if self.fit_per_class % (self.batch_size // 2):
            raise ValueError("fit_per_class must divide into paired half-batches")
        if self.optimizer_steps * self.batch_size % (2 * self.fit_per_class):
            raise ValueError("optimizer steps must make complete fit cycles")
        finite = {
            "learning_rate": self.learning_rate,
            "min_raw_gain": self.min_raw_gain,
            "min_valid_label_rate": self.min_valid_label_rate,
            "max_class_drop": self.max_class_drop,
        }
        if any(not math.isfinite(value) for value in finite.values()):
            raise ValueError(f"finite transfer constants required: {finite}")
        if self.learning_rate <= 0 or self.min_raw_gain < 0:
            raise ValueError("learning rate must be positive and gain nonnegative")
        if not 0 < self.min_valid_label_rate <= 1:
            raise ValueError("minimum valid-label rate must be in (0,1]")
        if not 0 <= self.max_class_drop < 1:
            raise ValueError("max_class_drop must be in [0,1)")
        if self.truncation_mode not in bap.TRUNCATION_MODES:
            raise ValueError(f"unsupported truncation mode {self.truncation_mode!r}")

        if not self.tiny_random_model:
            fixed = {
                "learning_rate": (self.learning_rate, FORMAL_LEARNING_RATE),
                "fit_per_class": (self.fit_per_class, FORMAL_FIT_PER_CLASS),
                "calibration_per_class": (
                    self.calibration_per_class,
                    FORMAL_CALIBRATION_PER_CLASS,
                ),
                "tune_audit_per_class": (
                    self.tune_audit_per_class,
                    FORMAL_TUNE_PER_CLASS,
                ),
                "confirm_audit_per_class": (
                    self.confirm_audit_per_class,
                    FORMAL_CONFIRM_PER_CLASS,
                ),
                "optimizer_steps": (self.optimizer_steps, FORMAL_OPTIMIZER_STEPS),
                "seed": (self.seed, FORMAL_SEED),
                "batch_size": (self.batch_size, FORMAL_BATCH_SIZE),
                "max_source_length": (
                    self.max_source_length,
                    FORMAL_MAX_SOURCE_LENGTH,
                ),
                "max_target_length": (
                    self.max_target_length,
                    FORMAL_MAX_TARGET_LENGTH,
                ),
                "truncation_mode": (
                    self.truncation_mode,
                    FORMAL_TRUNCATION_MODE,
                ),
                "min_raw_gain": (self.min_raw_gain, 0.05),
                "min_valid_label_rate": (self.min_valid_label_rate, 0.99),
                "max_class_drop": (self.max_class_drop, 0.05),
            }
            drift = [name for name, (actual, expected) in fixed.items() if actual != expected]
            if drift:
                raise ValueError(
                    "target-task transfer recipe is frozen; drifted fields: "
                    + ", ".join(drift)
                )
            if not self.fresh_confirm_against_manifests:
                raise ValueError("formal transfer requires at least one freshness source")
            resolved_sources = [
                str(Path(value).expanduser().resolve())
                for value in self.fresh_confirm_against_manifests
            ]
            if len(set(resolved_sources)) != len(resolved_sources):
                raise ValueError("formal transfer freshness source paths must be distinct")
            output = Path(self.output_root).expanduser().resolve()
            protected_paths = [
                ("data_root", Path(self.data_root).expanduser().resolve()),
                ("model_path", Path(self.model_path).expanduser().resolve()),
            ]
            for name, protected in protected_paths:
                if (
                    output == protected
                    or output.is_relative_to(protected)
                    or protected.is_relative_to(output)
                ):
                    raise ValueError(
                        f"output_root must not overlap locked {name}: {output} vs {protected}"
                    )
            external_inputs = [
                *(Path(value).expanduser().resolve() for value in self.fresh_confirm_against_manifests),
                *(
                    ()
                    if self.task_protocol_registry is None
                    else (Path(self.task_protocol_registry).expanduser().resolve(),)
                ),
            ]
            if any(path == output or path.is_relative_to(output) for path in external_inputs):
                raise ValueError("output_root must not contain freshness/registry inputs")

    @property
    def update_per_class(self) -> int:
        return self.fit_per_class + self.calibration_per_class

    @property
    def fit_cycles(self) -> int:
        return self.optimizer_steps * self.batch_size // (2 * self.fit_per_class)


def _dependency_fingerprints() -> dict[str, Any]:
    module_dir = Path(__file__).resolve().parent
    expected = {
        "pcsm_v1": (Path(pcsm.__file__).resolve(), FROZEN_PCSM_SHA256),
        "bap_v1": (Path(bap.__file__).resolve(), FROZEN_BAP_SHA256),
        "shared_rescue": (Path(rescue.__file__).resolve(), FROZEN_SHARED_SHA256),
        "order4_data": (module_dir / "order4_data.py", FROZEN_ORDER4_DATA_SHA256),
        "objectives": (module_dir / "objectives.py", FROZEN_OBJECTIVES_SHA256),
        "t5_rank_bank": (module_dir / "t5_rank_bank.py", FROZEN_T5_RANK_BANK_SHA256),
    }
    result: dict[str, Any] = {}
    for name, (path, expected_sha) in expected.items():
        current_sha = rescue._sha256_file(path).lower()
        if current_sha != expected_sha:
            raise ValueError(f"frozen transfer dependency {name} SHA-256 changed")
        result[name] = {
            "path": str(path),
            "sha256": current_sha,
            "expected_sha256": expected_sha,
        }
    runner_path = Path(__file__).resolve()
    result["transfer_runner"] = {
        "path": str(runner_path),
        "sha256": rescue._sha256_file(runner_path).lower(),
        "binding": "selection-time implementation identity; rechecked before confirm",
    }
    return result


def _recipe_payload() -> dict[str, Any]:
    return {
        "origin": "BoolQA PCSM-LoRA v1 positive-control recipe",
        "origin_runner_sha256": FROZEN_PCSM_SHA256,
        "target_task_recipe_hparam_checkpoint_tuning": False,
        "target_task_gradient_adaptation": True,
        "target_task_internal_threshold_calibration": True,
        "objective": {
            "name": OBJECTIVE,
            "per_example_loss": "sequence-mean target-token NLL of observed label",
            "importance_weight": "pinned target natural prior pi(y) / paired sampler q(y)=0.5",
        },
        "learning_rate": FORMAL_LEARNING_RATE,
        "fit_per_class": FORMAL_FIT_PER_CLASS,
        "internal_calibration_per_class": FORMAL_CALIBRATION_PER_CLASS,
        "development_per_class": FORMAL_TUNE_PER_CLASS,
        "confirm_per_class": FORMAL_CONFIRM_PER_CLASS,
        "optimizer_steps": FORMAL_OPTIMIZER_STEPS,
        "eligible_checkpoint_steps": [FORMAL_OPTIMIZER_STEPS],
        "batch_size": FORMAL_BATCH_SIZE,
        "fit_cycles": 2,
        "seed": FORMAL_SEED,
        "sampler": "paired_fit_batches; equal True/False per batch",
        "gradient_forward_mode": "eval with autograd; dropout disabled",
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "max_source_length": FORMAL_MAX_SOURCE_LENGTH,
        "max_target_length": FORMAL_MAX_TARGET_LENGTH,
        "truncation_mode": FORMAL_TRUNCATION_MODE,
        "primary_gate": {
            "raw_natural_prior_gain": "strictly > 0.05",
            "post_valid_label_rate": ">= 0.99",
            "raw_minimum_per_class_delta": ">= -0.05",
        },
        "secondary_gate": {
            "same_calibration_gain": "strictly > 0",
            "calibrated_minimum_per_class_delta": ">= -0.05",
            "margin_roc_auc_delta": "strictly > 0",
        },
        "threshold_rule": (
            "fit base/post independently with the identical exhaustive target-internal "
            "calibration fitter; freeze both before development/confirm"
        ),
        "split": {
            "unit": "semantic_group",
            "group_key_version": rescue.SEMANTIC_GROUP_KEY_VERSION,
            "selection_algorithm_version": rescue.SEMANTIC_SELECTION_ALGORITHM_VERSION,
            "internal_split_seed_tag": "bap-internal-calibration",
        },
    }


def _protocol_payload(config: TransferConfig) -> dict[str, Any]:
    recipe = _recipe_payload()
    return {
        "format": FORMAT_VERSION,
        "claim_scope": "cross-task train-only acquisition transfer; not continual retention",
        "method": "PCSM-LoRA frozen-recipe cross-task transfer",
        "source_recipe": recipe,
        "source_recipe_sha256": bap._sha256_json(recipe),
        "transfer_task": config.task_name,
        "target_task_recipe_hparam_checkpoint_tuning": False,
        "target_task_gradient_adaptation": True,
        "target_task_internal_threshold_calibration": True,
        "selection_eligible_arm_count": 1,
        "learning_rate_candidate_count": 1,
        "checkpoint_candidate_count": 1,
        "freshness_policy": {
            "binding": "CLI paths plus loader-verified path/bytes/SHA metadata",
            "cli_source_paths": list(config.fresh_confirm_against_manifests),
            "minimum_source_count": 1,
            "semantic_group_required": True,
            "hardcoded_unreviewed_target_paths": False,
            "task_protocol_registry": config.task_protocol_registry,
        },
        "partition_roles": {
            "fit": "gradient only",
            "internal_calibration": "first qualification and threshold fitting",
            "development_tune": "second qualification; opened only after internal pass",
            "confirm": "sealed until durable two-gate selection and one-shot claim",
        },
        "confirmation": {
            "primary_endpoint": "same raw official gate",
            "calibrated_and_auc": "reported with internal-frozen thresholds; not retuned",
        },
        "development_independence_note": (
            "recipe was fixed on BoolQA before this target-task trajectory; "
            "target development cannot change any recipe field"
        ),
    }


def _as_bap_config(config: TransferConfig) -> bap.BAPConfig:
    # PCSM's projection is deliberately duck-typed and contains no BoolQA-only
    # logic; BAPConfig validates the target has exactly True/False labels.
    return pcsm._as_bap_config(config)  # type: ignore[arg-type]


def _model_config(config: TransferConfig) -> rescue.RescueConfig:
    return pcsm._model_config(config)  # type: ignore[arg-type]


def _internal_manifest(split: bap.BAPInternalSplit) -> dict[str, Any]:
    payload = bap.bap_internal_split_manifest(split)
    return {
        **payload,
        "format": FORMAT_VERSION,
        "constructor": "frozen BAP group-atomic internal split",
    }


def _validate_freshness_sources(
    sources: Sequence[Mapping[str, Any]],
    outer: rescue.RescueSplit,
    *,
    config: TransferConfig,
) -> dict[str, Any]:
    if config.tiny_random_model and not config.fresh_confirm_against_manifests:
        return {
            "status": "tiny_test_without_freshness_sources",
            "source_count": 0,
            "semantic_group_strict": True,
        }
    if not sources or len(sources) != len(config.fresh_confirm_against_manifests):
        raise ValueError("transfer freshness loader/source count mismatch")
    configured = [
        str(Path(value).expanduser().resolve())
        for value in config.fresh_confirm_against_manifests
    ]
    returned = [
        str(Path(str(source.get("path", ""))).expanduser().resolve())
        for source in sources
    ]
    if configured != returned:
        raise ValueError("transfer freshness source order/path changed")
    if any(source.get("task") != config.task_name for source in sources):
        raise ValueError("transfer freshness source escaped the target task")
    source_hashes = [str(source.get("sha256", "")).lower() for source in sources]
    if any(len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value) for value in source_hashes):
        raise ValueError("transfer freshness source lacks a SHA-256 fingerprint")
    if len(set(source_hashes)) != len(source_hashes):
        raise ValueError("transfer freshness source contents must be distinct")
    source_file_bindings = []
    for source, expected_path, expected_sha in zip(
        sources, returned, source_hashes
    ):
        path = Path(expected_path)
        if not path.is_file() or rescue._sha256_file(path).lower() != expected_sha:
            raise ValueError("transfer freshness source file changed during validation")
        source_file_bindings.append(
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": expected_sha,
                "loader_metadata_sha256": bap._sha256_json(dict(source)),
            }
        )
    if outer.split_unit != "semantic_group":
        raise ValueError("transfer outer split is not semantic-group strict")
    if outer.group_key_version != rescue.SEMANTIC_GROUP_KEY_VERSION:
        raise ValueError("transfer semantic group-key version changed")
    if outer.selection_algorithm_version != rescue.SEMANTIC_SELECTION_ALGORITHM_VERSION:
        raise ValueError("transfer semantic selection algorithm changed")
    assert outer.group_key_by_example_id is not None
    confirm_groups = {
        outer.group_key_by_example_id[item.example_id] for item in outer.confirm_audit
    }
    prior_overlap = len(set(outer.prior_touched_group_keys) & confirm_groups)
    if prior_overlap:
        raise ValueError("fresh transfer confirm overlaps prior-touched semantic groups")
    type_counts = {
        "access_only": sum(
            source.get("format") == rescue.ACCESS_ONLY_FRESHNESS_FORMAT
            for source in sources
        ),
        "legacy_split": sum("format" not in source for source in sources),
    }
    if sum(type_counts.values()) != len(sources):
        raise ValueError("transfer freshness source has an unsupported format")
    return {
        "status": "verified",
        "source_count": len(sources),
        "ordered_source_sha256s": source_hashes,
        "source_file_bindings": source_file_bindings,
        "ordered_source_metadata_sha256": bap._sha256_json(
            [dict(item) for item in sources]
        ),
        "source_type_counts": type_counts,
        "semantic_group_strict": True,
        "prior_touched_confirm_semantic_group_overlap": prior_overlap,
    }


def _build_splits(
    examples: Sequence[OfficialExample], *, config: TransferConfig
) -> tuple[
    rescue.RescueSplit,
    bap.BAPInternalSplit,
    tuple[Mapping[str, Any], ...],
    dict[str, Any],
]:
    bap_config = _as_bap_config(config)
    outer, sources = bap._build_outer_split(examples, config=bap_config)
    internal = bap.build_bap_internal_split(outer, config=bap_config)
    audit = _validate_freshness_sources(sources, outer, config=config)
    return outer, internal, sources, audit


def _load_task_protocol_registry(
    config: TransferConfig, sources: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Optionally bind a target task to a later reviewed freshness registry."""

    if config.task_protocol_registry is None:
        return {
            "status": "not_supplied",
            "required_for_this_run": False,
            "run_local_cli_source_binding_only": True,
        }
    path = Path(config.task_protocol_registry).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing transfer task protocol registry: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load transfer task protocol registry") from error
    if not isinstance(payload, Mapping) or payload.get("format") != TASK_PROTOCOL_REGISTRY_FORMAT:
        raise ValueError("unsupported transfer task protocol registry format")
    tasks = payload.get("tasks")
    entry = tasks.get(config.task_name) if isinstance(tasks, Mapping) else None
    if not isinstance(entry, Mapping):
        raise ValueError("transfer registry lacks the requested task entry")
    recipe_sha = bap._sha256_json(_recipe_payload())
    if entry.get("source_recipe_sha256") != recipe_sha:
        raise ValueError("transfer registry recipe SHA-256 differs from frozen recipe")
    if entry.get("semantic_group_required") is not True:
        raise ValueError("transfer registry does not require semantic-group splitting")
    expected_shas = entry.get("freshness_source_sha256s")
    if (
        not isinstance(expected_shas, list)
        or not expected_shas
        or any(not isinstance(value, str) for value in expected_shas)
    ):
        raise ValueError("transfer registry freshness SHA set is malformed")
    actual_shas = sorted(str(source["sha256"]).lower() for source in sources)
    if sorted(value.lower() for value in expected_shas) != actual_shas:
        raise ValueError("transfer registry freshness SHA set does not match CLI sources")
    return {
        "status": "verified",
        "required_for_this_run": True,
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": rescue._sha256_file(path),
        "format": TASK_PROTOCOL_REGISTRY_FORMAT,
        "task": config.task_name,
        "task_entry_sha256": bap._sha256_json(entry),
        "source_recipe_sha256": recipe_sha,
        "freshness_source_sha256s": actual_shas,
        "semantic_group_required": True,
    }


def _protocol_lock_payload(
    config: TransferConfig,
    *,
    sources: Sequence[Mapping[str, Any]],
    freshness_audit: Mapping[str, Any],
    registry_binding: Mapping[str, Any],
    dependency_fingerprints: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "format": FORMAT_VERSION,
        "lock_scope": "run-local pre-model/pre-gradient; not external preregistration",
        "external_preregistration_claimed": False,
        "scientific_config": asdict(config),
        "protocol": _protocol_payload(config),
        "resolved_freshness_sources": [dict(item) for item in sources],
        "freshness_audit": dict(freshness_audit),
        "task_protocol_registry_binding": dict(registry_binding),
        "frozen_dependency_fingerprints": dict(dependency_fingerprints),
        "pilot_inventory": {
            "source_task": "BoolQA",
            "target_task": config.task_name,
            "target_task_specific_candidates": 1,
            "target_task_specific_hyperparameter_or_checkpoint_scan": False,
            "recipe_fixed_before_target_training": True,
        },
    }
    return {**payload, "payload_sha256": bap._sha256_json(payload)}


def _score_partition(
    bank: Any,
    tokenizer: Any,
    examples: Sequence[OfficialExample],
    *,
    config: TransferConfig,
    natural_prior: Mapping[str, float],
) -> tuple[dict[str, Any], list[float]]:
    return pcsm._score_partition(  # type: ignore[arg-type]
        bank,
        tokenizer,
        examples,
        config=config,
        natural_prior=natural_prior,
    )


def _train_fixed_trajectory(
    bank: Any,
    tokenizer: Any,
    fit_examples: Sequence[OfficialExample],
    *,
    config: TransferConfig,
    natural_prior: Mapping[str, float],
) -> dict[str, Any]:
    return pcsm._train_fixed_trajectory(  # type: ignore[arg-type]
        bank,
        tokenizer,
        fit_examples,
        config=config,
        natural_prior=natural_prior,
    )


def _dry_run_payload(
    config: TransferConfig,
    outer: rescue.RescueSplit,
    internal: bap.BAPInternalSplit,
    distribution: Mapping[str, Any],
    *,
    freshness_audit: Mapping[str, Any],
    registry_binding: Mapping[str, Any],
    dependencies: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": FORMAT_VERSION,
        "status": "dry_run_complete",
        "scientific_config": asdict(config),
        "protocol": _protocol_payload(config),
        "outer_split_manifest_sha256": rescue._atomic_json_payload_sha256(
            rescue._split_manifest(outer)
        ),
        "internal_split_manifest_sha256": rescue._atomic_json_payload_sha256(
            _internal_manifest(internal)
        ),
        "freshness_audit": dict(freshness_audit),
        "task_protocol_registry_binding": dict(registry_binding),
        "frozen_dependency_fingerprints": dict(dependencies),
        "pinned_train_label_distribution": dict(distribution),
        "model_or_tokenizer_constructed": False,
        "gradient_steps": 0,
        "development_tune_accessed": False,
        "confirm_audit_model_scored_or_tokenized": False,
    }


def _write_internal_failure(
    *,
    output_root: Path,
    task_dir: Path,
    manifest: dict[str, Any],
    config: TransferConfig,
    resource_ledger: Mapping[str, Any],
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    primary_pass: bool,
    secondary_pass: bool,
) -> dict[str, Any]:
    result = {
        "format": FORMAT_VERSION,
        "status": "internal_qualification_failed",
        "task": config.task_name,
        "source_recipe_task": "BoolQA",
        "target_task_recipe_hparam_checkpoint_tuning": False,
        "target_task_gradient_adaptation": True,
        "target_task_internal_threshold_calibration": True,
        "only_eligible_step": config.optimizer_steps,
        "resource_ledger": dict(resource_ledger),
        "internal_calibration": {
            "primary_raw_official": dict(primary),
            "secondary_calibrated_and_auc": dict(secondary),
            "primary_pass": primary_pass,
            "secondary_pass": secondary_pass,
            "qualification_pass": False,
        },
        "development_tune_accessed": False,
        "confirm_eligible": False,
        "confirm_audit_model_scored_or_tokenized": False,
    }
    rescue._atomic_json(task_dir / "qualification_failure.json", result)
    manifest.update(
        {
            "status": "internal_qualification_failed",
            "resource_ledger": dict(resource_ledger),
            "development_tune_accessed": False,
        }
    )
    rescue._atomic_json(output_root / "manifest.json", manifest)
    (output_root / "QUALIFICATION_FAILED").write_text(
        "target-task internal gate failed; development and confirm stayed unscored\n",
        encoding="utf-8",
    )
    return result


def run_selection(
    config: TransferConfig,
    *,
    train_examples: Sequence[OfficialExample] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Run the sole frozen-recipe target-task adaptation trajectory."""

    if config.confirm_selection is not None:
        raise ValueError("run_selection cannot execute transfer confirmation mode")
    output_root = Path(config.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing non-empty transfer output root {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    task_dir = output_root / config.task_name
    task_dir.mkdir(parents=True, exist_ok=False)

    injected_examples = train_examples is not None
    if injected_examples and not config.tiny_random_model:
        raise ValueError("formal transfer selection forbids train_examples injection")
    if train_examples is None:
        train_examples = load_official_examples(
            config.data_root, config.task_name, "train", verify=True
        )
    data_artifact = (
        None
        if injected_examples
        else bap._official_train_artifact_fingerprint(
            config.data_root, task_name=config.task_name
        )
    )
    corpus_fingerprint = bap._train_corpus_fingerprint(
        train_examples, task_name=config.task_name
    )
    execution_environment = bap._execution_environment_fingerprint(config.device)
    dependencies = _dependency_fingerprints()
    distribution = rescue.pinned_train_label_distribution(
        train_examples, task_name=config.task_name
    )
    natural_prior = distribution["natural_label_prior"]
    if set(natural_prior) != set(BINARY_LABELS) or not math.isclose(
        sum(float(value) for value in natural_prior.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("target natural prior must cover True/False and sum to one")
    outer, internal, sources, freshness_audit = _build_splits(
        train_examples, config=config
    )
    registry_binding = _load_task_protocol_registry(config, sources)
    outer_manifest = rescue._split_manifest(outer)
    internal_manifest = _internal_manifest(internal)
    rescue._atomic_json(task_dir / "split.json", outer_manifest)
    rescue._atomic_json(task_dir / "internal_split.json", internal_manifest)

    protocol_lock_path = task_dir / "protocol_lock.json"
    protocol_lock_payload = _protocol_lock_payload(
        config,
        sources=sources,
        freshness_audit=freshness_audit,
        registry_binding=registry_binding,
        dependency_fingerprints=dependencies,
    )
    rescue._atomic_json(protocol_lock_path, protocol_lock_payload)
    protocol_lock = {
        "path": str(protocol_lock_path.resolve()),
        "sha256": rescue._sha256_file(protocol_lock_path),
        "payload_sha256": protocol_lock_payload["payload_sha256"],
        "created_before_model_or_gradients": True,
        "external_preregistration_claimed": False,
    }
    manifest: dict[str, Any] = {
        "format": FORMAT_VERSION,
        "status": "running",
        "scientific_config": asdict(config),
        "protocol": _protocol_payload(config),
        "outer_split_manifest_sha256": rescue._atomic_json_payload_sha256(
            outer_manifest
        ),
        "internal_split_manifest_sha256": rescue._atomic_json_payload_sha256(
            internal_manifest
        ),
        "freshness_sources": [dict(item) for item in sources],
        "freshness_audit": freshness_audit,
        "task_protocol_registry_binding": registry_binding,
        "frozen_dependency_fingerprints": dependencies,
        "pinned_train_label_distribution": distribution,
        "train_corpus_fingerprint": corpus_fingerprint,
        "execution_environment": execution_environment,
        "verified_data_artifact": data_artifact,
        "run_local_protocol_lock": protocol_lock,
        "train_examples_injected_for_tiny_test": injected_examples,
        "development_tune_accessed": False,
        "confirm_audit_model_scored_or_tokenized": False,
    }
    rescue._atomic_json(output_root / "manifest.json", manifest)
    if config.dry_run:
        payload = _dry_run_payload(
            config,
            outer,
            internal,
            distribution,
            freshness_audit=freshness_audit,
            registry_binding=registry_binding,
            dependencies=dependencies,
        )
        rescue._atomic_json(output_root / "dry_run.json", payload)
        manifest["status"] = "dry_run_complete"
        rescue._atomic_json(output_root / "manifest.json", manifest)
        (output_root / "DRY_RUN_COMPLETE").write_text(
            "transfer dry run completed; no model, gradient, or audit scoring\n",
            encoding="utf-8",
        )
        return payload

    injected_tokenizer = tokenizer is not None
    if injected_tokenizer and not config.tiny_random_model:
        raise ValueError("formal transfer selection forbids tokenizer injection")
    model_artifact = bap._local_artifact_fingerprint(config.model_path)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, use_fast=True, local_files_only=True
        )
    tokenizer_fingerprint = bap._tokenizer_protocol_fingerprint(tokenizer)
    model_config = _model_config(config)
    bank = rescue._build_model(model_config, tokenizer)
    initial_fingerprint = rescue._adapter_fingerprint(bank)
    payload_audit = bank.payload_audit().to_dict()
    if not payload_audit["exact_budget"]:
        raise RuntimeError("transfer model violates the fixed adapter budget")

    base_internal_raw, base_internal_margins = _score_partition(
        bank,
        tokenizer,
        internal.calibration,
        config=config,
        natural_prior=natural_prior,
    )
    resource_ledger = _train_fixed_trajectory(
        bank,
        tokenizer,
        internal.fit,
        config=config,
        natural_prior=natural_prior,
    )
    post_internal_raw, post_internal_margins = _score_partition(
        bank,
        tokenizer,
        internal.calibration,
        config=config,
        natural_prior=natural_prior,
    )
    internal_primary = pcsm.raw_official_comparison(
        base_internal_raw, post_internal_raw
    )
    internal_secondary = pcsm.calibrated_internal_comparison(
        base_internal_margins,
        post_internal_margins,
        [item.label for item in internal.calibration],
        natural_prior=natural_prior,
        config=config,  # type: ignore[arg-type]
    )
    internal_primary_pass = pcsm.primary_gate(
        internal_primary, config=config  # type: ignore[arg-type]
    )
    internal_secondary_pass = pcsm.secondary_gate(
        internal_secondary, config=config  # type: ignore[arg-type]
    )
    if not (internal_primary_pass and internal_secondary_pass):
        failure = _write_internal_failure(
            output_root=output_root,
            task_dir=task_dir,
            manifest=manifest,
            config=config,
            resource_ledger=resource_ledger,
            primary=internal_primary,
            secondary=internal_secondary,
            primary_pass=internal_primary_pass,
            secondary_pass=internal_secondary_pass,
        )
        rescue._release_model(bank)
        return failure

    checkpoint_dir = task_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = checkpoint_dir / f"step_{config.optimizer_steps}.pt"
    rescue._atomic_torch_save(checkpoint_path, bank.compact_state_dict())
    checkpoint = {
        "path": str(checkpoint_path.resolve()),
        "sha256": rescue._sha256_file(checkpoint_path),
        "optimizer_step": config.optimizer_steps,
        "sole_eligible_checkpoint": True,
    }
    internal_lock_path = task_dir / "internal_qualification_lock.json"
    internal_lock_core = {
        "format": FORMAT_VERSION,
        "status": "internal_qualification_locked_before_development_access",
        "protocol_lock_sha256": protocol_lock["sha256"],
        "outer_split_manifest_sha256": manifest["outer_split_manifest_sha256"],
        "internal_split_manifest_sha256": manifest["internal_split_manifest_sha256"],
        "selected_checkpoint": checkpoint,
        "resource_ledger_sha256": bap._sha256_json(resource_ledger),
        "primary_raw_official": internal_primary,
        "secondary_calibrated_and_auc": internal_secondary,
        "primary_pass": True,
        "secondary_pass": True,
        "development_tune_accessed": False,
        "confirm_audit_model_scored_or_tokenized": False,
    }
    internal_lock_payload = {
        **internal_lock_core,
        "payload_sha256": bap._sha256_json(internal_lock_core),
    }
    rescue._atomic_json(internal_lock_path, internal_lock_payload)
    internal_lock = {
        "path": str(internal_lock_path.resolve()),
        "sha256": rescue._sha256_file(internal_lock_path),
        "payload_sha256": internal_lock_payload["payload_sha256"],
        "created_before_development_access": True,
    }

    base_dev_bank = rescue._build_model(model_config, tokenizer)
    if rescue._adapter_fingerprint(base_dev_bank) != initial_fingerprint:
        raise RuntimeError("transfer frozen development base initialization drifted")
    base_dev_raw, base_dev_margins = _score_partition(
        base_dev_bank,
        tokenizer,
        outer.tune_audit,
        config=config,
        natural_prior=natural_prior,
    )
    rescue._release_model(base_dev_bank)
    post_dev_raw, post_dev_margins = _score_partition(
        bank,
        tokenizer,
        outer.tune_audit,
        config=config,
        natural_prior=natural_prior,
    )
    dev_primary = pcsm.raw_official_comparison(base_dev_raw, post_dev_raw)
    base_threshold = float(
        internal_secondary["base_same_calibration_control"]["threshold"]
    )
    post_threshold = float(internal_secondary["pcsm_same_calibration"]["threshold"])
    dev_secondary = pcsm.calibrated_frozen_comparison(
        base_dev_margins,
        post_dev_margins,
        [item.label for item in outer.tune_audit],
        natural_prior=natural_prior,
        base_threshold=base_threshold,
        post_threshold=post_threshold,
    )
    dev_primary_pass = pcsm.primary_gate(
        dev_primary, config=config  # type: ignore[arg-type]
    )
    dev_secondary_pass = pcsm.secondary_gate(
        dev_secondary, config=config  # type: ignore[arg-type]
    )
    confirm_eligible = dev_primary_pass and dev_secondary_pass

    selection = {
        "format": FORMAT_VERSION,
        "status": "selection_locked",
        "task": config.task_name,
        "source_recipe_task": "BoolQA",
        "target_task_recipe_hparam_checkpoint_tuning": False,
        "target_task_gradient_adaptation": True,
        "target_task_internal_threshold_calibration": True,
        "scientific_config": asdict(config),
        "protocol": _protocol_payload(config),
        "selected": {
            "objective": OBJECTIVE,
            "step": config.optimizer_steps,
            "selection_rule": "sole frozen transfer step; no target checkpoint maximization",
            "base_threshold": base_threshold,
            "pcsm_threshold": post_threshold,
        },
        "selected_checkpoint": checkpoint,
        "resource_ledger": resource_ledger,
        "internal_calibration": {
            "primary_raw_official": internal_primary,
            "secondary_calibrated_and_auc": internal_secondary,
            "primary_pass": internal_primary_pass,
            "secondary_pass": internal_secondary_pass,
            "qualification_pass": True,
        },
        "internal_qualification_lock": internal_lock,
        "development_tune": {
            "role": "target-task second qualification gate; not recipe tuning",
            "accessed_only_after_internal_pass": True,
            "thresholds_frozen_before_access": True,
            "primary_raw_official": dev_primary,
            "secondary_calibrated_and_auc": dev_secondary,
            "primary_pass": dev_primary_pass,
            "secondary_pass": dev_secondary_pass,
            "qualification_pass": confirm_eligible,
            "did_not_select_recipe_checkpoint_or_threshold": True,
        },
        "confirm_eligible": confirm_eligible,
        "confirm_ineligibility_reason": (
            None
            if confirm_eligible
            else "target development primary and/or secondary gate failed"
        ),
        "outer_split_manifest_sha256": manifest["outer_split_manifest_sha256"],
        "internal_split_manifest_sha256": manifest["internal_split_manifest_sha256"],
        "freshness_sources": [dict(item) for item in sources],
        "freshness_audit": freshness_audit,
        "task_protocol_registry_binding": registry_binding,
        "frozen_dependency_fingerprints": dependencies,
        "pinned_train_label_distribution": distribution,
        "train_corpus_fingerprint": corpus_fingerprint,
        "execution_environment": execution_environment,
        "verified_data_artifact": data_artifact,
        "run_local_protocol_lock": protocol_lock,
        "train_examples_injected_for_tiny_test": injected_examples,
        "model_protocol": {
            "initial_adapter_fingerprint": initial_fingerprint,
            "frozen_model_tokenizer_artifact": model_artifact,
            "tokenizer_protocol_fingerprint": tokenizer_fingerprint,
            "tokenizer_injected_for_tiny_test": injected_tokenizer,
            "payload": payload_audit,
            "rescue_loss_objective": OBJECTIVE,
        },
        "development_tune_accessed": True,
        "confirm_audit_model_scored_or_tokenized": False,
    }
    selection_path = task_dir / "selection.json"
    rescue._atomic_json(selection_path, selection)
    selection_sha = rescue._sha256_file(selection_path)
    selection_lock = {
        "format": FORMAT_VERSION,
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": selection_sha,
        "selected_checkpoint_sha256": checkpoint["sha256"],
        "selected_step": config.optimizer_steps,
        "internal_qualification_pass": True,
        "development_qualification_pass": confirm_eligible,
        "confirm_eligible": confirm_eligible,
        "confirm_audit_model_scored_or_tokenized": False,
    }
    rescue._atomic_json(task_dir / "selection_lock.json", selection_lock)
    manifest.update(
        {
            "status": "selection_completed",
            "selection": {
                "path": str(selection_path.resolve()),
                "sha256": selection_sha,
                "selected_checkpoint_sha256": checkpoint["sha256"],
                "selected_step": config.optimizer_steps,
                "confirm_eligible": confirm_eligible,
            },
            "resource_ledger": resource_ledger,
            "internal_qualification_lock": internal_lock,
            "development_tune_accessed": True,
        }
    )
    rescue._atomic_json(output_root / "manifest.json", manifest)
    (output_root / "SELECTION_COMPLETED").write_text(
        "target transfer selection completed; confirm remains unscored and untokenized\n",
        encoding="utf-8",
    )
    rescue._release_model(bank)
    return selection


def load_selection(path: str | os.PathLike[str]) -> tuple[dict[str, Any], Path]:
    selection_path = Path(path).expanduser().resolve()
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load transfer selection {selection_path}") from error
    if not isinstance(selection, dict) or selection.get("format") != FORMAT_VERSION:
        raise ValueError("selection is not a PCSM transfer v1 artifact")
    if selection.get("status") != "selection_locked":
        raise ValueError("transfer selection is not durably locked")
    mappings = {
        "scientific_config",
        "protocol",
        "selected",
        "selected_checkpoint",
        "resource_ledger",
        "internal_calibration",
        "internal_qualification_lock",
        "development_tune",
        "model_protocol",
        "run_local_protocol_lock",
        "freshness_audit",
        "task_protocol_registry_binding",
        "frozen_dependency_fingerprints",
    }
    missing = [name for name in mappings if not isinstance(selection.get(name), Mapping)]
    if missing:
        raise ValueError("transfer selection lacks mappings: " + ", ".join(sorted(missing)))
    return selection, selection_path


def _config_from_selection(
    selection: Mapping[str, Any],
    *,
    output_root: str,
    selection_path: Path,
    device: str | None = None,
) -> TransferConfig:
    payload = selection.get("scientific_config")
    if not isinstance(payload, Mapping):
        raise ValueError("transfer selection lacks scientific_config")
    values = dict(payload)
    values["fresh_confirm_against_manifests"] = tuple(
        values.get("fresh_confirm_against_manifests", ())
    )
    locked_device = str(values.get("device"))
    if device is not None and str(device) != locked_device:
        raise ValueError(
            f"confirmation device {device!r} differs from locked {locked_device!r}"
        )
    values["output_root"] = output_root
    values["confirm_selection"] = str(selection_path.resolve())
    values["dry_run"] = False
    return TransferConfig(**values)


def _validate_confirmation_config(
    config: TransferConfig, selection: Mapping[str, Any]
) -> None:
    locked = selection.get("scientific_config")
    if not isinstance(locked, Mapping):
        raise ValueError("transfer selection lacks scientific_config")
    current = asdict(config)
    operational = {"output_root", "confirm_selection"}
    locked_scientific = {key: value for key, value in locked.items() if key not in operational}
    current_scientific = {
        key: value for key, value in current.items() if key not in operational
    }
    if bap._sha256_json(locked_scientific) != bap._sha256_json(current_scientific):
        differing = sorted(
            key
            for key in set(locked_scientific) | set(current_scientific)
            if bap._sha256_json(locked_scientific.get(key))
            != bap._sha256_json(current_scientific.get(key))
        )
        raise ValueError(
            "confirmation scientific config differs from locked transfer: "
            + ", ".join(differing)
        )


def _preflight_confirmation(
    selection: Mapping[str, Any],
    selection_path: Path,
    outer: rescue.RescueSplit,
    internal: bap.BAPInternalSplit,
    distribution: Mapping[str, Any],
    *,
    config: TransferConfig,
) -> dict[str, Any]:
    if selection.get("task") != config.task_name:
        raise ValueError("transfer selection task changed")
    if selection.get("source_recipe_task") != "BoolQA":
        raise ValueError("transfer selection lost its BoolQA recipe provenance")
    if selection.get("target_task_recipe_hparam_checkpoint_tuning") is not False:
        raise ValueError(
            "transfer selection does not attest zero target-task recipe/hparam/checkpoint tuning"
        )
    if selection.get("target_task_gradient_adaptation") is not True:
        raise ValueError("transfer selection lost target-task gradient-adaptation disclosure")
    if selection.get("target_task_internal_threshold_calibration") is not True:
        raise ValueError(
            "transfer selection lost target-task internal-threshold-calibration disclosure"
        )
    if selection.get("confirm_eligible") is not True:
        raise ValueError("transfer selection is not eligible for sealed confirmation")
    if selection.get("confirm_audit_model_scored_or_tokenized") is not False:
        raise ValueError("transfer selection does not attest a sealed confirm partition")
    for partition_name in ("internal_calibration", "development_tune"):
        partition = selection[partition_name]
        primary = partition.get("primary_raw_official")
        secondary = partition.get("secondary_calibrated_and_auc")
        if (
            not isinstance(primary, Mapping)
            or not isinstance(secondary, Mapping)
            or not pcsm.primary_gate(primary, config=config)  # type: ignore[arg-type]
            or not pcsm.secondary_gate(secondary, config=config)  # type: ignore[arg-type]
            or partition.get("primary_pass") is not True
            or partition.get("secondary_pass") is not True
            or partition.get("qualification_pass") is not True
        ):
            raise ValueError(f"transfer {partition_name} did not pass both gates")

    selected = selection["selected"]
    locked_step = int(selection["scientific_config"]["optimizer_steps"])
    if (
        selected.get("objective") != OBJECTIVE
        or int(selected.get("step", -1)) != locked_step
        or selected.get("selection_rule")
        != "sole frozen transfer step; no target checkpoint maximization"
    ):
        raise ValueError("transfer selected objective/step protocol is invalid")
    internal_secondary = selection["internal_calibration"][
        "secondary_calibrated_and_auc"
    ]
    for key in ("base_threshold", "pcsm_threshold"):
        if not math.isfinite(float(selected[key])):
            raise ValueError(f"transfer selected {key} is non-finite")
    if not math.isclose(
        float(selected["base_threshold"]),
        float(internal_secondary["base_same_calibration_control"]["threshold"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(selected["pcsm_threshold"]),
        float(internal_secondary["pcsm_same_calibration"]["threshold"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("transfer thresholds differ from internal calibration")

    selection_sha = rescue._sha256_file(selection_path)
    selection_lock_path = selection_path.parent / "selection_lock.json"
    if not selection_lock_path.is_file():
        raise ValueError("transfer selection lock is missing")
    try:
        selection_lock = json.loads(
            selection_lock_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load transfer selection lock") from error
    checkpoint_payload = selection["selected_checkpoint"]
    if (
        selection_lock.get("format") != FORMAT_VERSION
        or Path(str(selection_lock.get("selection_path", ""))).expanduser().resolve()
        != selection_path.resolve()
        or selection_lock.get("selection_sha256") != selection_sha
        or selection_lock.get("selected_checkpoint_sha256")
        != checkpoint_payload.get("sha256")
        or int(selection_lock.get("selected_step", -1)) != locked_step
        or selection_lock.get("internal_qualification_pass") is not True
        or selection_lock.get("development_qualification_pass") is not True
        or selection_lock.get("confirm_eligible") is not True
        or selection_lock.get("confirm_audit_model_scored_or_tokenized") is not False
    ):
        raise ValueError("transfer selection lock does not bind the selection")

    protocol_metadata = selection["run_local_protocol_lock"]
    protocol_path = (selection_path.parent / "protocol_lock.json").resolve()
    if (
        Path(str(protocol_metadata.get("path", ""))).expanduser().resolve()
        != protocol_path
        or not protocol_path.is_file()
        or rescue._sha256_file(protocol_path) != protocol_metadata.get("sha256")
        or protocol_metadata.get("external_preregistration_claimed") is not False
    ):
        raise ValueError("transfer protocol lock path or SHA-256 changed")
    try:
        protocol_lock = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load transfer protocol lock") from error
    protocol_without_hash = dict(protocol_lock)
    recorded_protocol_sha = protocol_without_hash.pop("payload_sha256", None)
    if (
        recorded_protocol_sha != bap._sha256_json(protocol_without_hash)
        or recorded_protocol_sha != protocol_metadata.get("payload_sha256")
        or bap._sha256_json(protocol_lock.get("scientific_config"))
        != bap._sha256_json(selection.get("scientific_config"))
        or bap._sha256_json(protocol_lock.get("protocol"))
        != bap._sha256_json(selection.get("protocol"))
        or protocol_lock.get("resolved_freshness_sources")
        != selection.get("freshness_sources")
        or protocol_lock.get("freshness_audit") != selection.get("freshness_audit")
        or protocol_lock.get("task_protocol_registry_binding")
        != selection.get("task_protocol_registry_binding")
        or protocol_lock.get("frozen_dependency_fingerprints")
        != selection.get("frozen_dependency_fingerprints")
    ):
        raise ValueError("transfer protocol lock payload changed")

    checkpoint = Path(str(checkpoint_payload.get("path", ""))).expanduser().resolve()
    expected_checkpoint = (
        selection_path.parent / "checkpoints" / f"step_{locked_step}.pt"
    ).resolve()
    if checkpoint != expected_checkpoint or not checkpoint.is_file():
        raise ValueError("transfer selected checkpoint path is invalid")
    checkpoint_sha = rescue._sha256_file(checkpoint)
    if checkpoint_sha != checkpoint_payload.get("sha256"):
        raise ValueError("transfer selected checkpoint SHA-256 mismatch")
    if (
        int(checkpoint_payload.get("optimizer_step", -1)) != locked_step
        or checkpoint_payload.get("sole_eligible_checkpoint") is not True
    ):
        raise ValueError("transfer checkpoint violates the sole-step rule")

    internal_lock_metadata = selection["internal_qualification_lock"]
    internal_lock_path = (
        selection_path.parent / "internal_qualification_lock.json"
    ).resolve()
    if (
        Path(str(internal_lock_metadata.get("path", ""))).expanduser().resolve()
        != internal_lock_path
        or not internal_lock_path.is_file()
        or rescue._sha256_file(internal_lock_path)
        != internal_lock_metadata.get("sha256")
        or internal_lock_metadata.get("created_before_development_access") is not True
    ):
        raise ValueError("transfer internal qualification lock changed")
    try:
        internal_lock = json.loads(internal_lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load transfer internal qualification lock") from error
    internal_without_hash = dict(internal_lock)
    recorded_internal_sha = internal_without_hash.pop("payload_sha256", None)
    if (
        recorded_internal_sha != bap._sha256_json(internal_without_hash)
        or recorded_internal_sha != internal_lock_metadata.get("payload_sha256")
        or internal_lock.get("status")
        != "internal_qualification_locked_before_development_access"
        or internal_lock.get("protocol_lock_sha256") != protocol_metadata.get("sha256")
        or internal_lock.get("outer_split_manifest_sha256")
        != selection.get("outer_split_manifest_sha256")
        or internal_lock.get("internal_split_manifest_sha256")
        != selection.get("internal_split_manifest_sha256")
        or internal_lock.get("selected_checkpoint") != checkpoint_payload
        or internal_lock.get("resource_ledger_sha256")
        != bap._sha256_json(selection["resource_ledger"])
        or internal_lock.get("primary_raw_official")
        != selection["internal_calibration"].get("primary_raw_official")
        or internal_lock.get("secondary_calibrated_and_auc")
        != selection["internal_calibration"].get("secondary_calibrated_and_auc")
        or internal_lock.get("primary_pass") is not True
        or internal_lock.get("secondary_pass") is not True
        or internal_lock.get("development_tune_accessed") is not False
        or internal_lock.get("confirm_audit_model_scored_or_tokenized") is not False
    ):
        raise ValueError("transfer internal qualification lock payload changed")

    config_payload = selection["scientific_config"]
    ledger = selection["resource_ledger"]
    expected_exposures = locked_step * int(config_payload["batch_size"])
    expected_per_class = expected_exposures // 2
    if (
        ledger.get("objective") != OBJECTIVE
        or int(ledger.get("optimizer_steps", -1)) != locked_step
        or ledger.get("eligible_checkpoint_steps") != [locked_step]
        or int(ledger.get("checkpoint_candidates_evaluated", -1)) != 1
        or int(ledger.get("gradient_example_exposures", -1)) != expected_exposures
        or ledger.get("paired_exposures_per_class")
        != {label: expected_per_class for label in BINARY_LABELS}
        or ledger.get("exact_budget") is not True
        or ledger.get("dropout_disabled") is not True
    ):
        raise ValueError("transfer resource ledger violates the frozen trajectory")
    exposure_ids = [
        item.example_id
        for _step, batch in bap.paired_fit_batches(
            internal.fit,
            batch_size=int(config_payload["batch_size"]),
            steps=locked_step,
            seed=int(config_payload["seed"]),
        )
        for item in batch
    ]
    if ledger.get("ordered_exposure_ids_sha256") != bap._sha256_json(exposure_ids):
        raise ValueError("transfer ordered exposure schedule changed")

    outer_sha = rescue._atomic_json_payload_sha256(rescue._split_manifest(outer))
    internal_sha = rescue._atomic_json_payload_sha256(_internal_manifest(internal))
    if outer_sha != selection.get("outer_split_manifest_sha256"):
        raise ValueError("reconstructed transfer outer split changed")
    if internal_sha != selection.get("internal_split_manifest_sha256"):
        raise ValueError("reconstructed transfer internal split changed")
    if distribution != selection.get("pinned_train_label_distribution"):
        raise ValueError("target train label distribution changed")

    model_protocol = selection["model_protocol"]
    current_model_artifact = bap._local_artifact_fingerprint(
        str(config_payload["model_path"])
    )
    if current_model_artifact != model_protocol.get("frozen_model_tokenizer_artifact"):
        raise ValueError("frozen model/tokenizer artifact changed after transfer selection")
    if model_protocol.get("rescue_loss_objective") != OBJECTIVE:
        raise ValueError("transfer model protocol objective changed")

    manifest_path = selection_path.parent.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("transfer selection manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load transfer selection manifest") from error
    manifest_selection = manifest.get("selection")
    if (
        manifest.get("format") != FORMAT_VERSION
        or manifest.get("status") != "selection_completed"
        or not isinstance(manifest_selection, Mapping)
        or manifest_selection.get("sha256") != selection_sha
        or manifest_selection.get("selected_checkpoint_sha256") != checkpoint_sha
        or int(manifest_selection.get("selected_step", -1)) != locked_step
        or manifest_selection.get("confirm_eligible") is not True
        or manifest.get("internal_qualification_lock")
        != selection.get("internal_qualification_lock")
        or manifest.get("freshness_sources") != selection.get("freshness_sources")
        or manifest.get("task_protocol_registry_binding")
        != selection.get("task_protocol_registry_binding")
    ):
        raise ValueError("transfer manifest does not bind the eligible selection")
    return {
        "selection_sha256": selection_sha,
        "selection_lock": selection_lock,
        "protocol_lock": dict(protocol_metadata),
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "outer_split_manifest_sha256": outer_sha,
        "internal_split_manifest_sha256": internal_sha,
        "frozen_model_tokenizer_artifact": current_model_artifact,
    }


def _claim_confirmation(
    selection_path: Path, *, output_root: Path, preflight: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    claim_path = selection_path.parent / "confirmation_claim.json"
    core = {
        "format": FORMAT_VERSION,
        "status": "claimed_immediately_before_confirm_scoring",
        "selection_sha256": preflight["selection_sha256"],
        "checkpoint_sha256": preflight["checkpoint_sha256"],
        "confirmation_output_root": str(output_root.resolve()),
        "claimed_unix_time": time.time(),
        "scope_note": (
            "confirm metadata was used to reconstruct the locked semantic split; "
            "no confirm row was model-scored or tokenized before this claim"
        ),
    }
    payload = {**core, "claimed_payload_sha256": bap._sha256_json(core)}
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with claim_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise RuntimeError(
            "this transfer selection already has a confirmation claim; refusing a second confirm read"
        ) from error
    return claim_path, payload


def _finalize_claim(
    claim_path: Path,
    claimed_payload: Mapping[str, Any],
    *,
    result_path: Path,
) -> None:
    current = json.loads(claim_path.read_text(encoding="utf-8"))
    if bap._sha256_json(current) != bap._sha256_json(claimed_payload):
        raise RuntimeError("transfer confirmation claim changed during scoring")
    core = dict(current)
    recorded = core.pop("claimed_payload_sha256", None)
    if recorded != bap._sha256_json(core):
        raise RuntimeError("transfer confirmation claim payload hash is invalid")
    rescue._atomic_json(
        claim_path,
        {
            **current,
            "status": "confirmation_completed",
            "initial_claim_payload_sha256": recorded,
            "confirmation_result": str(result_path.resolve()),
            "confirmation_result_sha256": rescue._sha256_file(result_path),
        },
    )


def run_confirmation(
    config: TransferConfig,
    selection: Mapping[str, Any],
    selection_path: Path,
    *,
    train_examples: Sequence[OfficialExample] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Independently score a target-task confirm partition exactly once."""

    disk_selection, disk_path = load_selection(selection_path)
    if bap._sha256_json(selection) != bap._sha256_json(disk_selection):
        raise ValueError("in-memory selection differs from locked transfer selection")
    selection = disk_selection
    selection_path = disk_path
    _validate_confirmation_config(config, selection)
    output_root = Path(config.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing non-empty transfer confirmation root {output_root}")
    if (selection_path.parent / "confirmation_claim.json").exists():
        raise RuntimeError(
            "this transfer selection already has a confirmation claim; refusing a second confirm read"
        )

    injected_examples = train_examples is not None
    if injected_examples and not config.tiny_random_model:
        raise ValueError("formal transfer confirmation forbids train_examples injection")
    if injected_examples != bool(
        selection.get("train_examples_injected_for_tiny_test", False)
    ):
        raise ValueError("transfer confirmation data-loading mode changed")
    current_data_artifact = (
        None
        if injected_examples
        else bap._official_train_artifact_fingerprint(
            config.data_root, task_name=config.task_name
        )
    )
    if current_data_artifact != selection.get("verified_data_artifact"):
        raise ValueError("verified target data artifact changed after selection")
    if train_examples is None:
        train_examples = load_official_examples(
            config.data_root, config.task_name, "train", verify=True
        )
    corpus_fingerprint = bap._train_corpus_fingerprint(
        train_examples, task_name=config.task_name
    )
    if corpus_fingerprint != selection.get("train_corpus_fingerprint"):
        raise ValueError("target train corpus changed after transfer selection")
    execution_environment = bap._execution_environment_fingerprint(config.device)
    if execution_environment != selection.get("execution_environment"):
        raise ValueError("transfer confirmation execution environment drifted")
    dependencies = _dependency_fingerprints()
    if dependencies != selection.get("frozen_dependency_fingerprints"):
        raise ValueError("frozen transfer dependencies changed after selection")
    distribution = rescue.pinned_train_label_distribution(
        train_examples, task_name=config.task_name
    )
    natural_prior = distribution["natural_label_prior"]
    outer, internal, sources, freshness_audit = _build_splits(
        train_examples, config=config
    )
    if [dict(item) for item in sources] != selection.get("freshness_sources"):
        raise ValueError("target freshness sources changed after transfer selection")
    if freshness_audit != selection.get("freshness_audit"):
        raise ValueError("target freshness audit changed after transfer selection")
    registry_binding = _load_task_protocol_registry(config, sources)
    if registry_binding != selection.get("task_protocol_registry_binding"):
        raise ValueError("target task protocol registry changed after selection")
    preflight = _preflight_confirmation(
        selection,
        selection_path,
        outer,
        internal,
        distribution,
        config=config,
    )

    injected_tokenizer = tokenizer is not None
    if injected_tokenizer and not config.tiny_random_model:
        raise ValueError("formal transfer confirmation forbids tokenizer injection")
    if injected_tokenizer != bool(
        selection["model_protocol"].get("tokenizer_injected_for_tiny_test", False)
    ):
        raise ValueError("transfer confirmation tokenizer-loading mode changed")
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, use_fast=True, local_files_only=True
        )
    tokenizer_fingerprint = bap._tokenizer_protocol_fingerprint(tokenizer)
    if tokenizer_fingerprint != selection["model_protocol"].get(
        "tokenizer_protocol_fingerprint"
    ):
        raise ValueError("target tokenizer protocol changed after selection")

    model_config = _model_config(config)
    expected_init = selection["model_protocol"]["initial_adapter_fingerprint"]
    base_bank = rescue._build_model(model_config, tokenizer)
    if rescue._adapter_fingerprint(base_bank) != expected_init:
        raise ValueError("transfer confirmation base initialization changed")
    post_bank = rescue._build_model(model_config, tokenizer)
    if rescue._adapter_fingerprint(post_bank) != expected_init:
        raise ValueError("transfer confirmation adapter initialization changed")
    try:
        checkpoint_state = torch.load(
            preflight["checkpoint"], map_location="cpu", weights_only=True
        )
        post_bank.load_compact_state_dict(checkpoint_state)
    except Exception:
        rescue._release_model(base_bank)
        rescue._release_model(post_bank)
        raise

    output_root.mkdir(parents=True, exist_ok=True)
    claim_path, claimed_payload = _claim_confirmation(
        selection_path, output_root=output_root, preflight=preflight
    )
    base_raw, base_margins = _score_partition(
        base_bank,
        tokenizer,
        outer.confirm_audit,
        config=config,
        natural_prior=natural_prior,
    )
    post_raw, post_margins = _score_partition(
        post_bank,
        tokenizer,
        outer.confirm_audit,
        config=config,
        natural_prior=natural_prior,
    )
    confirm_primary = pcsm.raw_official_comparison(base_raw, post_raw)
    confirm_primary_pass = pcsm.primary_gate(
        confirm_primary, config=config  # type: ignore[arg-type]
    )
    confirm_secondary = pcsm.calibrated_frozen_comparison(
        base_margins,
        post_margins,
        [item.label for item in outer.confirm_audit],
        natural_prior=natural_prior,
        base_threshold=float(selection["selected"]["base_threshold"]),
        post_threshold=float(selection["selected"]["pcsm_threshold"]),
    )
    confirm_secondary_diagnostic_pass = pcsm.secondary_gate(
        confirm_secondary, config=config  # type: ignore[arg-type]
    )
    result = {
        "format": FORMAT_VERSION,
        "status": "confirmation_completed",
        "task": config.task_name,
        "source_recipe_task": "BoolQA",
        "target_task_recipe_hparam_checkpoint_tuning": False,
        "target_task_gradient_adaptation": True,
        "target_task_internal_threshold_calibration": True,
        "selection": {
            "path": str(selection_path),
            "sha256": preflight["selection_sha256"],
            "checkpoint_sha256": preflight["checkpoint_sha256"],
            "selected_step": config.optimizer_steps,
        },
        "confirmation_claim": {
            "path": str(claim_path.resolve()),
            "created_immediately_before_confirm_scoring": True,
            "confirm_metadata_read_before_claim": True,
            "confirm_model_scored_or_tokenized_before_claim": False,
        },
        "primary_raw_official": confirm_primary,
        "confirm_primary_gate_pass": confirm_primary_pass,
        "official_positive_claim_metric": (
            "raw official free-generation natural-prior EM gain with validity/class guards"
        ),
        "secondary_calibrated_and_auc_report": confirm_secondary,
        "secondary_diagnostic_pass": confirm_secondary_diagnostic_pass,
        "secondary_used_for_confirmation_selection_or_tuning": False,
        "thresholds_frozen_on_target_internal_calibration": {
            "base": float(selection["selected"]["base_threshold"]),
            "pcsm": float(selection["selected"]["pcsm_threshold"]),
        },
        "calibration_gain_counted_as_primary": False,
        "confirm_rows_used_for_recipe_checkpoint_or_threshold_selection": False,
        "freshness_audit": freshness_audit,
        "task_protocol_registry_binding": registry_binding,
        "frozen_dependency_fingerprints": dependencies,
    }
    task_dir = output_root / config.task_name
    task_dir.mkdir(parents=True, exist_ok=False)
    result_path = task_dir / "confirmation.json"
    rescue._atomic_json(result_path, result)
    rescue._atomic_json(
        output_root / "manifest.json",
        {
            "format": FORMAT_VERSION,
            "status": "confirmation_completed",
            "scientific_config": asdict(config),
            "protocol": _protocol_payload(config),
            "freshness_audit": freshness_audit,
            "task_protocol_registry_binding": registry_binding,
            "frozen_dependency_fingerprints": dependencies,
            "preflight": {
                **dict(preflight),
                "checkpoint": str(preflight["checkpoint"]),
            },
            "summary": {
                "confirm_primary_gate_pass": confirm_primary_pass,
                "raw_natural_prior_gain": confirm_primary["natural_prior_gain"],
                "post_valid_label_rate": confirm_primary["post_valid_label_rate"],
                "raw_minimum_per_class_delta": confirm_primary[
                    "minimum_per_class_delta"
                ],
                "calibrated_gain_report_only": confirm_secondary[
                    "training_contribution_natural_prior"
                ],
                "roc_auc_delta_report_only": confirm_secondary[
                    "margin_diagnostics"
                ]["roc_auc_delta"],
            },
        },
    )
    _finalize_claim(claim_path, claimed_payload, result_path=result_path)
    (output_root / "COMPLETED").write_text(
        "PCSM target-task transfer one-shot confirmation completed\n",
        encoding="utf-8",
    )
    rescue._release_model(base_bank)
    rescue._release_model(post_bank)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument("--model-path")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task", dest="task_name", choices=TRANSFER_TASKS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-selection")
    parser.add_argument("--device")
    parser.add_argument("--task-protocol-registry")
    parser.add_argument(
        "--fresh-confirm-against-manifest",
        dest="fresh_confirm_against_manifests",
        action="append",
        default=[],
    )
    return parser


def _selection_config_from_args(args: argparse.Namespace) -> TransferConfig:
    if args.data_root is None or args.model_path is None or args.task_name is None:
        raise ValueError("transfer selection requires --data-root, --model-path, and --task")
    return TransferConfig(
        data_root=args.data_root,
        model_path=args.model_path,
        output_root=args.output_root,
        task_name=args.task_name,
        dry_run=args.dry_run,
        device="cpu" if args.device is None else args.device,
        fresh_confirm_against_manifests=tuple(
            args.fresh_confirm_against_manifests
        ),
        task_protocol_registry=args.task_protocol_registry,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        if args.confirm_selection is None:
            run_selection(_selection_config_from_args(args))
        else:
            selection, selection_path = load_selection(args.confirm_selection)
            config = _config_from_selection(
                selection,
                output_root=args.output_root,
                selection_path=selection_path,
                device=args.device,
            )
            if args.data_root is not None and (
                Path(args.data_root).expanduser().resolve()
                != Path(config.data_root).expanduser().resolve()
            ):
                raise ValueError("confirmation data_root differs from locked transfer")
            if args.model_path is not None and (
                Path(args.model_path).expanduser().resolve()
                != Path(config.model_path).expanduser().resolve()
            ):
                raise ValueError("confirmation model_path differs from locked transfer")
            if args.task_name is not None and args.task_name != config.task_name:
                raise ValueError("confirmation task differs from locked transfer")
            if args.fresh_confirm_against_manifests and tuple(
                args.fresh_confirm_against_manifests
            ) != config.fresh_confirm_against_manifests:
                raise ValueError("confirmation freshness sources differ from locked transfer")
            if (
                args.task_protocol_registry is not None
                and args.task_protocol_registry != config.task_protocol_registry
            ):
                raise ValueError("confirmation task registry differs from locked transfer")
            if args.dry_run:
                raise ValueError("--dry-run is incompatible with transfer confirmation")
            run_confirmation(config, selection, selection_path)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
