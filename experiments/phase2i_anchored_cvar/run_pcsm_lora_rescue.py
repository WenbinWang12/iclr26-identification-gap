"""Prior-Corrected Sequence-Mean (PCSM) LoRA acquisition rescue.

This runner promotes the successful ``sequence_mean_prior`` audit arm from the
BoolQA BAP study into a single, fixed qualification trajectory.  It deliberately
does not modify or consume the old BAP selection artifact.  The shared BAP code
is reused only for its semantic-group split, deterministic paired schedule,
sequence-margin audit, model construction, and integrity fingerprints.

The primary endpoint is the official raw free-generation natural-prior EM gain.
Identically calibrated base/post label decisions and margin ROC-AUC are secondary
guards; calibration gain is never counted as the primary training contribution.
Internal qualification happens before development rows are touched, and sealed
confirm rows may be scored once only after a durable two-gate selection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from . import run_acquisition_rescue as rescue
    from . import run_bap_lora_rescue as bap
    from .order4_data import OfficialExample, TASK_BY_NAME, load_official_examples
except ImportError:  # pragma: no cover - direct script execution.
    import run_acquisition_rescue as rescue  # type: ignore
    import run_bap_lora_rescue as bap  # type: ignore
    from order4_data import (  # type: ignore
        OfficialExample,
        TASK_BY_NAME,
        load_official_examples,
    )


FORMAT_VERSION = "phase2i.binary-pcsm-lora-rescue.v1"
OBJECTIVE = "sequence_mean_prior"
BINARY_LABELS = bap.BINARY_LABELS
FORMAL_FRESHNESS_SOURCE_COUNT = 4
FORMAL_FIT_PER_CLASS = 384
FORMAL_CALIBRATION_PER_CLASS = 128
FORMAL_TUNE_PER_CLASS = 128
FORMAL_CONFIRM_PER_CLASS = 32
FORMAL_LEARNING_RATE = 3e-4
FORMAL_BATCH_SIZE = 4
FORMAL_OPTIMIZER_STEPS = 384
FORMAL_SEED = 43
FORMAL_TASK = "BoolQA"
FORMAL_MAX_SOURCE_LENGTH = 512
FORMAL_MAX_TARGET_LENGTH = 50
FORMAL_TRUNCATION_MODE = "field_aware"
FORMAL_FRESHNESS_SHA256S = frozenset(
    {
        "022da431ad23bfe4f2583bc4dab06f8c8f7412be5a55fe74779bd1bf66c1cc91",
        "56c31e3411e5679f6e3c34b5b1dc3999925c5e598df52be2b7384721db3bc922",
        "e4fdc01e35235698272d2725e1e0cce7dcc171bc44b4dd23f25aae6d5f174b53",
        "c0a190eb440bf8309c1cf9b64342bafc999ed636d92c68211a99dbf21be72d92",
    }
)


@dataclass(frozen=True)
class PCSMConfig:
    """Scientific configuration for one fixed PCSM trajectory.

    Formal runs are intentionally non-configurable in the dimensions that were
    already examined in the BAP pointwise audit.  Smaller values are accepted
    only with ``tiny_random_model=True`` so tests do not masquerade as evidence.
    """

    data_root: str
    model_path: str
    output_root: str
    task_name: str = "BoolQA"
    learning_rate: float = FORMAL_LEARNING_RATE
    fit_per_class: int = FORMAL_FIT_PER_CLASS
    calibration_per_class: int = FORMAL_CALIBRATION_PER_CLASS
    tune_audit_per_class: int = FORMAL_TUNE_PER_CLASS
    confirm_audit_per_class: int = FORMAL_CONFIRM_PER_CLASS
    optimizer_steps: int = FORMAL_OPTIMIZER_STEPS
    seed: int = FORMAL_SEED
    batch_size: int = FORMAL_BATCH_SIZE
    max_source_length: int = 512
    max_target_length: int = 50
    truncation_mode: str = "field_aware"
    min_raw_gain: float = 0.05
    min_valid_label_rate: float = 0.99
    max_class_drop: float = 0.05
    tiny_random_model: bool = False
    dry_run: bool = False
    confirm_selection: str | None = None
    device: str = "cpu"
    fresh_confirm_against_manifests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.task_name not in bap.SUPPORTED_TASKS:
            raise ValueError(f"unsupported PCSM rescue task {self.task_name!r}")
        labels = TASK_BY_NAME[self.task_name].labels
        if len(labels) != 2 or set(labels) != set(BINARY_LABELS):
            raise ValueError("PCSM v1 is limited to binary True/False tasks")
        integer_fields = {
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
            for value in integer_fields.values()
        ):
            raise ValueError(f"positive integer config required: {integer_fields}")
        if self.batch_size % 2:
            raise ValueError("paired_fit_batches requires an even batch_size")
        half_batch = self.batch_size // 2
        if self.fit_per_class % half_batch:
            raise ValueError("fit_per_class must divide into exact paired half-batches")
        exposures = self.optimizer_steps * self.batch_size
        fit_rows = 2 * self.fit_per_class
        if exposures % fit_rows:
            raise ValueError("fixed steps must make an integer number of fit cycles")
        finite = {
            "learning_rate": self.learning_rate,
            "min_raw_gain": self.min_raw_gain,
            "min_valid_label_rate": self.min_valid_label_rate,
            "max_class_drop": self.max_class_drop,
        }
        if any(not math.isfinite(value) for value in finite.values()):
            raise ValueError(f"finite scientific constants required: {finite}")
        if self.learning_rate <= 0 or self.min_raw_gain < 0:
            raise ValueError("learning rate must be positive and gain nonnegative")
        if not 0 < self.min_valid_label_rate <= 1:
            raise ValueError("min_valid_label_rate must be in (0, 1]")
        if not 0 <= self.max_class_drop < 1:
            raise ValueError("max_class_drop must be in [0, 1)")
        if self.truncation_mode not in bap.TRUNCATION_MODES:
            raise ValueError(f"unsupported truncation_mode {self.truncation_mode!r}")
        if not self.tiny_random_model:
            if self.task_name != FORMAL_TASK:
                raise ValueError(
                    "PCSM v1 formal evidence is bound to the BoolQA positive-control reproduction"
                )
            locked = {
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
            drift = [name for name, (actual, expected) in locked.items() if actual != expected]
            if drift:
                raise ValueError(
                    "formal PCSM constants are fixed; drifted fields: "
                    + ", ".join(drift)
                )
            if len(self.fresh_confirm_against_manifests) != FORMAL_FRESHNESS_SOURCE_COUNT:
                raise ValueError("formal PCSM requires exactly four freshness sources")
            resolved_sources = {
                str(Path(value).expanduser().resolve())
                for value in self.fresh_confirm_against_manifests
            }
            if len(resolved_sources) != FORMAL_FRESHNESS_SOURCE_COUNT:
                raise ValueError("formal PCSM freshness sources must be four distinct paths")
            output = Path(self.output_root).expanduser().resolve()
            for name, value in (("data_root", self.data_root), ("model_path", self.model_path)):
                protected = Path(value).expanduser().resolve()
                if (
                    output == protected
                    or output.is_relative_to(protected)
                    or protected.is_relative_to(output)
                ):
                    raise ValueError(
                        f"output_root must not overlap locked {name}: {output} vs {protected}"
                    )

    @property
    def update_per_class(self) -> int:
        return self.fit_per_class + self.calibration_per_class

    @property
    def fit_cycles(self) -> int:
        return self.optimizer_steps * self.batch_size // (2 * self.fit_per_class)


def _as_bap_config(config: PCSMConfig) -> bap.BAPConfig:
    """Project PCSM onto the structural configuration reused from BAP."""

    return bap.BAPConfig(
        data_root=config.data_root,
        model_path=config.model_path,
        output_root=config.output_root,
        task_name=config.task_name,
        learning_rate=config.learning_rate,
        fit_per_class=config.fit_per_class,
        calibration_per_class=config.calibration_per_class,
        tune_audit_per_class=config.tune_audit_per_class,
        confirm_audit_per_class=config.confirm_audit_per_class,
        checkpoint_steps=(config.optimizer_steps,),
        seed=config.seed,
        batch_size=config.batch_size,
        max_source_length=config.max_source_length,
        max_target_length=config.max_target_length,
        truncation_mode=config.truncation_mode,
        min_rescue_gain=config.min_raw_gain,
        max_class_drop=config.max_class_drop,
        tiny_random_model=config.tiny_random_model,
        dry_run=config.dry_run,
        confirm_selection=config.confirm_selection,
        run_pointwise_control=False,
        device=config.device,
        fresh_confirm_against_manifests=config.fresh_confirm_against_manifests,
    )


def _protocol_payload(config: PCSMConfig) -> dict[str, Any]:
    return {
        "format": FORMAT_VERSION,
        "claim_scope": "binary train-only acquisition qualification; not continual retention",
        "method": "Prior-Corrected Sequence-Mean LoRA (PCSM-LoRA)",
        "objective": {
            "name": OBJECTIVE,
            "per_example_loss": "sequence-mean target-token NLL of the observed label",
            "weight": "pinned natural train prior pi(y) divided by balanced sampler q(y)=0.5",
            "formula": "mean_i[(pi(y_i)/0.5) * sequence_mean_NLL(y_i|x_i)]",
            "sampler": "paired_fit_batches: each batch has equal True/False rows",
            "gradient_forward_mode": "eval with autograd; dropout disabled",
        },
        "fixed_trajectory": {
            "learning_rate": config.learning_rate,
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "batch_size": config.batch_size,
            "optimizer_steps": config.optimizer_steps,
            "eligible_checkpoint_steps": [config.optimizer_steps],
            "fit_cycles": config.fit_cycles,
            "seed": config.seed,
            "candidate_count": 1,
            "checkpoint_selection": False,
            "coefficient_scan": False,
        },
        "partition_roles": {
            "fit": "gradient only",
            "internal_calibration": "first qualification and threshold fitting only",
            "development_tune": "second qualification gate; opened only after internal pass",
            "confirm": "sealed until durable two-gate selection; one-shot scoring only",
        },
        "formal_freshness": {
            "source_count": FORMAL_FRESHNESS_SOURCE_COUNT,
            "composition": "three locked legacy split manifests plus one locked StageC access-only ledger",
            "expected_source_sha256s": sorted(FORMAL_FRESHNESS_SHA256S),
        },
        "primary_gate": {
            "metric": "raw official free-generation natural-prior EM gain",
            "gain": f"strictly greater than {config.min_raw_gain}",
            "post_valid_label_rate": f">= {config.min_valid_label_rate}",
            "minimum_raw_class_recall_delta": f">= {-config.max_class_drop}",
        },
        "secondary_gate": {
            "same_calibration_training_gain": "strictly greater than 0",
            "minimum_calibrated_class_recall_delta": f">= {-config.max_class_drop}",
            "margin_roc_auc_delta": "strictly greater than 0",
            "calibration_gain_counted_as_primary": False,
        },
        "confirmation": {
            "eligibility": "internal primary+secondary AND development primary+secondary",
            "primary_endpoint": "same raw official gate",
            "calibrated_and_auc_metrics": "reported with internal-frozen thresholds; never retuned on confirm",
        },
        "evidence_status": (
            "fixed-for-this-run reproduction of a previously observed BAP pointwise control; "
            "development is not independent discovery evidence"
        ),
    }


def _protocol_lock_payload(config: PCSMConfig) -> dict[str, Any]:
    payload = {
        "format": FORMAT_VERSION,
        "lock_scope": "run-local pre-model/pre-gradient; not external preregistration",
        "external_preregistration_claimed": False,
        "scientific_config": asdict(config),
        "protocol": _protocol_payload(config),
        "pilot_inventory": {
            "selection_eligible_arms": [OBJECTIVE],
            "arm_count": 1,
            "learning_rate_candidate_count": 1,
            "eligible_checkpoint_count": 1,
            "prior_bap_pointwise_observation_disclosed": True,
        },
    }
    return {**payload, "payload_sha256": bap._sha256_json(payload)}


def raw_official_comparison(
    base_raw: Mapping[str, Any], post_raw: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare raw official free-generation metrics without calibration."""

    for name, metrics in (("base", base_raw), ("post", post_raw)):
        required = {
            "natural_prior_em",
            "balanced_em",
            "per_class_recall",
            "free_generation_valid_label_rate",
        }
        missing = required - set(metrics)
        if missing:
            raise ValueError(f"{name} raw audit is missing {sorted(missing)}")
        if set(metrics["per_class_recall"]) != set(BINARY_LABELS):
            raise ValueError(f"{name} raw audit must cover True/False")
    deltas = {
        label: float(post_raw["per_class_recall"][label])
        - float(base_raw["per_class_recall"][label])
        for label in BINARY_LABELS
    }
    return {
        "base": dict(base_raw),
        "post": dict(post_raw),
        "natural_prior_gain": float(
            post_raw["natural_prior_em"] - base_raw["natural_prior_em"]
        ),
        "balanced_gain": float(post_raw["balanced_em"] - base_raw["balanced_em"]),
        "per_class_recall_delta": deltas,
        "minimum_per_class_delta": min(deltas.values()),
        "post_valid_label_rate": float(post_raw["free_generation_valid_label_rate"]),
        "metric_is_raw_official_free_generation": True,
        "calibration_involved": False,
    }


def primary_gate(comparison: Mapping[str, Any], *, config: PCSMConfig) -> bool:
    return (
        bap._strictly_above(float(comparison["natural_prior_gain"]), config.min_raw_gain)
        and bap._at_least(
            float(comparison["post_valid_label_rate"]), config.min_valid_label_rate
        )
        and bap._at_least(
            float(comparison["minimum_per_class_delta"]), -config.max_class_drop
        )
    )


def _rename_calibrated_comparison(comparison: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "base_same_calibration_control": dict(
            comparison["base_same_calibration_control"]
        ),
        "pcsm_same_calibration": dict(comparison["bap_calibrated"]),
        "training_contribution_natural_prior": float(
            comparison["training_contribution_natural_prior"]
        ),
        "training_contribution_balanced": float(
            comparison["training_contribution_balanced"]
        ),
        "training_per_class_recall_delta": dict(
            comparison["training_per_class_recall_delta"]
        ),
        "training_min_per_class_delta": float(
            comparison["training_min_per_class_delta"]
        ),
        "margin_diagnostics": dict(comparison["margin_diagnostics"]),
        "calibration_gain_is_not_primary_training_contribution": True,
        "thresholds_frozen_before_this_partition": bool(
            comparison.get("thresholds_frozen_before_this_partition", False)
        ),
    }


def calibrated_internal_comparison(
    base_margins: Sequence[float],
    post_margins: Sequence[float],
    references: Sequence[str],
    *,
    natural_prior: Mapping[str, float],
    config: PCSMConfig,
) -> dict[str, Any]:
    return _rename_calibrated_comparison(
        bap.calibrated_training_comparison(
            base_margins,
            post_margins,
            references,
            natural_prior=natural_prior,
            max_class_drop=config.max_class_drop,
        )
    )


def calibrated_frozen_comparison(
    base_margins: Sequence[float],
    post_margins: Sequence[float],
    references: Sequence[str],
    *,
    natural_prior: Mapping[str, float],
    base_threshold: float,
    post_threshold: float,
) -> dict[str, Any]:
    return _rename_calibrated_comparison(
        bap.apply_frozen_thresholds(
            base_margins,
            post_margins,
            references,
            natural_prior=natural_prior,
            base_threshold=base_threshold,
            post_threshold=post_threshold,
        )
    )


def secondary_gate(comparison: Mapping[str, Any], *, config: PCSMConfig) -> bool:
    return (
        bap._strictly_above(
            float(comparison["training_contribution_natural_prior"]), 0.0
        )
        and bap._at_least(
            float(comparison["training_min_per_class_delta"]),
            -config.max_class_drop,
        )
        and bap._strictly_above(
            float(comparison["margin_diagnostics"]["roc_auc_delta"]), 0.0
        )
    )


def _build_splits(
    examples: Sequence[OfficialExample], *, config: PCSMConfig
) -> tuple[rescue.RescueSplit, bap.BAPInternalSplit, tuple[Mapping[str, Any], ...]]:
    bap_config = _as_bap_config(config)
    outer, sources = bap._build_outer_split(examples, config=bap_config)
    internal = bap.build_bap_internal_split(outer, config=bap_config)
    _validate_formal_freshness_sources(sources, config=config)
    return outer, internal, sources


def _validate_formal_freshness_sources(
    sources: Sequence[Mapping[str, Any]], *, config: PCSMConfig
) -> None:
    """Bind the formal BoolQA run to the audited four-source history."""

    if config.tiny_random_model:
        return
    if len(sources) != FORMAL_FRESHNESS_SOURCE_COUNT:
        raise ValueError("formal PCSM freshness loader did not return four sources")
    source_hashes = {str(source.get("sha256", "")).lower() for source in sources}
    if source_hashes != FORMAL_FRESHNESS_SHA256S:
        raise ValueError("formal PCSM freshness source SHA-256 set changed")
    access_sources = [
        source
        for source in sources
        if source.get("format") == rescue.ACCESS_ONLY_FRESHNESS_FORMAT
    ]
    legacy_sources = [source for source in sources if "format" not in source]
    if len(access_sources) != 1 or len(legacy_sources) != 3:
        raise ValueError(
            "formal PCSM requires three legacy splits and one access-only ledger"
        )
    if any(source.get("task") != FORMAL_TASK for source in sources):
        raise ValueError("formal PCSM freshness source escaped BoolQA")


def _model_config(config: PCSMConfig) -> rescue.RescueConfig:
    # BAP's model config uses a balanced-loss placeholder because its bespoke
    # loss is outside RescueConfig.  Bind PCSM's actual objective explicitly.
    return replace(
        bap._model_config(_as_bap_config(config)),
        loss_objectives=(OBJECTIVE,),
        update_sampler="balanced",
    )


def _score_partition(
    bank: Any,
    tokenizer: Any,
    examples: Sequence[OfficialExample],
    *,
    config: PCSMConfig,
    natural_prior: Mapping[str, float],
) -> tuple[dict[str, Any], list[float]]:
    bap_config = _as_bap_config(config)
    raw = bap._raw_audit(
        bank,
        tokenizer,
        examples,
        config=bap_config,
        natural_prior=natural_prior,
    )
    margins = bap.collect_label_margins(
        bank, tokenizer, examples, config=bap_config
    )
    return raw, margins


def _train_fixed_trajectory(
    bank: Any,
    tokenizer: Any,
    fit_examples: Sequence[OfficialExample],
    *,
    config: PCSMConfig,
    natural_prior: Mapping[str, float],
) -> dict[str, Any]:
    """Run the sole fixed trajectory; there is no checkpoint search."""

    if (
        set(natural_prior) != set(BINARY_LABELS)
        or not math.isclose(
            sum(float(value) for value in natural_prior.values()), 1.0, abs_tol=1e-9
        )
        or any(float(value) <= 0 for value in natural_prior.values())
    ):
        raise ValueError("PCSM natural prior must be positive, binary, and sum to one")
    bap_config = _as_bap_config(config)
    bank.eval()
    optimizer = torch.optim.AdamW(
        tuple(bank.all_atom_parameters()), lr=config.learning_rate, weight_decay=0.0
    )
    losses: list[float] = []
    exposure_ids: list[str] = []
    per_class = {label: 0 for label in BINARY_LABELS}
    source_tokens = 0
    target_tokens = 0
    observed_steps = 0
    for step, batch in bap.paired_fit_batches(
        fit_examples,
        batch_size=config.batch_size,
        steps=config.optimizer_steps,
        seed=config.seed,
    ):
        batch_counts = {
            label: sum(item.label == label for item in batch)
            for label in BINARY_LABELS
        }
        if batch_counts != {
            label: config.batch_size // 2 for label in BINARY_LABELS
        }:
            raise RuntimeError("PCSM received a non-paired gradient batch")
        loss, step_ledger = bap._pointwise_sequence_prior_step(
            bank,
            tokenizer,
            batch,
            optimizer,
            config=bap_config,
            natural_prior=natural_prior,
        )
        if not math.isfinite(float(loss)):
            raise FloatingPointError(f"non-finite PCSM loss at step {step}")
        observed_steps = step
        losses.append(float(loss))
        exposure_ids.extend(item.example_id for item in batch)
        for item in batch:
            per_class[item.label] += 1
        source_tokens += int(step_ledger["source_tokens_processed"])
        target_tokens += int(step_ledger["target_tokens_processed"])
    expected_per_class = config.optimizer_steps * config.batch_size // 2
    if (
        observed_steps != config.optimizer_steps
        or len(exposure_ids) != config.optimizer_steps * config.batch_size
        or per_class != {label: expected_per_class for label in BINARY_LABELS}
    ):
        raise RuntimeError("PCSM fixed trajectory violated its exact resource budget")
    return {
        "objective": OBJECTIVE,
        "optimizer": "AdamW",
        "learning_rate": config.learning_rate,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "optimizer_steps": observed_steps,
        "eligible_checkpoint_steps": [config.optimizer_steps],
        "checkpoint_candidates_evaluated": 1,
        "gradient_example_exposures": len(exposure_ids),
        "scored_label_sequences": len(exposure_ids),
        "paired_exposures_per_class": per_class,
        "fit_rows_per_class": config.fit_per_class,
        "complete_fit_cycles": config.fit_cycles,
        "batch_size": config.batch_size,
        "paired_schedule_seed": config.seed,
        "ordered_exposure_ids_sha256": bap._sha256_json(exposure_ids),
        "source_tokens_processed": source_tokens,
        "target_tokens_processed": target_tokens,
        "mean_loss": float(np.mean(losses)),
        "gradient_forward_mode": "eval (gradients enabled; dropout disabled)",
        "dropout_disabled": True,
        "exact_budget": True,
    }


def _dry_run_payload(
    config: PCSMConfig,
    outer: rescue.RescueSplit,
    internal: bap.BAPInternalSplit,
    distribution: Mapping[str, Any],
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
            bap.bap_internal_split_manifest(internal)
        ),
        "pinned_train_label_distribution": dict(distribution),
        "model_or_tokenizer_constructed": False,
        "gradient_steps": 0,
        "development_tune_accessed": False,
        "confirm_audit_model_scored_or_tokenized": False,
    }


def _write_failure(
    *,
    output_root: Path,
    task_dir: Path,
    manifest: dict[str, Any],
    config: PCSMConfig,
    resource_ledger: Mapping[str, Any],
    internal_primary: Mapping[str, Any],
    internal_secondary: Mapping[str, Any],
    primary_pass: bool,
    secondary_pass: bool,
) -> dict[str, Any]:
    failure = {
        "format": FORMAT_VERSION,
        "status": "internal_qualification_failed",
        "task": config.task_name,
        "only_eligible_step": config.optimizer_steps,
        "resource_ledger": dict(resource_ledger),
        "internal_calibration": {
            "primary_raw_official": dict(internal_primary),
            "secondary_calibrated_and_auc": dict(internal_secondary),
            "primary_pass": primary_pass,
            "secondary_pass": secondary_pass,
            "qualification_pass": False,
        },
        "development_tune_accessed": False,
        "confirm_eligible": False,
        "confirm_audit_model_scored_or_tokenized": False,
    }
    rescue._atomic_json(task_dir / "qualification_failure.json", failure)
    manifest.update(
        {
            "status": "internal_qualification_failed",
            "resource_ledger": dict(resource_ledger),
            "development_tune_accessed": False,
        }
    )
    rescue._atomic_json(output_root / "manifest.json", manifest)
    (output_root / "QUALIFICATION_FAILED").write_text(
        "PCSM internal gate failed; development and confirm remained unscored\n",
        encoding="utf-8",
    )
    return failure


def run_selection(
    config: PCSMConfig,
    *,
    train_examples: Sequence[OfficialExample] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Run one fixed PCSM trajectory and at most two qualification audits."""

    if config.confirm_selection is not None:
        raise ValueError("run_selection cannot execute confirmation mode")
    output_root = Path(config.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing non-empty PCSM output root {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    task_dir = output_root / config.task_name
    task_dir.mkdir(parents=True, exist_ok=False)

    injected_examples = train_examples is not None
    if injected_examples and not config.tiny_random_model:
        raise ValueError("formal PCSM selection forbids train_examples injection")
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
    distribution = rescue.pinned_train_label_distribution(
        train_examples, task_name=config.task_name
    )
    natural_prior = distribution["natural_label_prior"]
    if set(natural_prior) != set(BINARY_LABELS) or not math.isclose(
        sum(float(value) for value in natural_prior.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("pinned natural prior must cover True/False and sum to one")
    outer, internal, freshness_sources = _build_splits(train_examples, config=config)
    outer_manifest = rescue._split_manifest(outer)
    internal_manifest = bap.bap_internal_split_manifest(internal)
    rescue._atomic_json(task_dir / "split.json", outer_manifest)
    rescue._atomic_json(task_dir / "internal_split.json", internal_manifest)

    protocol_lock_path = task_dir / "protocol_lock.json"
    protocol_lock_payload = _protocol_lock_payload(config)
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
        "freshness_sources": [dict(item) for item in freshness_sources],
        "freshness_source_count": len(freshness_sources),
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
        payload = _dry_run_payload(config, outer, internal, distribution)
        rescue._atomic_json(output_root / "dry_run.json", payload)
        manifest["status"] = "dry_run_complete"
        rescue._atomic_json(output_root / "manifest.json", manifest)
        (output_root / "DRY_RUN_COMPLETE").write_text(
            "PCSM dry run completed; no model, gradient, or audit scoring\n",
            encoding="utf-8",
        )
        return payload

    injected_tokenizer = tokenizer is not None
    if injected_tokenizer and not config.tiny_random_model:
        raise ValueError("formal PCSM selection forbids tokenizer injection")
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
        raise RuntimeError("PCSM model violates the fixed adapter budget")

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
    internal_primary = raw_official_comparison(
        base_internal_raw, post_internal_raw
    )
    internal_secondary = calibrated_internal_comparison(
        base_internal_margins,
        post_internal_margins,
        [item.label for item in internal.calibration],
        natural_prior=natural_prior,
        config=config,
    )
    internal_primary_pass = primary_gate(internal_primary, config=config)
    internal_secondary_pass = secondary_gate(internal_secondary, config=config)
    if not (internal_primary_pass and internal_secondary_pass):
        failure = _write_failure(
            output_root=output_root,
            task_dir=task_dir,
            manifest=manifest,
            config=config,
            resource_ledger=resource_ledger,
            internal_primary=internal_primary,
            internal_secondary=internal_secondary,
            primary_pass=internal_primary_pass,
            secondary_pass=internal_secondary_pass,
        )
        rescue._release_model(bank)
        return failure

    checkpoints_dir = task_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = checkpoints_dir / f"step_{config.optimizer_steps}.pt"
    rescue._atomic_torch_save(checkpoint_path, bank.compact_state_dict())
    checkpoint = {
        "path": str(checkpoint_path.resolve()),
        "sha256": rescue._sha256_file(checkpoint_path),
        "optimizer_step": config.optimizer_steps,
        "sole_eligible_checkpoint": True,
    }

    # Make the internal decision durable before any development row is scored.
    internal_lock_path = task_dir / "internal_qualification_lock.json"
    internal_lock_core = {
        "format": FORMAT_VERSION,
        "status": "internal_qualification_locked_before_development_access",
        "protocol_lock_sha256": protocol_lock["sha256"],
        "outer_split_manifest_sha256": rescue._atomic_json_payload_sha256(
            outer_manifest
        ),
        "internal_split_manifest_sha256": rescue._atomic_json_payload_sha256(
            internal_manifest
        ),
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
        raise RuntimeError("PCSM frozen development base initialization drifted")
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
    dev_primary = raw_official_comparison(base_dev_raw, post_dev_raw)
    base_threshold = float(
        internal_secondary["base_same_calibration_control"]["threshold"]
    )
    post_threshold = float(internal_secondary["pcsm_same_calibration"]["threshold"])
    dev_secondary = calibrated_frozen_comparison(
        base_dev_margins,
        post_dev_margins,
        [item.label for item in outer.tune_audit],
        natural_prior=natural_prior,
        base_threshold=base_threshold,
        post_threshold=post_threshold,
    )
    dev_primary_pass = primary_gate(dev_primary, config=config)
    dev_secondary_pass = secondary_gate(dev_secondary, config=config)
    confirm_eligible = dev_primary_pass and dev_secondary_pass

    selection = {
        "format": FORMAT_VERSION,
        "status": "selection_locked",
        "task": config.task_name,
        "scientific_config": asdict(config),
        "protocol": _protocol_payload(config),
        "selected": {
            "objective": OBJECTIVE,
            "step": config.optimizer_steps,
            "selection_rule": "sole fixed step; no checkpoint maximization",
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
            "role": "second qualification gate",
            "accessed_only_after_internal_pass": True,
            "thresholds_frozen_before_access": True,
            "primary_raw_official": dev_primary,
            "secondary_calibrated_and_auc": dev_secondary,
            "primary_pass": dev_primary_pass,
            "secondary_pass": dev_secondary_pass,
            "qualification_pass": confirm_eligible,
            "did_not_select_checkpoint_or_threshold": True,
        },
        "confirm_eligible": confirm_eligible,
        "confirm_ineligibility_reason": (
            None
            if confirm_eligible
            else "development primary and/or secondary qualification gate failed"
        ),
        "outer_split_manifest_sha256": rescue._atomic_json_payload_sha256(
            outer_manifest
        ),
        "internal_split_manifest_sha256": rescue._atomic_json_payload_sha256(
            internal_manifest
        ),
        "freshness_sources": [dict(item) for item in freshness_sources],
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
        "PCSM selection completed; confirm remains model-unscored and untokenized\n",
        encoding="utf-8",
    )
    rescue._release_model(bank)
    return selection


def load_selection(path: str | os.PathLike[str]) -> tuple[dict[str, Any], Path]:
    selection_path = Path(path).expanduser().resolve()
    try:
        with selection_path.open("r", encoding="utf-8") as handle:
            selection = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load PCSM selection {selection_path}") from error
    if not isinstance(selection, dict) or selection.get("format") != FORMAT_VERSION:
        raise ValueError("selection is not a PCSM-LoRA v1 artifact")
    if selection.get("status") != "selection_locked":
        raise ValueError("PCSM selection is not durably locked")
    required_mappings = {
        "scientific_config",
        "protocol",
        "selected",
        "selected_checkpoint",
        "resource_ledger",
        "internal_calibration",
        "internal_qualification_lock",
        "development_tune",
        "model_protocol",
    }
    missing = [name for name in required_mappings if not isinstance(selection.get(name), Mapping)]
    if missing:
        raise ValueError("PCSM selection lacks mappings: " + ", ".join(sorted(missing)))
    return selection, selection_path


def _config_from_selection(
    selection: Mapping[str, Any],
    *,
    output_root: str,
    selection_path: Path,
    device: str | None = None,
) -> PCSMConfig:
    payload = selection.get("scientific_config")
    if not isinstance(payload, Mapping):
        raise ValueError("PCSM selection lacks scientific_config")
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
    return PCSMConfig(**values)


def _validate_confirmation_config(
    config: PCSMConfig, selection: Mapping[str, Any]
) -> None:
    locked = selection.get("scientific_config")
    if not isinstance(locked, Mapping):
        raise ValueError("PCSM selection lacks scientific_config")
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
            "confirmation scientific config differs from locked selection: "
            + ", ".join(differing)
        )


def _preflight_confirmation(
    selection: Mapping[str, Any],
    selection_path: Path,
    outer: rescue.RescueSplit,
    internal: bap.BAPInternalSplit,
    distribution: Mapping[str, Any],
    *,
    config: PCSMConfig,
) -> dict[str, Any]:
    """Fail closed on every locked resource before confirm tokenization/scoring."""

    if selection.get("confirm_eligible") is not True:
        raise ValueError("PCSM selection is not eligible for sealed confirmation")
    if selection.get("confirm_audit_model_scored_or_tokenized") is not False:
        raise ValueError("PCSM selection does not attest a sealed confirm partition")
    for partition in ("internal_calibration", "development_tune"):
        payload = selection[partition]
        if (
            not isinstance(payload.get("primary_raw_official"), Mapping)
            or not isinstance(payload.get("secondary_calibrated_and_auc"), Mapping)
            or not primary_gate(payload["primary_raw_official"], config=config)
            or not secondary_gate(
                payload["secondary_calibrated_and_auc"], config=config
            )
            or payload.get("primary_pass") is not True
            or payload.get("secondary_pass") is not True
            or payload.get("qualification_pass") is not True
        ):
            raise ValueError(f"PCSM {partition} did not pass both locked gates")
    selected = selection["selected"]
    config_payload = selection["scientific_config"]
    locked_step = int(config_payload["optimizer_steps"])
    if (
        selected.get("objective") != OBJECTIVE
        or int(selected.get("step", -1)) != locked_step
        or selected.get("selection_rule")
        != "sole fixed step; no checkpoint maximization"
    ):
        raise ValueError("PCSM selected objective/step protocol is invalid")
    for key in ("base_threshold", "pcsm_threshold"):
        if not math.isfinite(float(selected[key])):
            raise ValueError(f"PCSM selected {key} is non-finite")
    locked_internal = selection["internal_calibration"][
        "secondary_calibrated_and_auc"
    ]
    if not math.isclose(
        float(selected["base_threshold"]),
        float(locked_internal["base_same_calibration_control"]["threshold"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(selected["pcsm_threshold"]),
        float(locked_internal["pcsm_same_calibration"]["threshold"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("PCSM selected thresholds differ from internal calibration")

    selection_sha = rescue._sha256_file(selection_path)
    lock_path = selection_path.parent / "selection_lock.json"
    if not lock_path.is_file():
        raise ValueError("PCSM selection lock is missing")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load PCSM selection lock") from error
    checkpoint_payload = selection["selected_checkpoint"]
    if (
        lock.get("format") != FORMAT_VERSION
        or Path(str(lock.get("selection_path", ""))).expanduser().resolve()
        != selection_path.resolve()
        or lock.get("selection_sha256") != selection_sha
        or lock.get("selected_checkpoint_sha256") != checkpoint_payload.get("sha256")
        or int(lock.get("selected_step", -1)) != locked_step
        or lock.get("internal_qualification_pass") is not True
        or lock.get("development_qualification_pass") is not True
        or lock.get("confirm_eligible") is not True
        or lock.get("confirm_audit_model_scored_or_tokenized") is not False
    ):
        raise ValueError("PCSM selection lock does not bind the selection")

    protocol_metadata = selection["run_local_protocol_lock"]
    if not isinstance(protocol_metadata, Mapping):
        raise ValueError("PCSM selection lacks its protocol lock")
    protocol_path = (selection_path.parent / "protocol_lock.json").resolve()
    if (
        Path(str(protocol_metadata.get("path", ""))).expanduser().resolve()
        != protocol_path
        or not protocol_path.is_file()
        or rescue._sha256_file(protocol_path) != protocol_metadata.get("sha256")
        or protocol_metadata.get("external_preregistration_claimed") is not False
    ):
        raise ValueError("PCSM protocol lock path or SHA-256 changed")
    try:
        protocol_lock = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load PCSM protocol lock") from error
    protocol_without_hash = dict(protocol_lock)
    recorded_protocol_sha = protocol_without_hash.pop("payload_sha256", None)
    if (
        recorded_protocol_sha != bap._sha256_json(protocol_without_hash)
        or recorded_protocol_sha != protocol_metadata.get("payload_sha256")
        or bap._sha256_json(protocol_lock.get("scientific_config"))
        != bap._sha256_json(selection.get("scientific_config"))
        or bap._sha256_json(protocol_lock.get("protocol"))
        != bap._sha256_json(selection.get("protocol"))
    ):
        raise ValueError("PCSM protocol lock payload changed")

    checkpoint = Path(str(checkpoint_payload.get("path", ""))).expanduser().resolve()
    expected_checkpoint = (
        selection_path.parent / "checkpoints" / f"step_{locked_step}.pt"
    ).resolve()
    if checkpoint != expected_checkpoint or not checkpoint.is_file():
        raise ValueError("PCSM selected checkpoint path is invalid")
    checkpoint_sha = rescue._sha256_file(checkpoint)
    if checkpoint_sha != checkpoint_payload.get("sha256"):
        raise ValueError("PCSM selected checkpoint SHA-256 mismatch")
    if (
        int(checkpoint_payload.get("optimizer_step", -1)) != locked_step
        or checkpoint_payload.get("sole_eligible_checkpoint") is not True
    ):
        raise ValueError("PCSM selected checkpoint metadata violates the sole-step rule")

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
        raise ValueError("PCSM internal qualification lock path or SHA-256 changed")
    try:
        internal_lock = json.loads(internal_lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load PCSM internal qualification lock") from error
    internal_without_hash = dict(internal_lock)
    recorded_internal_sha = internal_without_hash.pop("payload_sha256", None)
    locked_internal_result = selection["internal_calibration"]
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
        != locked_internal_result.get("primary_raw_official")
        or internal_lock.get("secondary_calibrated_and_auc")
        != locked_internal_result.get("secondary_calibrated_and_auc")
        or internal_lock.get("primary_pass") is not True
        or internal_lock.get("secondary_pass") is not True
        or internal_lock.get("development_tune_accessed") is not False
        or internal_lock.get("confirm_audit_model_scored_or_tokenized") is not False
    ):
        raise ValueError("PCSM internal qualification lock payload changed")
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
        raise ValueError("PCSM resource ledger violates the fixed trajectory")
    reconstructed_exposure_ids = [
        item.example_id
        for _step, batch in bap.paired_fit_batches(
            internal.fit,
            batch_size=int(config_payload["batch_size"]),
            steps=locked_step,
            seed=int(config_payload["seed"]),
        )
        for item in batch
    ]
    if ledger.get("ordered_exposure_ids_sha256") != bap._sha256_json(
        reconstructed_exposure_ids
    ):
        raise ValueError("PCSM ordered exposure schedule changed")

    outer_sha = rescue._atomic_json_payload_sha256(rescue._split_manifest(outer))
    internal_sha = rescue._atomic_json_payload_sha256(
        bap.bap_internal_split_manifest(internal)
    )
    if outer_sha != selection.get("outer_split_manifest_sha256"):
        raise ValueError("reconstructed PCSM outer split does not match selection")
    if internal_sha != selection.get("internal_split_manifest_sha256"):
        raise ValueError("reconstructed PCSM internal split does not match selection")
    if distribution != selection.get("pinned_train_label_distribution"):
        raise ValueError("pinned train label distribution changed after selection")

    model_protocol = selection["model_protocol"]
    current_model_artifact = bap._local_artifact_fingerprint(
        str(config_payload["model_path"])
    )
    if current_model_artifact != model_protocol.get(
        "frozen_model_tokenizer_artifact"
    ):
        raise ValueError("frozen model/tokenizer artifact changed after selection")
    if model_protocol.get("rescue_loss_objective") != OBJECTIVE:
        raise ValueError("PCSM model protocol objective changed")

    manifest_path = selection_path.parent.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("PCSM selection manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load PCSM selection manifest") from error
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
    ):
        raise ValueError("PCSM manifest does not bind the eligible selection")
    return {
        "selection_sha256": selection_sha,
        "selection_lock": lock,
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
            "confirm row metadata was used to reconstruct the locked split; no "
            "confirm row was model-scored or tokenized before this claim"
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
            "this PCSM selection already has a confirmation claim; refusing a second confirm read"
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
        raise RuntimeError("PCSM confirmation claim changed during scoring")
    core = dict(current)
    recorded = core.pop("claimed_payload_sha256", None)
    if recorded != bap._sha256_json(core):
        raise RuntimeError("PCSM confirmation claim payload hash is invalid")
    completed = {
        **current,
        "status": "confirmation_completed",
        "initial_claim_payload_sha256": recorded,
        "confirmation_result": str(result_path.resolve()),
        "confirmation_result_sha256": rescue._sha256_file(result_path),
    }
    rescue._atomic_json(claim_path, completed)


def run_confirmation(
    config: PCSMConfig,
    selection: Mapping[str, Any],
    selection_path: Path,
    *,
    train_examples: Sequence[OfficialExample] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Score the sealed outer confirm partition exactly once."""

    disk_selection, disk_path = load_selection(selection_path)
    if bap._sha256_json(selection) != bap._sha256_json(disk_selection):
        raise ValueError("in-memory selection differs from locked on-disk selection")
    selection = disk_selection
    selection_path = disk_path
    _validate_confirmation_config(config, selection)
    output_root = Path(config.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing non-empty PCSM confirmation root {output_root}")
    if (selection_path.parent / "confirmation_claim.json").exists():
        raise RuntimeError(
            "this PCSM selection already has a confirmation claim; refusing a second confirm read"
        )

    injected_examples = train_examples is not None
    if injected_examples and not config.tiny_random_model:
        raise ValueError("formal PCSM confirmation forbids train_examples injection")
    if injected_examples != bool(
        selection.get("train_examples_injected_for_tiny_test", False)
    ):
        raise ValueError("confirmation data-loading mode differs from selection")
    current_data_artifact = (
        None
        if injected_examples
        else bap._official_train_artifact_fingerprint(
            config.data_root, task_name=config.task_name
        )
    )
    if current_data_artifact != selection.get("verified_data_artifact"):
        raise ValueError("verified official data artifact changed after PCSM selection")
    if train_examples is None:
        train_examples = load_official_examples(
            config.data_root, config.task_name, "train", verify=True
        )
    corpus_fingerprint = bap._train_corpus_fingerprint(
        train_examples, task_name=config.task_name
    )
    if corpus_fingerprint != selection.get("train_corpus_fingerprint"):
        raise ValueError("train corpus content changed after PCSM selection")
    execution_environment = bap._execution_environment_fingerprint(config.device)
    if execution_environment != selection.get("execution_environment"):
        raise ValueError("confirmation numerical execution environment drifted")
    distribution = rescue.pinned_train_label_distribution(
        train_examples, task_name=config.task_name
    )
    natural_prior = distribution["natural_label_prior"]
    outer, internal, freshness_sources = _build_splits(train_examples, config=config)
    if [dict(item) for item in freshness_sources] != selection.get("freshness_sources"):
        raise ValueError("fresh-confirm source manifests changed after PCSM selection")
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
        raise ValueError("formal PCSM confirmation forbids tokenizer injection")
    if injected_tokenizer != bool(
        selection["model_protocol"].get("tokenizer_injected_for_tiny_test", False)
    ):
        raise ValueError("confirmation tokenizer-loading mode differs from selection")
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, use_fast=True, local_files_only=True
        )
    tokenizer_fingerprint = bap._tokenizer_protocol_fingerprint(tokenizer)
    if tokenizer_fingerprint != selection["model_protocol"].get(
        "tokenizer_protocol_fingerprint"
    ):
        raise ValueError("tokenizer protocol changed after PCSM selection")

    # Build and validate every non-confirm resource before burning the claim.
    model_config = _model_config(config)
    expected_init = selection["model_protocol"]["initial_adapter_fingerprint"]
    base_bank = rescue._build_model(model_config, tokenizer)
    if rescue._adapter_fingerprint(base_bank) != expected_init:
        raise ValueError("PCSM confirmation base initialization fingerprint changed")
    post_bank = rescue._build_model(model_config, tokenizer)
    if rescue._adapter_fingerprint(post_bank) != expected_init:
        raise ValueError("PCSM confirmation adapter initialization fingerprint changed")
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
    confirm_primary = raw_official_comparison(base_raw, post_raw)
    confirm_primary_pass = primary_gate(confirm_primary, config=config)
    confirm_secondary = calibrated_frozen_comparison(
        base_margins,
        post_margins,
        [item.label for item in outer.confirm_audit],
        natural_prior=natural_prior,
        base_threshold=float(selection["selected"]["base_threshold"]),
        post_threshold=float(selection["selected"]["pcsm_threshold"]),
    )
    confirm_secondary_diagnostic_pass = secondary_gate(
        confirm_secondary, config=config
    )
    result = {
        "format": FORMAT_VERSION,
        "status": "confirmation_completed",
        "task": config.task_name,
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
        "thresholds_frozen_on_internal_calibration": {
            "base": float(selection["selected"]["base_threshold"]),
            "pcsm": float(selection["selected"]["pcsm_threshold"]),
        },
        "calibration_gain_counted_as_primary": False,
        "confirm_rows_used_for_checkpoint_threshold_or_coefficient_selection": False,
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
        "PCSM independent one-shot confirmation completed\n", encoding="utf-8"
    )
    rescue._release_model(base_bank)
    rescue._release_model(post_bank)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument("--model-path")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task", dest="task_name")
    parser.add_argument("--max-source-length", type=int)
    parser.add_argument("--max-target-length", type=int)
    parser.add_argument(
        "--truncation-mode", choices=bap.TRUNCATION_MODES
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-selection")
    parser.add_argument("--device")
    parser.add_argument(
        "--fresh-confirm-against-manifest",
        dest="fresh_confirm_against_manifests",
        action="append",
        default=[],
    )
    return parser


def _selection_config_from_args(args: argparse.Namespace) -> PCSMConfig:
    if args.data_root is None or args.model_path is None:
        raise ValueError("selection requires --data-root and --model-path")
    return PCSMConfig(
        data_root=args.data_root,
        model_path=args.model_path,
        output_root=args.output_root,
        task_name="BoolQA" if args.task_name is None else args.task_name,
        max_source_length=(
            512 if args.max_source_length is None else args.max_source_length
        ),
        max_target_length=(
            50 if args.max_target_length is None else args.max_target_length
        ),
        truncation_mode=(
            "field_aware" if args.truncation_mode is None else args.truncation_mode
        ),
        dry_run=args.dry_run,
        device="cpu" if args.device is None else args.device,
        fresh_confirm_against_manifests=tuple(
            args.fresh_confirm_against_manifests
        ),
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
                raise ValueError("confirmation data_root differs from locked selection")
            if args.model_path is not None and (
                Path(args.model_path).expanduser().resolve()
                != Path(config.model_path).expanduser().resolve()
            ):
                raise ValueError("confirmation model_path differs from locked selection")
            if args.fresh_confirm_against_manifests and tuple(
                args.fresh_confirm_against_manifests
            ) != config.fresh_confirm_against_manifests:
                raise ValueError("confirmation freshness sources differ from locked selection")
            if args.task_name is not None and args.task_name != config.task_name:
                raise ValueError("confirmation task differs from locked selection")
            if (
                args.max_source_length is not None
                and args.max_source_length != config.max_source_length
            ):
                raise ValueError(
                    "confirmation max_source_length differs from locked selection"
                )
            if (
                args.max_target_length is not None
                and args.max_target_length != config.max_target_length
            ):
                raise ValueError(
                    "confirmation max_target_length differs from locked selection"
                )
            if (
                args.truncation_mode is not None
                and args.truncation_mode != config.truncation_mode
            ):
                raise ValueError(
                    "confirmation truncation_mode differs from locked selection"
                )
            if args.dry_run:
                raise ValueError("--dry-run is incompatible with confirmation")
            run_confirmation(config, selection, selection_path)
    except (ValueError, FileExistsError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
