"""Locked PCSM adaptive-endpoint v2 transfer protocol.

This runner leaves the frozen PCSM/transfer-v1 implementation untouched.  It
keeps the exact v1 training trajectory and adds one train-derived, ordered
endpoint decision.  Raw free generation has priority.  A calibrated endpoint
is eligible only when raw acquisition and representation checks succeeded and
the sole raw failure is the predeclared class-safety guard.  The selected
endpoint and both internal thresholds are durably locked before development is
opened and can never be switched on development or confirmation.

The method-development protocol is content-addressed in
``notes/phase2i_pcsm_adaptive_endpoint_v2_protocol.md``.  It is a run-local
lock, not an external preregistration; QQP internal metrics are discovery
evidence under that protocol.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
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
    from . import run_pcsm_lora_rescue as pcsm
    from . import run_pcsm_lora_transfer as transfer_v1
    from .order4_data import OfficialExample, TASK_BY_NAME, load_official_examples
except ImportError:  # pragma: no cover - direct script execution.
    import run_acquisition_rescue as rescue  # type: ignore
    import run_bap_lora_rescue as bap  # type: ignore
    import run_pcsm_lora_rescue as pcsm  # type: ignore
    import run_pcsm_lora_transfer as transfer_v1  # type: ignore
    from order4_data import (  # type: ignore
        OfficialExample,
        TASK_BY_NAME,
        load_official_examples,
    )


FORMAT_VERSION = "phase2i.binary-pcsm-adaptive-endpoint-transfer.v2"
ENDPOINT_RULE_VERSION = "phase2i.pcsm-adaptive-endpoint-rule.v2"
TASK_PROTOCOL_REGISTRY_FORMAT = (
    "phase2i.pcsm-adaptive-endpoint-transfer-task-registry.v2"
)
RAW_ENDPOINT = "raw"
CALIBRATED_ENDPOINT = "internal_frozen_calibrated"
ENDPOINTS = (RAW_ENDPOINT, CALIBRATED_ENDPOINT)

TRANSFER_TASKS = transfer_v1.TRANSFER_TASKS
BINARY_LABELS = pcsm.BINARY_LABELS
OBJECTIVE = pcsm.OBJECTIVE
TransferConfig = transfer_v1.TransferConfig

FROZEN_TRANSFER_V1_SHA256 = (
    "30b92b26bf9dd0d4e6f2cbc86e76834c74707e4c7ecf2ea8e498524fdd2b7167"
)
FROZEN_PROTOCOL_SHA256 = (
    "c0bb4f10d17fe10afb30fdb499549492a2191d8e391ed3544e8610a34b51b040"
)
FROZEN_EXECUTION_LOCK_SHA256 = (
    "1051bf4de41634714430943ab7cd9b4c54e4420f71031739b96523fa0662f42d"
)
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 43002
BOOTSTRAP_INTERVAL_LEVEL = 0.95
EVALUATION_MICROBATCH_SIZE = 2


def _assert_frozen_uncertainty_protocol() -> None:
    """Reject runtime monkeypatching of the locked confirmation estimand."""

    if (
        BOOTSTRAP_SEED != 43002
        or BOOTSTRAP_REPLICATES != 10000
        or not math.isclose(
            BOOTSTRAP_INTERVAL_LEVEL, 0.95, rel_tol=0.0, abs_tol=0.0
        )
    ):
        raise RuntimeError(
            "frozen uncertainty protocol drifted from seed=43002, "
            "replicates=10000, interval_level=0.95"
        )


def _protocol_note_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "notes"
        / "phase2i_pcsm_adaptive_endpoint_v2_protocol.md"
    )


def _execution_lock_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "notes"
        / "phase2i_pcsm_adaptive_endpoint_v2_execution_lock.md"
    )


def _dependency_fingerprints() -> dict[str, Any]:
    """Verify every frozen implementation/protocol dependency and bind v2."""

    _assert_frozen_uncertainty_protocol()

    module_dir = Path(__file__).resolve().parent
    expected = {
        "pcsm_v1": (
            Path(pcsm.__file__).resolve(),
            transfer_v1.FROZEN_PCSM_SHA256,
        ),
        "bap_v1": (
            Path(bap.__file__).resolve(),
            transfer_v1.FROZEN_BAP_SHA256,
        ),
        "shared_rescue": (
            Path(rescue.__file__).resolve(),
            transfer_v1.FROZEN_SHARED_SHA256,
        ),
        "order4_data": (
            module_dir / "order4_data.py",
            transfer_v1.FROZEN_ORDER4_DATA_SHA256,
        ),
        "objectives": (
            module_dir / "objectives.py",
            transfer_v1.FROZEN_OBJECTIVES_SHA256,
        ),
        "t5_rank_bank": (
            module_dir / "t5_rank_bank.py",
            transfer_v1.FROZEN_T5_RANK_BANK_SHA256,
        ),
        "transfer_v1": (
            Path(transfer_v1.__file__).resolve(),
            FROZEN_TRANSFER_V1_SHA256,
        ),
        "adaptive_endpoint_protocol": (
            _protocol_note_path(),
            FROZEN_PROTOCOL_SHA256,
        ),
        "adaptive_endpoint_execution_lock": (
            _execution_lock_path(),
            FROZEN_EXECUTION_LOCK_SHA256,
        ),
    }
    result: dict[str, Any] = {}
    for name, (path, expected_sha) in expected.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen adaptive-endpoint dependency: {path}")
        current_sha = rescue._sha256_file(path).lower()
        if current_sha != expected_sha:
            raise ValueError(f"frozen adaptive-endpoint dependency {name} SHA-256 changed")
        result[name] = {
            "path": str(path),
            "sha256": current_sha,
            "expected_sha256": expected_sha,
        }
    runner_path = Path(__file__).resolve()
    result["adaptive_endpoint_v2_runner"] = {
        "path": str(runner_path),
        "sha256": rescue._sha256_file(runner_path).lower(),
        "binding": "selection-time implementation identity; rechecked before confirm",
    }
    return result


def _require_finite_metrics(**values: float) -> None:
    invalid = sorted(name for name, value in values.items() if not math.isfinite(value))
    if invalid:
        raise ValueError("adaptive endpoint metrics must be finite: " + ", ".join(invalid))


def _common_endpoint_checks(
    raw: Mapping[str, Any],
    calibrated: Mapping[str, Any],
    *,
    config: TransferConfig,
) -> dict[str, Any]:
    raw_valid = float(raw["post_valid_label_rate"])
    auc_delta = float(calibrated["margin_diagnostics"]["roc_auc_delta"])
    _require_finite_metrics(raw_valid=raw_valid, roc_auc_delta=auc_delta)
    valid_pass = bap._at_least(raw_valid, config.min_valid_label_rate)
    auc_pass = bap._strictly_above(auc_delta, 0.0)
    ledger = {
        "post_raw_free_generation_valid_label_rate": raw_valid,
        "minimum_post_raw_valid_label_rate": config.min_valid_label_rate,
        "raw_validity_source": "raw_free_generation.post_valid_label_rate",
        "raw_validity_pass": valid_pass,
        "margin_roc_auc_delta": auc_delta,
        "minimum_margin_roc_auc_delta": "strictly greater than 0",
        "margin_roc_auc_pass": auc_pass,
        "common_checks_pass": valid_pass and auc_pass,
    }
    return ledger


def raw_endpoint_gate_audit(
    raw: Mapping[str, Any],
    calibrated: Mapping[str, Any],
    *,
    config: TransferConfig,
) -> dict[str, Any]:
    common = _common_endpoint_checks(raw, calibrated, config=config)
    gain = float(raw["natural_prior_gain"])
    minimum_class_delta = float(raw["minimum_per_class_delta"])
    _require_finite_metrics(
        raw_natural_prior_gain=gain,
        raw_minimum_per_class_delta=minimum_class_delta,
    )
    gain_pass = bap._strictly_above(gain, config.min_raw_gain)
    class_pass = bap._at_least(minimum_class_delta, -config.max_class_drop)
    return {
        "endpoint": RAW_ENDPOINT,
        "common_checks": common,
        "natural_prior_gain": gain,
        "minimum_natural_prior_gain": f"strictly greater than {config.min_raw_gain}",
        "natural_prior_gain_pass": gain_pass,
        "minimum_per_class_recall_delta": minimum_class_delta,
        "minimum_allowed_per_class_recall_delta": -config.max_class_drop,
        "class_safety_pass": class_pass,
        "gate_pass": bool(common["common_checks_pass"] and gain_pass and class_pass),
    }


def calibrated_deployment_gate_audit(
    raw: Mapping[str, Any],
    calibrated: Mapping[str, Any],
    *,
    config: TransferConfig,
) -> dict[str, Any]:
    """Gate a previously selected calibrated endpoint on dev/confirm."""

    common = _common_endpoint_checks(raw, calibrated, config=config)
    gain = float(calibrated["training_contribution_natural_prior"])
    minimum_class_delta = float(calibrated["training_min_per_class_delta"])
    _require_finite_metrics(
        calibrated_natural_prior_gain=gain,
        calibrated_minimum_per_class_delta=minimum_class_delta,
    )
    gain_pass = bap._strictly_above(gain, config.min_raw_gain)
    class_pass = bap._at_least(minimum_class_delta, -config.max_class_drop)
    return {
        "endpoint": CALIBRATED_ENDPOINT,
        "common_checks": common,
        "natural_prior_gain": gain,
        "minimum_natural_prior_gain": f"strictly greater than {config.min_raw_gain}",
        "natural_prior_gain_pass": gain_pass,
        "minimum_per_class_recall_delta": minimum_class_delta,
        "minimum_allowed_per_class_recall_delta": -config.max_class_drop,
        "class_safety_pass": class_pass,
        "gate_pass": bool(common["common_checks_pass"] and gain_pass and class_pass),
        "thresholds_must_be_internal_frozen": True,
    }


def calibrated_internal_fallback_gate_audit(
    raw: Mapping[str, Any],
    calibrated: Mapping[str, Any],
    *,
    config: TransferConfig,
) -> dict[str, Any]:
    """Narrow fallback: raw acquisition passed and only raw class safety failed."""

    deployment = calibrated_deployment_gate_audit(
        raw, calibrated, config=config
    )
    raw_gain = float(raw["natural_prior_gain"])
    raw_minimum_class_delta = float(raw["minimum_per_class_delta"])
    _require_finite_metrics(
        raw_natural_prior_gain=raw_gain,
        raw_minimum_per_class_delta=raw_minimum_class_delta,
    )
    raw_gain_pass = bap._strictly_above(raw_gain, config.min_raw_gain)
    raw_class_safety_failure = not bap._at_least(
        raw_minimum_class_delta, -config.max_class_drop
    )
    return {
        "endpoint": CALIBRATED_ENDPOINT,
        "deployment_gate": deployment,
        "raw_natural_prior_gain": raw_gain,
        "raw_acquisition_gain_pass": raw_gain_pass,
        "raw_minimum_per_class_recall_delta": raw_minimum_class_delta,
        "raw_class_safety_failure_required": True,
        "raw_class_safety_failure": raw_class_safety_failure,
        "calibrated_metric_gate_pass": bool(deployment["gate_pass"]),
        "fallback_gate_pass": bool(
            deployment["gate_pass"]
            and raw_gain_pass
            and raw_class_safety_failure
        ),
        "cannot_rescue_absent_raw_acquisition": True,
        "cannot_rescue_invalid_raw_free_generation": True,
        "cannot_rescue_non_improving_ranking": True,
    }


def internal_endpoint_decision(
    raw: Mapping[str, Any],
    calibrated: Mapping[str, Any],
    *,
    config: TransferConfig,
) -> dict[str, Any]:
    raw_audit = raw_endpoint_gate_audit(raw, calibrated, config=config)
    fallback_audit = calibrated_internal_fallback_gate_audit(
        raw, calibrated, config=config
    )
    if raw_audit["gate_pass"]:
        selected_endpoint: str | None = RAW_ENDPOINT
        reason = "raw_passed_and_has_permanent_priority"
    elif fallback_audit["fallback_gate_pass"]:
        selected_endpoint = CALIBRATED_ENDPOINT
        reason = "raw_only_failed_class_safety_and_calibrated_fallback_passed"
    else:
        selected_endpoint = None
        reason = "neither_ordered_internal_endpoint_rule_passed"
    return {
        "endpoint_rule_version": ENDPOINT_RULE_VERSION,
        "endpoint_priority": list(ENDPOINTS),
        "endpoint_candidate_count": 2,
        "raw_endpoint_gate": raw_audit,
        "calibrated_internal_fallback_gate": fallback_audit,
        "selected_endpoint": selected_endpoint,
        "selection_reason": reason,
        "raw_priority_applied": bool(raw_audit["gate_pass"]),
        "endpoint_selected": selected_endpoint is not None,
        "endpoint_switch_allowed_after_internal": False,
        "endpoint_switch_count": 0,
    }


def deployment_endpoint_gate_audit(
    selected_endpoint: str,
    raw: Mapping[str, Any],
    calibrated: Mapping[str, Any],
    *,
    config: TransferConfig,
) -> dict[str, Any]:
    if selected_endpoint == RAW_ENDPOINT:
        result = raw_endpoint_gate_audit(raw, calibrated, config=config)
    elif selected_endpoint == CALIBRATED_ENDPOINT:
        result = calibrated_deployment_gate_audit(
            raw, calibrated, config=config
        )
    else:
        raise ValueError(f"unsupported selected endpoint {selected_endpoint!r}")
    return {
        **result,
        "selected_endpoint": selected_endpoint,
        "alternative_endpoint_cannot_rescue": True,
        "endpoint_switch_allowed": False,
    }


def _recipe_payload() -> dict[str, Any]:
    return {
        "origin": "BoolQA PCSM-LoRA v1 training recipe plus locked adaptive endpoint v2",
        "source_training_recipe_sha256": bap._sha256_json(
            transfer_v1._recipe_payload()
        ),
        "adaptive_endpoint_protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "adaptive_endpoint_execution_lock_sha256": FROZEN_EXECUTION_LOCK_SHA256,
        "target_task_recipe_hparam_checkpoint_tuning": False,
        "target_task_gradient_adaptation": True,
        "target_task_internal_threshold_calibration": True,
        "target_task_endpoint_selection": True,
        "objective": {
            "name": OBJECTIVE,
            "per_example_loss": "sequence-mean target-token NLL of observed label",
            "importance_weight": "pinned target natural prior pi(y) / paired sampler q(y)=0.5",
        },
        "learning_rate": transfer_v1.FORMAL_LEARNING_RATE,
        "fit_per_class": transfer_v1.FORMAL_FIT_PER_CLASS,
        "internal_calibration_per_class": transfer_v1.FORMAL_CALIBRATION_PER_CLASS,
        "development_per_class": transfer_v1.FORMAL_TUNE_PER_CLASS,
        "confirm_per_class": transfer_v1.FORMAL_CONFIRM_PER_CLASS,
        "optimizer_steps": transfer_v1.FORMAL_OPTIMIZER_STEPS,
        "eligible_checkpoint_steps": [transfer_v1.FORMAL_OPTIMIZER_STEPS],
        "batch_size": transfer_v1.FORMAL_BATCH_SIZE,
        "training_batch_size": transfer_v1.FORMAL_BATCH_SIZE,
        "physical_training_microbatch_size": 2,
        "gradient_accumulation_microbatches_per_logical_step": 2,
        "execution_version": "PCSM adaptive-endpoint execution-v2",
        "algebraically_same_logical_batch_objective": True,
        "bitwise_identical_to_v1_claimed": False,
        "evaluation_microbatch_size": EVALUATION_MICROBATCH_SIZE,
        "maximum_label_expanded_evaluation_sequences": (
            EVALUATION_MICROBATCH_SIZE * len(BINARY_LABELS)
        ),
        "evaluation_microbatch_semantics": (
            "pure resource control; preserve ordered rows, outputs, metrics, and endpoint rule"
        ),
        "fit_cycles": 2,
        "seed": transfer_v1.FORMAL_SEED,
        "sampler": "paired_fit_batches; equal True/False per batch",
        "gradient_forward_mode": "eval with autograd; dropout disabled",
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "max_source_length": transfer_v1.FORMAL_MAX_SOURCE_LENGTH,
        "max_target_length": transfer_v1.FORMAL_MAX_TARGET_LENGTH,
        "truncation_mode": transfer_v1.FORMAL_TRUNCATION_MODE,
        "endpoint_selection": {
            "rule_version": ENDPOINT_RULE_VERSION,
            "ordered_candidates": list(ENDPOINTS),
            "raw_priority": True,
            "calibrated_fallback_only_for_raw_class_safety_failure": True,
            "raw_acquisition_gain_required_for_fallback": "strictly > 0.05",
            "post_raw_valid_label_rate": ">= 0.99",
            "margin_roc_auc_delta": "strictly > 0",
            "selected_gain": "strictly > 0.05",
            "selected_minimum_per_class_delta": ">= -0.05",
            "endpoint_switch_after_internal": False,
        },
        "threshold_rule": (
            "fit base/post independently with the identical exhaustive target-internal "
            "natural-prior fitter; freeze both before development/confirm"
        ),
        "split": {
            "unit": "semantic_group",
            "group_key_version": rescue.SEMANTIC_GROUP_KEY_VERSION,
            "selection_algorithm_version": rescue.SEMANTIC_SELECTION_ALGORITHM_VERSION,
            "internal_split_seed_tag": "bap-internal-calibration",
        },
    }


def _protocol_payload(config: TransferConfig) -> dict[str, Any]:
    _assert_frozen_uncertainty_protocol()
    recipe = _recipe_payload()
    return {
        "format": FORMAT_VERSION,
        "claim_scope": (
            "cross-task train-only adaptive-endpoint acquisition; not continual retention"
        ),
        "method": "PCSM-LoRA adaptive endpoint v2",
        "source_recipe": recipe,
        "source_recipe_sha256": bap._sha256_json(recipe),
        "adaptive_endpoint_protocol": {
            "path": str(_protocol_note_path()),
            "sha256": FROZEN_PROTOCOL_SHA256,
            "external_preregistration_claimed": False,
            "qqp_internal_role": "post-hoc discovery evidence",
            "boolqa_role": "source/pilot; not a prospective v2 test",
            "multirc_role": "determined by its pre-v2 access ledger; no blindness inferred here",
        },
        "adaptive_endpoint_execution_lock": {
            "path": str(_execution_lock_path()),
            "sha256": FROZEN_EXECUTION_LOCK_SHA256,
            "metric_free_infrastructure_interruption_preceded_lock": True,
            "scientific_endpoint_rule_changed": False,
            "bitwise_identical_to_v1_claimed": False,
        },
        "transfer_task": config.task_name,
        "target_task_recipe_hparam_checkpoint_tuning": False,
        "target_task_gradient_adaptation": True,
        "target_task_internal_threshold_calibration": True,
        "target_task_endpoint_selection": True,
        "selection_eligible_training_arm_count": 1,
        "endpoint_candidate_count": 2,
        "endpoint_priority": list(ENDPOINTS),
        "learning_rate_candidate_count": 1,
        "checkpoint_candidate_count": 1,
        "resource_protocol": {
            "training_batch_size": config.batch_size,
            "physical_training_microbatch_size": 2,
            "gradient_accumulation_microbatches_per_logical_step": 2,
            "single_clip_and_optimizer_step_per_logical_batch": True,
            "training_execution": _training_execution_protocol(),
            "evaluation_microbatch_size": EVALUATION_MICROBATCH_SIZE,
            "maximum_label_expanded_evaluation_sequences": (
                EVALUATION_MICROBATCH_SIZE * len(BINARY_LABELS)
            ),
            "evaluation_order_and_metric_change_allowed": False,
        },
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
            "internal_calibration": (
                "ordered endpoint selection plus base/post threshold fitting"
            ),
            "development_tune": (
                "selected-endpoint qualification only; opened after endpoint lock"
            ),
            "confirm": "fresh one-shot selected-endpoint scoring only",
        },
        "confirmation": {
            "selected_endpoint_only": True,
            "alternative_endpoint_cannot_rescue": True,
            "thresholds_refit": False,
            "uncertainty": {
                "method": "paired semantic-group cluster bootstrap",
                "seed": BOOTSTRAP_SEED,
                "valid_replicates": BOOTSTRAP_REPLICATES,
                "interval_level": BOOTSTRAP_INTERVAL_LEVEL,
                "interval_type": "percentile",
                "discard_and_redraw_replicates_missing_either_label": True,
            },
            "calibrated_claim_is_not_raw_generation": True,
        },
    }


def _as_bap_config(config: TransferConfig) -> bap.BAPConfig:
    return transfer_v1._as_bap_config(config)


def _model_config(config: TransferConfig) -> rescue.RescueConfig:
    return transfer_v1._model_config(config)


def _internal_manifest(split: bap.BAPInternalSplit) -> dict[str, Any]:
    payload = bap.bap_internal_split_manifest(split)
    return {
        **payload,
        "format": FORMAT_VERSION,
        "constructor": "frozen BAP group-atomic internal split",
    }


_validate_freshness_sources = transfer_v1._validate_freshness_sources
def _training_execution_protocol() -> dict[str, Any]:
    return {
        "execution_version": "PCSM adaptive-endpoint execution-v2",
        "execution_lock_path": str(_execution_lock_path()),
        "execution_lock_sha256": FROZEN_EXECUTION_LOCK_SHA256,
        "logical_training_batch_size": 4,
        "logical_batch_class_counts": {label: 2 for label in BINARY_LABELS},
        "physical_training_microbatch_size": 2,
        "microbatch_slices": [[0, 2], [2, 4]],
        "microbatch_loss_scale": "len(microbatch) / logical_batch_size",
        "zero_grad_calls_per_logical_step": 1,
        "backward_calls_per_logical_step": 2,
        "gradient_clip_calls_per_logical_step": 1,
        "optimizer_step_calls_per_logical_step": 1,
        "algebraically_same_logical_batch_objective": True,
        "bitwise_identical_to_v1_claimed": False,
        "ordered_v1_logical_exposure_schedule_preserved": True,
        "intermediate_recovery_artifacts_implemented": False,
        "step_384_pre_internal_eval_durability_required": True,
    }


def _accumulate_pcsm_logical_batch(
    bank: Any,
    tokenizer: Any,
    logical_batch: Sequence[OfficialExample],
    optimizer: torch.optim.Optimizer,
    *,
    config: TransferConfig,
    natural_prior: Mapping[str, float],
) -> tuple[float, dict[str, Any]]:
    """Backpropagate one logical 2+2 batch via consecutive 2-row slices."""

    logical_size = len(logical_batch)
    if logical_size != 4 or config.batch_size != 4:
        raise ValueError("execution-v2 requires an exact logical training batch of 4")
    if {
        label: sum(item.label == label for item in logical_batch)
        for label in BINARY_LABELS
    } != {label: 2 for label in BINARY_LABELS}:
        raise ValueError("execution-v2 logical batch must contain two rows per class")
    bap_config = _as_bap_config(config)
    optimizer.zero_grad(set_to_none=True)
    logical_loss = 0.0
    source_tokens = 0
    target_tokens = 0
    microbatch_ids: list[list[str]] = []
    for start in range(0, logical_size, 2):
        microbatch = tuple(logical_batch[start : start + 2])
        if len(microbatch) != 2:
            raise RuntimeError("execution-v2 produced an incomplete training microbatch")
        encoded = rescue._tokenize(tokenizer, microbatch, config=bap_config)
        output = bank(**encoded)
        sequence_nll = bap.per_example_target_token_nll(
            output.logits, encoded["labels"]
        )
        weights = torch.tensor(
            [float(natural_prior[item.label]) / 0.5 for item in microbatch],
            dtype=sequence_nll.dtype,
            device=sequence_nll.device,
        )
        micro_loss = torch.mean(weights * sequence_nll)
        scale = len(microbatch) / logical_size
        scaled_loss = micro_loss * scale
        scaled_loss.backward()
        logical_loss += float(scaled_loss.detach().cpu())
        source_tokens += int(encoded["attention_mask"].sum())
        target_tokens += int(encoded["labels"].ne(-100).sum())
        microbatch_ids.append([item.example_id for item in microbatch])
    torch.nn.utils.clip_grad_norm_(tuple(bank.active_atom_parameters()), 1.0)
    optimizer.step()
    bank.assert_exact_budget()
    return logical_loss, {
        "gradient_examples": logical_size,
        "physical_microbatches": 2,
        "physical_microbatch_size": 2,
        "microbatch_example_ids": microbatch_ids,
        "source_tokens_processed": source_tokens,
        "target_tokens_processed": target_tokens,
        "zero_grad_calls": 1,
        "backward_calls": 2,
        "gradient_clip_calls": 1,
        "optimizer_step_calls": 1,
    }


def _train_fixed_trajectory(
    bank: Any,
    tokenizer: Any,
    fit_examples: Sequence[OfficialExample],
    *,
    config: TransferConfig,
    natural_prior: Mapping[str, float],
) -> dict[str, Any]:
    """Run the frozen logical trajectory with execution-v2 accumulation."""

    if config.batch_size != 4:
        raise ValueError("execution-v2 logical training batch size is frozen at 4")
    if (
        set(natural_prior) != set(BINARY_LABELS)
        or not math.isclose(
            sum(float(value) for value in natural_prior.values()),
            1.0,
            abs_tol=1e-9,
        )
        or any(float(value) <= 0 for value in natural_prior.values())
    ):
        raise ValueError("execution-v2 natural prior must be positive and binary")
    bank.eval()
    optimizer = torch.optim.AdamW(
        tuple(bank.all_atom_parameters()),
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    losses: list[float] = []
    exposure_ids: list[str] = []
    per_class = {label: 0 for label in BINARY_LABELS}
    source_tokens = 0
    target_tokens = 0
    observed_steps = 0
    physical_microbatches = 0
    zero_grad_calls = 0
    backward_calls = 0
    gradient_clip_calls = 0
    optimizer_step_calls = 0
    for step, logical_batch in bap.paired_fit_batches(
        fit_examples,
        batch_size=config.batch_size,
        steps=config.optimizer_steps,
        seed=config.seed,
    ):
        expected_order = [item.example_id for item in logical_batch]
        loss, step_ledger = _accumulate_pcsm_logical_batch(
            bank,
            tokenizer,
            logical_batch,
            optimizer,
            config=config,
            natural_prior=natural_prior,
        )
        flattened_microbatch_order = [
            example_id
            for microbatch_ids in step_ledger["microbatch_example_ids"]
            for example_id in microbatch_ids
        ]
        if flattened_microbatch_order != expected_order:
            raise RuntimeError("execution-v2 changed logical exposure order")
        if not math.isfinite(float(loss)):
            raise FloatingPointError(f"non-finite execution-v2 loss at step {step}")
        observed_steps = step
        losses.append(float(loss))
        exposure_ids.extend(expected_order)
        for item in logical_batch:
            per_class[item.label] += 1
        source_tokens += int(step_ledger["source_tokens_processed"])
        target_tokens += int(step_ledger["target_tokens_processed"])
        physical_microbatches += int(step_ledger["physical_microbatches"])
        zero_grad_calls += int(step_ledger["zero_grad_calls"])
        backward_calls += int(step_ledger["backward_calls"])
        gradient_clip_calls += int(step_ledger["gradient_clip_calls"])
        optimizer_step_calls += int(step_ledger["optimizer_step_calls"])
    expected_exposures = config.optimizer_steps * config.batch_size
    expected_per_class = expected_exposures // 2
    if (
        observed_steps != config.optimizer_steps
        or len(exposure_ids) != expected_exposures
        or per_class != {label: expected_per_class for label in BINARY_LABELS}
        or physical_microbatches != config.optimizer_steps * 2
        or zero_grad_calls != config.optimizer_steps
        or backward_calls != config.optimizer_steps * 2
        or gradient_clip_calls != config.optimizer_steps
        or optimizer_step_calls != config.optimizer_steps
    ):
        raise RuntimeError("execution-v2 violated its logical resource budget")
    ledger = {
        "objective": OBJECTIVE,
        "execution_version": "PCSM adaptive-endpoint execution-v2",
        "bitwise_identical_to_v1_claimed": False,
        "algebraically_same_logical_batch_objective": True,
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
        "logical_training_batch_size": config.batch_size,
        "physical_training_microbatch_size": 2,
        "physical_microbatches": physical_microbatches,
        "zero_grad_calls": zero_grad_calls,
        "post_trajectory_cleanup_zero_grad_calls": 1,
        "total_optimizer_zero_grad_calls": zero_grad_calls + 1,
        "backward_calls": backward_calls,
        "gradient_clip_calls": gradient_clip_calls,
        "optimizer_step_calls": optimizer_step_calls,
        "paired_schedule_seed": config.seed,
        "ordered_exposure_ids_sha256": bap._sha256_json(exposure_ids),
        "ordered_exposure_count": len(exposure_ids),
        "source_tokens_processed": source_tokens,
        "target_tokens_processed": target_tokens,
        "mean_logical_batch_loss": float(np.mean(losses)),
        "gradient_forward_mode": "eval (gradients enabled; dropout disabled)",
        "dropout_disabled": True,
        "exact_budget": True,
        "training_execution_protocol": _training_execution_protocol(),
        "post_trajectory_cleanup": {
            "adapter_gradients_cleared": True,
            "optimizer_released_before_internal_evaluation": True,
            "python_gc_collected_before_internal_evaluation": True,
        },
    }
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    gc.collect()
    return ledger


def _evaluation_protocol() -> dict[str, Any]:
    return {
        "training_batch_size_is_unchanged": True,
        "evaluation_microbatch_size": EVALUATION_MICROBATCH_SIZE,
        "label_count": len(BINARY_LABELS),
        "maximum_label_expanded_sequences_per_forward": (
            EVALUATION_MICROBATCH_SIZE * len(BINARY_LABELS)
        ),
        "ordered_example_coverage_preserved": True,
        "metric_definition_preserved": True,
        "endpoint_rule_preserved": True,
        "purpose": "CPU peak-memory control only",
    }


def _score_partition(
    bank: Any,
    tokenizer: Any,
    examples: Sequence[OfficialExample],
    *,
    config: TransferConfig,
    natural_prior: Mapping[str, float],
) -> tuple[dict[str, Any], list[float]]:
    """Evaluate in fixed two-example chunks; training remains batch four."""

    evaluation_config = replace(
        _as_bap_config(config), batch_size=EVALUATION_MICROBATCH_SIZE
    )
    raw = bap._raw_audit(
        bank,
        tokenizer,
        examples,
        config=evaluation_config,
        natural_prior=natural_prior,
    )
    margins = bap.collect_label_margins(
        bank, tokenizer, examples, config=evaluation_config
    )
    if list(raw.get("example_ids", ())) != [item.example_id for item in examples]:
        raise RuntimeError("v2 evaluation microbatch changed raw example order/coverage")
    if len(margins) != len(examples):
        raise RuntimeError("v2 evaluation microbatch changed margin row coverage")
    return raw, margins


def _build_splits(
    examples: Sequence[OfficialExample], *, config: TransferConfig
) -> tuple[
    rescue.RescueSplit,
    bap.BAPInternalSplit,
    tuple[Mapping[str, Any], ...],
    dict[str, Any],
]:
    return transfer_v1._build_splits(examples, config=config)


def _load_task_protocol_registry(
    config: TransferConfig, sources: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if config.task_protocol_registry is None:
        return {
            "status": "not_supplied",
            "required_for_this_run": False,
            "run_local_cli_source_binding_only": True,
        }
    path = Path(config.task_protocol_registry).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing v2 task protocol registry: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load v2 task protocol registry") from error
    if not isinstance(payload, Mapping) or payload.get("format") != TASK_PROTOCOL_REGISTRY_FORMAT:
        raise ValueError("unsupported v2 task protocol registry format")
    tasks = payload.get("tasks")
    entry = tasks.get(config.task_name) if isinstance(tasks, Mapping) else None
    if not isinstance(entry, Mapping):
        raise ValueError("v2 registry lacks the requested task entry")
    recipe_sha = bap._sha256_json(_recipe_payload())
    if entry.get("source_recipe_sha256") != recipe_sha:
        raise ValueError("v2 registry recipe SHA-256 differs from frozen recipe")
    if entry.get("semantic_group_required") is not True:
        raise ValueError("v2 registry does not require semantic-group splitting")
    expected_shas = entry.get("freshness_source_sha256s")
    if (
        not isinstance(expected_shas, list)
        or not expected_shas
        or any(not isinstance(value, str) for value in expected_shas)
    ):
        raise ValueError("v2 registry freshness SHA set is malformed")
    actual_shas = sorted(str(source["sha256"]).lower() for source in sources)
    if sorted(value.lower() for value in expected_shas) != actual_shas:
        raise ValueError("v2 registry freshness SHA set does not match CLI sources")
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
        "method_development_inventory": {
            "qqp_v1_internal_observed_before_rule": True,
            "qqp_v1_internal_role": "discovery_only",
            "boolqa_v1_result_observed_before_rule": True,
            "boolqa_role": "source_or_pilot_only",
            "multirc_metrics_inspected_before_protocol_lock": False,
            "multirc_partition_role_must_follow_actual_access_ledger": True,
            "endpoint_candidate_count": 2,
            "training_hyperparameter_or_checkpoint_scan_in_v2": False,
        },
    }
    return {**payload, "payload_sha256": bap._sha256_json(payload)}


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
        "evaluation_protocol": _evaluation_protocol(),
        "training_execution_protocol": _training_execution_protocol(),
        "model_or_tokenizer_constructed": False,
        "gradient_steps": 0,
        "endpoint_selected": False,
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
    raw: Mapping[str, Any],
    calibrated: Mapping[str, Any],
    decision: Mapping[str, Any],
    trajectory_lock: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "format": FORMAT_VERSION,
        "status": "internal_endpoint_qualification_failed",
        "task": config.task_name,
        "source_training_recipe_task": "BoolQA",
        "qqp_internal_is_discovery_evidence": config.task_name == "QQP",
        "target_task_recipe_hparam_checkpoint_tuning": False,
        "target_task_gradient_adaptation": True,
        "target_task_internal_threshold_calibration": True,
        "target_task_endpoint_selection": True,
        "only_eligible_step": config.optimizer_steps,
        "resource_ledger": dict(resource_ledger),
        "trajectory_completed_lock": dict(trajectory_lock),
        "internal_calibration": {
            "raw_free_generation": dict(raw),
            "internal_fitted_calibrated": dict(calibrated),
            "endpoint_decision": dict(decision),
            "qualification_pass": False,
        },
        "selected_endpoint": None,
        "development_tune_accessed": False,
        "confirm_eligible": False,
        "confirm_audit_model_scored_or_tokenized": False,
    }
    rescue._atomic_json(task_dir / "qualification_failure.json", result)
    manifest.update(
        {
            "status": "internal_endpoint_qualification_failed",
            "resource_ledger": dict(resource_ledger),
            "trajectory_completed_lock": dict(trajectory_lock),
            "endpoint_selected": False,
            "development_tune_accessed": False,
        }
    )
    rescue._atomic_json(output_root / "manifest.json", manifest)
    (output_root / "QUALIFICATION_FAILED").write_text(
        "adaptive endpoint internal gate failed; development and confirm stayed unscored\n",
        encoding="utf-8",
    )
    return result


def _write_trajectory_completed_lock(
    *,
    task_dir: Path,
    bank: Any,
    config: TransferConfig,
    resource_ledger: Mapping[str, Any],
    protocol_lock: Mapping[str, Any],
    outer_split_manifest_sha256: str,
    internal_split_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Durably bind step 384 before any post-training internal evaluation."""

    checkpoint_dir = task_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = checkpoint_dir / f"step_{config.optimizer_steps}.pt"
    rescue._atomic_torch_save(checkpoint_path, bank.compact_state_dict())
    checkpoint = {
        "path": str(checkpoint_path.resolve()),
        "sha256": rescue._sha256_file(checkpoint_path),
        "optimizer_step": config.optimizer_steps,
        "sole_eligible_checkpoint": True,
        "written_before_post_training_internal_evaluation": True,
    }
    ledger_path = task_dir / "trajectory_resource_ledger.json"
    rescue._atomic_json(ledger_path, dict(resource_ledger))
    ledger_file = {
        "path": str(ledger_path.resolve()),
        "sha256": rescue._sha256_file(ledger_path),
        "payload_sha256": bap._sha256_json(resource_ledger),
    }
    lock_path = task_dir / "trajectory_completed_pending_internal_eval.json"
    lock_core = {
        "format": FORMAT_VERSION,
        "status": "trajectory_completed_pending_internal_eval",
        "execution_lock_sha256": FROZEN_EXECUTION_LOCK_SHA256,
        "protocol_lock_sha256": protocol_lock["sha256"],
        "outer_split_manifest_sha256": outer_split_manifest_sha256,
        "internal_split_manifest_sha256": internal_split_manifest_sha256,
        "checkpoint": checkpoint,
        "resource_ledger_file": ledger_file,
        "ordered_exposure_ids_sha256": resource_ledger[
            "ordered_exposure_ids_sha256"
        ],
        "optimizer_steps_completed": resource_ledger["optimizer_steps"],
        "gradient_example_exposures": resource_ledger[
            "gradient_example_exposures"
        ],
        "paired_exposures_per_class": resource_ledger[
            "paired_exposures_per_class"
        ],
        "training_execution_protocol": _training_execution_protocol(),
        "post_training_internal_eval_accessed": False,
        "development_tune_accessed": False,
        "confirm_audit_model_scored_or_tokenized": False,
        "intermediate_recovery_artifacts_implemented": False,
        "recovery_artifact_selection_eligible": False,
    }
    lock_payload = {**lock_core, "payload_sha256": bap._sha256_json(lock_core)}
    rescue._atomic_json(lock_path, lock_payload)
    lock_metadata = {
        "path": str(lock_path.resolve()),
        "sha256": rescue._sha256_file(lock_path),
        "payload_sha256": lock_payload["payload_sha256"],
        "checkpoint_sha256": checkpoint["sha256"],
        "resource_ledger_sha256": ledger_file["sha256"],
        "ordered_exposure_ids_sha256": resource_ledger[
            "ordered_exposure_ids_sha256"
        ],
        "created_before_post_training_internal_evaluation": True,
    }
    return checkpoint, lock_metadata


def run_selection(
    config: TransferConfig,
    *,
    train_examples: Sequence[OfficialExample] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Run one frozen PCSM trajectory and lock an ordered endpoint."""

    if config.confirm_selection is not None:
        raise ValueError("run_selection cannot execute v2 confirmation mode")
    output_root = Path(config.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing non-empty v2 output root {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    task_dir = output_root / config.task_name
    task_dir.mkdir(parents=True, exist_ok=False)

    injected_examples = train_examples is not None
    if injected_examples and not config.tiny_random_model:
        raise ValueError("formal v2 selection forbids train_examples injection")
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
        "method_development_protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "execution_resource_lock_sha256": FROZEN_EXECUTION_LOCK_SHA256,
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
        "evaluation_protocol": _evaluation_protocol(),
        "training_execution_protocol": _training_execution_protocol(),
        "train_corpus_fingerprint": corpus_fingerprint,
        "execution_environment": execution_environment,
        "verified_data_artifact": data_artifact,
        "run_local_protocol_lock": protocol_lock,
        "train_examples_injected_for_tiny_test": injected_examples,
        "endpoint_selected": False,
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
            "v2 dry run completed; no model, gradient, endpoint, or audit scoring\n",
            encoding="utf-8",
        )
        return payload

    injected_tokenizer = tokenizer is not None
    if injected_tokenizer and not config.tiny_random_model:
        raise ValueError("formal v2 selection forbids tokenizer injection")
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
        raise RuntimeError("v2 model violates the fixed adapter budget")

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
    checkpoint, trajectory_lock = _write_trajectory_completed_lock(
        task_dir=task_dir,
        bank=bank,
        config=config,
        resource_ledger=resource_ledger,
        protocol_lock=protocol_lock,
        outer_split_manifest_sha256=manifest["outer_split_manifest_sha256"],
        internal_split_manifest_sha256=manifest[
            "internal_split_manifest_sha256"
        ],
    )
    manifest.update(
        {
            "status": "trajectory_completed_pending_internal_eval",
            "resource_ledger": resource_ledger,
            "trajectory_completed_lock": trajectory_lock,
            "post_training_internal_eval_accessed": False,
        }
    )
    rescue._atomic_json(output_root / "manifest.json", manifest)
    post_internal_raw, post_internal_margins = _score_partition(
        bank,
        tokenizer,
        internal.calibration,
        config=config,
        natural_prior=natural_prior,
    )
    internal_raw = pcsm.raw_official_comparison(
        base_internal_raw, post_internal_raw
    )
    internal_calibrated = pcsm.calibrated_internal_comparison(
        base_internal_margins,
        post_internal_margins,
        [item.label for item in internal.calibration],
        natural_prior=natural_prior,
        config=config,  # type: ignore[arg-type]
    )
    decision = internal_endpoint_decision(
        internal_raw, internal_calibrated, config=config
    )
    decision_inputs_sha256 = bap._sha256_json(
        {
            "raw_free_generation": internal_raw,
            "internal_fitted_calibrated": internal_calibrated,
        }
    )
    endpoint_decision_sha256 = bap._sha256_json(decision)
    selected_endpoint = decision["selected_endpoint"]
    if selected_endpoint is None:
        failure = _write_internal_failure(
            output_root=output_root,
            task_dir=task_dir,
            manifest=manifest,
            config=config,
            resource_ledger=resource_ledger,
            raw=internal_raw,
            calibrated=internal_calibrated,
            decision=decision,
            trajectory_lock=trajectory_lock,
        )
        rescue._release_model(bank)
        return failure

    base_threshold = float(
        internal_calibrated["base_same_calibration_control"]["threshold"]
    )
    post_threshold = float(
        internal_calibrated["pcsm_same_calibration"]["threshold"]
    )
    if not math.isfinite(base_threshold) or not math.isfinite(post_threshold):
        raise RuntimeError("v2 internal threshold fitter produced a non-finite value")

    # This is the critical causal ordering: endpoint, checkpoint, thresholds,
    # metrics, and decision inputs become durable before a development row is
    # scored or tokenized.
    internal_lock_path = task_dir / "internal_endpoint_lock.json"
    internal_lock_core = {
        "format": FORMAT_VERSION,
        "status": "internal_endpoint_locked_before_development_access",
        "endpoint_rule_version": ENDPOINT_RULE_VERSION,
        "protocol_lock_sha256": protocol_lock["sha256"],
        "outer_split_manifest_sha256": manifest["outer_split_manifest_sha256"],
        "internal_split_manifest_sha256": manifest[
            "internal_split_manifest_sha256"
        ],
        "selected_checkpoint": checkpoint,
        "trajectory_completed_lock": trajectory_lock,
        "resource_ledger_sha256": bap._sha256_json(resource_ledger),
        "raw_free_generation": internal_raw,
        "internal_fitted_calibrated": internal_calibrated,
        "endpoint_decision": decision,
        "endpoint_decision_inputs_sha256": decision_inputs_sha256,
        "endpoint_decision_sha256": endpoint_decision_sha256,
        "selected_endpoint": selected_endpoint,
        "base_threshold": base_threshold,
        "post_threshold": post_threshold,
        "thresholds_frozen": True,
        "endpoint_switch_allowed": False,
        "endpoint_switch_count": 0,
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
        "selected_endpoint": selected_endpoint,
        "created_before_development_access": True,
    }

    base_dev_bank = rescue._build_model(model_config, tokenizer)
    if rescue._adapter_fingerprint(base_dev_bank) != initial_fingerprint:
        raise RuntimeError("v2 frozen development base initialization drifted")
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
    dev_raw = pcsm.raw_official_comparison(base_dev_raw, post_dev_raw)
    dev_calibrated = pcsm.calibrated_frozen_comparison(
        base_dev_margins,
        post_dev_margins,
        [item.label for item in outer.tune_audit],
        natural_prior=natural_prior,
        base_threshold=base_threshold,
        post_threshold=post_threshold,
    )
    dev_selected_gate = deployment_endpoint_gate_audit(
        selected_endpoint,
        dev_raw,
        dev_calibrated,
        config=config,
    )
    confirm_eligible = bool(dev_selected_gate["gate_pass"])

    selection = {
        "format": FORMAT_VERSION,
        "status": "selection_locked",
        "task": config.task_name,
        "source_training_recipe_task": "BoolQA",
        "qqp_internal_is_discovery_evidence": config.task_name == "QQP",
        "target_task_recipe_hparam_checkpoint_tuning": False,
        "target_task_gradient_adaptation": True,
        "target_task_internal_threshold_calibration": True,
        "target_task_endpoint_selection": True,
        "scientific_config": asdict(config),
        "protocol": _protocol_payload(config),
        "selected": {
            "objective": OBJECTIVE,
            "step": config.optimizer_steps,
            "selection_rule": (
                "sole frozen training step plus ordered internal endpoint rule"
            ),
            "endpoint_rule_version": ENDPOINT_RULE_VERSION,
            "endpoint": selected_endpoint,
            "base_threshold": base_threshold,
            "post_threshold": post_threshold,
            "thresholds_used_for_selected_endpoint": (
                selected_endpoint == CALIBRATED_ENDPOINT
            ),
            "endpoint_switch_allowed": False,
        },
        "selected_checkpoint": checkpoint,
        "resource_ledger": resource_ledger,
        "internal_calibration": {
            "raw_free_generation": internal_raw,
            "internal_fitted_calibrated": internal_calibrated,
            "endpoint_decision": decision,
            "endpoint_decision_inputs_sha256": decision_inputs_sha256,
            "endpoint_decision_sha256": endpoint_decision_sha256,
            "selected_endpoint": selected_endpoint,
            "qualification_pass": True,
        },
        "internal_endpoint_lock": internal_lock,
        "trajectory_completed_lock": trajectory_lock,
        "development_tune": {
            "role": "selected-endpoint second qualification; not endpoint tuning",
            "accessed_only_after_endpoint_lock": True,
            "thresholds_frozen_before_access": True,
            "raw_free_generation_diagnostic": dev_raw,
            "internal_frozen_calibrated_diagnostic": dev_calibrated,
            "selected_endpoint": selected_endpoint,
            "selected_endpoint_gate": dev_selected_gate,
            "qualification_pass": confirm_eligible,
            "alternative_endpoint_cannot_rescue": True,
            "endpoint_switched": False,
            "endpoint_switch_count": 0,
            "did_not_select_recipe_checkpoint_or_threshold": True,
        },
        "confirm_eligible": confirm_eligible,
        "confirm_ineligibility_reason": (
            None
            if confirm_eligible
            else "selected endpoint failed development; alternative endpoint cannot rescue"
        ),
        "outer_split_manifest_sha256": manifest["outer_split_manifest_sha256"],
        "internal_split_manifest_sha256": manifest[
            "internal_split_manifest_sha256"
        ],
        "freshness_sources": [dict(item) for item in sources],
        "freshness_audit": freshness_audit,
        "task_protocol_registry_binding": registry_binding,
        "frozen_dependency_fingerprints": dependencies,
        "pinned_train_label_distribution": distribution,
        "evaluation_protocol": _evaluation_protocol(),
        "training_execution_protocol": _training_execution_protocol(),
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
            "evaluation_protocol": _evaluation_protocol(),
            "training_execution_protocol": _training_execution_protocol(),
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
        "selected_endpoint": selected_endpoint,
        "endpoint_rule_version": ENDPOINT_RULE_VERSION,
        "internal_endpoint_lock_sha256": internal_lock["sha256"],
        "trajectory_completed_lock_sha256": trajectory_lock["sha256"],
        "endpoint_decision_inputs_sha256": decision_inputs_sha256,
        "endpoint_decision_sha256": endpoint_decision_sha256,
        "internal_qualification_pass": True,
        "development_selected_endpoint_pass": confirm_eligible,
        "confirm_eligible": confirm_eligible,
        "endpoint_switch_allowed": False,
        "endpoint_switch_count": 0,
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
                "selected_endpoint": selected_endpoint,
                "confirm_eligible": confirm_eligible,
            },
            "resource_ledger": resource_ledger,
            "internal_endpoint_lock": internal_lock,
            "trajectory_completed_lock": trajectory_lock,
            "endpoint_selected": True,
            "development_tune_accessed": True,
        }
    )
    rescue._atomic_json(output_root / "manifest.json", manifest)
    (output_root / "SELECTION_COMPLETED").write_text(
        "v2 endpoint selection completed; confirm remains unscored and untokenized\n",
        encoding="utf-8",
    )
    rescue._release_model(bank)
    return selection


def load_selection(path: str | os.PathLike[str]) -> tuple[dict[str, Any], Path]:
    selection_path = Path(path).expanduser().resolve()
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load v2 selection {selection_path}") from error
    if not isinstance(selection, dict) or selection.get("format") != FORMAT_VERSION:
        raise ValueError("selection is not a PCSM adaptive-endpoint v2 artifact")
    if selection.get("status") != "selection_locked":
        raise ValueError("v2 selection is not durably locked")
    mappings = {
        "scientific_config",
        "protocol",
        "selected",
        "selected_checkpoint",
        "resource_ledger",
        "internal_calibration",
        "internal_endpoint_lock",
        "trajectory_completed_lock",
        "development_tune",
        "model_protocol",
        "run_local_protocol_lock",
        "freshness_audit",
        "task_protocol_registry_binding",
        "frozen_dependency_fingerprints",
    }
    missing = [name for name in mappings if not isinstance(selection.get(name), Mapping)]
    if missing:
        raise ValueError("v2 selection lacks mappings: " + ", ".join(sorted(missing)))
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
        raise ValueError("v2 selection lacks scientific_config")
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
        raise ValueError("v2 selection lacks scientific_config")
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
            "confirmation scientific config differs from locked v2 selection: "
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
    """Re-derive the scientific state machine before claiming confirmation."""

    if selection.get("task") != config.task_name:
        raise ValueError("v2 selection task changed")
    if selection.get("source_training_recipe_task") != "BoolQA":
        raise ValueError("v2 selection lost BoolQA training-recipe provenance")
    disclosures = {
        "target_task_recipe_hparam_checkpoint_tuning": False,
        "target_task_gradient_adaptation": True,
        "target_task_internal_threshold_calibration": True,
        "target_task_endpoint_selection": True,
    }
    for key, expected in disclosures.items():
        if selection.get(key) is not expected:
            raise ValueError(f"v2 selection disclosure changed: {key}")
    if selection.get("qqp_internal_is_discovery_evidence") is not (
        config.task_name == "QQP"
    ):
        raise ValueError("v2 selection changed the QQP discovery-evidence disclosure")
    if selection.get("confirm_eligible") is not True:
        raise ValueError("v2 selection is not eligible for sealed confirmation")
    if selection.get("confirm_audit_model_scored_or_tokenized") is not False:
        raise ValueError("v2 selection does not attest a sealed confirm partition")
    if bap._sha256_json(selection.get("protocol")) != bap._sha256_json(
        _protocol_payload(config)
    ):
        raise ValueError("v2 selection protocol differs from the locked implementation")
    if (
        selection.get("evaluation_protocol") != _evaluation_protocol()
        or selection["model_protocol"].get("evaluation_protocol")
        != _evaluation_protocol()
    ):
        raise ValueError("v2 evaluation microbatch protocol changed")
    if (
        selection.get("training_execution_protocol")
        != _training_execution_protocol()
        or selection["model_protocol"].get("training_execution_protocol")
        != _training_execution_protocol()
    ):
        raise ValueError("v2 training microbatch execution protocol changed")

    internal_partition = selection["internal_calibration"]
    internal_raw = internal_partition.get("raw_free_generation")
    internal_calibrated = internal_partition.get("internal_fitted_calibrated")
    stored_decision = internal_partition.get("endpoint_decision")
    if not all(
        isinstance(item, Mapping)
        for item in (internal_raw, internal_calibrated, stored_decision)
    ):
        raise ValueError("v2 internal calibration lacks endpoint decision inputs")
    expected_decision = internal_endpoint_decision(
        internal_raw, internal_calibrated, config=config  # type: ignore[arg-type]
    )
    if bap._sha256_json(stored_decision) != bap._sha256_json(expected_decision):
        raise ValueError("v2 stored endpoint decision does not follow raw priority")
    decision_inputs_sha = bap._sha256_json(
        {
            "raw_free_generation": internal_raw,
            "internal_fitted_calibrated": internal_calibrated,
        }
    )
    decision_sha = bap._sha256_json(expected_decision)
    if (
        internal_partition.get("endpoint_decision_inputs_sha256")
        != decision_inputs_sha
        or internal_partition.get("endpoint_decision_sha256") != decision_sha
    ):
        raise ValueError("v2 endpoint decision SHA binding changed")
    selected_endpoint = expected_decision.get("selected_endpoint")
    if selected_endpoint not in ENDPOINTS:
        raise ValueError("v2 internal endpoint qualification did not pass")
    if (
        internal_partition.get("selected_endpoint") != selected_endpoint
        or internal_partition.get("qualification_pass") is not True
    ):
        raise ValueError("v2 internal selected endpoint state changed")

    selected = selection["selected"]
    locked_step = int(selection["scientific_config"]["optimizer_steps"])
    if (
        selected.get("objective") != OBJECTIVE
        or int(selected.get("step", -1)) != locked_step
        or selected.get("selection_rule")
        != "sole frozen training step plus ordered internal endpoint rule"
        or selected.get("endpoint_rule_version") != ENDPOINT_RULE_VERSION
        or selected.get("endpoint") != selected_endpoint
        or selected.get("thresholds_used_for_selected_endpoint")
        is not (selected_endpoint == CALIBRATED_ENDPOINT)
        or selected.get("endpoint_switch_allowed") is not False
    ):
        raise ValueError("v2 selected objective/step/endpoint protocol is invalid")
    base_threshold = float(selected.get("base_threshold"))
    post_threshold = float(selected.get("post_threshold"))
    _require_finite_metrics(
        base_threshold=base_threshold, post_threshold=post_threshold
    )
    if not math.isclose(
        base_threshold,
        float(
            internal_calibrated["base_same_calibration_control"]["threshold"]
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        post_threshold,
        float(internal_calibrated["pcsm_same_calibration"]["threshold"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("v2 thresholds differ from internal calibration")

    development = selection["development_tune"]
    dev_raw = development.get("raw_free_generation_diagnostic")
    dev_calibrated = development.get("internal_frozen_calibrated_diagnostic")
    stored_dev_gate = development.get("selected_endpoint_gate")
    if not all(
        isinstance(item, Mapping)
        for item in (dev_raw, dev_calibrated, stored_dev_gate)
    ):
        raise ValueError("v2 development lacks selected-endpoint diagnostics")
    expected_dev_gate = deployment_endpoint_gate_audit(
        selected_endpoint,
        dev_raw,  # type: ignore[arg-type]
        dev_calibrated,  # type: ignore[arg-type]
        config=config,
    )
    if (
        bap._sha256_json(stored_dev_gate) != bap._sha256_json(expected_dev_gate)
        or expected_dev_gate.get("gate_pass") is not True
        or development.get("selected_endpoint") != selected_endpoint
        or development.get("qualification_pass") is not True
        or development.get("accessed_only_after_endpoint_lock") is not True
        or development.get("thresholds_frozen_before_access") is not True
        or development.get("alternative_endpoint_cannot_rescue") is not True
        or development.get("endpoint_switched") is not False
        or int(development.get("endpoint_switch_count", -1)) != 0
    ):
        raise ValueError("v2 development did not pass the locked selected endpoint")

    selection_sha = rescue._sha256_file(selection_path)
    selection_lock_path = selection_path.parent / "selection_lock.json"
    if not selection_lock_path.is_file():
        raise ValueError("v2 selection lock is missing")
    try:
        selection_lock = json.loads(selection_lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load v2 selection lock") from error
    checkpoint_payload = selection["selected_checkpoint"]
    internal_lock_metadata = selection["internal_endpoint_lock"]
    trajectory_lock_metadata = selection["trajectory_completed_lock"]
    if (
        selection_lock.get("format") != FORMAT_VERSION
        or Path(str(selection_lock.get("selection_path", ""))).expanduser().resolve()
        != selection_path.resolve()
        or selection_lock.get("selection_sha256") != selection_sha
        or selection_lock.get("selected_checkpoint_sha256")
        != checkpoint_payload.get("sha256")
        or int(selection_lock.get("selected_step", -1)) != locked_step
        or selection_lock.get("selected_endpoint") != selected_endpoint
        or selection_lock.get("endpoint_rule_version") != ENDPOINT_RULE_VERSION
        or selection_lock.get("internal_endpoint_lock_sha256")
        != internal_lock_metadata.get("sha256")
        or selection_lock.get("trajectory_completed_lock_sha256")
        != trajectory_lock_metadata.get("sha256")
        or selection_lock.get("endpoint_decision_inputs_sha256")
        != decision_inputs_sha
        or selection_lock.get("endpoint_decision_sha256") != decision_sha
        or selection_lock.get("internal_qualification_pass") is not True
        or selection_lock.get("development_selected_endpoint_pass") is not True
        or selection_lock.get("confirm_eligible") is not True
        or selection_lock.get("endpoint_switch_allowed") is not False
        or int(selection_lock.get("endpoint_switch_count", -1)) != 0
        or selection_lock.get("confirm_audit_model_scored_or_tokenized") is not False
    ):
        raise ValueError("v2 selection lock does not bind the selected endpoint")

    protocol_metadata = selection["run_local_protocol_lock"]
    protocol_path = (selection_path.parent / "protocol_lock.json").resolve()
    if (
        Path(str(protocol_metadata.get("path", ""))).expanduser().resolve()
        != protocol_path
        or not protocol_path.is_file()
        or rescue._sha256_file(protocol_path) != protocol_metadata.get("sha256")
        or protocol_metadata.get("external_preregistration_claimed") is not False
        or protocol_metadata.get("method_development_protocol_sha256")
        != FROZEN_PROTOCOL_SHA256
        or protocol_metadata.get("execution_resource_lock_sha256")
        != FROZEN_EXECUTION_LOCK_SHA256
    ):
        raise ValueError("v2 protocol lock path or SHA-256 changed")
    try:
        protocol_lock = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load v2 protocol lock") from error
    protocol_without_hash = dict(protocol_lock)
    recorded_protocol_sha = protocol_without_hash.pop("payload_sha256", None)
    locked_config_values = dict(selection["scientific_config"])
    locked_config_values["fresh_confirm_against_manifests"] = tuple(
        locked_config_values.get("fresh_confirm_against_manifests", ())
    )
    locked_selection_config = TransferConfig(**locked_config_values)
    expected_protocol_lock = _protocol_lock_payload(
        locked_selection_config,
        sources=selection.get("freshness_sources", ()),
        freshness_audit=selection["freshness_audit"],
        registry_binding=selection["task_protocol_registry_binding"],
        dependency_fingerprints=selection["frozen_dependency_fingerprints"],
    )
    if (
        recorded_protocol_sha != bap._sha256_json(protocol_without_hash)
        or recorded_protocol_sha != protocol_metadata.get("payload_sha256")
        or bap._sha256_json(protocol_lock) != bap._sha256_json(expected_protocol_lock)
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
        raise ValueError("v2 protocol lock payload changed")

    checkpoint = Path(str(checkpoint_payload.get("path", ""))).expanduser().resolve()
    expected_checkpoint = (
        selection_path.parent / "checkpoints" / f"step_{locked_step}.pt"
    ).resolve()
    if checkpoint != expected_checkpoint or not checkpoint.is_file():
        raise ValueError("v2 selected checkpoint path is invalid")
    checkpoint_sha = rescue._sha256_file(checkpoint)
    if (
        checkpoint_sha != checkpoint_payload.get("sha256")
        or int(checkpoint_payload.get("optimizer_step", -1)) != locked_step
        or checkpoint_payload.get("sole_eligible_checkpoint") is not True
        or checkpoint_payload.get(
            "written_before_post_training_internal_evaluation"
        )
        is not True
    ):
        raise ValueError("v2 selected checkpoint binding is invalid")

    trajectory_lock_path = (
        selection_path.parent / "trajectory_completed_pending_internal_eval.json"
    ).resolve()
    if (
        Path(str(trajectory_lock_metadata.get("path", ""))).expanduser().resolve()
        != trajectory_lock_path
        or not trajectory_lock_path.is_file()
        or rescue._sha256_file(trajectory_lock_path)
        != trajectory_lock_metadata.get("sha256")
        or trajectory_lock_metadata.get("checkpoint_sha256") != checkpoint_sha
        or trajectory_lock_metadata.get("ordered_exposure_ids_sha256")
        != selection["resource_ledger"].get("ordered_exposure_ids_sha256")
        or trajectory_lock_metadata.get(
            "created_before_post_training_internal_evaluation"
        )
        is not True
    ):
        raise ValueError("v2 trajectory-completed lock changed")
    try:
        trajectory_lock = json.loads(
            trajectory_lock_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load v2 trajectory-completed lock") from error
    trajectory_without_hash = dict(trajectory_lock)
    recorded_trajectory_sha = trajectory_without_hash.pop("payload_sha256", None)
    trajectory_ledger_file = trajectory_lock.get("resource_ledger_file")
    if not isinstance(trajectory_ledger_file, Mapping):
        raise ValueError("v2 trajectory lock lacks a resource-ledger binding")
    trajectory_ledger_path = (
        selection_path.parent / "trajectory_resource_ledger.json"
    ).resolve()
    if (
        Path(str(trajectory_ledger_file.get("path", ""))).expanduser().resolve()
        != trajectory_ledger_path
        or not trajectory_ledger_path.is_file()
        or rescue._sha256_file(trajectory_ledger_path)
        != trajectory_ledger_file.get("sha256")
        or trajectory_lock_metadata.get("resource_ledger_sha256")
        != trajectory_ledger_file.get("sha256")
    ):
        raise ValueError("v2 trajectory resource-ledger file changed")
    try:
        trajectory_resource_ledger = json.loads(
            trajectory_ledger_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load v2 trajectory resource ledger") from error
    if (
        recorded_trajectory_sha != bap._sha256_json(trajectory_without_hash)
        or recorded_trajectory_sha != trajectory_lock_metadata.get("payload_sha256")
        or trajectory_lock.get("status")
        != "trajectory_completed_pending_internal_eval"
        or trajectory_lock.get("execution_lock_sha256")
        != FROZEN_EXECUTION_LOCK_SHA256
        or trajectory_lock.get("protocol_lock_sha256")
        != protocol_metadata.get("sha256")
        or trajectory_lock.get("outer_split_manifest_sha256")
        != selection.get("outer_split_manifest_sha256")
        or trajectory_lock.get("internal_split_manifest_sha256")
        != selection.get("internal_split_manifest_sha256")
        or trajectory_lock.get("checkpoint") != checkpoint_payload
        or trajectory_ledger_file.get("payload_sha256")
        != bap._sha256_json(selection["resource_ledger"])
        or trajectory_resource_ledger != selection["resource_ledger"]
        or trajectory_lock.get("ordered_exposure_ids_sha256")
        != selection["resource_ledger"].get("ordered_exposure_ids_sha256")
        or int(trajectory_lock.get("optimizer_steps_completed", -1))
        != locked_step
        or int(trajectory_lock.get("gradient_example_exposures", -1))
        != int(selection["resource_ledger"].get("gradient_example_exposures", -2))
        or trajectory_lock.get("paired_exposures_per_class")
        != selection["resource_ledger"].get("paired_exposures_per_class")
        or trajectory_lock.get("training_execution_protocol")
        != _training_execution_protocol()
        or trajectory_lock.get("post_training_internal_eval_accessed") is not False
        or trajectory_lock.get("development_tune_accessed") is not False
        or trajectory_lock.get("confirm_audit_model_scored_or_tokenized") is not False
        or trajectory_lock.get("intermediate_recovery_artifacts_implemented")
        is not False
        or trajectory_lock.get("recovery_artifact_selection_eligible") is not False
    ):
        raise ValueError("v2 trajectory-completed lock payload changed")

    internal_lock_path = (selection_path.parent / "internal_endpoint_lock.json").resolve()
    if (
        Path(str(internal_lock_metadata.get("path", ""))).expanduser().resolve()
        != internal_lock_path
        or not internal_lock_path.is_file()
        or rescue._sha256_file(internal_lock_path)
        != internal_lock_metadata.get("sha256")
        or internal_lock_metadata.get("selected_endpoint") != selected_endpoint
        or internal_lock_metadata.get("created_before_development_access") is not True
    ):
        raise ValueError("v2 internal endpoint lock changed")
    try:
        internal_lock = json.loads(internal_lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load v2 internal endpoint lock") from error
    internal_without_hash = dict(internal_lock)
    recorded_internal_sha = internal_without_hash.pop("payload_sha256", None)
    if (
        recorded_internal_sha != bap._sha256_json(internal_without_hash)
        or recorded_internal_sha != internal_lock_metadata.get("payload_sha256")
        or internal_lock.get("status")
        != "internal_endpoint_locked_before_development_access"
        or internal_lock.get("endpoint_rule_version") != ENDPOINT_RULE_VERSION
        or internal_lock.get("protocol_lock_sha256") != protocol_metadata.get("sha256")
        or internal_lock.get("outer_split_manifest_sha256")
        != selection.get("outer_split_manifest_sha256")
        or internal_lock.get("internal_split_manifest_sha256")
        != selection.get("internal_split_manifest_sha256")
        or internal_lock.get("selected_checkpoint") != checkpoint_payload
        or internal_lock.get("trajectory_completed_lock")
        != trajectory_lock_metadata
        or internal_lock.get("resource_ledger_sha256")
        != bap._sha256_json(selection["resource_ledger"])
        or internal_lock.get("raw_free_generation") != internal_raw
        or internal_lock.get("internal_fitted_calibrated") != internal_calibrated
        or internal_lock.get("endpoint_decision") != expected_decision
        or internal_lock.get("endpoint_decision_inputs_sha256")
        != decision_inputs_sha
        or internal_lock.get("endpoint_decision_sha256") != decision_sha
        or internal_lock.get("selected_endpoint") != selected_endpoint
        or not math.isclose(
            float(internal_lock.get("base_threshold")),
            base_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(internal_lock.get("post_threshold")),
            post_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or internal_lock.get("thresholds_frozen") is not True
        or internal_lock.get("endpoint_switch_allowed") is not False
        or int(internal_lock.get("endpoint_switch_count", -1)) != 0
        or internal_lock.get("development_tune_accessed") is not False
        or internal_lock.get("confirm_audit_model_scored_or_tokenized") is not False
    ):
        raise ValueError("v2 internal endpoint lock payload changed")

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
        or int(ledger.get("batch_size", -1)) != int(config_payload["batch_size"])
        or ledger.get("paired_exposures_per_class")
        != {label: expected_per_class for label in BINARY_LABELS}
        or ledger.get("exact_budget") is not True
        or ledger.get("dropout_disabled") is not True
        or ledger.get("execution_version")
        != "PCSM adaptive-endpoint execution-v2"
        or ledger.get("bitwise_identical_to_v1_claimed") is not False
        or ledger.get("algebraically_same_logical_batch_objective") is not True
        or int(ledger.get("logical_training_batch_size", -1)) != 4
        or int(ledger.get("physical_training_microbatch_size", -1)) != 2
        or int(ledger.get("physical_microbatches", -1)) != locked_step * 2
        or int(ledger.get("zero_grad_calls", -1)) != locked_step
        or int(ledger.get("post_trajectory_cleanup_zero_grad_calls", -1)) != 1
        or int(ledger.get("total_optimizer_zero_grad_calls", -1))
        != locked_step + 1
        or int(ledger.get("backward_calls", -1)) != locked_step * 2
        or int(ledger.get("gradient_clip_calls", -1)) != locked_step
        or int(ledger.get("optimizer_step_calls", -1)) != locked_step
        or ledger.get("training_execution_protocol")
        != _training_execution_protocol()
    ):
        raise ValueError("v2 resource ledger violates the frozen trajectory")
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
        raise ValueError("v2 ordered exposure schedule changed")

    outer_sha = rescue._atomic_json_payload_sha256(rescue._split_manifest(outer))
    internal_sha = rescue._atomic_json_payload_sha256(_internal_manifest(internal))
    if outer_sha != selection.get("outer_split_manifest_sha256"):
        raise ValueError("reconstructed v2 outer split changed")
    if internal_sha != selection.get("internal_split_manifest_sha256"):
        raise ValueError("reconstructed v2 internal split changed")
    if distribution != selection.get("pinned_train_label_distribution"):
        raise ValueError("v2 target train label distribution changed")

    model_protocol = selection["model_protocol"]
    current_model_artifact = bap._local_artifact_fingerprint(
        str(config_payload["model_path"])
    )
    if current_model_artifact != model_protocol.get("frozen_model_tokenizer_artifact"):
        raise ValueError("frozen model/tokenizer artifact changed after v2 selection")
    if model_protocol.get("rescue_loss_objective") != OBJECTIVE:
        raise ValueError("v2 model protocol objective changed")

    manifest_path = selection_path.parent.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("v2 selection manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load v2 selection manifest") from error
    manifest_selection = manifest.get("selection")
    if (
        manifest.get("format") != FORMAT_VERSION
        or manifest.get("status") != "selection_completed"
        or not isinstance(manifest_selection, Mapping)
        or manifest_selection.get("sha256") != selection_sha
        or manifest_selection.get("selected_checkpoint_sha256") != checkpoint_sha
        or int(manifest_selection.get("selected_step", -1)) != locked_step
        or manifest_selection.get("selected_endpoint") != selected_endpoint
        or manifest_selection.get("confirm_eligible") is not True
        or manifest.get("internal_endpoint_lock")
        != selection.get("internal_endpoint_lock")
        or manifest.get("trajectory_completed_lock")
        != selection.get("trajectory_completed_lock")
        or manifest.get("freshness_sources") != selection.get("freshness_sources")
        or manifest.get("task_protocol_registry_binding")
        != selection.get("task_protocol_registry_binding")
        or manifest.get("evaluation_protocol") != _evaluation_protocol()
        or manifest.get("training_execution_protocol")
        != _training_execution_protocol()
    ):
        raise ValueError("v2 manifest does not bind the eligible selection")
    return {
        "selection_sha256": selection_sha,
        "selection_lock": selection_lock,
        "protocol_lock": dict(protocol_metadata),
        "internal_endpoint_lock": dict(internal_lock_metadata),
        "trajectory_completed_lock": dict(trajectory_lock_metadata),
        "selected_endpoint": selected_endpoint,
        "base_threshold": base_threshold,
        "post_threshold": post_threshold,
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
        "selected_endpoint": preflight["selected_endpoint"],
        "confirmation_output_root": str(output_root.resolve()),
        "claimed_unix_time": time.time(),
        "scope_note": (
            "confirm metadata reconstructed the locked semantic split; no confirm row "
            "was model-scored or tokenized before this exclusive claim"
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
            "this v2 selection already has a confirmation claim; refusing a second confirm read"
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
        raise RuntimeError("v2 confirmation claim changed during scoring")
    core = dict(current)
    recorded = core.pop("claimed_payload_sha256", None)
    if recorded != bap._sha256_json(core):
        raise RuntimeError("v2 confirmation claim payload hash is invalid")
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


def _raw_metrics_for_indices(
    raw: Mapping[str, Any],
    indices: Sequence[int],
    *,
    natural_prior: Mapping[str, float],
) -> dict[str, Any]:
    predictions = raw.get("predictions")
    references = raw.get("references")
    if (
        not isinstance(predictions, Sequence)
        or isinstance(predictions, (str, bytes))
        or not isinstance(references, Sequence)
        or isinstance(references, (str, bytes))
        or len(predictions) != len(references)
    ):
        raise ValueError("raw audit lacks aligned per-example predictions/references")
    selected_predictions = [str(predictions[index]) for index in indices]
    selected_references = [str(references[index]) for index in indices]
    return rescue.free_generation_classification_metrics(
        selected_predictions,
        selected_references,
        labels=BINARY_LABELS,
        natural_label_prior=natural_prior,
    )


def _resampled_statistics(
    indices: Sequence[int],
    *,
    base_raw: Mapping[str, Any],
    post_raw: Mapping[str, Any],
    base_margins: Sequence[float],
    post_margins: Sequence[float],
    references: Sequence[str],
    natural_prior: Mapping[str, float],
    selected_endpoint: str,
    base_threshold: float,
    post_threshold: float,
) -> dict[str, Any]:
    base_subset_raw = _raw_metrics_for_indices(
        base_raw, indices, natural_prior=natural_prior
    )
    post_subset_raw = _raw_metrics_for_indices(
        post_raw, indices, natural_prior=natural_prior
    )
    raw_comparison = pcsm.raw_official_comparison(
        base_subset_raw, post_subset_raw
    )
    selected_references = [references[index] for index in indices]
    calibrated_comparison = pcsm.calibrated_frozen_comparison(
        [float(base_margins[index]) for index in indices],
        [float(post_margins[index]) for index in indices],
        selected_references,
        natural_prior=natural_prior,
        base_threshold=base_threshold,
        post_threshold=post_threshold,
    )
    if selected_endpoint == RAW_ENDPOINT:
        natural_gain = float(raw_comparison["natural_prior_gain"])
        minimum_class_delta = float(raw_comparison["minimum_per_class_delta"])
        per_class_delta = dict(raw_comparison["per_class_recall_delta"])
    elif selected_endpoint == CALIBRATED_ENDPOINT:
        natural_gain = float(
            calibrated_comparison["training_contribution_natural_prior"]
        )
        minimum_class_delta = float(
            calibrated_comparison["training_min_per_class_delta"]
        )
        per_class_delta = dict(
            calibrated_comparison["training_per_class_recall_delta"]
        )
    else:  # pragma: no cover - guarded by caller/preflight.
        raise ValueError(f"unsupported bootstrap endpoint {selected_endpoint!r}")
    return {
        "selected_natural_prior_gain": natural_gain,
        "selected_minimum_per_class_delta": minimum_class_delta,
        "selected_per_class_delta": {
            label: float(per_class_delta[label]) for label in BINARY_LABELS
        },
        "post_raw_valid_label_rate": float(
            raw_comparison["post_valid_label_rate"]
        ),
        "margin_roc_auc_delta": float(
            calibrated_comparison["margin_diagnostics"]["roc_auc_delta"]
        ),
    }


def _percentile_interval(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("cluster-bootstrap statistics must be finite and non-empty")
    alpha = (1.0 - BOOTSTRAP_INTERVAL_LEVEL) / 2.0
    lower, upper = np.quantile(array, [alpha, 1.0 - alpha])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "interval_level": BOOTSTRAP_INTERVAL_LEVEL,
        "interval_type": "percentile",
        "contains_zero": bool(float(lower) <= 0.0 <= float(upper)),
    }


def semantic_group_cluster_uncertainty(
    examples: Sequence[OfficialExample],
    group_key_by_example_id: Mapping[str, str],
    *,
    base_raw: Mapping[str, Any],
    post_raw: Mapping[str, Any],
    base_margins: Sequence[float],
    post_margins: Sequence[float],
    natural_prior: Mapping[str, float],
    selected_endpoint: str,
    base_threshold: float,
    post_threshold: float,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Deterministic whole-semantic-group percentile bootstrap."""

    _assert_frozen_uncertainty_protocol()

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("positive integer cluster-bootstrap replicates required")
    if len(examples) != len(base_margins) or len(examples) != len(post_margins):
        raise ValueError("cluster-bootstrap margins are not aligned with examples")
    if not examples:
        raise ValueError("cluster bootstrap requires a non-empty confirmation partition")
    example_ids = [item.example_id for item in examples]
    references = [item.label for item in examples]
    if set(references) != set(BINARY_LABELS):
        raise ValueError("cluster bootstrap confirmation lacks a binary class")
    for name, raw in (("base", base_raw), ("post", post_raw)):
        if list(raw.get("example_ids", ())) != example_ids:
            raise ValueError(f"{name} raw audit example order changed")
        if list(raw.get("references", ())) != references:
            raise ValueError(f"{name} raw audit reference order changed")
    clusters: dict[str, list[int]] = {}
    cluster_order: list[str] = []
    for index, example_id in enumerate(example_ids):
        if example_id not in group_key_by_example_id:
            raise ValueError("cluster bootstrap lacks a semantic group key")
        group = str(group_key_by_example_id[example_id])
        if group not in clusters:
            clusters[group] = []
            cluster_order.append(group)
        clusters[group].append(index)
    if not cluster_order:
        raise ValueError("cluster bootstrap found no semantic groups")

    full_indices = list(range(len(examples)))
    point = _resampled_statistics(
        full_indices,
        base_raw=base_raw,
        post_raw=post_raw,
        base_margins=base_margins,
        post_margins=post_margins,
        references=references,
        natural_prior=natural_prior,
        selected_endpoint=selected_endpoint,
        base_threshold=base_threshold,
        post_threshold=post_threshold,
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    scalar_samples: dict[str, list[float]] = {
        "selected_natural_prior_gain": [],
        "selected_minimum_per_class_delta": [],
        "post_raw_valid_label_rate": [],
        "margin_roc_auc_delta": [],
    }
    class_samples = {label: [] for label in BINARY_LABELS}
    attempts = 0
    maximum_attempts = max(replicates * 20, replicates + 100)
    while (
        len(scalar_samples["selected_natural_prior_gain"]) < replicates
        and attempts < maximum_attempts
    ):
        attempts += 1
        drawn = rng.integers(0, len(cluster_order), size=len(cluster_order))
        indices = [
            index
            for cluster_index in drawn
            for index in clusters[cluster_order[int(cluster_index)]]
        ]
        if set(references[index] for index in indices) != set(BINARY_LABELS):
            continue
        statistics = _resampled_statistics(
            indices,
            base_raw=base_raw,
            post_raw=post_raw,
            base_margins=base_margins,
            post_margins=post_margins,
            references=references,
            natural_prior=natural_prior,
            selected_endpoint=selected_endpoint,
            base_threshold=base_threshold,
            post_threshold=post_threshold,
        )
        for key in scalar_samples:
            scalar_samples[key].append(float(statistics[key]))
        for label in BINARY_LABELS:
            class_samples[label].append(
                float(statistics["selected_per_class_delta"][label])
            )
    accepted = len(scalar_samples["selected_natural_prior_gain"])
    if accepted != replicates:
        raise RuntimeError(
            f"cluster bootstrap accepted only {accepted}/{replicates} binary replicates"
        )
    return {
        "method": "semantic-group cluster bootstrap with replacement",
        "cluster_sampling_rule": (
            "sample the observed number of semantic groups with replacement and "
            "retain every row of each drawn group; reject draws missing a class"
        ),
        "seed": int(seed),
        "replicates_requested": replicates,
        "replicates_accepted": accepted,
        "draws_attempted": attempts,
        "interval_level": BOOTSTRAP_INTERVAL_LEVEL,
        "interval_type": "percentile",
        "row_count": len(examples),
        "semantic_group_count": len(cluster_order),
        "semantic_group_size_summary": {
            "minimum": min(len(clusters[key]) for key in cluster_order),
            "maximum": max(len(clusters[key]) for key in cluster_order),
            "mean": float(np.mean([len(clusters[key]) for key in cluster_order])),
        },
        "selected_endpoint": selected_endpoint,
        "point_estimates": point,
        "confidence_intervals": {
            **{
                key: _percentile_interval(values)
                for key, values in scalar_samples.items()
            },
            "selected_per_class_delta": {
                label: _percentile_interval(class_samples[label])
                for label in BINARY_LABELS
            },
        },
        "statistical_significance_claimed_from_point_gate": False,
    }


def run_confirmation(
    config: TransferConfig,
    selection: Mapping[str, Any],
    selection_path: Path,
    *,
    train_examples: Sequence[OfficialExample] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Score the locked endpoint on a fresh confirmation partition once."""

    _assert_frozen_uncertainty_protocol()

    disk_selection, disk_path = load_selection(selection_path)
    if bap._sha256_json(selection) != bap._sha256_json(disk_selection):
        raise ValueError("in-memory selection differs from locked v2 selection")
    selection = disk_selection
    selection_path = disk_path
    _validate_confirmation_config(config, selection)
    output_root = Path(config.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing non-empty v2 confirmation root {output_root}")
    if (selection_path.parent / "confirmation_claim.json").exists():
        raise RuntimeError(
            "this v2 selection already has a confirmation claim; refusing a second confirm read"
        )

    injected_examples = train_examples is not None
    if injected_examples and not config.tiny_random_model:
        raise ValueError("formal v2 confirmation forbids train_examples injection")
    if injected_examples != bool(
        selection.get("train_examples_injected_for_tiny_test", False)
    ):
        raise ValueError("v2 confirmation data-loading mode changed")
    current_data_artifact = (
        None
        if injected_examples
        else bap._official_train_artifact_fingerprint(
            config.data_root, task_name=config.task_name
        )
    )
    if current_data_artifact != selection.get("verified_data_artifact"):
        raise ValueError("verified target data artifact changed after v2 selection")
    if train_examples is None:
        train_examples = load_official_examples(
            config.data_root, config.task_name, "train", verify=True
        )
    corpus_fingerprint = bap._train_corpus_fingerprint(
        train_examples, task_name=config.task_name
    )
    if corpus_fingerprint != selection.get("train_corpus_fingerprint"):
        raise ValueError("target train corpus changed after v2 selection")
    execution_environment = bap._execution_environment_fingerprint(config.device)
    if execution_environment != selection.get("execution_environment"):
        raise ValueError("v2 confirmation execution environment drifted")
    dependencies = _dependency_fingerprints()
    if dependencies != selection.get("frozen_dependency_fingerprints"):
        raise ValueError("frozen v2 dependencies changed after selection")
    distribution = rescue.pinned_train_label_distribution(
        train_examples, task_name=config.task_name
    )
    natural_prior = distribution["natural_label_prior"]
    outer, internal, sources, freshness_audit = _build_splits(
        train_examples, config=config
    )
    if [dict(item) for item in sources] != selection.get("freshness_sources"):
        raise ValueError("v2 freshness sources changed after selection")
    if freshness_audit != selection.get("freshness_audit"):
        raise ValueError("v2 freshness audit changed after selection")
    registry_binding = _load_task_protocol_registry(config, sources)
    if registry_binding != selection.get("task_protocol_registry_binding"):
        raise ValueError("v2 task protocol registry changed after selection")
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
        raise ValueError("formal v2 confirmation forbids tokenizer injection")
    if injected_tokenizer != bool(
        selection["model_protocol"].get("tokenizer_injected_for_tiny_test", False)
    ):
        raise ValueError("v2 confirmation tokenizer-loading mode changed")
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, use_fast=True, local_files_only=True
        )
    tokenizer_fingerprint = bap._tokenizer_protocol_fingerprint(tokenizer)
    if tokenizer_fingerprint != selection["model_protocol"].get(
        "tokenizer_protocol_fingerprint"
    ):
        raise ValueError("v2 tokenizer protocol changed after selection")

    model_config = _model_config(config)
    expected_init = selection["model_protocol"]["initial_adapter_fingerprint"]
    base_bank = rescue._build_model(model_config, tokenizer)
    if rescue._adapter_fingerprint(base_bank) != expected_init:
        raise ValueError("v2 confirmation base initialization changed")
    post_bank = rescue._build_model(model_config, tokenizer)
    if rescue._adapter_fingerprint(post_bank) != expected_init:
        rescue._release_model(base_bank)
        raise ValueError("v2 confirmation adapter initialization changed")
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
    try:
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
        confirm_raw = pcsm.raw_official_comparison(base_raw, post_raw)
        confirm_calibrated = pcsm.calibrated_frozen_comparison(
            base_margins,
            post_margins,
            [item.label for item in outer.confirm_audit],
            natural_prior=natural_prior,
            base_threshold=float(preflight["base_threshold"]),
            post_threshold=float(preflight["post_threshold"]),
        )
        selected_endpoint = str(preflight["selected_endpoint"])
        selected_gate = deployment_endpoint_gate_audit(
            selected_endpoint,
            confirm_raw,
            confirm_calibrated,
            config=config,
        )
        uncertainty = semantic_group_cluster_uncertainty(
            outer.confirm_audit,
            outer.group_key_by_example_id or {},
            base_raw=base_raw,
            post_raw=post_raw,
            base_margins=base_margins,
            post_margins=post_margins,
            natural_prior=natural_prior,
            selected_endpoint=selected_endpoint,
            base_threshold=float(preflight["base_threshold"]),
            post_threshold=float(preflight["post_threshold"]),
            seed=43002,
            replicates=10000,
        )
    except Exception:
        # The exclusive claim intentionally remains.  A failed scoring attempt
        # cannot be silently retried against the same sealed partition.
        rescue._release_model(base_bank)
        rescue._release_model(post_bank)
        raise

    selected_comparison = (
        confirm_raw
        if selected_endpoint == RAW_ENDPOINT
        else confirm_calibrated
    )
    selected_metric_name = (
        "raw official free-generation natural-prior EM gain"
        if selected_endpoint == RAW_ENDPOINT
        else (
            "internal-frozen calibrated constrained-label natural-prior EM gain; "
            "not raw free-generation accuracy"
        )
    )
    result = {
        "format": FORMAT_VERSION,
        "status": "confirmation_completed",
        "task": config.task_name,
        "source_training_recipe_task": "BoolQA",
        "qqp_internal_is_discovery_evidence": config.task_name == "QQP",
        "target_task_recipe_hparam_checkpoint_tuning": False,
        "target_task_gradient_adaptation": True,
        "target_task_internal_threshold_calibration": True,
        "target_task_endpoint_selection": True,
        "selection": {
            "path": str(selection_path),
            "sha256": preflight["selection_sha256"],
            "checkpoint_sha256": preflight["checkpoint_sha256"],
            "selected_step": config.optimizer_steps,
            "selected_endpoint": selected_endpoint,
        },
        "confirmation_claim": {
            "path": str(claim_path.resolve()),
            "created_immediately_before_confirm_scoring": True,
            "confirm_metadata_read_before_claim": True,
            "confirm_model_scored_or_tokenized_before_claim": False,
        },
        "selected_endpoint": selected_endpoint,
        "selected_endpoint_comparison": selected_comparison,
        "selected_endpoint_gate": selected_gate,
        "confirm_selected_endpoint_gate_pass": bool(selected_gate["gate_pass"]),
        "positive_claim_metric": selected_metric_name,
        "metric_is_raw_official_free_generation": selected_endpoint == RAW_ENDPOINT,
        "calibrated_endpoint_must_not_be_relabelled_as_raw": True,
        "raw_free_generation_diagnostic": confirm_raw,
        "internal_frozen_calibrated_diagnostic": confirm_calibrated,
        "alternative_endpoint_is_diagnostic_only": True,
        "alternative_endpoint_cannot_rescue": True,
        "endpoint_switched": False,
        "endpoint_switch_count": 0,
        "thresholds_frozen_on_target_internal_calibration": {
            "base": float(preflight["base_threshold"]),
            "post": float(preflight["post_threshold"]),
            "refit_on_development_or_confirmation": False,
        },
        "calibrated_comparison_semantics": (
            "adapted model plus internal calibration versus frozen base with the "
            "identical internal calibration procedure"
        ),
        "confirm_rows_used_for_recipe_checkpoint_endpoint_or_threshold_selection": False,
        "semantic_group_cluster_uncertainty": uncertainty,
        "evaluation_protocol": _evaluation_protocol(),
        "point_gate_is_not_a_statistical_significance_claim": True,
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
            "evaluation_protocol": _evaluation_protocol(),
            "preflight": {
                **dict(preflight),
                "checkpoint": str(preflight["checkpoint"]),
            },
            "summary": {
                "selected_endpoint": selected_endpoint,
                "confirm_selected_endpoint_gate_pass": bool(
                    selected_gate["gate_pass"]
                ),
                "selected_natural_prior_gain": float(
                    selected_gate["natural_prior_gain"]
                ),
                "selected_minimum_per_class_delta": float(
                    selected_gate["minimum_per_class_recall_delta"]
                ),
                "post_raw_valid_label_rate": float(
                    selected_gate["common_checks"][
                        "post_raw_free_generation_valid_label_rate"
                    ]
                ),
                "margin_roc_auc_delta": float(
                    selected_gate["common_checks"]["margin_roc_auc_delta"]
                ),
                "cluster_bootstrap_natural_gain_interval": uncertainty[
                    "confidence_intervals"
                ]["selected_natural_prior_gain"],
            },
        },
    )
    _finalize_claim(claim_path, claimed_payload, result_path=result_path)
    (output_root / "COMPLETED").write_text(
        "PCSM adaptive-endpoint v2 one-shot confirmation completed\n",
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
        raise ValueError("v2 selection requires --data-root, --model-path, and --task")
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
                raise ValueError("confirmation data_root differs from locked v2 selection")
            if args.model_path is not None and (
                Path(args.model_path).expanduser().resolve()
                != Path(config.model_path).expanduser().resolve()
            ):
                raise ValueError("confirmation model_path differs from locked v2 selection")
            if args.task_name is not None and args.task_name != config.task_name:
                raise ValueError("confirmation task differs from locked v2 selection")
            if args.fresh_confirm_against_manifests and tuple(
                args.fresh_confirm_against_manifests
            ) != config.fresh_confirm_against_manifests:
                raise ValueError("confirmation freshness sources differ from locked v2 selection")
            if (
                args.task_protocol_registry is not None
                and args.task_protocol_registry != config.task_protocol_registry
            ):
                raise ValueError("confirmation registry differs from locked v2 selection")
            run_confirmation(config, selection, selection_path)
    except Exception as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
