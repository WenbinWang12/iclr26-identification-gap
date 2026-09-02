"""Train-only acquisition rescue search for hard Order-4 tasks.

This executable diagnoses whether weak diagonal acquisition on QQP, BoolQA, or
MultiRC is caused by an underpowered update schedule rather than continual
forgetting.  Its data envelope is intentionally narrower than the five-arm
panel: it reads each task's pinned ``train.json`` only and creates three
deterministic, class-balanced partitions:

``update``
    The only examples authorized for gradients.  Candidate caps select nested
    prefixes of this pool.
``tune_audit``
    Selects learning-rate/epoch/cap candidates by frozen-base-to-post-train
    normalized exact-match acquisition.
``confirm_audit``
    Remains sealed until one candidate has been selected.  It is evaluated
    exactly once for the chosen candidate and cannot affect selection.

No development or held-out benchmark split is available to this module.  A
successful run writes content-addressed split ledgers, all tune measurements,
the selected compact adapter, one confirmation result, and completion
sentinels.  The search is an acquisition diagnostic, not a continual-learning
result and not a substitute for the full Order-4 panel.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

import numpy as np
import torch
from torch import Tensor

try:
    from .objectives import per_example_target_token_nll
    from .order4_data import (
        OFFICIAL_COMMIT,
        OFFICIAL_REPOSITORY,
        TASK_BY_NAME,
        OfficialExample,
        load_official_examples,
        normalize_answer,
        normalized_em,
    )
    from .t5_rank_bank import T5GlobalRankBank
except ImportError:  # pragma: no cover - direct script execution.
    from objectives import per_example_target_token_nll  # type: ignore
    from order4_data import (  # type: ignore
        OFFICIAL_COMMIT,
        OFFICIAL_REPOSITORY,
        TASK_BY_NAME,
        OfficialExample,
        load_official_examples,
        normalize_answer,
        normalized_em,
    )
    from t5_rank_bank import T5GlobalRankBank  # type: ignore


SUPPORTED_TASKS: tuple[str, ...] = ("QQP", "BoolQA", "MultiRC")
LEGACY_FORMAT_VERSION = "phase2i.train-only-acquisition-rescue.v1"
FORMAT_VERSION = "phase2i.train-only-acquisition-rescue.v2"
ACCESS_ONLY_FRESHNESS_FORMAT = "phase2i.access-only-freshness.v1"
LOSS_OBJECTIVES: tuple[str, ...] = (
    "hf_token_ce",
    "sequence_mean_balanced",
    "sequence_mean_prior",
)
CONTROL_OBJECTIVE = "hf_token_ce"
UPDATE_SAMPLERS: tuple[str, ...] = ("balanced",)


def _strictly_above(value: float, threshold: float) -> bool:
    """Numerically stable strict comparison for metric fractions."""

    return value > threshold and not math.isclose(
        value, threshold, rel_tol=0.0, abs_tol=1e-12
    )


def _at_least(value: float, threshold: float) -> bool:
    return value >= threshold or math.isclose(
        value, threshold, rel_tol=0.0, abs_tol=1e-12
    )
SPLIT_UNITS: tuple[str, ...] = ("row", "semantic_group")
LEGACY_ROW_KEY_VERSION = "row-example-id.v1"
SEMANTIC_GROUP_KEY_VERSION = "semantic-group.v1"
LEGACY_SEMANTIC_SELECTION_ALGORITHM = "size-first-greedy.v1"
SEMANTIC_SELECTION_ALGORITHM_VERSION = "hash-priority-exact-dp.v2"
SEMANTIC_SELECTION_ALGORITHMS: tuple[str, ...] = (
    LEGACY_SEMANTIC_SELECTION_ALGORITHM,
    SEMANTIC_SELECTION_ALGORITHM_VERSION,
)


@dataclass(frozen=True)
class RescueSplit:
    """Disjoint train-derived update, tune-audit, and confirm-audit sets."""

    task_name: str
    update_by_label: Mapping[str, tuple[OfficialExample, ...]]
    tune_audit: tuple[OfficialExample, ...]
    confirm_audit: tuple[OfficialExample, ...]
    excluded_count: int
    # Defaults preserve the constructor and byte-for-byte manifest used by
    # legacy selections.  New strict runs explicitly request semantic_group.
    split_unit: str = "row"
    group_key_version: str = LEGACY_ROW_KEY_VERSION
    group_key_by_example_id: Mapping[str, str] | None = None
    freshness_sources: tuple[Mapping[str, Any], ...] = ()
    prior_touched_requested_example_ids: tuple[str, ...] = ()
    prior_touched_example_ids: tuple[str, ...] = ()
    prior_touched_group_keys: tuple[str, ...] = ()
    selection_algorithm_version: str | None = None

    @property
    def update_pool(self) -> tuple[OfficialExample, ...]:
        return tuple(
            sorted(
                (example for group in self.update_by_label.values() for example in group),
                key=lambda example: example.source_index,
            )
        )

    def update_for_cap(self, cap_per_class: int) -> tuple[OfficialExample, ...]:
        if isinstance(cap_per_class, bool) or not isinstance(cap_per_class, int):
            raise TypeError("cap_per_class must be an integer")
        if cap_per_class <= 0:
            raise ValueError("cap_per_class must be positive")
        short = {
            label: len(group)
            for label, group in self.update_by_label.items()
            if len(group) < cap_per_class
        }
        if short:
            raise ValueError(f"update cap {cap_per_class} exceeds label pools: {short}")
        return tuple(
            sorted(
                (
                    example
                    for group in self.update_by_label.values()
                    for example in group[:cap_per_class]
                ),
                key=lambda example: example.source_index,
            )
        )

    def assert_disjoint(self) -> None:
        partitions = (
            {example.example_id for example in self.update_pool},
            {example.example_id for example in self.tune_audit},
            {example.example_id for example in self.confirm_audit},
        )
        if partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2]:
            raise AssertionError("update/tune-audit/confirm-audit partitions overlap")
        all_examples = (*self.update_pool, *self.tune_audit, *self.confirm_audit)
        if any(example.subset != "train" for example in all_examples):
            raise AssertionError("acquisition rescue partitions must be train-derived")
        if self.split_unit == "semantic_group":
            if self.group_key_version != SEMANTIC_GROUP_KEY_VERSION:
                raise AssertionError("unsupported semantic group-key version")
            if self.selection_algorithm_version not in SEMANTIC_SELECTION_ALGORITHMS:
                raise AssertionError("unsupported semantic selection algorithm")
            if self.group_key_by_example_id is None:
                raise AssertionError("semantic split is missing its group-key ledger")
            group_partitions = tuple(
                {
                    self.group_key_by_example_id[example.example_id]
                    for example in partition
                }
                for partition in (self.update_pool, self.tune_audit, self.confirm_audit)
            )
            if (
                group_partitions[0] & group_partitions[1]
                or group_partitions[0] & group_partitions[2]
                or group_partitions[1] & group_partitions[2]
            ):
                raise AssertionError(
                    "update/tune-audit/confirm-audit semantic groups overlap"
                )
            if group_partitions[2] & set(self.prior_touched_group_keys):
                raise AssertionError(
                    "confirm-audit contains a semantic group touched by a prior run"
                )


@dataclass(frozen=True)
class CandidateSpec:
    learning_rate: float
    epochs: int
    cap_per_class: int
    loss_objective: str = "hf_token_ce"
    update_sampler: str = "balanced"

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("candidate learning rate must be finite and positive")
        if isinstance(self.epochs, bool) or self.epochs <= 0:
            raise ValueError("candidate epochs must be positive")
        if isinstance(self.cap_per_class, bool) or self.cap_per_class <= 0:
            raise ValueError("candidate cap_per_class must be positive")
        if self.loss_objective not in LOSS_OBJECTIVES:
            raise ValueError(f"unsupported loss objective {self.loss_objective!r}")
        if self.update_sampler not in UPDATE_SAMPLERS:
            raise ValueError(f"unsupported update sampler {self.update_sampler!r}")

    @property
    def candidate_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
        lr_text = format(self.learning_rate, ".3g").replace("+", "").replace(".", "p")
        objective = {
            "hf_token_ce": "tokce",
            "sequence_mean_balanced": "seqbal",
            "sequence_mean_prior": "seqprior",
        }[self.loss_objective]
        return (
            f"{objective}_{self.update_sampler}_lr{lr_text}_ep{self.epochs}_"
            f"cap{self.cap_per_class}_{digest}"
        )


@dataclass(frozen=True)
class CandidateMeasurement:
    candidate: CandidateSpec
    base_tune_em: float
    post_tune_em: float
    base_tune_label_constrained_accuracy: float
    post_tune_label_constrained_accuracy: float
    update_examples_per_epoch: int
    checkpoint: str
    base_tune_natural_prior_em: float | None = None
    post_tune_natural_prior_em: float | None = None
    base_tune_valid_rate: float = 1.0
    post_tune_valid_rate: float = 1.0
    base_tune_per_class_recall: Mapping[str, float] | None = None
    post_tune_per_class_recall: Mapping[str, float] | None = None

    @property
    def tune_acquisition(self) -> float:
        return self.post_tune_em - self.base_tune_em

    @property
    def tune_natural_prior_acquisition(self) -> float:
        base = (
            self.base_tune_em
            if self.base_tune_natural_prior_em is None
            else self.base_tune_natural_prior_em
        )
        post = (
            self.post_tune_em
            if self.post_tune_natural_prior_em is None
            else self.post_tune_natural_prior_em
        )
        return post - base

    @property
    def min_per_class_delta(self) -> float:
        if self.base_tune_per_class_recall is None or self.post_tune_per_class_recall is None:
            return 0.0
        labels = set(self.base_tune_per_class_recall) | set(
            self.post_tune_per_class_recall
        )
        return min(
            float(self.post_tune_per_class_recall[label])
            - float(self.base_tune_per_class_recall[label])
            for label in labels
        )

    def v2_gate_pass(self, min_gain: float) -> bool:
        return (
            _strictly_above(self.tune_natural_prior_acquisition, min_gain)
            and self.post_tune_valid_rate >= 0.99
            and _at_least(self.min_per_class_delta, -0.05)
        )

    @property
    def tune_label_constrained_acquisition(self) -> float:
        return (
            self.post_tune_label_constrained_accuracy
            - self.base_tune_label_constrained_accuracy
        )

    @property
    def update_exposures(self) -> int:
        return self.update_examples_per_epoch * self.candidate.epochs


@dataclass(frozen=True)
class RescueConfig:
    data_root: str
    model_path: str
    output_root: str
    task_names: tuple[str, ...]
    learning_rates: tuple[float, ...]
    epochs: tuple[int, ...]
    caps_per_class: tuple[int, ...]
    tune_audit_per_class: int
    confirm_audit_per_class: int
    seed: int
    batch_size: int
    max_source_length: int
    max_target_length: int
    truncation_mode: str
    min_rescue_gain: float
    tiny_random_model: bool
    selection_only: bool
    confirm_selection: str | None
    device: str
    split_unit: str = "row"
    fresh_confirm_against_manifests: tuple[str, ...] = ()
    loss_objectives: tuple[str, ...] = ("hf_token_ce",)
    update_sampler: str = "balanced"
    semantic_selection_algorithm: str = SEMANTIC_SELECTION_ALGORITHM_VERSION


@dataclass(frozen=True)
class SelectionObjectivePlan:
    """Separate pre-registered selection objectives from the matched control.

    In a selection-only acquisition-rescue run, token CE is a diagnostic
    control whenever at least one sequence-level objective is requested.  It is
    therefore trained exactly once *after* selection, at the selected
    LR/epoch/cap, instead of being crossed through the full search grid.  A
    token-CE-only invocation keeps its historical behavior for compatibility.
    Search-then-confirm invocations are also left unchanged.
    """

    requested_objectives: tuple[str, ...]
    selection_objectives: tuple[str, ...]
    train_matched_control_after_selection: bool


def selection_objective_plan(
    loss_objectives: Sequence[str], *, selection_only: bool
) -> SelectionObjectivePlan:
    requested = tuple(dict.fromkeys(str(item) for item in loss_objectives))
    if not requested:
        raise ValueError("loss objective list cannot be empty")
    unsupported = [item for item in requested if item not in LOSS_OBJECTIVES]
    if unsupported:
        raise ValueError(f"unsupported loss objectives: {unsupported}")
    non_control = tuple(item for item in requested if item != CONTROL_OBJECTIVE)
    if selection_only and non_control:
        return SelectionObjectivePlan(
            requested_objectives=requested,
            selection_objectives=non_control,
            train_matched_control_after_selection=True,
        )
    return SelectionObjectivePlan(
        requested_objectives=requested,
        selection_objectives=requested,
        train_matched_control_after_selection=False,
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_ids_sha256(example_ids: Sequence[str]) -> str:
    encoded = json.dumps(
        list(example_ids), sort_keys=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _access_content_row(example: OfficialExample) -> dict[str, Any]:
    """Bind every train field that can alter grouping, prompts, or labels."""

    return {
        "task_name": example.task_name,
        "category": example.category,
        "dataset": example.dataset,
        "subset": example.subset,
        "source_index": int(example.source_index),
        "example_id": example.example_id,
        "sentence": example.sentence,
        "label": example.label,
        "prompt": example.prompt,
    }


def _read_json_mapping(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description} {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{description} is not a JSON object: {path}")
    return payload


def _source_split_partition_ids(
    split_payload: Mapping[str, Any], *, task_name: str
) -> dict[str, list[str]]:
    if split_payload.get("task") != task_name:
        raise ValueError(f"source split manifest has no direct ledger for {task_name}")
    source_ids: dict[str, list[str]] = {}
    for partition_name in ("update_pool", "tune_audit", "confirm_audit"):
        partition = split_payload.get(partition_name)
        ids = partition.get("example_ids") if isinstance(partition, Mapping) else None
        if not isinstance(ids, list) or any(
            not isinstance(example_id, str) or not example_id for example_id in ids
        ):
            raise ValueError(
                f"source split manifest lacks valid {partition_name}.example_ids"
            )
        if len(ids) != len(set(ids)):
            raise ValueError(f"source split {partition_name} contains duplicate IDs")
        source_ids[partition_name] = list(ids)
    source_sets = [set(source_ids[name]) for name in source_ids]
    if (
        source_sets[0] & source_sets[1]
        or source_sets[0] & source_sets[2]
        or source_sets[1] & source_sets[2]
    ):
        raise ValueError("source split partitions overlap")
    return source_ids


def _file_evidence_record(path_value: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"access evidence is not a file: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _validate_selection_only_access_evidence(
    *,
    source_ids: Mapping[str, Sequence[str]],
    examples_by_id: Mapping[str, OfficialExample],
    candidate_metrics: Mapping[str, Any],
    base_tune: Mapping[str, Any],
) -> dict[str, Any]:
    """Enforce the runner semantics that justify an access-only ledger."""

    tune_ids = list(source_ids["tune_audit"])
    if base_tune.get("example_ids") != tune_ids:
        raise ValueError("base-tune evidence does not exactly match tune_audit IDs/order")
    for audit_name in ("base_tune", "post_tune"):
        audit = candidate_metrics.get(audit_name)
        if not isinstance(audit, Mapping) or audit.get("example_ids") != tune_ids:
            raise ValueError(
                f"candidate {audit_name} evidence does not exactly match tune_audit IDs/order"
            )
    if candidate_metrics.get("base_tune") != base_tune:
        raise ValueError("candidate and standalone base-tune evidence differ")
    if (
        candidate_metrics.get("confirm_audit_accessed") is not False
        or candidate_metrics.get("selection_partition") != "tune_audit"
    ):
        raise ValueError("candidate evidence does not attest selection-only confirm access")

    update_counts: dict[str, int] = {}
    for example_id in source_ids["update_pool"]:
        try:
            label = examples_by_id[example_id].label
        except KeyError as error:
            raise ValueError(f"update evidence references unknown ID {example_id!r}") from error
        update_counts[label] = update_counts.get(label, 0) + 1
    candidate = candidate_metrics.get("candidate")
    training = candidate_metrics.get("training")
    if not isinstance(candidate, Mapping) or not isinstance(training, Mapping):
        raise ValueError("candidate evidence lacks candidate/training ledgers")
    cap = candidate.get("cap_per_class")
    epochs = candidate.get("epochs")
    completed_epochs = training.get("completed_epochs")
    if (
        isinstance(cap, bool)
        or not isinstance(cap, int)
        or isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs < 1
        or completed_epochs != epochs
        or any(count != cap for count in update_counts.values())
    ):
        raise ValueError("candidate evidence does not cover the exact full update pool")
    resource = training.get("resource_ledger")
    if not isinstance(resource, Mapping):
        raise ValueError("candidate evidence lacks its training resource ledger")
    total_update = len(source_ids["update_pool"])
    if (
        resource.get("gradient_examples_per_epoch") != total_update
        or resource.get("gradient_example_exposures") != total_update * epochs
        or resource.get("sampler_label_counts_per_epoch") != update_counts
        or resource.get("tune_audit_gradient_examples") != 0
        or resource.get("confirm_audit_gradient_examples") != 0
    ):
        raise ValueError("candidate resource ledger does not prove exact update exposures")
    per_class = resource.get("per_class")
    if not isinstance(per_class, Mapping):
        raise ValueError("candidate resource ledger lacks per-class exposures")
    for label, count in update_counts.items():
        class_row = per_class.get(label)
        if not isinstance(class_row, Mapping) or class_row.get("row_exposures") != count * epochs:
            raise ValueError(
                f"candidate resource ledger has wrong {label} row exposures"
            )
    return {
        "candidate_id": candidate_metrics.get("candidate_id"),
        "candidate_cap_per_class": cap,
        "completed_epochs": epochs,
        "update_pool_exact_once_per_epoch": True,
        "gradient_example_exposures": total_update * epochs,
        "base_and_post_tune_exact_ids_and_order": True,
        "tune_audit_gradient_examples": 0,
        "confirm_audit_access_flag": False,
        "confirm_audit_gradient_examples": 0,
        "confirm_nonaccess_provenance": {
            "claim_type": "runner-artifact attestation; not independently observable",
            "candidate_confirm_audit_accessed": False,
            "scope": "not model-scored, tokenized, or selection-used",
            "split_id_metadata_was_read": True,
        },
    }


def build_access_only_freshness_ledger(
    source_split_path: str | os.PathLike[str],
    *,
    train_examples: Sequence[OfficialExample],
    task_name: str,
    candidate_metrics_evidence_path: str | os.PathLike[str],
    base_tune_evidence_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build a content-addressed ledger of rows that were actually accessed.

    A selection-only run preallocates confirmation rows in ``split.json``.  The
    builder necessarily reads those ID records to verify the partition ledger,
    but the selection-only artifacts must attest that they were never
    model-scored, tokenized, gradient-bearing, or used for selection.  Only the update and
    tune IDs are emitted as touched rows.
    """

    source = Path(source_split_path).expanduser().resolve()
    split_payload = _read_json_mapping(source, description="source split manifest")

    examples_by_id: dict[str, OfficialExample] = {}
    for example in train_examples:
        if example.task_name != task_name or example.subset != "train":
            raise ValueError("access ledger accepts the complete pinned train task only")
        if example.example_id in examples_by_id:
            raise ValueError(f"duplicate train example ID {example.example_id!r}")
        examples_by_id[example.example_id] = example
    if not examples_by_id:
        raise ValueError("access ledger cannot fingerprint an empty train corpus")

    source_ids = _source_split_partition_ids(split_payload, task_name=task_name)
    for partition_name, ids in source_ids.items():
        unknown = set(ids) - set(examples_by_id)
        if unknown:
            raise ValueError(
                f"source split {partition_name} contains {len(unknown)} unknown IDs"
            )

    def accessed_partition(partition_name: str) -> dict[str, Any]:
        ids = source_ids[partition_name]
        examples = [examples_by_id[example_id] for example_id in ids]
        counts: dict[str, int] = {}
        for example in examples:
            counts[example.label] = counts.get(example.label, 0) + 1
        content_rows = [_access_content_row(example) for example in examples]
        return {
            "accessed": True,
            "count": len(ids),
            "per_class": counts,
            "example_ids": ids,
            "ordered_ids_sha256": _ordered_ids_sha256(ids),
            "ordered_full_content_sha256": _canonical_payload_sha256(content_rows),
            "source_subset": "train",
        }

    empty_ids: list[str] = []
    empty_content_rows: list[dict[str, Any]] = []
    partitions = {
        "update_pool": accessed_partition("update_pool"),
        "tune_audit": accessed_partition("tune_audit"),
        "confirm_audit": {
            "accessed": False,
            "count": 0,
            "per_class": {},
            "example_ids": empty_ids,
            "ordered_ids_sha256": _ordered_ids_sha256(empty_ids),
            "ordered_full_content_sha256": _canonical_payload_sha256(
                empty_content_rows
            ),
            "source_subset": "train",
            "status": "runner-attested-not-model-accessed",
            "nonaccess_claim_type": (
                "provenance from selection-only artifacts; not independently observable"
            ),
            "id_metadata_read_by_ledger_builder": True,
            "runner_attested_model_scored": False,
            "runner_attested_tokenized": False,
            "runner_attested_selection_used": False,
            "preallocated_but_not_touched_count": len(
                source_ids["confirm_audit"]
            ),
        },
    }
    accessed_ids = [
        *partitions["update_pool"]["example_ids"],
        *partitions["tune_audit"]["example_ids"],
    ]
    accessed_rows = [
        _access_content_row(examples_by_id[example_id]) for example_id in accessed_ids
    ]

    candidate_record = _file_evidence_record(candidate_metrics_evidence_path)
    base_record = _file_evidence_record(base_tune_evidence_path)
    candidate_metrics = _read_json_mapping(
        Path(candidate_record["path"]), description="candidate metrics evidence"
    )
    base_tune = _read_json_mapping(
        Path(base_record["path"]), description="base-tune evidence"
    )
    enforced_access_checks = _validate_selection_only_access_evidence(
        source_ids=source_ids,
        examples_by_id=examples_by_id,
        candidate_metrics=candidate_metrics,
        base_tune=base_tune,
    )
    evidence_payload = {
        "candidate_metrics": candidate_record,
        "base_tune": base_record,
    }

    full_train_rows = [
        _access_content_row(example)
        for example in sorted(train_examples, key=lambda item: item.source_index)
    ]
    unsigned: dict[str, Any] = {
        "format": ACCESS_ONLY_FRESHNESS_FORMAT,
        "task": task_name,
        "access_policy": {
            "touched_definition": (
                "gradient-bearing update rows or development-scored tune rows only"
            ),
            "source_split_preallocation_is_not_access": True,
            "confirm_id_metadata_read_by_ledger_builder": True,
            "confirm_nonaccess_claim_type": (
                "provenance from selection-only artifacts; not independently observable"
            ),
            "confirm_nonaccess_scope": (
                "runner-attested not model-scored, tokenized, gradient-bearing, or selection-used"
            ),
        },
        "source_split": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": _sha256_file(source),
        },
        "pinned_train": {
            "row_count": len(full_train_rows),
            "ordered_full_content_sha256": _canonical_payload_sha256(full_train_rows),
        },
        **partitions,
        "accessed_union": {
            "count": len(accessed_ids),
            "ordered_ids_sha256": _ordered_ids_sha256(accessed_ids),
            "ordered_full_content_sha256": _canonical_payload_sha256(accessed_rows),
        },
        "access_evidence": evidence_payload,
        "enforced_access_checks": enforced_access_checks,
    }
    return {
        **unsigned,
        "ledger_payload_sha256": _canonical_payload_sha256(unsigned),
    }


def _verify_file_evidence_record(
    record: Any, *, description: str
) -> Path:
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise ValueError(f"access-only freshness ledger lacks {description} record")
    path = Path(record["path"]).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"access-only freshness {description} is missing: {path}")
    if record.get("bytes") != path.stat().st_size:
        raise ValueError(f"access-only freshness {description} byte count drifted")
    if record.get("sha256") != _sha256_file(path):
        raise ValueError(f"access-only freshness {description} SHA-256 drifted")
    return path


def _load_access_only_freshness_ledger(
    payload: Mapping[str, Any],
    *,
    source: Path,
    task_name: str,
    train_examples: Sequence[OfficialExample] | None,
) -> tuple[set[str], dict[str, Any]]:
    if payload.get("task") != task_name:
        raise ValueError(f"access-only freshness ledger {source} is not for {task_name}")
    supplied_payload_sha = payload.get("ledger_payload_sha256")
    unsigned = dict(payload)
    unsigned.pop("ledger_payload_sha256", None)
    expected_payload_sha = _canonical_payload_sha256(unsigned)
    if supplied_payload_sha != expected_payload_sha:
        raise ValueError(f"access-only freshness ledger payload hash mismatch: {source}")
    policy = payload.get("access_policy")
    if not isinstance(policy, Mapping) or (
        policy.get("source_split_preallocation_is_not_access") is not True
        or policy.get("confirm_id_metadata_read_by_ledger_builder") is not True
        or policy.get("confirm_nonaccess_claim_type")
        != "provenance from selection-only artifacts; not independently observable"
        or policy.get("confirm_nonaccess_scope")
        != "runner-attested not model-scored, tokenized, gradient-bearing, or selection-used"
    ):
        raise ValueError(f"access-only freshness ledger has invalid policy: {source}")

    if train_examples is None:
        raise ValueError(
            "access-only freshness ledger requires the verified pinned train examples"
        )
    examples_by_id: dict[str, OfficialExample] = {}
    for example in train_examples:
        if example.task_name != task_name or example.subset != "train":
            raise ValueError("access-only freshness validation escaped pinned train")
        if example.example_id in examples_by_id:
            raise ValueError(f"duplicate train example ID {example.example_id!r}")
        examples_by_id[example.example_id] = example
    full_train_rows = [
        _access_content_row(example)
        for example in sorted(train_examples, key=lambda item: item.source_index)
    ]
    pinned = payload.get("pinned_train")
    if not isinstance(pinned, Mapping) or (
        pinned.get("row_count") != len(full_train_rows)
        or pinned.get("ordered_full_content_sha256")
        != _canonical_payload_sha256(full_train_rows)
    ):
        raise ValueError("access-only freshness pinned-train content drifted")

    source_split_path = _verify_file_evidence_record(
        payload.get("source_split"), description="source split"
    )
    split_payload = _read_json_mapping(
        source_split_path, description="source split manifest"
    )
    source_ids = _source_split_partition_ids(split_payload, task_name=task_name)
    for partition_name, ids in source_ids.items():
        unknown = set(ids) - set(examples_by_id)
        if unknown:
            raise ValueError(
                f"source split {partition_name} contains {len(unknown)} unknown IDs"
            )

    example_ids: set[str] = set()
    ordered_accessed_ids: list[str] = []
    partition_counts: dict[str, int] = {}
    for partition_name in ("update_pool", "tune_audit", "confirm_audit"):
        partition = payload.get(partition_name)
        if not isinstance(partition, Mapping):
            raise ValueError(
                f"access-only freshness ledger lacks {partition_name}: {source}"
            )
        ids = partition.get("example_ids")
        if not isinstance(ids, list) or any(
            not isinstance(example_id, str) or not example_id for example_id in ids
        ):
            raise ValueError(
                f"access-only freshness ledger has invalid {partition_name} IDs"
            )
        should_be_accessed = partition_name != "confirm_audit"
        if partition.get("accessed") is not should_be_accessed:
            raise ValueError(
                f"access-only freshness ledger has invalid {partition_name} access flag"
            )
        if partition_name == "confirm_audit":
            if (
                ids
                or partition.get("status") != "runner-attested-not-model-accessed"
                or partition.get("nonaccess_claim_type")
                != "provenance from selection-only artifacts; not independently observable"
                or partition.get("id_metadata_read_by_ledger_builder") is not True
                or partition.get("runner_attested_model_scored") is not False
                or partition.get("runner_attested_tokenized") is not False
                or partition.get("runner_attested_selection_used") is not False
                or partition.get("preallocated_but_not_touched_count")
                != len(source_ids["confirm_audit"])
            ):
                raise ValueError(
                    "access-only freshness confirm nonaccess metadata is invalid"
                )
        elif ids != source_ids[partition_name]:
            raise ValueError(
                f"access-only freshness {partition_name} differs from source split"
            )
        if partition.get("count") != len(ids):
            raise ValueError(
                f"access-only freshness ledger {partition_name} count mismatch"
            )
        if partition.get("ordered_ids_sha256") != _ordered_ids_sha256(ids):
            raise ValueError(
                f"access-only freshness ledger {partition_name} ID hash mismatch"
            )
        content_rows = [
            _access_content_row(examples_by_id[example_id]) for example_id in ids
        ]
        if partition.get("ordered_full_content_sha256") != _canonical_payload_sha256(
            content_rows
        ):
            raise ValueError(
                f"access-only freshness ledger {partition_name} content hash drifted"
            )
        counts: dict[str, int] = {}
        for example_id in ids:
            label = examples_by_id[example_id].label
            counts[label] = counts.get(label, 0) + 1
        if partition.get("per_class") != counts:
            raise ValueError(
                f"access-only freshness ledger {partition_name} class counts drifted"
            )
        duplicate_or_overlap = example_ids.intersection(ids)
        if duplicate_or_overlap:
            raise ValueError(
                f"access-only freshness ledger repeats {len(duplicate_or_overlap)} IDs"
            )
        example_ids.update(ids)
        if should_be_accessed:
            ordered_accessed_ids.extend(ids)
        partition_counts[partition_name] = len(ids)

    union = payload.get("accessed_union")
    if not isinstance(union, Mapping) or (
        union.get("count") != len(ordered_accessed_ids)
        or union.get("ordered_ids_sha256")
        != _ordered_ids_sha256(ordered_accessed_ids)
    ):
        raise ValueError(f"access-only freshness ledger union mismatch: {source}")
    accessed_rows = [
        _access_content_row(examples_by_id[example_id])
        for example_id in ordered_accessed_ids
    ]
    if union.get("ordered_full_content_sha256") != _canonical_payload_sha256(
        accessed_rows
    ):
        raise ValueError("access-only freshness ledger union content hash drifted")

    evidence = payload.get("access_evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "candidate_metrics",
        "base_tune",
    }:
        raise ValueError("access-only freshness ledger evidence roles are invalid")
    candidate_path = _verify_file_evidence_record(
        evidence["candidate_metrics"], description="candidate metrics evidence"
    )
    base_tune_path = _verify_file_evidence_record(
        evidence["base_tune"], description="base-tune evidence"
    )
    enforced_checks = _validate_selection_only_access_evidence(
        source_ids=source_ids,
        examples_by_id=examples_by_id,
        candidate_metrics=_read_json_mapping(
            candidate_path, description="candidate metrics evidence"
        ),
        base_tune=_read_json_mapping(
            base_tune_path, description="base-tune evidence"
        ),
    )
    if payload.get("enforced_access_checks") != enforced_checks:
        raise ValueError("access-only freshness enforced evidence summary drifted")
    return example_ids, {
        "path": str(source),
        "sha256": _sha256_file(source),
        "format": ACCESS_ONLY_FRESHNESS_FORMAT,
        "task": task_name,
        "access_policy": "actual-access-only",
        "partition_row_counts": partition_counts,
        "unique_requested_example_ids": len(example_ids),
        "ledger_payload_sha256": supplied_payload_sha,
    }


def _load_freshness_split_manifest(
    path: str | os.PathLike[str],
    *,
    task_name: str,
    train_examples: Sequence[OfficialExample] | None = None,
) -> tuple[set[str], dict[str, Any]]:
    """Read one prior split ledger and return all rows known to be accessed."""

    source = Path(path).expanduser().resolve()
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read freshness split manifest {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"freshness split manifest is not an object: {source}")
    if payload.get("format") == ACCESS_ONLY_FRESHNESS_FORMAT:
        return _load_access_only_freshness_ledger(
            payload,
            source=source,
            task_name=task_name,
            train_examples=train_examples,
        )
    candidate: Any = payload
    if isinstance(payload.get("splits"), Mapping):
        candidate = payload["splits"].get(task_name)
    elif payload.get("task") not in (None, task_name):
        candidate = None
    if not isinstance(candidate, Mapping) or candidate.get("task") != task_name:
        raise ValueError(
            f"freshness split manifest {source} has no ledger for {task_name}"
        )
    example_ids: set[str] = set()
    partition_counts: dict[str, int] = {}
    for partition_name in ("update_pool", "tune_audit", "confirm_audit"):
        ledger = candidate.get(partition_name)
        if not isinstance(ledger, Mapping) or not isinstance(
            ledger.get("example_ids"), list
        ):
            raise ValueError(
                f"freshness split manifest {source} lacks {partition_name}.example_ids"
            )
        ids = ledger["example_ids"]
        if any(not isinstance(item, str) or not item for item in ids):
            raise ValueError(
                f"freshness split manifest {source} has invalid example IDs"
            )
        example_ids.update(ids)
        partition_counts[partition_name] = len(ids)
    return example_ids, {
        "path": str(source),
        "sha256": _sha256_file(source),
        "task": task_name,
        "partition_row_counts": partition_counts,
        "unique_requested_example_ids": len(example_ids),
    }


def load_prior_touched_ids(
    paths: Sequence[str],
    *,
    task_name: str,
    train_examples: Sequence[OfficialExample] | None = None,
) -> tuple[set[str], tuple[Mapping[str, Any], ...]]:
    requested: set[str] = set()
    sources: list[Mapping[str, Any]] = []
    for path in paths:
        ids, source = _load_freshness_split_manifest(
            path, task_name=task_name, train_examples=train_examples
        )
        requested.update(ids)
        sources.append(source)
    return requested, tuple(sources)


def _stable_seed(base_seed: int, *parts: object) -> int:
    material = "\0".join([str(int(base_seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _partition_key(example: OfficialExample, seed: int) -> str:
    payload = f"acquisition-rescue\0{seed}\0{example.task_name}\0{example.label}\0{example.example_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_group_text(value: str) -> str:
    """Canonicalize benchmark text only for semantic leakage grouping."""

    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def _split_once(value: str, marker: str, *, task_name: str) -> tuple[str, str]:
    head, separator, tail = value.partition(marker)
    if not separator:
        raise ValueError(f"{task_name} sentence lacks {marker.strip()!r} field")
    return head, tail


def _semantic_group_keys(
    examples: Sequence[OfficialExample], *, task_name: str
) -> dict[str, str]:
    """Return full-corpus semantic group IDs for strict train-only splitting.

    MultiRC candidates sharing paragraph+question are one unit.  BoolQA rows
    sharing a passage are one unit.  QQP uses connected components in the
    normalized-question graph, so even transitive shared-question chains
    cannot cross partitions.
    """

    raw_key_by_id: dict[str, str] = {}
    if task_name == "MultiRC":
        for example in examples:
            body, candidate = _split_once(
                example.sentence,
                "\ncandidate answer: ",
                task_name=task_name,
            )
            paragraph, question = _split_once(
                body,
                "\nquestion: ",
                task_name=task_name,
            )
            if not paragraph.startswith("paragraph: ") or not candidate:
                raise ValueError("MultiRC sentence has malformed semantic fields")
            payload = json.dumps(
                {
                    "paragraph": _normalize_group_text(paragraph[len("paragraph: ") :]),
                    "question": _normalize_group_text(question),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            raw_key_by_id[example.example_id] = payload
    elif task_name == "BoolQA":
        for example in examples:
            question, passage = _split_once(
                example.sentence,
                "\npassage: ",
                task_name=task_name,
            )
            if not question.startswith("question: ") or not passage:
                raise ValueError("BoolQA sentence has malformed semantic fields")
            # Context isolation is deliberately stricter than exact-row
            # deduplication: every question about one passage stays together.
            raw_key_by_id[example.example_id] = _normalize_group_text(passage)
    elif task_name == "QQP":
        pairs: list[tuple[OfficialExample, str, str]] = []
        parent: dict[str, str] = {}

        def find(node: str) -> str:
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                return
            # Lexical rooting makes components independent of source order.
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

        for example in examples:
            first, second = _split_once(
                example.sentence,
                "\nsecond sentence: ",
                task_name=task_name,
            )
            if not first.startswith("first sentence: ") or not second:
                raise ValueError("QQP sentence has malformed semantic fields")
            q1 = _normalize_group_text(first[len("first sentence: ") :])
            q2 = _normalize_group_text(second)
            union(q1, q2)
            pairs.append((example, q1, q2))
        nodes_by_root: dict[str, list[str]] = {}
        for node in sorted(parent):
            nodes_by_root.setdefault(find(node), []).append(node)
        component_payload = {
            root: json.dumps(nodes, ensure_ascii=False, separators=(",", ":"))
            for root, nodes in nodes_by_root.items()
        }
        for example, q1, _q2 in pairs:
            raw_key_by_id[example.example_id] = component_payload[find(q1)]
    else:  # pragma: no cover - guarded by the public builder.
        raise ValueError(f"unsupported rescue task {task_name!r}")

    return {
        example_id: hashlib.sha256(
            (
                f"{SEMANTIC_GROUP_KEY_VERSION}\0{task_name}\0{raw_key}"
            ).encode("utf-8")
        ).hexdigest()
        for example_id, raw_key in raw_key_by_id.items()
    }


def _semantic_partition_key(task_name: str, group_key: str, seed: int) -> str:
    payload = f"acquisition-rescue-semantic\0{seed}\0{task_name}\0{group_key}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _take_exact_group_partition(
    remaining: dict[str, tuple[OfficialExample, ...]],
    *,
    task_name: str,
    labels: Sequence[str],
    quota_per_class: int,
    seed: int,
    algorithm_version: str = SEMANTIC_SELECTION_ALGORITHM_VERSION,
) -> tuple[tuple[OfficialExample, ...], dict[str, tuple[OfficialExample, ...]]]:
    """Take whole groups and meet every class quota exactly.

    The strict v2 algorithm gives every semantic group a deterministic
    hash-random priority, independent of group size, then uses bounded exact
    subset-sum dynamic programming.  The first exact state reached in that
    priority stream is used.  This preserves group atomicity and exact label
    quotas without the v1 size-first sampling bias.

    ``size-first-greedy.v1`` remains available only so an independent
    confirmation can reconstruct and hash-check already persisted selections.
    """

    if algorithm_version not in SEMANTIC_SELECTION_ALGORITHMS:
        raise ValueError(
            f"unsupported semantic selection algorithm {algorithm_version!r}"
        )
    deficits = {label: quota_per_class for label in labels}
    available = dict(remaining)
    if algorithm_version == LEGACY_SEMANTIC_SELECTION_ALGORITHM:
        chosen: list[OfficialExample] = []
        ordered = sorted(
            available.items(),
            key=lambda item: (
                -len(item[1]),
                _semantic_partition_key(task_name, item[0], seed),
                item[0],
            ),
        )
        for group_key, group in ordered:
            counts = {label: 0 for label in labels}
            for example in group:
                counts[example.label] += 1
            if all(counts[label] <= deficits[label] for label in labels):
                chosen.extend(group)
                del available[group_key]
                for label in labels:
                    deficits[label] -= counts[label]
                if not any(deficits.values()):
                    break
        if any(deficits.values()):
            raise ValueError(
                f"{task_name} semantic groups cannot satisfy exact per-class quota "
                f"{quota_per_class}; remaining deficits={deficits}"
            )
        return (
            tuple(sorted(chosen, key=lambda example: example.source_index)),
            available,
        )

    label_index = {label: index for index, label in enumerate(labels)}
    target = tuple(quota_per_class for _ in labels)
    ordered = sorted(
        available.items(),
        key=lambda item: (
            _semantic_partition_key(task_name, item[0], seed),
            item[0],
        ),
    )
    vectors: list[tuple[str, tuple[OfficialExample, ...], tuple[int, ...]]] = []
    for group_key, group in ordered:
        counts = [0] * len(labels)
        for example in group:
            try:
                counts[label_index[example.label]] += 1
            except KeyError as error:  # pragma: no cover - public builder validates.
                raise ValueError(f"unexpected label {example.label!r}") from error
        vector = tuple(counts)
        if all(vector[index] <= target[index] for index in range(len(labels))):
            vectors.append((group_key, group, vector))

    origin = tuple(0 for _ in labels)
    # All pinned rescue tasks are binary.  Bitset rows make the bounded exact
    # solver O(groups * quota) big-integer operations rather than scanning the
    # full quota square for every group.  Predecessors are materialized only
    # once per newly reachable state, so exact reconstruction remains cheap.
    predecessor: dict[tuple[int, ...], tuple[tuple[int, ...], str] | None] = {
        origin: None
    }
    if len(labels) == 2:
        quota_first, quota_second = target
        reachable = [0] * (quota_first + 1)
        reachable[0] = 1  # bit b denotes reachable state (0, b)
        bit_mask = (1 << (quota_second + 1)) - 1
        for group_key, _group, vector in vectors:
            delta_first, delta_second = vector
            before = tuple(reachable)
            for first in range(quota_first - delta_first + 1):
                shifted = (before[first] << delta_second) & bit_mask
                destination_first = first + delta_first
                new_bits = shifted & ~reachable[destination_first]
                reachable[destination_first] |= shifted
                while new_bits:
                    lowest = new_bits & -new_bits
                    destination_second = lowest.bit_length() - 1
                    predecessor[(destination_first, destination_second)] = (
                        (first, destination_second - delta_second),
                        group_key,
                    )
                    new_bits ^= lowest
            if reachable[quota_first] & (1 << quota_second):
                break
    else:  # pragma: no cover - the public rescue task registry is binary.
        for group_key, _group, vector in vectors:
            snapshot = tuple(predecessor)
            for state in snapshot:
                candidate = tuple(
                    state[index] + vector[index] for index in range(len(labels))
                )
                if any(
                    candidate[index] > target[index]
                    for index in range(len(labels))
                ):
                    continue
                if candidate not in predecessor:
                    predecessor[candidate] = (state, group_key)
                if candidate == target:
                    break
            if target in predecessor:
                break

    if target not in predecessor:
        raise ValueError(
            f"{task_name} semantic groups cannot satisfy exact per-class quota "
            f"{quota_per_class}; exact hash-priority DP found no feasible subset"
        )

    chosen_keys: list[str] = []
    cursor = target
    while cursor != origin:
        step = predecessor[cursor]
        if step is None:  # pragma: no cover - only origin has no predecessor.
            raise AssertionError("broken semantic subset predecessor chain")
        cursor, group_key = step
        chosen_keys.append(group_key)
    chosen_key_set = set(chosen_keys)
    chosen = [
        example
        for group_key in chosen_keys
        for example in available[group_key]
    ]
    available = {
        group_key: group
        for group_key, group in available.items()
        if group_key not in chosen_key_set
    }
    return (
        tuple(sorted(chosen, key=lambda example: example.source_index)),
        available,
    )


def build_rescue_split(
    examples: Sequence[OfficialExample],
    *,
    task_name: str,
    max_update_per_class: int,
    tune_audit_per_class: int,
    confirm_audit_per_class: int,
    seed: int,
    split_unit: str = "row",
    semantic_selection_algorithm: str = SEMANTIC_SELECTION_ALGORITHM_VERSION,
    prior_touched_example_ids: Iterable[str] = (),
    freshness_sources: Sequence[Mapping[str, Any]] = (),
) -> RescueSplit:
    """Partition train rows with fixed per-class quotas.

    ``row`` exactly reproduces the legacy hash split.  ``semantic_group`` is
    the strict protocol: related rows are assigned atomically and quotas are
    still exact for every label.
    """

    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported rescue task {task_name!r}")
    if split_unit not in SPLIT_UNITS:
        raise ValueError(f"split_unit must be one of {SPLIT_UNITS}, got {split_unit!r}")
    if (
        split_unit == "semantic_group"
        and semantic_selection_algorithm not in SEMANTIC_SELECTION_ALGORITHMS
    ):
        raise ValueError(
            "unsupported semantic selection algorithm "
            f"{semantic_selection_algorithm!r}"
        )
    quotas = (max_update_per_class, tune_audit_per_class, confirm_audit_per_class)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in quotas):
        raise ValueError("all per-class split quotas must be positive integers")
    if not examples:
        raise ValueError("cannot split an empty training set")
    if any(example.subset != "train" for example in examples):
        raise ValueError("acquisition rescue accepts train.json examples only")
    if any(example.task_name != task_name for example in examples):
        raise ValueError("all examples must belong to the requested task")

    requested_touched_ids = set(prior_touched_example_ids)
    if requested_touched_ids and split_unit != "semantic_group":
        raise ValueError(
            "fresh-confirm manifests require split_unit='semantic_group'"
        )
    known_ids = {example.example_id for example in examples}
    unknown_touched_ids = requested_touched_ids - known_ids
    if unknown_touched_ids:
        raise ValueError(
            f"{task_name} freshness manifests contain {len(unknown_touched_ids)} "
            "example IDs absent from the pinned train split"
        )

    expected_labels = TASK_BY_NAME[task_name].labels
    required = sum(quotas)

    group_key_by_example_id: Mapping[str, str] | None = None
    touched_group_keys: set[str] = set()
    expanded_touched_example_ids: set[str] = set()
    if split_unit == "semantic_group":
        group_key_by_example_id = _semantic_group_keys(examples, task_name=task_name)
        touched_group_keys = {
            group_key_by_example_id[example_id]
            for example_id in requested_touched_ids
        }
        expanded_touched_example_ids = {
            example.example_id
            for example in examples
            if group_key_by_example_id[example.example_id] in touched_group_keys
        }

    grouped: dict[str, list[OfficialExample]] = {label: [] for label in expected_labels}
    for example in examples:
        try:
            grouped[example.label].append(example)
        except KeyError as error:
            raise ValueError(f"unexpected label {example.label!r}") from error
    available_by_label = {label: len(grouped[label]) for label in expected_labels}
    insufficient = {
        label: count for label, count in available_by_label.items() if count < required
    }
    if insufficient:
        raise ValueError(
            f"{task_name} has insufficient train rows; "
            f"available_by_label={available_by_label}, required_per_label={required}"
        )

    if split_unit == "row":
        update_by_label: dict[str, tuple[OfficialExample, ...]] = {}
        tune: list[OfficialExample] = []
        confirm: list[OfficialExample] = []
        assigned = 0
        for label in expected_labels:
            ranked = sorted(
                grouped[label],
                key=lambda example: (_partition_key(example, seed), example.example_id),
            )
            # Confirmation receives the first deterministic block and remains
            # untouched until after selection.  Tune and nested update prefixes
            # use disjoint subsequent blocks.
            confirm_block = ranked[:confirm_audit_per_class]
            tune_start = confirm_audit_per_class
            tune_end = tune_start + tune_audit_per_class
            tune_block = ranked[tune_start:tune_end]
            update_block = ranked[tune_end : tune_end + max_update_per_class]
            confirm.extend(confirm_block)
            tune.extend(tune_block)
            update_by_label[label] = tuple(update_block)
            assigned += required
    else:
        assert group_key_by_example_id is not None
        all_group_lists: dict[str, list[OfficialExample]] = {}
        for example in examples:
            all_group_lists.setdefault(
                group_key_by_example_id[example.example_id], []
            ).append(example)
        all_groups: dict[str, tuple[OfficialExample, ...]] = {
            key: tuple(sorted(group, key=lambda item: item.source_index))
            for key, group in all_group_lists.items()
        }
        fresh_confirm_groups = {
            key: group
            for key, group in all_groups.items()
            if key not in touched_group_keys
        }
        fresh_available_by_label = {
            label: sum(
                example.label == label
                for group in fresh_confirm_groups.values()
                for example in group
            )
            for label in expected_labels
        }
        if any(
            count < confirm_audit_per_class
            for count in fresh_available_by_label.values()
        ):
            raise ValueError(
                f"{task_name} has insufficient untouched rows for fresh confirm; "
                f"fresh_available_by_label={fresh_available_by_label}, "
                f"required_per_label={confirm_audit_per_class}"
            )
        confirm_tuple, _unused_fresh = _take_exact_group_partition(
            fresh_confirm_groups,
            task_name=task_name,
            labels=expected_labels,
            quota_per_class=confirm_audit_per_class,
            seed=_stable_seed(seed, "confirm"),
            algorithm_version=semantic_selection_algorithm,
        )
        confirm_group_keys = {
            group_key_by_example_id[example.example_id]
            for example in confirm_tuple
        }
        # Prior development rows may be reused by tune/update.  Only the new
        # confirmation partition must be genuinely untouched.
        remaining = {
            key: group
            for key, group in all_groups.items()
            if key not in confirm_group_keys
        }
        tune_tuple, remaining = _take_exact_group_partition(
            remaining,
            task_name=task_name,
            labels=expected_labels,
            quota_per_class=tune_audit_per_class,
            seed=_stable_seed(seed, "tune"),
            algorithm_version=semantic_selection_algorithm,
        )
        update_tuple, remaining = _take_exact_group_partition(
            remaining,
            task_name=task_name,
            labels=expected_labels,
            quota_per_class=max_update_per_class,
            seed=_stable_seed(seed, "update"),
            algorithm_version=semantic_selection_algorithm,
        )
        confirm = list(confirm_tuple)
        tune = list(tune_tuple)
        update_by_label = {
            label: tuple(
                sorted(
                    (example for example in update_tuple if example.label == label),
                    key=lambda example: (
                        _semantic_partition_key(
                            task_name,
                            group_key_by_example_id[example.example_id],
                            _stable_seed(seed, "update-prefix"),
                        ),
                        example.example_id,
                    ),
                )
            )
            for label in expected_labels
        }
        assigned = len(confirm) + len(tune) + len(update_tuple)
    split = RescueSplit(
        task_name=task_name,
        update_by_label=update_by_label,
        tune_audit=tuple(sorted(tune, key=lambda example: example.source_index)),
        confirm_audit=tuple(sorted(confirm, key=lambda example: example.source_index)),
        excluded_count=len(examples) - assigned,
        split_unit=split_unit,
        group_key_version=(
            SEMANTIC_GROUP_KEY_VERSION
            if split_unit == "semantic_group"
            else LEGACY_ROW_KEY_VERSION
        ),
        group_key_by_example_id=group_key_by_example_id,
        freshness_sources=tuple(dict(item) for item in freshness_sources),
        prior_touched_requested_example_ids=tuple(sorted(requested_touched_ids)),
        prior_touched_example_ids=tuple(sorted(expanded_touched_example_ids)),
        prior_touched_group_keys=tuple(sorted(touched_group_keys)),
        selection_algorithm_version=(
            semantic_selection_algorithm
            if split_unit == "semantic_group"
            else None
        ),
    )
    split.assert_disjoint()
    return split


def pinned_train_label_distribution(
    examples: Sequence[OfficialExample], *, task_name: str
) -> dict[str, Any]:
    """Derive and content-address the natural label prior from pinned train rows."""

    if not examples or any(
        example.task_name != task_name or example.subset != "train"
        for example in examples
    ):
        raise ValueError("label prior requires the complete pinned train task rows")
    labels = tuple(TASK_BY_NAME[task_name].labels)
    counts = {label: 0 for label in labels}
    rows: list[dict[str, Any]] = []
    for example in sorted(examples, key=lambda item: item.source_index):
        if example.label not in counts:
            raise ValueError(f"unexpected train label {example.label!r}")
        counts[example.label] += 1
        rows.append(
            {
                "source_index": int(example.source_index),
                "example_id": example.example_id,
                "label": example.label,
            }
        )
    total = sum(counts.values())
    prior = {label: counts[label] / total for label in labels}
    content = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    statistics = json.dumps(
        {"label_counts": counts, "natural_label_prior": prior, "total": total},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "source": "complete verified pinned train.json",
        "task": task_name,
        "total_rows": total,
        "label_counts": counts,
        "natural_label_prior": prior,
        "ordered_id_label_rows_sha256": hashlib.sha256(content).hexdigest(),
        "statistics_sha256": hashlib.sha256(statistics).hexdigest(),
        "manual_prior_allowed": False,
    }


def candidate_grid(
    learning_rates: Sequence[float],
    epochs: Sequence[int],
    caps_per_class: Sequence[int],
    loss_objectives: Sequence[str] = ("hf_token_ce",),
    update_sampler: str = "balanced",
) -> tuple[CandidateSpec, ...]:
    candidates = {
        CandidateSpec(
            float(lr), int(epoch), int(cap), str(objective), str(update_sampler)
        )
        for cap in caps_per_class
        for epoch in epochs
        for lr in learning_rates
        for objective in loss_objectives
    }
    if not candidates:
        raise ValueError("candidate grid is empty")
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.cap_per_class,
                item.loss_objective,
                item.epochs,
                item.learning_rate,
            ),
        )
    )


def select_candidate(
    measurements: Sequence[CandidateMeasurement], *, min_gain: float = 0.05
) -> CandidateMeasurement:
    """Select by the v2 natural-prior acquisition gate and resource tie-breaks.

    Remaining exact ties prefer fewer epochs, a smaller cap, a lower learning
    rate, and finally the canonical candidate ID.  Confirmation metrics are not
    accepted by this function, making selection leakage structurally harder.
    """

    if not measurements:
        raise ValueError("cannot select from an empty measurement set")
    if len({item.candidate.candidate_id for item in measurements}) != len(measurements):
        raise ValueError("candidate measurements contain duplicates")
    for item in measurements:
        for value in (
            item.base_tune_em,
            item.post_tune_em,
            item.base_tune_label_constrained_accuracy,
            item.post_tune_label_constrained_accuracy,
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("tune EM scores must be finite fractions")
        for value in (
            item.tune_natural_prior_acquisition,
            item.base_tune_valid_rate,
            item.post_tune_valid_rate,
            item.min_per_class_delta,
        ):
            if not math.isfinite(value):
                raise ValueError("v2 tune metrics must be finite")
        if not 0.0 <= item.base_tune_valid_rate <= 1.0 or not 0.0 <= item.post_tune_valid_rate <= 1.0:
            raise ValueError("valid-label rates must be fractions")
    return min(
        measurements,
        key=lambda item: (
            -int(item.v2_gate_pass(min_gain)),
            -item.tune_natural_prior_acquisition,
            -(
                item.post_tune_em
                if item.post_tune_natural_prior_em is None
                else item.post_tune_natural_prior_em
            ),
            -item.post_tune_valid_rate,
            -item.min_per_class_delta,
            item.update_exposures,
            item.candidate.epochs,
            item.candidate.cap_per_class,
            item.candidate.learning_rate,
            item.candidate.candidate_id,
        ),
    )


def _chunks(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _tokenize(
    tokenizer: Any,
    examples: Sequence[OfficialExample],
    *,
    config: RescueConfig,
) -> dict[str, Tensor]:
    encoded = _encode_source_inputs(
        tokenizer,
        examples,
        max_source_length=config.max_source_length,
        truncation_mode=config.truncation_mode,
    )
    targets = tokenizer(
        text_target=[example.label for example in examples],
        padding=True,
        truncation=True,
        max_length=config.max_target_length,
        return_tensors="pt",
    )
    labels = targets["input_ids"].masked_fill(
        targets["input_ids"].eq(tokenizer.pad_token_id), -100
    )
    device = torch.device(config.device)
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "labels": labels.to(device),
    }


def _field_aware_components(example: OfficialExample) -> tuple[str, str, str] | None:
    """Return fixed head, flexible passage/paragraph, and protected suffix."""

    answer = "\nAnswer:"
    if not example.prompt.endswith(answer):
        raise ValueError("official prompt does not end in Answer:")
    body = example.prompt[: -len(answer)]
    if not body.endswith(example.sentence):
        raise ValueError("official prompt/body alignment failed")
    header = body[: -len(example.sentence)]
    if example.task_name == "BoolQA":
        question_marker = "question: "
        passage_marker = "\npassage: "
        if not example.sentence.startswith(question_marker) or passage_marker not in example.sentence:
            raise ValueError("BoolQA sentence fields are malformed")
        question, passage = example.sentence[len(question_marker) :].split(
            passage_marker, 1
        )
        return (
            header + question_marker + question + passage_marker,
            passage,
            answer,
        )
    if example.task_name == "MultiRC":
        paragraph_marker = "paragraph: "
        question_marker = "\nquestion: "
        candidate_marker = "\ncandidate answer: "
        if not example.sentence.startswith(paragraph_marker):
            raise ValueError("MultiRC sentence lacks paragraph field")
        paragraph_and_tail = example.sentence[len(paragraph_marker) :]
        paragraph, separator, question_and_candidate = paragraph_and_tail.rpartition(
            question_marker
        )
        if not separator:
            raise ValueError("MultiRC sentence lacks question field")
        question, separator, candidate = question_and_candidate.rpartition(candidate_marker)
        if not separator:
            raise ValueError("MultiRC sentence lacks candidate-answer field")
        return (
            header + paragraph_marker,
            paragraph,
            question_marker + question + candidate_marker + candidate + answer,
        )
    return None


def _field_aware_ids(
    tokenizer: Any, example: OfficialExample, max_source_length: int
) -> list[int]:
    components = _field_aware_components(example)
    if components is None:
        return tokenizer(
            example.prompt,
            truncation=True,
            max_length=max_source_length,
        )["input_ids"]
    head, flexible, suffix = components
    head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
    flexible_ids = tokenizer(flexible, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    special = int(tokenizer.num_special_tokens_to_add(pair=False))
    available = max_source_length - len(head_ids) - len(suffix_ids) - special
    if available < 0:
        raise ValueError(
            f"protected {example.task_name} fields require "
            f"{len(head_ids) + len(suffix_ids) + special} tokens, exceeding "
            f"max_source_length={max_source_length}"
        )
    content = [*head_ids, *flexible_ids[:available], *suffix_ids]
    input_ids = tokenizer.build_inputs_with_special_tokens(content)
    if len(input_ids) > max_source_length:
        raise RuntimeError("field-aware token construction exceeded its budget")
    return [int(value) for value in input_ids]


def _encode_source_inputs(
    tokenizer: Any,
    examples: Sequence[OfficialExample],
    *,
    max_source_length: int,
    truncation_mode: str,
) -> dict[str, Tensor]:
    if truncation_mode == "official_right":
        encoded = tokenizer(
            [example.prompt for example in examples],
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
    if truncation_mode != "field_aware":
        raise ValueError(f"unsupported truncation_mode {truncation_mode!r}")
    ids = [_field_aware_ids(tokenizer, example, max_source_length) for example in examples]
    width = max(len(row) for row in ids)
    input_ids = torch.full(
        (len(ids), width), int(tokenizer.pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros((len(ids), width), dtype=torch.long)
    for row_index, row in enumerate(ids):
        input_ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask[row_index, : len(row)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def source_token_length_report(
    tokenizer: Any,
    examples: Sequence[OfficialExample],
    *,
    actual_max_source_length: int,
    truncation_mode: str,
) -> dict[str, Any]:
    """Measure untruncated prompt lengths and would-truncate rates."""

    if not examples:
        raise ValueError("token-length report requires non-empty examples")
    lengths: list[int] = []
    for batch in _chunks(examples, 256):
        encoded = tokenizer(
            [example.prompt for example in batch],
            add_special_tokens=True,
            truncation=False,
            return_length=True,
        )
        batch_lengths = encoded.get("length")
        if batch_lengths is None:
            batch_lengths = [len(ids) for ids in encoded["input_ids"]]
        lengths.extend(int(value) for value in batch_lengths)
    values = np.asarray(lengths, dtype=np.int64)
    cutoffs = sorted({128, 256, 512, int(actual_max_source_length)})
    applicable = 0
    official_suffix_preserved = 0
    field_flexible_truncated = 0
    field_protected_overflow = 0
    for example in examples:
        components = _field_aware_components(example)
        if components is None:
            continue
        applicable += 1
        head, flexible, suffix = components
        suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
        official_ids = tokenizer(
            example.prompt,
            truncation=True,
            max_length=actual_max_source_length,
        )["input_ids"]
        if any(
            official_ids[index : index + len(suffix_ids)] == suffix_ids
            for index in range(max(0, len(official_ids) - len(suffix_ids) + 1))
        ):
            official_suffix_preserved += 1
        head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
        flexible_ids = tokenizer(flexible, add_special_tokens=False)["input_ids"]
        special = int(tokenizer.num_special_tokens_to_add(pair=False))
        available = actual_max_source_length - len(head_ids) - len(suffix_ids) - special
        if available < 0:
            field_protected_overflow += 1
        elif len(flexible_ids) > available:
            field_flexible_truncated += 1
    return {
        "example_count": int(values.size),
        "untruncated_tokens": {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": int(values.max()),
        },
        "cutoffs": {
            str(cutoff): {
                "would_truncate_count": int((values > cutoff).sum()),
                "would_truncate_rate": float((values > cutoff).mean()),
            }
            for cutoff in cutoffs
        },
        "actual_max_source_length": int(actual_max_source_length),
        "selected_truncation_mode": truncation_mode,
        "mode_comparison_at_actual_max": {
            "field_aware_applicable_count": applicable,
            "official_right_protected_suffix_preserved_count": official_suffix_preserved,
            "official_right_protected_suffix_preserved_rate": (
                float(official_suffix_preserved / applicable) if applicable else None
            ),
            "field_aware_flexible_field_truncated_count": field_flexible_truncated,
            "field_aware_flexible_field_truncated_rate": (
                float(field_flexible_truncated / applicable) if applicable else None
            ),
            "field_aware_protected_field_overflow_count": field_protected_overflow,
        },
    }


def _label_constrained_predictions(
    bank: T5GlobalRankBank,
    tokenizer: Any,
    examples: Sequence[OfficialExample],
    *,
    config: RescueConfig,
) -> tuple[list[str], list[list[float]]]:
    """Choose the official label sequence with minimum conditional token NLL."""

    labels = TASK_BY_NAME[examples[0].task_name].labels
    prompts: list[str] = []
    targets: list[str] = []
    for example in examples:
        prompts.extend([example.prompt] * len(labels))
        targets.extend(labels)
    losses: list[float] = []
    device = torch.device(config.device)
    for start in range(0, len(prompts), config.batch_size):
        batch_targets = targets[start : start + config.batch_size]
        # A batch can straddle example/label rows.  Encode the exact repeated
        # source sequence rather than relying on prompt strings that would
        # bypass the field-aware policy.
        repeated_examples = [
            examples[pair_index // len(labels)]
            for pair_index in range(start, start + len(batch_targets))
        ]
        encoded = _encode_source_inputs(
            tokenizer,
            repeated_examples,
            max_source_length=config.max_source_length,
            truncation_mode=config.truncation_mode,
        )
        encoded_targets = tokenizer(
            text_target=batch_targets,
            padding=True,
            truncation=True,
            max_length=config.max_target_length,
            return_tensors="pt",
        )
        target_ids = encoded_targets["input_ids"]
        target_ids = target_ids.masked_fill(target_ids.eq(tokenizer.pad_token_id), -100)
        inputs = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
            "labels": target_ids.to(device),
        }
        output = bank(**inputs)
        nll = per_example_target_token_nll(output.logits, inputs["labels"])
        losses.extend(float(value) for value in nll.detach().cpu())
    matrix = np.asarray(losses, dtype=np.float64).reshape(len(examples), len(labels))
    chosen = matrix.argmin(axis=1).tolist()
    return [labels[int(index)] for index in chosen], matrix.tolist()


def free_generation_classification_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    labels: Sequence[str],
    natural_label_prior: Mapping[str, float],
) -> dict[str, Any]:
    if not predictions or len(predictions) != len(references):
        raise ValueError("predictions/references must be non-empty and aligned")
    if set(natural_label_prior) != set(labels) or not math.isclose(
        sum(float(value) for value in natural_label_prior.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("natural_label_prior must cover labels and sum to one")
    normalized_predictions = [normalize_answer(value) for value in predictions]
    normalized_labels = {label: normalize_answer(label) for label in labels}
    valid_labels = set(normalized_labels.values())
    per_class_recall: dict[str, float] = {}
    for label in labels:
        indices = [index for index, reference in enumerate(references) if reference == label]
        if not indices:
            raise ValueError("audit partition is missing an official class")
        per_class_recall[label] = float(
            sum(
                normalized_predictions[index] == normalized_labels[label]
                for index in indices
            )
            / len(indices)
        )
    balanced_em = float(np.mean(tuple(per_class_recall.values())))
    return {
        "natural_prior_em": float(
            sum(
                float(natural_label_prior[label]) * per_class_recall[label]
                for label in labels
            )
        ),
        "balanced_em": balanced_em,
        "per_class_recall": per_class_recall,
        "worst_class_recall": min(per_class_recall.values()),
        "free_generation_valid_label_rate": float(
            sum(value in valid_labels for value in normalized_predictions)
            / len(predictions)
        ),
    }


def evaluate_train_audit(
    bank: T5GlobalRankBank,
    tokenizer: Any,
    examples: Sequence[OfficialExample],
    *,
    config: RescueConfig,
    natural_label_prior: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if not examples or any(example.subset != "train" for example in examples):
        raise ValueError("audit evaluation requires a non-empty train-derived partition")
    was_training = bank.training
    bank.eval()
    predictions: list[str] = []
    device = torch.device(config.device)
    with torch.no_grad():
        for batch in _chunks(examples, config.batch_size):
            encoded = _encode_source_inputs(
                tokenizer,
                batch,
                max_source_length=config.max_source_length,
                truncation_mode=config.truncation_mode,
            )
            generated = bank.generate(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
                max_new_tokens=config.max_target_length,
                num_beams=1,
                do_sample=False,
            )
            predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        constrained_predictions, constrained_nll = _label_constrained_predictions(
            bank, tokenizer, examples, config=config
        )
    bank.train(was_training)
    references = [example.label for example in examples]
    official_labels = tuple(TASK_BY_NAME[examples[0].task_name].labels)
    if natural_label_prior is None:
        empirical_counts = {
            label: sum(example.label == label for example in examples)
            for label in official_labels
        }
        empirical_total = sum(empirical_counts.values())
        natural_label_prior = {
            label: empirical_counts[label] / empirical_total for label in official_labels
        }
    classification = free_generation_classification_metrics(
        predictions,
        references,
        labels=official_labels,
        natural_label_prior=natural_label_prior,
    )
    return {
        # Compatibility alias.  Audit sets are class-balanced by construction,
        # so legacy normalized_em has the balanced-EM meaning in this runner.
        "normalized_em": classification["balanced_em"],
        "normalized_em_semantics": "balanced_em on class-balanced train audit",
        **classification,
        "label_constrained_sequence_nll_accuracy": (
            normalized_em(constrained_predictions, references) / 100.0
        ),
        "example_ids": [example.example_id for example in examples],
        "predictions": predictions,
        "label_constrained_predictions": constrained_predictions,
        "label_sequence_mean_nll": constrained_nll,
        "references": references,
        "free_generation_empty_rate": float(
            sum(not value.strip() for value in predictions) / len(predictions)
        ),
        "natural_label_prior": {
            label: float(natural_label_prior[label]) for label in official_labels
        },
        "source": "train.json audit partition",
    }


def _build_model(config: RescueConfig, tokenizer: Any) -> T5GlobalRankBank:
    from transformers import AutoModelForSeq2SeqLM, T5Config, T5ForConditionalGeneration

    _set_seed(config.seed)
    if config.tiny_random_model:
        base_config = T5Config(
            vocab_size=len(tokenizer),
            d_model=32,
            d_kv=8,
            d_ff=64,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            dropout_rate=0.0,
            decoder_start_token_id=tokenizer.pad_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = T5ForConditionalGeneration(base_config)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_path, local_files_only=True
        )
    model.to(torch.device(config.device))
    return T5GlobalRankBank(
        model,
        max_rank=8,
        initial_rank=4,
        alpha=16.0,
        reference_rank=4,
        lora_dropout=0.1,
        require_t5_small=not config.tiny_random_model,
    )


def _adapter_fingerprint(bank: T5GlobalRankBank) -> str:
    digest = hashlib.sha256()
    for name, adapter in zip(bank.adapter_names, bank.adapters):
        digest.update(name.encode("utf-8"))
        digest.update(adapter.active_mask.detach().cpu().numpy().tobytes())
        for atom in adapter.atoms:
            digest.update(atom.a.detach().cpu().contiguous().numpy().tobytes())
            digest.update(atom.b.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sequence_objective_weights(
    loss_objective: str,
    labels: Sequence[str],
    *,
    natural_label_prior: Mapping[str, float],
    sampler_q: Mapping[str, float],
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Return per-row sequence weights, without batch re-normalization."""

    if loss_objective not in {"sequence_mean_balanced", "sequence_mean_prior"}:
        raise ValueError("sequence weights require a sequence_mean objective")
    if loss_objective == "sequence_mean_balanced":
        values = [1.0 for _ in labels]
    else:
        values = [
            float(natural_label_prior[label]) / float(sampler_q[label])
            for label in labels
        ]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("objective weights must be finite and positive")
    return torch.tensor(values, dtype=dtype, device=device)


def _new_resource_ledger(
    task_name: str, update_examples: Sequence[OfficialExample]
) -> dict[str, Any]:
    return {
        "gradient_examples_per_epoch": len(update_examples),
        "gradient_example_exposures": 0,
        "source_tokens": 0,
        "target_tokens": 0,
        "optimizer_steps": 0,
        "tune_audit_gradient_examples": 0,
        "confirm_audit_gradient_examples": 0,
        "per_class": {
            label: {
                "row_exposures": 0,
                "target_tokens": 0,
                "objective_weight_sum": 0.0,
            }
            for label in TASK_BY_NAME[task_name].labels
        },
    }


def _validate_completed_resource_ledger(
    ledger: Mapping[str, Any],
    candidate: CandidateSpec,
    update_examples: Sequence[OfficialExample],
    *,
    task_name: str,
    batch_size: int,
    natural_label_prior: Mapping[str, float],
) -> dict[str, Any]:
    """Fail closed if a completed trajectory has an incomplete resource ledger."""

    rows_per_epoch = len(update_examples)
    expected_exposures = rows_per_epoch * candidate.epochs
    expected_steps = math.ceil(rows_per_epoch / batch_size) * candidate.epochs
    expected_per_class = {
        label: sum(example.label == label for example in update_examples)
        * candidate.epochs
        for label in TASK_BY_NAME[task_name].labels
    }
    counts_per_epoch = {
        label: sum(example.label == label for example in update_examples)
        for label in TASK_BY_NAME[task_name].labels
    }
    sampler_total = sum(counts_per_epoch.values())
    expected_sampler_q = {
        label: count / sampler_total for label, count in counts_per_epoch.items()
    }
    recorded_counts = ledger.get("sampler_label_counts_per_epoch")
    if not isinstance(recorded_counts, Mapping) or {
        str(label): int(count) for label, count in recorded_counts.items()
    } != counts_per_epoch:
        raise RuntimeError("resource ledger sampler label counts are incomplete")
    recorded_q = ledger.get("sampler_q")
    recorded_prior = ledger.get("natural_label_prior")
    if (
        not isinstance(recorded_q, Mapping)
        or set(recorded_q) != set(expected_sampler_q)
        or not isinstance(recorded_prior, Mapping)
        or set(recorded_prior) != set(natural_label_prior)
    ):
        raise RuntimeError("resource ledger sampler/prior schema is incomplete")
    for label in counts_per_epoch:
        if not math.isclose(
            float(recorded_q[label]),
            expected_sampler_q[label],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(recorded_prior[label]),
            float(natural_label_prior[label]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError("resource ledger sampler/prior values drifted")
    if not math.isclose(
        sum(float(recorded_prior[label]) for label in recorded_prior),
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("resource ledger natural prior is not normalized")
    required_scalars = {
        "gradient_examples_per_epoch": rows_per_epoch,
        "gradient_example_exposures": expected_exposures,
        "optimizer_steps": expected_steps,
        "tune_audit_gradient_examples": 0,
        "confirm_audit_gradient_examples": 0,
    }
    for key, expected in required_scalars.items():
        if int(ledger.get(key, -1)) != expected:
            raise RuntimeError(
                f"incomplete resource ledger for {candidate.candidate_id}: "
                f"{key}={ledger.get(key)!r}, expected={expected}"
            )
    per_class = ledger.get("per_class")
    if not isinstance(per_class, Mapping) or set(per_class) != set(expected_per_class):
        raise RuntimeError("resource ledger has an invalid per-class schema")
    for label, expected_rows in expected_per_class.items():
        row = per_class[label]
        if not isinstance(row, Mapping) or int(row.get("row_exposures", -1)) != expected_rows:
            raise RuntimeError(
                f"resource ledger row exposures are incomplete for class {label}"
            )
        target_tokens = int(row.get("target_tokens", -1))
        objective_weight_sum = float(row.get("objective_weight_sum", float("nan")))
        if target_tokens < expected_rows or not math.isfinite(objective_weight_sum):
            raise RuntimeError(
                f"resource ledger token/weight accounting is incomplete for class {label}"
            )
        expected_weight_sum = float(expected_rows)
        if candidate.loss_objective == "sequence_mean_prior":
            expected_weight_sum *= float(natural_label_prior[label]) / float(
                expected_sampler_q[label]
            )
        if not math.isclose(
            objective_weight_sum,
            expected_weight_sum,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                f"resource ledger objective mass is incorrect for class {label}"
            )
    per_class_target_tokens = sum(
        int(per_class[label]["target_tokens"]) for label in expected_per_class
    )
    if int(ledger.get("target_tokens", -1)) != per_class_target_tokens:
        raise RuntimeError("resource ledger target-token totals do not reconcile")
    if int(ledger.get("source_tokens", -1)) < expected_exposures:
        raise RuntimeError("resource ledger source-token total is incomplete")
    return {
        "status": "pass",
        "expected": {
            **required_scalars,
            "per_class_row_exposures": expected_per_class,
            "sampler_label_counts_per_epoch": counts_per_epoch,
            "sampler_q": expected_sampler_q,
            "natural_label_prior": {
                label: float(natural_label_prior[label])
                for label in counts_per_epoch
            },
        },
        "observed": {
            key: int(ledger[key]) for key in required_scalars
        },
        "target_token_totals_reconciled": True,
        "source_token_total_at_least_row_exposures": True,
    }


def train_update_segment(
    bank: T5GlobalRankBank,
    tokenizer: Any,
    update_examples: Sequence[OfficialExample],
    candidate: CandidateSpec,
    optimizer: torch.optim.Optimizer,
    ledger: dict[str, Any],
    *,
    task_name: str,
    config: RescueConfig,
    start_epoch: int,
    stop_epoch: int,
    natural_label_prior: Mapping[str, float],
) -> list[float]:
    """Advance one shared LR/cap trajectory over a half-open epoch range."""

    if not update_examples or any(example.subset != "train" for example in update_examples):
        raise ValueError("gradient examples must be a non-empty train-derived update set")
    if not 0 <= start_epoch < stop_epoch <= candidate.epochs:
        raise ValueError("invalid trajectory epoch range")
    epoch_means: list[float] = []
    update_counts = {
        label: sum(example.label == label for example in update_examples)
        for label in TASK_BY_NAME[task_name].labels
    }
    update_total = sum(update_counts.values())
    sampler_q = {label: count / update_total for label, count in update_counts.items()}
    if candidate.update_sampler == "balanced" and len(set(update_counts.values())) != 1:
        raise ValueError("balanced update sampler requires equal per-class rows")
    if set(natural_label_prior) != set(update_counts):
        raise ValueError("natural prior and update sampler labels disagree")
    ledger["sampler_label_counts_per_epoch"] = update_counts
    ledger["sampler_q"] = sampler_q
    ledger["natural_label_prior"] = {
        label: float(natural_label_prior[label]) for label in update_counts
    }
    bank.train()
    for epoch_index in range(start_epoch, stop_epoch):
        _set_seed(_stable_seed(config.seed, task_name, candidate.cap_per_class, "epoch", epoch_index))
        rng = np.random.Generator(
            np.random.PCG64(
                _stable_seed(config.seed, task_name, candidate.cap_per_class, "order", epoch_index)
            )
        )
        order = rng.permutation(len(update_examples)).tolist()
        ordered = tuple(update_examples[int(index)] for index in order)
        losses: list[float] = []
        for batch in _chunks(ordered, config.batch_size):
            encoded = _tokenize(tokenizer, batch, config=config)
            output = bank(**encoded)
            if candidate.loss_objective == "hf_token_ce":
                # Hugging Face Seq2SeqTrainer/T5 token-weighted reduction.
                loss = output.loss
                example_weights = torch.ones(len(batch), device=encoded["labels"].device)
            else:
                sequence_nll = per_example_target_token_nll(
                    output.logits, encoded["labels"]
                )
                if candidate.loss_objective == "sequence_mean_balanced":
                    example_weights = sequence_objective_weights(
                        candidate.loss_objective,
                        [example.label for example in batch],
                        natural_label_prior=natural_label_prior,
                        sampler_q=sampler_q,
                        dtype=sequence_nll.dtype,
                        device=sequence_nll.device,
                    )
                elif candidate.loss_objective == "sequence_mean_prior":
                    example_weights = sequence_objective_weights(
                        candidate.loss_objective,
                        [example.label for example in batch],
                        natural_label_prior=natural_label_prior,
                        sampler_q=sampler_q,
                        dtype=sequence_nll.dtype,
                        device=sequence_nll.device,
                    )
                else:  # pragma: no cover - CandidateSpec validates this.
                    raise AssertionError("unreachable loss objective")
                # Deliberately do not re-normalize within a batch: pi/q is an
                # importance weight under the exactly recorded sampler q.
                loss = torch.mean(example_weights * sequence_nll)
            if loss is None or loss.ndim != 0 or not bool(torch.isfinite(loss.detach())):
                raise RuntimeError("model did not return a finite configured loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tuple(bank.active_atom_parameters()), 1.0)
            optimizer.step()
            bank.assert_exact_budget()
            losses.append(float(loss.detach().cpu()))
            ledger["gradient_example_exposures"] += len(batch)
            ledger["source_tokens"] += int(encoded["attention_mask"].sum())
            ledger["target_tokens"] += int(encoded["labels"].ne(-100).sum())
            ledger["optimizer_steps"] += 1
            token_counts = encoded["labels"].ne(-100).sum(dim=1).detach().cpu().tolist()
            weight_values = example_weights.detach().cpu().tolist()
            for example, target_tokens, weight in zip(batch, token_counts, weight_values):
                class_row = ledger["per_class"][example.label]
                class_row["row_exposures"] += 1
                class_row["target_tokens"] += int(target_tokens)
                class_row["objective_weight_sum"] += float(weight)
        epoch_means.append(float(np.mean(losses)))
    return epoch_means


def _split_manifest(split: RescueSplit) -> dict[str, Any]:
    def ledger(examples: Sequence[OfficialExample]) -> dict[str, Any]:
        ids = [example.example_id for example in examples]
        encoded = json.dumps(ids, sort_keys=False, separators=(",", ":")).encode("utf-8")
        counts: dict[str, int] = {}
        for example in examples:
            counts[example.label] = counts.get(example.label, 0) + 1
        return {
            "count": len(examples),
            "per_class": counts,
            "example_ids": ids,
            "ordered_ids_sha256": hashlib.sha256(encoded).hexdigest(),
            "source_subset": "train",
        }

    manifest = {
        "task": split.task_name,
        "update_pool": ledger(split.update_pool),
        "tune_audit": ledger(split.tune_audit),
        "confirm_audit": ledger(split.confirm_audit),
        "excluded_train_examples": split.excluded_count,
        "pairwise_disjoint": True,
    }
    # Do not change legacy row manifests: independent confirmations of already
    # persisted selections verify their exact file hash.
    if split.split_unit == "semantic_group":
        if split.group_key_by_example_id is None:
            raise AssertionError("semantic split is missing group keys")
        partitions = (split.update_pool, split.tune_audit, split.confirm_audit)
        group_sets = [
            {
                split.group_key_by_example_id[example.example_id]
                for example in partition
            }
            for partition in partitions
        ]
        overlap_counts = {
            "update_tune": len(group_sets[0] & group_sets[1]),
            "update_confirm": len(group_sets[0] & group_sets[2]),
            "tune_confirm": len(group_sets[1] & group_sets[2]),
        }
        if any(overlap_counts.values()):
            raise AssertionError(f"semantic-group overlap detected: {overlap_counts}")
        touched_groups = set(split.prior_touched_group_keys)
        # Reusing old development groups for new tune/update is authorized;
        # only confirm must remain pristine.
        prior_touched_overlap = {
            name: len(group_set & touched_groups)
            for name, group_set in zip(
                ("update_pool", "tune_audit", "confirm_audit"), group_sets
            )
        }
        if prior_touched_overlap["confirm_audit"]:
            raise AssertionError(
                f"prior-touched groups entered fresh confirm: {prior_touched_overlap}"
            )
        requested_ids_encoded = json.dumps(
            list(split.prior_touched_requested_example_ids), separators=(",", ":")
        ).encode("utf-8")
        expanded_ids_encoded = json.dumps(
            list(split.prior_touched_example_ids), separators=(",", ":")
        ).encode("utf-8")
        excluded_groups_encoded = json.dumps(
            list(split.prior_touched_group_keys), separators=(",", ":")
        ).encode("utf-8")
        manifest.update(
            {
                "split_unit": split.split_unit,
                "group_key_version": split.group_key_version,
                "group_counts": {
                    name: len(group_set)
                    for name, group_set in zip(
                        ("update_pool", "tune_audit", "confirm_audit"),
                        group_sets,
                    )
                },
                "semantic_group_pairwise_overlap_counts": overlap_counts,
                "semantic_group_pairwise_disjoint": True,
                "fresh_confirm_against_prior_splits": {
                    "sources": [dict(item) for item in split.freshness_sources],
                    "requested_example_ids": list(
                        split.prior_touched_requested_example_ids
                    ),
                    "requested_example_id_count": len(
                        split.prior_touched_requested_example_ids
                    ),
                    "requested_example_ids_sha256": hashlib.sha256(
                        requested_ids_encoded
                    ).hexdigest(),
                    "expanded_prior_touched_example_ids": list(
                        split.prior_touched_example_ids
                    ),
                    "expanded_prior_touched_row_count": len(
                        split.prior_touched_example_ids
                    ),
                    "expanded_prior_touched_example_ids_sha256": hashlib.sha256(
                        expanded_ids_encoded
                    ).hexdigest(),
                    "prior_touched_group_keys": list(split.prior_touched_group_keys),
                    "prior_touched_group_count": len(split.prior_touched_group_keys),
                    "prior_touched_group_keys_sha256": hashlib.sha256(
                        excluded_groups_encoded
                    ).hexdigest(),
                    "partition_overlap_with_prior_touched_groups": prior_touched_overlap,
                    "confirm_vs_prior_touched_group_overlap": 0,
                },
            }
        )
        if (
            split.selection_algorithm_version
            == SEMANTIC_SELECTION_ALGORITHM_VERSION
        ):
            group_population_sizes: dict[str, int] = {}
            for group_key in split.group_key_by_example_id.values():
                group_population_sizes[group_key] = (
                    group_population_sizes.get(group_key, 0) + 1
                )

            def group_size_statistics(group_keys: Iterable[str]) -> dict[str, Any]:
                sizes = sorted(group_population_sizes[key] for key in set(group_keys))
                histogram: dict[str, int] = {}
                for size in sizes:
                    key = str(size)
                    histogram[key] = histogram.get(key, 0) + 1
                row_count = sum(sizes)
                squared_mass = sum(size * size for size in sizes)
                return {
                    "row_count": row_count,
                    "group_count": len(sizes),
                    "effective_group_count": (
                        (row_count * row_count) / squared_mass
                        if squared_mass
                        else 0.0
                    ),
                    "group_size_histogram": histogram,
                    "min_group_size": min(sizes) if sizes else None,
                    "max_group_size": max(sizes) if sizes else None,
                    "mean_group_size": (
                        row_count / len(sizes) if sizes else 0.0
                    ),
                }

            partition_names = ("update_pool", "tune_audit", "confirm_audit")
            for name, group_set in zip(partition_names, group_sets):
                statistics = group_size_statistics(group_set)
                if statistics["row_count"] != manifest[name]["count"]:
                    raise AssertionError(
                        f"{name} contains a partial semantic group"
                    )
                manifest[name].update(
                    {
                        "group_size_histogram": statistics[
                            "group_size_histogram"
                        ],
                        "effective_group_count": statistics[
                            "effective_group_count"
                        ],
                    }
                )
            manifest["semantic_group_sampling"] = {
                "selection_algorithm_version": split.selection_algorithm_version,
                "priority": "deterministic SHA-256 order independent of group size",
                "quota_solver": "bounded exact subset-sum dynamic programming",
                "corpus": group_size_statistics(group_population_sizes),
                "partitions": {
                    name: group_size_statistics(group_set)
                    for name, group_set in zip(partition_names, group_sets)
                },
            }
    return manifest


def _release_model(bank: T5GlobalRankBank) -> None:
    del bank
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _canonical_payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json_payload_sha256(payload: Any) -> str:
    """SHA-256 of the exact pretty JSON bytes emitted by :func:`_atomic_json`."""

    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    # _atomic_json opens a text stream with platform newline translation.
    if os.linesep != "\n":
        rendered = rendered.replace("\n", os.linesep)
    encoded = rendered.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_matched_resource_ledgers(
    selected_ledger: Mapping[str, Any], control_ledger: Mapping[str, Any]
) -> dict[str, Any]:
    exact_scalar_fields = (
        "gradient_examples_per_epoch",
        "gradient_example_exposures",
        "source_tokens",
        "target_tokens",
        "optimizer_steps",
        "tune_audit_gradient_examples",
        "confirm_audit_gradient_examples",
    )
    for field in exact_scalar_fields:
        if int(selected_ledger.get(field, -1)) != int(
            control_ledger.get(field, -2)
        ):
            raise RuntimeError(f"matched control resource mismatch: {field}")
    selected_per_class = selected_ledger.get("per_class")
    control_per_class = control_ledger.get("per_class")
    if (
        not isinstance(selected_per_class, Mapping)
        or not isinstance(control_per_class, Mapping)
        or set(selected_per_class) != set(control_per_class)
    ):
        raise RuntimeError("matched control per-class resource schema mismatch")
    per_class_exact: dict[str, dict[str, int]] = {}
    for label in selected_per_class:
        selected_row = selected_per_class[label]
        control_row = control_per_class[label]
        if not isinstance(selected_row, Mapping) or not isinstance(
            control_row, Mapping
        ):
            raise RuntimeError("matched control per-class resource row is invalid")
        per_class_exact[label] = {}
        for field in ("row_exposures", "target_tokens"):
            selected_value = int(selected_row.get(field, -1))
            if selected_value != int(control_row.get(field, -2)):
                raise RuntimeError(
                    f"matched control resource mismatch: {label}.{field}"
                )
            per_class_exact[label][field] = selected_value
    return {
        "status": "pass",
        "exact_scalar_fields": {
            field: int(selected_ledger[field]) for field in exact_scalar_fields
        },
        "exact_per_class_fields": per_class_exact,
        "objective_weight_sum_excluded_because_objectives_differ": True,
    }


def _train_post_selection_hf_control(
    task_name: str,
    update_examples: Sequence[OfficialExample],
    selected: CandidateMeasurement,
    tokenizer: Any,
    *,
    config: RescueConfig,
    task_dir: Path,
    expected_fingerprint: str,
    train_label_distribution: Mapping[str, Any],
    selected_registry_entry: Mapping[str, Any],
    selection_lock_path: Path,
    selection_lock_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train one matched token-CE control after the selection is durable.

    This helper receives update rows, not the full split, so neither tune nor
    confirmation examples are in its data scope.  The durable selection lock is
    verified before and after training.  The returned registry entry is
    explicitly ineligible for selection.
    """

    if selected.candidate.loss_objective == CONTROL_OBJECTIVE:
        raise ValueError("a selected token-CE candidate does not need a new control")
    if not selection_lock_path.is_file():
        raise RuntimeError("post-selection control requires a durable selection lock")
    if _sha256_file(selection_lock_path) != selection_lock_sha256:
        raise RuntimeError("selection lock changed before control training")
    with selection_lock_path.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    if (
        lock.get("selected_candidate_id") != selected.candidate.candidate_id
        or lock.get("selected_checkpoint_sha256")
        != _sha256_file(Path(selected.checkpoint))
        or lock.get("confirm_audit_accessed") is not False
    ):
        raise RuntimeError("selection lock does not match the selected checkpoint")
    if str(selected_registry_entry.get("candidate_id")) != selected.candidate.candidate_id:
        raise RuntimeError("selected registry entry does not match the selection lock")
    selected_metrics_path = Path(
        str(selected_registry_entry.get("metrics", ""))
    ).expanduser().resolve()
    if (
        not selected_metrics_path.is_file()
        or _sha256_file(selected_metrics_path)
        != str(selected_registry_entry.get("metrics_sha256"))
    ):
        raise RuntimeError("selected metrics registry entry is not content-addressed")
    with selected_metrics_path.open("r", encoding="utf-8") as handle:
        selected_metrics = json.load(handle)
    selected_ledger = selected_metrics.get("training", {}).get("resource_ledger")
    if not isinstance(selected_ledger, Mapping):
        raise RuntimeError("selected candidate is missing its resource ledger")

    control = CandidateSpec(
        learning_rate=selected.candidate.learning_rate,
        epochs=selected.candidate.epochs,
        cap_per_class=selected.candidate.cap_per_class,
        loss_objective=CONTROL_OBJECTIVE,
        update_sampler=selected.candidate.update_sampler,
    )
    bank = _build_model(config, tokenizer)
    fingerprint = _adapter_fingerprint(bank)
    if fingerprint != expected_fingerprint:
        raise RuntimeError("matched-control adapter initialization drifted")
    optimizer = torch.optim.AdamW(
        tuple(bank.all_atom_parameters()),
        lr=control.learning_rate,
        weight_decay=0.0,
    )
    ledger = _new_resource_ledger(task_name, update_examples)
    control_started = time.perf_counter()
    epoch_means = train_update_segment(
        bank,
        tokenizer,
        update_examples,
        control,
        optimizer,
        ledger,
        task_name=task_name,
        config=config,
        start_epoch=0,
        stop_epoch=control.epochs,
        natural_label_prior=train_label_distribution["natural_label_prior"],
    )
    resource_validation = _validate_completed_resource_ledger(
        ledger,
        control,
        update_examples,
        task_name=task_name,
        batch_size=config.batch_size,
        natural_label_prior=train_label_distribution["natural_label_prior"],
    )
    matched_resource_validation = _validate_matched_resource_ledgers(
        selected_ledger, ledger
    )
    if _sha256_file(selection_lock_path) != selection_lock_sha256:
        raise RuntimeError("selection lock changed during control training")

    candidate_dir = task_dir / "candidates" / control.candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = candidate_dir / "adapter.pt"
    _atomic_torch_save(checkpoint, bank.compact_state_dict())
    checkpoint_sha = _sha256_file(checkpoint)
    artifact = {
        "candidate_index": None,
        "candidate": asdict(control),
        "candidate_id": control.candidate_id,
        "registry_role": "post_selection_matched_hf_token_ce_control",
        "selection_eligible": False,
        "selection_partition": "not accessed; selection already locked",
        "tune_audit_accessed": False,
        "confirm_audit_accessed": False,
        "trained_after_selection_lock": True,
        "selection_lock": {
            "path": str(selection_lock_path.resolve()),
            "sha256": selection_lock_sha256,
            "selected_candidate_id": selected.candidate.candidate_id,
        },
        "matching_fields": {
            "learning_rate": control.learning_rate,
            "epochs": control.epochs,
            "cap_per_class": control.cap_per_class,
            "update_sampler": control.update_sampler,
        },
        "training": {
            "optimizer": "AdamW",
            "training_loss_reduction": CONTROL_OBJECTIVE,
            "loss_objective": CONTROL_OBJECTIVE,
            "update_sampler": control.update_sampler,
            "pinned_train_label_distribution": dict(train_label_distribution),
            "weight_decay": 0.0,
            "learning_rate": control.learning_rate,
            "checkpoint_epochs": [control.epochs],
            "completed_epochs": control.epochs,
            "epoch_mean_nll": epoch_means,
            "resource_ledger": ledger,
            "resource_ledger_validation": resource_validation,
            "matched_selected_resource_validation": matched_resource_validation,
        },
        "payload": bank.payload_audit().to_dict(),
        "initial_adapter_fingerprint": fingerprint,
        "checkpoint": {
            "path": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_sha,
        },
        "wall_seconds": time.perf_counter() - control_started,
    }
    metrics_path = candidate_dir / "metrics.json"
    _atomic_json(metrics_path, artifact)
    registry_entry = {
        "candidate_id": control.candidate_id,
        "candidate": asdict(control),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "metrics": str(metrics_path.resolve()),
        "metrics_sha256": _sha256_file(metrics_path),
        "registry_role": "post_selection_matched_hf_token_ce_control",
        "selection_eligible": False,
        "trained_after_selection_lock": True,
        "confirm_audit_accessed": False,
        "selection_lock_sha256": selection_lock_sha256,
        "resource_ledger_validation": resource_validation,
        "matched_selected_resource_validation": matched_resource_validation,
    }
    bundle_entry = {
        "candidate_id": control.candidate_id,
        "candidate": asdict(control),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "same_as_selected": False,
        "selection_eligible": False,
        "trained_after_selection_lock": True,
        "confirm_audit_accessed": False,
        "selection_lock_sha256": selection_lock_sha256,
        "matched_selected_resource_validation": matched_resource_validation,
    }
    _release_model(bank)
    return registry_entry, bundle_entry


def _candidate_spec_from_artifact(
    payload: Any, *, context: str
) -> CandidateSpec:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} is missing candidate hyperparameters")
    expected_fields = {
        "learning_rate",
        "epochs",
        "cap_per_class",
        "loss_objective",
        "update_sampler",
    }
    if set(payload) != expected_fields:
        raise ValueError(f"{context} candidate schema is invalid")
    try:
        learning_rate = payload["learning_rate"]
        epochs = payload["epochs"]
        cap_per_class = payload["cap_per_class"]
        loss_objective = payload["loss_objective"]
        update_sampler = payload["update_sampler"]
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or isinstance(epochs, bool)
            or not isinstance(epochs, int)
            or isinstance(cap_per_class, bool)
            or not isinstance(cap_per_class, int)
            or not isinstance(loss_objective, str)
            or not isinstance(update_sampler, str)
        ):
            raise TypeError("candidate field types are invalid")
        return CandidateSpec(
            learning_rate=float(learning_rate),
            epochs=epochs,
            cap_per_class=cap_per_class,
            loss_objective=loss_objective,
            update_sampler=update_sampler,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} candidate hyperparameters are invalid") from error


def _validate_matched_control_bundle(
    selection: Mapping[str, Any],
    confirmation_bundle: Mapping[str, Any],
    *,
    selected_checkpoint: Path,
    selected_checkpoint_sha256: str,
) -> tuple[Mapping[str, Any] | None, Path | None, str | None]:
    """Fail closed on control identity, matching, registry, and lock tampering."""

    selected_id = str(selection.get("selected_candidate_id", ""))
    selected_payload = selection.get("selected")
    if not isinstance(selected_payload, Mapping):
        raise ValueError("selection is missing selected candidate metadata")
    selected_spec = _candidate_spec_from_artifact(
        selected_payload.get("candidate"), context="selected"
    )
    if selected_spec.candidate_id != selected_id:
        raise ValueError("selected candidate ID is not content-addressed correctly")
    locked_selected = confirmation_bundle.get("selected")
    if not isinstance(locked_selected, Mapping) or (
        str(locked_selected.get("candidate_id")) != selected_id
        or str(locked_selected.get("checkpoint_sha256"))
        != selected_checkpoint_sha256
        or Path(str(locked_selected.get("checkpoint", ""))).expanduser().resolve()
        != selected_checkpoint
    ):
        raise ValueError("v2 confirmation bundle selected checkpoint is not locked")

    matched = confirmation_bundle.get("matched_hf_token_ce")
    if matched is None:
        return None, None, None
    if not isinstance(matched, Mapping):
        raise ValueError("matched hf_token_ce bundle entry is invalid")
    control_id = str(matched.get("candidate_id", ""))
    registry = selection.get("candidate_registry")
    if not isinstance(registry, list):
        raise ValueError("selection candidate registry is missing")
    registry_matches = [
        item
        for item in registry
        if isinstance(item, Mapping) and str(item.get("candidate_id")) == control_id
    ]
    if len(registry_matches) != 1:
        raise ValueError("matched hf_token_ce must have one unique registry entry")
    registry_entry = registry_matches[0]
    registry_spec = _candidate_spec_from_artifact(
        registry_entry.get("candidate"), context="matched-control registry"
    )
    if "candidate" in matched:
        control_spec = _candidate_spec_from_artifact(
            matched.get("candidate"), context="matched hf_token_ce"
        )
        if asdict(control_spec) != asdict(registry_spec):
            raise ValueError("matched hf_token_ce candidate disagrees with registry")
    else:
        # Older v2 bundles omitted this redundant payload; their registry is
        # still the authoritative content-addressed source.
        control_spec = registry_spec
    if control_spec.candidate_id != control_id:
        raise ValueError("matched hf_token_ce candidate ID is not content-addressed")
    if control_spec.loss_objective != CONTROL_OBJECTIVE:
        raise ValueError("matched control objective is not hf_token_ce")
    for field in ("learning_rate", "epochs", "cap_per_class", "update_sampler"):
        if getattr(control_spec, field) != getattr(selected_spec, field):
            raise ValueError(f"matched hf_token_ce has mismatched {field}")

    control_checkpoint = Path(
        str(matched.get("checkpoint", ""))
    ).expanduser().resolve()
    if not control_checkpoint.is_file():
        raise FileNotFoundError(
            f"matched hf_token_ce checkpoint is missing: {control_checkpoint}"
        )
    control_sha = _sha256_file(control_checkpoint)
    if control_sha != str(matched.get("checkpoint_sha256")):
        raise ValueError("matched hf_token_ce checkpoint SHA-256 mismatch")
    if (
        Path(str(registry_entry.get("checkpoint", ""))).expanduser().resolve()
        != control_checkpoint
        or str(registry_entry.get("checkpoint_sha256")) != control_sha
    ):
        raise ValueError("matched hf_token_ce registry checkpoint is inconsistent")
    same_as_selected = bool(matched.get("same_as_selected"))
    if same_as_selected:
        if (
            control_id != selected_id
            or control_checkpoint != selected_checkpoint
            or control_sha != selected_checkpoint_sha256
            or selected_spec.loss_objective != CONTROL_OBJECTIVE
        ):
            raise ValueError("same_as_selected control identity is forged")
    elif control_id == selected_id or control_checkpoint == selected_checkpoint:
        raise ValueError("distinct matched control aliases the selected checkpoint")

    control_phase = confirmation_bundle.get("control_training_phase")
    if control_phase not in {
        None,
        "post_selection_lock_pre_confirmation",
        "selection_grid_or_selected",
        "not_available",
    }:
        raise ValueError("confirmation bundle has an invalid control-training phase")
    if control_phase == "not_available":
        raise ValueError("non-null matched control is marked unavailable")
    post_selection_control = (
        control_phase == "post_selection_lock_pre_confirmation"
        or bool(matched.get("trained_after_selection_lock"))
    )
    if post_selection_control:
        if (
            matched.get("selection_eligible") is not False
            or matched.get("trained_after_selection_lock") is not True
            or matched.get("confirm_audit_accessed") is not False
            or registry_entry.get("registry_role")
            != "post_selection_matched_hf_token_ce_control"
            or registry_entry.get("selection_eligible") is not False
            or registry_entry.get("trained_after_selection_lock") is not True
            or registry_entry.get("confirm_audit_accessed") is not False
        ):
            raise ValueError("post-selection control registry role is inconsistent")
        lock_metadata = selection.get("selection_lock")
        bundle_lock = confirmation_bundle.get("selection_lock")
        if not isinstance(lock_metadata, Mapping) or not isinstance(
            bundle_lock, Mapping
        ):
            raise ValueError("post-selection control is missing its selection lock")
        lock_path = Path(str(lock_metadata.get("path", ""))).expanduser().resolve()
        if not lock_path.is_file():
            raise FileNotFoundError(f"selection lock is missing: {lock_path}")
        lock_sha = _sha256_file(lock_path)
        claimed_lock_shas = {
            str(lock_metadata.get("sha256", "")),
            str(bundle_lock.get("sha256", "")),
            str(matched.get("selection_lock_sha256", "")),
            str(registry_entry.get("selection_lock_sha256", "")),
        }
        if claimed_lock_shas != {lock_sha}:
            raise ValueError("post-selection control selection-lock SHA is inconsistent")
        with lock_path.open("r", encoding="utf-8") as handle:
            lock = json.load(handle)
        if (
            lock.get("selected_candidate_id") != selected_id
            or lock.get("selected_checkpoint_sha256")
            != selected_checkpoint_sha256
            or lock.get("confirm_audit_accessed") is not False
        ):
            raise ValueError("selection lock does not bind the selected checkpoint")
        eligible_registry = sorted(
            (
                dict(item)
                for item in registry
                if isinstance(item, Mapping)
                and bool(item.get("selection_eligible", True))
            ),
            key=lambda item: item["candidate_id"],
        )
        if (
            lock.get("selection_candidate_ids")
            != sorted(item["candidate_id"] for item in eligible_registry)
            or lock.get("selection_candidate_registry_sha256")
            != _canonical_payload_sha256(eligible_registry)
        ):
            raise ValueError("selection lock does not bind the eligible registry")
    elif control_phase == "selection_grid_or_selected":
        # New artifacts make the role explicit.  Older v2 registries omitted
        # these additive fields, so their absence remains compatible.
        if registry_entry.get("registry_role") not in (None, "selection_candidate"):
            raise ValueError("grid control has an invalid registry role")
        if registry_entry.get("selection_eligible") not in (None, True):
            raise ValueError("grid control is incorrectly marked selection-ineligible")
        if matched.get("selection_eligible") not in (None, True):
            raise ValueError("grid control bundle has an invalid selection role")
    return matched, control_checkpoint, control_sha


def _preflight_selection_confirmation(
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all artifact-only invariants before any confirm-row inspection."""

    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("selection artifact is missing its selected candidate")
    checkpoint = Path(str(selected.get("checkpoint", ""))).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"selected adapter checkpoint is missing: {checkpoint}")
    actual_sha = _sha256_file(checkpoint)
    expected_sha = selected.get("checkpoint_sha256")
    if expected_sha is not None and actual_sha != expected_sha:
        raise ValueError("selected adapter checkpoint SHA-256 mismatch")
    artifact_format = selection.get("format")
    if artifact_format == FORMAT_VERSION:
        if selection.get("gate_version") != "v2":
            raise ValueError("v2 selection is missing its non-downgradable v2 gate")
        is_v2 = True
    elif artifact_format == LEGACY_FORMAT_VERSION:
        is_v2 = False
    else:
        raise ValueError("selection artifact has an unsupported format")
    confirmation_bundle = selection.get("confirmation_bundle")
    matched_control: Mapping[str, Any] | None = None
    control_checkpoint: Path | None = None
    control_sha: str | None = None
    if is_v2 and isinstance(confirmation_bundle, Mapping):
        matched_control, control_checkpoint, control_sha = (
            _validate_matched_control_bundle(
                selection,
                confirmation_bundle,
                selected_checkpoint=checkpoint,
                selected_checkpoint_sha256=actual_sha,
            )
        )
    elif is_v2 and confirmation_bundle is not None:
        raise ValueError("v2 confirmation bundle is invalid")
    return {
        "selected_checkpoint": checkpoint,
        "selected_checkpoint_sha256": actual_sha,
        "is_v2": is_v2,
        "confirmation_bundle": confirmation_bundle,
        "matched_control": matched_control,
        "control_checkpoint": control_checkpoint,
        "control_checkpoint_sha256": control_sha,
    }


def evaluate_selected_confirmation(
    selection: Mapping[str, Any],
    split: RescueSplit,
    tokenizer: Any,
    *,
    config: RescueConfig,
    selection_file_preexisted: bool,
) -> dict[str, Any]:
    """Evaluate confirm-audit once for an already durable selection."""

    if selection.get("task") != split.task_name:
        raise ValueError("selection task does not match the reconstructed split")
    preflight = _preflight_selection_confirmation(selection)
    checkpoint = preflight["selected_checkpoint"]
    actual_sha = preflight["selected_checkpoint_sha256"]
    is_v2 = bool(preflight["is_v2"])
    confirmation_bundle = preflight["confirmation_bundle"]
    matched_control = preflight["matched_control"]
    control_checkpoint = preflight["control_checkpoint"]
    control_sha = preflight["control_checkpoint_sha256"]
    train_distribution = selection.get("pinned_train_label_distribution")
    if is_v2 and not isinstance(train_distribution, Mapping):
        raise ValueError("v2 selection is missing pinned train label distribution")
    natural_prior = (
        train_distribution["natural_label_prior"]
        if isinstance(train_distribution, Mapping)
        else None
    )

    confirm_bank = _build_model(config, tokenizer)
    expected_fingerprint = selection.get("model_protocol", {}).get(
        "initial_adapter_fingerprint"
    )
    if expected_fingerprint is not None and (
        _adapter_fingerprint(confirm_bank) != expected_fingerprint
    ):
        raise ValueError("confirmation adapter initialization fingerprint mismatch")
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    confirm_bank.load_compact_state_dict(checkpoint_state)
    base_confirm_bank = _build_model(config, tokenizer)
    base_confirm = evaluate_train_audit(
        base_confirm_bank,
        tokenizer,
        split.confirm_audit,
        config=config,
        natural_label_prior=natural_prior,
    )
    _release_model(base_confirm_bank)
    post_confirm = evaluate_train_audit(
        confirm_bank,
        tokenizer,
        split.confirm_audit,
        config=config,
        natural_label_prior=natural_prior,
    )
    confirm_gain = float(post_confirm["normalized_em"] - base_confirm["normalized_em"])
    natural_confirm_gain = float(
        post_confirm["natural_prior_em"] - base_confirm["natural_prior_em"]
    )
    per_class_deltas = {
        label: float(post_confirm["per_class_recall"][label])
        - float(base_confirm["per_class_recall"][label])
        for label in base_confirm["per_class_recall"]
    }
    min_per_class_delta = min(per_class_deltas.values())
    v2_gate_pass = (
        _strictly_above(natural_confirm_gain, config.min_rescue_gain)
        and float(post_confirm["free_generation_valid_label_rate"]) >= 0.99
        and _at_least(min_per_class_delta, -0.05)
    )
    matched_control_confirmation: dict[str, Any] | None = None
    if is_v2 and isinstance(confirmation_bundle, Mapping):
        if matched_control is not None:
            assert control_checkpoint is not None and control_sha is not None
            if bool(matched_control.get("same_as_selected")):
                control_post = post_confirm
                control_payload = confirm_bank.payload_audit().to_dict()
            else:
                control_bank = _build_model(config, tokenizer)
                control_bank.load_compact_state_dict(
                    torch.load(
                        control_checkpoint, map_location="cpu", weights_only=False
                    )
                )
                control_post = evaluate_train_audit(
                    control_bank,
                    tokenizer,
                    split.confirm_audit,
                    config=config,
                    natural_label_prior=natural_prior,
                )
                control_payload = control_bank.payload_audit().to_dict()
                _release_model(control_bank)
            control_deltas = {
                label: float(control_post["per_class_recall"][label])
                - float(base_confirm["per_class_recall"][label])
                for label in base_confirm["per_class_recall"]
            }
            control_natural_gain = float(
                control_post["natural_prior_em"]
                - base_confirm["natural_prior_em"]
            )
            matched_control_confirmation = {
                "candidate_id": matched_control["candidate_id"],
                "checkpoint": {
                    "path": str(control_checkpoint),
                    "sha256": control_sha,
                },
                "same_as_selected": bool(matched_control.get("same_as_selected")),
                "post_confirm": control_post,
                "confirm_natural_prior_acquisition": control_natural_gain,
                "confirm_balanced_acquisition": float(
                    control_post["balanced_em"] - base_confirm["balanced_em"]
                ),
                "confirm_per_class_recall_delta": control_deltas,
                "confirm_min_per_class_delta": min(control_deltas.values()),
                "v2_gate_pass": (
                    _strictly_above(control_natural_gain, config.min_rescue_gain)
                    and float(control_post["free_generation_valid_label_rate"])
                    >= 0.99
                    and _at_least(min(control_deltas.values()), -0.05)
                ),
                "payload": control_payload,
            }
    constrained_confirm_gain = float(
        post_confirm["label_constrained_sequence_nll_accuracy"]
        - base_confirm["label_constrained_sequence_nll_accuracy"]
    )
    confirmation = {
        "selected_candidate_id": selection["selected_candidate_id"],
        "accessed_after_selection": True,
        "selection_file_preexisted": bool(selection_file_preexisted),
        "selected_checkpoint": {
            "path": str(checkpoint),
            "sha256": actual_sha,
        },
        "base_confirm": base_confirm,
        "post_confirm": post_confirm,
        "confirm_acquisition": confirm_gain,
        "confirm_balanced_acquisition": confirm_gain,
        "confirm_natural_prior_acquisition": natural_confirm_gain,
        "confirm_per_class_recall_delta": per_class_deltas,
        "confirm_min_per_class_delta": min_per_class_delta,
        "confirm_label_constrained_acquisition": constrained_confirm_gain,
        "min_rescue_gain": config.min_rescue_gain,
        "confirm_rescue_pass": (
            v2_gate_pass if is_v2 else confirm_gain >= config.min_rescue_gain
        ),
        "v2_gate_pass": v2_gate_pass if is_v2 else None,
        "gate_version": "v2" if is_v2 else "legacy_v1",
        "legacy_confirmation_not_v2_evidence": not is_v2,
        "v2_gate_rule": {
            "natural_prior_gain": "strictly greater than min_rescue_gain",
            "post_valid_label_rate": ">= 0.99",
            "minimum_per_class_recall_delta": ">= -0.05",
        },
        "payload": confirm_bank.payload_audit().to_dict(),
        "confirmation_bundle": (
            dict(confirmation_bundle)
            if isinstance(confirmation_bundle, Mapping)
            else None
        ),
        "matched_hf_token_ce_control": matched_control_confirmation,
        "official_gate_metric": (
            "free-generation natural-prior EM acquisition plus validity/class guards"
            if is_v2
            else "legacy free-generation balanced normalized EM acquisition"
        ),
    }
    _release_model(confirm_bank)
    return confirmation


def run_task_search(
    task_name: str,
    split: RescueSplit,
    candidates: Sequence[CandidateSpec],
    tokenizer: Any,
    *,
    config: RescueConfig,
    perform_confirmation: bool,
    train_label_distribution: Mapping[str, Any],
) -> dict[str, Any]:
    task_dir = Path(config.output_root) / task_name
    task_dir.mkdir(parents=True, exist_ok=False)
    objective_plan = selection_objective_plan(
        config.loss_objectives, selection_only=config.selection_only
    )
    actual_selection_objectives = {
        candidate.loss_objective for candidate in candidates
    }
    if actual_selection_objectives != set(objective_plan.selection_objectives):
        raise RuntimeError(
            "candidate grid objectives do not match the pre-registered selection plan"
        )
    task_started = time.perf_counter()
    source_spec = TASK_BY_NAME[task_name].train
    split_path = task_dir / "split.json"
    _atomic_json(split_path, _split_manifest(split))
    token_lengths: dict[str, Any] = {
        "update_pool": source_token_length_report(
            tokenizer,
            split.update_pool,
            actual_max_source_length=config.max_source_length,
            truncation_mode=config.truncation_mode,
        ),
        "tune_audit": source_token_length_report(
            tokenizer,
            split.tune_audit,
            actual_max_source_length=config.max_source_length,
            truncation_mode=config.truncation_mode,
        ),
    }
    token_lengths["confirm_audit"] = {
        "status": (
            "deferred_until_durable_selection"
            if perform_confirmation
            else "deferred_until_independent_confirmation"
        )
    }
    _atomic_json(task_dir / "source_token_lengths.json", token_lengths)
    _atomic_json(
        task_dir / "manifest.json",
        {
            "format": FORMAT_VERSION,
            "status": "running",
            "task": task_name,
            "authorized_data": {
                "relative_path": source_spec.relative_path,
                "git_blob_sha1": source_spec.git_blob_sha1,
                "bytes": source_spec.size,
                "subset": "train",
            },
            "selection_partition": "tune_audit",
            "selection_objective_plan": asdict(objective_plan),
            "confirmation_access": (
                "only after candidate selection"
                if perform_confirmation
                else "disabled in selection-only mode"
            ),
            "audit_gradient_access": False,
            "source_token_lengths_file": "source_token_lengths.json",
            "pinned_train_label_distribution": dict(train_label_distribution),
        },
    )

    base_bank = _build_model(config, tokenizer)
    expected_fingerprint = _adapter_fingerprint(base_bank)
    payload = base_bank.payload_audit()
    if not payload.exact_budget or (not config.tiny_random_model and payload.active_atoms != 144):
        raise RuntimeError("initial model violates the fixed adapter budget")
    base_tune = evaluate_train_audit(
        base_bank,
        tokenizer,
        split.tune_audit,
        config=config,
        natural_label_prior=train_label_distribution["natural_label_prior"],
    )
    _atomic_json(task_dir / "base_tune.json", base_tune)
    _release_model(base_bank)

    measurements: list[CandidateMeasurement] = []
    candidate_registry: list[dict[str, Any]] = []
    candidate_indices = {
        candidate.candidate_id: index for index, candidate in enumerate(candidates)
    }
    trajectories = sorted(
        {
            (
                candidate.cap_per_class,
                candidate.learning_rate,
                candidate.loss_objective,
                candidate.update_sampler,
            )
            for candidate in candidates
        }
    )
    for cap_per_class, learning_rate, loss_objective, update_sampler in trajectories:
        trajectory_started = time.perf_counter()
        checkpoints = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.cap_per_class == cap_per_class
                and candidate.learning_rate == learning_rate
                and candidate.loss_objective == loss_objective
                and candidate.update_sampler == update_sampler
            ),
            key=lambda candidate: candidate.epochs,
        )
        bank = _build_model(config, tokenizer)
        fingerprint = _adapter_fingerprint(bank)
        if fingerprint != expected_fingerprint:
            raise RuntimeError("candidate adapter initialization drifted")
        update_examples = split.update_for_cap(cap_per_class)
        optimizer = torch.optim.AdamW(
            tuple(bank.all_atom_parameters()), lr=learning_rate, weight_decay=0.0
        )
        ledger = _new_resource_ledger(task_name, update_examples)
        epoch_means: list[float] = []
        completed_epochs = 0
        trajectory_id = (
            f"{loss_objective}_{update_sampler}_"
            f"lr{format(learning_rate, '.3g')}_cap{cap_per_class}"
        )
        for candidate in checkpoints:
            checkpoint_started = time.perf_counter()
            epoch_means.extend(
                train_update_segment(
                    bank,
                    tokenizer,
                    update_examples,
                    candidate,
                    optimizer,
                    ledger,
                    task_name=task_name,
                    config=config,
                    start_epoch=completed_epochs,
                    stop_epoch=candidate.epochs,
                    natural_label_prior=train_label_distribution[
                        "natural_label_prior"
                    ],
                )
            )
            completed_epochs = candidate.epochs
            post_tune = evaluate_train_audit(
                bank,
                tokenizer,
                split.tune_audit,
                config=config,
                natural_label_prior=train_label_distribution[
                    "natural_label_prior"
                ],
            )
            candidate_dir = task_dir / "candidates" / candidate.candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=False)
            checkpoint = candidate_dir / "adapter.pt"
            _atomic_torch_save(checkpoint, bank.compact_state_dict())
            measurement = CandidateMeasurement(
                candidate=candidate,
                base_tune_em=float(base_tune["normalized_em"]),
                post_tune_em=float(post_tune["normalized_em"]),
                base_tune_label_constrained_accuracy=float(
                    base_tune["label_constrained_sequence_nll_accuracy"]
                ),
                post_tune_label_constrained_accuracy=float(
                    post_tune["label_constrained_sequence_nll_accuracy"]
                ),
                update_examples_per_epoch=len(update_examples),
                checkpoint=str(checkpoint.resolve()),
                base_tune_natural_prior_em=float(base_tune["natural_prior_em"]),
                post_tune_natural_prior_em=float(post_tune["natural_prior_em"]),
                base_tune_valid_rate=float(
                    base_tune["free_generation_valid_label_rate"]
                ),
                post_tune_valid_rate=float(
                    post_tune["free_generation_valid_label_rate"]
                ),
                base_tune_per_class_recall=dict(base_tune["per_class_recall"]),
                post_tune_per_class_recall=dict(post_tune["per_class_recall"]),
            )
            measurements.append(measurement)
            resource_validation = _validate_completed_resource_ledger(
                ledger,
                candidate,
                update_examples,
                task_name=task_name,
                batch_size=config.batch_size,
                natural_label_prior=train_label_distribution[
                    "natural_label_prior"
                ],
            )
            artifact = {
                "candidate_index": candidate_indices[candidate.candidate_id],
                "candidate": asdict(candidate),
                "candidate_id": candidate.candidate_id,
                "registry_role": "selection_candidate",
                "selection_eligible": True,
                "trained_after_selection_lock": False,
                "trajectory": {
                    "trajectory_id": trajectory_id,
                    "shared_lr_cap_trajectory": True,
                    "checkpoint_epoch": candidate.epochs,
                    "trained_from_previous_checkpoint_epoch": (
                        checkpoints[checkpoints.index(candidate) - 1].epochs
                        if checkpoints.index(candidate) > 0
                        else 0
                    ),
                },
                "selection_partition": "tune_audit",
                "official_gate_metric": "free-generation natural-prior EM acquisition",
                "base_tune": base_tune,
                "post_tune": post_tune,
                "tune_acquisition": measurement.tune_acquisition,
                "tune_natural_prior_acquisition": (
                    measurement.tune_natural_prior_acquisition
                ),
                "min_per_class_delta": measurement.min_per_class_delta,
                "v2_gate_pass": measurement.v2_gate_pass(config.min_rescue_gain),
                "tune_label_constrained_acquisition": float(
                    measurement.tune_label_constrained_acquisition
                ),
                "training": {
                    "optimizer": "AdamW",
                    "training_loss_reduction": (
                        loss_objective
                    ),
                    "loss_objective": loss_objective,
                    "update_sampler": update_sampler,
                    "sequence_mean_formula": (
                        "mean_i(w_y * per_example_target_token_nll_i); no batch renormalization"
                        if loss_objective.startswith("sequence_mean")
                        else None
                    ),
                    "pinned_train_label_distribution": dict(
                        train_label_distribution
                    ),
                    "weight_decay": 0.0,
                    "learning_rate": learning_rate,
                    "checkpoint_epochs": [item.epochs for item in checkpoints],
                    "completed_epochs": candidate.epochs,
                    "epoch_mean_nll": list(epoch_means),
                    "resource_ledger": dict(ledger),
                    "resource_ledger_validation": resource_validation,
                },
                "payload": bank.payload_audit().to_dict(),
                "initial_adapter_fingerprint": fingerprint,
                "checkpoint": {
                    "path": checkpoint.name,
                    "bytes": checkpoint.stat().st_size,
                    "sha256": _sha256_file(checkpoint),
                },
                "confirm_audit_accessed": False,
                "checkpoint_wall_seconds": time.perf_counter() - checkpoint_started,
                "trajectory_wall_seconds": time.perf_counter() - trajectory_started,
            }
            metrics_path = candidate_dir / "metrics.json"
            _atomic_json(metrics_path, artifact)
            candidate_registry.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate": asdict(candidate),
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": _sha256_file(checkpoint),
                    "metrics": str(metrics_path.resolve()),
                    "metrics_sha256": _sha256_file(metrics_path),
                    "registry_role": "selection_candidate",
                    "selection_eligible": True,
                    "trained_after_selection_lock": False,
                    "confirm_audit_accessed": False,
                    "resource_ledger_validation": resource_validation,
                }
            )
            print(
                f"[{task_name}] {candidate_indices[candidate.candidate_id] + 1}/"
                f"{len(candidates)} {candidate.candidate_id}: "
                f"natural-prior tune gain="
                f"{measurement.tune_natural_prior_acquisition:+.4f}",
                flush=True,
            )
        _release_model(bank)

    actual_selection_objectives = {
        item.candidate.loss_objective for item in measurements
    }
    if actual_selection_objectives != set(objective_plan.selection_objectives):
        raise RuntimeError("executed objectives drifted from the selection plan")
    selected = select_candidate(measurements, min_gain=config.min_rescue_gain)
    selected_checkpoint_sha = _sha256_file(Path(selected.checkpoint))
    selection_lock_metadata: dict[str, Any] | None = None
    if objective_plan.train_matched_control_after_selection:
        if perform_confirmation:
            raise RuntimeError(
                "post-selection control optimization is restricted to selection-only mode"
            )
        selection_registry = sorted(
            candidate_registry, key=lambda item: item["candidate_id"]
        )
        selection_lock = {
            "format": f"{FORMAT_VERSION}.selection-lock.v1",
            "status": "selection_locked_before_matched_control",
            "task": task_name,
            "selection_rule": (
                "prefer v2 gate pass; max natural-prior tune acquisition; "
                "max post natural-prior EM; valid rate; min class delta; min exposures; "
                "fewer epochs; smaller cap; lower LR; canonical ID"
            ),
            "selected_candidate_id": selected.candidate.candidate_id,
            "selected_checkpoint": selected.checkpoint,
            "selected_checkpoint_sha256": selected_checkpoint_sha,
            "selection_candidate_ids": sorted(
                item.candidate.candidate_id for item in measurements
            ),
            "selection_candidate_count": len(measurements),
            "selection_candidate_registry_sha256": _canonical_payload_sha256(
                selection_registry
            ),
            "selection_objectives": list(objective_plan.selection_objectives),
            "control_objective": CONTROL_OBJECTIVE,
            "control_candidate_metrics_available": False,
            "confirm_audit_accessed": False,
        }
        selection_lock_path = task_dir / "selection_lock.json"
        _atomic_json(selection_lock_path, selection_lock)
        selection_lock_sha = _sha256_file(selection_lock_path)
        selection_lock_metadata = {
            "path": str(selection_lock_path.resolve()),
            "sha256": selection_lock_sha,
            "status": selection_lock["status"],
            "confirm_audit_accessed": False,
        }
        selected_registry_entry = next(
            item
            for item in selection_registry
            if item["candidate_id"] == selected.candidate.candidate_id
        )
        control_registry, control_bundle = _train_post_selection_hf_control(
            task_name,
            split.update_for_cap(selected.candidate.cap_per_class),
            selected,
            tokenizer,
            config=config,
            task_dir=task_dir,
            expected_fingerprint=expected_fingerprint,
            train_label_distribution=train_label_distribution,
            selected_registry_entry=selected_registry_entry,
            selection_lock_path=selection_lock_path,
            selection_lock_sha256=selection_lock_sha,
        )
        candidate_registry.append(control_registry)
        selected_after_control = select_candidate(
            measurements, min_gain=config.min_rescue_gain
        )
        if selected_after_control.candidate.candidate_id != selected.candidate.candidate_id:
            raise RuntimeError("selected candidate changed after control training")
        if _sha256_file(selection_lock_path) != selection_lock_sha:
            raise RuntimeError("selection lock changed after control training")
    else:
        control_bundle = None
    matched_hf = next(
        (
            item
            for item in measurements
            if item.candidate.loss_objective == CONTROL_OBJECTIVE
            and item.candidate.update_sampler == selected.candidate.update_sampler
            and item.candidate.learning_rate == selected.candidate.learning_rate
            and item.candidate.epochs == selected.candidate.epochs
            and item.candidate.cap_per_class == selected.candidate.cap_per_class
        ),
        None,
    )
    matched_hf_bundle = control_bundle
    if matched_hf_bundle is None and matched_hf is not None:
        matched_hf_bundle = {
            "candidate_id": matched_hf.candidate.candidate_id,
            "candidate": asdict(matched_hf.candidate),
            "checkpoint": matched_hf.checkpoint,
            "checkpoint_sha256": _sha256_file(Path(matched_hf.checkpoint)),
            "same_as_selected": (
                matched_hf.candidate.candidate_id
                == selected.candidate.candidate_id
            ),
            "selection_eligible": True,
            "trained_after_selection_lock": False,
            "confirm_audit_accessed": False,
        }
    confirmation_bundle = {
        "selected": {
            "candidate_id": selected.candidate.candidate_id,
            "checkpoint": selected.checkpoint,
            "checkpoint_sha256": _sha256_file(Path(selected.checkpoint)),
        },
        "matched_hf_token_ce": matched_hf_bundle,
        "matching_fields": [
            "learning_rate",
            "epochs",
            "cap_per_class",
            "update_sampler",
        ],
        "locked_before_confirm_access": True,
        "selection_lock": selection_lock_metadata,
        "control_training_phase": (
            "post_selection_lock_pre_confirmation"
            if objective_plan.train_matched_control_after_selection
            else "selection_grid_or_selected"
            if matched_hf_bundle is not None
            else "not_available"
        ),
        "control_excluded_from_selection": bool(
            objective_plan.train_matched_control_after_selection
        ),
    }
    selection = {
        "format": FORMAT_VERSION,
        "task": task_name,
        "rule": (
            "prefer v2 gate pass; max natural-prior tune acquisition; "
            "max post natural-prior EM; valid rate; min class delta; min exposures; "
            "fewer epochs; smaller cap; lower LR; canonical ID"
        ),
        "selected_candidate_id": selected.candidate.candidate_id,
        "selection_objective_plan": asdict(objective_plan),
        "selection_candidate_ids": sorted(
            item.candidate.candidate_id for item in measurements
        ),
        "selection_candidate_count": len(measurements),
        "post_selection_control_candidate_ids": sorted(
            item["candidate_id"]
            for item in candidate_registry
            if not bool(item.get("selection_eligible", True))
        ),
        "selection_lock": selection_lock_metadata,
        "selected": {
            "candidate": asdict(selected.candidate),
            "base_tune_em": selected.base_tune_em,
            "post_tune_em": selected.post_tune_em,
            "tune_acquisition": selected.tune_acquisition,
            "base_tune_natural_prior_em": selected.base_tune_natural_prior_em,
            "post_tune_natural_prior_em": selected.post_tune_natural_prior_em,
            "tune_natural_prior_acquisition": (
                selected.tune_natural_prior_acquisition
            ),
            "post_tune_valid_rate": selected.post_tune_valid_rate,
            "min_per_class_delta": selected.min_per_class_delta,
            "base_tune_per_class_recall": selected.base_tune_per_class_recall,
            "post_tune_per_class_recall": selected.post_tune_per_class_recall,
            "tune_label_constrained_acquisition": (
                selected.tune_label_constrained_acquisition
            ),
            "update_exposures": selected.update_exposures,
            "checkpoint": selected.checkpoint,
            "checkpoint_sha256": selected_checkpoint_sha,
        },
        "candidate_registry": sorted(
            candidate_registry, key=lambda item: item["candidate_id"]
        ),
        "confirmation_bundle": confirmation_bundle,
        "split_protocol": {
            "base_seed": config.seed,
            "derived_partition_seed": _stable_seed(
                config.seed, task_name, "three-way-split"
            ),
            "split_unit": split.split_unit,
            "group_key_version": split.group_key_version,
            "semantic_selection_algorithm": split.selection_algorithm_version,
            "fresh_confirm_against_manifests": [
                dict(item) for item in split.freshness_sources
            ],
            "prior_touched_requested_example_id_count": len(
                split.prior_touched_requested_example_ids
            ),
            "prior_touched_row_count": len(split.prior_touched_example_ids),
            "prior_touched_group_count": len(split.prior_touched_group_keys),
            "max_update_per_class": max(config.caps_per_class),
            "tune_audit_per_class": config.tune_audit_per_class,
            "confirm_audit_per_class": config.confirm_audit_per_class,
            "split_manifest_sha256": _sha256_file(split_path),
        },
        "evaluation_protocol": {
            "max_source_length": config.max_source_length,
            "max_target_length": config.max_target_length,
            "truncation_mode": config.truncation_mode,
            "official_gate_metric": "free-generation natural-prior EM acquisition",
        },
        "model_protocol": {
            "local_model_path": str(Path(config.model_path).resolve()),
            "tiny_random_model": config.tiny_random_model,
            "initial_adapter_fingerprint": expected_fingerprint,
            "lora": {
                "target_modules": "all T5 q/v projections",
                "rank": 4,
                "alpha": 16.0,
                "dropout": 0.1,
            },
        },
        "min_rescue_gain": config.min_rescue_gain,
        "v2_gate_rule": {
            "natural_prior_gain": "strictly greater than min_rescue_gain",
            "post_valid_label_rate": ">= 0.99",
            "minimum_per_class_recall_delta": ">= -0.05",
        },
        "pinned_train_label_distribution": dict(train_label_distribution),
        "tune_rescue_pass": selected.v2_gate_pass(config.min_rescue_gain),
        "gate_version": "v2",
        "confirm_metrics_used_for_selection": False,
    }
    _atomic_json(task_dir / "selection.json", selection)

    if not perform_confirmation:
        result = {
            "format": FORMAT_VERSION,
            "status": "selection_completed",
            "task": task_name,
            "candidate_count": len(candidates),
            "selection_candidate_count": len(measurements),
            "trained_candidate_count": len(candidate_registry),
            "post_selection_control_count": sum(
                not bool(item.get("selection_eligible", True))
                for item in candidate_registry
            ),
            "selection": selection,
            "confirmation": None,
            "wall_seconds": time.perf_counter() - task_started,
        }
        _atomic_json(task_dir / "result.json", result)
        _atomic_json(
            task_dir / "manifest.json",
            {
                "format": FORMAT_VERSION,
                "status": "selection_completed",
                "task": task_name,
                "authorized_data": {
                    "relative_path": source_spec.relative_path,
                    "git_blob_sha1": source_spec.git_blob_sha1,
                    "bytes": source_spec.size,
                    "subset": "train",
                },
                "selection_partition": "tune_audit",
                "confirmation_access": "not accessed",
                "audit_gradient_access": False,
                "selected_candidate_id": selected.candidate.candidate_id,
                "gate_version": "v2",
                "pinned_train_label_distribution": dict(
                    train_label_distribution
                ),
                "candidate_registry": selection["candidate_registry"],
                "confirmation_bundle": selection["confirmation_bundle"],
                "selection_objective_plan": selection["selection_objective_plan"],
                "selection_lock": selection["selection_lock"],
            },
        )
        (task_dir / "SELECTION_COMPLETED").write_text(
            "selection completed; confirmation not accessed\n", encoding="utf-8"
        )
        return result

    # Confirmation is first accessed here, after selection.json is durable.
    # Even descriptive token-length inspection is kept behind this boundary.
    _preflight_selection_confirmation(selection)
    token_lengths["confirm_audit"] = source_token_length_report(
        tokenizer,
        split.confirm_audit,
        actual_max_source_length=config.max_source_length,
        truncation_mode=config.truncation_mode,
    )
    _atomic_json(task_dir / "source_token_lengths.json", token_lengths)
    confirmation = evaluate_selected_confirmation(
        selection,
        split,
        tokenizer,
        config=config,
        selection_file_preexisted=(task_dir / "selection.json").is_file(),
    )
    _atomic_json(task_dir / "confirmation.json", confirmation)

    result = {
        "format": FORMAT_VERSION,
        "status": "completed",
        "task": task_name,
        "candidate_count": len(candidates),
        "selection_candidate_count": len(measurements),
        "trained_candidate_count": len(candidate_registry),
        "post_selection_control_count": sum(
            not bool(item.get("selection_eligible", True))
            for item in candidate_registry
        ),
        "selection": selection,
        "confirmation": confirmation,
        "wall_seconds": time.perf_counter() - task_started,
    }
    _atomic_json(task_dir / "result.json", result)
    _atomic_json(
        task_dir / "manifest.json",
        {
            "format": FORMAT_VERSION,
            "status": "completed",
            "task": task_name,
            "authorized_data": {
                "relative_path": source_spec.relative_path,
                "git_blob_sha1": source_spec.git_blob_sha1,
                "bytes": source_spec.size,
                "subset": "train",
            },
            "selection_partition": "tune_audit",
            "confirmation_access": "only after candidate selection",
            "audit_gradient_access": False,
            "source_token_lengths_file": "source_token_lengths.json",
            "selected_candidate_id": selected.candidate.candidate_id,
            "gate_version": "v2",
            "pinned_train_label_distribution": dict(train_label_distribution),
            "candidate_registry": selection["candidate_registry"],
            "confirmation_bundle": selection["confirmation_bundle"],
            "selection_objective_plan": selection["selection_objective_plan"],
            "selection_lock": selection["selection_lock"],
        },
    )
    (task_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return result


def parse_tasks(values: Sequence[str]) -> tuple[str, ...]:
    flattened = [part.strip() for value in values for part in value.split(",") if part.strip()]
    if len(flattened) == 1 and flattened[0].lower() == "all":
        return SUPPORTED_TASKS
    lookup = {task.lower(): task for task in SUPPORTED_TASKS}
    try:
        selected = {lookup[item.lower()] for item in flattened}
    except KeyError as error:
        raise ValueError(
            f"unsupported task {error.args[0]!r}; allowed={SUPPORTED_TASKS}"
        ) from error
    if not selected:
        raise ValueError("task selection cannot be empty")
    return tuple(task for task in SUPPORTED_TASKS if task in selected)


def _parse_numbers(values: Sequence[str], *, kind: str) -> tuple[float | int, ...]:
    flattened = [part.strip() for value in values for part in value.split(",") if part.strip()]
    if not flattened:
        raise ValueError(f"{kind} list cannot be empty")
    if kind == "learning rate":
        parsed: tuple[float | int, ...] = tuple(float(value) for value in flattened)
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in parsed):
            raise ValueError("learning rates must be finite and positive")
    else:
        parsed = tuple(int(value) for value in flattened)
        if any(isinstance(value, bool) or int(value) <= 0 for value in parsed):
            raise ValueError(f"{kind} values must be positive integers")
    return tuple(sorted(set(parsed)))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--lrs", nargs="+", default=["0.0003", "0.001", "0.003"])
    parser.add_argument("--epochs", nargs="+", default=["1", "3", "5"])
    parser.add_argument("--caps-per-class", nargs="+", default=["16", "32"])
    parser.add_argument(
        "--loss-objectives",
        nargs="+",
        choices=LOSS_OBJECTIVES,
        default=["hf_token_ce"],
        help=(
            "requested objectives; in selection-only mode with any sequence objective, "
            "hf_token_ce is trained once post-selection as a matched control"
        ),
    )
    parser.add_argument(
        "--update-sampler",
        choices=UPDATE_SAMPLERS,
        default="balanced",
        help="class-balanced update rows; its exact q is recorded per candidate",
    )
    parser.add_argument("--tune-audit-per-class", type=_positive_int, default=32)
    parser.add_argument("--confirm-audit-per-class", type=_positive_int, default=64)
    parser.add_argument(
        "--split-unit",
        choices=SPLIT_UNITS,
        default="row",
        help=(
            "row reproduces legacy artifacts; semantic_group prevents related "
            "QQP/BoolQA/MultiRC rows from crossing update/audit partitions"
        ),
    )
    parser.add_argument(
        "--fresh-confirm-against-manifest",
        action="append",
        default=[],
        help=(
            "prior split.json or root manifest.json whose accessed "
            "update/tune/confirm rows define groups forbidden only from the new "
            "confirmation partition; "
            "repeat for multiple prior runs"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--max-source-length", type=_positive_int, default=512)
    parser.add_argument("--max-target-length", type=_positive_int, default=50)
    parser.add_argument(
        "--truncation-mode",
        choices=("official_right", "field_aware"),
        default="official_right",
    )
    parser.add_argument("--min-rescue-gain", type=float, default=0.05)
    parser.add_argument("--tiny-random-model", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--selection-only",
        action="store_true",
        help="search/tune and write selection, without evaluating confirm-audit",
    )
    mode.add_argument(
        "--confirm-selection",
        help="independently confirm a prior selection.json or result.json",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> RescueConfig:
    tasks = parse_tasks(args.tasks)
    lrs = tuple(float(value) for value in _parse_numbers(args.lrs, kind="learning rate"))
    epochs = tuple(int(value) for value in _parse_numbers(args.epochs, kind="epoch"))
    caps = tuple(int(value) for value in _parse_numbers(args.caps_per_class, kind="cap"))
    if not math.isfinite(args.min_rescue_gain) or not 0.0 <= args.min_rescue_gain <= 1.0:
        raise ValueError("min-rescue-gain must be a finite fraction in [0, 1]")
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise ValueError("model-path must be an existing local model directory")
    return RescueConfig(
        data_root=str(Path(args.data_root).expanduser().resolve()),
        model_path=str(model_path),
        output_root=str(Path(args.output_root).expanduser().resolve()),
        task_names=tasks,
        learning_rates=lrs,
        epochs=epochs,
        caps_per_class=caps,
        tune_audit_per_class=args.tune_audit_per_class,
        confirm_audit_per_class=args.confirm_audit_per_class,
        seed=args.seed,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        truncation_mode=args.truncation_mode,
        min_rescue_gain=float(args.min_rescue_gain),
        tiny_random_model=bool(args.tiny_random_model),
        selection_only=bool(args.selection_only),
        confirm_selection=(
            str(Path(args.confirm_selection).expanduser().resolve())
            if args.confirm_selection
            else None
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
        split_unit=args.split_unit,
        fresh_confirm_against_manifests=tuple(
            str(Path(path).expanduser().resolve())
            for path in args.fresh_confirm_against_manifest
        ),
        loss_objectives=tuple(dict.fromkeys(args.loss_objectives)),
        update_sampler=str(args.update_sampler),
    )


def _environment_manifest(config: RescueConfig) -> dict[str, Any]:
    versions = {}
    for package in ("torch", "transformers", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "device": config.device,
    }


def load_selection_artifact(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read selection artifact {source}") from error
    if isinstance(payload, dict) and isinstance(payload.get("selection"), dict):
        payload = payload["selection"]
    if not isinstance(payload, dict) or payload.get("format") not in {
        FORMAT_VERSION,
        LEGACY_FORMAT_VERSION,
    }:
        raise ValueError("selection artifact has an unsupported format")
    if payload.get("task") not in SUPPORTED_TASKS:
        raise ValueError("selection artifact has an unsupported task")
    if not isinstance(payload.get("split_protocol"), dict) or not isinstance(
        payload.get("evaluation_protocol"), dict
    ):
        raise ValueError("selection artifact is missing locked protocol metadata")
    return payload


def config_for_confirmation(
    config: RescueConfig, selection: Mapping[str, Any]
) -> RescueConfig:
    split_protocol = selection["split_protocol"]
    evaluation = selection["evaluation_protocol"]
    model_protocol = selection.get("model_protocol")
    if not isinstance(model_protocol, Mapping):
        raise ValueError("selection artifact is missing model protocol metadata")
    artifact_split_unit = str(split_protocol.get("split_unit", "row"))
    if artifact_split_unit not in SPLIT_UNITS:
        raise ValueError("selection artifact has an unsupported split unit")
    if artifact_split_unit == "semantic_group" and split_protocol.get(
        "group_key_version"
    ) != SEMANTIC_GROUP_KEY_VERSION:
        raise ValueError("selection artifact has an unsupported group-key version")
    artifact_semantic_algorithm = split_protocol.get(
        "semantic_selection_algorithm",
        LEGACY_SEMANTIC_SELECTION_ALGORITHM,
    )
    if (
        artifact_split_unit == "semantic_group"
        and artifact_semantic_algorithm not in SEMANTIC_SELECTION_ALGORITHMS
    ):
        raise ValueError("selection artifact has an unsupported split algorithm")
    freshness_manifests = split_protocol.get(
        "fresh_confirm_against_manifests", []
    )
    if not isinstance(freshness_manifests, list) or any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("path"), str)
        or not isinstance(item.get("sha256"), str)
        for item in freshness_manifests
    ):
        raise ValueError("selection artifact has invalid fresh-confirm manifests")
    if bool(model_protocol.get("tiny_random_model")) != config.tiny_random_model:
        raise ValueError(
            "--tiny-random-model must match the selection invocation"
        )
    if not config.tiny_random_model and Path(
        str(model_protocol.get("local_model_path", ""))
    ).resolve() != Path(config.model_path).resolve():
        raise ValueError(
            "confirmation model-path must match the selection's local base model"
        )
    return replace(
        config,
        task_names=(str(selection["task"]),),
        seed=int(split_protocol["base_seed"]),
        caps_per_class=(int(split_protocol["max_update_per_class"]),),
        tune_audit_per_class=int(split_protocol["tune_audit_per_class"]),
        confirm_audit_per_class=int(split_protocol["confirm_audit_per_class"]),
        max_source_length=int(evaluation["max_source_length"]),
        max_target_length=int(evaluation["max_target_length"]),
        truncation_mode=str(evaluation["truncation_mode"]),
        min_rescue_gain=float(selection["min_rescue_gain"]),
        selection_only=False,
        # Artifacts predating semantic grouping have no field and must rebuild
        # the exact legacy row split for hash verification.
        split_unit=artifact_split_unit,
        semantic_selection_algorithm=str(artifact_semantic_algorithm),
        fresh_confirm_against_manifests=tuple(
            str(item["path"])
            for item in freshness_manifests
        ),
        loss_objectives=(
            str(selection.get("selected", {})
                .get("candidate", {})
                .get("loss_objective", "hf_token_ce")),
        ),
        update_sampler=str(
            selection.get("selected", {})
            .get("candidate", {})
            .get("update_sampler", "balanced")
        ),
    )


def run_independent_confirmation(
    selection: Mapping[str, Any],
    selection_path: Path,
    split: RescueSplit,
    tokenizer: Any,
    *,
    config: RescueConfig,
) -> dict[str, Any]:
    """Run the one-shot confirmation phase without touching tune-audit."""

    task_name = split.task_name
    task_dir = Path(config.output_root) / task_name
    task_dir.mkdir(parents=True, exist_ok=False)
    split_path = task_dir / "split.json"
    _atomic_json(split_path, _split_manifest(split))
    expected_split_sha = selection["split_protocol"]["split_manifest_sha256"]
    actual_split_sha = _sha256_file(split_path)
    if actual_split_sha != expected_split_sha:
        raise ValueError(
            "reconstructed train-only split does not match the selection artifact"
        )
    _preflight_selection_confirmation(selection)
    token_lengths = source_token_length_report(
        tokenizer,
        split.confirm_audit,
        actual_max_source_length=config.max_source_length,
        truncation_mode=config.truncation_mode,
    )
    _atomic_json(
        task_dir / "source_token_lengths.json",
        {
            "update_pool": {"status": "not_accessed_in_confirmation"},
            "tune_audit": {"status": "not_accessed_in_confirmation"},
            "confirm_audit": token_lengths,
        },
    )
    confirmation = evaluate_selected_confirmation(
        selection,
        split,
        tokenizer,
        config=config,
        selection_file_preexisted=selection_path.is_file(),
    )
    result = {
        "format": FORMAT_VERSION,
        "status": "confirmation_completed",
        "task": task_name,
        "selection_artifact": {
            "path": str(selection_path),
            "sha256": _sha256_file(selection_path),
        },
        "selection": dict(selection),
        "confirmation": confirmation,
        "tune_audit_accessed_in_this_invocation": False,
    }
    _atomic_json(task_dir / "confirmation.json", confirmation)
    _atomic_json(task_dir / "result.json", result)
    _atomic_json(
        task_dir / "manifest.json",
        {
            "format": FORMAT_VERSION,
            "status": "confirmation_completed",
            "task": task_name,
            "authorized_source_subset": "train only",
            "selection_artifact": str(selection_path),
            "tune_audit_accessed": False,
            "confirm_audit_gradient_access": False,
            "truncation_mode": config.truncation_mode,
            "max_source_length": config.max_source_length,
            "selection_format": selection.get("format"),
            "gate_version": confirmation["gate_version"],
            "legacy_confirmation_not_v2_evidence": confirmation[
                "legacy_confirmation_not_v2_evidence"
            ],
            "pinned_train_label_distribution": selection.get(
                "pinned_train_label_distribution"
            ),
            "confirmation_bundle": selection.get("confirmation_bundle"),
        },
    )
    (task_dir / "COMPLETED").write_text("confirmation completed\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    selection_payload: dict[str, Any] | None = None
    selection_path: Path | None = None
    try:
        config = _config_from_args(args)
        if config.confirm_selection is not None:
            selection_path = Path(config.confirm_selection)
            selection_payload = load_selection_artifact(selection_path)
            config = config_for_confirmation(config, selection_payload)
            candidates: tuple[CandidateSpec, ...] = ()
            objective_plan: SelectionObjectivePlan | None = None
        else:
            objective_plan = selection_objective_plan(
                config.loss_objectives, selection_only=config.selection_only
            )
            candidates = candidate_grid(
                config.learning_rates,
                config.epochs,
                config.caps_per_class,
                objective_plan.selection_objectives,
                config.update_sampler,
            )
    except ValueError as error:
        parser.error(str(error))
    output_root = Path(config.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"refusing non-empty output root {output_root}; choose a fresh path"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    # This is the only dataset loader call in the executable.  Its split is a
    # constant train value; every downstream function re-validates subset tags.
    splits: dict[str, RescueSplit] = {}
    train_label_distributions: dict[str, dict[str, Any]] = {}
    for task_name in config.task_names:
        train_examples = load_official_examples(
            config.data_root, task_name, "train", verify=True
        )
        train_label_distributions[task_name] = pinned_train_label_distribution(
            train_examples, task_name=task_name
        )
        if selection_payload is not None and selection_payload.get("format") == FORMAT_VERSION:
            expected_distribution = selection_payload.get(
                "pinned_train_label_distribution"
            )
            if expected_distribution != train_label_distributions[task_name]:
                raise ValueError(
                    "complete pinned train label distribution/content hash no longer "
                    "matches the v2 selection"
                )
        touched_ids, freshness_sources = load_prior_touched_ids(
            config.fresh_confirm_against_manifests,
            task_name=task_name,
            train_examples=train_examples,
        )
        if selection_payload is not None:
            expected_sources = selection_payload["split_protocol"].get(
                "fresh_confirm_against_manifests", []
            )
            if [dict(item) for item in freshness_sources] != expected_sources:
                raise ValueError(
                    "fresh-confirm manifests no longer match the selection "
                    "artifact (path, SHA, or ledger counts changed)"
                )
        splits[task_name] = build_rescue_split(
            train_examples,
            task_name=task_name,
            max_update_per_class=max(config.caps_per_class),
            tune_audit_per_class=config.tune_audit_per_class,
            confirm_audit_per_class=config.confirm_audit_per_class,
            seed=_stable_seed(config.seed, task_name, "three-way-split"),
            split_unit=config.split_unit,
            semantic_selection_algorithm=config.semantic_selection_algorithm,
            prior_touched_example_ids=touched_ids,
            freshness_sources=freshness_sources,
        )

    if selection_payload is not None:
        confirmation_task = config.task_names[0]
        expected_split_sha = selection_payload["split_protocol"].get(
            "split_manifest_sha256"
        )
        actual_split_sha = _atomic_json_payload_sha256(
            _split_manifest(splits[confirmation_task])
        )
        if actual_split_sha != expected_split_sha:
            raise ValueError(
                "reconstructed train-only split does not match the selection artifact"
            )
        # Fail closed on every checkpoint/bundle/registry/lock invariant before
        # tokenizer construction or any confirm-row length inspection.
        _preflight_selection_confirmation(selection_payload)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, use_fast=True, local_files_only=True
    )

    def root_token_report(task: str, split: RescueSplit) -> dict[str, Any]:
        common = {
            "actual_max_source_length": config.max_source_length,
            "truncation_mode": config.truncation_mode,
        }
        if selection_payload is not None:
            return {
                "update_pool": {"status": "not_accessed_in_confirmation"},
                "tune_audit": {"status": "not_accessed_in_confirmation"},
                "confirm_audit": source_token_length_report(
                    tokenizer,
                    split.confirm_audit,
                    actual_max_source_length=config.max_source_length,
                    truncation_mode=config.truncation_mode,
                ),
                "protocol": common,
            }
        report = {
            "update_pool": source_token_length_report(
                tokenizer,
                split.update_pool,
                actual_max_source_length=config.max_source_length,
                truncation_mode=config.truncation_mode,
            ),
            "tune_audit": source_token_length_report(
                tokenizer,
                split.tune_audit,
                actual_max_source_length=config.max_source_length,
                truncation_mode=config.truncation_mode,
            ),
            "protocol": common,
        }
        report["confirm_audit"] = {
            "status": (
                "deferred_until_independent_confirmation"
                if config.selection_only
                else "deferred_until_durable_selection"
            )
        }
        return report

    root_manifest = {
        "format": FORMAT_VERSION,
        "status": "running",
        "config": asdict(config),
        "candidate_grid": [asdict(candidate) for candidate in candidates],
        "selection_objective_plan": (
            asdict(objective_plan) if objective_plan is not None else None
        ),
        "execution_mode": (
            "independent_confirmation"
            if selection_payload is not None
            else "selection_only"
            if config.selection_only
            else "search_then_confirmation"
        ),
        "protocol": {
            "repository": OFFICIAL_REPOSITORY,
            "commit": OFFICIAL_COMMIT,
            "authorized_source_subset": "train only",
            "gradient_partition": "update only",
            "training_loss_reduction": (
                "candidate-specific: hf token CE or unnormalized weighted mean of "
                "per-example target-token-mean NLL"
            ),
            "loss_objectives": list(config.loss_objectives),
            "selection_objectives": (
                list(objective_plan.selection_objectives)
                if objective_plan is not None
                else None
            ),
            "post_selection_matched_control": (
                {
                    "objective": CONTROL_OBJECTIVE,
                    "enabled": objective_plan.train_matched_control_after_selection,
                    "matching_fields": [
                        "learning_rate",
                        "epochs",
                        "cap_per_class",
                        "update_sampler",
                    ],
                    "selection_eligible": False,
                    "timing": "after durable selection lock; before confirm access",
                }
                if objective_plan is not None
                else None
            ),
            "update_sampler": config.update_sampler,
            "selection_partition": "tune_audit only",
            "confirmation_partition": (
                "confirm_audit only"
                if selection_payload is not None
                else "not accessed"
                if config.selection_only
                else "confirm_audit accessed after selection"
            ),
            "official_gate_metric": (
                "free-generation natural-prior exact-match acquisition, strict > threshold; "
                "post valid-label rate >=.99; min class-recall delta >=-.05"
            ),
            "diagnostic_metric": "label-constrained sequence-NLL accuracy acquisition",
            "epoch_search": "shared trajectory checkpoints within each learning-rate/cap pair",
            "claim_scope": "acquisition rescue diagnostic, not continual retention",
        },
        "environment": _environment_manifest(config),
        "model_files": {
            name: {
                "bytes": (Path(config.model_path) / name).stat().st_size,
                "sha256": _sha256_file(Path(config.model_path) / name),
            }
            for name in ("config.json", "pytorch_model.bin", "model.safetensors")
            if (Path(config.model_path) / name).is_file()
        },
        "splits": {task: _split_manifest(split) for task, split in splits.items()},
        "pinned_train_label_distributions": train_label_distributions,
        "source_token_lengths": {
            task: root_token_report(task, split) for task, split in splits.items()
        },
    }
    _atomic_json(output_root / "manifest.json", root_manifest)

    if selection_payload is not None:
        assert selection_path is not None
        task_name = config.task_names[0]
        result = run_independent_confirmation(
            selection_payload,
            selection_path,
            splits[task_name],
            tokenizer,
            config=config,
        )
        summary = {
            task_name: {
                "selected_candidate_id": selection_payload["selected_candidate_id"],
                "confirm_acquisition": result["confirmation"]["confirm_acquisition"],
                "confirm_natural_prior_acquisition": result["confirmation"][
                    "confirm_natural_prior_acquisition"
                ],
                "confirm_label_constrained_acquisition": result["confirmation"][
                    "confirm_label_constrained_acquisition"
                ],
                "confirm_rescue_pass": result["confirmation"]["confirm_rescue_pass"],
            }
        }
        _atomic_json(output_root / "summary.json", summary)
        root_manifest["status"] = "confirmation_completed"
        root_manifest["summary"] = summary
        _atomic_json(output_root / "manifest.json", root_manifest)
        (output_root / "COMPLETED").write_text(
            "independent confirmation completed\n", encoding="utf-8"
        )
        return 0

    results = {
        task_name: run_task_search(
            task_name,
            splits[task_name],
            candidates,
            tokenizer,
            config=config,
            perform_confirmation=not config.selection_only,
            train_label_distribution=train_label_distributions[task_name],
        )
        for task_name in config.task_names
    }
    # Refresh the root ledger from each task artifact.  Search-then-confirm
    # tasks replace the deferred confirm report only after selection.json was
    # durable; selection-only tasks remain explicitly deferred.
    refreshed_token_lengths: dict[str, Any] = {}
    for task_name in config.task_names:
        with (output_root / task_name / "source_token_lengths.json").open(
            "r", encoding="utf-8"
        ) as handle:
            refreshed_token_lengths[task_name] = json.load(handle)
    root_manifest["source_token_lengths"] = refreshed_token_lengths
    summary: dict[str, Any] = {}
    for task, result in results.items():
        item = {
            "selected_candidate_id": result["selection"]["selected_candidate_id"],
            "tune_acquisition": result["selection"]["selected"]["tune_acquisition"],
            "tune_natural_prior_acquisition": result["selection"]["selected"][
                "tune_natural_prior_acquisition"
            ],
            "tune_rescue_pass": result["selection"]["tune_rescue_pass"],
        }
        if result["confirmation"] is not None:
            item.update(
                {
                    "confirm_acquisition": result["confirmation"][
                        "confirm_acquisition"
                    ],
                    "confirm_natural_prior_acquisition": result["confirmation"][
                        "confirm_natural_prior_acquisition"
                    ],
                    "confirm_label_constrained_acquisition": result["confirmation"][
                        "confirm_label_constrained_acquisition"
                    ],
                    "confirm_rescue_pass": result["confirmation"][
                        "confirm_rescue_pass"
                    ],
                }
            )
        summary[task] = item
    _atomic_json(output_root / "summary.json", summary)
    root_manifest["status"] = (
        "selection_completed" if config.selection_only else "completed"
    )
    root_manifest["summary"] = summary
    _atomic_json(output_root / "manifest.json", root_manifest)
    if config.selection_only:
        (output_root / "SELECTION_COMPLETED").write_text(
            "selection completed; confirmation not accessed\n", encoding="utf-8"
        )
    else:
        (output_root / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
