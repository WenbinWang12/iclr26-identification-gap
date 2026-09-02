"""Binary Boundary-Anchored Pairwise LoRA acquisition rescue.

This executable is deliberately separate from :mod:`run_acquisition_rescue`.
It is a narrow, train-only qualification path for the three binary Order-4
tasks (QQP, BoolQA, and MultiRC) after ordinary pointwise objectives have shown
class-boundary drift.  It never treats calibration gain as training gain.

The protocol has four disjoint train-derived partitions:

* ``fit``: the only rows authorized for gradients;
* ``internal_calibration``: chooses a scalar label-margin threshold and the
  first run-locally locked optimizer-step checkpoint that passes the gate;
* ``development_tune``: a post-selection second qualification gate only;
* ``confirm``: sealed until a durable selection is independently confirmed.

All partitions are semantic-group atomic.  The trained score is the
sequence-mean official-label margin ``NLL(False) - NLL(True)``.  The pairwise
loss improves ranking without rewarding a common margin translation, while a
common-shift anchor and a worst-class base-correct hinge guard the decision
boundary.  A frozen base model receives the exact same train-only calibration
procedure and is the sole denominator of the BAP training-contribution gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

try:
    from . import run_acquisition_rescue as rescue
    from .objectives import per_example_target_token_nll
    from .order4_data import OfficialExample, TASK_BY_NAME, load_official_examples
except ImportError:  # pragma: no cover - direct script execution.
    import run_acquisition_rescue as rescue  # type: ignore
    from objectives import per_example_target_token_nll  # type: ignore
    from order4_data import (  # type: ignore
        OfficialExample,
        TASK_BY_NAME,
        load_official_examples,
    )


FORMAT_VERSION = "phase2i.binary-bap-lora-rescue.v1"
BINARY_LABELS = ("True", "False")
SUPPORTED_TASKS = rescue.SUPPORTED_TASKS
DEFAULT_CHECKPOINT_STEPS = (96, 192, 288, 384)
CONTROL_OBJECTIVE = "sequence_mean_prior"
TRUNCATION_MODES = ("official_right", "field_aware")


def _strictly_above(value: float, threshold: float) -> bool:
    return value > threshold and not math.isclose(
        value, threshold, rel_tol=0.0, abs_tol=1e-12
    )


def _at_least(value: float, threshold: float) -> bool:
    return value >= threshold or math.isclose(
        value, threshold, rel_tol=0.0, abs_tol=1e-12
    )


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_float_rows(
    example_ids: Sequence[str], values: Sequence[float]
) -> str:
    if len(example_ids) != len(values):
        raise ValueError("margin IDs and values are not aligned")
    rows = [
        [example_id, float(value)]
        for example_id, value in zip(example_ids, values)
    ]
    return _sha256_json(rows)


def _local_artifact_fingerprint(path_value: str) -> dict[str, Any]:
    """Hash every file backing the local model/tokenizer artifact."""

    root = Path(path_value).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"BAP requires an existing local model/tokenizer artifact: {root}"
        )
    if root.is_file():
        paths = (root,)
        kind = "file"
    elif root.is_dir():
        paths = tuple(
            sorted(
                (path for path in root.rglob("*") if path.is_file()),
                key=lambda path: path.relative_to(root).as_posix(),
            )
        )
        kind = "directory"
    else:
        raise ValueError(f"unsupported model/tokenizer artifact type: {root}")
    if not paths:
        raise ValueError(f"model/tokenizer artifact contains no files: {root}")
    entries = []
    total_bytes = 0
    for path in paths:
        size = int(path.stat().st_size)
        total_bytes += size
        entries.append(
            {
                "relative_path": (
                    path.name if root.is_file() else path.relative_to(root).as_posix()
                ),
                "bytes": size,
                "sha256": rescue._sha256_file(path),
            }
        )
    return {
        "resolved_path": str(root),
        "kind": kind,
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
        "aggregate_sha256": _sha256_json(entries),
    }


def _official_train_artifact_fingerprint(
    data_root: str, *, task_name: str
) -> dict[str, Any]:
    """Hash only the pinned train and label blobs, never test/dev bytes."""

    root = Path(data_root).expanduser().resolve()
    spec = TASK_BY_NAME[task_name]
    entries = []
    for role, blob in (("train", spec.train), ("labels", spec.label_file)):
        path = (root / blob.relative_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing official {role} blob: {path}")
        entries.append(
            {
                "role": role,
                "relative_path": blob.relative_path.replace("\\", "/"),
                "bytes": int(path.stat().st_size),
                "sha256": rescue._sha256_file(path),
                "pinned_git_blob_sha1": blob.git_blob_sha1,
            }
        )
    return {
        "resolved_data_root": str(root),
        "task": task_name,
        "split_scope": "train+labels only; test/dev not opened",
        "files": entries,
        "aggregate_sha256": _sha256_json(entries),
    }


def _tokenizer_protocol_fingerprint(tokenizer: Any) -> dict[str, Any]:
    """Bind tokenizer vocabulary, special IDs, label encoding, and software."""

    try:
        import transformers
    except ImportError:  # pragma: no cover - the real runner requires it.
        transformers = None
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise ValueError("tokenizer must expose a non-empty vocabulary")
    vocabulary_rows = sorted(
        ((str(token), int(index)) for token, index in vocabulary.items()),
        key=lambda item: (item[0], item[1]),
    )
    label_encodings: dict[str, list[int]] = {}
    for label in BINARY_LABELS:
        encoded = tokenizer(label, add_special_tokens=True)
        values = encoded["input_ids"]
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().tolist()
        if values and isinstance(values[0], (list, tuple)):
            values = values[0]
        label_encodings[label] = [int(value) for value in values]
    payload = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocabulary_size": len(vocabulary_rows),
        "vocabulary_sha256": _sha256_json(vocabulary_rows),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "decoder_label_encodings": label_encodings,
        "transformers_version": (
            None if transformers is None else str(transformers.__version__)
        ),
        "torch_version": str(torch.__version__),
    }
    return {**payload, "aggregate_sha256": _sha256_json(payload)}


def _train_corpus_fingerprint(
    examples: Sequence[OfficialExample], *, task_name: str
) -> dict[str, Any]:
    """Bind every ordered field that can affect splits, prompts, or labels."""

    if not examples:
        raise ValueError("cannot fingerprint an empty train corpus")
    rows = []
    for example in examples:
        if example.task_name != task_name or example.subset != "train":
            raise ValueError("train corpus fingerprint escaped the locked task/subset")
        rows.append(
            {
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
        )
    return {
        "task": task_name,
        "ordered_row_count": len(rows),
        "ordered_full_content_sha256": _sha256_json(rows),
        "field_schema": list(rows[0]),
    }


def _execution_environment_fingerprint(device_value: str) -> dict[str, Any]:
    """Lock numerical execution settings that may change threshold decisions."""

    device = torch.device(device_value)
    payload: dict[str, Any] = {
        "configured_device": str(device_value),
        "resolved_device_type": device.type,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "default_dtype": str(torch.get_default_dtype()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cpu_machine": platform.machine(),
        "cpu_processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "mkl_available": bool(torch.backends.mkl.is_available()),
        "openmp_available": bool(torch.backends.openmp.is_available()),
        "torch_build_config_sha256": hashlib.sha256(
            torch.__config__.show().encode("utf-8")
        ).hexdigest(),
        "numerical_environment_variables": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("locked CUDA device is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        payload.update(
            {
                "resolved_cuda_index": int(index),
                "cuda_device_name": torch.cuda.get_device_name(index),
                "cuda_device_capability": list(torch.cuda.get_device_capability(index)),
                "cuda_driver_version": (
                    int(torch._C._cuda_getDriverVersion())
                    if hasattr(torch._C, "_cuda_getDriverVersion")
                    else None
                ),
            }
        )
    return {**payload, "aggregate_sha256": _sha256_json(payload)}


@dataclass(frozen=True)
class BAPConfig:
    data_root: str
    model_path: str
    output_root: str
    task_name: str = "BoolQA"
    learning_rate: float = 3e-4
    fit_per_class: int = 384
    calibration_per_class: int = 128
    tune_audit_per_class: int = 128
    confirm_audit_per_class: int = 32
    checkpoint_steps: tuple[int, ...] = DEFAULT_CHECKPOINT_STEPS
    seed: int = 43
    batch_size: int = 4
    max_source_length: int = 512
    max_target_length: int = 50
    truncation_mode: str = "field_aware"
    min_rescue_gain: float = 0.05
    max_class_drop: float = 0.05
    pair_temperature: float = 1.0
    common_shift_weight: float = 1.0
    safe_hinge_weight: float = 2.0
    safe_margin_cap: float = 0.5
    scale_floor: float = 0.05
    tiny_random_model: bool = False
    dry_run: bool = False
    confirm_selection: str | None = None
    run_pointwise_control: bool = False
    device: str = "cpu"
    fresh_confirm_against_manifests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.task_name not in SUPPORTED_TASKS:
            raise ValueError(f"unsupported BAP rescue task {self.task_name!r}")
        task_labels = set(TASK_BY_NAME[self.task_name].labels)
        if task_labels != set(BINARY_LABELS) or len(TASK_BY_NAME[self.task_name].labels) != 2:
            raise ValueError(
                "BAP-LoRA v1 is explicitly limited to two-label True/False tasks"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        integer_fields = {
            "fit_per_class": self.fit_per_class,
            "calibration_per_class": self.calibration_per_class,
            "tune_audit_per_class": self.tune_audit_per_class,
            "confirm_audit_per_class": self.confirm_audit_per_class,
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
            raise ValueError("BAP paired batches require an even batch_size")
        half_batch = self.batch_size // 2
        if self.fit_per_class % half_batch:
            raise ValueError(
                "fit_per_class must be divisible by batch_size/2 for exact paired cycles"
            )
        if (
            not self.checkpoint_steps
            or tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps
            or any(
                isinstance(step, bool) or not isinstance(step, int) or step <= 0
                for step in self.checkpoint_steps
            )
        ):
            raise ValueError("checkpoint_steps must be unique increasing positive integers")
        finite_positive = {
            "pair_temperature": self.pair_temperature,
            "scale_floor": self.scale_floor,
        }
        if any(not math.isfinite(value) or value <= 0 for value in finite_positive.values()):
            raise ValueError(f"finite positive coefficients required: {finite_positive}")
        finite_nonnegative = {
            "min_rescue_gain": self.min_rescue_gain,
            "max_class_drop": self.max_class_drop,
            "common_shift_weight": self.common_shift_weight,
            "safe_hinge_weight": self.safe_hinge_weight,
            "safe_margin_cap": self.safe_margin_cap,
        }
        if any(
            not math.isfinite(value) or value < 0
            for value in finite_nonnegative.values()
        ):
            raise ValueError(
                f"finite nonnegative coefficients required: {finite_nonnegative}"
            )
        if self.max_class_drop >= 1:
            raise ValueError("max_class_drop must be smaller than one")
        if self.truncation_mode not in TRUNCATION_MODES:
            raise ValueError(
                f"truncation_mode must be one of {TRUNCATION_MODES}"
            )
        if not self.tiny_random_model:
            output = Path(self.output_root).expanduser().resolve()
            for name, protected_value in (
                ("data_root", self.data_root),
                ("model_path", self.model_path),
            ):
                protected = Path(protected_value).expanduser().resolve()
                if (
                    output == protected
                    or output.is_relative_to(protected)
                    or protected.is_relative_to(output)
                ):
                    raise ValueError(
                        f"output_root must not overlap locked {name}: "
                        f"{output} vs {protected}"
                    )

    @property
    def update_per_class(self) -> int:
        return self.fit_per_class + self.calibration_per_class

    @property
    def max_steps(self) -> int:
        return self.checkpoint_steps[-1]


@dataclass(frozen=True)
class BAPInternalSplit:
    fit: tuple[OfficialExample, ...]
    calibration: tuple[OfficialExample, ...]
    group_key_by_example_id: Mapping[str, str]
    selection_algorithm_version: str = rescue.SEMANTIC_SELECTION_ALGORITHM_VERSION

    def assert_valid(self, *, config: BAPConfig) -> None:
        fit_ids = {example.example_id for example in self.fit}
        calibration_ids = {example.example_id for example in self.calibration}
        if fit_ids & calibration_ids:
            raise AssertionError("BAP fit/calibration rows overlap")
        fit_groups = {self.group_key_by_example_id[item] for item in fit_ids}
        calibration_groups = {
            self.group_key_by_example_id[item] for item in calibration_ids
        }
        if fit_groups & calibration_groups:
            raise AssertionError("BAP fit/calibration semantic groups overlap")
        for partition_name, partition, quota in (
            ("fit", self.fit, config.fit_per_class),
            ("calibration", self.calibration, config.calibration_per_class),
        ):
            counts = {
                label: sum(example.label == label for example in partition)
                for label in BINARY_LABELS
            }
            if counts != {label: quota for label in BINARY_LABELS}:
                raise AssertionError(
                    f"BAP {partition_name} exact quotas failed: {counts}"
                )
            if any(
                example.subset != "train" or example.task_name != config.task_name
                for example in partition
            ):
                raise AssertionError("BAP internal split escaped the requested train task")


@dataclass(frozen=True)
class ThresholdFit:
    threshold: float
    metrics: Mapping[str, Any]
    feasible_threshold_count: int
    candidate_threshold_count: int
    constraint_reference_threshold: float | None
    selection_rule: str


def build_bap_internal_split(
    split: rescue.RescueSplit, *, config: BAPConfig
) -> BAPInternalSplit:
    """Split the update pool into group-atomic fit and calibration rows."""

    if split.split_unit != "semantic_group" or split.group_key_by_example_id is None:
        raise ValueError("BAP requires a semantic_group outer rescue split")
    if len(split.update_pool) != 2 * config.update_per_class:
        raise ValueError("outer update pool does not match BAP fit+calibration quotas")
    grouped: dict[str, list[OfficialExample]] = {}
    for example in split.update_pool:
        grouped.setdefault(
            split.group_key_by_example_id[example.example_id], []
        ).append(example)
    groups = {
        key: tuple(sorted(items, key=lambda item: item.source_index))
        for key, items in grouped.items()
    }
    calibration, remaining = rescue._take_exact_group_partition(
        groups,
        task_name=config.task_name,
        labels=BINARY_LABELS,
        quota_per_class=config.calibration_per_class,
        seed=rescue._stable_seed(
            config.seed, config.task_name, "bap-internal-calibration"
        ),
        algorithm_version=rescue.SEMANTIC_SELECTION_ALGORITHM_VERSION,
    )
    fit = tuple(
        sorted(
            (
                example
                for group in remaining.values()
                for example in group
            ),
            key=lambda example: example.source_index,
        )
    )
    result = BAPInternalSplit(
        fit=fit,
        calibration=calibration,
        group_key_by_example_id=split.group_key_by_example_id,
    )
    result.assert_valid(config=config)
    return result


def _partition_ledger(
    examples: Sequence[OfficialExample], group_keys: Mapping[str, str]
) -> dict[str, Any]:
    ids = [example.example_id for example in examples]
    groups = {group_keys[example.example_id] for example in examples}
    return {
        "count": len(examples),
        "per_class": {
            label: sum(example.label == label for example in examples)
            for label in BINARY_LABELS
        },
        "semantic_group_count": len(groups),
        "example_ids": ids,
        "ordered_ids_sha256": _sha256_json(ids),
        "source_subset": "train",
    }


def bap_internal_split_manifest(split: BAPInternalSplit) -> dict[str, Any]:
    fit_groups = {
        split.group_key_by_example_id[example.example_id] for example in split.fit
    }
    calibration_groups = {
        split.group_key_by_example_id[example.example_id]
        for example in split.calibration
    }
    return {
        "format": FORMAT_VERSION,
        "selection_algorithm_version": split.selection_algorithm_version,
        "split_unit": "semantic_group",
        "fit": _partition_ledger(split.fit, split.group_key_by_example_id),
        "internal_calibration": _partition_ledger(
            split.calibration, split.group_key_by_example_id
        ),
        "row_overlap": 0,
        "semantic_group_overlap": len(fit_groups & calibration_groups),
    }


def _binary_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    natural_prior: Mapping[str, float],
) -> dict[str, Any]:
    if len(predictions) != len(references) or not predictions:
        raise ValueError("predictions and references must be non-empty and aligned")
    if set(natural_prior) != set(BINARY_LABELS) or not math.isclose(
        sum(float(value) for value in natural_prior.values()),
        1.0,
        abs_tol=1e-9,
    ):
        raise ValueError("natural prior must cover True/False and sum to one")
    per_class: dict[str, float] = {}
    for label in BINARY_LABELS:
        indices = [index for index, value in enumerate(references) if value == label]
        if not indices:
            raise ValueError(f"metrics partition is missing {label!r}")
        per_class[label] = float(
            sum(predictions[index] == label for index in indices) / len(indices)
        )
    natural = float(
        sum(float(natural_prior[label]) * per_class[label] for label in BINARY_LABELS)
    )
    return {
        "natural_prior_em": natural,
        "balanced_em": float(np.mean(tuple(per_class.values()))),
        "per_class_recall": per_class,
        "worst_class_recall": min(per_class.values()),
        "valid_label_rate": float(
            sum(value in set(BINARY_LABELS) for value in predictions)
            / len(predictions)
        ),
        "count": len(predictions),
    }


def margin_predictions(
    margins: Sequence[float], *, threshold: float
) -> list[str]:
    if not math.isfinite(threshold):
        raise ValueError("deployed threshold must be finite")
    values = np.asarray(margins, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("margins must be a finite non-empty vector")
    return ["True" if float(value) > threshold else "False" for value in values]


def _margin_diagnostics(
    margins: Sequence[float], references: Sequence[str]
) -> dict[str, Any]:
    values = np.asarray(margins, dtype=np.float64)
    labels = np.asarray(references, dtype=object)
    if (
        values.ndim != 1
        or values.size == 0
        or values.size != labels.size
        or not np.isfinite(values).all()
    ):
        raise ValueError("margin diagnostics require finite aligned vectors")
    true_values = values[labels == "True"]
    false_values = values[labels == "False"]
    if not true_values.size or not false_values.size:
        raise ValueError("margin diagnostics require both binary classes")
    pairwise = true_values[:, None] - false_values[None, :]
    auc = float(np.mean((pairwise > 0).astype(np.float64) + 0.5 * (pairwise == 0)))
    return {
        "roc_auc": auc,
        "mean_margin": float(np.mean(values)),
        "mean_margin_by_class": {
            "True": float(np.mean(true_values)),
            "False": float(np.mean(false_values)),
        },
        "mean_true_minus_false_margin": float(
            np.mean(true_values) - np.mean(false_values)
        ),
    }


def _margin_change_diagnostics(
    base_margins: Sequence[float],
    post_margins: Sequence[float],
    references: Sequence[str],
) -> dict[str, Any]:
    base = _margin_diagnostics(base_margins, references)
    post = _margin_diagnostics(post_margins, references)
    base_values = np.asarray(base_margins, dtype=np.float64)
    post_values = np.asarray(post_margins, dtype=np.float64)
    labels = np.asarray(references, dtype=object)
    shifts = post_values - base_values
    return {
        "base": base,
        "post": post,
        "roc_auc_delta": float(post["roc_auc"] - base["roc_auc"]),
        "common_margin_shift": float(np.mean(shifts)),
        "mean_margin_shift_by_class": {
            label: float(np.mean(shifts[labels == label])) for label in BINARY_LABELS
        },
    }


def _threshold_candidates(margins: Sequence[float]) -> tuple[float, ...]:
    values = np.asarray(margins, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("margins must be a finite non-empty vector")
    unique = np.unique(values)
    low = float(np.nextafter(unique[0], -np.inf))
    high = float(np.nextafter(unique[-1], np.inf))
    if not math.isfinite(low):
        low = float(unique[0] - max(1.0, abs(float(unique[0]))))
    if not math.isfinite(high):
        high = float(unique[-1] + max(1.0, abs(float(unique[-1]))))
    return (low, *(float(value) for value in unique), high)


def fit_margin_threshold(
    margins: Sequence[float],
    references: Sequence[str],
    *,
    natural_prior: Mapping[str, float],
    constraint_reference: Mapping[str, float] | None = None,
    max_class_drop: float = 0.05,
    reference_threshold: float | None = None,
) -> ThresholdFit:
    """Fit one deterministic threshold on train-derived calibration rows.

    When ``constraint_reference`` is supplied, every class must retain at least
    its calibrated-base recall minus ``max_class_drop``.  Ties prefer larger
    worst-class recall and then the threshold closest to the reference/base
    threshold; this avoids using an arbitrary extreme decision boundary.
    """

    if len(margins) != len(references):
        raise ValueError("threshold margins/references are not aligned")
    if constraint_reference is not None and set(constraint_reference) != set(
        BINARY_LABELS
    ):
        raise ValueError("constraint reference must cover both classes")
    if not 0 <= max_class_drop < 1:
        raise ValueError("max_class_drop must be in [0, 1)")
    center = 0.0 if reference_threshold is None else float(reference_threshold)
    candidates = _threshold_candidates(margins)
    feasible: list[tuple[tuple[float, float, float, float], float, dict[str, Any]]] = []
    for threshold in candidates:
        metrics = _binary_metrics(
            margin_predictions(margins, threshold=threshold),
            references,
            natural_prior=natural_prior,
        )
        if constraint_reference is not None and any(
            not _at_least(
                float(metrics["per_class_recall"][label]),
                float(constraint_reference[label]) - max_class_drop,
            )
            for label in BINARY_LABELS
        ):
            continue
        key = (
            float(metrics["natural_prior_em"]),
            float(metrics["worst_class_recall"]),
            -abs(float(threshold) - center),
            -float(threshold),
        )
        feasible.append((key, float(threshold), metrics))
    if not feasible:
        raise ValueError("no threshold satisfies the calibrated-base class guards")
    _key, threshold, metrics = max(feasible, key=lambda item: item[0])
    return ThresholdFit(
        threshold=threshold,
        metrics=metrics,
        feasible_threshold_count=len(feasible),
        candidate_threshold_count=len(candidates),
        constraint_reference_threshold=(
            None if reference_threshold is None else float(reference_threshold)
        ),
        selection_rule=(
            "maximize pinned-natural-prior EM; tie-break worst-class recall, "
            "distance to reference threshold, then lower threshold"
        ),
    )


def calibrated_training_comparison(
    base_margins: Sequence[float],
    post_margins: Sequence[float],
    references: Sequence[str],
    *,
    natural_prior: Mapping[str, float],
    max_class_drop: float,
) -> dict[str, Any]:
    """Fit base and BAP identically, then isolate the training contribution."""

    if not 0 <= max_class_drop < 1:
        raise ValueError("max_class_drop must be in [0, 1)")

    base_fit = fit_margin_threshold(
        base_margins, references, natural_prior=natural_prior
    )
    post_fit = fit_margin_threshold(
        post_margins,
        references,
        natural_prior=natural_prior,
    )
    deltas = {
        label: float(post_fit.metrics["per_class_recall"][label])
        - float(base_fit.metrics["per_class_recall"][label])
        for label in BINARY_LABELS
    }
    return {
        "base_same_calibration_control": asdict(base_fit),
        "bap_calibrated": asdict(post_fit),
        "training_contribution_natural_prior": float(
            post_fit.metrics["natural_prior_em"]
            - base_fit.metrics["natural_prior_em"]
        ),
        "training_contribution_balanced": float(
            post_fit.metrics["balanced_em"] - base_fit.metrics["balanced_em"]
        ),
        "training_per_class_recall_delta": deltas,
        "training_min_per_class_delta": min(deltas.values()),
        "margin_diagnostics": _margin_change_diagnostics(
            base_margins, post_margins, references
        ),
        "denominator": "frozen base with the exact same exhaustive threshold fitter",
        "class_safety_applied_after_identical_calibration": True,
        "calibration_gain_is_not_training_contribution": True,
    }


def apply_frozen_thresholds(
    base_margins: Sequence[float],
    post_margins: Sequence[float],
    references: Sequence[str],
    *,
    natural_prior: Mapping[str, float],
    base_threshold: float,
    post_threshold: float,
) -> dict[str, Any]:
    base_metrics = _binary_metrics(
        margin_predictions(base_margins, threshold=base_threshold),
        references,
        natural_prior=natural_prior,
    )
    post_metrics = _binary_metrics(
        margin_predictions(post_margins, threshold=post_threshold),
        references,
        natural_prior=natural_prior,
    )
    deltas = {
        label: float(post_metrics["per_class_recall"][label])
        - float(base_metrics["per_class_recall"][label])
        for label in BINARY_LABELS
    }
    return {
        "base_same_calibration_control": {
            "threshold": float(base_threshold),
            "metrics": base_metrics,
        },
        "bap_calibrated": {
            "threshold": float(post_threshold),
            "metrics": post_metrics,
        },
        "training_contribution_natural_prior": float(
            post_metrics["natural_prior_em"] - base_metrics["natural_prior_em"]
        ),
        "training_contribution_balanced": float(
            post_metrics["balanced_em"] - base_metrics["balanced_em"]
        ),
        "training_per_class_recall_delta": deltas,
        "training_min_per_class_delta": min(deltas.values()),
        "margin_diagnostics": _margin_change_diagnostics(
            base_margins, post_margins, references
        ),
        "thresholds_frozen_before_this_partition": True,
        "calibration_gain_is_not_training_contribution": True,
    }


def qualification_gate(
    comparison: Mapping[str, Any], *, min_gain: float, max_class_drop: float
) -> bool:
    """Gate calibrated training gain; label validity is true by construction."""

    return (
        _strictly_above(
            float(comparison["training_contribution_natural_prior"]), min_gain
        )
        and _at_least(
            float(comparison["training_min_per_class_delta"]), -max_class_drop
        )
    )


def robust_margin_scale(values: Sequence[float], *, floor: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("base margins must be a finite non-empty vector")
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("scale floor must be finite and positive")
    q25, q75 = np.percentile(array, [25, 75])
    return float(max((q75 - q25) / 1.349, floor))


def bap_loss_components(
    current_margins: Tensor,
    base_margins: Tensor,
    labels: Sequence[str],
    *,
    base_threshold: float,
    margin_scale: float,
    pair_temperature: float,
    safe_margin_cap: float,
) -> dict[str, Tensor]:
    """Return translation-invariant rank loss and two boundary guards."""

    if current_margins.ndim != 1 or base_margins.shape != current_margins.shape:
        raise ValueError("current/base margins must be aligned vectors")
    if len(labels) != current_margins.numel() or set(labels) != set(BINARY_LABELS):
        raise ValueError("each BAP batch must contain both True and False rows")
    if not math.isfinite(margin_scale) or margin_scale <= 0:
        raise ValueError("margin_scale must be finite and positive")
    if not math.isfinite(pair_temperature) or pair_temperature <= 0:
        raise ValueError("pair_temperature must be finite and positive")
    if not math.isfinite(safe_margin_cap) or safe_margin_cap < 0:
        raise ValueError("safe_margin_cap must be finite and nonnegative")
    true_indices = torch.tensor(
        [label == "True" for label in labels],
        dtype=torch.bool,
        device=current_margins.device,
    )
    false_indices = ~true_indices
    if int(true_indices.sum()) != int(false_indices.sum()):
        raise ValueError("BAP pairwise loss requires equal True/False batch counts")
    temperature = margin_scale * pair_temperature
    # Use the complete within-batch True x False U-statistic.  A zipped pairing
    # would make the objective depend on an arbitrary row ordering and discard
    # valid RankNet comparisons when the half-batch is larger than one.
    pair_differences = (
        current_margins[true_indices, None]
        - current_margins[false_indices][None, :]
    )
    pairwise_rank = F.softplus(-pair_differences / temperature).mean()
    common_shift = torch.square(
        torch.mean(current_margins - base_margins) / margin_scale
    )
    signed = torch.where(
        true_indices,
        torch.ones_like(current_margins),
        -torch.ones_like(current_margins),
    )
    base_signed = signed * (base_margins - float(base_threshold))
    current_signed = signed * (current_margins - float(base_threshold))
    # Deployment predicts True for m > threshold and False otherwise, so a
    # False example exactly on the threshold is base-correct by definition.
    base_correct = torch.where(
        true_indices,
        base_margins > float(base_threshold),
        base_margins <= float(base_threshold),
    )
    class_safe_losses: list[Tensor] = []
    for class_mask in (true_indices, false_indices):
        protected = class_mask & base_correct
        if bool(protected.any()):
            required = torch.clamp(
                base_signed[protected],
                min=0.0,
                max=float(safe_margin_cap) * margin_scale,
            )
            violations = F.relu(required - current_signed[protected]) / margin_scale
            class_safe_losses.append(torch.square(violations).mean())
    if class_safe_losses:
        worst_class_safe = torch.stack(class_safe_losses).max()
    else:
        worst_class_safe = current_margins.sum() * 0.0
    return {
        "pairwise_rank": pairwise_rank,
        "common_shift_anchor": common_shift,
        "worst_class_base_correct_hinge": worst_class_safe,
        "base_correct_count": base_correct.to(current_margins.dtype).sum(),
    }


def bap_total_loss(
    components: Mapping[str, Tensor],
    *,
    common_shift_weight: float,
    safe_hinge_weight: float,
) -> Tensor:
    return (
        components["pairwise_rank"]
        + float(common_shift_weight) * components["common_shift_anchor"]
        + float(safe_hinge_weight)
        * components["worst_class_base_correct_hinge"]
    )


def _chunks(
    values: Sequence[OfficialExample], size: int
) -> Iterator[tuple[OfficialExample, ...]]:
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])


def _label_margin_batch(
    bank: Any,
    tokenizer: Any,
    examples: Sequence[OfficialExample],
    *,
    config: BAPConfig,
) -> tuple[Tensor, dict[str, int]]:
    """Compute differentiable True-vs-False sequence-mean margins."""

    if not examples:
        raise ValueError("cannot score an empty BAP batch")
    repeated: list[OfficialExample] = []
    targets: list[str] = []
    for example in examples:
        if example.label not in BINARY_LABELS:
            raise ValueError("BAP encountered a non-binary label")
        for label in BINARY_LABELS:
            repeated.append(example)
            targets.append(label)
    encoded = rescue._encode_source_inputs(
        tokenizer,
        repeated,
        max_source_length=config.max_source_length,
        truncation_mode=config.truncation_mode,
    )
    encoded_targets = tokenizer(
        text_target=targets,
        padding=True,
        truncation=True,
        max_length=config.max_target_length,
        return_tensors="pt",
    )
    target_ids = encoded_targets["input_ids"]
    target_ids = target_ids.masked_fill(target_ids.eq(tokenizer.pad_token_id), -100)
    device = torch.device(config.device)
    inputs = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "labels": target_ids.to(device),
    }
    output = bank(**inputs)
    nll = per_example_target_token_nll(output.logits, inputs["labels"])
    matrix = nll.reshape(len(examples), len(BINARY_LABELS))
    true_index = BINARY_LABELS.index("True")
    false_index = BINARY_LABELS.index("False")
    margins = matrix[:, false_index] - matrix[:, true_index]
    return margins, {
        "gradient_examples": len(examples),
        "scored_label_sequences": len(repeated),
        "source_tokens_processed": int(encoded["attention_mask"].sum()),
        "target_tokens_processed": int(target_ids.ne(-100).sum()),
    }


def collect_label_margins(
    bank: Any,
    tokenizer: Any,
    examples: Sequence[OfficialExample],
    *,
    config: BAPConfig,
) -> list[float]:
    if not examples or any(example.subset != "train" for example in examples):
        raise ValueError("BAP margin audits must be non-empty and train-derived")
    was_training = bool(bank.training)
    bank.eval()
    output: list[float] = []
    with torch.no_grad():
        for batch in _chunks(examples, config.batch_size):
            margins, _ledger = _label_margin_batch(
                bank, tokenizer, batch, config=config
            )
            output.extend(float(value) for value in margins.detach().cpu())
    bank.train(was_training)
    return output


def paired_fit_batches(
    examples: Sequence[OfficialExample],
    *,
    batch_size: int,
    steps: int,
    seed: int,
) -> Iterator[tuple[int, tuple[OfficialExample, ...]]]:
    """Yield deterministic, exactly balanced batches for a fixed step budget."""

    if batch_size <= 0 or batch_size % 2:
        raise ValueError("paired batch_size must be positive and even")
    if steps <= 0:
        raise ValueError("paired schedule requires positive steps")
    half = batch_size // 2
    by_label = {
        label: tuple(example for example in examples if example.label == label)
        for label in BINARY_LABELS
    }
    if len(by_label["True"]) != len(by_label["False"]):
        raise ValueError("paired fit partition must be class balanced")
    if not by_label["True"] or len(by_label["True"]) % half:
        raise ValueError("per-class fit rows must divide exactly into half-batches")
    steps_per_cycle = len(by_label["True"]) // half
    for step_index in range(steps):
        cycle = step_index // steps_per_cycle
        within = step_index % steps_per_cycle
        selected: dict[str, list[OfficialExample]] = {}
        for label in BINARY_LABELS:
            rng = np.random.Generator(
                np.random.PCG64(rescue._stable_seed(seed, "bap-pairs", cycle, label))
            )
            order = rng.permutation(len(by_label[label])).tolist()
            start = within * half
            selected[label] = [
                by_label[label][int(index)] for index in order[start : start + half]
            ]
        interleaved: list[OfficialExample] = []
        for local_index in range(half):
            interleaved.extend(
                [selected["True"][local_index], selected["False"][local_index]]
            )
        yield step_index + 1, tuple(interleaved)


def _raw_audit(
    bank: Any,
    tokenizer: Any,
    examples: Sequence[OfficialExample],
    *,
    config: BAPConfig,
    natural_prior: Mapping[str, float],
) -> dict[str, Any]:
    return rescue.evaluate_train_audit(
        bank,
        tokenizer,
        examples,
        config=config,  # Rescue evaluator uses this config structurally.
        natural_label_prior=natural_prior,
    )


def _raw_gain_decomposition(
    base_raw: Mapping[str, Any],
    post_raw: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    base_cal = comparison["base_same_calibration_control"]["metrics"]
    post_cal = comparison["bap_calibrated"]["metrics"]
    return {
        "raw_free_generation": {
            "base": dict(base_raw),
            "post": dict(post_raw),
            "natural_prior_gain": float(
                post_raw["natural_prior_em"] - base_raw["natural_prior_em"]
            ),
            "balanced_gain": float(
                post_raw["balanced_em"] - base_raw["balanced_em"]
            ),
        },
        "calibration_only_gain": {
            "base_natural_prior": float(
                base_cal["natural_prior_em"] - base_raw["natural_prior_em"]
            ),
            "post_natural_prior": float(
                post_cal["natural_prior_em"] - post_raw["natural_prior_em"]
            ),
            "not_counted_as_training_contribution": True,
        },
        "bap_training_contribution": {
            "natural_prior": float(
                comparison["training_contribution_natural_prior"]
            ),
            "balanced": float(comparison["training_contribution_balanced"]),
            "per_class_recall_delta": dict(
                comparison["training_per_class_recall_delta"]
            ),
            "minimum_per_class_delta": float(
                comparison["training_min_per_class_delta"]
            ),
            "denominator": "frozen base with the exact same calibration procedure",
        },
    }


def _initial_resource_ledger(config: BAPConfig) -> dict[str, Any]:
    return {
        "optimizer_steps": 0,
        "gradient_example_exposures": 0,
        "paired_exposures_per_class": {label: 0 for label in BINARY_LABELS},
        "scored_label_sequences": 0,
        "source_tokens_processed": 0,
        "target_tokens_processed": 0,
        "fit_rows_per_class": config.fit_per_class,
        "batch_size": config.batch_size,
        "paired_schedule_seed": config.seed,
        "gradient_forward_mode": "eval (gradients enabled; dropout disabled)",
        "anchor_scoring_mode_matched": True,
        "ordered_exposure_ids_sha256": None,
    }


def _train_bap_trajectory(
    bank: Any,
    tokenizer: Any,
    internal_split: BAPInternalSplit,
    base_fit_margins: Mapping[str, float],
    base_calibration_margins: Sequence[float],
    base_calibration_raw: Mapping[str, Any],
    *,
    config: BAPConfig,
    natural_prior: Mapping[str, float],
    task_dir: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    """Train one shared trajectory and stop at the first internal gate pass."""

    fit_margin_values = [base_fit_margins[item.example_id] for item in internal_split.fit]
    margin_scale = robust_margin_scale(fit_margin_values, floor=config.scale_floor)
    # The safety boundary used inside the gradient objective must itself be
    # fit-derived.  Internal calibration is reserved for threshold fitting and
    # early stopping after optimizer steps; allowing its threshold to enter the
    # hinge would make those held-out rows influence the learned parameters.
    base_threshold_fit = fit_margin_threshold(
        fit_margin_values,
        [example.label for example in internal_split.fit],
        natural_prior=natural_prior,
    )
    optimizer = torch.optim.AdamW(
        tuple(bank.all_atom_parameters()),
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    checkpoint_set = set(config.checkpoint_steps)
    checkpoint_records: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    ledger = _initial_resource_ledger(config)
    loss_accumulator = {
        "total": [],
        "pairwise_rank": [],
        "common_shift_anchor": [],
        "worst_class_base_correct_hinge": [],
    }
    exposure_ids: list[str] = []
    # Gradients remain enabled in eval mode.  Disabling T5/LoRA dropout is
    # essential here: both candidate labels must share a deterministic scoring
    # function, and the cached base anchors were also measured in eval mode.
    bank.eval()
    for step, batch in paired_fit_batches(
        internal_split.fit,
        batch_size=config.batch_size,
        steps=config.max_steps,
        seed=config.seed,
    ):
        current, step_ledger = _label_margin_batch(
            bank, tokenizer, batch, config=config
        )
        base_values = torch.tensor(
            [base_fit_margins[example.example_id] for example in batch],
            dtype=current.dtype,
            device=current.device,
        )
        components = bap_loss_components(
            current,
            base_values,
            [example.label for example in batch],
            base_threshold=base_threshold_fit.threshold,
            margin_scale=margin_scale,
            pair_temperature=config.pair_temperature,
            safe_margin_cap=config.safe_margin_cap,
        )
        loss = bap_total_loss(
            components,
            common_shift_weight=config.common_shift_weight,
            safe_hinge_weight=config.safe_hinge_weight,
        )
        if loss.ndim != 0 or not bool(torch.isfinite(loss.detach())):
            raise RuntimeError("BAP objective produced a non-finite scalar")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tuple(bank.active_atom_parameters()), 1.0)
        optimizer.step()
        bank.assert_exact_budget()
        ledger["optimizer_steps"] += 1
        ledger["gradient_example_exposures"] += len(batch)
        ledger["scored_label_sequences"] += step_ledger["scored_label_sequences"]
        ledger["source_tokens_processed"] += step_ledger["source_tokens_processed"]
        ledger["target_tokens_processed"] += step_ledger["target_tokens_processed"]
        for example in batch:
            ledger["paired_exposures_per_class"][example.label] += 1
            exposure_ids.append(example.example_id)
        loss_accumulator["total"].append(float(loss.detach().cpu()))
        for name in (
            "pairwise_rank",
            "common_shift_anchor",
            "worst_class_base_correct_hinge",
        ):
            loss_accumulator[name].append(float(components[name].detach().cpu()))
        if step not in checkpoint_set:
            continue
        post_calibration_margins = collect_label_margins(
            bank, tokenizer, internal_split.calibration, config=config
        )
        comparison = calibrated_training_comparison(
            base_calibration_margins,
            post_calibration_margins,
            [example.label for example in internal_split.calibration],
            natural_prior=natural_prior,
            max_class_drop=config.max_class_drop,
        )
        gate_pass = qualification_gate(
            comparison,
            min_gain=config.min_rescue_gain,
            max_class_drop=config.max_class_drop,
        )
        record = {
            "step": step,
            "internal_calibration": comparison,
            "internal_gate_pass": gate_pass,
            "fit_derived_safety_threshold": float(base_threshold_fit.threshold),
            "safety_threshold_source": "fit only; internal calibration excluded",
            "mean_losses_through_step": {
                name: float(np.mean(values))
                for name, values in loss_accumulator.items()
            },
            "margin_scale": margin_scale,
            "raw_free_generation_not_used_for_checkpoint_selection": True,
            "confirm_audit_accessed": False,
        }
        checkpoint_records.append(record)
        rescue._atomic_json(
            task_dir / "checkpoint_metrics" / f"step_{step:06d}.json", record
        )
        if gate_pass:
            selected_dir = task_dir / "selected"
            selected_dir.mkdir(parents=True, exist_ok=False)
            checkpoint_path = selected_dir / "adapter.pt"
            rescue._atomic_torch_save(checkpoint_path, bank.compact_state_dict())
            post_calibration_raw = _raw_audit(
                bank,
                tokenizer,
                internal_split.calibration,
                config=config,
                natural_prior=natural_prior,
            )
            comparison_with_raw = {
                **comparison,
                "gain_decomposition": _raw_gain_decomposition(
                    base_calibration_raw, post_calibration_raw, comparison
                ),
            }
            selected = {
                "step": step,
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": rescue._sha256_file(checkpoint_path),
                "internal_calibration": comparison_with_raw,
                "base_threshold": float(
                    comparison["base_same_calibration_control"]["threshold"]
                ),
                "bap_threshold": float(
                    comparison["bap_calibrated"]["threshold"]
                ),
                "fit_derived_safety_threshold": float(
                    base_threshold_fit.threshold
                ),
                "selection_rule": "first pre-gradient-declared optimizer-step checkpoint passing the internal gate",
                "selected_without_development_tune": True,
                "confirm_audit_accessed": False,
            }
            break
    ledger["ordered_exposure_ids_sha256"] = _sha256_json(exposure_ids)
    ledger["selected_step"] = None if selected is None else selected["step"]
    ledger["exact_equal_class_exposures"] = (
        ledger["paired_exposures_per_class"]["True"]
        == ledger["paired_exposures_per_class"]["False"]
    )
    return selected, checkpoint_records, ledger


def _model_config(config: BAPConfig) -> rescue.RescueConfig:
    """Create the structural config expected by the shared model builder."""

    return rescue.RescueConfig(
        data_root=config.data_root,
        model_path=config.model_path,
        output_root=config.output_root,
        task_names=(config.task_name,),
        learning_rates=(config.learning_rate,),
        epochs=(1,),
        caps_per_class=(config.update_per_class,),
        tune_audit_per_class=config.tune_audit_per_class,
        confirm_audit_per_class=config.confirm_audit_per_class,
        seed=config.seed,
        batch_size=config.batch_size,
        max_source_length=config.max_source_length,
        max_target_length=config.max_target_length,
        truncation_mode=config.truncation_mode,
        min_rescue_gain=config.min_rescue_gain,
        tiny_random_model=config.tiny_random_model,
        selection_only=True,
        confirm_selection=config.confirm_selection,
        device=config.device,
        split_unit="semantic_group",
        fresh_confirm_against_manifests=config.fresh_confirm_against_manifests,
        loss_objectives=("sequence_mean_balanced",),
        update_sampler="balanced",
    )


def _build_outer_split(
    train_examples: Sequence[OfficialExample], *, config: BAPConfig
) -> tuple[rescue.RescueSplit, tuple[Mapping[str, Any], ...]]:
    touched_ids, sources = rescue.load_prior_touched_ids(
        config.fresh_confirm_against_manifests,
        task_name=config.task_name,
        train_examples=train_examples,
    )
    split = rescue.build_rescue_split(
        train_examples,
        task_name=config.task_name,
        max_update_per_class=config.update_per_class,
        tune_audit_per_class=config.tune_audit_per_class,
        confirm_audit_per_class=config.confirm_audit_per_class,
        seed=rescue._stable_seed(config.seed, config.task_name, "three-way-split"),
        split_unit="semantic_group",
        semantic_selection_algorithm=rescue.SEMANTIC_SELECTION_ALGORITHM_VERSION,
        prior_touched_example_ids=touched_ids,
        freshness_sources=sources,
    )
    return split, sources


def _protocol_payload(config: BAPConfig) -> dict[str, Any]:
    return {
        "format": FORMAT_VERSION,
        "claim_scope": "binary train-only acquisition qualification; not continual retention",
        "binary_only": True,
        "score": "sequence-mean NLL(False) - sequence-mean NLL(True)",
        "objective": {
            "pairwise_rank": "softplus(-(margin_true-margin_false)/(robust_scale*temperature))",
            "translation_invariant": True,
            "common_shift_anchor": "square(mean(post_margin-base_margin)/robust_scale)",
            "worst_class_safety": "max classwise squared hinge on base-correct fit rows",
            "gradient_forward_mode": (
                "eval with autograd enabled; dropout disabled to match frozen anchors"
            ),
            "coefficients": {
                "pair_temperature": config.pair_temperature,
                "common_shift_weight": config.common_shift_weight,
                "safe_hinge_weight": config.safe_hinge_weight,
                "safe_margin_cap_in_robust_scales": config.safe_margin_cap,
                "status": (
                    "fixed in the run-local pre-gradient protocol; no coefficient "
                    "scan and no external preregistration claim"
                ),
            },
        },
        "partition_roles": {
            "fit": "gradient only",
            "internal_calibration": "threshold plus first-pass step stopping only",
            "development_tune": (
                "post-selection second qualification gate; cannot select step/threshold"
            ),
            "confirm": "sealed until independent durable-selection confirmation",
        },
        "gate": {
            "numerator": "BAP calibrated label-output",
            "denominator": "frozen base with the exact same calibration fitter",
            "natural_prior_gain": f"strictly greater than {config.min_rescue_gain}",
            "minimum_per_class_delta": f">= {-config.max_class_drop}",
            "calibrated_valid_label_rate": (
                "1 by construction for the binary threshold rule; reported, not gated"
            ),
            "calibration_gain_counted_as_training": False,
            "raw_free_generation_counted_as_training": False,
            "positive_claim_scope": (
                "calibrated official-label decision only; raw free generation "
                "is a separate diagnostic"
            ),
        },
        "checkpoint_steps": list(config.checkpoint_steps),
        "checkpoint_rule": "first passing checkpoint; no best-of-checkpoints maximization",
        "optional_post_selection_control": {
            "objective": CONTROL_OBJECTIVE,
            "enabled": config.run_pointwise_control,
            "selection_eligible": False,
            "conditional_on_bap_selected_step": True,
            "independently_tuned": False,
            "claim_scope": (
                "local objective comparison under the BAP-chosen schedule; "
                "not a claim against an independently tuned pointwise optimum"
            ),
            "matching": "same optimizer steps and paired example exposures",
            "forward_compute_matched": False,
            "compute_caveat": (
                "BAP scores both official labels per example; pointwise scores "
                "the observed label once"
            ),
        },
    }


def _run_local_protocol_lock_payload(config: BAPConfig) -> dict[str, Any]:
    """Declare every arm/candidate fixed inside this run before gradients."""

    payload = {
        "format": FORMAT_VERSION,
        "lock_scope": "run-local, pre-gradient; not an external preregistration",
        "external_preregistration_claimed": False,
        "scientific_config": asdict(config),
        "pilot_inventory": {
            "selection_eligible_arms": [
                {
                    "name": "BAP-LoRA",
                    "learning_rate": config.learning_rate,
                    "coefficient_set": {
                        "pair_temperature": config.pair_temperature,
                        "common_shift_weight": config.common_shift_weight,
                        "safe_hinge_weight": config.safe_hinge_weight,
                        "safe_margin_cap": config.safe_margin_cap,
                        "scale_floor": config.scale_floor,
                    },
                    "coefficient_scan": False,
                    "checkpoint_steps": list(config.checkpoint_steps),
                    "selection_rule": "first passing checkpoint",
                }
            ],
            "post_selection_audit_arms": (
                [
                    {
                        "name": CONTROL_OBJECTIVE,
                        "conditional_on_bap_selected_step": True,
                        "independently_tuned": False,
                        "selection_eligible": False,
                    }
                ]
                if config.run_pointwise_control
                else []
            ),
            "run_local_coefficient_candidate_count": 1,
            "run_local_learning_rate_candidate_count": 1,
            "external_or_prior_pilots_not_governed_by_this_lock": True,
        },
        "protocol": _protocol_payload(config),
    }
    return {**payload, "payload_sha256": _sha256_json(payload)}


def _dry_run_payload(
    config: BAPConfig,
    split: rescue.RescueSplit,
    internal: BAPInternalSplit,
    distribution: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": FORMAT_VERSION,
        "status": "dry_run_complete",
        "config": asdict(config),
        "protocol": _protocol_payload(config),
        "outer_split_sha256": rescue._atomic_json_payload_sha256(
            rescue._split_manifest(split)
        ),
        "internal_split": bap_internal_split_manifest(internal),
        "pinned_train_label_distribution": dict(distribution),
        "model_or_tokenizer_constructed": False,
        "gradient_steps": 0,
        "tune_audit_accessed": False,
        "confirm_audit_accessed": False,
    }


def run_selection(
    config: BAPConfig,
    *,
    train_examples: Sequence[OfficialExample] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Run dry planning or one BAP selection/qualification trajectory."""

    if config.confirm_selection is not None:
        raise ValueError("run_selection cannot execute confirmation mode")
    output_root = Path(config.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing non-empty BAP output root {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    task_dir = output_root / config.task_name
    task_dir.mkdir(parents=True, exist_ok=False)
    train_examples_injected = train_examples is not None
    if train_examples_injected and not config.tiny_random_model:
        raise ValueError(
            "formal BAP selection must load verified official data; "
            "train_examples injection is tiny-test-only"
        )
    if train_examples is None:
        train_examples = load_official_examples(
            config.data_root, config.task_name, "train", verify=True
        )
    data_artifact = (
        None
        if train_examples_injected
        else _official_train_artifact_fingerprint(
            config.data_root, task_name=config.task_name
        )
    )
    distribution = rescue.pinned_train_label_distribution(
        train_examples, task_name=config.task_name
    )
    train_corpus_fingerprint = _train_corpus_fingerprint(
        train_examples, task_name=config.task_name
    )
    execution_environment = _execution_environment_fingerprint(config.device)
    natural_prior = distribution["natural_label_prior"]
    split, freshness_sources = _build_outer_split(train_examples, config=config)
    internal = build_bap_internal_split(split, config=config)
    outer_manifest = rescue._split_manifest(split)
    internal_manifest = bap_internal_split_manifest(internal)
    rescue._atomic_json(task_dir / "split.json", outer_manifest)
    rescue._atomic_json(task_dir / "internal_split.json", internal_manifest)
    protocol_lock_path = task_dir / "protocol_lock.json"
    protocol_lock_payload = _run_local_protocol_lock_payload(config)
    rescue._atomic_json(protocol_lock_path, protocol_lock_payload)
    protocol_lock = {
        "path": str(protocol_lock_path.resolve()),
        "sha256": rescue._sha256_file(protocol_lock_path),
        "payload_sha256": protocol_lock_payload["payload_sha256"],
        "created_before_model_or_gradients": True,
        "external_preregistration_claimed": False,
    }
    manifest = {
        "format": FORMAT_VERSION,
        "status": "running",
        "config": asdict(config),
        "protocol": _protocol_payload(config),
        "outer_split_sha256": rescue._atomic_json_payload_sha256(outer_manifest),
        "internal_split_sha256": rescue._atomic_json_payload_sha256(
            internal_manifest
        ),
        "freshness_sources": [dict(item) for item in freshness_sources],
        "pinned_train_label_distribution": distribution,
        "train_corpus_fingerprint": train_corpus_fingerprint,
        "execution_environment": execution_environment,
        "run_local_protocol_lock": protocol_lock,
        "verified_data_artifact": data_artifact,
        "train_examples_injected_for_tiny_test": train_examples_injected,
        "tune_audit_accessed": False,
        "confirm_audit_accessed": False,
    }
    rescue._atomic_json(output_root / "manifest.json", manifest)
    if config.dry_run:
        payload = _dry_run_payload(config, split, internal, distribution)
        rescue._atomic_json(output_root / "dry_run.json", payload)
        manifest["status"] = "dry_run_complete"
        rescue._atomic_json(output_root / "manifest.json", manifest)
        (output_root / "DRY_RUN_COMPLETE").write_text(
            "BAP dry run completed; no model/tokenizer/confirm scoring\n",
            encoding="utf-8",
        )
        return payload
    tokenizer_injected = tokenizer is not None
    if tokenizer_injected and not config.tiny_random_model:
        raise ValueError(
            "formal BAP selection must load the locked local tokenizer; "
            "tokenizer injection is tiny-test-only"
        )
    model_tokenizer_artifact = _local_artifact_fingerprint(config.model_path)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, use_fast=True, local_files_only=True
        )
    tokenizer_fingerprint = _tokenizer_protocol_fingerprint(tokenizer)
    model_config = _model_config(config)
    base_bank = rescue._build_model(model_config, tokenizer)
    initial_fingerprint = rescue._adapter_fingerprint(base_bank)
    payload_audit = base_bank.payload_audit().to_dict()
    if not payload_audit["exact_budget"]:
        raise RuntimeError("BAP base model violates the fixed adapter budget")
    base_fit_values = collect_label_margins(
        base_bank, tokenizer, internal.fit, config=config
    )
    base_fit_margins = {
        example.example_id: value
        for example, value in zip(internal.fit, base_fit_values)
    }
    base_calibration_margins = collect_label_margins(
        base_bank, tokenizer, internal.calibration, config=config
    )
    base_calibration_raw = _raw_audit(
        base_bank,
        tokenizer,
        internal.calibration,
        config=config,
        natural_prior=natural_prior,
    )
    base_cache = {
        "fit_margin_sha256": _sha256_float_rows(
            [example.example_id for example in internal.fit], base_fit_values
        ),
        "internal_calibration_margin_sha256": _sha256_float_rows(
            [example.example_id for example in internal.calibration],
            base_calibration_margins,
        ),
        "internal_calibration_raw": base_calibration_raw,
        "individual_fit_margins_persisted": False,
        "confirm_audit_accessed": False,
    }
    rescue._atomic_json(task_dir / "base_internal_calibration.json", base_cache)
    selected, checkpoint_records, ledger = _train_bap_trajectory(
        base_bank,
        tokenizer,
        internal,
        base_fit_margins,
        base_calibration_margins,
        base_calibration_raw,
        config=config,
        natural_prior=natural_prior,
        task_dir=task_dir,
    )
    if selected is None:
        failure = {
            "format": FORMAT_VERSION,
            "status": "qualification_failed",
            "task": config.task_name,
            "checkpoint_records": checkpoint_records,
            "resource_ledger": ledger,
            "tune_audit_accessed": False,
            "confirm_audit_accessed": False,
            "reason": "no pre-gradient-declared optimizer-step checkpoint passed internal calibration",
        }
        rescue._atomic_json(task_dir / "qualification_failure.json", failure)
        manifest["status"] = "qualification_failed"
        manifest["resource_ledger"] = ledger
        rescue._atomic_json(output_root / "manifest.json", manifest)
        (output_root / "QUALIFICATION_FAILED").write_text(
            "no BAP checkpoint qualified; tune and confirm remained unaccessed\n",
            encoding="utf-8",
        )
        rescue._release_model(base_bank)
        return failure
    tune_base_bank = rescue._build_model(model_config, tokenizer)
    if rescue._adapter_fingerprint(tune_base_bank) != initial_fingerprint:
        raise RuntimeError("frozen tune-base adapter initialization drifted")
    base_tune_raw = _raw_audit(
        tune_base_bank,
        tokenizer,
        split.tune_audit,
        config=config,
        natural_prior=natural_prior,
    )
    # The selected bank is still resident at the exact selected step.
    post_tune_raw = _raw_audit(
        base_bank,
        tokenizer,
        split.tune_audit,
        config=config,
        natural_prior=natural_prior,
    )
    tune_base_margins = collect_label_margins(
        tune_base_bank, tokenizer, split.tune_audit, config=config
    )
    rescue._release_model(tune_base_bank)
    tune_post_margins = collect_label_margins(
        base_bank, tokenizer, split.tune_audit, config=config
    )
    tune_comparison = apply_frozen_thresholds(
        tune_base_margins,
        tune_post_margins,
        [example.label for example in split.tune_audit],
        natural_prior=natural_prior,
        base_threshold=selected["base_threshold"],
        post_threshold=selected["bap_threshold"],
    )
    development_pass = qualification_gate(
        tune_comparison,
        min_gain=config.min_rescue_gain,
        max_class_drop=config.max_class_drop,
    )
    tune_decomposition = _raw_gain_decomposition(
        base_tune_raw, post_tune_raw, tune_comparison
    )
    selection = {
        "format": FORMAT_VERSION,
        "status": "selection_locked",
        "task": config.task_name,
        "scientific_config": asdict(config),
        "protocol": _protocol_payload(config),
        "selected": selected,
        "selected_checkpoint": {
            "path": selected["checkpoint"],
            "sha256": selected["checkpoint_sha256"],
        },
        "checkpoint_records": checkpoint_records,
        "resource_ledger": ledger,
        "internal_qualification_pass": True,
        "development_tune": {
            "role": "post-selection second qualification gate",
            "thresholds_frozen_before_access": True,
            "comparison": tune_comparison,
            "gain_decomposition": tune_decomposition,
            "second_qualification_pass": development_pass,
            "did_not_select_checkpoint_or_threshold": True,
        },
        "confirm_eligible": development_pass,
        "confirm_ineligibility_reason": (
            None
            if development_pass
            else "post-selection development qualification gate did not pass"
        ),
        "outer_split_manifest_sha256": rescue._atomic_json_payload_sha256(
            outer_manifest
        ),
        "internal_split_manifest_sha256": rescue._atomic_json_payload_sha256(
            internal_manifest
        ),
        "freshness_sources": [dict(item) for item in freshness_sources],
        "pinned_train_label_distribution": distribution,
        "train_corpus_fingerprint": train_corpus_fingerprint,
        "execution_environment": execution_environment,
        "run_local_protocol_lock": protocol_lock,
        "verified_data_artifact": data_artifact,
        "train_examples_injected_for_tiny_test": train_examples_injected,
        "model_protocol": {
            "initial_adapter_fingerprint": initial_fingerprint,
            "frozen_model_tokenizer_artifact": model_tokenizer_artifact,
            "tokenizer_protocol_fingerprint": tokenizer_fingerprint,
            "tokenizer_injected_for_tiny_test": tokenizer_injected,
            "payload": payload_audit,
        },
        "tune_audit_accessed_after_internal_selection": True,
        "confirm_audit_accessed": False,
        "pointwise_control": {
            "enabled": config.run_pointwise_control,
            "status": "not_run" if config.run_pointwise_control else "disabled",
            "selection_eligible": False,
        },
    }
    selection_path = task_dir / "selection.json"
    rescue._atomic_json(selection_path, selection)
    selection_sha = rescue._sha256_file(selection_path)
    selection_lock = {
        "format": FORMAT_VERSION,
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": selection_sha,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_step": selected["step"],
        "confirm_eligible": development_pass,
        "confirm_audit_accessed": False,
    }
    rescue._atomic_json(task_dir / "selection_lock.json", selection_lock)
    # The optional pointwise audit arm is implemented below and is deliberately
    # invoked only after the selection file and lock are durable.
    pointwise_bundle = None
    if config.run_pointwise_control:
        pointwise_bundle = run_optional_pointwise_control(
            config,
            tokenizer,
            internal,
            split.tune_audit,
            base_calibration_margins,
            base_tune_margins=tune_base_margins,
            base_calibration_raw=base_calibration_raw,
            base_tune_raw=base_tune_raw,
            natural_prior=natural_prior,
            selected_steps=int(selected["step"]),
            task_dir=task_dir,
            selection_lock_path=task_dir / "selection_lock.json",
            expected_selection_sha=selection_sha,
            expected_exposure_sha=str(ledger["ordered_exposure_ids_sha256"]),
            expected_initial_fingerprint=initial_fingerprint,
        )
    manifest["status"] = "selection_completed"
    manifest["selection"] = {
        "path": str(selection_path.resolve()),
        "sha256": selection_sha,
        "selected_step": selected["step"],
        "internal_qualification_pass": True,
        "development_second_qualification_pass": development_pass,
        "confirm_eligible": development_pass,
    }
    manifest["resource_ledger"] = ledger
    manifest["pointwise_control_bundle"] = pointwise_bundle
    manifest["tune_audit_accessed"] = True
    rescue._atomic_json(output_root / "manifest.json", manifest)
    (output_root / "SELECTION_COMPLETED").write_text(
        "BAP selection completed; confirm remains sealed\n", encoding="utf-8"
    )
    rescue._release_model(base_bank)
    return selection


def _pointwise_sequence_prior_step(
    bank: Any,
    tokenizer: Any,
    batch: Sequence[OfficialExample],
    optimizer: torch.optim.Optimizer,
    *,
    config: BAPConfig,
    natural_prior: Mapping[str, float],
) -> tuple[float, dict[str, int]]:
    encoded = rescue._tokenize(tokenizer, batch, config=config)
    output = bank(**encoded)
    sequence_nll = per_example_target_token_nll(output.logits, encoded["labels"])
    weights = torch.tensor(
        [float(natural_prior[item.label]) / 0.5 for item in batch],
        dtype=sequence_nll.dtype,
        device=sequence_nll.device,
    )
    loss = torch.mean(weights * sequence_nll)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(tuple(bank.active_atom_parameters()), 1.0)
    optimizer.step()
    bank.assert_exact_budget()
    return float(loss.detach().cpu()), {
        "gradient_examples": len(batch),
        "source_tokens_processed": int(encoded["attention_mask"].sum()),
        "target_tokens_processed": int(encoded["labels"].ne(-100).sum()),
    }


def run_optional_pointwise_control(
    config: BAPConfig,
    tokenizer: Any,
    internal: BAPInternalSplit,
    tune_examples: Sequence[OfficialExample],
    base_calibration_margins: Sequence[float],
    *,
    base_tune_margins: Sequence[float],
    base_calibration_raw: Mapping[str, Any],
    base_tune_raw: Mapping[str, Any],
    natural_prior: Mapping[str, float],
    selected_steps: int,
    task_dir: Path,
    selection_lock_path: Path,
    expected_selection_sha: str,
    expected_exposure_sha: str,
    expected_initial_fingerprint: str,
) -> dict[str, Any]:
    """Train an ineligible, same-step/same-exposure pointwise audit arm."""

    if rescue._sha256_file(Path(json.loads(selection_lock_path.read_text(encoding="utf-8"))["selection_path"])) != expected_selection_sha:
        raise RuntimeError("selection changed before pointwise control training")
    model_config = _model_config(config)
    bank = rescue._build_model(model_config, tokenizer)
    control_initial_fingerprint = rescue._adapter_fingerprint(bank)
    if control_initial_fingerprint != expected_initial_fingerprint:
        raise RuntimeError("pointwise control adapter initialization drifted")
    # Match BAP's deterministic gradient forward mode.  This changes module
    # stochasticity only; autograd and optimizer updates remain active.
    bank.eval()
    optimizer = torch.optim.AdamW(
        tuple(bank.all_atom_parameters()), lr=config.learning_rate, weight_decay=0.0
    )
    losses: list[float] = []
    exposure_ids: list[str] = []
    tokens = {"source": 0, "target": 0}
    for _step, batch in paired_fit_batches(
        internal.fit,
        batch_size=config.batch_size,
        steps=selected_steps,
        seed=config.seed,
    ):
        loss, ledger = _pointwise_sequence_prior_step(
            bank,
            tokenizer,
            batch,
            optimizer,
            config=config,
            natural_prior=natural_prior,
        )
        losses.append(loss)
        exposure_ids.extend(item.example_id for item in batch)
        tokens["source"] += ledger["source_tokens_processed"]
        tokens["target"] += ledger["target_tokens_processed"]
    post_cal_margins = collect_label_margins(
        bank, tokenizer, internal.calibration, config=config
    )
    cal_comparison = calibrated_training_comparison(
        base_calibration_margins,
        post_cal_margins,
        [item.label for item in internal.calibration],
        natural_prior=natural_prior,
        max_class_drop=config.max_class_drop,
    )
    post_cal_raw = _raw_audit(
        bank,
        tokenizer,
        internal.calibration,
        config=config,
        natural_prior=natural_prior,
    )
    post_tune_margins = collect_label_margins(
        bank, tokenizer, tune_examples, config=config
    )
    tune_comparison = apply_frozen_thresholds(
        base_tune_margins,
        post_tune_margins,
        [item.label for item in tune_examples],
        natural_prior=natural_prior,
        base_threshold=float(
            cal_comparison["base_same_calibration_control"]["threshold"]
        ),
        post_threshold=float(cal_comparison["bap_calibrated"]["threshold"]),
    )
    post_tune_raw = _raw_audit(
        bank,
        tokenizer,
        tune_examples,
        config=config,
        natural_prior=natural_prior,
    )
    control_dir = task_dir / "pointwise_control"
    control_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = control_dir / "adapter.pt"
    rescue._atomic_torch_save(checkpoint, bank.compact_state_dict())
    control_exposure_sha = _sha256_json(exposure_ids)
    if control_exposure_sha != expected_exposure_sha:
        raise RuntimeError("pointwise control did not replay the exact BAP exposure order")
    bundle = {
        "format": FORMAT_VERSION,
        "role": "post-selection same-step same-paired-exposure audit control",
        "objective": CONTROL_OBJECTIVE,
        "selection_eligible": False,
        "trained_after_durable_selection": True,
        "conditional_on_bap_selected_step": True,
        "independently_tuned": False,
        "selected_steps_matched": selected_steps,
        "gradient_example_exposures": selected_steps * config.batch_size,
        "scored_label_sequences": selected_steps * config.batch_size,
        "ordered_exposure_ids_sha256": control_exposure_sha,
        "matched_bap_exposure_sha256": expected_exposure_sha,
        "exact_ordered_exposure_match": True,
        "initial_adapter_fingerprint": control_initial_fingerprint,
        "matched_bap_initialization": True,
        "gradient_forward_mode": "eval (gradients enabled; dropout disabled)",
        "forward_compute_matched_to_bap": False,
        "compute_caveat": (
            "example exposure and optimizer steps are matched, but BAP scores "
            "two label sequences per example"
        ),
        "mean_loss": float(np.mean(losses)),
        "tokens": tokens,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": rescue._sha256_file(checkpoint),
        },
        "internal_calibration": {
            **cal_comparison,
            "gain_decomposition": _raw_gain_decomposition(
                base_calibration_raw, post_cal_raw, cal_comparison
            ),
        },
        "development_tune": {
            "comparison": tune_comparison,
            "gain_decomposition": _raw_gain_decomposition(
                base_tune_raw, post_tune_raw, tune_comparison
            ),
        },
        "confirm_audit_accessed": False,
    }
    rescue._atomic_json(control_dir / "control.json", bundle)
    if rescue._sha256_file(Path(json.loads(selection_lock_path.read_text(encoding="utf-8"))["selection_path"])) != expected_selection_sha:
        raise RuntimeError("selection changed during pointwise control training")
    rescue._release_model(bank)
    return {
        "path": str((control_dir / "control.json").resolve()),
        "sha256": rescue._sha256_file(control_dir / "control.json"),
        "checkpoint_sha256": bundle["checkpoint"]["sha256"],
        "selection_eligible": False,
    }


def load_selection(path: str | os.PathLike[str]) -> tuple[dict[str, Any], Path]:
    selection_path = Path(path).expanduser().resolve()
    try:
        with selection_path.open("r", encoding="utf-8") as handle:
            selection = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load BAP selection {selection_path}") from error
    if not isinstance(selection, dict) or selection.get("format") != FORMAT_VERSION:
        raise ValueError("selection is not a BAP-LoRA v1 artifact")
    if selection.get("status") != "selection_locked":
        raise ValueError("BAP selection is not durably locked")
    return selection, selection_path


def _config_from_selection(
    selection: Mapping[str, Any], *, output_root: str, device: str
) -> BAPConfig:
    payload = selection.get("scientific_config")
    if not isinstance(payload, Mapping):
        raise ValueError("BAP selection lacks scientific_config")
    values = dict(payload)
    values["checkpoint_steps"] = tuple(values["checkpoint_steps"])
    values["fresh_confirm_against_manifests"] = tuple(
        values.get("fresh_confirm_against_manifests", ())
    )
    locked_device = str(values.get("device"))
    if str(device) != locked_device:
        raise ValueError(
            f"confirmation device {device!r} differs from locked {locked_device!r}"
        )
    values["output_root"] = output_root
    values["confirm_selection"] = "locked"
    values["dry_run"] = False
    return BAPConfig(**values)


def _validate_confirmation_config(
    config: BAPConfig, selection: Mapping[str, Any]
) -> None:
    """Reject scientific drift; only output location/mode marker may differ."""

    locked = selection.get("scientific_config")
    if not isinstance(locked, Mapping):
        raise ValueError("BAP selection lacks scientific_config")
    current = asdict(config)
    operational_fields = {"output_root", "confirm_selection"}
    locked_scientific = {
        key: value for key, value in locked.items() if key not in operational_fields
    }
    current_scientific = {
        key: value for key, value in current.items() if key not in operational_fields
    }
    if _sha256_json(locked_scientific) != _sha256_json(current_scientific):
        differing = sorted(
            key
            for key in set(locked_scientific) | set(current_scientific)
            if _sha256_json(locked_scientific.get(key))
            != _sha256_json(current_scientific.get(key))
        )
        raise ValueError(
            "confirmation scientific config differs from locked selection: "
            + ", ".join(differing)
        )


def _preflight_pointwise_control(
    selection: Mapping[str, Any], selection_path: Path
) -> dict[str, Any] | None:
    """Validate an optional post-selection control before any confirm scoring."""

    scientific_config = selection.get("scientific_config")
    if not isinstance(scientific_config, Mapping):
        raise ValueError("BAP selection lacks scientific_config")
    if not bool(scientific_config.get("run_pointwise_control", False)):
        return None
    control_path = selection_path.parent / "pointwise_control" / "control.json"
    if not control_path.is_file():
        raise ValueError("locked BAP protocol requires a missing pointwise control")
    try:
        control = json.loads(control_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load the pointwise control artifact") from error
    if (
        not isinstance(control, Mapping)
        or control.get("format") != FORMAT_VERSION
        or control.get("objective") != CONTROL_OBJECTIVE
        or control.get("selection_eligible") is not False
        or control.get("trained_after_durable_selection") is not True
        or control.get("conditional_on_bap_selected_step") is not True
        or control.get("independently_tuned") is not False
        or control.get("confirm_audit_accessed") is not False
    ):
        raise ValueError("pointwise control protocol metadata is invalid")
    control_sha = rescue._sha256_file(control_path)
    manifest_path = selection_path.parent.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("BAP selection manifest is missing for pointwise control")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load BAP selection manifest") from error
    bundle = manifest.get("pointwise_control_bundle")
    if (
        manifest.get("status") != "selection_completed"
        or not isinstance(manifest.get("selection"), Mapping)
        or manifest["selection"].get("sha256")
        != rescue._sha256_file(selection_path)
        or not isinstance(bundle, Mapping)
        or Path(str(bundle.get("path", ""))).expanduser().resolve()
        != control_path.resolve()
        or bundle.get("sha256") != control_sha
        or bundle.get("selection_eligible") is not False
    ):
        raise ValueError("selection manifest does not bind the pointwise control")
    selected_step = int(selection["selected"]["step"])
    expected_exposure_sha = str(
        selection["resource_ledger"]["ordered_exposure_ids_sha256"]
    )
    if (
        int(control.get("selected_steps_matched", -1)) != selected_step
        or int(control.get("gradient_example_exposures", -1))
        != selected_step * int(scientific_config["batch_size"])
        or int(control.get("scored_label_sequences", -1))
        != selected_step * int(scientific_config["batch_size"])
        or control.get("ordered_exposure_ids_sha256") != expected_exposure_sha
        or control.get("matched_bap_exposure_sha256") != expected_exposure_sha
        or control.get("exact_ordered_exposure_match") is not True
        or control.get("matched_bap_initialization") is not True
        or control.get("forward_compute_matched_to_bap") is not False
        or control.get("initial_adapter_fingerprint")
        != selection["model_protocol"]["initial_adapter_fingerprint"]
    ):
        raise ValueError("pointwise control is not exactly resource-matched to BAP")
    checkpoint_payload = control.get("checkpoint")
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("pointwise control checkpoint metadata is missing")
    checkpoint = Path(str(checkpoint_payload.get("path", ""))).expanduser().resolve()
    expected_checkpoint = (control_path.parent / "adapter.pt").resolve()
    if checkpoint != expected_checkpoint or not checkpoint.is_file():
        raise ValueError("pointwise control checkpoint path is invalid")
    checkpoint_sha = rescue._sha256_file(checkpoint)
    if checkpoint_sha != checkpoint_payload.get("sha256"):
        raise ValueError("pointwise control checkpoint SHA-256 mismatch")
    if bundle.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("selection manifest does not bind the control checkpoint")
    internal = control.get("internal_calibration")
    if not isinstance(internal, Mapping):
        raise ValueError("pointwise control lacks internal calibration metadata")
    try:
        base_threshold = float(
            internal["base_same_calibration_control"]["threshold"]
        )
        control_threshold = float(internal["bap_calibrated"]["threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("pointwise control thresholds are malformed") from error
    if not math.isclose(
        base_threshold,
        float(selection["selected"]["base_threshold"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("pointwise and BAP calibrated-base thresholds disagree")
    return {
        "control_path": str(control_path.resolve()),
        "control_sha256": control_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "base_threshold": base_threshold,
        "control_threshold": control_threshold,
        "selected_steps_matched": selected_step,
        "ordered_exposure_ids_sha256": expected_exposure_sha,
    }


def _preflight_confirmation(
    selection: Mapping[str, Any],
    selection_path: Path,
    split: rescue.RescueSplit,
    internal: BAPInternalSplit,
    distribution: Mapping[str, Any],
) -> dict[str, Any]:
    if selection.get("confirm_eligible") is not True:
        raise ValueError("BAP selection is not eligible for sealed confirmation")
    if selection.get("confirm_audit_accessed") is not False:
        raise ValueError("BAP selection does not attest a sealed confirm audit")
    lock_path = selection_path.parent / "selection_lock.json"
    if not lock_path.is_file():
        raise ValueError("BAP selection lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selection_sha = rescue._sha256_file(selection_path)
    if (
        lock.get("selection_sha256") != selection_sha
        or Path(str(lock.get("selection_path", ""))).expanduser().resolve()
        != selection_path.resolve()
        or lock.get("selected_checkpoint_sha256")
        != selection["selected_checkpoint"]["sha256"]
        or int(lock.get("selected_step", -1)) != int(selection["selected"]["step"])
        or lock.get("confirm_eligible") is not True
        or lock.get("confirm_audit_accessed") is not False
    ):
        raise ValueError("BAP selection lock does not bind the selection")
    protocol_lock_metadata = selection.get("run_local_protocol_lock")
    if not isinstance(protocol_lock_metadata, Mapping):
        raise ValueError("BAP selection lacks its run-local protocol lock")
    protocol_lock_path = (selection_path.parent / "protocol_lock.json").resolve()
    if (
        Path(str(protocol_lock_metadata.get("path", ""))).expanduser().resolve()
        != protocol_lock_path
        or not protocol_lock_path.is_file()
        or rescue._sha256_file(protocol_lock_path)
        != protocol_lock_metadata.get("sha256")
        or protocol_lock_metadata.get("external_preregistration_claimed") is not False
    ):
        raise ValueError("run-local protocol lock path or hash changed")
    try:
        protocol_lock_payload = json.loads(
            protocol_lock_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not load run-local protocol lock") from error
    payload_without_hash = dict(protocol_lock_payload)
    recorded_payload_sha = payload_without_hash.pop("payload_sha256", None)
    if (
        recorded_payload_sha != _sha256_json(payload_without_hash)
        or recorded_payload_sha != protocol_lock_metadata.get("payload_sha256")
        or _sha256_json(protocol_lock_payload.get("scientific_config"))
        != _sha256_json(selection.get("scientific_config"))
        or _sha256_json(protocol_lock_payload.get("protocol"))
        != _sha256_json(selection.get("protocol"))
    ):
        raise ValueError("run-local protocol lock payload changed")
    checkpoint = Path(
        str(selection["selected_checkpoint"]["path"])
    ).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"BAP selected checkpoint is missing: {checkpoint}")
    checkpoint_sha = rescue._sha256_file(checkpoint)
    if checkpoint_sha != selection["selected_checkpoint"]["sha256"]:
        raise ValueError("BAP selected checkpoint SHA-256 mismatch")
    outer_sha = rescue._atomic_json_payload_sha256(rescue._split_manifest(split))
    internal_sha = rescue._atomic_json_payload_sha256(
        bap_internal_split_manifest(internal)
    )
    if outer_sha != selection.get("outer_split_manifest_sha256"):
        raise ValueError("reconstructed BAP outer split does not match selection")
    if internal_sha != selection.get("internal_split_manifest_sha256"):
        raise ValueError("reconstructed BAP internal split does not match selection")
    if distribution != selection.get("pinned_train_label_distribution"):
        raise ValueError("pinned train label distribution changed after selection")
    locked_model_protocol = selection.get("model_protocol")
    if not isinstance(locked_model_protocol, Mapping):
        raise ValueError("BAP selection lacks locked model protocol metadata")
    locked_artifact = locked_model_protocol.get("frozen_model_tokenizer_artifact")
    current_artifact = _local_artifact_fingerprint(
        str(selection["scientific_config"]["model_path"])
    )
    if current_artifact != locked_artifact:
        raise ValueError("frozen model/tokenizer artifact changed after selection")
    pointwise_control = _preflight_pointwise_control(selection, selection_path)
    return {
        "selection_sha256": selection_sha,
        "selection_lock": lock,
        "run_local_protocol_lock": dict(protocol_lock_metadata),
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "outer_split_sha256": outer_sha,
        "internal_split_sha256": internal_sha,
        "frozen_model_tokenizer_artifact": current_artifact,
        "pointwise_control": pointwise_control,
    }


def _claim_confirmation(
    selection_path: Path, *, output_root: Path, preflight: Mapping[str, Any]
) -> Path:
    claim_path = selection_path.parent / "confirmation_claim.json"
    payload = {
        "format": FORMAT_VERSION,
        "status": "claimed_before_confirm_scoring",
        "selection_sha256": preflight["selection_sha256"],
        "checkpoint_sha256": preflight["checkpoint_sha256"],
        "confirmation_output_root": str(output_root.resolve()),
        "claimed_unix_time": time.time(),
    }
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with claim_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise RuntimeError(
            "this BAP selection already has a confirmation claim; refusing a second confirm read"
        ) from error
    return claim_path


def run_confirmation(
    config: BAPConfig,
    selection: Mapping[str, Any],
    selection_path: Path,
    *,
    train_examples: Sequence[OfficialExample] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Independently evaluate the sealed confirm partition exactly once."""

    disk_selection, resolved_selection_path = load_selection(selection_path)
    if _sha256_json(selection) != _sha256_json(disk_selection):
        raise ValueError(
            "in-memory selection differs from the locked on-disk selection"
        )
    selection = disk_selection
    selection_path = resolved_selection_path
    _validate_confirmation_config(config, selection)
    output_root = Path(config.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing non-empty BAP confirmation root {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    train_examples_injected = train_examples is not None
    if train_examples_injected and not config.tiny_random_model:
        raise ValueError(
            "formal BAP confirmation forbids train_examples injection"
        )
    if train_examples_injected != bool(
        selection.get("train_examples_injected_for_tiny_test", False)
    ):
        raise ValueError("confirmation data loading mode differs from selection")
    current_data_artifact = (
        None
        if train_examples_injected
        else _official_train_artifact_fingerprint(
            config.data_root, task_name=config.task_name
        )
    )
    if current_data_artifact != selection.get("verified_data_artifact"):
        raise ValueError("verified official data artifact changed after selection")
    if train_examples is None:
        train_examples = load_official_examples(
            config.data_root, config.task_name, "train", verify=True
        )
    train_corpus_fingerprint = _train_corpus_fingerprint(
        train_examples, task_name=config.task_name
    )
    if train_corpus_fingerprint != selection.get("train_corpus_fingerprint"):
        raise ValueError("train corpus content changed after BAP selection")
    execution_environment = _execution_environment_fingerprint(config.device)
    if execution_environment != selection.get("execution_environment"):
        raise ValueError("confirmation numerical execution environment drifted")
    distribution = rescue.pinned_train_label_distribution(
        train_examples, task_name=config.task_name
    )
    natural_prior = distribution["natural_label_prior"]
    split, freshness_sources = _build_outer_split(train_examples, config=config)
    if [dict(item) for item in freshness_sources] != selection.get(
        "freshness_sources"
    ):
        raise ValueError("fresh-confirm source manifests changed after BAP selection")
    internal = build_bap_internal_split(split, config=config)
    # All data/model files and eligibility are checked before tokenizer
    # construction and before any confirm-row scoring or token-length inspection.
    preflight = _preflight_confirmation(
        selection, selection_path, split, internal, distribution
    )
    tokenizer_injected = tokenizer is not None
    if tokenizer_injected and not config.tiny_random_model:
        raise ValueError("formal BAP confirmation forbids tokenizer injection")
    if tokenizer_injected != bool(
        selection["model_protocol"].get("tokenizer_injected_for_tiny_test", False)
    ):
        raise ValueError("confirmation tokenizer loading mode differs from selection")
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, use_fast=True, local_files_only=True
        )
    tokenizer_fingerprint = _tokenizer_protocol_fingerprint(tokenizer)
    if tokenizer_fingerprint != selection["model_protocol"].get(
        "tokenizer_protocol_fingerprint"
    ):
        raise ValueError("tokenizer protocol changed after BAP selection")
    # The tokenizer check above touches no confirm row.  Claim atomically only
    # after every non-confirm preflight succeeds, then score confirm exactly once.
    claim_path = _claim_confirmation(
        selection_path, output_root=output_root, preflight=preflight
    )
    model_config = _model_config(config)
    base_bank = rescue._build_model(model_config, tokenizer)
    expected_fingerprint = selection["model_protocol"][
        "initial_adapter_fingerprint"
    ]
    if rescue._adapter_fingerprint(base_bank) != expected_fingerprint:
        raise ValueError("BAP confirmation base initialization fingerprint changed")
    post_bank = rescue._build_model(model_config, tokenizer)
    if rescue._adapter_fingerprint(post_bank) != expected_fingerprint:
        raise ValueError("BAP confirmation adapter initialization fingerprint changed")
    checkpoint_state = torch.load(
        preflight["checkpoint"], map_location="cpu", weights_only=False
    )
    post_bank.load_compact_state_dict(checkpoint_state)
    control_bank = None
    control_preflight = preflight.get("pointwise_control")
    if control_preflight is not None:
        control_bank = rescue._build_model(model_config, tokenizer)
        if rescue._adapter_fingerprint(control_bank) != expected_fingerprint:
            raise ValueError(
                "BAP confirmation pointwise-control initialization fingerprint changed"
            )
        control_state = torch.load(
            Path(str(control_preflight["checkpoint"])),
            map_location="cpu",
            weights_only=False,
        )
        control_bank.load_compact_state_dict(control_state)
    base_raw = _raw_audit(
        base_bank,
        tokenizer,
        split.confirm_audit,
        config=config,
        natural_prior=natural_prior,
    )
    post_raw = _raw_audit(
        post_bank,
        tokenizer,
        split.confirm_audit,
        config=config,
        natural_prior=natural_prior,
    )
    base_margins = collect_label_margins(
        base_bank, tokenizer, split.confirm_audit, config=config
    )
    post_margins = collect_label_margins(
        post_bank, tokenizer, split.confirm_audit, config=config
    )
    comparison = apply_frozen_thresholds(
        base_margins,
        post_margins,
        [item.label for item in split.confirm_audit],
        natural_prior=natural_prior,
        base_threshold=float(selection["selected"]["base_threshold"]),
        post_threshold=float(selection["selected"]["bap_threshold"]),
    )
    gate_pass = qualification_gate(
        comparison,
        min_gain=config.min_rescue_gain,
        max_class_drop=config.max_class_drop,
    )
    pointwise_confirmation = None
    if control_bank is not None:
        control_raw = _raw_audit(
            control_bank,
            tokenizer,
            split.confirm_audit,
            config=config,
            natural_prior=natural_prior,
        )
        control_margins = collect_label_margins(
            control_bank, tokenizer, split.confirm_audit, config=config
        )
        control_comparison = apply_frozen_thresholds(
            base_margins,
            control_margins,
            [item.label for item in split.confirm_audit],
            natural_prior=natural_prior,
            base_threshold=float(control_preflight["base_threshold"]),
            post_threshold=float(control_preflight["control_threshold"]),
        )
        control_gate_pass = qualification_gate(
            control_comparison,
            min_gain=config.min_rescue_gain,
            max_class_drop=config.max_class_drop,
        )
        bap_metrics = comparison["bap_calibrated"]["metrics"]
        control_metrics = control_comparison["bap_calibrated"]["metrics"]
        pointwise_confirmation = {
            "status": "confirmation_completed_in_same_one_shot",
            "objective": CONTROL_OBJECTIVE,
            "selection_eligible": False,
            "checkpoint_or_coefficient_selection_influence": False,
            "conditional_on_bap_selected_step": True,
            "independently_tuned": False,
            "claim_scope": (
                "matched local audit only; not superiority to an independently "
                "tuned pointwise method"
            ),
            "exact_resource_match": {
                "scope": "optimizer steps, initialization, and ordered example exposures",
                "optimizer_steps": int(
                    control_preflight["selected_steps_matched"]
                ),
                "ordered_exposure_ids_sha256": control_preflight[
                    "ordered_exposure_ids_sha256"
                ],
                "same_initial_adapter": True,
                "forward_compute_matched": False,
                "compute_caveat": (
                    "BAP scores two official-label sequences per example; "
                    "pointwise scores one"
                ),
            },
            "comparison_to_same_calibrated_base": control_comparison,
            "gain_decomposition": _raw_gain_decomposition(
                base_raw, control_raw, control_comparison
            ),
            "control_gate_pass_for_audit_only": control_gate_pass,
            "bap_minus_matched_pointwise": {
                "calibrated_natural_prior_em": float(
                    bap_metrics["natural_prior_em"]
                    - control_metrics["natural_prior_em"]
                ),
                "calibrated_balanced_em": float(
                    bap_metrics["balanced_em"]
                    - control_metrics["balanced_em"]
                ),
                "per_class_recall": {
                    label: float(bap_metrics["per_class_recall"][label])
                    - float(control_metrics["per_class_recall"][label])
                    for label in BINARY_LABELS
                },
                "raw_free_generation_natural_prior_em": float(
                    post_raw["natural_prior_em"]
                    - control_raw["natural_prior_em"]
                ),
            },
            "preflight": dict(control_preflight),
        }
    result = {
        "format": FORMAT_VERSION,
        "status": "confirmation_completed",
        "task": config.task_name,
        "selection": {
            "path": str(selection_path),
            "sha256": preflight["selection_sha256"],
            "checkpoint_sha256": preflight["checkpoint_sha256"],
            "selected_step": selection["selected"]["step"],
        },
        "confirmation_claim": {
            "path": str(claim_path.resolve()),
            "created_before_confirm_scoring": True,
            "scope_note": (
                "train metadata and semantic split were reconstructed during "
                "preflight before the claim; no confirm model scoring/tokenization occurred"
            ),
        },
        "thresholds_frozen_on_internal_calibration": {
            "base": float(selection["selected"]["base_threshold"]),
            "bap": float(selection["selected"]["bap_threshold"]),
        },
        "comparison": comparison,
        "gain_decomposition": _raw_gain_decomposition(
            base_raw, post_raw, comparison
        ),
        "confirm_gate_pass": gate_pass,
        "official_positive_claim_metric": "BAP-calibrated minus base-calibrated natural-prior EM",
        "calibration_gain_counted_as_training_contribution": False,
        "raw_free_generation_positive_claim": False,
        "calibrated_label_validity": "1 by construction; not an empirical gate",
        "confirm_rows_used_for_selection_or_threshold": False,
        "pointwise_control_confirmation": pointwise_confirmation,
    }
    task_dir = output_root / config.task_name
    task_dir.mkdir(parents=True, exist_ok=False)
    rescue._atomic_json(task_dir / "confirmation.json", result)
    rescue._atomic_json(
        output_root / "manifest.json",
        {
            "format": FORMAT_VERSION,
            "status": "confirmation_completed",
            "config": asdict(config),
            "protocol": _protocol_payload(config),
            "preflight": {
                **dict(preflight),
                "checkpoint": str(preflight["checkpoint"]),
            },
            "summary": {
                "confirm_gate_pass": gate_pass,
                "training_contribution_natural_prior": comparison[
                    "training_contribution_natural_prior"
                ],
                "training_min_per_class_delta": comparison[
                    "training_min_per_class_delta"
                ],
            },
        },
    )
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    claim["status"] = "confirmation_completed"
    claim["confirmation_result"] = str(
        (task_dir / "confirmation.json").resolve()
    )
    claim["confirmation_result_sha256"] = rescue._sha256_file(
        task_dir / "confirmation.json"
    )
    rescue._atomic_json(claim_path, claim)
    (output_root / "COMPLETED").write_text(
        "BAP independent confirmation completed\n", encoding="utf-8"
    )
    rescue._release_model(base_bank)
    rescue._release_model(post_bank)
    if control_bank is not None:
        rescue._release_model(control_bank)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task", dest="task_name", default="BoolQA")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--fit-per-class", type=int, default=384)
    parser.add_argument("--calibration-per-class", type=int, default=128)
    parser.add_argument("--tune-audit-per-class", type=int, default=128)
    parser.add_argument("--confirm-audit-per-class", type=int, default=32)
    parser.add_argument(
        "--checkpoint-steps", type=int, nargs="+", default=DEFAULT_CHECKPOINT_STEPS
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=50)
    parser.add_argument(
        "--truncation-mode", choices=TRUNCATION_MODES, default="field_aware"
    )
    parser.add_argument("--min-rescue-gain", type=float, default=0.05)
    parser.add_argument("--max-class-drop", type=float, default=0.05)
    parser.add_argument("--pair-temperature", type=float, default=1.0)
    parser.add_argument("--common-shift-weight", type=float, default=1.0)
    parser.add_argument("--safe-hinge-weight", type=float, default=2.0)
    parser.add_argument("--safe-margin-cap", type=float, default=0.5)
    parser.add_argument("--scale-floor", type=float, default=0.05)
    parser.add_argument("--tiny-random-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-selection")
    parser.add_argument("--run-pointwise-control", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--fresh-confirm-against-manifest",
        dest="fresh_confirm_against_manifests",
        action="append",
        default=[],
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> BAPConfig:
    return BAPConfig(
        data_root=args.data_root,
        model_path=args.model_path,
        output_root=args.output_root,
        task_name=args.task_name,
        learning_rate=args.learning_rate,
        fit_per_class=args.fit_per_class,
        calibration_per_class=args.calibration_per_class,
        tune_audit_per_class=args.tune_audit_per_class,
        confirm_audit_per_class=args.confirm_audit_per_class,
        checkpoint_steps=tuple(args.checkpoint_steps),
        seed=args.seed,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        truncation_mode=args.truncation_mode,
        min_rescue_gain=args.min_rescue_gain,
        max_class_drop=args.max_class_drop,
        pair_temperature=args.pair_temperature,
        common_shift_weight=args.common_shift_weight,
        safe_hinge_weight=args.safe_hinge_weight,
        safe_margin_cap=args.safe_margin_cap,
        scale_floor=args.scale_floor,
        tiny_random_model=args.tiny_random_model,
        dry_run=args.dry_run,
        confirm_selection=args.confirm_selection,
        run_pointwise_control=args.run_pointwise_control,
        device=args.device,
        fresh_confirm_against_manifests=tuple(
            args.fresh_confirm_against_manifests
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        cli_config = _config_from_args(args)
        if cli_config.confirm_selection is None:
            run_selection(cli_config)
        else:
            selection, selection_path = load_selection(cli_config.confirm_selection)
            config = _config_from_selection(
                selection, output_root=cli_config.output_root, device=cli_config.device
            )
            if Path(config.data_root).resolve() != Path(cli_config.data_root).resolve():
                raise ValueError("confirmation data_root differs from locked selection")
            if Path(config.model_path).resolve() != Path(cli_config.model_path).resolve():
                raise ValueError("confirmation model_path differs from locked selection")
            run_confirmation(config, selection, selection_path)
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
