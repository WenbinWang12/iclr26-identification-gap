"""Read-only reconciliation of Phase-2 primary and confirmation artifacts.

The validator deliberately does not import PyTorch and never writes below either
input directory.  Its only output is ``reconciliation.json`` in a new, empty
directory supplied by the caller.
"""

from __future__ import annotations

import csv
import functools
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import artifacts
from .core_numpy import (
    BridgeConfig,
    PRIMARY_SEEDS,
    PROTOCOL_SHA256,
    RANKS,
    REGIMES,
    bootstrap_summary,
    build_seed_dataset,
    build_prefix_oracle,
    make_base_draws,
    make_factor_initializations,
    make_seed_streams,
    mode_count,
    mode_schedule,
    population_risk,
    q_min_keep,
    run_all_sanity,
    sanity_contraction_bound,
    weighted_modal_svd,
)
from .replay import ClockBalancedReplay, ReservoirReplay


SCHEMA_VERSION = 1
ORACLE_ATOL = 2e-10
TRAINED_ATOL = 1e-6
TRAINABLE_ARMS = (
    "sequential_dense_rank_R",
    "clock_balanced_replay_rank_R",
    "reservoir_random_replay_rank_R",
)
REPLAY_ARMS = TRAINABLE_ARMS[1:]
REQUIRED_OUTPUTS = {
    "config.json",
    "gates.json",
    "per_seed.csv",
    "per_window.csv",
    "resource_ledger.csv",
    "split_manifest.json",
    "summary.json",
    "traces.json",
}
CANONICAL_SOURCE_FILES = frozenset(
    {
        "__init__.py",
        "artifacts.py",
        "core_numpy.py",
        "evidence.py",
        "phase2_gpu_bridge.py",
        "replay.py",
        "test_phase2_gpu_bridge.py",
        "test_validation.py",
        "torch_runner.py",
        "validation.py",
    }
)

PER_WINDOW_KEY = ("seed", "regime", "R", "window", "arm")
PER_SEED_KEY = ("seed", "regime", "R", "arm")
RESOURCE_KEY = PER_SEED_KEY
TRANSITION_KEY = ("seed", "regime", "R", "arm", "completed_window")
TRANSITION_FIELDS = frozenset(
    {
        *TRANSITION_KEY,
        "history_windows",
        "capacity",
        "eligible_count",
        "occupied_count",
        "unique_count",
        "duplicate_count",
        "replacement_count",
        "priority_draws",
        "selection_comparisons",
        "priority_tie_count",
        "controller_cpu_seconds",
        "selected_priority_order_sha256",
    }
)
_REPLAY_DETERMINISTIC_FIELDS = frozenset(TRANSITION_FIELDS - {"controller_cpu_seconds"})
TRANSITION_INTEGER_FIELDS = frozenset(
    {
        "seed",
        "R",
        "completed_window",
        "history_windows",
        "capacity",
        "eligible_count",
        "occupied_count",
        "unique_count",
        "duplicate_count",
        "replacement_count",
        "priority_draws",
        "selection_comparisons",
        "priority_tie_count",
    }
)
EXPECTED_TRANSITION_ROWS = len(PRIMARY_SEEDS) * len(REGIMES) * len(RANKS) * len(REPLAY_ARMS) * 11
_ALLOWED_GATE_STATUSES = {
    "evaluated",
    "not_evaluated",
    "primary_only",
    "pending",
    "pending_reconciliation",
    "passed_non_citable_single_run",
    "failed",
}
_EXPECTED_PEAK_SCOPE = (
    "arm_cell_isolated_full_12_window_training_from_pre_model_process_resident_baseline; "
    "inference_independently_reset_from_post_training_model_resident_baseline"
)
_GATE_NUMERIC_KEYS = {
    "contraction_bound",
    "contraction_bound_limit",
    "max_analytic_tail_delta",
    "max_sanity_normalized_error",
    "max_weighted_modal_svd_delta",
    "q_min_keep",
    "q_min_keep_expected",
    "lower_bound_05",
    "upper_bound_95",
    "bootstrap_lower_bound_05",
    "gap_fraction_median",
    "sequential_online_ratio_lower_bound",
    "sequential_online_ratio_threshold",
    "selection_gain_lower_bound",
}
_G1_FIELDS = {
    "passed",
    "oracle_rows",
    "q_min_keep",
    "q_min_keep_expected",
    "contraction_bound",
    "contraction_bound_limit",
    "support_failures",
    "residual_failures",
    "max_sanity_normalized_error",
    "passed_support",
    "passed_error",
    "weighted_modal_svd_mismatches",
    "max_weighted_modal_svd_delta",
    "max_analytic_tail_delta",
    "passed_oracle_agreement",
    "passed_capacity",
    "passed_contraction_bound",
}

PER_WINDOW_ORACLE_FIELDS = {
    "L_rank",
    "L_inf",
    "D_t",
    "D_current",
    "capacity_ratio",
}
PER_WINDOW_TRAINED_FIELDS = {
    "L_prefix",
    "L_current",
    "online_ratio",
    "current_acquisition_ratio",
    "gap",
    "bounded_history_reduction",
    "selection_gain",
    "gap_fraction",
    "paired_gap_seq",
    "paired_gap_clock",
    "paired_gap_reservoir",
}
OPTIONAL_FLOAT_FIELDS = {"gap_fraction", "final_gap_fraction"}
PER_SEED_TRAINED_FIELDS = {
    "final_gap",
    "final_online_ratio",
    "final_current_acquisition_ratio",
    "final_bounded_history_reduction",
    "final_selection_gain",
    "final_gap_fraction",
}

PER_WINDOW_REQUIRED_FIELDS = set(PER_WINDOW_KEY) | PER_WINDOW_ORACLE_FIELDS | PER_WINDOW_TRAINED_FIELDS | {
    "requested_rank",
    "numerical_rank",
    "validity",
    "updates",
    "forward",
    "backward",
    "current_examples",
    "replay_examples",
    "factor_sha256",
    "occupied",
    "unique",
    "duplicates",
    "priority_draws",
    "replay_sample_draws",
    "selection_comparisons",
    "controller_cpu_seconds",
    "replay_assembly_seconds",
    "device_staging_seconds",
    "metadata_leakage_events",
    "model_input_contract_passed",
    "replay_sampler_contract_passed",
    "train_wall_seconds",
    "peak_allocated_gpu_bytes_so_far",
    "peak_reserved_gpu_bytes_so_far",
}
PER_SEED_REQUIRED_FIELDS = set(PER_SEED_KEY) | PER_SEED_TRAINED_FIELDS | {"final_validity"}
RESOURCE_REQUIRED_FIELDS = set(RESOURCE_KEY) | {
    "schema_version",
    "requested_rank",
    "active_rank",
    "deployed_rank",
    "numerical_rank_max",
    "factor_shapes",
    "trainable_parameter_count",
    "deployed_parameter_count",
    "trainable_parameter_bytes",
    "deployed_parameter_bytes",
    "parameter_breakdown",
    "optimizer_state_bytes",
    "optimizer_state_breakdown",
    "optimizer_steps",
    "optimizer_contract",
    "optimizer_contract_sha256",
    "batch_size",
    "replay_fraction",
    "allocated_capacity",
    "replay_bytes",
    "replay_x_y_slot_bytes",
    "replay_priority_bytes",
    "replay_occupancy_bytes",
    "pointer_counter_bytes",
    "persistent_controller_sidecar_records",
    "persistent_controller_sidecar_bytes",
    "controller_transient_records_peak",
    "controller_transient_python_bytes_peak",
    "controller_transient_numpy_bytes_peak",
    "curvature_sketch_bytes",
    "checkpoint_persistent_bytes",
    "evaluator_checkpoint_staging_bytes",
    "common_W0_cpu_bytes",
    "common_W0_gpu_bytes",
    "common_cpu_cell_dataset_bytes",
    "common_cpu_seed_dataset_resident_bytes",
    "common_cpu_determinism_rebuild_increment_bytes",
    "common_cpu_determinism_rebuild_peak_bytes",
    "common_cpu_batch_staging_bytes",
    "common_cpu_index_staging_bytes_max",
    "arm_cpu_batch_staging_peak_bytes",
    "arm_cpu_index_staging_peak_bytes",
    "common_gpu_batch_staging_bytes",
    "updates",
    "forward",
    "backward",
    "current_examples",
    "replay_examples",
    "replay_sample_draws",
    "priority_draws",
    "selection_comparisons",
    "priority_tie_count",
    "replacement_count",
    "occupied_records_max",
    "unique_records_max",
    "replay_sampling_duplicate_count",
    "controller_cpu_seconds",
    "replay_assembly_seconds",
    "device_staging_seconds",
    "training_wall_seconds",
    "wall_seconds",
    "gpu_baseline_allocated_bytes",
    "gpu_baseline_reserved_bytes",
    "peak_allocated_gpu_bytes",
    "peak_reserved_gpu_bytes",
    "peak_allocated_increment_bytes",
    "peak_reserved_increment_bytes",
    "inference_peak_allocated_gpu_bytes",
    "inference_peak_reserved_gpu_bytes",
    "inference_baseline_allocated_bytes",
    "inference_baseline_reserved_bytes",
    "inference_peak_allocated_increment_bytes",
    "inference_peak_reserved_increment_bytes",
    "inference_latency_mean_batch_seconds",
    "inference_latency_mean_example_seconds",
    "inference_warmup_forwards",
    "inference_measured_forwards",
    "metadata_leakage_events",
    "model_input_contract_passed",
    "replay_sampler_contract_passed",
    "protocol_sha256",
    "frozen_config_sha256",
    "source_hashes",
    "test_sha256",
    "data_sha256",
    "split_sha256",
    "factor_sha256",
    "device",
    "device_index",
    "gpu_name",
    "driver_version",
    "hostname",
    "python_version",
    "numpy_version",
    "torch_version",
    "cuda_version",
    "cudnn_version",
    "nvidia_smi_devices",
    "deterministic_flags",
}

_INTEGER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_HEX_256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_MAX_REPORTED_ISSUES = 256
_MAX_JSON_INTEGER_DIGITS = 1024


class StrictJSONError(ValueError):
    """Raised when an artifact is not finite, duplicate-free strict JSON."""


@dataclass(frozen=True)
class Issue:
    check: str
    code: str
    location: str
    message: str
    disposition: str = "REVISE"

    def payload(self) -> dict[str, str]:
        return {
            "check": self.check,
            "code": self.code,
            "location": self.location,
            "message": self.message,
            "disposition": self.disposition,
        }


class IssueCollector:
    def __init__(self) -> None:
        self.issues: list[Issue] = []
        self.suppressed = 0
        self._totals: dict[tuple[str, str], int] = {}

    def add(
        self,
        check: str,
        code: str,
        location: str,
        message: str,
        disposition: str = "REVISE",
    ) -> None:
        issue = Issue(check, code, location, message, disposition)
        total_key = (check, disposition)
        self._totals[total_key] = self._totals.get(total_key, 0) + 1
        if len(self.issues) < _MAX_REPORTED_ISSUES:
            self.issues.append(issue)
        else:
            self.suppressed += 1

    def count(self, check: str | None = None, disposition: str | None = None) -> int:
        if check is not None and disposition is not None:
            return int(self._totals.get((check, disposition), 0))
        if check is not None:
            return int(sum(value for (item_check, _), value in self._totals.items() if item_check == check))
        if disposition is not None:
            return int(sum(value for (_, item_disposition), value in self._totals.items() if item_disposition == disposition))
        return int(sum(self._totals.values()))


@dataclass
class RunBundle:
    directory: Path
    label: str
    expected_role: str
    manifest: dict[str, Any]
    documents: dict[str, Any]


@dataclass
class CSVTable:
    name: str
    header: tuple[str, ...]
    rows: dict[tuple[Any, ...], dict[str, str]]


def _reject_constant(token: str) -> Any:
    raise StrictJSONError(f"non-finite JSON token {token!r}")


def _parse_strict_int(token: str) -> int:
    digits = token.lstrip("-")
    if len(digits) > _MAX_JSON_INTEGER_DIGITS:
        raise StrictJSONError(f"JSON integer exceeds {_MAX_JSON_INTEGER_DIGITS} digits")
    try:
        return int(token)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StrictJSONError(f"invalid JSON integer: {token!r}") from exc


def _object_without_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _assert_finite_json(value: Any, path: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StrictJSONError(f"non-finite number at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_json(item, f"{path}.{key}")
        return
    raise StrictJSONError(f"unsupported JSON value {type(value).__name__} at {path}")


def _load_json_strict(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            parse_int=_parse_strict_int,
            object_pairs_hook=_object_without_duplicates,
        )
        _assert_finite_json(value)
    except (OSError, UnicodeError, json.JSONDecodeError, StrictJSONError, ValueError, RecursionError) as exc:
        raise StrictJSONError(f"{path.name}: {exc}") from exc
    return value


def load_json_strict(path: str | Path) -> Any:
    """Public strict-JSON loader used by run preflight checks."""

    return _load_json_strict(Path(path))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HEX_256_RE.fullmatch(value) is not None


def _safe_declared_file(directory: Path, name: Any) -> Path | None:
    if not isinstance(name, str) or not name or Path(name).name != name:
        return None
    candidate = (directory / name).resolve()
    if candidate.parent != directory:
        return None
    return candidate


def _load_run(
    directory: Path,
    label: str,
    expected_role: str,
    collector: IssueCollector,
) -> RunBundle | None:
    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    if not directory.is_dir():
        collector.add("artifact_integrity", "missing_run_directory", label, str(directory))
        return None
    if not manifest_path.is_file():
        collector.add("artifact_integrity", "missing_manifest", label, "manifest.json is absent")
        return None
    try:
        manifest = _load_json_strict(manifest_path)
    except StrictJSONError as exc:
        collector.add("artifact_integrity", "invalid_manifest_json", label, str(exc))
        return None
    if not isinstance(manifest, dict):
        collector.add("artifact_integrity", "manifest_not_object", label, "manifest must be an object")
        return None

    output_hashes = manifest.get("output_hashes")
    if not isinstance(output_hashes, dict):
        collector.add("artifact_integrity", "invalid_output_hashes", label, "output_hashes must be an object")
        output_hashes = {}
    missing_required = sorted(REQUIRED_OUTPUTS - set(output_hashes))
    if missing_required:
        collector.add(
            "artifact_integrity",
            "missing_required_outputs",
            label,
            ",".join(missing_required),
        )

    documents: dict[str, Any] = {"manifest.json": manifest}
    for name in sorted(output_hashes):
        expected_hash = output_hashes[name]
        path = _safe_declared_file(directory, name)
        if path is None:
            collector.add("artifact_integrity", "unsafe_output_path", f"{label}.output_hashes", repr(name))
            continue
        if not _is_sha256(expected_hash):
            collector.add("artifact_integrity", "invalid_output_hash", f"{label}.{name}", repr(expected_hash))
            continue
        if not path.is_file():
            collector.add("artifact_integrity", "missing_hashed_output", f"{label}.{name}", "file is absent")
            continue
        actual_hash = artifacts.sha256_file(path)
        if actual_hash.lower() != expected_hash.lower():
            collector.add(
                "artifact_integrity",
                "output_hash_mismatch",
                f"{label}.{name}",
                f"expected={expected_hash},actual={actual_hash}",
            )
        if name.endswith(".json"):
            try:
                documents[name] = _load_json_strict(path)
            except StrictJSONError as exc:
                collector.add("artifact_integrity", "invalid_json_output", f"{label}.{name}", str(exc))

    hash_references = {
        "config.json": "config_sha256",
        "split_manifest.json": "split_manifest_sha256",
        "gates.json": "gates_sha256",
        "traces.json": "traces_sha256",
    }
    for filename, field in hash_references.items():
        declared = manifest.get(field)
        output_declared = output_hashes.get(filename)
        if not _is_sha256(declared) or declared != output_declared:
            collector.add(
                "artifact_integrity",
                "manifest_hash_reference_mismatch",
                f"{label}.manifest.{field}",
                f"value={declared!r},output_hash={output_declared!r}",
            )

    declared_files = {name for name in output_hashes if _safe_declared_file(directory, name) is not None}
    direct_files = {path.name for path in directory.iterdir() if path.is_file() and path.name != "manifest.json"}
    if declared_files != direct_files:
        collector.add(
            "artifact_integrity",
            "unbound_direct_output",
            label,
            f"declared={sorted(declared_files)!r},direct={sorted(direct_files)!r}",
        )

    return RunBundle(directory, label, expected_role, manifest, documents)


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, dict) else None


def _validate_runtime_metadata(runtime: Any, label: str, collector: IssueCollector) -> None:
    if not isinstance(runtime, dict):
        return
    required = {
        "hostname",
        "python_version",
        "numpy_version",
        "torch_version",
        "cuda_version",
        "cudnn_version",
        "torch_available",
        "cuda_available",
        "deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "float32_matmul_precision",
        "environment_controls",
        "nvidia_smi_devices",
        "device_count",
        "devices",
    }
    for field in sorted(required):
        if field not in runtime or runtime.get(field) is None:
            collector.add("run_schema", "runtime_field_missing", f"{label}.manifest.runtime.{field}", "required runtime identity is absent")
    for field in ("hostname", "python_version", "numpy_version", "torch_version", "cuda_version", "float32_matmul_precision"):
        value = runtime.get(field)
        if not isinstance(value, str) or not value:
            collector.add("run_schema", "runtime_string_invalid", f"{label}.manifest.runtime.{field}", repr(value))
    if type(runtime.get("cudnn_version")) is not int or runtime.get("cudnn_version") <= 0:
        collector.add("run_schema", "runtime_cudnn_version_invalid", f"{label}.manifest.runtime.cudnn_version", repr(runtime.get("cudnn_version")))
    for field in (
        "torch_available",
        "cuda_available",
        "deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
    ):
        if type(runtime.get(field)) is not bool:
            collector.add("run_schema", "runtime_boolean_invalid", f"{label}.manifest.runtime.{field}", repr(runtime.get(field)))
    if runtime.get("torch_available") is not True or runtime.get("cuda_available") is not True:
        collector.add("run_schema", "runtime_cuda_unavailable", f"{label}.manifest.runtime", "primary/confirmation requires Torch with CUDA")
    expected_determinism = {
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }
    for field, expected in expected_determinism.items():
        if runtime.get(field) != expected:
            collector.add(
                "run_schema",
                "runtime_determinism_mismatch",
                f"{label}.manifest.runtime.{field}",
                f"expected={expected!r},actual={runtime.get(field)!r}",
            )
    environment = runtime.get("environment_controls")
    expected_environment_keys = {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MKL_DYNAMIC",
        "CUBLAS_WORKSPACE_CONFIG",
    }
    if not isinstance(environment, dict) or set(environment) != expected_environment_keys:
        collector.add("run_schema", "runtime_environment_controls_invalid", f"{label}.manifest.runtime.environment_controls", repr(environment))
    else:
        expected_environment = {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MKL_DYNAMIC": "FALSE",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
        if environment != expected_environment:
            collector.add(
                "run_schema",
                "runtime_environment_controls_mismatch",
                f"{label}.manifest.runtime.environment_controls",
                f"expected={expected_environment!r},actual={environment!r}",
            )
    devices = runtime.get("devices")
    smi_devices = runtime.get("nvidia_smi_devices")
    if not isinstance(devices, list) or not devices or any(not isinstance(name, str) or not name for name in devices):
        collector.add("run_schema", "runtime_devices_invalid", f"{label}.manifest.runtime.devices", repr(devices))
    if type(runtime.get("device_count")) is not int or runtime.get("device_count") <= 0:
        collector.add("run_schema", "runtime_device_count_invalid", f"{label}.manifest.runtime.device_count", repr(runtime.get("device_count")))
    elif isinstance(devices, list) and runtime.get("device_count") != len(devices):
        collector.add("run_schema", "runtime_device_count_mismatch", f"{label}.manifest.runtime.device_count", repr(runtime.get("device_count")))
    if not isinstance(smi_devices, list) or not smi_devices:
        collector.add("run_schema", "runtime_smi_devices_invalid", f"{label}.manifest.runtime.nvidia_smi_devices", repr(smi_devices))
    else:
        for index, item in enumerate(smi_devices):
            if not isinstance(item, dict) or type(item.get("index")) is not int or item.get("index") < 0 or not isinstance(item.get("name"), str) or not item.get("name") or not isinstance(item.get("driver_version"), str) or not item.get("driver_version"):
                collector.add("run_schema", "runtime_smi_device_invalid", f"{label}.manifest.runtime.nvidia_smi_devices[{index}]", repr(item))
        count = runtime.get("device_count")
        if type(count) is int and count > 0:
            valid_smi = [
                item
                for item in smi_devices
                if isinstance(item, dict) and type(item.get("index")) is int and item.get("index") >= 0
            ]
            indices = [item["index"] for item in valid_smi]
            if len(valid_smi) != count or len(smi_devices) != count or len(set(indices)) != len(indices) or set(indices) != set(range(count)):
                collector.add(
                    "run_schema",
                    "runtime_smi_device_coverage_mismatch",
                    f"{label}.manifest.runtime.nvidia_smi_devices",
                    f"device_count={count!r},indices={indices!r}",
                )
            if isinstance(devices, list) and len(devices) == count:
                by_index = {item["index"]: item for item in valid_smi}
                for index, name in enumerate(devices):
                    if not isinstance(by_index.get(index), dict) or by_index[index].get("name") != name:
                        collector.add(
                            "run_schema",
                            "runtime_device_name_mismatch",
                            f"{label}.manifest.runtime.devices[{index}]",
                            f"torch={name!r},nvidia_smi={None if index not in by_index else by_index[index].get('name')!r}",
                        )


def _validate_run_schema(
    bundle: RunBundle,
    collector: IssueCollector,
    seeds: Sequence[int] = PRIMARY_SEEDS,
) -> None:
    manifest = bundle.manifest
    label = bundle.label
    expected_seeds = [int(seed) for seed in seeds]
    expected_seed_keys = {str(seed) for seed in expected_seeds}

    scalar_expectations = {
        "schema_version": 3,
        "stage": "phase2_gpu_training",
        "run_role": bundle.expected_role,
        "ordered_seeds": expected_seeds,
        "protocol_sha256": PROTOCOL_SHA256,
    }
    for field, expected in scalar_expectations.items():
        if manifest.get(field) != expected:
            collector.add(
                "run_schema",
                "manifest_field_mismatch",
                f"{label}.manifest.{field}",
                f"expected={expected!r},actual={manifest.get(field)!r}",
            )

    for field in ("source_hashes", "runtime", "data_sha256_by_seed", "split_sha256_by_seed"):
        if not isinstance(manifest.get(field), dict):
            collector.add("run_schema", "manifest_field_not_object", f"{label}.manifest.{field}", "expected object")
    source_hashes = manifest.get("source_hashes")
    if isinstance(source_hashes, dict):
        if set(source_hashes) != CANONICAL_SOURCE_FILES:
            collector.add(
                "run_schema",
                "source_hash_set_mismatch",
                f"{label}.manifest.source_hashes",
                f"expected={sorted(CANONICAL_SOURCE_FILES)!r},actual={sorted(source_hashes)!r}",
            )
        source_root = Path(__file__).resolve().parent
        for name in sorted(CANONICAL_SOURCE_FILES & set(source_hashes)):
            expected = source_hashes.get(name)
            path = source_root / name
            if not _is_sha256(expected) or not path.is_file() or artifacts.sha256_file(path).lower() != str(expected).lower():
                collector.add(
                    "run_schema",
                    "source_hash_current_file_mismatch",
                    f"{label}.manifest.source_hashes.{name}",
                    repr(expected),
                )
    _validate_runtime_metadata(manifest.get("runtime"), label, collector)
    for field in (
        "test_hash",
        "frozen_config_sha256",
        "base_sha256",
        "config_sha256",
    ):
        if not _is_sha256(manifest.get(field)):
            collector.add("run_schema", "invalid_identity_hash", f"{label}.manifest.{field}", repr(manifest.get(field)))
    if bundle.expected_role in {"primary", "confirmation"}:
        if not _is_sha256(manifest.get("calibration_freeze_sha256")):
            collector.add("run_schema", "invalid_identity_hash", f"{label}.manifest.calibration_freeze_sha256", repr(manifest.get("calibration_freeze_sha256")))
    elif manifest.get("calibration_freeze_sha256") is not None:
        collector.add("run_schema", "unexpected_calibration_freeze", f"{label}.manifest.calibration_freeze_sha256", repr(manifest.get("calibration_freeze_sha256")))
    if manifest.get("base_sha256") != _expected_base_sha256():
        collector.add("run_schema", "noncanonical_base_hash", f"{label}.manifest.base_sha256", repr(manifest.get("base_sha256")))
    if manifest.get("run_id") != bundle.directory.name:
        collector.add(
            "run_schema",
            "run_id_directory_mismatch",
            f"{label}.manifest.run_id",
            f"expected={bundle.directory.name!r},actual={manifest.get('run_id')!r}",
        )
    confirmation_of = manifest.get("confirmation_of_manifest_sha256")
    if bundle.expected_role == "primary" and confirmation_of is not None:
        collector.add(
            "run_schema",
            "primary_has_confirmation_target",
            f"{label}.manifest.confirmation_of_manifest_sha256",
            repr(confirmation_of),
        )
    if bundle.expected_role == "confirmation" and not _is_sha256(confirmation_of):
        collector.add(
            "run_schema",
            "confirmation_target_hash_missing",
            f"{label}.manifest.confirmation_of_manifest_sha256",
            repr(confirmation_of),
        )

    sources = _mapping(manifest.get("source_hashes"))
    if sources is not None:
        if not sources or any(not _is_sha256(value) for value in sources.values()):
            collector.add("run_schema", "invalid_source_hashes", f"{label}.manifest.source_hashes", "all values must be SHA-256")
        source_test = sources.get("test_phase2_gpu_bridge.py")
        if source_test != manifest.get("test_hash"):
            collector.add(
                "run_schema",
                "test_hash_not_in_source_hashes",
                f"{label}.manifest.test_hash",
                f"manifest={manifest.get('test_hash')!r},source={source_test!r}",
            )

    canonical_seed_hashes: dict[int, tuple[str, str]] | None = None
    for field in ("data_sha256_by_seed", "split_sha256_by_seed"):
        values = _mapping(manifest.get(field))
        if values is not None:
            if set(values) != expected_seed_keys or any(not _is_sha256(value) for value in values.values()):
                collector.add(
                    "run_schema",
                    "invalid_seed_hash_map",
                    f"{label}.manifest.{field}",
                    "keys must be the exact ordered run seeds and values must be SHA-256",
                )
            else:
                if canonical_seed_hashes is None:
                    canonical_seed_hashes = _expected_seed_hashes(tuple(expected_seeds))
                hash_index = 0 if field == "data_sha256_by_seed" else 1
                expected_values = {
                    str(seed): canonical_seed_hashes[seed][hash_index]
                    for seed in expected_seeds
                }
                if values != expected_values:
                    collector.add(
                        "run_schema",
                        "noncanonical_seed_hash_map",
                        f"{label}.manifest.{field}",
                        "does not match a deterministic rebuild from BridgeConfig()",
                    )

    config = bundle.documents.get("config.json")
    if not isinstance(config, dict):
        collector.add("run_schema", "missing_config_object", f"{label}.config", "strict config object unavailable")
    else:
        for field, expected in (
            ("kind", "phase2_frozen_experiment_config"),
            ("schema_version", 1),
            ("seeds", expected_seeds),
            ("protocol_sha256", PROTOCOL_SHA256),
        ):
            if config.get(field) != expected:
                collector.add(
                    "run_schema",
                    "config_field_mismatch",
                    f"{label}.config.{field}",
                    f"expected={expected!r},actual={config.get(field)!r}",
                )
        if "run_role" in config:
            collector.add(
                "run_schema",
                "config_contains_run_role",
                f"{label}.config.run_role",
                "run_role must remain outside the shared frozen experiment config",
            )
        frozen = config.get("frozen")
        if not isinstance(frozen, dict):
            collector.add("run_schema", "missing_frozen_config", f"{label}.config.frozen", "expected object")
        else:
            canonical_config = BridgeConfig()
            canonical_frozen = canonical_config.payload()
            if frozen.get("primary_seeds") != list(PRIMARY_SEEDS):
                collector.add("run_schema", "frozen_seed_mismatch", f"{label}.config.frozen.primary_seeds", repr(frozen.get("primary_seeds")))
            computed = artifacts.hash_json(frozen).upper()
            if computed != str(manifest.get("frozen_config_sha256", "")).upper():
                collector.add(
                    "run_schema",
                    "frozen_config_hash_mismatch",
                    f"{label}.manifest.frozen_config_sha256",
                    f"computed={computed},declared={manifest.get('frozen_config_sha256')}",
                )
            if frozen != canonical_frozen:
                collector.add(
                    "run_schema",
                    "noncanonical_frozen_config",
                    f"{label}.config.frozen",
                    "frozen payload differs from BridgeConfig() protocol constants",
                )
            if str(manifest.get("frozen_config_sha256", "")).upper() != canonical_config.sha256():
                collector.add(
                    "run_schema",
                    "noncanonical_frozen_config_hash",
                    f"{label}.manifest.frozen_config_sha256",
                    f"expected={canonical_config.sha256()},actual={manifest.get('frozen_config_sha256')!r}",
                )

    split = bundle.documents.get("split_manifest.json")
    if not isinstance(split, dict):
        collector.add("run_schema", "missing_split_manifest_object", f"{label}.split_manifest", "strict object unavailable")
    else:
        if split.get("protocol_sha256") != manifest.get("protocol_sha256"):
            collector.add("run_schema", "split_protocol_mismatch", f"{label}.split_manifest.protocol_sha256", "does not match manifest")
        if split.get("base_sha256") != manifest.get("base_sha256"):
            collector.add("run_schema", "split_base_mismatch", f"{label}.split_manifest.base_sha256", "does not match manifest")
        split_seeds = _mapping(split.get("seeds"))
        if split_seeds is None or set(split_seeds) != expected_seed_keys:
            collector.add("run_schema", "split_seed_set_mismatch", f"{label}.split_manifest.seeds", "expected exact ordered run-seed set")
        else:
            for seed in sorted(expected_seed_keys, key=int):
                entry = _mapping(split_seeds.get(seed))
                if entry is None:
                    continue
                for entry_field, manifest_field in (
                    ("data_sha256", "data_sha256_by_seed"),
                    ("split_sha256", "split_sha256_by_seed"),
                ):
                    mapping = _mapping(manifest.get(manifest_field))
                    if mapping is not None and entry.get(entry_field) != mapping.get(seed):
                        collector.add(
                            "run_schema",
                            "split_seed_hash_mismatch",
                            f"{label}.split_manifest.seeds.{seed}.{entry_field}",
                            "does not match manifest map",
                        )

    summary = bundle.documents.get("summary.json")
    gates = bundle.documents.get("gates.json")
    if not isinstance(gates, dict):
        collector.add("run_schema", "missing_gates_object", f"{label}.gates", "strict gates object unavailable")
    if not isinstance(summary, dict):
        collector.add("run_schema", "missing_summary_object", f"{label}.summary", "strict object unavailable")
    else:
        if summary.get("run_role") != bundle.expected_role:
            collector.add("run_schema", "summary_role_mismatch", f"{label}.summary.run_role", repr(summary.get("run_role")))
        expected_per_window = len(expected_seeds) * len(REGIMES) * len(RANKS) * BridgeConfig().T * len(TRAINABLE_ARMS)
        expected_per_seed = len(expected_seeds) * len(REGIMES) * len(RANKS) * len(TRAINABLE_ARMS)
        if summary.get("per_window_rows") != expected_per_window or summary.get("per_seed_rows") != expected_per_seed:
            collector.add(
                "run_schema",
                "summary_row_count_mismatch",
                f"{label}.summary",
                f"expected {expected_per_window} per-window and {expected_per_seed} per-seed rows",
            )
        if isinstance(gates, dict) and summary.get("gates") != gates:
            collector.add("artifact_integrity", "summary_gate_copy_mismatch", f"{label}.summary.gates", "does not equal gates.json")
        if isinstance(gates, dict):
            gate_pass = {
                name: gates.get(name, {}).get("passed") if isinstance(gates.get(name), dict) else None
                for name in ("G0_numpy_identity", "G1_oracle_capacity", "G2_phenomenon", "G3_bounded_history")
            }
            g4_single = gates.get("G4_resource_and_reproducibility", {}).get("single_run_passed") if isinstance(gates.get("G4_resource_and_reproducibility"), dict) else None
            expected_execution = (
                all(type(value) is bool and value for value in gate_pass.values())
                and type(g4_single) is bool
                and g4_single
            )
            if bundle.expected_role not in {"primary", "confirmation"}:
                expected_execution = (
                    type(gate_pass["G0_numpy_identity"]) is bool
                    and gate_pass["G0_numpy_identity"]
                    and type(gate_pass["G1_oracle_capacity"]) is bool
                    and gate_pass["G1_oracle_capacity"]
                    and type(g4_single) is bool
                    and g4_single
                )
            if type(summary.get("execution_pass")) is not bool or summary.get("execution_pass") != expected_execution:
                collector.add("run_schema", "summary_execution_mismatch", f"{label}.summary.execution_pass", f"expected={expected_execution},actual={summary.get('execution_pass')!r}")
            expected_decision = (
                "AWAITING_CONFIRMATION_AND_POST_RESULT_AUDIT"
                if bundle.expected_role == "primary" and expected_execution
                else "AWAITING_RECONCILIATION_AND_POST_RESULT_AUDIT"
                if bundle.expected_role == "confirmation" and expected_execution
                else "NO-GO"
                if bundle.expected_role in {"primary", "confirmation"}
                else "NON_CITABLE_PASS"
                if expected_execution
                else "NON_CITABLE_FAIL"
            )
            if summary.get("decision") != expected_decision:
                collector.add("run_schema", "summary_decision_mismatch", f"{label}.summary.decision", f"expected={expected_decision!r},actual={summary.get('decision')!r}")

    traces = bundle.documents.get("traces.json")
    if not isinstance(traces, dict):
        collector.add("run_schema", "missing_traces_object", f"{label}.traces", "strict object unavailable")
    else:
        for field in ("protocol_sha256", "base_sha256", "source_hashes"):
            if traces.get(field) != manifest.get(field):
                collector.add("run_schema", "trace_identity_mismatch", f"{label}.traces.{field}", "does not match manifest")
        if traces.get("base_sha256") != _expected_base_sha256():
            collector.add("run_schema", "noncanonical_trace_base", f"{label}.traces.base_sha256", repr(traces.get("base_sha256")))
        expected_controls = manifest.get("runtime", {}).get("environment_controls") if isinstance(manifest.get("runtime"), dict) else None
        if traces.get("deterministic_controls") != expected_controls:
            collector.add(
                "run_schema",
                "trace_deterministic_controls_mismatch",
                f"{label}.traces.deterministic_controls",
                "does not match the manifest runtime controls",
            )
        boundary = traces.get("runtime_learner_boundary")
        expected_boundary_fields = {
            "passed",
            "gradient_model_input_names",
            "gradient_model_input_shape",
            "gradient_model_input_dtype",
            "latent_metadata_tensor_events",
            "replay_sampler_argument_names",
            "permitted_scheduler_fields",
            "permitted_post_run_evaluator_fields",
        }
        if not isinstance(boundary, dict) or set(boundary) != expected_boundary_fields:
            collector.add("run_schema", "runtime_boundary_schema_mismatch", f"{label}.traces.runtime_learner_boundary", repr(boundary))
        elif (
            boundary.get("passed") is not True
            or boundary.get("gradient_model_input_names") != ["x", "y"]
            or boundary.get("gradient_model_input_shape") != [128, 32]
            or boundary.get("gradient_model_input_dtype") != "torch.float32"
            or not _is_nonnegative_int(boundary.get("latent_metadata_tensor_events"))
            or boundary.get("replay_sampler_argument_names") != ["draw_rng", "size"]
            or boundary.get("permitted_scheduler_fields") != ["window", "pi_t", "insertion_boundary"]
            or boundary.get("permitted_post_run_evaluator_fields") != ["window", "mode_ids", "schedule", "split_indices"]
        ):
            collector.add("run_schema", "runtime_boundary_noncanonical", f"{label}.traces.runtime_learner_boundary", repr(boundary))


def _compare_identity(primary: RunBundle, confirmation: RunBundle, collector: IssueCollector) -> None:
    exact_manifest_fields = (
        "schema_version",
        "stage",
        "protocol_sha256",
        "source_hashes",
        "test_hash",
        "frozen_config_sha256",
        "config_sha256",
        "calibration_freeze_sha256",
        "base_sha256",
        "data_sha256_by_seed",
        "split_sha256_by_seed",
        "runtime",
        "resource_ledger",
        "hash_schemas",
        "claim_boundary",
    )
    for field in exact_manifest_fields:
        if primary.manifest.get(field) != confirmation.manifest.get(field):
            collector.add("identity", "manifest_identity_mismatch", f"manifest.{field}", "primary and confirmation differ")

    primary_config = primary.documents.get("config.json")
    confirmation_config = confirmation.documents.get("config.json")
    if isinstance(primary_config, dict) and isinstance(confirmation_config, dict):
        if primary_config != confirmation_config:
            collector.add("identity", "config_identity_mismatch", "config", "primary and confirmation differ")

    primary_manifest_hash = artifacts.sha256_file(primary.directory / "manifest.json")
    if confirmation.manifest.get("confirmation_of_manifest_sha256") != primary_manifest_hash:
        collector.add(
            "identity",
            "confirmation_primary_binding_mismatch",
            "confirmation.manifest.confirmation_of_manifest_sha256",
            f"expected={primary_manifest_hash},actual={confirmation.manifest.get('confirmation_of_manifest_sha256')!r}",
        )

    if primary.manifest.get("split_manifest_sha256") != confirmation.manifest.get("split_manifest_sha256"):
        collector.add("identity", "split_manifest_hash_mismatch", "manifest.split_manifest_sha256", "primary and confirmation differ")

    primary_traces = primary.documents.get("traces.json")
    confirmation_traces = confirmation.documents.get("traces.json")
    if isinstance(primary_traces, dict) and isinstance(confirmation_traces, dict):
        for field in (
            "protocol_sha256",
            "base_sha256",
            "source_hashes",
            "factor_initializations",
            "deterministic_controls",
            "runtime_learner_boundary",
        ):
            if primary_traces.get(field) != confirmation_traces.get(field):
                collector.add("identity", "trace_identity_mismatch", f"traces.{field}", "primary and confirmation differ")


def _strict_int(value: str, location: str) -> int:
    if _INTEGER_RE.fullmatch(value) is None or len(value.lstrip("-")) > _MAX_JSON_INTEGER_DIGITS:
        raise ValueError(f"{location}: expected canonical integer, got {value!r}")
    try:
        return int(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{location}: invalid integer {value!r}") from exc


def _canonical_csv_key(row: Mapping[str, str], key_fields: Sequence[str], location: str) -> tuple[Any, ...]:
    values: list[Any] = []
    for field in key_fields:
        if field not in row or row[field] is None:
            raise ValueError(f"{location}: missing key field {field}")
        raw = row[field]
        if field in {"seed", "R", "window", "completed_window"}:
            values.append(_strict_int(raw, f"{location}.{field}"))
        else:
            if raw == "":
                raise ValueError(f"{location}.{field}: empty key")
            values.append(raw)
    return tuple(values)


def _read_csv_table(
    bundle: RunBundle,
    filename: str,
    key_fields: Sequence[str],
    required_fields: set[str],
    expected_keys: set[tuple[Any, ...]],
    collector: IssueCollector,
) -> CSVTable | None:
    path = bundle.directory / filename
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames is None:
                raise ValueError("missing header")
            header = tuple(reader.fieldnames)
            if len(header) != len(set(header)):
                raise ValueError("duplicate header fields")
            missing_fields = sorted(required_fields - set(header))
            if missing_fields:
                collector.add(
                    "csv_schema",
                    "missing_csv_fields",
                    f"{bundle.label}.{filename}",
                    ",".join(missing_fields),
                )
            extra_fields = sorted(set(header) - required_fields)
            if extra_fields:
                collector.add(
                    "csv_schema",
                    "unexpected_csv_fields",
                    f"{bundle.label}.{filename}",
                    ",".join(extra_fields),
                )
            rows: dict[tuple[Any, ...], dict[str, str]] = {}
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    collector.add("csv_schema", "extra_csv_columns", f"{bundle.label}.{filename}:{row_number}", repr(row[None]))
                    continue
                missing_values = [field for field, value in row.items() if value is None]
                if missing_values:
                    collector.add(
                        "csv_schema",
                        "missing_csv_values",
                        f"{bundle.label}.{filename}:{row_number}",
                        ",".join(sorted(missing_values)),
                    )
                    row = {field: ("" if value is None else value) for field, value in row.items()}
                try:
                    key = _canonical_csv_key(row, key_fields, f"{bundle.label}.{filename}:{row_number}")
                except ValueError as exc:
                    collector.add("csv_schema", "invalid_canonical_key", f"{bundle.label}.{filename}:{row_number}", str(exc))
                    continue
                if key in rows:
                    collector.add("csv_schema", "duplicate_canonical_key", f"{bundle.label}.{filename}:{row_number}", repr(key))
                    continue
                _validate_csv_row_types(filename, row, f"{bundle.label}.{filename}:{row_number}", collector)
                rows[key] = dict(row)
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        collector.add("csv_schema", "invalid_csv", f"{bundle.label}.{filename}", str(exc))
        return None

    actual_keys = set(rows)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing:
        collector.add(
            "csv_schema",
            "missing_canonical_keys",
            f"{bundle.label}.{filename}",
            f"count={len(missing)},first={missing[:3]!r}",
        )
    if extra:
        collector.add(
            "csv_schema",
            "unexpected_canonical_keys",
            f"{bundle.label}.{filename}",
            f"count={len(extra)},first={extra[:3]!r}",
        )
    return CSVTable(filename, header, rows)


def _csv_runtime_value(row: Mapping[str, str], field: str) -> Any:
    raw = row.get(field, "")
    return None if raw == "" else raw


def _validate_resource_identity(bundle: RunBundle, table: CSVTable | None, collector: IssueCollector) -> None:
    if table is None:
        return
    manifest = bundle.manifest
    source_hashes = manifest.get("source_hashes")
    data_by_seed = manifest.get("data_sha256_by_seed")
    split_by_seed = manifest.get("split_sha256_by_seed")
    runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}
    traces = bundle.documents.get("traces.json")
    factor_initializations = traces.get("factor_initializations", {}) if isinstance(traces, dict) else {}
    for key, row in table.rows.items():
        location = f"{bundle.label}.resource_ledger.csv[{key!r}]"
        expected_text = {
            "protocol_sha256": manifest.get("protocol_sha256"),
            "frozen_config_sha256": manifest.get("frozen_config_sha256"),
            "test_sha256": manifest.get("test_hash"),
            "data_sha256": None if not isinstance(data_by_seed, dict) else data_by_seed.get(str(key[0])),
            "split_sha256": None if not isinstance(split_by_seed, dict) else split_by_seed.get(str(key[0])),
            "hostname": runtime.get("hostname"),
            "python_version": runtime.get("python_version"),
            "numpy_version": runtime.get("numpy_version"),
            "torch_version": runtime.get("torch_version"),
            "cuda_version": runtime.get("cuda_version"),
            "cudnn_version": runtime.get("cudnn_version"),
        }
        for field, expected in expected_text.items():
            observed = _csv_runtime_value(row, field)
            if expected is not None and str(observed) != str(expected):
                collector.add("artifact_integrity", "resource_identity_mismatch", f"{location}.{field}", f"expected={expected!r},actual={observed!r}")
        try:
            row_sources = _validate_inline_json(row["source_hashes"], f"{location}.source_hashes")
        except (KeyError, ValueError) as exc:
            collector.add("artifact_integrity", "resource_source_hashes_invalid", f"{location}.source_hashes", str(exc))
        else:
            if row_sources != source_hashes:
                collector.add("artifact_integrity", "resource_source_hashes_mismatch", f"{location}.source_hashes", "does not match manifest")
        expected_deterministic_flags = {
            "deterministic_algorithms": runtime.get("deterministic_algorithms"),
            "cudnn_deterministic": runtime.get("cudnn_deterministic"),
            "cudnn_benchmark": runtime.get("cudnn_benchmark"),
            "cuda_matmul_allow_tf32": runtime.get("cuda_matmul_allow_tf32"),
            "cudnn_allow_tf32": runtime.get("cudnn_allow_tf32"),
            "float32_matmul_precision": runtime.get("float32_matmul_precision"),
            "environment_controls": runtime.get("environment_controls"),
        }
        for field, expected in (
            ("nvidia_smi_devices", runtime.get("nvidia_smi_devices", [])),
            ("deterministic_flags", expected_deterministic_flags),
        ):
            try:
                observed = _validate_inline_json(row[field], f"{location}.{field}")
            except (KeyError, ValueError) as exc:
                collector.add("artifact_integrity", "resource_runtime_json_invalid", f"{location}.{field}", str(exc))
            else:
                if observed != expected:
                    collector.add("artifact_integrity", "resource_runtime_json_mismatch", f"{location}.{field}", "does not match manifest runtime")
        config = bundle.documents.get("config.json")
        if isinstance(config, dict) and row.get("device") != str(config.get("device")):
            collector.add("artifact_integrity", "resource_device_mismatch", f"{location}.device", f"expected={config.get('device')!r},actual={row.get('device')!r}")
        config_device_index: int | None = None
        if isinstance(config, dict):
            config_device = config.get("device")
            if isinstance(config_device, str) and re.fullmatch(r"cuda:[0-9]+", config_device):
                config_device_index = int(config_device.split(":", 1)[1])
            elif str(config_device).startswith("cuda:"):
                collector.add("artifact_integrity", "config_device_invalid", f"{bundle.label}.config.json.device", repr(config_device))
        raw_selected_index = row.get("device_index", "")
        if not isinstance(raw_selected_index, str) or _INTEGER_RE.fullmatch(raw_selected_index) is None:
            collector.add(
                "artifact_integrity",
                "resource_device_index_invalid",
                f"{location}.device_index",
                repr(raw_selected_index),
            )
            selected_index: int | None = None
        else:
            if len(raw_selected_index.lstrip("-")) > _MAX_JSON_INTEGER_DIGITS:
                collector.add(
                    "artifact_integrity",
                    "resource_device_index_invalid",
                    f"{location}.device_index",
                    "integer exceeds strict digit limit",
                )
                selected_index = None
            else:
                try:
                    selected_index = int(raw_selected_index)
                except (ValueError, OverflowError):
                    collector.add(
                        "artifact_integrity",
                        "resource_device_index_invalid",
                        f"{location}.device_index",
                        repr(raw_selected_index),
                    )
                    selected_index = None
            if selected_index is not None and selected_index < 0:
                collector.add(
                    "artifact_integrity",
                    "resource_device_index_invalid",
                    f"{location}.device_index",
                    repr(raw_selected_index),
                )
                selected_index = None
        if config_device_index is not None and selected_index != config_device_index:
            collector.add(
                "artifact_integrity",
                "resource_device_index_config_mismatch",
                f"{location}.device_index",
                f"config cuda index={config_device_index}, observed={selected_index}",
            )
        runtime_devices = runtime.get("nvidia_smi_devices", [])
        if not isinstance(runtime_devices, list):
            collector.add(
                "artifact_integrity",
                "runtime_devices_not_list",
                f"{bundle.label}.manifest.runtime.nvidia_smi_devices",
                type(runtime_devices).__name__,
            )
            runtime_devices = []
        selected: list[Mapping[str, Any]] = []
        seen_runtime_indices: set[int] = set()
        for device_position, item in enumerate(runtime_devices):
            if not isinstance(item, dict):
                collector.add(
                    "artifact_integrity",
                    "runtime_device_not_object",
                    f"{bundle.label}.manifest.runtime.nvidia_smi_devices[{device_position}]",
                    type(item).__name__,
                )
                continue
            runtime_index = item.get("index")
            if type(runtime_index) is not int or runtime_index < 0:
                collector.add(
                    "artifact_integrity",
                    "runtime_device_index_invalid",
                    f"{bundle.label}.manifest.runtime.nvidia_smi_devices[{device_position}].index",
                    repr(runtime_index),
                )
                continue
            if runtime_index in seen_runtime_indices:
                collector.add(
                    "artifact_integrity",
                    "runtime_device_index_duplicate",
                    f"{bundle.label}.manifest.runtime.nvidia_smi_devices[{device_position}].index",
                    repr(runtime_index),
                )
            seen_runtime_indices.add(runtime_index)
            if not isinstance(item.get("name"), str) or not item["name"].strip():
                collector.add(
                    "artifact_integrity",
                    "runtime_device_name_invalid",
                    f"{bundle.label}.manifest.runtime.nvidia_smi_devices[{device_position}].name",
                    repr(item.get("name")),
                )
            if not isinstance(item.get("driver_version"), str) or not item["driver_version"].strip():
                collector.add(
                    "artifact_integrity",
                    "runtime_driver_version_invalid",
                    f"{bundle.label}.manifest.runtime.nvidia_smi_devices[{device_position}].driver_version",
                    repr(item.get("driver_version")),
                )
            if selected_index is not None and runtime_index == selected_index:
                selected.append(item)
        if len(selected) == 1:
            if row.get("gpu_name") != str(selected[0].get("name")):
                collector.add("artifact_integrity", "resource_gpu_name_mismatch", f"{location}.gpu_name", "does not match runtime device")
            if row.get("driver_version") != str(selected[0].get("driver_version")):
                collector.add("artifact_integrity", "resource_driver_mismatch", f"{location}.driver_version", "does not match runtime device")
        else:
            collector.add("artifact_integrity", "resource_device_index_unbound", f"{location}.device_index", "does not select one runtime device")
        factor_key = f"{key[1]}:{key[2]}"
        expected_factor = None
        if isinstance(factor_initializations, dict):
            seed_factors = factor_initializations.get(str(key[0]), {})
            if isinstance(seed_factors, dict) and isinstance(seed_factors.get(factor_key), dict):
                expected_factor = seed_factors[factor_key].get("sha256")
        if expected_factor is None:
            collector.add("artifact_integrity", "resource_factor_hash_unbound", f"{location}.factor_sha256", "factor initialization is missing from traces")
        elif row.get("factor_sha256") != expected_factor:
            collector.add("artifact_integrity", "resource_factor_hash_mismatch", f"{location}.factor_sha256", f"expected={expected_factor!r},actual={row.get('factor_sha256')!r}")


def _finite_float(value: str, location: str, allow_empty: bool = False) -> float | None:
    if value == "" and allow_empty:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: expected float, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{location}: non-finite float {value!r}")
    return parsed


def _reject_inline_constant(token: str) -> Any:
    raise ValueError(f"non-finite JSON token {token!r}")


def _reject_inline_duplicate(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_inline_json(raw: str, location: str) -> Any:
    try:
        value = json.loads(
            raw,
            parse_constant=_reject_inline_constant,
            parse_int=_parse_strict_int,
            object_pairs_hook=_reject_inline_duplicate,
        )
        _assert_finite_json(value)
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"{location}: invalid strict JSON: {exc}") from exc
    return value


def _resource_int(row: Mapping[str, str], field: str, location: str, collector: IssueCollector) -> int | None:
    raw = row.get(field, "")
    if not isinstance(raw, str) or _INTEGER_RE.fullmatch(raw) is None or len(raw.lstrip("-")) > _MAX_JSON_INTEGER_DIGITS:
        collector.add("artifact_integrity", "resource_integer_invalid", f"{location}.{field}", repr(raw))
        return None
    try:
        return int(raw)
    except (ValueError, OverflowError) as exc:
        collector.add("artifact_integrity", "resource_integer_invalid", f"{location}.{field}", str(exc))
        return None


def _resource_json(row: Mapping[str, str], field: str, location: str, collector: IssueCollector) -> Any:
    try:
        return _validate_inline_json(row[field], f"{location}.{field}")
    except (KeyError, ValueError) as exc:
        collector.add("artifact_integrity", "resource_json_invalid", f"{location}.{field}", str(exc))
        return None


def _validate_resource_semantics(bundle: RunBundle, table: CSVTable | None, collector: IssueCollector) -> None:
    if table is None:
        return
    config = BridgeConfig()
    expected_optimizer_contract = {
        "class": "AdamW",
        "parameter_order": ["B", "A"],
        "lr": config.lr,
        "betas": list(config.betas),
        "eps": config.eps,
        "weight_decay": config.weight_decay,
        "amsgrad": False,
        "foreach": False,
        "fused": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "scheduler": None,
    }
    expected_optimizer_hash = artifacts.hash_json(expected_optimizer_contract)
    expected_updates = config.T * config.updates_per_window
    expected_param_bytes = lambda rank: rank * (config.d_in + config.d_out) * 4
    for key, row in table.rows.items():
        location = f"{bundle.label}.resource_ledger.csv[{key!r}]"
        rank = _resource_int(row, "R", location, collector)
        if rank is None:
            continue
        if rank not in RANKS:
            collector.add("artifact_integrity", "resource_rank_invalid", f"{location}.R", repr(rank))
            continue
        if _resource_int(row, "schema_version", location, collector) != 1:
            collector.add("artifact_integrity", "resource_schema_version_invalid", f"{location}.schema_version", repr(row.get("schema_version")))
        for field in ("requested_rank", "active_rank", "deployed_rank"):
            if _resource_int(row, field, location, collector) != rank:
                collector.add("artifact_integrity", "resource_rank_mismatch", f"{location}.{field}", repr(row.get(field)))
        numerical_rank = _resource_int(row, "numerical_rank_max", location, collector)
        if numerical_rank is not None and numerical_rank > rank:
            collector.add("artifact_integrity", "resource_numerical_rank_exceeds_requested", f"{location}.numerical_rank_max", repr(numerical_rank))
        shapes = _resource_json(row, "factor_shapes", location, collector)
        expected_shapes = {"B": [config.d_out, rank], "A": [rank, config.d_in]}
        if shapes != expected_shapes:
            collector.add("artifact_integrity", "resource_factor_shapes_invalid", f"{location}.factor_shapes", repr(shapes))
        parameter_count = rank * (config.d_in + config.d_out)
        parameter_bytes = expected_param_bytes(rank)
        for field, expected in (
            ("trainable_parameter_count", parameter_count),
            ("deployed_parameter_count", parameter_count),
            ("trainable_parameter_bytes", parameter_bytes),
            ("deployed_parameter_bytes", parameter_bytes),
            ("optimizer_state_bytes", 2 * parameter_bytes + 2 * 4),
            ("batch_size", config.batch_size),
            ("updates", expected_updates),
            ("forward", expected_updates),
            ("backward", expected_updates),
        ):
            if _resource_int(row, field, location, collector) != expected:
                collector.add("artifact_integrity", "resource_scalar_contract_mismatch", f"{location}.{field}", f"expected={expected},actual={row.get(field)!r}")
        current_examples = _resource_int(row, "current_examples", location, collector)
        replay_examples = _resource_int(row, "replay_examples", location, collector)
        if current_examples is not None and replay_examples is not None and current_examples + replay_examples != expected_updates * config.batch_size:
            collector.add("artifact_integrity", "resource_example_count_mismatch", f"{location}.current_examples", repr((current_examples, replay_examples)))
        contract = _resource_json(row, "optimizer_contract", location, collector)
        if contract != expected_optimizer_contract:
            collector.add("artifact_integrity", "resource_optimizer_contract_mismatch", f"{location}.optimizer_contract", repr(contract))
        if row.get("optimizer_contract_sha256", "").upper() != expected_optimizer_hash.upper():
            collector.add("artifact_integrity", "resource_optimizer_contract_hash_mismatch", f"{location}.optimizer_contract_sha256", repr(row.get("optimizer_contract_sha256")))
        steps = _resource_json(row, "optimizer_steps", location, collector)
        if steps != [expected_updates, expected_updates]:
            collector.add("artifact_integrity", "resource_optimizer_steps_mismatch", f"{location}.optimizer_steps", repr(steps))
        replay_arm = key[3] in REPLAY_ARMS
        expected_replay = {
            "replay_fraction": 0.5 if replay_arm else 0.0,
            "allocated_capacity": config.replay_capacity if replay_arm else 0,
            "replay_bytes": config.fixed_replay_bytes if replay_arm else 0,
            "replay_x_y_slot_bytes": 90112 if replay_arm else 0,
            "replay_priority_bytes": 2816 if replay_arm else 0,
            "replay_occupancy_bytes": 352 if replay_arm else 0,
            "pointer_counter_bytes": 208 if replay_arm else 0,
            "replay_sample_draws": (config.T - 1) * config.updates_per_window if replay_arm else 0,
            "priority_draws": (config.T - 1) * config.n_train if replay_arm else 0,
        }
        for field, expected in expected_replay.items():
            raw = row.get(field, "")
            if field == "replay_fraction":
                try:
                    observed = float(raw)
                except (TypeError, ValueError):
                    observed = None
                if observed != expected:
                    collector.add("artifact_integrity", "resource_replay_contract_mismatch", f"{location}.{field}", f"expected={expected},actual={raw!r}")
            elif _resource_int(row, field, location, collector) != expected:
                collector.add("artifact_integrity", "resource_replay_contract_mismatch", f"{location}.{field}", f"expected={expected},actual={raw!r}")


_COMMON_INT_FIELDS = {
    "seed", "R", "window", "completed_window", "requested_rank", "active_rank", "deployed_rank",
    "numerical_rank", "numerical_rank_max", "updates", "forward", "backward", "current_examples",
    "replay_examples", "occupied", "unique", "duplicates", "priority_draws", "replay_sample_draws",
    "selection_comparisons", "metadata_leakage_events", "peak_allocated_gpu_bytes_so_far",
    "peak_reserved_gpu_bytes_so_far", "schema_version", "trainable_parameter_count",
    "deployed_parameter_count", "trainable_parameter_bytes", "deployed_parameter_bytes",
    "optimizer_state_bytes", "batch_size", "allocated_capacity", "replay_bytes", "replay_x_y_slot_bytes",
    "replay_priority_bytes", "replay_occupancy_bytes", "pointer_counter_bytes",
    "persistent_controller_sidecar_records", "persistent_controller_sidecar_bytes",
    "controller_transient_records_peak", "controller_transient_python_bytes_peak",
    "controller_transient_numpy_bytes_peak", "curvature_sketch_bytes", "checkpoint_persistent_bytes",
    "evaluator_checkpoint_staging_bytes", "common_W0_cpu_bytes", "common_W0_gpu_bytes",
    "common_cpu_cell_dataset_bytes", "common_cpu_seed_dataset_resident_bytes",
    "common_cpu_determinism_rebuild_increment_bytes", "common_cpu_determinism_rebuild_peak_bytes",
    "common_cpu_batch_staging_bytes", "common_cpu_index_staging_bytes_max",
    "arm_cpu_batch_staging_peak_bytes", "arm_cpu_index_staging_peak_bytes",
    "common_gpu_batch_staging_bytes", "priority_tie_count", "replacement_count",
    "occupied_records_max", "unique_records_max", "replay_sampling_duplicate_count",
    "gpu_baseline_allocated_bytes", "gpu_baseline_reserved_bytes", "peak_allocated_gpu_bytes",
    "peak_reserved_gpu_bytes", "peak_allocated_increment_bytes", "peak_reserved_increment_bytes",
    "inference_peak_allocated_gpu_bytes", "inference_peak_reserved_gpu_bytes",
    "inference_baseline_allocated_bytes", "inference_baseline_reserved_bytes",
    "inference_peak_allocated_increment_bytes", "inference_peak_reserved_increment_bytes",
    "inference_warmup_forwards", "inference_measured_forwards", "device_index",
}
_COMMON_FLOAT_FIELDS = {
    "L_prefix", "L_current", "L_rank", "L_inf", "D_t", "D_current", "capacity_ratio", "online_ratio",
    "current_acquisition_ratio", "gap", "bounded_history_reduction", "selection_gain", "gap_fraction",
    "paired_gap_seq", "paired_gap_clock", "paired_gap_reservoir",
    "final_gap", "final_online_ratio", "final_current_acquisition_ratio", "final_bounded_history_reduction",
    "final_selection_gain", "final_gap_fraction", "controller_cpu_seconds", "replay_assembly_seconds",
    "device_staging_seconds", "train_wall_seconds", "training_wall_seconds", "wall_seconds",
    "inference_latency_mean_batch_seconds", "inference_latency_mean_example_seconds", "replay_fraction",
}
_COMMON_BOOL_FIELDS = {"model_input_contract_passed", "replay_sampler_contract_passed"}
_COMMON_JSON_FIELDS = {
    "factor_shapes", "parameter_breakdown", "optimizer_state_breakdown", "optimizer_steps", "optimizer_contract",
    "source_hashes", "nvidia_smi_devices", "deterministic_flags",
}
_COMMON_HASH_FIELDS = {
    "factor_sha256", "optimizer_contract_sha256", "test_sha256", "data_sha256", "split_sha256",
    "protocol_sha256", "frozen_config_sha256",
}
_OPTIONAL_CSV_FLOAT_FIELDS = {"gap_fraction", "final_gap_fraction", "bounded_history_reduction", "selection_gain", "final_bounded_history_reduction", "final_selection_gain"}


def _validate_csv_row_types(filename: str, row: Mapping[str, str], location: str, collector: IssueCollector) -> None:
    for field, raw in row.items():
        if field is None or raw is None:
            continue
        try:
            if field in _COMMON_INT_FIELDS:
                if _INTEGER_RE.fullmatch(raw) is None or int(raw) < 0:
                    raise ValueError("expected a non-negative canonical integer")
            elif field in _COMMON_FLOAT_FIELDS:
                _finite_float(raw, f"{location}.{field}", allow_empty=field in _OPTIONAL_CSV_FLOAT_FIELDS)
            elif field in _COMMON_BOOL_FIELDS:
                if raw not in {"True", "False"}:
                    raise ValueError("expected canonical True/False")
            elif field in _COMMON_JSON_FIELDS:
                _validate_inline_json(raw, f"{location}.{field}")
            elif field in _COMMON_HASH_FIELDS:
                if not _is_sha256(raw):
                    raise ValueError("expected SHA-256")
            elif field in {"regime"} and raw not in set(REGIMES):
                raise ValueError("unknown regime")
            elif field in {"arm"} and raw not in set(TRAINABLE_ARMS):
                raise ValueError("unknown arm")
            elif field in {"validity", "final_validity"} and raw not in {"valid", "invalid_denominator", "risk_inconsistency", "indeterminate_gap"}:
                raise ValueError("unknown validity status")
        except ValueError as exc:
            collector.add("csv_schema", "invalid_csv_field_type", f"{location}.{field}", str(exc))


_NONDETERMINISTIC_PEAK_FIELDS = {
    "gpu_baseline_allocated_bytes",
    "gpu_baseline_reserved_bytes",
    "peak_allocated_gpu_bytes",
    "peak_reserved_gpu_bytes",
    "peak_allocated_increment_bytes",
    "peak_reserved_increment_bytes",
    "inference_peak_allocated_gpu_bytes",
    "inference_peak_reserved_gpu_bytes",
    "inference_peak_allocated_increment_bytes",
    "inference_peak_reserved_increment_bytes",
    "peak_allocated_gpu_bytes_so_far",
    "peak_reserved_gpu_bytes_so_far",
}


def _is_timing_or_peak(field: str) -> bool:
    lowered = field.lower()
    return (
        "wall" in lowered
        or "latency" in lowered
        or "seconds" in lowered
        or lowered.endswith("_time")
        or lowered in _NONDETERMINISTIC_PEAK_FIELDS
    )


def _compare_csv_tables(
    primary: CSVTable | None,
    confirmation: CSVTable | None,
    scalar_tolerances: Mapping[str, float],
    collector: IssueCollector,
) -> dict[str, Any]:
    stats = {
        "rows_compared": 0,
        "oracle_max_abs_delta": 0.0,
        "trained_max_abs_delta": 0.0,
        "timing_or_peak_fields_checked": 0,
    }
    if primary is None or confirmation is None:
        return stats
    if primary.header != confirmation.header:
        collector.add("csv_schema", "csv_header_mismatch", primary.name, "primary and confirmation headers differ")
    common_keys = sorted(set(primary.rows) & set(confirmation.rows))
    for key in common_keys:
        stats["rows_compared"] += 1
        left = primary.rows[key]
        right = confirmation.rows[key]
        for field in sorted(set(left) | set(right)):
            location = f"{primary.name}[{key!r}].{field}"
            if field not in left or field not in right:
                collector.add("csv_reproducibility", "csv_field_presence_mismatch", location, "field missing from one run")
                continue
            left_raw = left[field]
            right_raw = right[field]
            if _is_timing_or_peak(field):
                try:
                    left_value = _finite_float(left_raw, f"primary.{location}")
                    right_value = _finite_float(right_raw, f"confirmation.{location}")
                    if left_value is not None and left_value < 0.0:
                        raise ValueError(f"primary.{location}: negative value")
                    if right_value is not None and right_value < 0.0:
                        raise ValueError(f"confirmation.{location}: negative value")
                    stats["timing_or_peak_fields_checked"] += 1
                except ValueError as exc:
                    collector.add("csv_reproducibility", "invalid_timing_or_peak", location, str(exc))
                continue
            if field in scalar_tolerances:
                allow_empty = field in OPTIONAL_FLOAT_FIELDS
                try:
                    left_value = _finite_float(left_raw, f"primary.{location}", allow_empty)
                    right_value = _finite_float(right_raw, f"confirmation.{location}", allow_empty)
                except ValueError as exc:
                    collector.add("csv_reproducibility", "invalid_scalar", location, str(exc))
                    continue
                if left_value is None or right_value is None:
                    if left_value is not right_value:
                        collector.add("csv_reproducibility", "optional_scalar_presence_mismatch", location, "one run is empty")
                    continue
                delta = abs(left_value - right_value)
                tolerance = scalar_tolerances[field]
                if not math.isfinite(delta):
                    collector.add(
                        "csv_reproducibility",
                        "scalar_delta_nonfinite",
                        location,
                        f"primary={left_value!r},confirmation={right_value!r}",
                    )
                    continue
                stat_field = "oracle_max_abs_delta" if tolerance == ORACLE_ATOL else "trained_max_abs_delta"
                stats[stat_field] = max(float(stats[stat_field]), delta)
                if delta > tolerance:
                    collector.add(
                        "csv_reproducibility",
                        "scalar_tolerance_exceeded",
                        location,
                        f"delta={delta:.17g},tolerance={tolerance:.17g}",
                    )
                continue
            if left_raw != right_raw:
                collector.add(
                    "csv_reproducibility",
                    "exact_field_mismatch",
                    location,
                    f"primary={left_raw!r},confirmation={right_raw!r}",
                )
    return stats


def _json_path_is_timing(path: tuple[str, ...]) -> bool:
    return bool(path) and _is_timing_or_peak(path[-1])


def _json_path_is_oracle(path: tuple[str, ...]) -> bool:
    text = ".".join(path).lower()
    # Risk-failure gaps are trained outputs, even though they live under the
    # G0 gate object.  Keep them on TRAINED_ATOL so a tiny measured variation
    # cannot be misclassified as a scientific gate identity failure.
    if "risk_consistency_failures" in text and path and path[-1].lower() == "gap":
        return False
    return (
        "g0_numpy_identity" in text
        or "g1_oracle_capacity" in text
        or "oracle" in text
        or "capacity" in text
        or "q_min" in text
        or "contraction" in text
        or "sanity" in text
        or "analytic_tail" in text
    )


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _derived_mismatch(collector: IssueCollector, location: str, message: str) -> None:
    collector.add("artifact_integrity", "derived_value_mismatch", location, message)


def _compare_derived_value(
    observed: Any,
    expected: Any,
    location: str,
    collector: IssueCollector,
    tolerance: float,
) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            _derived_mismatch(collector, location, f"expected object, got {type(observed).__name__}")
            return
        if set(observed) != set(expected):
            _derived_mismatch(
                collector,
                location,
                f"field sets differ: expected={sorted(expected)!r},actual={sorted(observed)!r}",
            )
        for key in sorted(set(observed) & set(expected)):
            _compare_derived_value(observed[key], expected[key], f"{location}.{key}", collector, tolerance)
        return
    if isinstance(expected, list):
        if not isinstance(observed, list):
            _derived_mismatch(collector, location, f"expected list, got {type(observed).__name__}")
            return
        if len(observed) != len(expected):
            _derived_mismatch(collector, location, f"length expected={len(expected)},actual={len(observed)}")
        for index, (left, right) in enumerate(zip(observed, expected)):
            _compare_derived_value(left, right, f"{location}[{index}]", collector, tolerance)
        return
    if isinstance(expected, float):
        if not _is_finite_number(observed):
            _derived_mismatch(collector, location, f"expected finite number, got {observed!r}")
            return
        delta = abs(float(observed) - expected)
        if not math.isfinite(delta) or delta > tolerance:
            _derived_mismatch(collector, location, f"expected={expected!r},actual={observed!r},tolerance={tolerance}")
        return
    if type(observed) is not type(expected) or observed != expected:
        _derived_mismatch(collector, location, f"expected={expected!r},actual={observed!r}")


@functools.lru_cache(maxsize=1)
def _expected_g1_gate() -> dict[str, Any]:
    config = BridgeConfig()
    base = make_base_draws(config)
    sanity = run_all_sanity(base, config)
    oracle_rows: list[dict[str, Any]] = []
    oracle_mismatch = 0
    max_oracle_delta = 0.0
    max_analytic_tail_delta = 0.0
    for regime in REGIMES:
        for rank in RANKS:
            for window in range(1, config.T + 1):
                oracle = build_prefix_oracle(base, regime, rank, window)
                U, V = base.basis(regime, rank)
                weighted_delta, weighted_selected, _ = weighted_modal_svd(U, V, oracle.alpha, oracle.q, rank)
                direct_weighted_loss = population_risk(weighted_delta, U, V, oracle.alpha, oracle.q)
                delta = max(
                    abs(float(direct_weighted_loss - oracle.loss_rank)),
                    abs(float(len(weighted_selected) - len(oracle.selected))),
                )
                max_oracle_delta = max(max_oracle_delta, delta)
                max_analytic_tail_delta = max(
                    max_analytic_tail_delta,
                    abs(float(oracle.loss_rank - oracle.loss_inf - oracle.capacity_tail)),
                )
                if weighted_selected != oracle.selected or abs(direct_weighted_loss - oracle.loss_rank) > 1e-10:
                    oracle_mismatch += 1
                oracle_rows.append(
                    {
                        "regime": regime,
                        "rank": rank,
                        "window": window,
                        "capacity_ratio": oracle.capacity_tail / oracle.D_t,
                        "oracle_loss": oracle.loss_rank,
                        "inf_loss": oracle.loss_inf,
                    }
                )
    support_failures = sum(
        1
        for result in sanity.values()
        for selected, support in zip(result.pre_selected, result.post_support)
        if selected != result.oracle.selected or support != result.oracle.selected
    )
    residual_failures = sum(
        1 for result in sanity.values() for residual, tau in zip(result.residual_norms, result.taus) if residual > tau
    )
    max_oracle_error = max((result.final_normalized_error for result in sanity.values()), default=float("nan"))
    q_min = q_min_keep(sanity)
    passed_capacity = all(
        row["capacity_ratio"] <= 1e-10 + 1e-15 for row in oracle_rows if row["regime"] == "compatible"
    )
    passed_bound = sanity_contraction_bound(q_min) < 1.2e-19
    passed = bool(
        passed_capacity
        and passed_bound
        and q_min == 1.0 / 12.0
        and support_failures == 0
        and residual_failures == 0
        and max_oracle_error <= 1e-4
        and oracle_mismatch == 0
        and max_oracle_delta <= 1e-10
        and max_analytic_tail_delta <= 1e-10
    )
    return {
        "passed": passed,
        "oracle_rows": oracle_rows,
        "q_min_keep": q_min,
        "q_min_keep_expected": 1.0 / 12.0,
        "contraction_bound": sanity_contraction_bound(q_min),
        "contraction_bound_limit": 1.2e-19,
        "support_failures": support_failures,
        "residual_failures": residual_failures,
        "max_sanity_normalized_error": max_oracle_error,
        "passed_support": support_failures == 0 and residual_failures == 0,
        "passed_error": max_oracle_error <= 1e-4,
        "weighted_modal_svd_mismatches": oracle_mismatch,
        "max_weighted_modal_svd_delta": max_oracle_delta,
        "max_analytic_tail_delta": max_analytic_tail_delta,
        "passed_oracle_agreement": oracle_mismatch == 0 and max_oracle_delta <= 1e-10,
        "passed_capacity": passed_capacity,
        "passed_contraction_bound": passed_bound,
    }


@functools.lru_cache(maxsize=1)
def _expected_oracle_cells() -> dict[tuple[str, int, int], dict[str, float]]:
    """Recompute the immutable per-cell oracle scalars once per process."""

    config = BridgeConfig()
    base = make_base_draws(config)
    result: dict[tuple[str, int, int], dict[str, float]] = {}
    for regime in REGIMES:
        for rank in RANKS:
            for window in range(1, config.T + 1):
                oracle = build_prefix_oracle(base, regime, rank, window)
                result[(regime, rank, window)] = {
                    "L_rank": float(oracle.loss_rank),
                    "L_inf": float(oracle.loss_inf),
                    "D_t": float(oracle.D_t),
                    "D_current": float(oracle.D_current),
                    "capacity_ratio": float(oracle.capacity_tail / oracle.D_t),
                }
    return result


@functools.lru_cache(maxsize=None)
def _expected_factor_hashes(seeds: tuple[int, ...]) -> dict[tuple[int, str, int], str]:
    """Recompute factor identities from the frozen seed streams.

    The run artifacts carry these hashes for provenance, but the validator must
    derive them independently so a consistently forged traces/resource table
    cannot establish a false shared-factor identity.
    """

    config = BridgeConfig()
    result: dict[tuple[int, str, int], str] = {}
    for seed in seeds:
        initializations = make_factor_initializations(make_seed_streams(seed, config), config)
        for regime in REGIMES:
            for rank in RANKS:
                result[(int(seed), regime, int(rank))] = initializations[(regime, rank)].sha256()
    return result


@functools.lru_cache(maxsize=1)
def _expected_base_sha256() -> str:
    return make_base_draws(BridgeConfig()).sha256()


@functools.lru_cache(maxsize=None)
def _expected_seed_hash(seed: int) -> tuple[str, str]:
    config = BridgeConfig()
    base = make_base_draws(config)
    dataset = build_seed_dataset(base, int(seed), config)
    return dataset.data_sha256, dataset.split_sha256


def _expected_seed_hashes(seeds: tuple[int, ...]) -> dict[int, tuple[str, str]]:
    return {int(seed): _expected_seed_hash(int(seed)) for seed in seeds}


@functools.lru_cache(maxsize=None)
def _expected_replay_transitions(seeds: tuple[int, ...]) -> dict[tuple[int, str, int, str, int], dict[str, Any]]:
    """Rebuild replay chronology from the frozen independent insertion streams."""

    config = BridgeConfig()
    zeros = np.zeros((config.n_train, config.d_in), dtype=np.float32)
    rank_codes = {rank: index for index, rank in enumerate(RANKS)}
    regime_codes = {regime: index for index, regime in enumerate(REGIMES)}
    result: dict[tuple[int, str, int, str, int], dict[str, Any]] = {}
    for seed in seeds:
        streams = make_seed_streams(int(seed), config)
        for arm, controller_type, insertion_rng in (
            (REPLAY_ARMS[0], ClockBalancedReplay, streams.clock_insert),
            (REPLAY_ARMS[1], ReservoirReplay, streams.reservoir_priority),
        ):
            for regime in REGIMES:
                for rank in RANKS:
                    controller = controller_type(regime_code=regime_codes[regime], rank_code=rank_codes[rank])
                    for window in range(1, config.T):
                        before_priority = int(controller.priority_draws)
                        before_comparisons = int(controller.comparison_count)
                        transition = controller.insert_completed_window(window, zeros, zeros, insertion_rng)
                        key = (int(seed), regime, int(rank), arm, window)
                        result[key] = {
                            "seed": int(seed),
                            "regime": regime,
                            "R": int(rank),
                            "arm": arm,
                            "completed_window": window,
                            "history_windows": int(transition.history_windows),
                            "capacity": int(transition.capacity),
                            "eligible_count": int(transition.eligible_count),
                            "occupied_count": int(transition.occupied_count),
                            "unique_count": int(transition.unique_count),
                            "duplicate_count": int(transition.duplicate_count),
                            "replacement_count": int(transition.replacement_count),
                            "priority_draws": int(controller.priority_draws - before_priority),
                            "selection_comparisons": int(controller.comparison_count - before_comparisons),
                            "priority_tie_count": int(transition.priority_tie_count),
                            "selected_priority_order_sha256": transition.selected_priority_order_sha256,
                        }
    return result


def _replay_transition_matches(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    """Check the deterministic replay payload, leaving only timing variable."""

    if any(observed.get(field) != expected.get(field) for field in _REPLAY_DETERMINISTIC_FIELDS):
        return False
    elapsed = observed.get("controller_cpu_seconds")
    return _is_finite_number(elapsed) and float(elapsed) >= 0.0


def _table_float(row: Mapping[str, str], field: str, location: str) -> float:
    value = _finite_float(row.get(field, ""), f"{location}.{field}")
    if value is None:
        raise ValueError(f"{location}.{field}: value is unavailable")
    return value


def _table_bool(row: Mapping[str, str], field: str, location: str) -> bool:
    value = row.get(field)
    if value not in {"True", "False"}:
        raise ValueError(f"{location}.{field}: invalid boolean {value!r}")
    return value == "True"


def _compare_table_scalar(
    row: Mapping[str, str],
    field: str,
    expected: float | None,
    location: str,
    collector: IssueCollector,
    tolerance: float = TRAINED_ATOL,
) -> None:
    raw = row.get(field, "")
    try:
        observed = _finite_float(raw, f"{location}.{field}", allow_empty=expected is None)
    except ValueError as exc:
        _derived_mismatch(collector, f"{location}.{field}", str(exc))
        return
    if expected is None:
        if observed is not None:
            _derived_mismatch(collector, f"{location}.{field}", f"expected empty, got {observed!r}")
        return
    if observed is None or abs(observed - expected) > tolerance:
        _derived_mismatch(collector, f"{location}.{field}", f"expected={expected!r},actual={observed!r}")


def _derive_scientific_gates(
    bundle: RunBundle,
    per_window: CSVTable | None,
    per_seed: CSVTable | None,
    collector: IssueCollector,
    seeds: Sequence[int] = PRIMARY_SEEDS,
    compare_scientific_gates: bool = True,
) -> None:
    if per_window is None or per_seed is None:
        return
    config = BridgeConfig()
    expected_seeds = tuple(int(seed) for seed in seeds)
    try:
        oracle_cells = _expected_oracle_cells()
        derived_metrics: dict[tuple[Any, ...], dict[str, Any]] = {}
        for key, row in per_window.rows.items():
            seed, regime, rank, window, arm = key
            oracle = oracle_cells[(regime, rank, window)]
            location = f"{bundle.label}.per_window.csv[{key!r}]"
            l_prefix = _table_float(row, "L_prefix", location)
            l_current = _table_float(row, "L_current", location)
            gap = l_prefix - oracle["L_rank"]
            online_ratio = gap / oracle["D_t"] if oracle["D_t"] >= 1e-8 else None
            current_acquisition = l_current / oracle["D_current"] if oracle["D_current"] >= 1e-8 else None
            expected_validity = (
                "invalid_denominator"
                if oracle["D_t"] < 1e-8 or oracle["D_current"] < 1e-8
                else "risk_inconsistency"
                if gap < -1e-10
                else "indeterminate_gap"
                if gap < 1e-8
                else "valid"
            )
            for field, expected in oracle.items():
                _compare_table_scalar(row, field, expected, location, collector, ORACLE_ATOL)
            for field, expected in (
                ("gap", gap),
                ("online_ratio", online_ratio),
                ("current_acquisition_ratio", current_acquisition),
            ):
                _compare_table_scalar(row, field, expected, location, collector, TRAINED_ATOL)
            if row.get("validity") != expected_validity:
                _derived_mismatch(collector, f"{location}.validity", f"expected={expected_validity!r},actual={row.get('validity')!r}")
            if row.get("requested_rank") not in {str(rank), rank}:
                _derived_mismatch(collector, f"{location}.requested_rank", f"expected={rank},actual={row.get('requested_rank')!r}")
            derived_metrics[key] = {
                **oracle,
                "gap": gap,
                "online_ratio": online_ratio,
                "current_acquisition_ratio": current_acquisition,
                "validity": expected_validity,
            }
        acquisition: dict[str, dict[str, Any]] = {}
        for arm in TRAINABLE_ARMS:
            values = [
                derived_metrics[key]["current_acquisition_ratio"]
                for key in per_window.rows
                if key[1] == "compatible" and key[2] == 4 and key[4] == arm
            ]
            acquisition[arm] = {
                "count": int(sum(value <= 0.05 for value in values)),
                "total": len(values),
                "threshold": 228,
                "status": "primary_only",
            }

        derived_by_cell: dict[tuple[int, str, int, int], tuple[float, float, float | None]] = {}
        final_rows: dict[int, dict[str, Mapping[str, str]]] = {}
        for seed, regime, rank, window in itertools.product(expected_seeds, REGIMES, RANKS, range(1, config.T + 1)):
            rows = {
                arm: per_window.rows[(seed, regime, rank, window, arm)]
                for arm in TRAINABLE_ARMS
            }
            cell_key = (seed, regime, rank, window)
            seq_gap = derived_metrics[(seed, regime, rank, window, TRAINABLE_ARMS[0])]["gap"]
            clock_gap = derived_metrics[(seed, regime, rank, window, TRAINABLE_ARMS[1])]["gap"]
            reservoir_gap = derived_metrics[(seed, regime, rank, window, TRAINABLE_ARMS[2])]["gap"]
            denominator = oracle_cells[(regime, rank, window)]["D_t"]
            if denominator < 1e-8:
                raise ValueError(f"non-positive paired denominator for {(seed, regime, rank, window)!r}")
            bounded = (seq_gap - clock_gap) / denominator
            selection = (reservoir_gap - clock_gap) / denominator
            fraction = None if seq_gap < 1e-8 else (seq_gap - clock_gap) / seq_gap
            derived_by_cell[cell_key] = (bounded, selection, fraction)
            for arm, row in rows.items():
                location = f"{bundle.label}.per_window.csv[{(seed, regime, rank, window, arm)!r}]"
                for field, expected in (
                    ("bounded_history_reduction", bounded),
                    ("selection_gain", selection),
                    ("gap_fraction", fraction),
                    ("paired_gap_seq", seq_gap),
                    ("paired_gap_clock", clock_gap),
                    ("paired_gap_reservoir", reservoir_gap),
                ):
                    _compare_table_scalar(row, field, expected, location, collector)
            if regime == "compatible" and rank == 4 and window == config.T:
                final_rows[seed] = rows

        if set(final_rows) != set(expected_seeds):
            raise ValueError("paired final row set is incomplete")
        seq_ratios: list[float] = []
        bounded_values: list[float] = []
        selection_values: list[float] = []
        seq_gaps: list[float] = []
        clock_gaps: list[float] = []
        reservoir_gaps: list[float] = []
        finite = True
        for seed in expected_seeds:
            rows = final_rows[seed]
            seq_key = (seed, "compatible", 4, config.T, TRAINABLE_ARMS[0])
            seq_ratios.append(float(derived_metrics[seq_key]["online_ratio"]))
            bounded, selection, _ = derived_by_cell[(seed, "compatible", 4, config.T)]
            bounded_values.append(bounded)
            selection_values.append(selection)
            seq_gaps.append(float(derived_metrics[seq_key]["gap"]))
            clock_gaps.append(float(derived_metrics[(seed, "compatible", 4, config.T, TRAINABLE_ARMS[1])]["gap"]))
            reservoir_gaps.append(float(derived_metrics[(seed, "compatible", 4, config.T, TRAINABLE_ARMS[2])]["gap"]))
            for arm, row in rows.items():
                metrics = derived_metrics[(seed, "compatible", 4, config.T, arm)]
                finite = finite and all(
                    value is not None and math.isfinite(float(value))
                    for value in (metrics["gap"], metrics["online_ratio"], metrics["current_acquisition_ratio"], metrics["D_t"])
                )

        # Scientific bootstrap gates are defined only for the frozen 20-seed
        # primary population.  Calibration and smoke artifacts deliberately
        # use smaller seed sets, but still need all per-window/per-seed
        # formula and provenance checks above and below this block.
        if compare_scientific_gates:
            if expected_seeds != tuple(PRIMARY_SEEDS):
                raise ValueError("scientific gate comparison requires the exact frozen primary seeds")
            valid_fraction = [
                (seq_gap - clock_gap) / seq_gap
                for seq_gap, clock_gap in zip(seq_gaps, clock_gaps)
                if seq_gap >= 1e-8
            ]
            bootstrap = bootstrap_summary(
                {
                    "sequential_online_ratio_T": seq_ratios,
                    "bounded_history_reduction_T": bounded_values,
                    "selection_gain_T": selection_values,
                },
                valid_fraction if valid_fraction else None,
                config,
            )
            acquisition_ok = {
                arm: acquisition[arm]["count"] >= acquisition[arm]["threshold"]
                for arm in TRAINABLE_ARMS
            }
            sequential_lower = bootstrap["statistics"]["sequential_online_ratio_T"]["lower_bound_05"]
            expected_g2 = {
                "status": "evaluated",
                "acquisition": acquisition,
                "paired_final_rows_complete": True,
                "bootstrap": bootstrap,
                "sequential_online_ratio_lower_bound": sequential_lower,
                "sequential_online_ratio_threshold": 0.10,
                "acquisition_ok": acquisition_ok,
                "passed": bool(
                    sequential_lower >= 0.10
                    and all(acquisition_ok.values())
                    and finite
                    and all(acquisition[arm]["total"] == 240 for arm in TRAINABLE_ARMS)
                ),
            }
            valid_count = len(valid_fraction)
            risk_failures = int(sum(gap < -1e-10 for gap in seq_gaps))
            gap_stats = bootstrap["statistics"]["gap_fraction"]
            expected_g3 = {
                "status": "evaluated",
                "paired_final_rows_complete": True,
                "bootstrap": bootstrap,
                "valid_positive_gap_count": valid_count,
                "indeterminate_count": int(sum(0.0 < gap < 1e-8 for gap in seq_gaps)),
                "nonpositive_count": int(sum(gap <= 0.0 for gap in seq_gaps)),
                "minimum_valid_count": 16,
                "gap_fraction_median": gap_stats.get("median"),
                "gap_fraction_bootstrap_lower_bound": gap_stats.get("bootstrap_lower_bound_05"),
                "risk_consistency_failures": risk_failures,
                "passed": bool(
                    bootstrap["statistics"]["bounded_history_reduction_T"]["lower_bound_05"] > 0.0
                    and valid_count >= 16
                    and gap_stats.get("median") is not None
                    and gap_stats["median"] >= 0.50
                    and risk_failures == 0
                    and finite
                ),
                "selection_gain_lower_bound": bootstrap["statistics"]["selection_gain_T"]["lower_bound_05"],
            }
    except (ArithmeticError, KeyError, TypeError, ValueError, OverflowError) as exc:
        collector.add("artifact_integrity", "gate_rederivation_failed", f"{bundle.label}.per_window.csv", str(exc))
        return

    gates = bundle.documents.get("gates.json")
    if isinstance(gates, dict) and compare_scientific_gates:
        _compare_derived_value(gates.get("G2_phenomenon"), expected_g2, f"{bundle.label}.gates.G2_phenomenon", collector, TRAINED_ATOL)
        _compare_derived_value(gates.get("G3_bounded_history"), expected_g3, f"{bundle.label}.gates.G3_bounded_history", collector, TRAINED_ATOL)

    try:
        for key, summary_row in per_seed.rows.items():
            window_key = (key[0], key[1], key[2], config.T, key[3])
            window_row = per_window.rows.get(window_key)
            if window_row is None:
                _derived_mismatch(collector, f"{bundle.label}.per_seed.csv[{key!r}]", "missing final per-window row")
                continue
            for summary_field, window_field in (
                ("final_gap", "gap"),
                ("final_online_ratio", "online_ratio"),
                ("final_current_acquisition_ratio", "current_acquisition_ratio"),
                ("final_bounded_history_reduction", "bounded_history_reduction"),
                ("final_selection_gain", "selection_gain"),
                ("final_gap_fraction", "gap_fraction"),
            ):
                raw = window_row.get(window_field, "")
                try:
                    expected = None if raw == "" else _finite_float(raw, f"{window_key!r}.{window_field}")
                except (TypeError, ValueError, OverflowError) as exc:
                    _derived_mismatch(collector, f"{bundle.label}.per_window.csv[{window_key!r}].{window_field}", str(exc))
                    continue
                _compare_table_scalar(summary_row, summary_field, expected, f"{bundle.label}.per_seed.csv[{key!r}]", collector)
            if summary_row.get("final_validity") != window_row.get("validity"):
                _derived_mismatch(collector, f"{bundle.label}.per_seed.csv[{key!r}].final_validity", "does not match window 12 validity")
    except (ArithmeticError, KeyError, TypeError, ValueError, OverflowError) as exc:
        collector.add("artifact_integrity", "per_seed_rederivation_failed", f"{bundle.label}.per_seed.csv", str(exc))


def _native_resource_row(row: Mapping[str, str], location: str) -> dict[str, Any]:
    native: dict[str, Any] = {}
    for field in RESOURCE_REQUIRED_FIELDS:
        if field not in row or row[field] is None:
            raise ValueError(f"{location}.{field}: missing value")
        raw = row[field]
        if field in _COMMON_INT_FIELDS:
            native[field] = _strict_int(raw, f"{location}.{field}")
        elif field in _COMMON_FLOAT_FIELDS:
            value = _finite_float(raw, f"{location}.{field}")
            if value is None:
                raise ValueError(f"{location}.{field}: missing float")
            native[field] = value
        elif field in _COMMON_BOOL_FIELDS:
            native[field] = _table_bool(row, field, location)
        elif field in _COMMON_JSON_FIELDS:
            native[field] = _validate_inline_json(raw, f"{location}.{field}")
        else:
            native[field] = raw
    return native


def _split_manifest_semantics(bundle: RunBundle, seeds: Sequence[int] = PRIMARY_SEEDS) -> bool:
    config = BridgeConfig()
    expected_seeds = tuple(int(seed) for seed in seeds)
    split = bundle.documents.get("split_manifest.json")
    if not isinstance(split, dict) or set(split) != {"protocol_sha256", "base_sha256", "seeds"}:
        return False
    seed_payload = split.get("seeds")
    if not isinstance(seed_payload, dict):
        return False
    try:
        expected_base = make_base_draws(config)
        if split.get("protocol_sha256") != PROTOCOL_SHA256 or split.get("base_sha256") != expected_base.sha256():
            return False
        # Reuse the runner's canonical serializer so every recorded split detail
        # (IDs, permutations, signs, casts, and per-window hashes) is bound to
        # the deterministic dataset rather than merely to self-consistent flags.
        from .phase2_gpu_bridge import _split_manifest_entry  # local import avoids module initialization cycles
        for seed in expected_seeds:
            rebuilt = build_seed_dataset(expected_base, seed, config)
            observed_entry = seed_payload.get(str(seed))
            if not isinstance(observed_entry, dict) or observed_entry != _split_manifest_entry(rebuilt, expected_base, config):
                return False
    except (ImportError, KeyError, TypeError, ValueError, ArithmeticError, OverflowError):
        return False
    seeds = split.get("seeds")
    if not isinstance(seeds, dict) or set(seeds) != {str(seed) for seed in expected_seeds}:
        return False
    expected_cells = {f"{regime}:{rank}" for regime in REGIMES for rank in RANKS}
    required_block_flags = (
        "mode_counts_match",
        "balanced_signs_passed",
        "record_ids_passed",
        "global_key_schema_passed",
        "storage_permutation_passed",
        "finite_arrays_passed",
        "float32_casts_passed",
    )
    for seed in expected_seeds:
        entry = seeds.get(str(seed))
        if not isinstance(entry, dict) or entry.get("seed") != seed:
            return False
        if any(entry.get(field) is not True for field in ("deterministic_rebuild_passed", "base_finite_passed", "basis_orthogonality_passed")):
            return False
        errors = entry.get("basis_orthogonality_errors")
        if not isinstance(errors, dict) or not errors or any(not _is_finite_number(value) or value > 1e-10 for value in errors.values()):
            return False
        cells = entry.get("cells")
        if not isinstance(cells, dict) or set(cells) != expected_cells:
            return False
        for cell_key, cell in cells.items():
            if not isinstance(cell, dict) or cell.get("windows") != config.T:
                return False
            try:
                regime, rank_text = str(cell_key).split(":", 1)
                rank = int(rank_text)
            except (TypeError, ValueError):
                return False
            if regime not in REGIMES or rank not in RANKS:
                return False
            if cell.get("train_counts") != [config.n_train] * config.T or cell.get("eval_counts") != [config.n_eval] * config.T:
                return False
            if cell.get("train_eval_global_key_overlap") != [0] * config.T:
                return False
            expected_mode_counts = []
            for window in range(1, config.T + 1):
                count = mode_count(regime, rank)
                current, previous = mode_schedule(regime, rank, window)
                train_expected = [0] * count
                eval_expected = [0] * count
                if window == 1:
                    train_expected[current] = config.n_train
                    eval_expected[current] = config.n_eval
                else:
                    train_expected[current] = 3 * config.n_train // 4
                    train_expected[previous] = config.n_train // 4
                    eval_expected[current] = 3 * config.n_eval // 4
                    eval_expected[previous] = config.n_eval // 4
                expected_mode_counts.append({"train": train_expected, "eval": eval_expected})
            if cell.get("mode_counts") != expected_mode_counts:
                return False
            hashes = cell.get("pi_hashes")
            if not isinstance(hashes, list) or len(hashes) != config.T or any(not _is_sha256(value) for value in hashes):
                return False
            integrity = cell.get("integrity")
            if not isinstance(integrity, list) or len(integrity) != config.T:
                return False
            for expected_window, item in enumerate(integrity, start=1):
                if not isinstance(item, dict) or item.get("window") != expected_window:
                    return False
                if item.get("pi_matches_train_storage_permutation") is not True or item.get("pi_is_full_permutation") is not True:
                    return False
                for split_name in ("train", "eval"):
                    block = item.get(split_name)
                    if not isinstance(block, dict) or any(block.get(field) is not True for field in required_block_flags):
                        return False
                    count = mode_count(regime, rank)
                    current, previous = mode_schedule(regime, rank, expected_window)
                    total = config.n_train if split_name == "train" else config.n_eval
                    expected_counts = [0] * count
                    if expected_window == 1:
                        expected_counts[current] = total
                    else:
                        expected_counts[current] = 3 * total // 4
                        expected_counts[previous] = total // 4
                    if block.get("expected_mode_counts") != expected_counts or block.get("observed_mode_counts") != expected_counts:
                        return False
                    signs = block.get("sign_counts")
                    if not isinstance(signs, list) or len(signs) != count:
                        return False
                    for mode, sign in enumerate(signs):
                        requested = expected_counts[mode]
                        if (
                            not isinstance(sign, dict)
                            or set(sign) != {"mode", "requested", "positive", "negative", "balanced"}
                            or sign.get("mode") != mode
                            or sign.get("requested") != requested
                            or sign.get("positive") != requested // 2
                            or sign.get("negative") != requested // 2
                            or sign.get("balanced") is not True
                        ):
                            return False
            for cast_name in ("train_cast_roundoff", "eval_cast_roundoff"):
                casts = cell.get(cast_name)
                if not isinstance(casts, list) or len(casts) != config.T:
                    return False
                for cast in casts:
                    if (
                        not isinstance(cast, dict)
                        or set(cast) != {"base_matrix_cast_loss", "target_nonadditivity_loss"}
                        or any(not _is_finite_number(cast.get(field)) or cast.get(field) < 0.0 for field in cast)
                    ):
                        return False
    return True


def _derive_g0_gate(
    bundle: RunBundle,
    per_window: CSVTable | None,
    collector: IssueCollector,
    resource_table: CSVTable | None = None,
    seeds: Sequence[int] = PRIMARY_SEEDS,
) -> None:
    if per_window is None:
        return
    expected_seeds = tuple(int(seed) for seed in seeds)
    try:
        split_passed = _split_manifest_semantics(bundle, expected_seeds)
        finite_fields = (
            "L_prefix",
            "L_current",
            "L_rank",
            "L_inf",
            "D_t",
            "D_current",
            "capacity_ratio",
            "online_ratio",
            "current_acquisition_ratio",
            "gap",
            "bounded_history_reduction",
            "selection_gain",
        )
        finite_passed = True
        risk_failures: list[dict[str, Any]] = []
        rank_passed = True
        learner_boundary_passed = True
        total_metadata_events = 0
        factor_groups: dict[tuple[int, str, int], set[str]] = {}
        regime_order = {name: index for index, name in enumerate(REGIMES)}
        arm_order = {name: index for index, name in enumerate(TRAINABLE_ARMS)}
        ordered_keys = sorted(
            per_window.rows,
            key=lambda key: (int(key[0]), regime_order.get(key[1], len(REGIMES)), int(key[2]), int(key[3]), arm_order.get(key[4], len(TRAINABLE_ARMS))),
        )
        for key in ordered_keys:
            row = per_window.rows[key]
            location = f"{bundle.label}.per_window.csv[{key!r}]"
            values = {field: _table_float(row, field, location) for field in finite_fields}
            finite_passed = finite_passed and all(math.isfinite(value) for value in values.values())
            numerical_rank = _strict_int(row.get("numerical_rank", ""), f"{location}.numerical_rank")
            requested_rank = _strict_int(row.get("requested_rank", ""), f"{location}.requested_rank")
            rank_passed = rank_passed and requested_rank == key[2] and numerical_rank <= key[2]
            leakage = _strict_int(row.get("metadata_leakage_events", ""), f"{location}.metadata_leakage_events")
            total_metadata_events += leakage
            learner_boundary_passed = learner_boundary_passed and bool(
                leakage == 0
                and _table_bool(row, "model_input_contract_passed", location)
                and _table_bool(row, "replay_sampler_contract_passed", location)
            )
            factor_groups.setdefault((key[0], key[1], key[2]), set()).add(row.get("factor_sha256", ""))
            if values["gap"] < -1e-10:
                risk_failures.append(
                    {
                        "seed": key[0],
                        "regime": key[1],
                        "R": key[2],
                        "window": key[3],
                        "arm": key[4],
                        "gap": values["gap"],
                    }
                )
        traces = bundle.documents.get("traces.json")
        boundary = traces.get("runtime_learner_boundary") if isinstance(traces, dict) else None
        learner_boundary_passed = learner_boundary_passed and bool(
            isinstance(boundary, dict)
            and boundary.get("passed") is True
            and boundary.get("gradient_model_input_names") == ["x", "y"]
            and boundary.get("gradient_model_input_shape") == [128, 32]
            and boundary.get("gradient_model_input_dtype") == "torch.float32"
            and boundary.get("latent_metadata_tensor_events") == total_metadata_events
            and boundary.get("replay_sampler_argument_names") == ["draw_rng", "size"]
            and boundary.get("permitted_scheduler_fields") == ["window", "pi_t", "insertion_boundary"]
            and boundary.get("permitted_post_run_evaluator_fields") == ["window", "mode_ids", "schedule", "split_indices"]
        )
        factor_shared = bool(factor_groups) and all(len(values) == 1 and all(_is_sha256(value) for value in values) for values in factor_groups.values())
        expected_factors = _expected_factor_hashes(expected_seeds)
        traces = bundle.documents.get("traces.json")
        trace_factors = traces.get("factor_initializations") if isinstance(traces, dict) else None
        resource_by_cell: dict[tuple[int, str, int], set[str]] = {}
        if resource_table is not None:
            for resource_key, resource_row in resource_table.rows.items():
                resource_by_cell.setdefault(resource_key[:3], set()).add(resource_row.get("factor_sha256", ""))
        factor_binding_passed = True
        for cell_key, values in factor_groups.items():
            seed, regime, rank = cell_key
            expected_factor = expected_factors.get(cell_key)
            trace_factor = None
            if isinstance(trace_factors, dict):
                seed_entry = trace_factors.get(str(seed))
                if isinstance(seed_entry, dict):
                    cell_entry = seed_entry.get(f"{regime}:{rank}")
                    if isinstance(cell_entry, dict):
                        trace_factor = cell_entry.get("sha256")
            factor_binding_passed = factor_binding_passed and bool(
                expected_factor
                and trace_factor == expected_factor
                and values == {expected_factor}
                and (resource_table is None or resource_by_cell.get(cell_key) == {expected_factor})
            )
        risk_passed = len(risk_failures) == 0
        expected = {
            "protocol_hash": PROTOCOL_SHA256,
            "passed": bool(
                split_passed
                and finite_passed
                and risk_passed
                and rank_passed
                and factor_shared
                and factor_binding_passed
                and learner_boundary_passed
            ),
            "source_hashes": bundle.manifest.get("source_hashes"),
            "split_integrity_passed": split_passed,
            "finite_metrics_passed": finite_passed,
            "risk_consistency_passed": risk_passed,
            "risk_consistency_failure_count": len(risk_failures),
            "risk_consistency_failures": risk_failures,
            "rank_passed": rank_passed,
            "factor_shared_across_arms": factor_shared,
            "factor_initializations_bound": factor_binding_passed,
            "runtime_learner_boundary_passed": learner_boundary_passed,
        }
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        collector.add("artifact_integrity", "g0_rederivation_failed", f"{bundle.label}.per_window.csv", str(exc))
        return
    gates = bundle.documents.get("gates.json")
    if isinstance(gates, dict):
        _compare_derived_value(gates.get("G0_numpy_identity"), expected, f"{bundle.label}.gates.G0_numpy_identity", collector, TRAINED_ATOL)


def _derive_g4_gate(
    bundle: RunBundle,
    per_window: CSVTable | None,
    resource_table: CSVTable | None,
    collector: IssueCollector,
    seeds: Sequence[int] = PRIMARY_SEEDS,
) -> None:
    if per_window is None or resource_table is None:
        return
    config = BridgeConfig()
    expected_seeds = tuple(int(seed) for seed in seeds)
    try:
        resource_native_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
        resource_rows: list[dict[str, Any]] = []
        for key, row in resource_table.rows.items():
            native = _native_resource_row(row, f"{bundle.label}.resource_ledger.csv[{key!r}]")
            resource_native_by_key[key] = native
            resource_rows.append(native)
        traces_document = bundle.documents.get("traces.json")
        transitions = traces_document.get("transitions") if isinstance(traces_document, dict) else None
        if not isinstance(transitions, list):
            transitions = []
        expected_transition_rows = len(expected_seeds) * len(REGIMES) * len(RANKS) * len(REPLAY_ARMS) * (config.T - 1)
        expected_replay = (
            _expected_replay_transitions(expected_seeds)
            if len(transitions) == expected_transition_rows
            else {}
        )
        observed_replay: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        for row in transitions:
            if not isinstance(row, Mapping):
                continue
            key = tuple(row.get(field) for field in TRANSITION_KEY)
            if key not in observed_replay:
                observed_replay[key] = row
        transition_passed = bool(
            len(transitions) == expected_transition_rows
            and set(observed_replay) == set(expected_replay)
            and all(_replay_transition_matches(observed_replay[key], expected_replay[key]) for key in expected_replay)
        )
        occupancy_passed = all(
            (
                _strict_int(row.get("occupied", ""), "occupied") == 0
                and _strict_int(row.get("unique", ""), "unique") == 0
            )
            if key[4] == TRAINABLE_ARMS[0] or key[3] == 1
            else (
                _strict_int(row.get("occupied", ""), "occupied") == config.replay_capacity
                and _strict_int(row.get("unique", ""), "unique") == config.replay_capacity
            )
            for key, row in per_window.rows.items()
        )
        expected_optimizer_contract = {
            "class": "AdamW",
            "parameter_order": ["B", "A"],
            "lr": config.lr,
            "betas": list(config.betas),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
            "amsgrad": False,
            "foreach": False,
            "fused": False,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
            "scheduler": None,
        }
        expected_optimizer_hash = artifacts.hash_json(expected_optimizer_contract)
        record_bytes = 2 * config.d_in * 8 + 2 * config.d_in * 4 + 8 + 1 + 8 + 5 * 8 + 8
        expected_cell_dataset_bytes = config.T * (
            (config.n_train + config.n_eval) * record_bytes + config.n_train * 8
        )
        expected_seed_dataset_bytes = len(REGIMES) * len(RANKS) * expected_cell_dataset_bytes
        optimizer_contract_passed = True
        compute_passed = True
        replay_resource_passed = True
        runtime = bundle.manifest.get("runtime") if isinstance(bundle.manifest.get("runtime"), dict) else {}
        warmup = runtime.get("measurement_backend_warmup") if isinstance(runtime, dict) else None
        warmup_baseline = (
            (warmup.get("allocated_bytes"), warmup.get("reserved_bytes"))
            if isinstance(warmup, dict)
            and set(warmup)
            == {
                "kind",
                "forward_count",
                "backward_count",
                "optimizer_step_count",
                "allocated_bytes",
                "reserved_bytes",
            }
            and warmup.get("kind") == "deterministic_max_rank_adamw_step_outside_measured_arms"
            and warmup.get("forward_count") == 1
            and warmup.get("backward_count") == 1
            and warmup.get("optimizer_step_count") == 1
            and type(warmup.get("allocated_bytes")) is int
            and type(warmup.get("reserved_bytes")) is int
            and warmup.get("allocated_bytes") >= 0
            and warmup.get("reserved_bytes") >= warmup.get("allocated_bytes")
            else None
        )
        accounting_complete_passed = bool(resource_rows and warmup_baseline is not None)
        for row in resource_rows:
            rank = row["requested_rank"]
            parameter_count = rank * (config.d_in + config.d_out)
            parameter_bytes = parameter_count * 4
            expected_parameter_breakdown = [
                {
                    "name": "B",
                    "shape": [config.d_out, rank],
                    "dtype": "torch.float32",
                    "numel": config.d_out * rank,
                    "bytes": config.d_out * rank * 4,
                },
                {
                    "name": "A",
                    "shape": [rank, config.d_in],
                    "dtype": "torch.float32",
                    "numel": config.d_in * rank,
                    "bytes": config.d_in * rank * 4,
                },
            ]
            expected_state_breakdown = [
                {
                    "parameter": parameter_name,
                    "state": state_name,
                    "shape": ([] if state_name == "step" else list(shape)),
                    "dtype": "torch.float32",
                    "bytes": (4 if state_name == "step" else shape[0] * shape[1] * 4),
                }
                for parameter_name, shape in (("B", (config.d_out, rank)), ("A", (rank, config.d_in)))
                for state_name in ("exp_avg", "exp_avg_sq", "step")
            ]
            optimizer_contract_passed = optimizer_contract_passed and bool(
                row["optimizer_contract"] == expected_optimizer_contract
                and row["optimizer_contract_sha256"].lower() == expected_optimizer_hash.lower()
                and row["parameter_breakdown"] == expected_parameter_breakdown
                and row["optimizer_state_bytes"] == 2 * parameter_bytes + 8
                and row["optimizer_state_breakdown"] == expected_state_breakdown
            )
            expected_updates = config.T * config.updates_per_window
            compute_passed = compute_passed and bool(
                row["requested_rank"] == row["active_rank"] == row["deployed_rank"]
                and row["numerical_rank_max"] <= rank
                and row["factor_shapes"] == {"B": [config.d_out, rank], "A": [rank, config.d_in]}
                and row["updates"] == expected_updates
                and row["forward"] == row["backward"] == expected_updates
                and row["current_examples"] + row["replay_examples"] == expected_updates * config.batch_size
                and row["trainable_parameter_count"] == parameter_count
                and row["trainable_parameter_bytes"] == parameter_bytes
                and row["deployed_parameter_count"] == parameter_count
                and row["deployed_parameter_bytes"] == parameter_bytes
                and row["batch_size"] == config.batch_size
                and row["optimizer_steps"] == [expected_updates, expected_updates]
                and row["curvature_sketch_bytes"] == 0
                and row["checkpoint_persistent_bytes"] == 0
            )
            replay_arm = row["arm"] in REPLAY_ARMS
            replay_resource_passed = replay_resource_passed and bool(
                (
                    row["replay_bytes"] == 0
                    and row["allocated_capacity"] == 0
                    and row["replay_fraction"] == 0.0
                    and row["replay_examples"] == 0
                    and row["current_examples"] == config.T * config.updates_per_window * config.batch_size
                    and row["replay_sample_draws"] == 0
                    and row["priority_draws"] == 0
                    and row["persistent_controller_sidecar_records"] == 0
                    and row["persistent_controller_sidecar_bytes"] == 0
                    and row["controller_transient_records_peak"] == 0
                )
                if not replay_arm
                else (
                    row["replay_bytes"] == config.fixed_replay_bytes
                    and row["allocated_capacity"] == config.replay_capacity
                    and row["replay_fraction"] == config.replay_batch / config.batch_size
                    and row["replay_examples"] == (config.T - 1) * config.updates_per_window * config.replay_batch
                    and row["current_examples"] == config.T * config.updates_per_window * config.batch_size - row["replay_examples"]
                    and row["replay_x_y_slot_bytes"] == 90112
                    and row["replay_priority_bytes"] == 2816
                    and row["replay_occupancy_bytes"] == 352
                    and row["pointer_counter_bytes"] == 208
                    and row["replay_sample_draws"] == (config.T - 1) * config.updates_per_window
                    and row["priority_draws"] == (config.T - 1) * config.n_train
                    and row["persistent_controller_sidecar_records"] == 0
                    and row["persistent_controller_sidecar_bytes"] == 0
                    and row["controller_transient_records_peak"] >= config.n_train
                    and row["controller_transient_python_bytes_peak"] > 0
                    and row["controller_transient_numpy_bytes_peak"] > 0
                )
            )
            expected_batch_staging = 2 * config.batch_size * config.d_in * 4
            if replay_arm:
                expected_batch_staging += 4 * config.replay_batch * config.d_in * 4
            expected_index_staging = config.batch_size * 8
            if replay_arm:
                expected_index_staging += 2 * config.replay_capacity * 8 + 2 * config.replay_batch * 8
            accounting_complete_passed = accounting_complete_passed and bool(
                    warmup_baseline is not None
                and (row["gpu_baseline_allocated_bytes"], row["gpu_baseline_reserved_bytes"]) == warmup_baseline
                and bool(row["driver_version"])
                and row["common_cpu_cell_dataset_bytes"] == expected_cell_dataset_bytes
                and row["common_cpu_seed_dataset_resident_bytes"] == expected_seed_dataset_bytes
                and row["common_cpu_determinism_rebuild_increment_bytes"] == row["common_cpu_seed_dataset_resident_bytes"]
                and row["common_cpu_determinism_rebuild_peak_bytes"] == 2 * row["common_cpu_seed_dataset_resident_bytes"]
                and row["evaluator_checkpoint_staging_bytes"] == parameter_count * 8 + config.d_in * config.d_out * 8
                and row["common_W0_cpu_bytes"] == config.d_in * config.d_out * 4
                and row["common_W0_gpu_bytes"] == config.d_in * config.d_out * 4
                and row["common_cpu_batch_staging_bytes"] == 2 * config.batch_size * config.d_in * 4
                and row["common_cpu_index_staging_bytes_max"] == config.batch_size * 8
                and row["common_gpu_batch_staging_bytes"] == 2 * config.batch_size * config.d_in * 4
                and row["arm_cpu_batch_staging_peak_bytes"] == expected_batch_staging
                and row["arm_cpu_index_staging_peak_bytes"] == expected_index_staging
                and row["peak_allocated_increment_bytes"] == row["peak_allocated_gpu_bytes"] - row["gpu_baseline_allocated_bytes"]
                and row["peak_reserved_increment_bytes"] == row["peak_reserved_gpu_bytes"] - row["gpu_baseline_reserved_bytes"]
                and row["peak_allocated_increment_bytes"] >= 0
                and row["peak_reserved_increment_bytes"] >= 0
                and row["inference_baseline_allocated_bytes"] > 0
                and row["inference_baseline_reserved_bytes"] >= row["inference_baseline_allocated_bytes"]
                and row["inference_peak_allocated_gpu_bytes"] >= row["inference_baseline_allocated_bytes"]
                and row["inference_peak_reserved_gpu_bytes"] >= row["inference_baseline_reserved_bytes"]
                and row["inference_peak_allocated_increment_bytes"] == row["inference_peak_allocated_gpu_bytes"] - row["inference_baseline_allocated_bytes"]
                and row["inference_peak_reserved_increment_bytes"] == row["inference_peak_reserved_gpu_bytes"] - row["inference_baseline_reserved_bytes"]
                and row["inference_latency_mean_batch_seconds"] > 0.0
                and abs(row["inference_latency_mean_example_seconds"] * config.batch_size - row["inference_latency_mean_batch_seconds"]) <= 1e-12
                and row["inference_warmup_forwards"] == 5
                and row["inference_measured_forwards"] == 20
                and row["metadata_leakage_events"] == 0
                and row["model_input_contract_passed"]
                and row["replay_sampler_contract_passed"]
            )

        # Reconcile the compact resource row against the twelve per-window
        # records and the eleven replay transitions.  These values are
        # independently serialized, so checking only each table in isolation
        # would allow a contradictory ledger to pass.
        transition_by_cell: dict[tuple[int, str, int, str], list[Mapping[str, Any]]] = {}
        for transition in transitions:
            if isinstance(transition, dict):
                transition_by_cell.setdefault(
                    (transition.get("seed"), transition.get("regime"), transition.get("R"), transition.get("arm")),
                    [],
                ).append(transition)
        aggregate_passed = True
        for resource_key, _raw_resource_row in resource_table.rows.items():
            resource_row = resource_native_by_key[resource_key]
            cell_key = resource_key[:3]
            arm = resource_key[3]
            windows = [
                row
                for key, row in per_window.rows.items()
                if key[:3] == cell_key and key[4] == arm
            ]
            windows.sort(key=lambda row: int(row["window"]))
            aggregate_passed = aggregate_passed and len(windows) == config.T
            if len(windows) != config.T:
                continue
            replay_arm = arm in REPLAY_ARMS
            expected_replay_per_window = 0 if not replay_arm else config.replay_batch * config.updates_per_window
            expected_current_per_window = config.batch_size * config.updates_per_window - expected_replay_per_window
            numerical_ranks = [int(row["numerical_rank"]) for row in windows]
            allocated_peaks = [int(row["peak_allocated_gpu_bytes_so_far"]) for row in windows]
            reserved_peaks = [int(row["peak_reserved_gpu_bytes_so_far"]) for row in windows]
            aggregate_passed = aggregate_passed and bool(
                numerical_ranks
                and all(left <= right for left, right in zip(allocated_peaks, allocated_peaks[1:]))
                and all(left <= right for left, right in zip(reserved_peaks, reserved_peaks[1:]))
                and resource_row["numerical_rank_max"] == max(numerical_ranks)
                and resource_row["peak_allocated_gpu_bytes"] == max(allocated_peaks)
                and resource_row["peak_reserved_gpu_bytes"] == max(reserved_peaks)
            )
            cell_transitions = transition_by_cell.get((*cell_key, arm), [])
            cell_transitions.sort(key=lambda item: int(item.get("completed_window", -1)))
            if replay_arm:
                aggregate_passed = aggregate_passed and [int(item.get("completed_window", -1)) for item in cell_transitions] == list(range(1, config.T))
            for index, window_row in enumerate(windows, start=1):
                prefix_transitions = [item for item in cell_transitions if int(item.get("completed_window", -1)) < index]
                expected_priority_prefix = sum(int(item["priority_draws"]) for item in prefix_transitions)
                expected_comparison_prefix = sum(int(item["selection_comparisons"]) for item in prefix_transitions)
                aggregate_passed = aggregate_passed and bool(
                    int(window_row["window"]) == index
                    and int(window_row["updates"]) == config.updates_per_window
                    and int(window_row["forward"]) == config.updates_per_window
                    and int(window_row["backward"]) == config.updates_per_window
                    and int(window_row["current_examples"]) == (
                        config.batch_size * config.updates_per_window
                        if index == 1
                        else expected_current_per_window
                    )
                    and int(window_row["replay_examples"]) == (0 if index == 1 else expected_replay_per_window)
                    and int(window_row["replay_sample_draws"]) == (
                        0 if not replay_arm or index == 1 else config.updates_per_window
                    )
                    and int(window_row["occupied"]) == (0 if not replay_arm or index == 1 else config.replay_capacity)
                    and int(window_row["unique"]) == (0 if not replay_arm or index == 1 else config.replay_capacity)
                    and int(window_row["duplicates"]) == 0
                    and int(window_row["priority_draws"]) == (expected_priority_prefix if replay_arm else 0)
                    and int(window_row["selection_comparisons"]) == (expected_comparison_prefix if replay_arm else 0)
                )
            int_sums = {
                "updates": sum(int(row["updates"]) for row in windows),
                "forward": sum(int(row["forward"]) for row in windows),
                "backward": sum(int(row["backward"]) for row in windows),
                "current_examples": sum(int(row["current_examples"]) for row in windows),
                "replay_examples": sum(int(row["replay_examples"]) for row in windows),
                "replay_sample_draws": sum(int(row["replay_sample_draws"]) for row in windows),
                "replay_sampling_duplicate_count": sum(int(row["duplicates"]) for row in windows),
            }
            aggregate_passed = aggregate_passed and all(resource_row[field] == value for field, value in int_sums.items())
            aggregate_passed = aggregate_passed and bool(
                resource_row["occupied_records_max"] == max(int(row["occupied"]) for row in windows)
                and resource_row["unique_records_max"] == max(int(row["unique"]) for row in windows)
            )
            if replay_arm:
                aggregate_passed = aggregate_passed and len(cell_transitions) == config.T - 1
                if len(cell_transitions) == config.T - 1:
                    aggregate_passed = aggregate_passed and bool(
                        resource_row["priority_draws"] == sum(int(item["priority_draws"]) for item in cell_transitions)
                        and resource_row["selection_comparisons"] == sum(int(item["selection_comparisons"]) for item in cell_transitions)
                        and resource_row["priority_tie_count"] == sum(int(item["priority_tie_count"]) for item in cell_transitions)
                        and resource_row["replacement_count"] == sum(int(item["replacement_count"]) for item in cell_transitions)
                    )
            else:
                aggregate_passed = aggregate_passed and bool(
                    resource_row["priority_draws"] == 0
                    and resource_row["selection_comparisons"] == 0
                    and resource_row["priority_tie_count"] == 0
                    and resource_row["replacement_count"] == 0
                )
            for resource_field, window_field in (
                ("replay_assembly_seconds", "replay_assembly_seconds"),
                ("device_staging_seconds", "device_staging_seconds"),
                ("training_wall_seconds", "train_wall_seconds"),
            ):
                observed = float(resource_row[resource_field])
                expected_sum = sum(float(row[window_field]) for row in windows)
                aggregate_passed = aggregate_passed and math.isfinite(observed) and math.isfinite(expected_sum) and abs(observed - expected_sum) <= 1e-9
            controller_observed = float(resource_row["controller_cpu_seconds"])
            controller_expected = sum(float(row["controller_cpu_seconds"]) for row in windows) + sum(
                float(item["controller_cpu_seconds"]) for item in cell_transitions
            )
            aggregate_passed = aggregate_passed and bool(
                math.isfinite(controller_observed)
                and math.isfinite(controller_expected)
                and abs(controller_observed - controller_expected) <= 1e-9
            )
            aggregate_passed = aggregate_passed and float(resource_row["wall_seconds"]) >= float(resource_row["training_wall_seconds"])

        by_cell: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
        for row in resource_rows:
            by_cell.setdefault((row["seed"], row["regime"], row["R"]), []).append(row)
        expected_cells = set(itertools.product(expected_seeds, REGIMES, RANKS))
        matched_arm_passed = aggregate_passed and set(by_cell) == expected_cells
        for rows in by_cell.values():
            if len(rows) != len(TRAINABLE_ARMS) or {row["arm"] for row in rows} != set(TRAINABLE_ARMS):
                matched_arm_passed = False
                continue
            matched_arm_passed = matched_arm_passed and len({row["trainable_parameter_bytes"] for row in rows}) == 1
            matched_arm_passed = matched_arm_passed and len({row["optimizer_state_bytes"] for row in rows}) == 1
            matched_arm_passed = matched_arm_passed and len({row["optimizer_contract_sha256"] for row in rows}) == 1
            matched_arm_passed = matched_arm_passed and len({row["factor_sha256"] for row in rows}) == 1
            replay_rows = [row for row in rows if row["arm"] in REPLAY_ARMS]
            for field in ("replay_bytes", "replay_examples", "replay_sample_draws", "priority_draws"):
                matched_arm_passed = matched_arm_passed and len({row[field] for row in replay_rows}) == 1
        expected = {
            "transition_passed": transition_passed,
            "occupancy_passed": occupancy_passed,
            "compute_passed": compute_passed,
            "optimizer_contract_passed": optimizer_contract_passed,
            "replay_resource_passed": replay_resource_passed,
            "matched_arm_passed": matched_arm_passed,
            "accounting_complete_passed": accounting_complete_passed,
        }
        expected["single_run_passed"] = all(expected.values())
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        collector.add("artifact_integrity", "g4_rederivation_failed", f"{bundle.label}.resource_ledger.csv", str(exc))
        return
    gates = bundle.documents.get("gates.json")
    g4 = gates.get("G4_resource_and_reproducibility") if isinstance(gates, dict) else None
    if not isinstance(g4, dict):
        return
    for field, derived in expected.items():
        if g4.get(field) is not derived:
            _derived_mismatch(
                collector,
                f"{bundle.label}.gates.G4_resource_and_reproducibility.{field}",
                f"expected={derived!r},actual={g4.get(field)!r}",
            )


def _validate_gate_leaf_types(
    value: Any,
    label: str,
    collector: IssueCollector,
    path: tuple[str, ...],
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_gate_leaf_types(item, label, collector, path + (str(key),))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_gate_leaf_types(item, label, collector, path + (str(index),))
        return
    if not path:
        return
    field = path[-1]
    parent = path[-2] if len(path) > 1 else ""
    location = f"{label}.gates." + ".".join(path)
    count_field = (
        field in {
            "count",
            "total",
            "threshold",
            "transition_rows",
            "expected_transition_rows",
            "residual_failures",
            "support_failures",
            "weighted_modal_svd_mismatches",
        }
        or field.endswith("_count")
    )
    boolean_field = (
        field == "passed"
        or field.startswith("passed_")
        or field.endswith("_passed")
        or field.endswith("_complete")
        or field.endswith("_required_for_citation")
        or field == "factor_shared_across_arms"
        or parent == "acquisition_ok"
    )
    status_field = field == "status" or field.endswith("_status")
    numeric_field = (
        field in _GATE_NUMERIC_KEYS
        or any(token in field for token in ("loss", "ratio", "delta", "error", "fraction", "gain"))
        or field in {"mean", "median", "std", "minimum", "maximum"}
        or "bound" in field
    )
    if count_field and not _is_nonnegative_int(value):
        collector.add("run_schema", "invalid_gate_count", location, repr(value))
    elif boolean_field:
        if type(value) is not bool:
            collector.add("run_schema", "invalid_gate_boolean", location, repr(value))
    elif status_field:
        allowed_statuses = set(_ALLOWED_GATE_STATUSES)
        if "bootstrap" in path and "statistics" in path:
            allowed_statuses.update({"valid", "invalid"})
        if not isinstance(value, str) or value not in allowed_statuses:
            collector.add("run_schema", "invalid_gate_status", location, repr(value))
    elif numeric_field and not _is_finite_number(value):
        collector.add("run_schema", "invalid_gate_numeric", location, repr(value))


def _validate_gate_schema(
    bundle: RunBundle,
    collector: IssueCollector,
    seeds: Sequence[int] = PRIMARY_SEEDS,
) -> None:
    gates = bundle.documents.get("gates.json")
    if not isinstance(gates, dict):
        return
    expected_seeds = tuple(int(seed) for seed in seeds)
    scientific_run = bundle.expected_role in {"primary", "confirmation"}
    expected_gate_names = {
        "G0_numpy_identity",
        "G1_oracle_capacity",
        "G2_phenomenon",
        "G3_bounded_history",
        "G4_resource_and_reproducibility",
    }
    if set(gates) != expected_gate_names:
        collector.add(
            "run_schema",
            "gate_set_mismatch",
            f"{bundle.label}.gates",
            f"expected={sorted(expected_gate_names)!r},actual={sorted(gates)!r}",
        )
    _validate_gate_leaf_types(gates, bundle.label, collector, ())

    g0 = gates.get("G0_numpy_identity")
    if isinstance(g0, dict):
        expected_g0_fields = {
            "protocol_hash",
            "passed",
            "source_hashes",
            "split_integrity_passed",
            "finite_metrics_passed",
            "risk_consistency_passed",
            "risk_consistency_failure_count",
            "risk_consistency_failures",
            "rank_passed",
            "factor_shared_across_arms",
            "factor_initializations_bound",
            "runtime_learner_boundary_passed",
        }
        if set(g0) != expected_g0_fields:
            collector.add(
                "run_schema",
                "g0_field_set_mismatch",
                f"{bundle.label}.gates.G0_numpy_identity",
                f"expected={sorted(expected_g0_fields)!r},actual={sorted(g0)!r}",
            )
        if g0.get("protocol_hash") != bundle.manifest.get("protocol_sha256"):
            collector.add("run_schema", "gate_protocol_mismatch", f"{bundle.label}.gates.G0_numpy_identity.protocol_hash", "does not match manifest")
        if g0.get("source_hashes") != bundle.manifest.get("source_hashes"):
            collector.add("run_schema", "gate_source_hashes_mismatch", f"{bundle.label}.gates.G0_numpy_identity.source_hashes", "does not match manifest")
        g0_subgates = (
            "split_integrity_passed",
            "finite_metrics_passed",
            "risk_consistency_passed",
            "rank_passed",
            "factor_shared_across_arms",
            "factor_initializations_bound",
            "runtime_learner_boundary_passed",
        )
        if all(type(g0.get(field)) is bool for field in g0_subgates):
            expected_g0_pass = bool(
                g0.get("protocol_hash") == PROTOCOL_SHA256
                and g0.get("source_hashes") == bundle.manifest.get("source_hashes")
                and all(g0[field] for field in g0_subgates)
            )
            if g0.get("passed") is not expected_g0_pass:
                collector.add("run_schema", "inconsistent_g0_passed", f"{bundle.label}.gates.G0_numpy_identity.passed", f"expected={expected_g0_pass!r},actual={g0.get('passed')!r}")
        failures = g0.get("risk_consistency_failures")
        if isinstance(failures, list) and g0.get("risk_consistency_failure_count") != len(failures):
            collector.add("run_schema", "inconsistent_g0_risk_count", f"{bundle.label}.gates.G0_numpy_identity.risk_consistency_failure_count", repr(g0.get("risk_consistency_failure_count")))

    g1 = gates.get("G1_oracle_capacity")
    expected_g1_fields = _G1_FIELDS
    if not isinstance(g1, dict):
        collector.add("run_schema", "missing_g1_object", f"{bundle.label}.gates.G1_oracle_capacity", "expected object")
    elif set(g1) != expected_g1_fields:
        collector.add(
            "run_schema",
            "g1_field_set_mismatch",
            f"{bundle.label}.gates.G1_oracle_capacity",
            f"expected={sorted(expected_g1_fields)!r},actual={sorted(g1)!r}",
        )
    else:
        _compare_derived_value(g1, _expected_g1_gate(), f"{bundle.label}.gates.G1_oracle_capacity", collector, ORACLE_ATOL)

    for gate_name in ("G2_phenomenon", "G3_bounded_history"):
        gate = gates.get(gate_name)
        expected_status = "evaluated" if scientific_run else "not_evaluated"
        if isinstance(gate, dict) and gate.get("status") != expected_status:
            collector.add("run_schema", "gate_status_mismatch", f"{bundle.label}.gates.{gate_name}.status", repr(gate.get("status")))

    g2 = gates.get("G2_phenomenon")
    if isinstance(g2, dict):
        acquisition = g2.get("acquisition")
        if not isinstance(acquisition, dict) or set(acquisition) != set(TRAINABLE_ARMS):
            collector.add("run_schema", "invalid_acquisition_gate", f"{bundle.label}.gates.G2_phenomenon.acquisition", "expected exact trainable-arm map")
        else:
            for arm in TRAINABLE_ARMS:
                entry = acquisition.get(arm)
                location = f"{bundle.label}.gates.G2_phenomenon.acquisition.{arm}"
                if not isinstance(entry, dict):
                    collector.add("run_schema", "invalid_acquisition_entry", location, type(entry).__name__)
                    continue
                if set(entry) != {"count", "total", "threshold", "status"}:
                    collector.add("run_schema", "acquisition_field_set_mismatch", location, repr(sorted(entry)))
                count = entry.get("count")
                total = entry.get("total")
                threshold = entry.get("threshold")
                if not _is_nonnegative_int(count) or not _is_nonnegative_int(total) or not _is_nonnegative_int(threshold):
                    collector.add("run_schema", "invalid_acquisition_counts", location, repr(entry))
                elif total != len(expected_seeds) * BridgeConfig().T or threshold != 228 or count > total:
                    collector.add("run_schema", "noncanonical_acquisition_counts", location, repr(entry))
                if entry.get("status") != "primary_only":
                    collector.add("run_schema", "invalid_acquisition_status", f"{location}.status", repr(entry.get("status")))
        if g2.get("status") == "evaluated":
            acquisition_ok = g2.get("acquisition_ok")
            if not isinstance(acquisition_ok, dict) or set(acquisition_ok) != set(TRAINABLE_ARMS):
                collector.add("run_schema", "invalid_acquisition_ok", f"{bundle.label}.gates.G2_phenomenon.acquisition_ok", repr(acquisition_ok))
            else:
                for arm in TRAINABLE_ARMS:
                    entry = acquisition.get(arm) if isinstance(acquisition, dict) else None
                    expected_ok = isinstance(entry, dict) and _is_nonnegative_int(entry.get("count")) and _is_nonnegative_int(entry.get("threshold")) and entry["count"] >= entry["threshold"]
                    if type(acquisition_ok.get(arm)) is not bool or acquisition_ok[arm] != expected_ok:
                        collector.add(
                            "run_schema",
                            "inconsistent_acquisition_ok",
                            f"{bundle.label}.gates.G2_phenomenon.acquisition_ok.{arm}",
                            f"expected={expected_ok!r},actual={acquisition_ok.get(arm)!r}",
                        )
            if g2.get("passed") is True:
                if g2.get("paired_final_rows_complete") is not True:
                    collector.add("run_schema", "inconsistent_g2_passed", f"{bundle.label}.gates.G2_phenomenon.paired_final_rows_complete", "passed requires complete paired final rows")
                if isinstance(acquisition, dict) and any(
                    not (
                        isinstance(acquisition.get(arm), dict)
                        and _is_nonnegative_int(acquisition[arm].get("count"))
                        and _is_nonnegative_int(acquisition[arm].get("threshold"))
                        and acquisition[arm]["count"] >= acquisition[arm]["threshold"]
                    )
                    for arm in TRAINABLE_ARMS
                ):
                    collector.add("run_schema", "inconsistent_g2_passed", f"{bundle.label}.gates.G2_phenomenon.acquisition", "passed requires every acquisition count to meet threshold")
                lower = g2.get("sequential_online_ratio_lower_bound")
                threshold = g2.get("sequential_online_ratio_threshold")
                if not _is_finite_number(lower) or not _is_finite_number(threshold) or lower < threshold:
                    collector.add("run_schema", "inconsistent_g2_passed", f"{bundle.label}.gates.G2_phenomenon.sequential_online_ratio_lower_bound", "passed requires a finite lower bound at or above its threshold")

    g3 = gates.get("G3_bounded_history")
    if isinstance(g3, dict) and scientific_run:
        for field in (
            "valid_positive_gap_count",
            "indeterminate_count",
            "nonpositive_count",
            "minimum_valid_count",
            "risk_consistency_failures",
        ):
            if not _is_nonnegative_int(g3.get(field)):
                collector.add("run_schema", "invalid_g3_count", f"{bundle.label}.gates.G3_bounded_history.{field}", repr(g3.get(field)))
        if g3.get("minimum_valid_count") != 16:
            collector.add("run_schema", "noncanonical_g3_minimum", f"{bundle.label}.gates.G3_bounded_history.minimum_valid_count", repr(g3.get("minimum_valid_count")))
        if g3.get("status") == "evaluated" and g3.get("passed") is True:
            if g3.get("paired_final_rows_complete") is not True:
                collector.add("run_schema", "inconsistent_g3_passed", f"{bundle.label}.gates.G3_bounded_history.paired_final_rows_complete", "passed requires complete paired final rows")
            valid_count = g3.get("valid_positive_gap_count")
            minimum_count = g3.get("minimum_valid_count")
            if not _is_nonnegative_int(valid_count) or not _is_nonnegative_int(minimum_count) or valid_count < minimum_count:
                collector.add("run_schema", "inconsistent_g3_passed", f"{bundle.label}.gates.G3_bounded_history.valid_positive_gap_count", "passed requires minimum valid positive gaps")
            if g3.get("risk_consistency_failures") != 0:
                collector.add("run_schema", "inconsistent_g3_passed", f"{bundle.label}.gates.G3_bounded_history.risk_consistency_failures", "passed requires zero risk consistency failures")
            bounded_lower = None
            bootstrap = g3.get("bootstrap")
            if isinstance(bootstrap, dict):
                stats = bootstrap.get("statistics")
                if isinstance(stats, dict):
                    bounded = stats.get("bounded_history_reduction_T")
                    if isinstance(bounded, dict):
                        bounded_lower = bounded.get("lower_bound_05")
            median = g3.get("gap_fraction_median")
            if not _is_finite_number(bounded_lower) or bounded_lower <= 0.0:
                collector.add("run_schema", "inconsistent_g3_passed", f"{bundle.label}.gates.G3_bounded_history.bootstrap", "passed requires a positive bounded-history lower bound")
            if not _is_finite_number(median) or median < 0.50:
                collector.add("run_schema", "inconsistent_g3_passed", f"{bundle.label}.gates.G3_bounded_history.gap_fraction_median", "passed requires median gap fraction at least 0.50")

    g4 = gates.get("G4_resource_and_reproducibility")
    expected_g4_fields = {
        "passed",
        "single_run_passed",
        "transition_passed",
        "occupancy_passed",
        "compute_passed",
        "optimizer_contract_passed",
        "expected_optimizer_contract_sha256",
        "replay_resource_passed",
        "matched_arm_passed",
        "accounting_complete_passed",
        "transition_rows",
        "expected_transition_rows",
        "confirmation_required_for_citation",
        "confirmation_status",
        "post_result_four_provider_audit_status",
        "status",
        "peak_gpu_measurement_scope",
    }
    if not isinstance(g4, dict):
        collector.add("run_schema", "missing_g4_object", f"{bundle.label}.gates.G4_resource_and_reproducibility", "expected object")
        return
    if set(g4) != expected_g4_fields:
        collector.add("run_schema", "g4_field_set_mismatch", f"{bundle.label}.gates.G4_resource_and_reproducibility", f"expected={sorted(expected_g4_fields)!r},actual={sorted(g4)!r}")
    if not _is_sha256(g4.get("expected_optimizer_contract_sha256")):
        collector.add(
            "run_schema",
            "invalid_g4_optimizer_contract_hash",
            f"{bundle.label}.gates.G4_resource_and_reproducibility.expected_optimizer_contract_sha256",
            repr(g4.get("expected_optimizer_contract_sha256")),
        )
    expected_transition_rows = len(expected_seeds) * len(REGIMES) * len(RANKS) * len(REPLAY_ARMS) * (BridgeConfig().T - 1)
    for field in ("transition_rows", "expected_transition_rows"):
        if not _is_nonnegative_int(g4.get(field)) or g4.get(field) != expected_transition_rows:
            collector.add("run_schema", "invalid_g4_transition_count", f"{bundle.label}.gates.G4_resource_and_reproducibility.{field}", f"expected={expected_transition_rows},actual={g4.get(field)!r}")
    expected_status = (
        "pending_reconciliation"
        if scientific_run and g4.get("single_run_passed") is True
        else "passed_non_citable_single_run"
        if not scientific_run and g4.get("single_run_passed") is True
        else "failed"
    )
    for field, expected in (
        ("passed", False if scientific_run else g4.get("single_run_passed")),
        ("confirmation_required_for_citation", True),
        ("confirmation_status", "pending"),
        ("post_result_four_provider_audit_status", "pending"),
        ("status", expected_status),
        ("peak_gpu_measurement_scope", _EXPECTED_PEAK_SCOPE),
    ):
        if g4.get(field) != expected:
            collector.add("run_schema", "noncanonical_g4_field", f"{bundle.label}.gates.G4_resource_and_reproducibility.{field}", f"expected={expected!r},actual={g4.get(field)!r}")
    subgate_fields = (
        "transition_passed",
        "occupancy_passed",
        "compute_passed",
        "optimizer_contract_passed",
        "replay_resource_passed",
        "matched_arm_passed",
        "accounting_complete_passed",
    )
    if all(type(g4.get(field)) is bool for field in subgate_fields):
        expected_single_run = all(g4[field] for field in subgate_fields)
        if g4.get("single_run_passed") is not expected_single_run:
            collector.add(
                "run_schema",
                "inconsistent_g4_single_run_passed",
                f"{bundle.label}.gates.G4_resource_and_reproducibility.single_run_passed",
                f"expected={expected_single_run!r},actual={g4.get('single_run_passed')!r}",
            )


def _compare_gate_json(
    left: Any,
    right: Any,
    collector: IssueCollector,
    path: tuple[str, ...] = (),
    maxima: dict[str, float] | None = None,
) -> None:
    if maxima is None:
        maxima = {"oracle": 0.0, "trained": 0.0}
    location = "gates" + ("." + ".".join(path) if path else "")
    if type(left) is not type(right):
        collector.add("gate_reproducibility", "gate_type_mismatch", location, f"{type(left).__name__} != {type(right).__name__}")
        return
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            collector.add(
                "gate_reproducibility",
                "gate_key_set_mismatch",
                location,
                f"primary_only={sorted(left_keys-right_keys)!r},confirmation_only={sorted(right_keys-left_keys)!r}",
            )
        for key in sorted(left_keys & right_keys):
            _compare_gate_json(left[key], right[key], collector, path + (str(key),), maxima)
        return
    if isinstance(left, list):
        if len(left) != len(right):
            collector.add("gate_reproducibility", "gate_list_length_mismatch", location, f"{len(left)} != {len(right)}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _compare_gate_json(left_item, right_item, collector, path + (str(index),), maxima)
        return
    if isinstance(left, bool) or left is None or isinstance(left, (str, int)):
        if left != right:
            collector.add("gate_reproducibility", "gate_exact_value_mismatch", location, f"{left!r} != {right!r}")
        return
    if isinstance(left, float):
        if not math.isfinite(left) or not math.isfinite(right):
            collector.add("gate_reproducibility", "nonfinite_gate_scalar", location, "non-finite gate value")
            return
        if _json_path_is_timing(path):
            if left < 0.0 or right < 0.0:
                collector.add("gate_reproducibility", "negative_gate_timing", location, f"{left!r},{right!r}")
            return
        oracle = _json_path_is_oracle(path)
        tolerance = ORACLE_ATOL if oracle else TRAINED_ATOL
        delta = abs(left - right)
        if not math.isfinite(delta):
            collector.add(
                "gate_reproducibility",
                "gate_scalar_delta_nonfinite",
                location,
                f"primary={left!r},confirmation={right!r}",
            )
            return
        maxima["oracle" if oracle else "trained"] = max(maxima["oracle" if oracle else "trained"], delta)
        if delta > tolerance:
            collector.add(
                "gate_reproducibility",
                "gate_scalar_tolerance_exceeded",
                location,
                f"delta={delta:.17g},tolerance={tolerance:.17g}",
            )
        return
    collector.add("gate_reproducibility", "unsupported_gate_value", location, type(left).__name__)


def _index_transitions(
    transitions: Any,
    label: str,
    collector: IssueCollector,
    seeds: Sequence[int] = PRIMARY_SEEDS,
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    expected_seeds = tuple(int(seed) for seed in seeds)
    expected = {
        (seed, regime, rank, arm, window)
        for seed, regime, rank, arm, window in itertools.product(
            expected_seeds, REGIMES, RANKS, REPLAY_ARMS, range(1, 12)
        )
    }
    canonical_config = BridgeConfig()
    indexed: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    if not isinstance(transitions, list):
        collector.add("trace_reproducibility", "transitions_not_list", f"{label}.traces.transitions", "expected list")
        return indexed
    expected_transition_rows = len(expected)
    if len(transitions) != expected_transition_rows:
        collector.add(
            "trace_reproducibility",
            "transition_row_count_mismatch",
            f"{label}.traces.transitions",
            f"expected={expected_transition_rows},actual={len(transitions)}",
        )
    for index, row in enumerate(transitions):
        location = f"{label}.traces.transitions[{index}]"
        if not isinstance(row, dict):
            collector.add("trace_reproducibility", "transition_not_object", location, type(row).__name__)
            continue
        field_set_valid = set(row) == set(TRANSITION_FIELDS)
        if not field_set_valid:
            collector.add(
                "trace_reproducibility",
                "transition_field_set_mismatch",
                location,
                f"missing={sorted(set(TRANSITION_FIELDS)-set(row))!r},extra={sorted(set(row)-set(TRANSITION_FIELDS))!r}",
            )
        types_valid = True
        for field in TRANSITION_INTEGER_FIELDS:
            value = row.get(field)
            if not _is_nonnegative_int(value):
                collector.add("trace_reproducibility", "invalid_transition_integer", f"{location}.{field}", repr(value))
                types_valid = False
        regime = row.get("regime")
        arm = row.get("arm")
        if not isinstance(regime, str) or regime not in REGIMES:
            collector.add("trace_reproducibility", "invalid_transition_regime", f"{location}.regime", repr(regime))
            types_valid = False
        if not isinstance(arm, str) or arm not in REPLAY_ARMS:
            collector.add("trace_reproducibility", "invalid_transition_arm", f"{location}.arm", repr(arm))
            types_valid = False
        elapsed = row.get("controller_cpu_seconds")
        if (
            not _is_finite_number(elapsed)
            or elapsed < 0.0
        ):
            collector.add("trace_reproducibility", "invalid_transition_timing", f"{location}.controller_cpu_seconds", repr(elapsed))
            types_valid = False
        priority_hash = row.get("selected_priority_order_sha256")
        if not _is_sha256(priority_hash):
            collector.add("trace_reproducibility", "invalid_transition_hash", f"{location}.selected_priority_order_sha256", repr(priority_hash))
            types_valid = False
        if not field_set_valid or not types_valid:
            continue
        seed = row["seed"]
        rank = row["R"]
        completed_window = row["completed_window"]
        if seed not in expected_seeds or rank not in RANKS or completed_window not in range(1, 12):
            collector.add("trace_reproducibility", "transition_key_out_of_domain", location, repr(tuple(row[field] for field in TRANSITION_KEY)))
            continue
        semantic_expectations = {
            "history_windows": completed_window,
            "capacity": canonical_config.replay_capacity,
            "eligible_count": completed_window * canonical_config.n_train,
            "occupied_count": canonical_config.replay_capacity,
            "unique_count": canonical_config.replay_capacity,
            "duplicate_count": 0,
            "priority_draws": canonical_config.n_train,
        }
        semantic_valid = True
        for field, expected_value in semantic_expectations.items():
            if row[field] != expected_value:
                collector.add(
                    "trace_reproducibility",
                    "noncanonical_transition_value",
                    f"{location}.{field}",
                    f"expected={expected_value},actual={row[field]!r}",
                )
                semantic_valid = False
        if row["selection_comparisons"] <= 0:
            collector.add("trace_reproducibility", "nonpositive_transition_comparisons", f"{location}.selection_comparisons", repr(row["selection_comparisons"]))
            semantic_valid = False
        if not semantic_valid:
            continue
        key = tuple(row[field] for field in TRANSITION_KEY)
        if key in indexed:
            collector.add("trace_reproducibility", "duplicate_transition_key", location, repr(key))
            continue
        indexed[key] = row
    missing = sorted(expected - set(indexed))
    extra = sorted(set(indexed) - expected)
    if missing:
        collector.add("trace_reproducibility", "missing_transition_keys", f"{label}.traces.transitions", f"count={len(missing)},first={missing[:3]!r}")
    if extra:
        collector.add("trace_reproducibility", "unexpected_transition_keys", f"{label}.traces.transitions", f"count={len(extra)},first={extra[:3]!r}")
    # Bind every deterministic replay field to an independent reconstruction
    # from the frozen seed stream.  Only controller wall-clock time is allowed
    # to vary between machines.
    if set(indexed) == expected:
        expected_replay = _expected_replay_transitions(expected_seeds)
        for key in sorted(set(indexed) & set(expected_replay)):
            if not _replay_transition_matches(indexed[key], expected_replay[key]):
                collector.add(
                    "trace_reproducibility",
                    "noncanonical_replay_transition",
                    f"{label}.traces.transitions[{key!r}]",
                    "deterministic replay fields differ from the frozen reconstruction",
                )
    return indexed


def _compare_transitions(primary: RunBundle, confirmation: RunBundle, collector: IssueCollector) -> None:
    left_traces = primary.documents.get("traces.json")
    right_traces = confirmation.documents.get("traces.json")
    if not isinstance(left_traces, dict) or not isinstance(right_traces, dict):
        return
    left = _index_transitions(left_traces.get("transitions"), "primary", collector)
    right = _index_transitions(right_traces.get("transitions"), "confirmation", collector)
    for key in sorted(set(left) & set(right)):
        left_row = left[key]
        right_row = right[key]
        if set(left_row) != set(right_row):
            collector.add("trace_reproducibility", "transition_field_set_mismatch", repr(key), "field sets differ")
        for field in sorted(set(left_row) & set(right_row)):
            location = f"traces.transitions[{key!r}].{field}"
            if _is_timing_or_peak(field):
                for label, value in (("primary", left_row[field]), ("confirmation", right_row[field])):
                    if not _is_finite_number(value) or value < 0.0:
                        collector.add("trace_reproducibility", "invalid_transition_timing", location, f"{label}={value!r}")
            elif left_row[field] != right_row[field]:
                collector.add("trace_reproducibility", "transition_exact_mismatch", location, f"{left_row[field]!r} != {right_row[field]!r}")


def _gate_decision_checks(bundle: RunBundle, collector: IssueCollector) -> None:
    gates = bundle.documents.get("gates.json")
    summary = bundle.documents.get("summary.json")
    if not isinstance(gates, dict) or not isinstance(summary, dict):
        return
    required = (
        ("G0_numpy_identity", "passed"),
        ("G1_oracle_capacity", "passed"),
        ("G2_phenomenon", "passed"),
        ("G3_bounded_history", "passed"),
        ("G4_resource_and_reproducibility", "single_run_passed"),
    )
    for gate_name, field in required:
        gate = gates.get(gate_name)
        if not isinstance(gate, dict) or not isinstance(gate.get(field), bool):
            collector.add("gate_outcomes", "missing_gate_boolean", f"{bundle.label}.{gate_name}.{field}", "required boolean unavailable")
        elif not gate[field]:
            collector.add(
                "gate_outcomes",
                "scientific_gate_failed",
                f"{bundle.label}.{gate_name}.{field}",
                "false",
                disposition="NO-GO",
            )
    if not isinstance(summary.get("execution_pass"), bool):
        collector.add("gate_outcomes", "missing_execution_pass", f"{bundle.label}.summary.execution_pass", "required boolean unavailable")
    elif not summary["execution_pass"]:
        collector.add(
            "gate_outcomes",
            "execution_gate_failed",
            f"{bundle.label}.summary.execution_pass",
            "false",
            disposition="NO-GO",
        )


def _check_payload(collector: IssueCollector, check: str) -> dict[str, Any]:
    issue_count = collector.count(check=check)
    return {
        "passed": issue_count == 0,
        "issue_count": issue_count,
    }


def validate_single_run(
    run_dir: str | Path,
    *,
    expected_role: str,
    expected_seeds: Sequence[int],
    require_scientific_gates: bool = False,
) -> dict[str, Any]:
    """Deeply validate one immutable GPU run without writing any artifacts."""

    seeds = tuple(int(seed) for seed in expected_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("expected_seeds must be a non-empty sequence of unique integers")
    collector = IssueCollector()
    bundle = _load_run(Path(run_dir).resolve(), "single", expected_role, collector)
    tables: dict[str, CSVTable | None] = {}
    if bundle is not None:
        _validate_run_schema(bundle, collector, seeds)
        _validate_gate_schema(bundle, collector, seeds)
        config = BridgeConfig()
        per_window_expected = {
            (seed, regime, rank, window, arm)
            for seed, regime, rank, window, arm in itertools.product(
                seeds, REGIMES, RANKS, range(1, config.T + 1), TRAINABLE_ARMS
            )
        }
        per_seed_expected = {
            (seed, regime, rank, arm)
            for seed, regime, rank, arm in itertools.product(
                seeds, REGIMES, RANKS, TRAINABLE_ARMS
            )
        }
        tables = {
            "per_window.csv": _read_csv_table(
                bundle,
                "per_window.csv",
                PER_WINDOW_KEY,
                PER_WINDOW_REQUIRED_FIELDS,
                per_window_expected,
                collector,
            ),
            "per_seed.csv": _read_csv_table(
                bundle,
                "per_seed.csv",
                PER_SEED_KEY,
                PER_SEED_REQUIRED_FIELDS,
                per_seed_expected,
                collector,
            ),
            "resource_ledger.csv": _read_csv_table(
                bundle,
                "resource_ledger.csv",
                RESOURCE_KEY,
                RESOURCE_REQUIRED_FIELDS,
                per_seed_expected,
                collector,
            ),
        }
        resource = tables["resource_ledger.csv"]
        per_window = tables["per_window.csv"]
        per_seed = tables["per_seed.csv"]
        _validate_resource_identity(bundle, resource, collector)
        _derive_g0_gate(bundle, per_window, collector, resource, seeds)
        _derive_scientific_gates(
            bundle,
            per_window,
            per_seed,
            collector,
            seeds,
            compare_scientific_gates=require_scientific_gates,
        )
        _validate_resource_semantics(bundle, resource, collector)
        _derive_g4_gate(bundle, per_window, resource, collector, seeds)
        traces = bundle.documents.get("traces.json")
        _index_transitions(
            traces.get("transitions") if isinstance(traces, dict) else None,
            bundle.label,
            collector,
            seeds,
        )
        if require_scientific_gates:
            _gate_decision_checks(bundle, collector)
    return {
        "valid": collector.count() == 0,
        "issue_count": collector.count(),
        "suppressed_issue_count": collector.suppressed,
        "issues": [issue.payload() for issue in collector.issues],
        "expected_role": expected_role,
        "expected_seeds": list(seeds),
        "rows": {
            name: (0 if table is None else len(table.rows))
            for name, table in sorted(tables.items())
        },
    }


def reconcile_runs(
    primary_dir: str | Path,
    confirmation_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate and reconcile two immutable Phase-2 run directories.

    A structurally valid agreement returns ``GO_PENDING_POST_RESULT_AUDIT``.
    A trusted, exactly reproduced failed gate returns ``NO-GO``.  Artifact,
    identity, schema, or numerical reproducibility problems return ``REVISE``.
    """

    primary_path = Path(primary_dir).resolve()
    confirmation_path = Path(confirmation_dir).resolve()
    output_path = Path(output_dir).resolve()
    if primary_path == confirmation_path:
        raise ValueError("primary_dir and confirmation_dir must be distinct")
    if output_path in {primary_path, confirmation_path} or primary_path in output_path.parents or confirmation_path in output_path.parents:
        raise ValueError("output_dir must not be either input directory or a descendant of one")
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise FileExistsError(f"output directory is not empty: {output_path}")

    collector = IssueCollector()
    primary = _load_run(primary_path, "primary", "primary", collector)
    confirmation = _load_run(confirmation_path, "confirmation", "confirmation", collector)

    tables: dict[str, tuple[CSVTable | None, CSVTable | None]] = {}
    comparison_stats: dict[str, Any] = {
        "per_window": {},
        "per_seed": {},
        "resource_ledger": {},
        "gates": {"oracle_max_abs_delta": 0.0, "trained_max_abs_delta": 0.0},
    }
    if primary is not None:
        _validate_run_schema(primary, collector)
        _validate_gate_schema(primary, collector)
    if confirmation is not None:
        _validate_run_schema(confirmation, collector)
        _validate_gate_schema(confirmation, collector)
    if primary is not None and confirmation is not None:
        _compare_identity(primary, confirmation, collector)

        per_window_expected = {
            (seed, regime, rank, window, arm)
            for seed, regime, rank, window, arm in itertools.product(
                PRIMARY_SEEDS, REGIMES, RANKS, range(1, 13), TRAINABLE_ARMS
            )
        }
        per_seed_expected = {
            (seed, regime, rank, arm)
            for seed, regime, rank, arm in itertools.product(
                PRIMARY_SEEDS, REGIMES, RANKS, TRAINABLE_ARMS
            )
        }
        table_specs = (
            ("per_window.csv", PER_WINDOW_KEY, PER_WINDOW_REQUIRED_FIELDS, per_window_expected),
            ("per_seed.csv", PER_SEED_KEY, PER_SEED_REQUIRED_FIELDS, per_seed_expected),
            ("resource_ledger.csv", RESOURCE_KEY, RESOURCE_REQUIRED_FIELDS, per_seed_expected),
        )
        for filename, key_fields, required_fields, expected_keys in table_specs:
            tables[filename] = (
                _read_csv_table(primary, filename, key_fields, required_fields, expected_keys, collector),
                _read_csv_table(confirmation, filename, key_fields, required_fields, expected_keys, collector),
            )
        _validate_resource_identity(primary, tables["resource_ledger.csv"][0], collector)
        _validate_resource_identity(confirmation, tables["resource_ledger.csv"][1], collector)
        _derive_g0_gate(primary, tables["per_window.csv"][0], collector, tables["resource_ledger.csv"][0])
        _derive_g0_gate(confirmation, tables["per_window.csv"][1], collector, tables["resource_ledger.csv"][1])
        _derive_scientific_gates(primary, tables["per_window.csv"][0], tables["per_seed.csv"][0], collector)
        _derive_scientific_gates(confirmation, tables["per_window.csv"][1], tables["per_seed.csv"][1], collector)
        _validate_resource_semantics(primary, tables["resource_ledger.csv"][0], collector)
        _validate_resource_semantics(confirmation, tables["resource_ledger.csv"][1], collector)
        _derive_g4_gate(primary, tables["per_window.csv"][0], tables["resource_ledger.csv"][0], collector)
        _derive_g4_gate(confirmation, tables["per_window.csv"][1], tables["resource_ledger.csv"][1], collector)

        per_window_tolerances = {
            **{field: ORACLE_ATOL for field in PER_WINDOW_ORACLE_FIELDS},
            **{field: TRAINED_ATOL for field in PER_WINDOW_TRAINED_FIELDS},
        }
        per_seed_tolerances = {field: TRAINED_ATOL for field in PER_SEED_TRAINED_FIELDS}
        comparison_stats["per_window"] = _compare_csv_tables(
            *tables["per_window.csv"], per_window_tolerances, collector
        )
        comparison_stats["per_seed"] = _compare_csv_tables(
            *tables["per_seed.csv"], per_seed_tolerances, collector
        )
        comparison_stats["resource_ledger"] = _compare_csv_tables(
            *tables["resource_ledger.csv"], {}, collector
        )

        left_gates = primary.documents.get("gates.json")
        right_gates = confirmation.documents.get("gates.json")
        if isinstance(left_gates, dict) and isinstance(right_gates, dict):
            gate_maxima = {"oracle": 0.0, "trained": 0.0}
            _compare_gate_json(left_gates, right_gates, collector, maxima=gate_maxima)
            comparison_stats["gates"] = {
                "oracle_max_abs_delta": gate_maxima["oracle"],
                "trained_max_abs_delta": gate_maxima["trained"],
            }
        _compare_transitions(primary, confirmation, collector)
        _gate_decision_checks(primary, collector)
        _gate_decision_checks(confirmation, collector)

    revise_issues = collector.count(disposition="REVISE")
    no_go_issues = collector.count(disposition="NO-GO")
    if revise_issues:
        decision = "REVISE"
    elif no_go_issues:
        decision = "NO-GO"
    else:
        decision = "GO_PENDING_POST_RESULT_AUDIT"

    check_names = (
        "artifact_integrity",
        "run_schema",
        "identity",
        "csv_schema",
        "csv_reproducibility",
        "gate_reproducibility",
        "trace_reproducibility",
        "gate_outcomes",
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "phase2_primary_confirmation_reconciliation",
        "decision": decision,
        "positive_claim_authorized": False,
        "post_result_four_provider_audit_required": True,
        "tolerances": {
            "oracle_absolute": ORACLE_ATOL,
            "trained_absolute": TRAINED_ATOL,
            "integer_boolean_status_count": "exact",
            "wall_time_latency_peak": "finite_nonnegative_only",
        },
        "inputs": {
            "primary": {
                "run_id": None if primary is None else primary.manifest.get("run_id"),
                "manifest_sha256": artifacts.sha256_file(primary_path / "manifest.json") if (primary_path / "manifest.json").is_file() else None,
                "required_role": "primary",
            },
            "confirmation": {
                "run_id": None if confirmation is None else confirmation.manifest.get("run_id"),
                "manifest_sha256": artifacts.sha256_file(confirmation_path / "manifest.json") if (confirmation_path / "manifest.json").is_file() else None,
                "required_role": "confirmation",
            },
        },
        "expected_ordered_seeds": list(PRIMARY_SEEDS),
        "checks": {name: _check_payload(collector, name) for name in check_names},
        "comparison": comparison_stats,
        "issue_count": collector.count(),
        "reported_issue_count": len(collector.issues),
        "suppressed_issue_count": collector.suppressed,
        "issues": [
            issue.payload()
            for issue in sorted(
                collector.issues,
                key=lambda item: (item.disposition, item.check, item.code, item.location, item.message),
            )
        ],
        "claim_boundary": "no citation before a successful post-result four-provider read-only audit",
    }

    output_path.mkdir(parents=True, exist_ok=True)
    artifacts.write_json(output_path / "reconciliation.json", payload)
    return payload


__all__ = [
    "ORACLE_ATOL",
    "TRAINED_ATOL",
    "StrictJSONError",
    "load_json_strict",
    "reconcile_runs",
]
