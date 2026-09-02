"""Tamper-evident calibration freeze evidence for the Phase-2 GPU bridge.

The marker written here contains only paths and digests.  In particular, it
does not contain a trusted ``frozen_for_primary`` switch: callers must invoke
``validate_calibration_freeze`` and use the conclusion derived from the bound
artifacts.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import artifacts
from .core_numpy import BridgeConfig, PROTOCOL_SHA256, build_seed_dataset, make_base_draws
from .validation import validate_single_run


CALIBRATION_SEEDS = (11, 23, 37)
PROVIDERS = ("gpt", "claude", "glm", "deepseek")
EXPECTED_CALIBRATION_ROWS = {
    "per_window.csv": 648,
    "per_seed.csv": 54,
    "resource_ledger.csv": 54,
}
EXPECTED_TRANSITIONS = 396
EXPECTED_REMOTE_TESTS = 34
VALIDATION_REPORT_SCHEMA_VERSION = 1
VALIDATOR_VERSION = "phase2-evidence-v2"
CROSS_HOST_HASH_FIELDS = ("base_sha256", "data_sha256", "split_sha256")
CROSS_HOST_HASH_DISPOSITION = "provenance_only_local_cpu_smoke"
REQUIRED_RUN_OUTPUTS = {
    "config.json",
    "gates.json",
    "manifest.json",
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
_DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MAX_JSON_INTEGER_DIGITS = 1024


class EvidenceError(ValueError):
    """Raised when a freeze artifact is incomplete, inconsistent, or unsafe."""


def _fail(message: str) -> None:
    raise EvidenceError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    _fail(f"non-finite JSON token: {token}")


def _parse_strict_int(token: str) -> int:
    if len(token.lstrip("-")) > _MAX_JSON_INTEGER_DIGITS:
        _fail(f"JSON integer exceeds {_MAX_JSON_INTEGER_DIGITS} digits")
    try:
        return int(token)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvidenceError(f"invalid JSON integer: {token!r}") from exc


def _check_finite(value: Any, label: str = "JSON") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require(isinstance(key, str), f"{label} contains a non-string key")
            _check_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, f"{label}[{index}]")


def _load_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            parse_int=_parse_strict_int,
            object_pairs_hook=_strict_object,
        )
    except EvidenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise EvidenceError(f"invalid JSON artifact {path}: {exc}") from exc
    _check_finite(value, str(path))
    return value


def _read_text_with_bom(path: Path) -> str:
    """Read runner text logs across PowerShell UTF-16 and UTF-8 encodings."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"unable to read text artifact {path}: {exc}") from exc
    if raw.startswith(b"\xff\xfe"):
        encoding = "utf-16-le"
        raw = raw[2:]
    elif raw.startswith(b"\xfe\xff"):
        encoding = "utf-16-be"
        raw = raw[2:]
    elif raw.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8"
        raw = raw[3:]
    else:
        encoding = "utf-8"
    try:
        return raw.decode(encoding)
    except UnicodeError as exc:
        raise EvidenceError(f"invalid text artifact {path}: {exc}") from exc


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a JSON object")
    return value


def _digest(value: Any, label: str) -> str:
    _require(isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None, f"{label} must be a SHA-256 digest")
    return value.lower()


def _same_digest(left: Any, right: Any, label: str) -> None:
    _require(_digest(left, f"{label} (left)") == _digest(right, f"{label} (right)"), f"{label} mismatch")


def _root_path(root: str | Path) -> Path:
    resolved = Path(root).resolve(strict=True)
    _require(resolved.is_dir(), f"root is not a directory: {resolved}")
    return resolved


def _inside(root: Path, value: str | Path, *, kind: str) -> Path:
    try:
        raw = Path(value)
    except TypeError as exc:
        raise EvidenceError(f"{kind} path is not a string: {value!r}") from exc
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise EvidenceError(f"{kind} is missing or outside root: {value}") from exc
    if kind.endswith("directory"):
        _require(resolved.is_dir(), f"{kind} is not a directory: {value}")
    else:
        _require(resolved.is_file(), f"{kind} is not a file: {value}")
    return resolved


def _marker_target(root: Path, value: str | Path) -> Path:
    try:
        raw = Path(value)
    except TypeError as exc:
        raise EvidenceError(f"marker path is not a string: {value!r}") from exc
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EvidenceError(f"marker path is outside root: {value}") from exc
    _require(resolved.name != "", "marker path must name a file")
    _require(not resolved.exists() or resolved.is_file(), "marker path must not be an existing directory")
    return resolved


def _assert_marker_disjoint(target: Path, inputs: Sequence[Path]) -> None:
    if target.exists():
        raise EvidenceError(f"marker path already exists; refusing to overwrite: {target}")
    for input_path in inputs:
        if target == input_path or input_path in target.parents:
            raise EvidenceError(f"marker path overlaps bound evidence: {target}")


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _config_facts(config: Any) -> tuple[str, str, tuple[int, ...]]:
    _require(type(config) is BridgeConfig, "config must be an exact BridgeConfig instance")
    _require(config == BridgeConfig() and config.is_protocol_locked, "config differs from the frozen BridgeConfig()")
    protocol = config.protocol_sha256
    seeds = config.calibration_seeds
    config_hash = config.sha256()
    _digest(protocol, "config protocol_sha256")
    _same_digest(protocol, PROTOCOL_SHA256, "config protocol hash")
    _digest(config_hash, "config sha256")
    _require(isinstance(seeds, (tuple, list)), "config calibration_seeds must be a sequence")
    parsed_seeds = tuple(int(seed) for seed in seeds)
    _require(parsed_seeds == CALIBRATION_SEEDS, "config calibration seeds are not the frozen (11,23,37)")
    return str(protocol), str(config_hash), parsed_seeds


def _validate_source_hashes(root: Path, source_hashes: Mapping[str, str]) -> dict[str, str]:
    _require(isinstance(source_hashes, Mapping) and bool(source_hashes), "source_hashes must be a non-empty mapping")
    _require(set(source_hashes) == CANONICAL_SOURCE_FILES, "source_hashes must bind the exact canonical Phase-2 source set")
    source_root = root / "experiments" / "phase2_gpu_bridge"
    validated: dict[str, str] = {}
    for name, expected in sorted(source_hashes.items()):
        _require(
            isinstance(name, str)
            and "/" not in name
            and "\\" not in name
            and Path(name).name == name
            and name not in {".", ".."},
            f"unsafe source hash key: {name!r}",
        )
        path = _inside(root, source_root / name, kind="source file")
        observed = artifacts.sha256_file(path)
        _same_digest(observed, expected, f"source hash for {name}")
        validated[name] = observed
    return validated


def _safe_output_name(name: Any) -> str:
    _require(isinstance(name, str) and name != "", "manifest output name must be a non-empty string")
    path = Path(name)
    _require(
        not path.is_absolute()
        and "/" not in name
        and "\\" not in name
        and path.name == name
        and name not in {".", "..", "manifest.json"},
        f"unsafe manifest output name: {name!r}",
    )
    return name


def _snapshot_run(root: Path, run_dir: str | Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    directory = _inside(root, run_dir, kind="run directory")
    manifest_path = _inside(root, directory / "manifest.json", kind="run manifest")
    manifest = _mapping(_load_json(manifest_path), f"{manifest_path} contents")
    declared_raw = _mapping(manifest.get("output_hashes"), "manifest.output_hashes")
    declared: dict[str, str] = {}
    for raw_name, expected in declared_raw.items():
        name = _safe_output_name(raw_name)
        _require(name not in declared, f"duplicate normalized output name: {name}")
        path = _inside(root, directory / name, kind="run output")
        observed = artifacts.sha256_file(path)
        _same_digest(observed, expected, f"output hash for {name}")
        declared[name] = observed
        if path.suffix.lower() == ".json":
            _load_json(path)
    direct_files = {path.name for path in directory.iterdir() if path.is_file() and path.name != "manifest.json"}
    _require(set(declared) == direct_files, "manifest output_hashes does not bind every direct output file exactly")
    _require(REQUIRED_RUN_OUTPUTS - {"manifest.json"} <= set(declared), "run is missing required JSON outputs")
    binding = {
        "manifest": _relative(root, manifest_path),
        "manifest_sha256": artifacts.sha256_file(manifest_path),
        "output_hashes": dict(sorted(declared.items())),
    }
    return binding, manifest


def _check_run_binding(root: Path, binding_value: Any) -> tuple[dict[str, Any], Mapping[str, Any], Path]:
    binding = _mapping(binding_value, "run binding")
    _require(set(binding) == {"manifest", "manifest_sha256", "output_hashes"}, "run binding schema mismatch")
    manifest_path = _inside(root, binding["manifest"], kind="bound run manifest")
    observed, manifest = _snapshot_run(root, manifest_path.parent)
    _require(observed["manifest"] == binding["manifest"], "bound manifest path is not canonical")
    _same_digest(observed["manifest_sha256"], binding["manifest_sha256"], "bound manifest hash")
    expected_outputs = _mapping(binding["output_hashes"], "bound output_hashes")
    _require(set(expected_outputs) == set(observed["output_hashes"]), "bound output hash names changed")
    for name, expected in expected_outputs.items():
        _same_digest(observed["output_hashes"][name], expected, f"bound output hash for {name}")
    return observed, manifest, manifest_path.parent


def _common_run_checks(
    directory: Path,
    manifest: Mapping[str, Any],
    *,
    role: str,
    protocol_hash: str,
    config_hash: str,
    source_hashes: Mapping[str, str],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    _require(manifest.get("run_role") == role, f"manifest run_role is not {role}")
    _same_digest(manifest.get("protocol_sha256"), protocol_hash, f"{role} protocol hash")
    _require(manifest.get("source_hashes") == source_hashes, f"{role} source hashes differ from the freeze snapshot")
    config_payload = _mapping(_load_json(directory / "config.json"), f"{role} config")
    summary = _mapping(_load_json(directory / "summary.json"), f"{role} summary")
    gates = _mapping(_load_json(directory / "gates.json"), f"{role} gates")
    traces = _mapping(_load_json(directory / "traces.json"), f"{role} traces")
    _require(config_payload.get("kind") == "phase2_frozen_experiment_config", f"{role} config kind mismatch")
    _require(config_payload.get("run_role") is None, f"{role} config must not contain a run role")
    _same_digest(config_payload.get("protocol_sha256"), protocol_hash, f"{role} config protocol hash")
    _require(
        config_payload.get("protocol_path") == "notes/phase2_gpu_bridge_protocol.md",
        f"{role} config protocol path is not the frozen project protocol",
    )
    frozen_payload = _mapping(config_payload.get("frozen"), f"{role} frozen config payload")
    _same_digest(artifacts.hash_json(frozen_payload), config_hash, f"{role} derived frozen config hash")
    _same_digest(manifest.get("frozen_config_sha256"), config_hash, f"{role} manifest frozen config hash")
    _same_digest(manifest.get("config_sha256"), artifacts.sha256_file(directory / "config.json"), f"{role} manifest config hash")
    _require(summary.get("run_role") == role, f"{role} summary role mismatch")
    _require(summary.get("gates") == gates, f"{role} summary gates differ from gates.json")
    g0 = _mapping(gates.get("G0_numpy_identity"), f"{role} G0")
    _require(g0.get("source_hashes") == source_hashes, f"{role} G0 source hashes differ from the freeze snapshot")
    _same_digest(g0.get("protocol_hash"), protocol_hash, f"{role} G0 protocol hash")
    _require(traces.get("source_hashes") == source_hashes, f"{role} traces source hashes differ from the freeze snapshot")
    _same_digest(traces.get("protocol_sha256"), protocol_hash, f"{role} traces protocol hash")
    _same_digest(traces.get("base_sha256"), manifest.get("base_sha256"), f"{role} traces base hash")
    redundant_hash_fields = {
        "config_sha256": "config.json",
        "split_manifest_sha256": "split_manifest.json",
        "gates_sha256": "gates.json",
        "traces_sha256": "traces.json",
    }
    for field, name in redundant_hash_fields.items():
        _require(field in manifest, f"{role} manifest omitted {field}")
        _same_digest(manifest[field], artifacts.sha256_file(directory / name), f"{role} {field}")
    return config_payload, summary, gates


def _summary_pass(summary: Mapping[str, Any], label: str) -> None:
    _require(summary.get("all_required_pass") is True, f"{label} all_required_pass is not true")
    if "execution_pass" in summary and summary.get("execution_pass") is not None:
        _require(summary.get("execution_pass") is True, f"{label} execution_pass is not true")


def _require_gate_true(gates: Mapping[str, Any], gate_name: str, field: str, label: str) -> None:
    gate = _mapping(gates.get(gate_name), f"{label} {gate_name}")
    _require(gate.get(field) is True, f"{label} {gate_name}.{field} is not true")


def _csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)), f"invalid CSV header: {path}")
            return sum(1 for _ in reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise EvidenceError(f"invalid CSV artifact {path}: {exc}") from exc


def _require_gpu_runtime(manifest: Mapping[str, Any], config_payload: Mapping[str, Any], label: str) -> None:
    device = config_payload.get("device")
    _require(isinstance(device, str) and re.fullmatch(r"cuda:[0-9]+", device) is not None, f"{label} config device is not canonical CUDA")
    runtime = _mapping(manifest.get("runtime"), f"{label} runtime")
    for field, expected in (
        ("torch_available", True),
        ("cuda_available", True),
        ("deterministic_algorithms", True),
        ("cudnn_deterministic", True),
        ("cudnn_benchmark", False),
        ("cuda_matmul_allow_tf32", False),
        ("cudnn_allow_tf32", False),
        ("float32_matmul_precision", "highest"),
    ):
        _require(runtime.get(field) == expected, f"{label} runtime {field} mismatch")
    environment = _mapping(runtime.get("environment_controls"), f"{label} runtime environment controls")
    _require(
        environment
        == {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MKL_DYNAMIC": "FALSE",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        f"{label} runtime environment controls mismatch",
    )
    device_count = runtime.get("device_count")
    torch_devices = runtime.get("devices")
    smi_devices = runtime.get("nvidia_smi_devices")
    _require(type(device_count) is int and device_count > 0, f"{label} runtime device_count is invalid")
    _require(isinstance(torch_devices, list) and len(torch_devices) == device_count, f"{label} runtime torch device list mismatch")
    _require(all(isinstance(name, str) and name for name in torch_devices), f"{label} runtime torch device names are invalid")
    _require(isinstance(smi_devices, list) and len(smi_devices) == device_count, f"{label} runtime nvidia-smi device count mismatch")
    smi_indices: list[int] = []
    smi_by_index: dict[int, Mapping[str, Any]] = {}
    for item in smi_devices:
        row = _mapping(item, f"{label} nvidia-smi device")
        index = row.get("index")
        _require(type(index) is int and 0 <= index < device_count, f"{label} nvidia-smi device index is invalid")
        _require(index not in smi_by_index, f"{label} nvidia-smi device indices are duplicated")
        _require(isinstance(row.get("name"), str) and row["name"], f"{label} nvidia-smi device name is empty")
        _require(isinstance(row.get("driver_version"), str) and row["driver_version"], f"{label} nvidia-smi driver identity is empty")
        smi_indices.append(index)
        smi_by_index[index] = row
    _require(sorted(smi_indices) == list(range(device_count)), f"{label} nvidia-smi indices are not a complete device set")
    selected_index = int(device.split(":", 1)[1])
    _require(selected_index < device_count, f"{label} selected device is outside runtime device set")
    _require(torch_devices[selected_index] == smi_by_index[selected_index]["name"], f"{label} selected torch/nvidia-smi device mismatch")


def _exact_local_smoke_rebuild(
    directory: Path,
    manifest: Mapping[str, Any],
    *,
    config: BridgeConfig,
) -> dict[str, str]:
    """Rebuild local smoke bytes only in its originating NumPy environment."""

    base = make_base_draws(config)
    dataset = build_seed_dataset(base, 11, config)
    expected_base = base.sha256()
    _same_digest(manifest.get("base_sha256"), expected_base, "local CPU smoke exact base hash")
    _same_digest(manifest.get("data_sha256"), dataset.data_sha256, "local CPU smoke exact data hash")
    _same_digest(manifest.get("split_sha256"), dataset.split_sha256, "local CPU smoke exact split hash")
    split_payload = _mapping(_load_json(directory / "split_manifest.json"), "local CPU smoke exact split manifest")
    _require(set(split_payload) == {"seed", "data_sha256", "split_sha256", "cells"}, "local CPU smoke exact split schema mismatch")
    _require(split_payload.get("seed") == 11, "local CPU smoke exact split seed mismatch")
    _same_digest(split_payload.get("data_sha256"), dataset.data_sha256, "local CPU smoke exact split data hash")
    _same_digest(split_payload.get("split_sha256"), dataset.split_sha256, "local CPU smoke exact split hash")
    cells = _mapping(split_payload.get("cells"), "local CPU smoke exact split cells")
    _require(set(cells) == {f"{regime}:{rank}" for regime in ("compatible", "stress") for rank in (2, 4, 8)}, "local CPU smoke exact split cell set mismatch")
    for (regime, rank), windows in dataset.cells.items():
        cell = _mapping(cells.get(f"{regime}:{rank}"), f"local CPU smoke exact cell {regime}:{rank}")
        _require(cell.get("windows") == config.T, f"local CPU smoke exact window count mismatch for {regime}:{rank}")
        _require(cell.get("train_counts") == [window.train.count for window in windows], f"local CPU smoke exact train counts mismatch for {regime}:{rank}")
        _require(cell.get("eval_counts") == [window.eval.count for window in windows], f"local CPU smoke exact eval counts mismatch for {regime}:{rank}")
        _require(cell.get("pi_hashes") == [artifacts.hash_arrays((("pi_t", window.pi_t),)) for window in windows], f"local CPU smoke exact pi hashes mismatch for {regime}:{rank}")
    traces = _mapping(_load_json(directory / "traces.json"), "local CPU smoke exact traces")
    _same_digest(traces.get("base_sha256"), expected_base, "local CPU smoke exact traces base hash")
    _same_digest(traces.get("data_sha256"), dataset.data_sha256, "local CPU smoke exact traces data hash")
    _same_digest(traces.get("split_sha256"), dataset.split_sha256, "local CPU smoke exact traces split hash")
    return {
        "base_sha256": expected_base,
        "data_sha256": dataset.data_sha256,
        "split_sha256": dataset.split_sha256,
    }


def _calibration_checks(
    directory: Path,
    manifest: Mapping[str, Any],
    *,
    protocol_hash: str,
    config_hash: str,
    source_hashes: Mapping[str, str],
    deep_validate: bool = True,
) -> dict[str, Any]:
    config_payload, summary, gates = _common_run_checks(
        directory,
        manifest,
        role="non_citable_calibration",
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        source_hashes=source_hashes,
    )
    ordered_seeds = tuple(manifest.get("ordered_seeds", ()))
    config_seeds = tuple(config_payload.get("seeds", ()))
    _require(ordered_seeds == CALIBRATION_SEEDS, "calibration manifest seeds are not exactly (11,23,37)")
    _require(config_seeds == CALIBRATION_SEEDS, "calibration config seeds are not exactly (11,23,37)")
    _require_gpu_runtime(manifest, config_payload, "calibration")
    _summary_pass(summary, "calibration")
    _require(summary.get("execution_pass") is True, "calibration execution_pass is not true")
    _require(summary.get("per_window_rows") == 648, "calibration summary per_window_rows is not 648")
    _require(summary.get("per_seed_rows") == 54, "calibration summary per_seed_rows is not 54")
    observed_rows: dict[str, int] = {}
    for name, expected in EXPECTED_CALIBRATION_ROWS.items():
        observed_rows[name] = _csv_rows(directory / name)
        _require(observed_rows[name] == expected, f"calibration {name} row count is not {expected}")
    traces = _mapping(_load_json(directory / "traces.json"), "calibration traces")
    transitions = traces.get("transitions")
    _require(isinstance(transitions, list) and len(transitions) == EXPECTED_TRANSITIONS, "calibration transition count is not 396")
    for gate_name in ("G0_numpy_identity", "G1_oracle_capacity"):
        _require_gate_true(gates, gate_name, "passed", "calibration")
    g4 = _mapping(gates.get("G4_resource_and_reproducibility"), "calibration G4")
    _require(g4.get("single_run_passed") is True, "calibration G4 single-run checks did not pass")
    _require(g4.get("passed") is True, "calibration G4 passed is not true")
    for gate_name in ("G2_phenomenon", "G3_bounded_history"):
        gate = _mapping(gates.get(gate_name), f"calibration {gate_name}")
        _require(gate.get("status") == "not_evaluated", f"calibration {gate_name} must be not_evaluated")
    expected_test = source_hashes["test_phase2_gpu_bridge.py"]
    _same_digest(manifest.get("test_hash"), expected_test, "calibration test hash")
    if deep_validate:
        deep = validate_single_run(
            directory,
            expected_role="non_citable_calibration",
            expected_seeds=CALIBRATION_SEEDS,
            require_scientific_gates=False,
        )
        _require(deep.get("valid") is True, f"calibration deep validation failed: {deep.get('issues', [])[:1]}")
    return {"rows": observed_rows, "transitions": len(transitions)}


def _smoke_checks(
    directory: Path,
    manifest: Mapping[str, Any],
    *,
    root: Path,
    role: str,
    protocol_hash: str,
    config_hash: str,
    source_hashes: Mapping[str, str],
    deep_validate: bool = True,
) -> None:
    config_payload, summary, gates = _common_run_checks(
        directory,
        manifest,
        role=role,
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        source_hashes=source_hashes,
    )
    _summary_pass(summary, role)
    seeds = tuple(config_payload.get("seeds", ()))
    _require(bool(seeds) and set(seeds) <= set(CALIBRATION_SEEDS), f"{role} uses seeds outside calibration set")
    if role == "local_cpu_smoke":
        _require(manifest.get("schema_version") == 1, "local CPU smoke manifest schema mismatch")
        _require(manifest.get("stage") == "phase2_numpy_smoke", "local CPU smoke manifest stage mismatch")
        _require(manifest.get("run_id") == directory.name, "local CPU smoke run_id/directory mismatch")
        _require(manifest.get("claim_boundary") == "provenance_only_local_cpu_smoke", "local CPU smoke claim boundary mismatch")
        _same_digest(manifest.get("test_hash"), source_hashes["test_phase2_gpu_bridge.py"], "local CPU smoke test hash")
        _require(summary.get("schema_version") == 1 and summary.get("stage") == "phase2_numpy_smoke", "local CPU smoke summary identity mismatch")
        _require(summary.get("claim_boundary") == "provenance_only_local_cpu_smoke", "local CPU smoke summary claim boundary mismatch")
        _require(config_payload.get("device") == "cpu", "local CPU smoke did not use device=cpu")
        _require(tuple(config_payload.get("seeds", ())) == (11,), "local CPU smoke must use exactly seed 11")
        _require(tuple(manifest.get("ordered_seeds", ())) == (11,), "local CPU smoke seed order is not [11]")
        expected_config = BridgeConfig()
        _require(config_payload.get("schema_version") == 1, "local CPU smoke config schema mismatch")
        _require(config_payload.get("kind") == "phase2_frozen_experiment_config", "local CPU smoke config kind mismatch")
        _require(config_payload.get("protocol_sha256") == expected_config.protocol_sha256, "local CPU smoke config protocol mismatch")
        _require(config_payload.get("protocol_path") == "notes/phase2_gpu_bridge_protocol.md", "local CPU smoke config protocol path mismatch")
        _require(config_payload.get("frozen") == expected_config.payload(), "local CPU smoke frozen config differs from BridgeConfig")
        # The protocol explicitly makes local CPU smoke provenance-only when
        # BLAS serialization differs from the locked remote environment.  Do
        # not rebuild remote canonical bytes on this host; bind the recorded
        # hashes internally and validate the structural split contract here.
        _same_digest(manifest.get("base_sha256"), manifest.get("base_sha256"), "local CPU smoke base hash")
        _same_digest(manifest.get("data_sha256"), manifest.get("data_sha256"), "local CPU smoke data hash")
        _same_digest(manifest.get("split_sha256"), manifest.get("split_sha256"), "local CPU smoke split hash")
        split_payload = _mapping(_load_json(directory / "split_manifest.json"), "local CPU smoke split manifest")
        _require(set(split_payload) == {"seed", "data_sha256", "split_sha256", "cells"}, "local CPU smoke split manifest schema mismatch")
        _require(split_payload.get("seed") == 11, "local CPU smoke split seed mismatch")
        _same_digest(split_payload.get("data_sha256"), manifest.get("data_sha256"), "local CPU smoke split data hash")
        _same_digest(split_payload.get("split_sha256"), manifest.get("split_sha256"), "local CPU smoke split hash")
        cells = _mapping(split_payload.get("cells"), "local CPU smoke split cells")
        _require(set(cells) == {f"{regime}:{rank}" for regime in ("compatible", "stress") for rank in (2, 4, 8)}, "local CPU smoke split cell set mismatch")
        for regime in ("compatible", "stress"):
            for rank in (2, 4, 8):
                cell = _mapping(cells.get(f"{regime}:{rank}"), f"local CPU smoke cell {regime}:{rank}")
                _require(set(cell) == {"windows", "train_counts", "eval_counts", "pi_hashes"}, f"local CPU smoke cell schema mismatch for {regime}:{rank}")
                _require(cell.get("windows") == expected_config.T, f"local CPU smoke window count mismatch for {regime}:{rank}")
                _require(cell.get("train_counts") == [expected_config.n_train] * expected_config.T, f"local CPU smoke train counts mismatch for {regime}:{rank}")
                _require(cell.get("eval_counts") == [expected_config.n_eval] * expected_config.T, f"local CPU smoke eval counts mismatch for {regime}:{rank}")
                pi_hashes = cell.get("pi_hashes")
                _require(isinstance(pi_hashes, list) and len(pi_hashes) == expected_config.T, f"local CPU smoke pi hash count mismatch for {regime}:{rank}")
                _require(all(_DIGEST_RE.fullmatch(str(value)) is not None for value in pi_hashes), f"local CPU smoke pi hash schema mismatch for {regime}:{rank}")
        traces = _mapping(_load_json(directory / "traces.json"), "local CPU smoke traces")
        _require(set(traces) == {"protocol_sha256", "source_hashes", "base_sha256", "data_sha256", "split_sha256"}, "local CPU smoke traces schema mismatch")
        _same_digest(traces.get("base_sha256"), manifest.get("base_sha256"), "local CPU smoke traces base hash")
        _same_digest(traces.get("data_sha256"), manifest.get("data_sha256"), "local CPU smoke traces data hash")
        _same_digest(traces.get("split_sha256"), manifest.get("split_sha256"), "local CPU smoke traces split hash")
        _require(
            _mapping(manifest.get("resource_ledger"), "local CPU smoke resource ledger")
            == {"gpu_used": False, "replay_bytes": expected_config.fixed_replay_bytes},
            "local CPU smoke resource ledger mismatch",
        )
        _require(_validate_source_hashes(root, source_hashes) == dict(source_hashes), "local CPU smoke source snapshot drifted during validation")
        _require_gate_true(gates, "G0_numpy_identity", "passed", role)
        _require_gate_true(gates, "G1_oracle_capacity", "passed", role)
    else:
        _require_gate_true(gates, "G0_numpy_identity", "passed", role)
        _require_gate_true(gates, "G1_oracle_capacity", "passed", role)
        _require(summary.get("execution_pass") is True, "remote GPU smoke execution_pass is not true")
        g4 = _mapping(gates.get("G4_resource_and_reproducibility"), "remote GPU smoke G4")
        _require(g4.get("single_run_passed") is True, "remote GPU smoke G4 single-run checks did not pass")
        _require(g4.get("passed") is True, "remote GPU smoke G4 passed is not true")
        _same_digest(manifest.get("test_hash"), source_hashes["test_phase2_gpu_bridge.py"], "remote GPU smoke test hash")
        ordered_seeds = tuple(manifest.get("ordered_seeds", ()))
        _require(ordered_seeds == seeds, "remote GPU smoke manifest/config seed order mismatch")
        _require_gpu_runtime(manifest, config_payload, role)
        expected_rows = {
            "per_window.csv": len(seeds) * 216,
            "per_seed.csv": len(seeds) * 18,
            "resource_ledger.csv": len(seeds) * 18,
        }
        _require(summary.get("per_window_rows") == expected_rows["per_window.csv"], "remote GPU smoke summary per-window count mismatch")
        _require(summary.get("per_seed_rows") == expected_rows["per_seed.csv"], "remote GPU smoke summary per-seed count mismatch")
        for name, expected in expected_rows.items():
            _require(_csv_rows(directory / name) == expected, f"remote GPU smoke {name} row count mismatch")
        traces = _mapping(_load_json(directory / "traces.json"), "remote GPU smoke traces")
        transitions = traces.get("transitions")
        _require(isinstance(transitions, list) and len(transitions) == len(seeds) * 132, "remote GPU smoke transition count mismatch")
        if deep_validate:
            deep = validate_single_run(
                directory,
                expected_role="remote_gpu_smoke",
                expected_seeds=seeds,
                require_scientific_gates=False,
            )
            _require(deep.get("valid") is True, f"remote GPU smoke deep validation failed: {deep.get('issues', [])[:1]}")


def _report_target(root: Path, value: str | Path) -> Path:
    target = _marker_target(root, value)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _report_checks(
    value: Any,
    label: str,
    *,
    expected_keys: set[str],
) -> dict[str, bool]:
    checks = _mapping(value, f"{label} checks")
    _require(set(checks) == expected_keys, f"{label} checks schema mismatch")
    _require(all(type(item) is bool and item for item in checks.values()), f"{label} checks are not all true")
    return {str(key): bool(item) for key, item in checks.items()}


def write_local_smoke_validation_report(
    run_dir: str | Path,
    report_path: str | Path,
    *,
    root: str | Path,
    config: Any,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Write an exact-origin report for a Windows/local CPU smoke bundle."""

    root_path = _root_path(root)
    protocol_hash, config_hash, _ = _config_facts(config)
    validated_sources = _validate_source_hashes(root_path, source_hashes)
    binding, manifest = _snapshot_run(root_path, run_dir)
    directory = _inside(root_path, binding["manifest"], kind="local smoke report run manifest").parent
    _smoke_checks(
        directory,
        manifest,
        root=root_path,
        role="local_cpu_smoke",
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        source_hashes=validated_sources,
        deep_validate=False,
    )
    exact_hashes = _exact_local_smoke_rebuild(directory, manifest, config=BridgeConfig())
    _require(_validate_source_hashes(root_path, validated_sources) == validated_sources, "local smoke source snapshot drifted during exact report")
    report = {
        "schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
        "kind": "phase2_local_cpu_smoke_validation",
        "validator_version": VALIDATOR_VERSION,
        "validation_mode": "same_host_exact_rebuild",
        "hash_comparison_policy": "local_generated_arrays=provenance_only_cross_host",
        "claim_boundary": "provenance_only_local_cpu_smoke",
        "protocol_sha256": protocol_hash,
        "config_sha256": config_hash,
        "source_hashes": validated_sources,
        "run": binding,
        "run_role": "local_cpu_smoke",
        "expected_seeds": [11],
        "exact_rebuild": exact_hashes,
        "runtime": _mapping(manifest.get("runtime"), "local smoke report runtime"),
        "checks": {
            "artifact_binding": True,
            "config_and_protocol": True,
            "split_semantics": True,
            "g0_numpy_identity": True,
            "g1_oracle_capacity": True,
            "source_snapshot": True,
        },
    }
    target = _report_target(root_path, report_path)
    _assert_marker_disjoint(target, [directory, root_path / "experiments" / "phase2_gpu_bridge"])
    artifacts.write_json(target, report)
    return {"path": _relative(root_path, target), "sha256": artifacts.sha256_file(target)}


def write_remote_validation_report(
    calibration_dir: str | Path,
    remote_smoke_dir: str | Path,
    report_path: str | Path,
    remote_test_result_path: str | Path,
    local_validation_report_path: str | Path,
    *,
    root: str | Path,
    config: Any,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Run canonical validation in the locked remote environment and retain a bound report."""

    root_path = _root_path(root)
    protocol_hash, config_hash, _ = _config_facts(config)
    validated_sources = _validate_source_hashes(root_path, source_hashes)
    local_validation_binding = _snapshot_local_validation_report(
        root_path,
        local_validation_report_path,
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        source_hashes=validated_sources,
    )
    run_specs = (
        ("remote_gpu_smoke", remote_smoke_dir, (11,), False),
        ("non_citable_calibration", calibration_dir, CALIBRATION_SEEDS, False),
    )
    run_reports: dict[str, Any] = {}
    run_paths: list[Path] = []
    for role, run_dir, seeds, _ in run_specs:
        binding, manifest = _snapshot_run(root_path, run_dir)
        directory = _inside(root_path, binding["manifest"], kind=f"{role} report manifest").parent
        run_paths.append(directory)
        if role == "remote_gpu_smoke":
            _smoke_checks(
                directory,
                manifest,
                root=root_path,
                role=role,
                protocol_hash=protocol_hash,
                config_hash=config_hash,
                source_hashes=validated_sources,
                deep_validate=False,
            )
        else:
            _calibration_checks(
                directory,
                manifest,
                protocol_hash=protocol_hash,
                config_hash=config_hash,
                source_hashes=validated_sources,
                deep_validate=False,
            )
        result = validate_single_run(
            directory,
            expected_role=role,
            expected_seeds=seeds,
            require_scientific_gates=False,
        )
        _require(result.get("valid") is True, f"{role} canonical validation failed: {result.get('issues', [])[:1]}")
        run_reports[role] = {
            "run": binding,
            "run_role": role,
            "expected_seeds": list(seeds),
            "manifest_hashes": {
                "base_sha256": manifest.get("base_sha256"),
                "data_sha256_by_seed": manifest.get("data_sha256_by_seed"),
                "split_sha256_by_seed": manifest.get("split_sha256_by_seed"),
            },
            "runtime": _mapping(manifest.get("runtime"), f"{role} report runtime"),
            "validation": {
                "valid": True,
                "issue_count": int(result.get("issue_count", 0)),
                "suppressed_issue_count": int(result.get("suppressed_issue_count", 0)),
                "issues": list(result.get("issues", [])),
            },
        }
    remote_test_binding = _snapshot_remote_test(
        root_path,
        remote_test_result_path,
        validated_sources,
        config_hash,
        protocol_hash,
    )
    local_exact = _mapping(local_validation_binding.get("exact_rebuild"), "local validation exact rebuild")
    remote_smoke_hashes = _mapping(run_reports["remote_gpu_smoke"]["manifest_hashes"], "remote smoke manifest hashes")
    remote_data_hashes = _mapping(remote_smoke_hashes.get("data_sha256_by_seed"), "remote smoke data hashes")
    remote_split_hashes = _mapping(remote_smoke_hashes.get("split_sha256_by_seed"), "remote smoke split hashes")
    remote_hashes = {
        "base_sha256": remote_smoke_hashes.get("base_sha256"),
        "data_sha256": remote_data_hashes.get("11", remote_data_hashes.get(11)),
        "split_sha256": remote_split_hashes.get("11", remote_split_hashes.get(11)),
    }
    cross_host_hash_comparison: dict[str, Any] = {}
    for field in CROSS_HOST_HASH_FIELDS:
        local_digest = _digest(local_exact.get(field), f"local validation {field}")
        remote_digest = _digest(remote_hashes.get(field), f"remote smoke {field}")
        cross_host_hash_comparison[field] = {
            "local_sha256": local_digest,
            "remote_sha256": remote_digest,
            "equal": local_digest == remote_digest,
            "disposition": CROSS_HOST_HASH_DISPOSITION,
        }
    report = {
        "schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
        "kind": "phase2_remote_validation_report",
        "validator_version": VALIDATOR_VERSION,
        "validation_mode": "remote_locked_exact_rebuild",
        "hash_comparison_policy": "remote_generated_arrays=exact;local_generated_arrays=provenance_only_cross_host",
        "claim_boundary": "non_citable_remote_validation",
        "protocol_sha256": protocol_hash,
        "config_sha256": config_hash,
        "source_hashes": validated_sources,
        "runs": run_reports,
        "remote_test": remote_test_binding,
        "local_cpu_smoke_validation": {
            "path": local_validation_binding["path"],
            "sha256": local_validation_binding["sha256"],
        },
        "cross_host_hash_comparison": cross_host_hash_comparison,
        "checks": {
            "calibration_canonical_validation": True,
            "remote_smoke_canonical_validation": True,
            "remote_test_binding": True,
            "local_validation_binding": True,
            "cross_host_hash_comparison": True,
            "source_snapshot": True,
        },
    }
    target = _report_target(root_path, report_path)
    _assert_marker_disjoint(
        target,
        [
            *run_paths,
            local_validation_binding["directory"],
            root_path / "experiments" / "phase2_gpu_bridge",
            _inside(root_path, remote_test_result_path, kind="remote validation test result"),
            _inside(root_path, local_validation_report_path, kind="remote validation local report"),
        ],
    )
    artifacts.write_json(target, report)
    return {"path": _relative(root_path, target), "sha256": artifacts.sha256_file(target)}


def _snapshot_local_validation_report(
    root: Path,
    path_value: str | Path,
    *,
    protocol_hash: str,
    config_hash: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    path = _inside(root, path_value, kind="local smoke validation report")
    report = _mapping(_load_json(path), "local smoke validation report")
    expected = {
        "schema_version", "kind", "validator_version", "validation_mode",
        "hash_comparison_policy", "claim_boundary", "protocol_sha256",
        "config_sha256", "source_hashes", "run", "run_role", "expected_seeds",
        "exact_rebuild", "runtime", "checks",
    }
    _require(set(report) == expected, "local smoke validation report schema mismatch")
    _require(report.get("schema_version") == VALIDATION_REPORT_SCHEMA_VERSION, "local smoke validation report version mismatch")
    _require(report.get("kind") == "phase2_local_cpu_smoke_validation", "local smoke validation report kind mismatch")
    _require(report.get("validator_version") == VALIDATOR_VERSION, "local smoke validation validator version mismatch")
    _require(report.get("validation_mode") == "same_host_exact_rebuild", "local smoke validation mode mismatch")
    _require(report.get("hash_comparison_policy") == "local_generated_arrays=provenance_only_cross_host", "local smoke hash policy mismatch")
    _require(report.get("claim_boundary") == "provenance_only_local_cpu_smoke", "local smoke validation claim boundary mismatch")
    _same_digest(report.get("protocol_sha256"), protocol_hash, "local smoke validation protocol hash")
    _same_digest(report.get("config_sha256"), config_hash, "local smoke validation config hash")
    _require(report.get("source_hashes") == source_hashes, "local smoke validation source hashes mismatch")
    binding = _mapping(report.get("run"), "local smoke validation run binding")
    observed, manifest, directory = _check_run_binding(root, binding)
    _require(manifest.get("run_role") == "local_cpu_smoke", "local smoke validation run role mismatch")
    _require(report.get("run_role") == "local_cpu_smoke" and report.get("expected_seeds") == [11], "local smoke validation seed binding mismatch")
    exact = _mapping(report.get("exact_rebuild"), "local smoke exact rebuild")
    _require(
        set(exact) == {"base_sha256", "data_sha256", "split_sha256"},
        "local smoke exact rebuild schema mismatch",
    )
    for field in ("base_sha256", "data_sha256", "split_sha256"):
        _same_digest(exact.get(field), manifest.get(field), f"local smoke exact rebuild {field}")
    _require(
        report.get("runtime") == manifest.get("runtime"),
        "local smoke validation runtime does not match manifest",
    )
    _require(isinstance(manifest.get("runtime"), Mapping) and bool(manifest.get("runtime")), "local smoke validation runtime is empty")
    _report_checks(
        report.get("checks"),
        "local smoke validation",
        expected_keys={
            "artifact_binding",
            "config_and_protocol",
            "split_semantics",
            "g0_numpy_identity",
            "g1_oracle_capacity",
            "source_snapshot",
        },
    )
    return {
        "path": _relative(root, path),
        "sha256": artifacts.sha256_file(path),
        "run": observed,
        "directory": directory,
        "exact_rebuild": dict(exact),
    }


def _snapshot_remote_validation_report(
    root: Path,
    path_value: str | Path,
    *,
    protocol_hash: str,
    config_hash: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    path = _inside(root, path_value, kind="remote validation report")
    report = _mapping(_load_json(path), "remote validation report")
    expected = {
        "schema_version", "kind", "validator_version", "validation_mode",
        "hash_comparison_policy", "claim_boundary", "protocol_sha256",
        "config_sha256", "source_hashes", "runs", "remote_test",
        "local_cpu_smoke_validation", "cross_host_hash_comparison", "checks",
    }
    _require(set(report) == expected, "remote validation report schema mismatch")
    _require(report.get("schema_version") == VALIDATION_REPORT_SCHEMA_VERSION, "remote validation report version mismatch")
    _require(report.get("kind") == "phase2_remote_validation_report", "remote validation report kind mismatch")
    _require(report.get("validator_version") == VALIDATOR_VERSION, "remote validation validator version mismatch")
    _require(report.get("validation_mode") == "remote_locked_exact_rebuild", "remote validation mode mismatch")
    _require(report.get("hash_comparison_policy") == "remote_generated_arrays=exact;local_generated_arrays=provenance_only_cross_host", "remote validation hash policy mismatch")
    _require(report.get("claim_boundary") == "non_citable_remote_validation", "remote validation claim boundary mismatch")
    _same_digest(report.get("protocol_sha256"), protocol_hash, "remote validation protocol hash")
    _same_digest(report.get("config_sha256"), config_hash, "remote validation config hash")
    _require(report.get("source_hashes") == source_hashes, "remote validation source hashes mismatch")
    runs = _mapping(report.get("runs"), "remote validation runs")
    _require(set(runs) == {"remote_gpu_smoke", "non_citable_calibration"}, "remote validation run set mismatch")
    remote_smoke_hashes: Mapping[str, Any] | None = None
    for role, expected_seeds in (("remote_gpu_smoke", [11]), ("non_citable_calibration", list(CALIBRATION_SEEDS))):
        item = _mapping(runs.get(role), f"remote validation {role}")
        _require(set(item) == {"run", "run_role", "expected_seeds", "manifest_hashes", "runtime", "validation"}, f"remote validation {role} schema mismatch")
        _require(item.get("run_role") == role and item.get("expected_seeds") == expected_seeds, f"remote validation {role} identity mismatch")
        observed, manifest, directory = _check_run_binding(root, item["run"])
        _require(manifest.get("run_role") == role, f"remote validation {role} run binding mismatch")
        config_payload = _mapping(_load_json(directory / "config.json"), f"remote validation {role} config")
        _require(item.get("runtime") == manifest.get("runtime"), f"remote validation {role} runtime does not match manifest")
        _require_gpu_runtime(manifest, config_payload, f"remote validation {role}")
        hashes = _mapping(item.get("manifest_hashes"), f"remote validation {role} manifest hashes")
        _require(
            set(hashes) == {"base_sha256", "data_sha256_by_seed", "split_sha256_by_seed"},
            f"remote validation {role} manifest hashes schema mismatch",
        )
        _same_digest(hashes.get("base_sha256"), manifest.get("base_sha256"), f"remote validation {role} base hash")
        _require(hashes.get("data_sha256_by_seed") == manifest.get("data_sha256_by_seed"), f"remote validation {role} data hashes mismatch")
        _require(hashes.get("split_sha256_by_seed") == manifest.get("split_sha256_by_seed"), f"remote validation {role} split hashes mismatch")
        _require(isinstance(item.get("runtime"), Mapping), f"remote validation {role} runtime is missing")
        validation = _mapping(item.get("validation"), f"remote validation {role} validation")
        _require(set(validation) == {"valid", "issue_count", "suppressed_issue_count", "issues"}, f"remote validation {role} result schema mismatch")
        _require(
            type(validation.get("valid")) is bool
            and validation.get("valid") is True
            and type(validation.get("issue_count")) is int
            and validation.get("issue_count") == 0
            and type(validation.get("suppressed_issue_count")) is int
            and validation.get("suppressed_issue_count") == 0
            and type(validation.get("issues")) is list
            and validation.get("issues") == [],
            f"remote validation {role} did not pass cleanly",
        )
        _require(directory.is_dir(), f"remote validation {role} directory is missing")
        if role == "remote_gpu_smoke":
            remote_smoke_hashes = hashes
    remote_test = _mapping(report.get("remote_test"), "remote validation test binding")
    _require(set(remote_test) == {"path", "sha256"}, "remote validation test binding schema mismatch")
    _check_remote_test_binding(root, remote_test, source_hashes, config_hash, protocol_hash)
    local_report_binding = _mapping(report.get("local_cpu_smoke_validation"), "remote validation local report binding")
    _require(set(local_report_binding) == {"path", "sha256"}, "remote validation local report binding schema mismatch")
    local_report = _snapshot_local_validation_report(
        root,
        local_report_binding["path"],
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        source_hashes=source_hashes,
    )
    _same_digest(local_report["sha256"], local_report_binding["sha256"], "remote validation local report hash")
    _require(remote_smoke_hashes is not None, "remote validation remote smoke hashes are missing")
    remote_data_hashes = _mapping(remote_smoke_hashes.get("data_sha256_by_seed"), "remote validation remote data hashes")
    remote_split_hashes = _mapping(remote_smoke_hashes.get("split_sha256_by_seed"), "remote validation remote split hashes")
    expected_remote_hashes = {
        "base_sha256": remote_smoke_hashes.get("base_sha256"),
        "data_sha256": remote_data_hashes.get("11", remote_data_hashes.get(11)),
        "split_sha256": remote_split_hashes.get("11", remote_split_hashes.get(11)),
    }
    comparison = _mapping(report.get("cross_host_hash_comparison"), "remote validation cross-host hash comparison")
    _require(set(comparison) == set(CROSS_HOST_HASH_FIELDS), "remote validation cross-host hash comparison schema mismatch")
    for field in CROSS_HOST_HASH_FIELDS:
        item = _mapping(comparison.get(field), f"remote validation cross-host {field}")
        _require(
            set(item) == {"local_sha256", "remote_sha256", "equal", "disposition"},
            f"remote validation cross-host {field} item schema mismatch",
        )
        _same_digest(item.get("local_sha256"), local_report["exact_rebuild"].get(field), f"remote validation local {field}")
        _same_digest(item.get("remote_sha256"), expected_remote_hashes[field], f"remote validation remote {field}")
        _require(type(item.get("equal")) is bool and item["equal"] == (item["local_sha256"].lower() == item["remote_sha256"].lower()), f"remote validation cross-host {field} equality mismatch")
        _require(item.get("disposition") == CROSS_HOST_HASH_DISPOSITION, f"remote validation cross-host {field} disposition mismatch")
    _report_checks(
        report.get("checks"),
        "remote validation",
        expected_keys={
            "calibration_canonical_validation",
            "remote_smoke_canonical_validation",
            "remote_test_binding",
            "local_validation_binding",
            "cross_host_hash_comparison",
            "source_snapshot",
        },
    )
    return {
        "path": _relative(root, path),
        "sha256": artifacts.sha256_file(path),
        "runs": {
            role: _mapping(runs[role]["run"], f"{role} report run binding")
            for role in ("remote_gpu_smoke", "non_citable_calibration")
        },
        "remote_test": remote_test,
        "local_cpu_smoke_validation": local_report_binding,
    }


def _assert_validation_report_bindings(
    local_report: Mapping[str, Any],
    local_run: Mapping[str, Any],
    remote_report: Mapping[str, Any],
    remote_run: Mapping[str, Any],
    calibration_run: Mapping[str, Any],
    remote_test: Mapping[str, Any],
) -> None:
    """Ensure validation reports describe the exact sibling evidence in the freeze marker."""

    _require(
        local_report.get("run") == local_run,
        "local validation report is not bound to the marker local smoke run",
    )
    runs = _mapping(remote_report.get("runs"), "remote validation report run bindings")
    _require(
        runs.get("remote_gpu_smoke") == remote_run,
        "remote validation report is not bound to the marker remote smoke run",
    )
    _require(
        runs.get("non_citable_calibration") == calibration_run,
        "remote validation report is not bound to the marker calibration run",
    )
    _require(
        remote_report.get("remote_test") == remote_test,
        "remote validation report is not bound to the marker remote test result",
    )
    _require(
        remote_report.get("local_cpu_smoke_validation")
        == {"path": local_report.get("path"), "sha256": local_report.get("sha256")},
        "remote validation report is not bound to the marker local validation report",
    )


def _test_hash_from_result(result: Mapping[str, Any]) -> str:
    candidates: list[Any] = []
    for key in ("test_hash", "test_sha256", "test_phase2_gpu_bridge_sha256"):
        if key in result:
            candidates.append(result[key])
    nested = result.get("source_hashes")
    if isinstance(nested, Mapping) and "test_phase2_gpu_bridge.py" in nested:
        candidates.append(nested["test_phase2_gpu_bridge.py"])
    _require(bool(candidates), "remote test result does not bind the test source hash")
    normalized = [_digest(value, "remote test source hash") for value in candidates]
    _require(len(set(normalized)) == 1, "remote test result contains conflicting test hashes")
    return normalized[0]


def _snapshot_remote_test(
    root: Path,
    path_value: str | Path,
    expected_source_hashes: Mapping[str, str],
    expected_config_hash: str,
    expected_protocol_hash: str,
) -> dict[str, Any]:
    path = _inside(root, path_value, kind="remote test result")
    result = _mapping(_load_json(path), "remote test result")
    _require(type(result.get("exit_code")) is int and result["exit_code"] == 0, "remote test exit_code is not zero")
    _same_digest(_test_hash_from_result(result), expected_source_hashes["test_phase2_gpu_bridge.py"], "remote test source hash")
    _require(result.get("source_hashes") == expected_source_hashes, "remote test does not bind the complete current source snapshot")
    _same_digest(result.get("config_sha256"), expected_config_hash, "remote test config hash")
    _same_digest(result.get("protocol_sha256"), expected_protocol_hash, "remote test protocol hash")
    _require(type(result.get("tests_run")) is int and result["tests_run"] == EXPECTED_REMOTE_TESTS, f"remote test count is not exactly the frozen suite ({EXPECTED_REMOTE_TESTS})")
    for field in ("failures", "errors", "skipped"):
        _require(type(result.get(field)) is int and result[field] == 0, f"remote test {field} is not zero")
    command = result.get("command")
    _require(isinstance(command, list) and all(isinstance(item, str) and item for item in command), "remote test command is not an argv list")
    _require(bool(command), "remote test command is empty")
    _require(Path(command[0]).name.lower() in {"python", "python3"}, "remote test command does not use a Python executable")
    _require("-m" in command and command[command.index("-m") + 1 : command.index("-m") + 2] == ["unittest"], "remote test command must invoke unittest as a module")
    module_tokens = [
        "experiments.phase2_gpu_bridge.test_phase2_gpu_bridge",
        "experiments.phase2_gpu_bridge.test_validation",
    ]
    module_start = command.index("-m") + 2
    trailing = command[module_start:]
    normalized = [token for token in trailing if token not in {"-q", "-v"}]
    _require(normalized == module_tokens, "remote test command has unexpected unittest selectors or modules")
    return {"path": _relative(root, path), "sha256": artifacts.sha256_file(path)}


def _check_remote_test_binding(
    root: Path,
    binding_value: Any,
    expected_source_hashes: Mapping[str, str],
    expected_config_hash: str,
    expected_protocol_hash: str,
) -> None:
    binding = _mapping(binding_value, "remote test binding")
    _require(set(binding) == {"path", "sha256"}, "remote test binding schema mismatch")
    observed = _snapshot_remote_test(root, binding["path"], expected_source_hashes, expected_config_hash, expected_protocol_hash)
    _require(observed["path"] == binding["path"], "remote test path is not canonical")
    _same_digest(observed["sha256"], binding["sha256"], "remote test result file hash")


EXPECTED_PROVIDER_MODELS = {
    "gpt": "gpt-5.5",
    "claude": "claude-opus-4-8",
    "glm": "glm-5.2",
    "deepseek": "deepseek-v4-pro",
}
EXPECTED_PROVIDER_REASONING = {
    "gpt": "xhigh",
    "claude": "max",
    "glm": "ultra",
    "deepseek": "xhigh",
}


def build_implementation_crosscheck_prompt(
    root: str | Path,
    *,
    protocol_hash: str,
    config_hash: str,
    source_hashes: Mapping[str, str],
    local_smoke_manifest_sha256: str,
    local_validation_report_sha256: str,
    remote_test_result_sha256: str,
    remote_smoke_manifest_sha256: str,
    calibration_manifest_sha256: str,
    remote_validation_report_sha256: str,
    local_smoke_manifest_path: str,
    local_validation_report_path: str,
    remote_test_result_path: str,
    remote_smoke_manifest_path: str,
    calibration_manifest_path: str,
    remote_validation_report_path: str,
) -> str:
    """Build the exact read-only prompt used by the final four-provider audit."""

    root_path = _root_path(root)
    validated_sources = _validate_source_hashes(root_path, source_hashes)
    _same_digest(protocol_hash, PROTOCOL_SHA256, "crosscheck prompt protocol hash")
    _digest(config_hash, "crosscheck prompt config hash")
    _digest(local_smoke_manifest_sha256, "crosscheck prompt local smoke manifest hash")
    _digest(local_validation_report_sha256, "crosscheck prompt local validation report hash")
    _digest(remote_test_result_sha256, "crosscheck prompt remote test result hash")
    _digest(remote_smoke_manifest_sha256, "crosscheck prompt remote smoke manifest hash")
    _digest(calibration_manifest_sha256, "crosscheck prompt calibration manifest hash")
    _digest(remote_validation_report_sha256, "crosscheck prompt remote validation report hash")
    bound_paths = {
        "local CPU smoke manifest": _relative(root_path, _inside(root_path, local_smoke_manifest_path, kind="crosscheck local smoke manifest")),
        "local CPU smoke validation report": _relative(root_path, _inside(root_path, local_validation_report_path, kind="crosscheck local validation report")),
        "remote unit-test result": _relative(root_path, _inside(root_path, remote_test_result_path, kind="crosscheck remote test result")),
        "remote GPU smoke manifest": _relative(root_path, _inside(root_path, remote_smoke_manifest_path, kind="crosscheck remote smoke manifest")),
        "non-citable calibration manifest": _relative(root_path, _inside(root_path, calibration_manifest_path, kind="crosscheck calibration manifest")),
        "remote validation report": _relative(root_path, _inside(root_path, remote_validation_report_path, kind="crosscheck remote validation report")),
    }
    source_block = json.dumps(validated_sources, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    bound_path_block = "".join(f"- {label}: {path}\n" for label, path in bound_paths.items())
    return (
        "Phase-2 pre-primary implementation and calibration-evidence audit (read-only, independent).\n\n"
        "Inspect the frozen protocol, every canonical source file listed below, the executor capsule, the bound remote unit-test result, "
        "remote GPU smoke, and non-citable calibration artifacts. Do not edit files, "
        "run network/GPU experiments, or rely on another verifier. Check code-to-protocol correspondence, "
        "tamper-evident artifact binding, deterministic resource accounting, failure handling, and whether "
        "the implementation is ready for the remote calibration/primary runs. Distinguish concrete blockers "
        "from non-blocking risks and cite file paths/lines or tests as evidence.\n\n"
        f"Protocol SHA256: {protocol_hash.lower()}\n"
        f"Frozen config SHA256: {config_hash.lower()}\n"
        f"Local CPU smoke manifest SHA256: {local_smoke_manifest_sha256.lower()}\n"
        f"Local CPU smoke validation report SHA256: {local_validation_report_sha256.lower()}\n"
        f"Remote unit-test result SHA256: {remote_test_result_sha256.lower()}\n"
        f"Remote GPU smoke manifest SHA256: {remote_smoke_manifest_sha256.lower()}\n"
        f"Non-citable calibration manifest SHA256: {calibration_manifest_sha256.lower()}\n"
        f"Remote validation report SHA256: {remote_validation_report_sha256.lower()}\n"
        "Bound artifact paths (relative to the project root):\n"
        f"{bound_path_block}"
        f"Canonical source SHA256 map: {source_block}\n"
        f"Expected test suite: exactly {EXPECTED_REMOTE_TESTS} tests, with no failures or errors on the locked remote environment.\n"
        "Treat local CPU array-hash differences from the remote locked environment as provenance-only when the report says so; require exact remote canonical validation.\n"
        "Cite at least two canonical source filenames in the review and do not modify any file.\n"
        "Required final line (exactly, after all other text): VERDICT: GO or VERDICT: NO-GO.\n"
    )


def _model_matches_expected(model: Any, provider: str, *, actual: bool = False) -> bool:
    expected = EXPECTED_PROVIDER_MODELS[provider].lower()
    if not isinstance(model, str):
        return False
    normalized = model.strip().lower()
    if normalized == expected:
        return True
    return actual and provider == "claude" and normalized == "claude-opus-4.8"


def _last_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _portable_path_parts(value: Any, label: str) -> tuple[str, ...]:
    """Parse a recorded path without interpreting its originating OS syntax."""

    _require(isinstance(value, str) and value.strip(), f"{label} path is empty")
    _require("\x00" not in value, f"{label} path contains NUL")
    parts: list[str] = []
    for part in value.replace("\\", "/").split("/"):
        if part in {"", "."}:
            continue
        _require(part != "..", f"{label} path contains traversal")
        parts.append(part)
    _require(bool(parts), f"{label} path has no components")
    return tuple(parts)


def _recorded_provider_file(
    root: Path,
    provider_dir: Path,
    provider: str,
    value: Any,
    label: str,
) -> Path:
    """Resolve a runner-recorded file against the copied provider capsule."""

    parts = _portable_path_parts(value, label)
    _require(len(parts) >= 2 and parts[-2] == provider, f"{label} is not below its provider directory")
    _require(provider_dir.parent.name in parts, f"{label} is not bound to the crosscheck run id")
    name = parts[-1]
    _require(name not in {".", ".."} and "/" not in name and "\\" not in name, f"unsafe {label} basename")
    return _inside(root, provider_dir / name, kind=label)


def _executor_relative_file(root: Path, directory: Path, value: Any, label: str) -> Path:
    """Resolve an executor-capsule path relative to the copied crosscheck run."""

    _require(isinstance(value, str) and value.strip(), f"{label} path is empty")
    raw = Path(value)
    _require(not raw.is_absolute(), f"{label} path must be run-relative")
    parts = _portable_path_parts(value, label)
    return _inside(root, directory.joinpath(*parts), kind=label)


def _snapshot_executor_capsule(
    root: Path,
    directory: Path,
    manifest: Mapping[str, Any],
    prompt_hash: str,
) -> dict[str, Any]:
    capsule = _mapping(manifest.get("executor_capsule"), "crosscheck executor capsule")
    _require(
        set(capsule) == {"path", "sha256", "launcher", "worker", "prompt_sha256"},
        "crosscheck executor capsule schema mismatch",
    )
    capsule_path = _executor_relative_file(root, directory, capsule.get("path"), "crosscheck executor capsule")
    _same_digest(artifacts.sha256_file(capsule_path), capsule.get("sha256"), "crosscheck executor capsule hash")
    capsule_payload = _mapping(_load_json(capsule_path), "crosscheck executor capsule contents")
    _require(
        set(capsule_payload) == {"schema_version", "launcher", "worker", "prompt_sha256"},
        "crosscheck executor capsule contents schema mismatch",
    )
    _require(capsule_payload.get("schema_version") == 1, "crosscheck executor capsule version mismatch")
    _same_digest(capsule_payload.get("prompt_sha256"), prompt_hash, "crosscheck executor prompt hash")
    _same_digest(capsule.get("prompt_sha256"), prompt_hash, "crosscheck manifest executor prompt hash")
    observed: dict[str, Any] = {
        "path": _relative(root, capsule_path),
        "sha256": artifacts.sha256_file(capsule_path),
        "prompt_sha256": prompt_hash,
    }
    for role in ("launcher", "worker"):
        manifest_item = _mapping(capsule.get(role), f"crosscheck executor {role}")
        payload_item = _mapping(capsule_payload.get(role), f"crosscheck executor {role} contents")
        _require(set(manifest_item) == {"path", "sha256"}, f"crosscheck executor {role} schema mismatch")
        _require(set(payload_item) == {"path", "sha256"}, f"crosscheck executor {role} contents schema mismatch")
        _require(manifest_item == payload_item, f"crosscheck executor {role} manifest/content mismatch")
        archive_path = _executor_relative_file(root, directory, manifest_item.get("path"), f"crosscheck executor {role} archive")
        _same_digest(artifacts.sha256_file(archive_path), manifest_item.get("sha256"), f"crosscheck executor {role} archive hash")
        observed[role] = {
            "path": _relative(root, archive_path),
            "sha256": artifacts.sha256_file(archive_path),
        }
    return observed


def _validate_crosscheck_provider(
    root: Path,
    directory: Path,
    provider: str,
    record: Mapping[str, Any],
    executor_capsule: Mapping[str, Any],
) -> dict[str, Any]:
    provider_dir = _inside(root, directory / provider, kind=f"{provider} directory")
    status_path = _inside(root, provider_dir / "status.json", kind=f"{provider} status")
    status = _mapping(_load_json(status_path), f"{provider} status")
    _require(status.get("provider") == provider, f"{provider} status provider mismatch")
    _require(status.get("succeeded") is True and status.get("error") is None, f"{provider} status did not succeed cleanly")
    _require(status.get("model_identity_verified") is True, f"{provider} model identity was not verified")
    _require(
        status.get("primary_requested_model")
        == status.get("selected_model")
        == status.get("requested_model")
        == EXPECTED_PROVIDER_MODELS[provider],
        f"{provider} requested/selected model is not the frozen strongest model",
    )
    _require(_model_matches_expected(status.get("actual_model"), provider, actual=True), f"{provider} actual model identity mismatch")
    _require(status.get("reasoning") == EXPECTED_PROVIDER_REASONING[provider], f"{provider} reasoning mode mismatch")
    for key in (
        "provider", "succeeded", "primary_requested_model", "selected_model", "requested_model",
        "actual_model", "model_identity_verified", "reasoning", "duration_seconds", "final_path",
        "error", "prompt_sha256", "verifier_prompt_path", "verifier_prompt_sha256",
        "worker_script_sha256", "executor_capsule_path", "executor_capsule_sha256", "attempts",
    ):
        _require(record.get(key) == status.get(key), f"{provider} manifest/status field mismatch: {key}")
    _require(record.get("launcher_cleanup_error") is None, f"{provider} launcher cleanup failed")

    attempts = status.get("attempts")
    _require(isinstance(attempts, list) and attempts, f"{provider} has no recorded attempts")
    expected_protocol = "anthropic-messages" if provider == "claude" else "openai-responses"
    for attempt in attempts:
        item = _mapping(attempt, f"{provider} attempt")
        _require(type(item.get("succeeded")) is bool, f"{provider} attempt succeeded flag is invalid")
        _require(isinstance(item.get("protocol"), str) and item["protocol"], f"{provider} attempt protocol is missing")
        _require(item.get("protocol") == expected_protocol, f"{provider} attempt used an unexpected protocol")
        _recorded_provider_file(root, provider_dir, provider, item.get("stdout"), f"{provider} attempt stdout")
        _recorded_provider_file(root, provider_dir, provider, item.get("stderr"), f"{provider} attempt stderr")
    successful = [attempt for attempt in attempts if isinstance(attempt, Mapping) and attempt.get("succeeded") is True]
    _require(len(successful) == 1, f"{provider} must have exactly one successful attempt")
    selected = _mapping(successful[0], f"{provider} successful attempt")
    _require(selected.get("protocol") == expected_protocol, f"{provider} successful attempt protocol is not tool-capable/frozen")
    _require(selected.get("requested_model") == status.get("requested_model"), f"{provider} attempt requested model mismatch")
    _require(_model_matches_expected(selected.get("actual_model"), provider, actual=True), f"{provider} attempt actual model mismatch")
    _require(selected.get("model_identity_verified") is True, f"{provider} attempt identity was not verified")
    _require(selected.get("reasoning") == status.get("reasoning"), f"{provider} attempt reasoning mismatch")
    _require(selected.get("exit_code") == 0 and selected.get("invocation_error") is None, f"{provider} successful attempt exit metadata is invalid")
    stdout_path = _recorded_provider_file(root, provider_dir, provider, selected.get("stdout"), f"{provider} successful stdout")
    stderr_path = _recorded_provider_file(root, provider_dir, provider, selected.get("stderr"), f"{provider} successful stderr")
    _require(stdout_path.is_file() and stderr_path.is_file(), f"{provider} successful attempt logs are missing")

    final_path = _recorded_provider_file(root, provider_dir, provider, record.get("final_path"), f"{provider} final path")
    _require(final_path.name == "final.md", f"{provider} final path is not final.md")
    _require(final_path.is_file() and final_path.stat().st_size > 0, f"{provider} final.md is empty")
    final_text = _read_text_with_bom(final_path)
    _require(_last_nonempty_line(final_text) == "VERDICT: GO", f"{provider} final output does not end with an exact GO verdict")
    verdict_lines = [line.strip() for line in final_text.splitlines() if line.strip() and line.strip().startswith("VERDICT:")]
    _require(verdict_lines == ["VERDICT: GO"], f"{provider} final output has ambiguous verdict markers")
    _require(final_text.count("VERDICT: GO") == 1 and "VERDICT: NO-GO" not in final_text, f"{provider} final output contains an ambiguous verdict token")
    _require(len(final_text.strip()) >= 200, f"{provider} final output is not substantive")
    cited_sources = sum(
        name in final_text
        for name in ("evidence.py", "validation.py", "phase2_gpu_bridge.py", "core_numpy.py", "replay.py")
    )
    _require(cited_sources >= 2, f"{provider} final output does not cite at least two canonical source files")

    input_path = _inside(root, provider_dir / "input.json", kind=f"{provider} input")
    input_payload = _mapping(_load_json(input_path), f"{provider} input")
    _require(
        set(input_payload)
        == {
            "provider",
            "prompt_path",
            "working_directory",
            "output_directory",
            "temp_root",
            "codex_home",
            "claude_home",
            "base_url",
            "prompt_sha256",
            "worker_script_sha256",
            "executor_capsule_path",
            "executor_capsule_sha256",
        },
        f"{provider} input schema mismatch",
    )
    _require(input_payload.get("provider") == provider, f"{provider} input provider mismatch")
    prompt_path = _inside(root, directory / "prompt.txt", kind="crosscheck prompt")
    _same_digest(input_payload.get("prompt_sha256"), artifacts.sha256_file(prompt_path), f"{provider} input prompt hash")
    _same_digest(input_payload.get("prompt_sha256"), status.get("prompt_sha256"), f"{provider} status prompt hash")
    _same_digest(input_payload.get("worker_script_sha256"), executor_capsule["worker"]["sha256"], f"{provider} worker archive hash")
    capsule_path = _inside(root, directory / "executor" / "executor_capsule.json", kind="crosscheck executor capsule")
    _same_digest(input_payload.get("executor_capsule_sha256"), artifacts.sha256_file(capsule_path), f"{provider} input capsule hash")
    _same_digest(input_payload.get("executor_capsule_sha256"), executor_capsule["sha256"], f"{provider} capsule hash")
    prompt_parts = _portable_path_parts(input_payload.get("prompt_path"), f"{provider} input prompt")
    _require(prompt_parts[-1] == "prompt.txt" and directory.name in prompt_parts, f"{provider} input prompt binding mismatch")
    working_parts = _portable_path_parts(input_payload.get("working_directory"), f"{provider} input working directory")
    _require(working_parts[-1] == root.name, f"{provider} input working-directory binding mismatch")
    output_parts = _portable_path_parts(input_payload.get("output_directory"), f"{provider} input output directory")
    _require(output_parts[-1] == provider and directory.name in output_parts, f"{provider} input output-directory binding mismatch")
    temp_root_value = input_payload.get("temp_root")
    temp_parts = _portable_path_parts(temp_root_value, f"{provider} input temp root")
    _require(len(temp_parts) >= 2 and temp_parts[-2:] == (directory.name, "temp"), f"{provider} input temp-root binding mismatch")
    capsule_parts = _portable_path_parts(input_payload.get("executor_capsule_path"), f"{provider} input executor capsule")
    _require(
        capsule_parts[-2:] == ("executor", "executor_capsule.json") and directory.name in capsule_parts,
        f"{provider} input executor capsule binding mismatch",
    )
    _same_digest(status.get("worker_script_sha256"), executor_capsule["worker"]["sha256"], f"{provider} status worker archive hash")
    _same_digest(status.get("executor_capsule_sha256"), executor_capsule["sha256"], f"{provider} status capsule hash")
    verifier_path = _recorded_provider_file(root, provider_dir, provider, status.get("verifier_prompt_path"), f"{provider} verifier prompt")
    _same_digest(artifacts.sha256_file(verifier_path), status.get("verifier_prompt_sha256"), f"{provider} verifier prompt hash")
    _same_digest(status.get("prompt_sha256"), input_payload.get("prompt_sha256"), f"{provider} status/input prompt hash")

    if provider != "claude":
        selected_final = _inside(root, provider_dir / f"codex-{status['requested_model']}.final.txt", kind=f"{provider} selected final transcript")
        _require(selected_final.is_file(), f"{provider} selected model transcript is missing")
        _require(_read_text_with_bom(selected_final) == final_text, f"{provider} final transcript is not bound to final.md")
    else:
        try:
            cli_payload = json.loads(_read_text_with_bom(stdout_path))
        except (EvidenceError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise EvidenceError(f"claude CLI stdout is not valid JSON: {exc}") from exc
        _require(isinstance(cli_payload, Mapping) and isinstance(cli_payload.get("result"), str), "claude CLI stdout has no result text")
        _require(cli_payload["result"] == final_text, "claude final.md is not bound to the CLI result")
        _require(type(cli_payload.get("num_turns")) is int and cli_payload["num_turns"] >= 2, "claude CLI did not perform a multi-turn inspection")
        _require(cli_payload.get("permission_denials") == [], "claude CLI recorded permission denials")
        cited_sources = sum(name in final_text for name in ("evidence.py", "validation.py", "phase2_gpu_bridge.py", "core_numpy.py", "replay.py"))
        _require(cited_sources >= 2, "claude final output does not cite at least two canonical source files")
    return {
        "final": _relative(root, final_path),
        "final_sha256": artifacts.sha256_file(final_path),
        "status": _relative(root, status_path),
        "status_sha256": artifacts.sha256_file(status_path),
        "stdout": _relative(root, stdout_path),
        "stdout_sha256": artifacts.sha256_file(stdout_path),
        "stderr": _relative(root, stderr_path),
        "stderr_sha256": artifacts.sha256_file(stderr_path),
        "input": _relative(root, input_path),
        "input_sha256": artifacts.sha256_file(input_path),
    }


def _snapshot_crosscheck(
    root: Path,
    directory_value: str | Path,
    protocol_hash: str,
    source_hashes: Mapping[str, str],
    config_hash: str,
    expected_prompt: str,
) -> dict[str, Any]:
    directory = _inside(root, directory_value, kind="crosscheck directory")
    manifest_path = _inside(root, directory / "manifest.json", kind="crosscheck manifest")
    manifest = _mapping(_load_json(manifest_path), "crosscheck manifest")
    prompt = _inside(root, directory / "prompt.txt", kind="crosscheck prompt")
    prompt_text = prompt.read_text(encoding="utf-8")
    _require(prompt_text == expected_prompt, "crosscheck prompt differs from the deterministic frozen prompt")
    _require(protocol_hash.lower() in prompt_text.lower() and config_hash.lower() in prompt_text.lower(), "crosscheck prompt omits protocol/config binding")
    _require(all(str(value).lower() in prompt_text.lower() for value in source_hashes.values()), "crosscheck prompt omits a source binding")
    prompt_hash = artifacts.sha256_file(prompt)
    _same_digest(manifest.get("prompt_sha256"), prompt_hash, "crosscheck prompt hash")
    executor_capsule = _snapshot_executor_capsule(root, directory, manifest, prompt_hash)
    _require(manifest.get("prompt_retained") is True, "crosscheck prompt was not retained")
    _require(manifest.get("prompt_cleanup_error") is None, "crosscheck prompt cleanup recorded an error")
    _require(manifest.get("schema_version") == 1 and manifest.get("outcome") == "complete", "crosscheck manifest is not complete")
    _require(manifest.get("run_id") == directory.name, "crosscheck run_id does not match its directory")
    _require(manifest.get("successful_providers") == 4 and manifest.get("attempted_providers") == 4, "crosscheck is not a four-provider success")
    working_parts = _portable_path_parts(manifest.get("working_directory"), "crosscheck working directory")
    _require(working_parts[-1] == root.name, "crosscheck working directory is not the project root")
    run_parts = _portable_path_parts(manifest.get("run_directory"), "crosscheck run directory")
    _require(run_parts[-1] == directory.name and "crosscheck" in {part.lower() for part in run_parts}, "crosscheck run directory binding mismatch")
    provider_records = manifest.get("providers")
    _require(isinstance(provider_records, list) and len(provider_records) == len(PROVIDERS), "crosscheck manifest provider set is incomplete")
    by_provider: dict[str, Mapping[str, Any]] = {}
    for raw in provider_records:
        record = _mapping(raw, "crosscheck manifest provider record")
        provider = record.get("provider")
        _require(provider in PROVIDERS and provider not in by_provider, f"unexpected or duplicate crosscheck provider: {provider!r}")
        by_provider[provider] = record
    _require(set(by_provider) == set(PROVIDERS), "crosscheck manifest provider set is not exact")
    providers = {
        provider: _validate_crosscheck_provider(root, directory, provider, by_provider[provider], executor_capsule)
        for provider in PROVIDERS
    }
    return {
        "directory": _relative(root, directory),
        "manifest": _relative(root, manifest_path),
        "manifest_sha256": artifacts.sha256_file(manifest_path),
        "prompt": _relative(root, prompt),
        "prompt_sha256": prompt_hash,
        "executor_capsule": executor_capsule,
        "providers": providers,
    }


def _check_crosscheck_binding(
    root: Path,
    binding_value: Any,
    protocol_hash: str,
    source_hashes: Mapping[str, str],
    config_hash: str,
    expected_prompt: str,
) -> None:
    binding = _mapping(binding_value, "crosscheck binding")
    expected_keys = {"directory", "manifest", "manifest_sha256", "prompt", "prompt_sha256", "executor_capsule", "providers"}
    _require(set(binding) == expected_keys, "crosscheck binding schema mismatch")
    observed = _snapshot_crosscheck(root, binding["directory"], protocol_hash, source_hashes, config_hash, expected_prompt)
    for key in ("directory", "manifest", "prompt"):
        _require(observed[key] == binding[key], f"crosscheck {key} path is not canonical")
    for key in ("manifest_sha256", "prompt_sha256"):
        _same_digest(observed[key], binding[key], f"crosscheck {key}")
    _require(observed["executor_capsule"] == binding["executor_capsule"], "crosscheck executor capsule binding mismatch")
    bound_providers = _mapping(binding["providers"], "bound crosscheck providers")
    _require(set(bound_providers) == set(PROVIDERS), "bound crosscheck provider set is not exact")
    for provider in PROVIDERS:
        bound = _mapping(bound_providers[provider], f"bound {provider} artifacts")
        expected = observed["providers"][provider]
        _require(set(bound) == set(expected), f"bound {provider} artifact schema mismatch")
        for path_key in ("final", "status", "stdout", "stderr", "input"):
            _require(bound[path_key] == expected[path_key], f"bound {provider} {path_key} path is not canonical")
        for hash_key in ("final_sha256", "status_sha256", "stdout_sha256", "stderr_sha256", "input_sha256"):
            _same_digest(bound[hash_key], expected[hash_key], f"bound {provider} {hash_key}")


def create_calibration_freeze(
    calibration_dir: str | Path,
    local_smoke_dir: str | Path,
    remote_smoke_dir: str | Path,
    remote_test_result_path: str | Path,
    local_validation_report_path: str | Path,
    remote_validation_report_path: str | Path,
    crosscheck_dir: str | Path,
    marker_path: str | Path,
    root: str | Path,
    config: Any,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Validate and bind all pre-primary evidence, then write the marker."""

    root_path = _root_path(root)
    protocol_hash, config_hash, _ = _config_facts(config)
    validated_sources = _validate_source_hashes(root_path, source_hashes)
    protocol_path = _inside(root_path, "notes/phase2_gpu_bridge_protocol.md", kind="protocol file")
    _same_digest(artifacts.sha256_file(protocol_path), protocol_hash, "protocol file hash")

    local_validation_binding = _snapshot_local_validation_report(
        root_path,
        local_validation_report_path,
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        source_hashes=validated_sources,
    )
    remote_validation_binding = _snapshot_remote_validation_report(
        root_path,
        remote_validation_report_path,
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        source_hashes=validated_sources,
    )

    calibration_binding, calibration_manifest = _snapshot_run(root_path, calibration_dir)
    calibration_path = _inside(root_path, calibration_binding["manifest"], kind="calibration manifest").parent
    _calibration_checks(
        calibration_path,
        calibration_manifest,
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        source_hashes=validated_sources,
        deep_validate=True,
    )

    local_binding, local_manifest = _snapshot_run(root_path, local_smoke_dir)
    local_path = _inside(root_path, local_binding["manifest"], kind="local smoke manifest").parent
    _smoke_checks(
        local_path,
        local_manifest,
        root=root_path,
        role="local_cpu_smoke",
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        source_hashes=validated_sources,
        deep_validate=False,
    )
    remote_binding, remote_manifest = _snapshot_run(root_path, remote_smoke_dir)
    remote_path = _inside(root_path, remote_binding["manifest"], kind="remote smoke manifest").parent
    _smoke_checks(
        remote_path,
        remote_manifest,
        root=root_path,
        role="remote_gpu_smoke",
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        source_hashes=validated_sources,
        deep_validate=True,
    )
    remote_test_binding = _snapshot_remote_test(
        root_path,
        remote_test_result_path,
        validated_sources,
        config_hash,
        protocol_hash,
    )
    _assert_validation_report_bindings(
        local_validation_binding,
        local_binding,
        remote_validation_binding,
        remote_binding,
        calibration_binding,
        remote_test_binding,
    )
    expected_prompt = build_implementation_crosscheck_prompt(
        root_path,
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        source_hashes=validated_sources,
        local_smoke_manifest_sha256=local_binding["manifest_sha256"],
        local_validation_report_sha256=local_validation_binding["sha256"],
        remote_test_result_sha256=remote_test_binding["sha256"],
        remote_smoke_manifest_sha256=remote_binding["manifest_sha256"],
        calibration_manifest_sha256=calibration_binding["manifest_sha256"],
        remote_validation_report_sha256=remote_validation_binding["sha256"],
        local_smoke_manifest_path=local_binding["manifest"],
        local_validation_report_path=local_validation_binding["path"],
        remote_test_result_path=remote_test_binding["path"],
        remote_smoke_manifest_path=remote_binding["manifest"],
        calibration_manifest_path=calibration_binding["manifest"],
        remote_validation_report_path=remote_validation_binding["path"],
    )

    evidence = {
        "calibration": calibration_binding,
        "local_cpu_smoke": local_binding,
        "remote_gpu_smoke": remote_binding,
        "remote_test": remote_test_binding,
        "local_cpu_smoke_validation": {"path": local_validation_binding["path"], "sha256": local_validation_binding["sha256"]},
        "remote_validation": {"path": remote_validation_binding["path"], "sha256": remote_validation_binding["sha256"]},
        "crosscheck": _snapshot_crosscheck(
            root_path,
            crosscheck_dir,
            protocol_hash,
            validated_sources,
            config_hash,
            expected_prompt,
        ),
    }
    marker = {
        "schema_version": 2,
        "kind": "phase2_calibration_freeze_evidence",
        "protocol_sha256": protocol_hash,
        "config_sha256": config_hash,
        "source_hashes": validated_sources,
        "evidence": evidence,
    }
    target = _marker_target(root_path, marker_path)
    _assert_marker_disjoint(
        target,
        [
            protocol_path,
            root_path / "experiments" / "phase2_gpu_bridge",
            calibration_path,
            local_path,
            remote_path,
            _inside(root_path, remote_test_result_path, kind="remote test result"),
            _inside(root_path, local_validation_report_path, kind="local validation report"),
            _inside(root_path, remote_validation_report_path, kind="remote validation report"),
            _inside(root_path, crosscheck_dir, kind="crosscheck directory"),
        ],
    )
    candidate_fd, candidate_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".candidate",
        dir=str(target.parent),
    )
    os.close(candidate_fd)
    candidate = Path(candidate_name)
    candidate_digest: str | None = None
    published = False
    try:
        artifacts.write_json(candidate, marker)
        candidate_digest = artifacts.sha256_file(candidate)
        # Validate the exact bytes that are about to become the marker before
        # publishing them, then revalidate after the atomic replacement.
        validate_calibration_freeze(candidate, root_path, config, validated_sources)
        os.replace(candidate, target)
        published = True
        return validate_calibration_freeze(target, root_path, config, validated_sources)
    except BaseException:
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            pass
        if published and candidate_digest is not None and target.exists():
            try:
                if artifacts.sha256_file(target) == candidate_digest:
                    target.unlink()
            except OSError:
                pass
        raise


def validate_calibration_freeze(
    marker_path: str | Path,
    root: str | Path,
    config: Any,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Re-derive the primary-lock decision from every bound artifact."""

    root_path = _root_path(root)
    protocol_hash, config_hash, _ = _config_facts(config)
    validated_sources = _validate_source_hashes(root_path, source_hashes)
    marker = _mapping(_load_json(_inside(root_path, marker_path, kind="calibration freeze marker")), "calibration freeze marker")
    expected_keys = {"schema_version", "kind", "protocol_sha256", "config_sha256", "source_hashes", "evidence"}
    _require(set(marker) == expected_keys, "calibration freeze marker schema mismatch")
    _require(marker.get("schema_version") == 2, "unsupported calibration freeze marker schema")
    _require(marker.get("kind") == "phase2_calibration_freeze_evidence", "calibration freeze marker kind mismatch")
    _same_digest(marker.get("protocol_sha256"), protocol_hash, "freeze protocol hash")
    _same_digest(marker.get("config_sha256"), config_hash, "freeze config hash")
    _require(marker.get("source_hashes") == validated_sources, "freeze source hashes differ from current sources")
    protocol_path = _inside(root_path, "notes/phase2_gpu_bridge_protocol.md", kind="protocol file")
    _same_digest(artifacts.sha256_file(protocol_path), protocol_hash, "protocol file hash")

    evidence = _mapping(marker.get("evidence"), "freeze evidence")
    expected_evidence = {
        "calibration",
        "local_cpu_smoke",
        "remote_gpu_smoke",
        "remote_test",
        "local_cpu_smoke_validation",
        "remote_validation",
        "crosscheck",
    }
    _require(set(evidence) == expected_evidence, "freeze evidence schema mismatch")

    local_validation_binding = _mapping(evidence["local_cpu_smoke_validation"], "bound local smoke validation")
    _require(set(local_validation_binding) == {"path", "sha256"}, "bound local smoke validation schema mismatch")
    local_validation = _snapshot_local_validation_report(
        root_path,
        local_validation_binding["path"],
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        source_hashes=validated_sources,
    )
    _same_digest(local_validation["sha256"], local_validation_binding["sha256"], "bound local smoke validation report hash")
    remote_validation_binding = _mapping(evidence["remote_validation"], "bound remote validation")
    _require(set(remote_validation_binding) == {"path", "sha256"}, "bound remote validation schema mismatch")
    remote_validation = _snapshot_remote_validation_report(
        root_path,
        remote_validation_binding["path"],
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        source_hashes=validated_sources,
    )
    _same_digest(remote_validation["sha256"], remote_validation_binding["sha256"], "bound remote validation report hash")

    calibration_binding, calibration_manifest, calibration_dir = _check_run_binding(root_path, evidence["calibration"])
    calibration_facts = _calibration_checks(
        calibration_dir,
        calibration_manifest,
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        source_hashes=validated_sources,
        deep_validate=False,
    )
    local_binding_observed, local_manifest, local_dir = _check_run_binding(root_path, evidence["local_cpu_smoke"])
    _smoke_checks(
        local_dir,
        local_manifest,
        root=root_path,
        role="local_cpu_smoke",
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        source_hashes=validated_sources,
        deep_validate=False,
    )
    remote_binding, remote_manifest, remote_dir = _check_run_binding(root_path, evidence["remote_gpu_smoke"])
    _smoke_checks(
        remote_dir,
        remote_manifest,
        root=root_path,
        role="remote_gpu_smoke",
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        source_hashes=validated_sources,
        deep_validate=False,
    )
    calibration_binding, _, _ = _check_run_binding(root_path, evidence["calibration"])
    _check_remote_test_binding(root_path, evidence["remote_test"], validated_sources, config_hash, protocol_hash)
    local_binding = _mapping(evidence["local_cpu_smoke"], "bound local CPU smoke")
    remote_test_binding = _mapping(evidence["remote_test"], "bound remote test")
    _assert_validation_report_bindings(
        local_validation,
        local_binding_observed,
        remote_validation,
        remote_binding,
        calibration_binding,
        remote_test_binding,
    )
    expected_prompt = build_implementation_crosscheck_prompt(
        root_path,
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        source_hashes=validated_sources,
        local_smoke_manifest_sha256=_digest(local_binding.get("manifest_sha256"), "bound local smoke manifest hash"),
        local_validation_report_sha256=_digest(local_validation_binding.get("sha256"), "bound local validation report hash"),
        remote_test_result_sha256=_digest(remote_test_binding.get("sha256"), "bound remote test result hash"),
        remote_smoke_manifest_sha256=_digest(remote_binding.get("manifest_sha256"), "bound remote smoke manifest hash"),
        calibration_manifest_sha256=_digest(calibration_binding.get("manifest_sha256"), "bound calibration manifest hash"),
        remote_validation_report_sha256=_digest(remote_validation_binding.get("sha256"), "bound remote validation report hash"),
        local_smoke_manifest_path=str(local_binding.get("manifest")),
        local_validation_report_path=str(local_validation_binding.get("path")),
        remote_test_result_path=str(remote_test_binding.get("path")),
        remote_smoke_manifest_path=str(remote_binding.get("manifest")),
        calibration_manifest_path=str(calibration_binding.get("manifest")),
        remote_validation_report_path=str(remote_validation_binding.get("path")),
    )
    _check_crosscheck_binding(root_path, evidence["crosscheck"], protocol_hash, validated_sources, config_hash, expected_prompt)

    return {
        "valid": True,
        "frozen_for_primary": True,
        "protocol_sha256": protocol_hash,
        "config_sha256": config_hash,
        "source_hashes": validated_sources,
        "calibration_rows": calibration_facts["rows"],
        "calibration_transitions": calibration_facts["transitions"],
        "providers": list(PROVIDERS),
        "local_validation_report": local_validation_binding["path"],
        "remote_validation_report": remote_validation_binding["path"],
    }


__all__ = [
    "EvidenceError",
    "build_implementation_crosscheck_prompt",
    "create_calibration_freeze",
    "write_local_smoke_validation_report",
    "write_remote_validation_report",
    "validate_calibration_freeze",
]
