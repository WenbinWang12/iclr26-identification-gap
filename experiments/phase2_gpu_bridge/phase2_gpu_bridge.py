"""CLI and orchestration for the planted Phase-2 GPU bridge."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Keep the launcher-level controls before importing NumPy/core modules.
for _key, _value in {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}.items():
    os.environ[_key] = _value

import numpy as np

from . import artifacts
from .evidence import EvidenceError, create_calibration_freeze, validate_calibration_freeze
from .core_numpy import (
    ARMS,
    RANKS,
    REGIMES,
    BridgeConfig,
    assert_protocol_hash,
    bootstrap_summary,
    build_prefix_oracle,
    build_seed_dataset,
    environment_payload,
    make_factor_initializations,
    make_base_draws,
    make_seed_streams,
    mode_count,
    mode_schedule,
    one_sided_lower_quantile,
    population_risk,
    protocol_hash,
    q_min_keep,
    run_all_sanity,
    sanity_contraction_bound,
    weighted_modal_svd,
)
from .replay import B_MAX, REPLAY_BYTES, ClockBalancedReplay, ReservoirReplay
from .torch_runner import import_torch, make_model, make_optimizer, model_delta64, require_cuda, train_window
from .validation import RESOURCE_REQUIRED_FIELDS, load_json_strict, reconcile_runs


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "notes" / "phase2_gpu_bridge_protocol.md"
SOURCE_PATHS = (
    Path(__file__).resolve(),
    ROOT / "experiments" / "phase2_gpu_bridge" / "core_numpy.py",
    ROOT / "experiments" / "phase2_gpu_bridge" / "replay.py",
    ROOT / "experiments" / "phase2_gpu_bridge" / "torch_runner.py",
    ROOT / "experiments" / "phase2_gpu_bridge" / "artifacts.py",
    ROOT / "experiments" / "phase2_gpu_bridge" / "test_phase2_gpu_bridge.py",
    ROOT / "experiments" / "phase2_gpu_bridge" / "evidence.py",
    ROOT / "experiments" / "phase2_gpu_bridge" / "validation.py",
    ROOT / "experiments" / "phase2_gpu_bridge" / "test_validation.py",
    ROOT / "experiments" / "phase2_gpu_bridge" / "__init__.py",
)
_ACTIVE_RUN_STATE: dict[str, Any] | None = None


def _set_run_progress(**values: Any) -> None:
    if _ACTIVE_RUN_STATE is not None:
        _ACTIVE_RUN_STATE.setdefault("progress", {}).update(values)


def _parse_seeds(value: str | None, default: Sequence[int]) -> tuple[int, ...]:
    if not value:
        return tuple(int(seed) for seed in default)
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return seeds


def _json_dump(path: Path, payload: Any) -> None:
    artifacts.write_json(path, payload)


def _source_hashes() -> dict[str, str]:
    missing = [str(path) for path in SOURCE_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required Phase-2 source files are missing: {missing}")
    if len({path.name for path in SOURCE_PATHS}) != len(SOURCE_PATHS):
        raise RuntimeError("Phase-2 source snapshot contains duplicate basenames")
    return {path.name: artifacts.sha256_file(path) for path in SOURCE_PATHS}


def _verified_manifest_outputs(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    directory = manifest_path.parent
    manifest = load_json_strict(manifest_path)
    if not isinstance(manifest, dict):
        raise RuntimeError("run manifest must be a strict JSON object")
    declared = manifest.get("output_hashes")
    if not isinstance(declared, dict) or not declared:
        raise RuntimeError("run manifest output_hashes must be a non-empty object")
    verified_names: set[str] = set()
    for name, expected in declared.items():
        if not isinstance(name, str) or Path(name).name != name or name == "manifest.json":
            raise RuntimeError(f"unsafe run output name: {name!r}")
        if not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError(f"invalid run output hash for {name}")
        path = (directory / name).resolve(strict=True)
        if path.parent != directory or not path.is_file():
            raise RuntimeError(f"run output is missing or escapes its directory: {name}")
        if artifacts.sha256_file(path).lower() != expected.lower():
            raise RuntimeError(f"run output hash mismatch: {name}")
        if path.suffix.lower() == ".json":
            load_json_strict(path)
        verified_names.add(name)
    direct_names = {path.name for path in directory.iterdir() if path.is_file() and path.name != "manifest.json"}
    if verified_names != direct_names:
        raise RuntimeError("run manifest does not bind every direct output file exactly")
    return manifest


def _config_payload(config: BridgeConfig, seeds: Sequence[int], device: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "phase2_frozen_experiment_config",
        "protocol_sha256": config.protocol_sha256,
        "protocol_path": "notes/phase2_gpu_bridge_protocol.md",
        "seeds": [int(seed) for seed in seeds],
        "device": device,
        "frozen": config.payload(),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    artifacts.write_csv(path, rows, fields)


def _gates_for_numpy(
    base: Any,
    config: BridgeConfig,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    source_snapshot = dict(source_hashes) if source_hashes is not None else _source_hashes()
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
                weighted_delta, weighted_selected, _ = weighted_modal_svd(
                    U, V, oracle.alpha, oracle.q, rank
                )
                direct_weighted_loss = population_risk(weighted_delta, U, V, oracle.alpha, oracle.q)
                delta = max(abs(float(direct_weighted_loss - oracle.loss_rank)), abs(float(len(weighted_selected) - len(oracle.selected))))
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
        row["capacity_ratio"] <= 1e-10 + 1e-15
        for row in oracle_rows
        if row["regime"] == "compatible"
    )
    passed_bound = sanity_contraction_bound(q_min) < 1.2e-19
    passed_identity = config.protocol_sha256 == protocol_hash(PROTOCOL_PATH)
    passed_g1 = bool(
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
        "G0_numpy_identity": {
            "protocol_hash": config.protocol_sha256,
            "passed": passed_identity,
            "source_hashes": source_snapshot,
        },
        "G1_oracle_capacity": {
            "passed": passed_g1,
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
        },
    }


def run_numpy_smoke(output_dir: Path, run_role: str = "local_cpu_smoke") -> dict[str, Any]:
    config = BridgeConfig()
    if run_role != "local_cpu_smoke":
        raise ValueError("NumPy smoke is provenance-only and must use run_role=local_cpu_smoke")
    assert_protocol_hash(PROTOCOL_PATH, config.protocol_sha256)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_source_hashes = _source_hashes()
    started = time.time()
    base = make_base_draws(config)
    dataset = build_seed_dataset(base, config.calibration_seeds[0], config)
    gates = _gates_for_numpy(base, config, run_source_hashes)
    if _source_hashes() != run_source_hashes:
        raise RuntimeError("Phase-2 source files changed while the NumPy smoke was running")
    config_payload = _config_payload(config, config.calibration_seeds[:1], "cpu")
    split_manifest = {
        "seed": dataset.seed,
        "data_sha256": dataset.data_sha256,
        "split_sha256": dataset.split_sha256,
        "cells": {
            f"{regime}:{rank}": {
                "windows": len(windows),
                "train_counts": [window.train.count for window in windows],
                "eval_counts": [window.eval.count for window in windows],
                "pi_hashes": [artifacts.hash_arrays((("pi_t", window.pi_t),)) for window in windows],
            }
            for (regime, rank), windows in dataset.cells.items()
        },
    }
    _json_dump(output_dir / "config.json", config_payload)
    _json_dump(output_dir / "split_manifest.json", split_manifest)
    _json_dump(output_dir / "gates.json", gates)
    _json_dump(
        output_dir / "traces.json",
        {
            "protocol_sha256": config.protocol_sha256,
            "source_hashes": run_source_hashes,
            "base_sha256": base.sha256(),
            "data_sha256": dataset.data_sha256,
            "split_sha256": dataset.split_sha256,
        },
    )
    summary = {
        "schema_version": 1,
        "stage": "phase2_numpy_smoke",
        "run_role": run_role,
        "claim_boundary": "provenance_only_local_cpu_smoke",
        "all_required_pass": bool(gates["G0_numpy_identity"]["passed"] and gates["G1_oracle_capacity"]["passed"] and gates["G1_oracle_capacity"]["passed_support"] and gates["G1_oracle_capacity"]["passed_error"]),
        "elapsed_seconds": time.time() - started,
        "gates": gates,
    }
    _json_dump(output_dir / "summary.json", summary)
    manifest = {
        "schema_version": 1,
        "stage": "phase2_numpy_smoke",
        "run_role": run_role,
        "run_id": output_dir.name,
        "ordered_seeds": [int(config.calibration_seeds[0])],
        "protocol_sha256": config.protocol_sha256,
        "source_hashes": run_source_hashes,
        "test_hash": run_source_hashes["test_phase2_gpu_bridge.py"],
        "frozen_config_sha256": config.sha256(),
        "config_sha256": artifacts.sha256_file(output_dir / "config.json"),
        "split_manifest_sha256": artifacts.sha256_file(output_dir / "split_manifest.json"),
        "gates_sha256": artifacts.sha256_file(output_dir / "gates.json"),
        "traces_sha256": artifacts.sha256_file(output_dir / "traces.json"),
        "data_sha256": dataset.data_sha256,
        "split_sha256": dataset.split_sha256,
        "base_sha256": base.sha256(),
        "runtime": environment_payload(),
        "resource_ledger": {"gpu_used": False, "replay_bytes": 93488},
        "output_hashes": {path.name: artifacts.sha256_file(path) for path in output_dir.iterdir() if path.name != "manifest.json"},
        "claim_boundary": "provenance_only_local_cpu_smoke",
    }
    _json_dump(output_dir / "manifest.json", manifest)
    return summary


def _evaluate_checkpoint(base: Any, regime: str, rank: int, window: int, delta: np.ndarray, arm: str, seed: int, updates: int, replay_count: int) -> dict[str, Any]:
    oracle = build_prefix_oracle(base, regime, rank, window)
    prefix_loss = population_risk(delta, *base.basis(regime, rank), oracle.alpha, oracle.q)
    current_loss = population_risk(delta, *base.basis(regime, rank), oracle.alpha, oracle.p)
    gap = prefix_loss - oracle.loss_rank
    return {
        "seed": int(seed),
        "regime": regime,
        "R": int(rank),
        "window": int(window),
        "arm": arm,
        "requested_rank": int(rank),
        "numerical_rank": int(np.linalg.matrix_rank(delta, tol=1e-10)),
        "L_prefix": prefix_loss,
        "L_current": current_loss,
        "L_rank": oracle.loss_rank,
        "L_inf": oracle.loss_inf,
        "D_t": oracle.D_t,
        "D_current": oracle.D_current,
        "capacity_ratio": oracle.capacity_tail / oracle.D_t,
        "online_ratio": gap / oracle.D_t,
        "current_acquisition_ratio": current_loss / oracle.D_current,
        "gap": gap,
        "bounded_history_reduction": None,
        "selection_gain": None,
        "gap_fraction": None,
        "validity": "valid",
        "updates": int(updates),
        "forward": int(updates),
        "backward": int(updates),
        "current_examples": int(updates * (64 if replay_count else 128)),
        "replay_examples": int(replay_count),
    }


TRAINABLE_ARMS = (
    "sequential_dense_rank_R",
    "clock_balanced_replay_rank_R",
    "reservoir_random_replay_rank_R",
)


def _record_block_bytes(block: Any) -> int:
    return int(
        sum(
            int(getattr(block, name).nbytes)
            for name in (
                "x64",
                "y64",
                "x32",
                "y32",
                "mode_ids",
                "signs",
                "record_ids",
                "global_keys",
                "storage_permutation",
            )
        )
    )


def _cell_dataset_bytes(cell: Sequence[Any]) -> int:
    return int(
        sum(
            _record_block_bytes(window.train)
            + _record_block_bytes(window.eval)
            + int(window.pi_t.nbytes)
            for window in cell
        )
    )


def _expected_optimizer_contract() -> dict[str, Any]:
    return {
        "class": "AdamW",
        "parameter_order": ["B", "A"],
        "lr": 3e-2,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "foreach": False,
        "fused": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "scheduler": None,
    }


def _optimizer_resource_payload(torch: Any, model: Any, optimizer: Any) -> dict[str, Any]:
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("the frozen optimizer must contain exactly one parameter group")
    if list(optimizer.param_groups[0]["params"]) != [model.B, model.A]:
        raise RuntimeError("the frozen optimizer parameter order must be exactly [B, A]")
    parameter_names = {id(model.B): "B", id(model.A): "A"}
    parameter_rows = []
    for parameter in optimizer.param_groups[0]["params"]:
        parameter_rows.append(
            {
                "name": parameter_names.get(id(parameter), "unknown"),
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "numel": int(parameter.numel()),
                "bytes": int(parameter.numel()) * int(parameter.element_size()),
            }
        )
    state_rows = []
    optimizer_state_bytes = 0
    optimizer_steps: list[int] = []
    for parameter in optimizer.param_groups[0]["params"]:
        state = optimizer.state.get(parameter, {})
        for name, value in sorted(state.items()):
            if torch.is_tensor(value):
                size = int(value.numel()) * int(value.element_size())
                optimizer_state_bytes += size
                state_rows.append(
                    {
                        "parameter": parameter_names.get(id(parameter), "unknown"),
                        "state": str(name),
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "bytes": size,
                    }
                )
                if name == "step":
                    optimizer_steps.append(int(value.detach().cpu().item()))
    trainable_parameter_count = int(sum(row["numel"] for row in parameter_rows))
    trainable_parameter_bytes = int(sum(row["bytes"] for row in parameter_rows))
    optimizer_contract = {
        "class": type(optimizer).__name__,
        "parameter_order": [row["name"] for row in parameter_rows],
        "lr": float(optimizer.param_groups[0]["lr"]),
        "betas": list(optimizer.param_groups[0]["betas"]),
        "eps": float(optimizer.param_groups[0]["eps"]),
        "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
        "amsgrad": bool(optimizer.param_groups[0]["amsgrad"]),
        "foreach": optimizer.param_groups[0].get("foreach"),
        "fused": optimizer.param_groups[0].get("fused"),
        "maximize": bool(optimizer.param_groups[0].get("maximize", False)),
        "capturable": bool(optimizer.param_groups[0].get("capturable", False)),
        "differentiable": bool(optimizer.param_groups[0].get("differentiable", False)),
        "scheduler": None,
    }
    return {
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_parameter_bytes": trainable_parameter_bytes,
        "parameter_breakdown": parameter_rows,
        "optimizer_state_bytes": int(optimizer_state_bytes),
        "optimizer_state_breakdown": state_rows,
        "optimizer_steps": optimizer_steps,
        "optimizer_contract": optimizer_contract,
        "optimizer_contract_sha256": artifacts.hash_json(optimizer_contract),
    }


def _measure_inference_latency(
    torch: Any,
    model: Any,
    x32: np.ndarray,
    device: Any,
    *,
    warmup: int = 5,
    repetitions: int = 20,
) -> dict[str, Any]:
    batch = torch.as_tensor(np.ascontiguousarray(x32[:128]), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(batch)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(repetitions):
            model(batch)
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "batch_size": 128,
        "warmup_forwards": int(warmup),
        "measured_forwards": int(repetitions),
        "total_seconds": float(elapsed),
        "mean_batch_seconds": float(elapsed / repetitions),
        "mean_example_seconds": float(elapsed / (repetitions * 128)),
    }


def _warm_cuda_backend(torch: Any, device: Any) -> dict[str, int]:
    """Initialize backend workspaces outside all measured learner arms."""

    # Use deterministic zeros and a max-rank dummy AdamW step.  This consumes
    # no experiment RNG and is excluded from all learner counters.
    w0 = torch.zeros((32, 32), dtype=torch.float32, device=device)
    a_parameter = torch.nn.Parameter(torch.zeros((8, 32), dtype=torch.float32, device=device))
    b_parameter = torch.nn.Parameter(torch.zeros((32, 8), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        [b_parameter, a_parameter],
        lr=3e-2,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        foreach=False,
        fused=False,
        maximize=False,
        capturable=False,
        differentiable=False,
    )
    x = torch.zeros((128, 32), dtype=torch.float32, device=device)
    y = torch.zeros((128, 32), dtype=torch.float32, device=device)
    optimizer.zero_grad(set_to_none=True)
    prediction = x @ w0.transpose(0, 1) + (x @ a_parameter.transpose(0, 1)) @ b_parameter.transpose(0, 1)
    loss = (prediction - y).square().sum() / (2.0 * 128.0)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    del optimizer, prediction, loss, x, y, w0, a_parameter, b_parameter
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _split_manifest_entry(dataset: Any, base: Any, config: BridgeConfig) -> dict[str, Any]:
    """Serialize split counts and hashes without exposing records to training."""

    w0_32 = np.asarray(base.W0, dtype=np.float32, order="C")

    def cast_roundoff(block: Any) -> dict[str, float]:
        deployed_base = np.asarray(block.x32 @ w0_32.T, dtype=np.float32)
        exact_base_32 = np.asarray(block.x64 @ base.W0.T, dtype=np.float32)
        base_residual = deployed_base.astype(np.float64) - exact_base_32.astype(np.float64)
        signal_32 = np.asarray(block.y64 - block.x64 @ base.W0.T, dtype=np.float32)
        separate_cast = exact_base_32.astype(np.float64) + signal_32.astype(np.float64)
        target_residual = block.y32.astype(np.float64) - separate_cast
        return {
            "base_matrix_cast_loss": float(0.5 * np.mean(np.sum(base_residual * base_residual, axis=1))),
            "target_nonadditivity_loss": float(0.5 * np.mean(np.sum(target_residual * target_residual, axis=1))),
        }

    def block_integrity(block: Any, regime: str, rank: int, window: int, split: str) -> dict[str, Any]:
        count = mode_count(regime, rank)
        current, previous = mode_schedule(regime, rank, window)
        total = config.n_train if split == "train" else config.n_eval
        expected = np.zeros(count, dtype=np.int64)
        if window == 1:
            expected[current] = total
        else:
            expected[current] = 3 * total // 4
            expected[int(previous)] = total // 4
        observed = np.bincount(block.mode_ids, minlength=count)
        sign_counts = []
        for mode, requested in enumerate(expected.tolist()):
            mode_signs = block.signs[block.mode_ids == mode]
            positive = int(np.count_nonzero(mode_signs == 1))
            negative = int(np.count_nonzero(mode_signs == -1))
            sign_counts.append(
                {
                    "mode": mode,
                    "requested": int(requested),
                    "positive": positive,
                    "negative": negative,
                    "balanced": positive == negative == int(requested) // 2,
                }
            )
        regime_code = 0 if regime == "compatible" else 1
        rank_code = {2: 0, 4: 1, 8: 2}[int(rank)]
        split_code = 0 if split == "train" else 1
        expected_keys = np.column_stack(
            (
                np.full(total, regime_code, dtype=np.int64),
                np.full(total, rank_code, dtype=np.int64),
                np.full(total, window, dtype=np.int64),
                np.full(total, split_code, dtype=np.int64),
                np.arange(total, dtype=np.int64),
            )
        )
        permutation = np.asarray(block.storage_permutation, dtype=np.int64)
        return {
            "expected_mode_counts": expected.tolist(),
            "observed_mode_counts": observed.tolist(),
            "mode_counts_match": bool(np.array_equal(observed, expected)),
            "sign_counts": sign_counts,
            "balanced_signs_passed": bool(all(item["balanced"] for item in sign_counts)),
            "record_ids_passed": bool(np.array_equal(block.record_ids, np.arange(total, dtype=np.int64))),
            "global_key_schema_passed": bool(np.array_equal(block.global_keys, expected_keys)),
            "storage_permutation_passed": bool(
                permutation.shape == (total,)
                and np.array_equal(np.sort(permutation), np.arange(total, dtype=np.int64))
            ),
            "finite_arrays_passed": bool(
                all(np.all(np.isfinite(value)) for value in (block.x64, block.y64, block.x32, block.y32))
            ),
            "float32_casts_passed": bool(
                np.array_equal(block.x32, np.asarray(block.x64, dtype=np.float32))
                and np.array_equal(block.y32, np.asarray(block.y64, dtype=np.float32))
            ),
        }

    rebuilt = build_seed_dataset(base, int(dataset.seed), config)
    basis_errors = {
        name: float(np.linalg.norm(getattr(base, name).T @ getattr(base, name) - np.eye(getattr(base, name).shape[1])))
        for name in ("compatible_U", "compatible_V", "stress_U", "stress_V")
    }

    return {
        "seed": int(dataset.seed),
        "data_sha256": dataset.data_sha256,
        "split_sha256": dataset.split_sha256,
        "deterministic_rebuild_passed": bool(
            rebuilt.data_sha256 == dataset.data_sha256 and rebuilt.split_sha256 == dataset.split_sha256
        ),
        "base_finite_passed": bool(
            all(
                np.all(np.isfinite(getattr(base, name)))
                for name in ("W0", "compatible_U", "compatible_V", "stress_U", "stress_V")
            )
        ),
        "basis_orthogonality_errors": basis_errors,
        "basis_orthogonality_passed": bool(max(basis_errors.values()) <= 1e-10),
        "cells": {
            f"{regime}:{rank}": {
                "windows": len(windows),
                "train_counts": [window.train.count for window in windows],
                "eval_counts": [window.eval.count for window in windows],
                "pi_hashes": [
                    artifacts.hash_arrays((("pi_t", window.pi_t),)) for window in windows
                ],
                "mode_counts": [
                    {
                        "train": np.bincount(window.train.mode_ids, minlength=(rank if regime == "compatible" else 12)).tolist(),
                        "eval": np.bincount(window.eval.mode_ids, minlength=(rank if regime == "compatible" else 12)).tolist(),
                    }
                    for window in windows
                ],
                "integrity": [
                    {
                        "window": int(window.window),
                        "train": block_integrity(window.train, regime, rank, window.window, "train"),
                        "eval": block_integrity(window.eval, regime, rank, window.window, "eval"),
                        "pi_matches_train_storage_permutation": bool(
                            np.array_equal(window.pi_t, window.train.storage_permutation)
                        ),
                        "pi_is_full_permutation": bool(
                            np.asarray(window.pi_t).shape == (config.n_train,)
                            and np.array_equal(
                                np.sort(np.asarray(window.pi_t, dtype=np.int64)),
                                np.arange(config.n_train, dtype=np.int64),
                            )
                        ),
                    }
                    for window in windows
                ],
                "train_eval_global_key_overlap": [
                    int(
                        len(
                            set(map(tuple, window.train.global_keys.tolist())).intersection(
                                set(map(tuple, window.eval.global_keys.tolist()))
                            )
                        )
                    )
                    for window in windows
                ],
                "train_cast_roundoff": [cast_roundoff(window.train) for window in windows],
                "eval_cast_roundoff": [cast_roundoff(window.eval) for window in windows],
            }
            for (regime, rank), windows in dataset.cells.items()
        },
    }


def _derive_contrasts(per_window: list[dict[str, Any]]) -> None:
    """Fill paired contrast columns after all three arms have been evaluated."""

    grouped: dict[tuple[int, str, int, int], dict[str, dict[str, Any]]] = {}
    for row in per_window:
        key = (int(row["seed"]), str(row["regime"]), int(row["R"]), int(row["window"]))
        grouped.setdefault(key, {})[str(row["arm"])] = row
    for rows in grouped.values():
        missing = [arm for arm in TRAINABLE_ARMS if arm not in rows]
        if missing:
            raise AssertionError(f"missing paired arm rows: {missing}")
        seq = rows[TRAINABLE_ARMS[0]]
        clock = rows[TRAINABLE_ARMS[1]]
        reservoir = rows[TRAINABLE_ARMS[2]]
        denom = float(seq["D_t"])
        if denom < 1e-8:
            raise AssertionError("non-positive prefix denominator")
        bounded = (float(seq["gap"]) - float(clock["gap"])) / denom
        selection = (float(reservoir["gap"]) - float(clock["gap"])) / denom
        seq_gap = float(seq["gap"])
        fraction = None if seq_gap < 1e-8 else (float(seq["gap"]) - float(clock["gap"])) / seq_gap
        for row in rows.values():
            row["bounded_history_reduction"] = float(bounded)
            row["selection_gain"] = float(selection)
            row["gap_fraction"] = None if fraction is None else float(fraction)
            row["paired_gap_seq"] = seq_gap
            row["paired_gap_clock"] = float(clock["gap"])
            row["paired_gap_reservoir"] = float(reservoir["gap"])


def _post_run_gates(
    per_window: Sequence[Mapping[str, Any]],
    per_seed: Sequence[Mapping[str, Any]],
    config: BridgeConfig,
    run_role: str,
) -> dict[str, Any]:
    """Compute report-only G2/G3 summaries; calibration never gets primary thresholds."""

    target = [row for row in per_window if row["regime"] == "compatible" and int(row["R"]) == 4]
    acquisition = {
        arm: {
            "count": int(sum(float(row["current_acquisition_ratio"]) <= 0.05 for row in target if row["arm"] == arm)),
            "total": int(sum(row["arm"] == arm for row in target)),
            "threshold": 228,
            "status": "primary_only",
        }
        for arm in TRAINABLE_ARMS
    }
    finals = {
        int(row["seed"]): row
        for row in per_window
        if row["regime"] == "compatible" and int(row["R"]) == 4 and int(row["window"]) == config.T and row["arm"] == TRAINABLE_ARMS[0]
    }
    clock_by_seed = {
        int(row["seed"]): row
        for row in per_window
        if row["regime"] == "compatible" and int(row["R"]) == 4 and int(row["window"]) == config.T and row["arm"] == TRAINABLE_ARMS[1]
    }
    reservoir_by_seed = {
        int(row["seed"]): row
        for row in per_window
        if row["regime"] == "compatible" and int(row["R"]) == 4 and int(row["window"]) == config.T and row["arm"] == TRAINABLE_ARMS[2]
    }
    primary_result = run_role in {"primary", "confirmation"}
    ordered_seeds = [int(seed) for seed in config.primary_seeds] if primary_result else sorted(finals)
    final_seq = [finals[seed] for seed in ordered_seeds if seed in finals]
    final_clock = [clock_by_seed[seed] for seed in ordered_seeds if seed in clock_by_seed]
    final_reservoir = [reservoir_by_seed[seed] for seed in ordered_seeds if seed in reservoir_by_seed]
    expected_count = 20 if primary_result else len(ordered_seeds)
    complete = len(final_seq) == len(final_clock) == len(final_reservoir) == expected_count
    g2: dict[str, Any] = {
        "status": "evaluated" if primary_result else "not_evaluated",
        "acquisition": acquisition,
        "paired_final_rows_complete": complete,
    }
    g3: dict[str, Any] = {
        "status": "evaluated" if primary_result else "not_evaluated",
        "paired_final_rows_complete": complete,
    }
    if primary_result and complete:
        seq_ratios = [float(row["online_ratio"]) for row in final_seq]
        bounded = [float(row["bounded_history_reduction"]) for row in final_clock]
        selection = [float(row["selection_gain"]) for row in final_clock]
        seq_gaps = [float(row["gap"]) for row in final_seq]
        clock_gaps = [float(row["gap"]) for row in final_clock]
        reservoir_gaps = [float(row["gap"]) for row in final_reservoir]
        valid_fraction = [
            (seq_gap - clock_gap) / seq_gap
            for seq_gap, clock_gap in zip(seq_gaps, clock_gaps)
            if seq_gap >= 1e-8
        ]
        finite = all(
            np.isfinite(float(row[field]))
            for row in (*final_seq, *final_clock, *final_reservoir)
            for field in ("gap", "online_ratio", "current_acquisition_ratio", "D_t")
        )
        risk_failures = sum(gap < -1e-10 for gap in seq_gaps)
        bootstrap = bootstrap_summary(
            {
                "sequential_online_ratio_T": seq_ratios,
                "bounded_history_reduction_T": bounded,
                "selection_gain_T": selection,
            },
            valid_fraction if valid_fraction else None,
            config,
        )
        acquisition_ok = {
            arm: acquisition[arm]["count"] >= acquisition[arm]["threshold"]
            for arm in TRAINABLE_ARMS
        }
        g2.update(
            {
                "bootstrap": bootstrap,
                "sequential_online_ratio_lower_bound": bootstrap["statistics"]["sequential_online_ratio_T"]["lower_bound_05"],
                "sequential_online_ratio_threshold": 0.10,
                "acquisition_ok": acquisition_ok,
                "passed": bool(
                    bootstrap["statistics"]["sequential_online_ratio_T"]["lower_bound_05"] >= 0.10
                    and all(acquisition_ok.values())
                    and finite
                    and acquisition[TRAINABLE_ARMS[0]]["total"] == 240
                    and acquisition[TRAINABLE_ARMS[1]]["total"] == 240
                    and acquisition[TRAINABLE_ARMS[2]]["total"] == 240
                ),
            }
        )
        valid_count = len(valid_fraction)
        indeterminate = sum(0.0 < gap < 1e-8 for gap in seq_gaps)
        nonpositive = sum(gap <= 0.0 for gap in seq_gaps)
        g3.update(
            {
                "bootstrap": bootstrap,
                "valid_positive_gap_count": valid_count,
                "indeterminate_count": int(indeterminate),
                "nonpositive_count": int(nonpositive),
                "minimum_valid_count": 16,
                "gap_fraction_median": bootstrap["statistics"]["gap_fraction"].get("median"),
                "gap_fraction_bootstrap_lower_bound": bootstrap["statistics"]["gap_fraction"].get("bootstrap_lower_bound_05"),
                "risk_consistency_failures": int(risk_failures),
                "passed": bool(
                    bootstrap["statistics"]["bounded_history_reduction_T"]["lower_bound_05"] > 0.0
                    and valid_count >= 16
                    and bootstrap["statistics"]["gap_fraction"].get("median") is not None
                    and bootstrap["statistics"]["gap_fraction"]["median"] >= 0.50
                    and risk_failures == 0
                    and finite
                ),
                "selection_gain_lower_bound": bootstrap["statistics"]["selection_gain_T"]["lower_bound_05"],
            }
        )
    return {"G2_phenomenon": g2, "G3_bounded_history": g3}


def run_gpu(
    output_dir: Path,
    seeds: Sequence[int],
    run_role: str,
    device: str = "cuda:0",
    confirmation_of: Path | None = None,
) -> dict[str, Any]:
    global _ACTIVE_RUN_STATE
    config = BridgeConfig()
    assert_protocol_hash(PROTOCOL_PATH, config.protocol_sha256)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be unset; Phase-2 device indices are physical and unmasked")
    run_source_hashes = _source_hashes()
    if run_role not in {"remote_gpu_smoke", "non_citable_calibration", "primary", "confirmation"}:
        raise ValueError("GPU mode requires remote_gpu_smoke, non_citable_calibration, primary, or confirmation")
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("GPU runs require at least one seed")
    freeze_record_sha256: str | None = None
    confirmation_of_manifest_sha256: str | None = None
    if run_role in {"primary", "confirmation"}:
        if seeds != tuple(config.primary_seeds):
            raise ValueError("primary and confirmation runs must use the frozen 20-seed set")
        freeze_path = ROOT / "notes" / "phase2_calibration_freeze.json"
        if not freeze_path.is_file():
            raise RuntimeError("primary is locked until notes/phase2_calibration_freeze.json exists")
        try:
            freeze_report = validate_calibration_freeze(
                freeze_path,
                ROOT,
                config,
                run_source_hashes,
            )
        except EvidenceError as exc:
            raise RuntimeError(f"calibration freeze evidence validation failed: {exc}") from exc
        if freeze_report.get("valid") is not True or freeze_report.get("frozen_for_primary") is not True:
            raise RuntimeError("calibration freeze evidence did not derive a valid primary lock")
        freeze_record_sha256 = artifacts.sha256_file(freeze_path)
        if run_role == "confirmation":
            if confirmation_of is None:
                raise ValueError("confirmation requires --confirmation-of pointing to the primary manifest")
            primary_manifest_path = confirmation_of.resolve()
            if ROOT.resolve() not in primary_manifest_path.parents or not primary_manifest_path.is_file():
                raise RuntimeError("confirmation primary manifest is missing or outside the project")
            primary_manifest = _verified_manifest_outputs(primary_manifest_path)
            if primary_manifest.get("run_role") != "primary":
                raise RuntimeError("confirmation target is not a primary manifest")
            if primary_manifest.get("protocol_sha256") != config.protocol_sha256:
                raise RuntimeError("confirmation target protocol hash mismatch")
            if primary_manifest.get("source_hashes") != run_source_hashes:
                raise RuntimeError("confirmation target source hashes mismatch")
            if primary_manifest.get("frozen_config_sha256") != config.sha256():
                raise RuntimeError("confirmation target frozen config mismatch")
            if primary_manifest.get("ordered_seeds") != [int(seed) for seed in seeds]:
                raise RuntimeError("confirmation target seed order mismatch")
            if primary_manifest.get("calibration_freeze_sha256") != freeze_record_sha256:
                raise RuntimeError("confirmation target calibration freeze mismatch")
            primary_config_path = primary_manifest_path.parent / "config.json"
            if not primary_config_path.is_file():
                raise RuntimeError("confirmation target config artifact is missing")
            if artifacts.sha256_file(primary_config_path) != primary_manifest.get("config_sha256"):
                raise RuntimeError("confirmation target config artifact hash mismatch")
            primary_config = load_json_strict(primary_config_path)
            if primary_config != _config_payload(config, seeds, device):
                raise RuntimeError("confirmation must use the primary frozen experiment config")
            primary_summary = load_json_strict(primary_manifest_path.parent / "summary.json")
            if not isinstance(primary_summary, dict) or primary_summary.get("execution_pass") is not True:
                raise RuntimeError("confirmation target did not complete all single-run gates")
            if primary_summary.get("decision") != "AWAITING_CONFIRMATION_AND_POST_RESULT_AUDIT":
                raise RuntimeError("confirmation target is not awaiting confirmation")
            confirmation_of_manifest_sha256 = artifacts.sha256_file(primary_manifest_path)
        elif confirmation_of is not None:
            raise ValueError("--confirmation-of is valid only for confirmation runs")
    elif run_role == "non_citable_calibration" and seeds != tuple(config.calibration_seeds):
        raise ValueError("formal calibration must use exactly seeds (11,23,37)")
    elif run_role == "remote_gpu_smoke" and not set(seeds).issubset(set(config.calibration_seeds)):
        raise ValueError("remote GPU smoke seeds must be a subset of calibration seeds")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "protocol_sha256": config.protocol_sha256,
        "frozen_config_sha256": config.sha256(),
        "source_hashes": run_source_hashes,
        "test_sha256": run_source_hashes.get("test_phase2_gpu_bridge.py"),
    }
    _ACTIVE_RUN_STATE = {
        "stage": "phase2_gpu_training",
        "run_role": run_role,
        "requested_seeds": [int(seed) for seed in seeds],
        "device": device,
        "output_dir": str(output_dir),
        "identity": run_identity,
        "progress": {"phase": "import_torch"},
    }

    torch = import_torch()
    torch_device = require_cuda(torch, device)
    if torch_device.type != "cuda":
        raise RuntimeError("Phase-2 GPU runs require a CUDA device")
    torch.cuda.set_device(torch_device)
    base = make_base_draws(config)
    numpy_gates = _gates_for_numpy(base, config, run_source_hashes)
    per_window: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    split_manifest: dict[str, Any] = {}
    factor_manifest: dict[str, Any] = {}
    _ACTIVE_RUN_STATE.update(
        {
            "per_window": per_window,
            "per_seed": per_seed,
            "resource_rows": resource_rows,
            "transition_rows": transition_rows,
            "split_manifest": split_manifest,
            "factor_manifest": factor_manifest,
        }
    )
    start = time.time()
    rank_codes = {2: 0, 4: 1, 8: 2}
    regime_codes = {name: index for index, name in enumerate(REGIMES)}
    arm_codes = {name: index for index, name in enumerate(TRAINABLE_ARMS)}
    runtime_metadata = {**environment_payload(), **artifacts.torch_runtime_metadata()}
    _ACTIVE_RUN_STATE["runtime"] = runtime_metadata
    backend_baseline = _warm_cuda_backend(torch, torch_device)
    runtime_metadata["measurement_backend_warmup"] = {
        "kind": "deterministic_max_rank_adamw_step_outside_measured_arms",
        "forward_count": 1,
        "backward_count": 1,
        "optimizer_step_count": 1,
        "allocated_bytes": backend_baseline["allocated_bytes"],
        "reserved_bytes": backend_baseline["reserved_bytes"],
    }
    _ACTIVE_RUN_STATE["backend_baseline"] = dict(backend_baseline)
    device_index = int(torch_device.index or 0)
    matching_devices = [
        row
        for row in runtime_metadata.get("nvidia_smi_devices", [])
        if isinstance(row, Mapping) and int(row.get("index", -1)) == device_index
    ]
    if len(matching_devices) != 1 or not str(matching_devices[0].get("driver_version", "")):
        raise RuntimeError("unable to bind the selected CUDA device to a structured driver identity")
    selected_device_metadata = matching_devices[0]
    torch_device_name = torch.cuda.get_device_name(torch_device)
    if str(selected_device_metadata.get("name")) != torch_device_name:
        raise RuntimeError("torch and nvidia-smi disagree on the selected GPU identity")
    deterministic_flags = {
        "deterministic_algorithms": runtime_metadata.get("deterministic_algorithms"),
        "cudnn_deterministic": runtime_metadata.get("cudnn_deterministic"),
        "cudnn_benchmark": runtime_metadata.get("cudnn_benchmark"),
        "cuda_matmul_allow_tf32": runtime_metadata.get("cuda_matmul_allow_tf32"),
        "cudnn_allow_tf32": runtime_metadata.get("cudnn_allow_tf32"),
        "float32_matmul_precision": runtime_metadata.get("float32_matmul_precision"),
        "environment_controls": runtime_metadata.get("environment_controls"),
    }

    for seed in seeds:
        _set_run_progress(phase="build_seed_dataset", seed=int(seed))
        streams = make_seed_streams(seed, config)
        dataset = build_seed_dataset(base, seed, config, streams)
        split_manifest[str(seed)] = _split_manifest_entry(dataset, base, config)
        initializations = make_factor_initializations(streams, config)
        factor_manifest[str(seed)] = {
            f"{regime}:{rank}": {
                "sha256": initializations[(regime, rank)].sha256(),
                "A_shape": list(initializations[(regime, rank)].A32.shape),
                "B_shape": list(initializations[(regime, rank)].B32.shape),
            }
            for regime in REGIMES
            for rank in RANKS
        }
        seed_dataset_bytes = int(
            sum(_cell_dataset_bytes(dataset.cell(regime, rank)) for regime in REGIMES for rank in RANKS)
        )
        for arm in TRAINABLE_ARMS:
            for regime in REGIMES:
                for rank in RANKS:
                    _set_run_progress(
                        phase="initialize_arm_cell",
                        seed=int(seed),
                        regime=regime,
                        R=int(rank),
                        arm=arm,
                        window=None,
                        update=None,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(torch_device)
                    baseline_allocated = int(torch.cuda.memory_allocated(torch_device))
                    baseline_reserved = int(torch.cuda.memory_reserved(torch_device))
                    torch.cuda.reset_peak_memory_stats(torch_device)
                    cell_started = time.perf_counter()
                    training_wall_seconds = 0.0
                    init = initializations[(regime, rank)]
                    model = make_model(torch, base.W0.astype(np.float32), init.A32)
                    model.to(torch_device)
                    optimizer = make_optimizer(torch, model)
                    if arm == TRAINABLE_ARMS[1]:
                        controller: Any | None = ClockBalancedReplay(
                            regime_code=regime_codes[regime], rank_code=rank_codes[rank]
                        )
                        replay_rng: Any | None = streams.clock_replay_draws
                        insertion_rng: Any | None = streams.clock_insert
                    elif arm == TRAINABLE_ARMS[2]:
                        controller = ReservoirReplay(
                            regime_code=regime_codes[regime], rank_code=rank_codes[rank]
                        )
                        replay_rng = streams.reservoir_replay_draws
                        insertion_rng = streams.reservoir_priority
                    else:
                        controller = None
                        replay_rng = None
                        insertion_rng = None
                    cell = dataset.cell(regime, rank)
                    for window in cell:
                        _set_run_progress(
                            phase="train_window",
                            seed=int(seed),
                            regime=regime,
                            R=int(rank),
                            arm=arm,
                            window=int(window.window),
                            update="0..99",
                        )
                        before = 0 if controller is None else int(controller.n_occupied)
                        trace: list[Any] = []
                        torch.cuda.synchronize(torch_device)
                        train_started = time.perf_counter()
                        train_window(
                            torch,
                            model,
                            optimizer,
                            x_current=window.train.x32,
                            y_current=window.train.y32,
                            pi_t=window.pi_t,
                            replay_state=controller,
                            replay_rng=replay_rng,
                            device=torch_device,
                            updates=config.updates_per_window,
                            trace=trace,
                        )
                        torch.cuda.synchronize(torch_device)
                        train_elapsed = time.perf_counter() - train_started
                        training_wall_seconds += train_elapsed
                        replay_examples = int(sum(int(item.replay_count) for item in trace))
                        replay_draws = int(
                            sum(int(item.replay_count == config.replay_batch) for item in trace)
                        )
                        replay_duplicates = int(
                            sum(int(item.replay_duplicate_count) for item in trace)
                        )
                        replay_sampling_seconds = float(
                            sum(float(item.replay_sample_seconds) for item in trace)
                        )
                        replay_assembly_seconds = float(
                            sum(float(item.replay_assembly_seconds) for item in trace)
                        )
                        device_staging_seconds = float(
                            sum(float(item.device_staging_seconds) for item in trace)
                        )
                        metadata_leakage_events = int(
                            sum(int(item.latent_metadata_tensor_count) for item in trace)
                        )
                        model_input_contract_passed = bool(
                            len(trace) == config.updates_per_window
                            and all(
                                item.model_input_names == ("x", "y")
                                and item.model_input_shapes == ((128, 32), (128, 32))
                                and item.model_input_dtypes == ("torch.float32", "torch.float32")
                                and item.latent_metadata_tensor_count == 0
                                for item in trace
                            )
                        )
                        sampler_contract_passed = bool(
                            all(
                                item.replay_sampler_argument_names
                                == (("draw_rng", "size") if item.replay_count else ())
                                for item in trace
                            )
                        )
                        delta = model_delta64(model)
                        row = _evaluate_checkpoint(
                            base,
                            regime,
                            rank,
                            window.window,
                            delta,
                            arm,
                            seed,
                            config.updates_per_window,
                            replay_examples,
                        )
                        row["factor_sha256"] = init.sha256()
                        row["occupied"] = before
                        row["unique"] = before
                        row["duplicates"] = replay_duplicates
                        row["priority_draws"] = 0 if controller is None else int(controller.priority_draws)
                        row["replay_sample_draws"] = replay_draws
                        row["selection_comparisons"] = 0 if controller is None else int(controller.comparison_count)
                        row["controller_cpu_seconds"] = replay_sampling_seconds
                        row["replay_assembly_seconds"] = replay_assembly_seconds
                        row["device_staging_seconds"] = device_staging_seconds
                        row["metadata_leakage_events"] = metadata_leakage_events
                        row["model_input_contract_passed"] = model_input_contract_passed
                        row["replay_sampler_contract_passed"] = sampler_contract_passed
                        row["train_wall_seconds"] = float(train_elapsed)
                        row["peak_allocated_gpu_bytes_so_far"] = int(
                            torch.cuda.max_memory_allocated(torch_device)
                        )
                        row["peak_reserved_gpu_bytes_so_far"] = int(
                            torch.cuda.max_memory_reserved(torch_device)
                        )
                        if row["D_t"] < 1e-8 or row["D_current"] < 1e-8:
                            row["validity"] = "invalid_denominator"
                        elif row["gap"] < -1e-10:
                            row["validity"] = "risk_inconsistency"
                        elif row["gap"] < 1e-8:
                            row["validity"] = "indeterminate_gap"
                        per_window.append(row)
                        if window.window < config.T and controller is not None:
                            _set_run_progress(
                                phase="insert_completed_window",
                                seed=int(seed),
                                regime=regime,
                                R=int(rank),
                                arm=arm,
                                window=int(window.window),
                                update=None,
                            )
                            before_priority = int(controller.priority_draws)
                            before_comparison = int(controller.comparison_count)
                            insert_started = time.perf_counter()
                            transition = controller.insert_completed_window(
                                window.window,
                                window.train.x32,
                                window.train.y32,
                                insertion_rng,
                            )
                            insert_elapsed = time.perf_counter() - insert_started
                            transition_rows.append(
                                {
                                    "seed": seed,
                                    "regime": regime,
                                    "R": rank,
                                    "arm": arm,
                                    "completed_window": int(transition.completed_window),
                                    "history_windows": int(transition.history_windows),
                                    "capacity": int(transition.capacity),
                                    "eligible_count": int(transition.eligible_count),
                                    "occupied_count": int(transition.occupied_count),
                                    "unique_count": int(transition.unique_count),
                                    "duplicate_count": int(transition.duplicate_count),
                                    "replacement_count": int(transition.replacement_count),
                                    "priority_draws": int(controller.priority_draws - before_priority),
                                    "selection_comparisons": int(controller.comparison_count - before_comparison),
                                    "priority_tie_count": int(transition.priority_tie_count),
                                    "controller_cpu_seconds": float(insert_elapsed),
                                    "selected_priority_order_sha256": transition.selected_priority_order_sha256,
                                }
                            )

                    torch.cuda.synchronize(torch_device)
                    peak_allocated = int(torch.cuda.max_memory_allocated(torch_device))
                    peak_reserved = int(torch.cuda.max_memory_reserved(torch_device))
                    optimizer_payload = _optimizer_resource_payload(torch, model, optimizer)
                    optimizer.zero_grad(set_to_none=True)
                    del optimizer
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(torch_device)
                    inference_baseline_allocated = int(torch.cuda.memory_allocated(torch_device))
                    inference_baseline_reserved = int(torch.cuda.memory_reserved(torch_device))
                    torch.cuda.reset_peak_memory_stats(torch_device)
                    inference = _measure_inference_latency(
                        torch, model, cell[-1].train.x32, torch_device
                    )
                    inference_peak_allocated = int(torch.cuda.max_memory_allocated(torch_device))
                    inference_peak_reserved = int(torch.cuda.max_memory_reserved(torch_device))
                    cell_wall_seconds = time.perf_counter() - cell_started
                    cell_rows = [
                        row
                        for row in per_window
                        if row["seed"] == seed
                        and row["regime"] == regime
                        and row["R"] == rank
                        and row["arm"] == arm
                    ]
                    final = cell_rows[-1]
                    per_seed.append(
                        {
                            "seed": seed,
                            "regime": regime,
                            "R": rank,
                            "arm": arm,
                            "final_gap": final["gap"],
                            "final_online_ratio": final["online_ratio"],
                            "final_current_acquisition_ratio": final["current_acquisition_ratio"],
                            "final_bounded_history_reduction": final.get("bounded_history_reduction"),
                            "final_selection_gain": final.get("selection_gain"),
                            "final_gap_fraction": final.get("gap_fraction"),
                            "final_validity": final["validity"],
                        }
                    )
                    transition_for_arm = [
                        row
                        for row in transition_rows
                        if row["seed"] == seed
                        and row["regime"] == regime
                        and row["R"] == rank
                        and row["arm"] == arm
                    ]
                    replay_bytes = 0 if controller is None else REPLAY_BYTES
                    parameter_count = int(optimizer_payload["trainable_parameter_count"])
                    parameter_bytes = int(optimizer_payload["trainable_parameter_bytes"])
                    evaluator_checkpoint_staging_bytes = int(parameter_count * 8 + config.d_in * config.d_out * 8)
                    resource_rows.append(
                        {
                            "schema_version": 1,
                            "seed": seed,
                            "regime": regime,
                            "R": rank,
                            "arm": arm,
                            "requested_rank": rank,
                            "active_rank": rank,
                            "deployed_rank": rank,
                            "numerical_rank_max": int(max(row["numerical_rank"] for row in cell_rows)),
                            "factor_shapes": json.dumps(
                                {"B": list(model.B.shape), "A": list(model.A.shape)}, separators=(",", ":")
                            ),
                            "trainable_parameter_count": parameter_count,
                            "deployed_parameter_count": parameter_count,
                            "trainable_parameter_bytes": parameter_bytes,
                            "deployed_parameter_bytes": parameter_bytes,
                            "parameter_breakdown": json.dumps(optimizer_payload["parameter_breakdown"], separators=(",", ":")),
                            "optimizer_state_bytes": int(optimizer_payload["optimizer_state_bytes"]),
                            "optimizer_state_breakdown": json.dumps(optimizer_payload["optimizer_state_breakdown"], separators=(",", ":")),
                            "optimizer_steps": json.dumps(optimizer_payload["optimizer_steps"], separators=(",", ":")),
                            "optimizer_contract": json.dumps(optimizer_payload["optimizer_contract"], sort_keys=True, separators=(",", ":")),
                            "optimizer_contract_sha256": optimizer_payload["optimizer_contract_sha256"],
                            "batch_size": int(config.batch_size),
                            "replay_fraction": 0.0 if controller is None else float(config.replay_batch / config.batch_size),
                            "allocated_capacity": 0 if controller is None else B_MAX,
                            "replay_bytes": replay_bytes,
                            "replay_x_y_slot_bytes": 0 if replay_bytes == 0 else 90112,
                            "replay_priority_bytes": 0 if replay_bytes == 0 else 2816,
                            "replay_occupancy_bytes": 0 if replay_bytes == 0 else 352,
                            "pointer_counter_bytes": 0 if replay_bytes == 0 else 208,
                            "persistent_controller_sidecar_records": 0 if controller is None else int(controller.persistent_sidecar_records),
                            "persistent_controller_sidecar_bytes": 0 if controller is None else int(controller.persistent_sidecar_bytes),
                            "controller_transient_records_peak": 0 if controller is None else int(controller.transient_records_peak),
                            "controller_transient_python_bytes_peak": 0 if controller is None else int(controller.transient_python_bytes_peak),
                            "controller_transient_numpy_bytes_peak": 0 if controller is None else int(controller.transient_numpy_bytes_peak),
                            "curvature_sketch_bytes": 0,
                            "checkpoint_persistent_bytes": 0,
                            "evaluator_checkpoint_staging_bytes": evaluator_checkpoint_staging_bytes,
                            "common_W0_cpu_bytes": int(base.W0.astype(np.float32).nbytes),
                            "common_W0_gpu_bytes": int(base.W0.astype(np.float32).nbytes),
                            "common_cpu_cell_dataset_bytes": _cell_dataset_bytes(cell),
                            "common_cpu_seed_dataset_resident_bytes": seed_dataset_bytes,
                            "common_cpu_determinism_rebuild_increment_bytes": seed_dataset_bytes,
                            "common_cpu_determinism_rebuild_peak_bytes": 2 * seed_dataset_bytes,
                            "common_cpu_batch_staging_bytes": int(2 * config.batch_size * config.d_in * 4),
                            "common_cpu_index_staging_bytes_max": int(config.batch_size * 8),
                            "arm_cpu_batch_staging_peak_bytes": int(
                                2 * config.batch_size * config.d_in * 4
                                if controller is None
                                else 2 * (config.batch_size * config.d_in * 4)
                                + 4 * (config.replay_batch * config.d_in * 4)
                            ),
                            "arm_cpu_index_staging_peak_bytes": int(
                                config.batch_size * 8
                                if controller is None
                                else config.batch_size * 8
                                + 2 * B_MAX * np.dtype(np.int64).itemsize
                                + 2 * config.replay_batch * np.dtype(np.int64).itemsize
                            ),
                            "common_gpu_batch_staging_bytes": int(2 * config.batch_size * config.d_in * 4),
                            "updates": int(sum(row["updates"] for row in cell_rows)),
                            "forward": int(sum(row["forward"] for row in cell_rows)),
                            "backward": int(sum(row["backward"] for row in cell_rows)),
                            "current_examples": int(sum(row["current_examples"] for row in cell_rows)),
                            "replay_examples": int(sum(row["replay_examples"] for row in cell_rows)),
                            "replay_sample_draws": int(sum(row["replay_sample_draws"] for row in cell_rows)),
                            "priority_draws": int(sum(row["priority_draws"] for row in transition_for_arm)),
                            "selection_comparisons": int(sum(row["selection_comparisons"] for row in transition_for_arm)),
                            "priority_tie_count": int(sum(row["priority_tie_count"] for row in transition_for_arm)),
                            "replacement_count": int(sum(row["replacement_count"] for row in transition_for_arm)),
                            "occupied_records_max": int(max(row["occupied"] for row in cell_rows)),
                            "unique_records_max": int(max(row["unique"] for row in cell_rows)),
                            "replay_sampling_duplicate_count": int(sum(row["duplicates"] for row in cell_rows)),
                            "controller_cpu_seconds": float(
                                sum(row["controller_cpu_seconds"] for row in cell_rows)
                                + sum(row["controller_cpu_seconds"] for row in transition_for_arm)
                            ),
                            "replay_assembly_seconds": float(sum(row["replay_assembly_seconds"] for row in cell_rows)),
                            "device_staging_seconds": float(sum(row["device_staging_seconds"] for row in cell_rows)),
                            "training_wall_seconds": float(training_wall_seconds),
                            "wall_seconds": float(cell_wall_seconds),
                            "gpu_baseline_allocated_bytes": baseline_allocated,
                            "gpu_baseline_reserved_bytes": baseline_reserved,
                            "peak_allocated_gpu_bytes": peak_allocated,
                            "peak_reserved_gpu_bytes": peak_reserved,
                            "peak_allocated_increment_bytes": int(peak_allocated - baseline_allocated),
                            "peak_reserved_increment_bytes": int(peak_reserved - baseline_reserved),
                            "inference_peak_allocated_gpu_bytes": inference_peak_allocated,
                            "inference_peak_reserved_gpu_bytes": inference_peak_reserved,
                            "inference_baseline_allocated_bytes": inference_baseline_allocated,
                            "inference_baseline_reserved_bytes": inference_baseline_reserved,
                            "inference_peak_allocated_increment_bytes": int(
                                inference_peak_allocated - inference_baseline_allocated
                            ),
                            "inference_peak_reserved_increment_bytes": int(
                                inference_peak_reserved - inference_baseline_reserved
                            ),
                            "inference_latency_mean_batch_seconds": inference["mean_batch_seconds"],
                            "inference_latency_mean_example_seconds": inference["mean_example_seconds"],
                            "inference_warmup_forwards": inference["warmup_forwards"],
                            "inference_measured_forwards": inference["measured_forwards"],
                            "metadata_leakage_events": int(sum(row["metadata_leakage_events"] for row in cell_rows)),
                            "model_input_contract_passed": bool(all(row["model_input_contract_passed"] for row in cell_rows)),
                            "replay_sampler_contract_passed": bool(all(row["replay_sampler_contract_passed"] for row in cell_rows)),
                            "protocol_sha256": config.protocol_sha256,
                            "frozen_config_sha256": config.sha256(),
                            "source_hashes": json.dumps(run_source_hashes, sort_keys=True, separators=(",", ":")),
                            "test_sha256": run_source_hashes.get("test_phase2_gpu_bridge.py"),
                            "data_sha256": dataset.data_sha256,
                            "split_sha256": dataset.split_sha256,
                            "factor_sha256": init.sha256(),
                            "device": str(torch_device),
                            "device_index": device_index,
                            "gpu_name": torch_device_name,
                            "driver_version": str(selected_device_metadata["driver_version"]),
                            "hostname": runtime_metadata.get("hostname"),
                            "python_version": runtime_metadata.get("python_version"),
                            "numpy_version": runtime_metadata.get("numpy_version"),
                            "torch_version": runtime_metadata.get("torch_version"),
                            "cuda_version": runtime_metadata.get("cuda_version"),
                            "cudnn_version": runtime_metadata.get("cudnn_version"),
                            "nvidia_smi_devices": json.dumps(runtime_metadata.get("nvidia_smi_devices", []), separators=(",", ":")),
                            "deterministic_flags": json.dumps(deterministic_flags, sort_keys=True, separators=(",", ":")),
                        }
                    )
                    del model, controller
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(torch_device)

    per_window.sort(
        key=lambda row: (
            tuple(seeds).index(int(row["seed"])),
            regime_codes[str(row["regime"])],
            rank_codes[int(row["R"])],
            int(row["window"]),
            arm_codes[str(row["arm"])],
        )
    )
    per_seed.sort(
        key=lambda row: (
            tuple(seeds).index(int(row["seed"])),
            regime_codes[str(row["regime"])],
            rank_codes[int(row["R"])],
            arm_codes[str(row["arm"])],
        )
    )
    resource_rows.sort(
        key=lambda row: (
            tuple(seeds).index(int(row["seed"])),
            regime_codes[str(row["regime"])],
            rank_codes[int(row["R"])],
            arm_codes[str(row["arm"])],
        )
    )
    transition_rows.sort(
        key=lambda row: (
            tuple(seeds).index(int(row["seed"])),
            regime_codes[str(row["regime"])],
            rank_codes[int(row["R"])],
            arm_codes[str(row["arm"])],
            int(row["completed_window"]),
        )
    )
    _derive_contrasts(per_window)
    for row in per_seed:
        matching = [
            item
            for item in per_window
            if item["seed"] == row["seed"]
            and item["regime"] == row["regime"]
            and item["R"] == row["R"]
            and item["window"] == config.T
            and item["arm"] == row["arm"]
        ]
        if matching:
            final = matching[0]
            row["final_bounded_history_reduction"] = final.get("bounded_history_reduction")
            row["final_selection_gain"] = final.get("selection_gain")
            row["final_gap_fraction"] = final.get("gap_fraction")

    split_seed_entries = list(split_manifest.values())
    split_cells = [cell for seed_entry in split_seed_entries for cell in seed_entry["cells"].values()]
    split_passed = bool(
        all(
            entry["deterministic_rebuild_passed"]
            and entry["base_finite_passed"]
            and entry["basis_orthogonality_passed"]
            for entry in split_seed_entries
        )
        and all(
            cell["windows"] == config.T
            and cell["train_counts"] == [config.n_train] * config.T
            and cell["eval_counts"] == [config.n_eval] * config.T
            and cell["train_eval_global_key_overlap"] == [0] * config.T
            and all(
                item["pi_matches_train_storage_permutation"]
                and item["pi_is_full_permutation"]
                and all(
                    block["mode_counts_match"]
                    and block["balanced_signs_passed"]
                    and block["record_ids_passed"]
                    and block["global_key_schema_passed"]
                    and block["storage_permutation_passed"]
                    and block["finite_arrays_passed"]
                    and block["float32_casts_passed"]
                    for block in (item["train"], item["eval"])
                )
                for item in cell["integrity"]
            )
            for cell in split_cells
        )
    )
    finite_passed = all(
        np.isfinite(float(row[field]))
        for row in per_window
        for field in (
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
    )
    risk_consistency_failures = [
        {
            "seed": int(row["seed"]),
            "regime": str(row["regime"]),
            "R": int(row["R"]),
            "window": int(row["window"]),
            "arm": str(row["arm"]),
            "gap": float(row["gap"]),
        }
        for row in per_window
        if float(row["gap"]) < -1e-10
    ]
    risk_consistency_passed = len(risk_consistency_failures) == 0
    rank_passed = all(int(row["numerical_rank"]) <= int(row["R"]) for row in per_window)
    learner_boundary_passed = bool(
        all(
            int(row["metadata_leakage_events"]) == 0
            and bool(row["model_input_contract_passed"])
            and bool(row["replay_sampler_contract_passed"])
            for row in per_window
        )
    )
    factor_shared = all(
        len(
            {
                row["factor_sha256"]
                for row in per_window
                if row["seed"] == seed and row["regime"] == regime and row["R"] == rank
            }
        )
        == 1
        for seed in seeds
        for regime in REGIMES
        for rank in RANKS
    )
    factor_initializations_bound = all(
        {
            row["factor_sha256"]
            for row in per_window
            if row["seed"] == seed and row["regime"] == regime and row["R"] == rank
        }
        == {factor_manifest[str(seed)][f"{regime}:{rank}"]["sha256"]}
        and {
            row["factor_sha256"]
            for row in resource_rows
            if row["seed"] == seed and row["regime"] == regime and row["R"] == rank
        }
        == {factor_manifest[str(seed)][f"{regime}:{rank}"]["sha256"]}
        for seed in seeds
        for regime in REGIMES
        for rank in RANKS
    )
    numpy_gates["G0_numpy_identity"].update(
        {
            "split_integrity_passed": split_passed,
            "finite_metrics_passed": finite_passed,
            "risk_consistency_passed": risk_consistency_passed,
            "risk_consistency_failure_count": len(risk_consistency_failures),
            "risk_consistency_failures": risk_consistency_failures,
            "rank_passed": rank_passed,
            "factor_shared_across_arms": factor_shared,
            "factor_initializations_bound": factor_initializations_bound,
            "runtime_learner_boundary_passed": learner_boundary_passed,
        }
    )
    numpy_gates["G0_numpy_identity"]["passed"] = bool(
        numpy_gates["G0_numpy_identity"]["passed"]
        and split_passed
        and finite_passed
        and risk_consistency_passed
        and rank_passed
        and factor_shared
        and factor_initializations_bound
        and learner_boundary_passed
    )

    expected_transition_rows = len(seeds) * len(REGIMES) * len(RANKS) * 2 * (config.T - 1)
    transition_passed = len(transition_rows) == expected_transition_rows and all(
        row["capacity"] == B_MAX
        and row["occupied_count"] == B_MAX
        and row["unique_count"] == B_MAX
        and row["duplicate_count"] == 0
        and row["priority_draws"] == config.n_train
        and row["selection_comparisons"] > 0
        and row["replacement_count"] >= 0
        and row["priority_tie_count"] >= 0
        and len(str(row["selected_priority_order_sha256"])) == 64
        for row in transition_rows
    )
    occupancy_passed = all(
        (row["occupied"] == 0 and row["unique"] == 0)
        if row["arm"] == TRAINABLE_ARMS[0] or int(row["window"]) == 1
        else (row["occupied"] == B_MAX and row["unique"] == B_MAX)
        for row in per_window
    )
    expected_optimizer_contract = _expected_optimizer_contract()
    expected_optimizer_contract_sha256 = artifacts.hash_json(expected_optimizer_contract)
    optimizer_contract_passed = all(
        json.loads(row["optimizer_contract"]) == expected_optimizer_contract
        and row["optimizer_contract_sha256"] == expected_optimizer_contract_sha256
        and json.loads(row["parameter_breakdown"])
        == [
            {
                "name": "B",
                "shape": [config.d_out, int(row["requested_rank"])],
                "dtype": "torch.float32",
                "numel": config.d_out * int(row["requested_rank"]),
                "bytes": config.d_out * int(row["requested_rank"]) * np.dtype(np.float32).itemsize,
            },
            {
                "name": "A",
                "shape": [int(row["requested_rank"]), config.d_in],
                "dtype": "torch.float32",
                "numel": config.d_in * int(row["requested_rank"]),
                "bytes": config.d_in * int(row["requested_rank"]) * np.dtype(np.float32).itemsize,
            },
        ]
        and row["optimizer_state_bytes"]
        == 2 * row["trainable_parameter_bytes"] + 2 * np.dtype(np.float32).itemsize
        and json.loads(row["optimizer_state_breakdown"])
        == [
            {
                "parameter": parameter_name,
                "state": state_name,
                "shape": ([] if state_name == "step" else list(shape)),
                "dtype": "torch.float32",
                "bytes": (
                    np.dtype(np.float32).itemsize
                    if state_name == "step"
                    else int(np.prod(shape)) * np.dtype(np.float32).itemsize
                ),
            }
            for parameter_name, shape in (
                ("B", (config.d_out, int(row["requested_rank"]))),
                ("A", (int(row["requested_rank"]), config.d_in)),
            )
            for state_name in ("exp_avg", "exp_avg_sq", "step")
        ]
        for row in resource_rows
    )
    compute_passed = all(
        row["requested_rank"] == row["active_rank"] == row["deployed_rank"]
        and row["numerical_rank_max"] <= row["requested_rank"]
        and json.loads(row["factor_shapes"])
        == {"B": [config.d_out, int(row["requested_rank"])], "A": [int(row["requested_rank"]), config.d_in]}
        and row["updates"] == config.T * config.updates_per_window
        and row["forward"] == row["backward"] == row["updates"]
        and row["current_examples"] + row["replay_examples"] == row["updates"] * config.batch_size
        and row["trainable_parameter_count"] == row["requested_rank"] * (config.d_in + config.d_out)
        and row["trainable_parameter_bytes"] == row["trainable_parameter_count"] * np.dtype(np.float32).itemsize
        and row["deployed_parameter_count"] == row["trainable_parameter_count"]
        and row["deployed_parameter_bytes"] == row["trainable_parameter_bytes"]
        and row["batch_size"] == config.batch_size
        and json.loads(row["optimizer_steps"]) == [config.T * config.updates_per_window] * 2
        and row["curvature_sketch_bytes"] == 0
        and row["checkpoint_persistent_bytes"] == 0
        for row in resource_rows
    )
    replay_resource_passed = all(
        (
            row["replay_bytes"] == 0
            and row["allocated_capacity"] == 0
            and row["replay_fraction"] == 0.0
            and row["replay_examples"] == 0
            and row["replay_sample_draws"] == 0
            and row["priority_draws"] == 0
            and row["persistent_controller_sidecar_records"] == 0
            and row["persistent_controller_sidecar_bytes"] == 0
            and row["controller_transient_records_peak"] == 0
        )
        if row["arm"] == TRAINABLE_ARMS[0]
        else (
            row["replay_bytes"] == REPLAY_BYTES
            and row["allocated_capacity"] == B_MAX
            and row["replay_fraction"] == config.replay_batch / config.batch_size
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
        for row in resource_rows
    )
    matched_arm_passed = True
    for seed in seeds:
        for regime in REGIMES:
            for rank in RANKS:
                cell = [row for row in resource_rows if row["seed"] == seed and row["regime"] == regime and row["R"] == rank]
                if len(cell) != len(TRAINABLE_ARMS):
                    matched_arm_passed = False
                    continue
                matched_arm_passed = matched_arm_passed and len({row["trainable_parameter_bytes"] for row in cell}) == 1
                matched_arm_passed = matched_arm_passed and len({row["optimizer_state_bytes"] for row in cell}) == 1
                matched_arm_passed = matched_arm_passed and len({row["optimizer_contract_sha256"] for row in cell}) == 1
                matched_arm_passed = matched_arm_passed and len({row["factor_sha256"] for row in cell}) == 1
                replay_rows = [row for row in cell if row["arm"] != TRAINABLE_ARMS[0]]
                matched_arm_passed = matched_arm_passed and len({row["replay_bytes"] for row in replay_rows}) == 1
                matched_arm_passed = matched_arm_passed and len({row["replay_examples"] for row in replay_rows}) == 1
                matched_arm_passed = matched_arm_passed and len({row["replay_sample_draws"] for row in replay_rows}) == 1
                matched_arm_passed = matched_arm_passed and len({row["priority_draws"] for row in replay_rows}) == 1
    # Keep the in-run accounting gate aligned with the validator's complete
    # resource schema.  A partial subset here could mark calibration as valid
    # while omitting fields that the primary evidence later relies on.
    required_resource_fields = RESOURCE_REQUIRED_FIELDS
    accounting_complete_passed = bool(
        resource_rows
        and all(required_resource_fields.issubset(row) for row in resource_rows)
        and all(
            row["gpu_baseline_allocated_bytes"] == backend_baseline["allocated_bytes"]
            and row["gpu_baseline_reserved_bytes"] == backend_baseline["reserved_bytes"]
            and all(row[field] is not None for field in required_resource_fields)
            and bool(str(row["driver_version"]))
            and row["common_cpu_seed_dataset_resident_bytes"] >= row["common_cpu_cell_dataset_bytes"] > 0
            and row["common_cpu_determinism_rebuild_increment_bytes"]
            == row["common_cpu_seed_dataset_resident_bytes"]
            and row["common_cpu_determinism_rebuild_peak_bytes"]
            == 2 * row["common_cpu_seed_dataset_resident_bytes"]
            and row["common_cpu_index_staging_bytes_max"] == config.batch_size * np.dtype(np.int64).itemsize
            and row["arm_cpu_batch_staging_peak_bytes"]
            == (
                2 * config.batch_size * config.d_in * np.dtype(np.float32).itemsize
                if row["arm"] == TRAINABLE_ARMS[0]
                else 2 * config.batch_size * config.d_in * np.dtype(np.float32).itemsize
                + 4 * config.replay_batch * config.d_in * np.dtype(np.float32).itemsize
            )
            and row["arm_cpu_index_staging_peak_bytes"]
            == (
                config.batch_size * np.dtype(np.int64).itemsize
                if row["arm"] == TRAINABLE_ARMS[0]
                else config.batch_size * np.dtype(np.int64).itemsize
                + 2 * B_MAX * np.dtype(np.int64).itemsize
                + 2 * config.replay_batch * np.dtype(np.int64).itemsize
            )
            and row["peak_allocated_increment_bytes"] >= 0
            and row["peak_reserved_increment_bytes"] >= 0
            and row["inference_baseline_allocated_bytes"] > 0
            and row["inference_baseline_reserved_bytes"] >= row["inference_baseline_allocated_bytes"]
            and row["inference_peak_allocated_gpu_bytes"] >= row["inference_baseline_allocated_bytes"]
            and row["inference_peak_reserved_gpu_bytes"] >= row["inference_baseline_reserved_bytes"]
            and row["inference_peak_allocated_increment_bytes"] >= 0
            and row["inference_peak_reserved_increment_bytes"] >= 0
            and np.isfinite(float(row["controller_cpu_seconds"]))
            and np.isfinite(float(row["wall_seconds"]))
            and np.isfinite(float(row["inference_latency_mean_batch_seconds"]))
            and float(row["inference_latency_mean_batch_seconds"]) > 0.0
            and row["inference_warmup_forwards"] == 5
            and row["inference_measured_forwards"] == 20
            and int(row["metadata_leakage_events"]) == 0
            and bool(row["model_input_contract_passed"])
            and bool(row["replay_sampler_contract_passed"])
            for row in resource_rows
        )
    )
    g4_passed = bool(
        transition_passed
        and occupancy_passed
        and compute_passed
        and optimizer_contract_passed
        and replay_resource_passed
        and matched_arm_passed
        and accounting_complete_passed
    )
    gates = {
        **numpy_gates,
        **_post_run_gates(per_window, per_seed, config, run_role),
        "G4_resource_and_reproducibility": {
            "passed": g4_passed if run_role not in {"primary", "confirmation"} else False,
            "single_run_passed": g4_passed,
            "transition_passed": transition_passed,
            "occupancy_passed": occupancy_passed,
            "compute_passed": compute_passed,
            "optimizer_contract_passed": optimizer_contract_passed,
            "expected_optimizer_contract_sha256": expected_optimizer_contract_sha256,
            "replay_resource_passed": replay_resource_passed,
            "matched_arm_passed": matched_arm_passed,
            "accounting_complete_passed": accounting_complete_passed,
            "transition_rows": len(transition_rows),
            "expected_transition_rows": expected_transition_rows,
            "confirmation_required_for_citation": True,
            "confirmation_status": "pending",
            "post_result_four_provider_audit_status": "pending",
            "status": "pending_reconciliation" if run_role in {"primary", "confirmation"} and g4_passed else ("passed_non_citable_single_run" if g4_passed else "failed"),
            "peak_gpu_measurement_scope": "arm_cell_isolated_full_12_window_training_from_pre_model_process_resident_baseline; inference_independently_reset_from_post_training_model_resident_baseline",
        },
    }
    split_manifest_payload = {
        "protocol_sha256": config.protocol_sha256,
        "base_sha256": base.sha256(),
        "seeds": split_manifest,
    }
    traces = {
        "protocol_sha256": config.protocol_sha256,
        "base_sha256": base.sha256(),
        "source_hashes": run_source_hashes,
        "factor_initializations": factor_manifest,
        "transitions": transition_rows,
        "deterministic_controls": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "MKL_DYNAMIC", "CUBLAS_WORKSPACE_CONFIG")
        },
        "runtime_learner_boundary": {
            "passed": learner_boundary_passed,
            "gradient_model_input_names": ["x", "y"],
            "gradient_model_input_shape": [128, 32],
            "gradient_model_input_dtype": "torch.float32",
            "latent_metadata_tensor_events": int(sum(row["metadata_leakage_events"] for row in per_window)),
            "replay_sampler_argument_names": ["draw_rng", "size"],
            "permitted_scheduler_fields": ["window", "pi_t", "insertion_boundary"],
            "permitted_post_run_evaluator_fields": ["window", "mode_ids", "schedule", "split_indices"],
        },
    }
    if _source_hashes() != run_source_hashes:
        raise RuntimeError("Phase-2 source files changed while the run was in progress")
    _write_csv(output_dir / "per_window.csv", per_window, list(per_window[0].keys()) if per_window else ["seed"])
    _write_csv(output_dir / "per_seed.csv", per_seed, list(per_seed[0].keys()) if per_seed else ["seed"])
    _write_csv(output_dir / "resource_ledger.csv", resource_rows, list(resource_rows[0].keys()) if resource_rows else ["seed"])
    _json_dump(output_dir / "config.json", _config_payload(config, seeds, device))
    _json_dump(output_dir / "split_manifest.json", split_manifest_payload)
    _json_dump(output_dir / "gates.json", gates)
    _json_dump(output_dir / "traces.json", traces)
    required_gate_names = ["G0_numpy_identity", "G1_oracle_capacity", "G4_resource_and_reproducibility"]
    if run_role in {"primary", "confirmation"}:
        required_gate_names.extend(("G2_phenomenon", "G3_bounded_history"))
    execution_pass = bool(
        gates["G0_numpy_identity"]["passed"]
        and gates["G1_oracle_capacity"]["passed"]
        and gates["G4_resource_and_reproducibility"]["single_run_passed"]
        and (run_role not in {"primary", "confirmation"} or (gates["G2_phenomenon"].get("passed", False) and gates["G3_bounded_history"].get("passed", False)))
    )
    all_required_pass = bool(
        execution_pass
        and (run_role not in {"primary", "confirmation"} or gates["G4_resource_and_reproducibility"]["passed"])
    )
    decision = "NON_CITABLE_PASS" if all_required_pass else "NON_CITABLE_FAIL"
    if run_role == "primary":
        decision = "AWAITING_CONFIRMATION_AND_POST_RESULT_AUDIT" if execution_pass else "NO-GO"
    elif run_role == "confirmation":
        decision = "AWAITING_RECONCILIATION_AND_POST_RESULT_AUDIT" if execution_pass else "NO-GO"
    summary = {
        "schema_version": 3,
        "stage": "phase2_gpu_training",
        "run_role": run_role,
        "claim_boundary": "frozen_planted_stream_only_non_citable_until_primary_gates_and_post_result_audit",
        "elapsed_seconds": time.time() - start,
        "per_window_rows": len(per_window),
        "per_seed_rows": len(per_seed),
        "all_required_pass": all_required_pass,
        "execution_pass": execution_pass,
        "decision": decision,
        "required_gate_names": required_gate_names,
        "gates": gates,
    }
    _json_dump(output_dir / "summary.json", summary)
    manifest = {
        "schema_version": 3,
        "stage": "phase2_gpu_training",
        "run_role": run_role,
        "run_id": output_dir.name,
        "ordered_seeds": [int(seed) for seed in seeds],
        "calibration_freeze_sha256": freeze_record_sha256,
        "confirmation_of_manifest_sha256": confirmation_of_manifest_sha256,
        "protocol_sha256": config.protocol_sha256,
        "source_hashes": run_source_hashes,
        "test_hash": run_source_hashes.get("test_phase2_gpu_bridge.py"),
        "frozen_config_sha256": config.sha256(),
        "config_sha256": artifacts.sha256_file(output_dir / "config.json"),
        "split_manifest_sha256": artifacts.sha256_file(output_dir / "split_manifest.json"),
        "gates_sha256": artifacts.sha256_file(output_dir / "gates.json"),
        "traces_sha256": artifacts.sha256_file(output_dir / "traces.json"),
        "base_sha256": base.sha256(),
        "data_sha256_by_seed": {seed: item["data_sha256"] for seed, item in split_manifest.items()},
        "split_sha256_by_seed": {seed: item["split_sha256"] for seed, item in split_manifest.items()},
        "runtime": runtime_metadata,
        "resource_ledger": {"replay_bytes": REPLAY_BYTES, "gpu_used": torch_device.type == "cuda"},
        "hash_schemas": {
            "core_arrays": "phase2_core_v1_uppercase_sha256",
            "artifact_arrays": "phase2_artifact_v1_lowercase_sha256",
            "files": "raw_bytes_sha256",
        },
        "output_hashes": {path.name: artifacts.sha256_file(path) for path in output_dir.iterdir() if path.name != "manifest.json"},
        "claim_boundary": summary["claim_boundary"],
    }
    _json_dump(output_dir / "manifest.json", manifest)
    result = {
        "output_dir": str(output_dir),
        "rows": len(per_window),
        "all_required_pass": all_required_pass,
        "execution_pass": execution_pass,
        "decision": decision,
        "gates": gates,
    }
    _ACTIVE_RUN_STATE = None
    return result


def _retain_failure_artifacts(
    output_dir: Path,
    *,
    run_role: str,
    mode: str,
    requested_seeds: str | None,
    device: str,
    exc: BaseException,
    output_preexisting_nonempty: bool,
) -> Path | None:
    """Retain an auditable failure without touching a pre-existing run."""

    if output_preexisting_nonempty:
        return None
    output_dir = output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    state = _ACTIVE_RUN_STATE or {}
    if state and Path(str(state.get("output_dir", ""))).resolve() != output_dir:
        state = {}
    partial_specs = (
        ("per_window", "partial_per_window.csv"),
        ("per_seed", "partial_per_seed.csv"),
        ("resource_rows", "partial_resource_ledger.csv"),
        ("transition_rows", "partial_transitions.csv"),
    )
    for key, name in partial_specs:
        rows = state.get(key, [])
        if rows:
            artifacts.write_csv(output_dir / name, rows, list(rows[0].keys()))
    partial_context = {
        "split_manifest": state.get("split_manifest", {}),
        "factor_manifest": state.get("factor_manifest", {}),
    }
    if partial_context["split_manifest"] or partial_context["factor_manifest"]:
        artifacts.write_json(output_dir / "partial_context.json", partial_context)

    config = BridgeConfig()
    progress = dict(state.get("progress", {}))
    progress.update(
        {
            "per_window_rows": len(state.get("per_window", [])),
            "per_seed_rows": len(state.get("per_seed", [])),
            "resource_rows": len(state.get("resource_rows", [])),
            "transition_rows": len(state.get("transition_rows", [])),
        }
    )
    source_hash_error = None
    try:
        fallback_source_hashes = _source_hashes()
    except Exception as source_exc:
        fallback_source_hashes = {}
        source_hash_error = f"{type(source_exc).__name__}: {source_exc}"
    fallback_identity = {
        "protocol_sha256": config.protocol_sha256,
        "frozen_config_sha256": config.sha256(),
        "source_hashes": fallback_source_hashes,
        "test_sha256": fallback_source_hashes.get("test_phase2_gpu_bridge.py"),
    }
    state_identity = state.get("identity")
    run_identity = dict(state_identity) if isinstance(state_identity, Mapping) else fallback_identity
    identity = {
        **run_identity,
        "requested_seeds_argument": requested_seeds,
        "requested_seeds": state.get("requested_seeds"),
        "device": device,
    }
    failure_payload = {
        "schema_version": 1,
        "kind": "phase2_failure",
        "stage": state.get("stage", "phase2_run_dispatch"),
        "run_role": run_role,
        "mode": mode,
        "status": "failed",
        "decision": "NO-GO",
        "positive_claim_blocked": True,
        "phase": progress.get("phase", "dispatch"),
        "exception": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
        "progress": progress,
        "identity": identity,
        "source_hash_error": source_hash_error,
        "runtime": dict(state.get("runtime")) if isinstance(state.get("runtime"), Mapping) else artifacts.base_runtime_metadata(),
        "all_required_pass": False,
        "execution_pass": False,
    }
    artifacts.write_json(output_dir / "failure.json", failure_payload)
    partial_hashes = {
        path.name: artifacts.sha256_file(path)
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    failure_manifest = {
        "schema_version": 1,
        "kind": "phase2_failure_manifest",
        "complete": False,
        "run_role": run_role,
        "stage": failure_payload["stage"],
        "protocol_sha256": identity["protocol_sha256"],
        "frozen_config_sha256": identity["frozen_config_sha256"],
        "source_hashes": identity["source_hashes"],
        "failure_sha256": partial_hashes["failure.json"],
        "partial_output_hashes": partial_hashes,
        "claim_boundary": "failed_non_citable_positive_claim_blocked",
    }
    artifacts.write_json(output_dir / "manifest.json", failure_manifest)
    return output_dir / "failure.json"


def main(argv: Sequence[str] | None = None) -> int:
    global _ACTIVE_RUN_STATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--run-role",
        default="local_cpu_smoke",
        choices=("local_cpu_smoke", "remote_gpu_smoke", "non_citable_calibration", "primary", "confirmation"),
    )
    parser.add_argument("--mode", choices=("smoke", "gpu", "freeze", "reconcile"), default="smoke")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confirmation-of", type=Path, default=None)
    parser.add_argument("--primary-dir", type=Path, default=None)
    parser.add_argument("--confirmation-dir", type=Path, default=None)
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument("--local-smoke-dir", type=Path, default=None)
    parser.add_argument("--remote-smoke-dir", type=Path, default=None)
    parser.add_argument("--remote-test-result", type=Path, default=None)
    parser.add_argument("--local-validation-report", type=Path, default=None)
    parser.add_argument("--remote-validation-report", type=Path, default=None)
    parser.add_argument("--crosscheck-dir", type=Path, default=None)
    parser.add_argument(
        "--freeze-marker",
        type=Path,
        default=Path("notes/phase2_calibration_freeze.json"),
    )
    args = parser.parse_args(argv)
    if args.mode != "freeze" and args.output_dir is None:
        parser.error("--output-dir is required for smoke, gpu, and reconcile modes")
    output_preexisting_nonempty = bool(
        args.output_dir is not None
        and args.output_dir.exists()
        and (not args.output_dir.is_dir() or any(args.output_dir.iterdir()))
    )
    try:
        if args.mode == "freeze":
            required_evidence = {
                "--calibration-dir": args.calibration_dir,
                "--local-smoke-dir": args.local_smoke_dir,
                "--remote-smoke-dir": args.remote_smoke_dir,
                "--remote-test-result": args.remote_test_result,
                "--local-validation-report": args.local_validation_report,
                "--remote-validation-report": args.remote_validation_report,
                "--crosscheck-dir": args.crosscheck_dir,
            }
            missing = [name for name, value in required_evidence.items() if value is None]
            if missing:
                raise ValueError(f"freeze mode requires {', '.join(missing)}")
            result = create_calibration_freeze(
                calibration_dir=args.calibration_dir,
                local_smoke_dir=args.local_smoke_dir,
                remote_smoke_dir=args.remote_smoke_dir,
                remote_test_result_path=args.remote_test_result,
                local_validation_report_path=args.local_validation_report,
                remote_validation_report_path=args.remote_validation_report,
                crosscheck_dir=args.crosscheck_dir,
                marker_path=args.freeze_marker,
                root=ROOT,
                config=BridgeConfig(),
                source_hashes=_source_hashes(),
            )
        elif args.mode == "reconcile":
            if args.primary_dir is None or args.confirmation_dir is None:
                raise ValueError("reconcile mode requires --primary-dir and --confirmation-dir")
            result = reconcile_runs(args.primary_dir, args.confirmation_dir, args.output_dir)
        elif args.mode == "smoke":
            result = run_numpy_smoke(args.output_dir, args.run_role)
        else:
            if args.run_role == "remote_gpu_smoke":
                default = BridgeConfig().calibration_seeds[:1]
            elif args.run_role == "non_citable_calibration":
                default = BridgeConfig().calibration_seeds
            else:
                default = BridgeConfig().primary_seeds
            result = run_gpu(
                args.output_dir,
                _parse_seeds(args.seeds, default),
                args.run_role,
                args.device,
                confirmation_of=args.confirmation_of,
            )
    except (Exception, KeyboardInterrupt) as exc:
        if args.mode not in {"freeze", "reconcile"} and args.output_dir is not None:
            try:
                failure_path = _retain_failure_artifacts(
                    args.output_dir,
                    run_role=args.run_role,
                    mode=args.mode,
                    requested_seeds=args.seeds,
                    device=args.device,
                    exc=exc,
                    output_preexisting_nonempty=output_preexisting_nonempty,
                )
                if failure_path is not None:
                    print(f"failure artifact: {failure_path}", file=sys.stderr)
            except Exception as artifact_exc:
                print(f"failure artifact write also failed: {artifact_exc}", file=sys.stderr)
        _ACTIVE_RUN_STATE = None
        print(f"phase2 run failed: {exc}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if args.mode == "freeze":
        return 0 if result.get("valid") is True and result.get("frozen_for_primary") is True else 1
    if args.mode == "reconcile":
        return 0 if result.get("decision") == "GO_PENDING_POST_RESULT_AUDIT" else 1
    if not result.get("execution_pass", result.get("all_required_pass", False)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
