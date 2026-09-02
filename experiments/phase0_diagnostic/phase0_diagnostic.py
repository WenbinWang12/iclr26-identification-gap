"""Small exact diagnostics before any neural continual-PEFT experiment.

The compatible stream is a matrix-sensing/Kaczmarz negative control.  The
full-observation stream is a conflict and acquisition-matching control.  The
module intentionally uses NumPy only: no optimizer or model checkpoint is
needed for these probes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EPS = 1e-12
NUMERICAL_GAP_TOL = 1e-10
PINV_RCOND = 1e-12
BOOTSTRAP_SEED = 91
BOOTSTRAP_DRAWS = 20_000


@dataclass(frozen=True)
class CompatibleConfig:
    d_out: int = 6
    d_in: int = 12
    true_rank: int = 2
    latent_dim: int = 6
    windows: int = 48
    correlation: float = 0.35


@dataclass(frozen=True)
class MatrixConfig:
    dimension: int = 8
    rank: int = 2
    windows: int = 12
    amplitude: float = 1.0


def _orthonormal(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    if cols > rows:
        raise ValueError("cannot request more orthonormal columns than rows")
    q, _ = np.linalg.qr(rng.normal(size=(rows, cols)))
    return q[:, :cols]


def make_truth(config: CompatibleConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 100_003)
    left = _orthonormal(rng, config.d_out, config.true_rank)
    right = _orthonormal(rng, config.d_in, config.true_rank)
    singular = np.linspace(2.0, 1.0, config.true_rank)
    return left @ np.diag(singular) @ right.T


def make_directions(
    config: CompatibleConfig, seed: int, regime: str
) -> np.ndarray:
    if regime not in {"orthogonal", "correlated"}:
        raise ValueError(f"unknown compatible regime: {regime}")
    rng = np.random.default_rng(seed + 200_003)
    basis = _orthonormal(rng, config.d_in, config.latent_dim)
    vectors: list[np.ndarray] = []
    for t in range(config.windows):
        z = np.zeros(config.latent_dim)
        if regime == "orthogonal":
            z[t % config.latent_dim] = 1.0
        else:
            # A rotating, correlated sequence.  It is deterministic given the
            # seed and has a nonzero current innovation at every window.
            z[0] = 1.0
            j = 1 + (t % (config.latent_dim - 1))
            z[j] = config.correlation * ((-1.0) ** (t // (config.latent_dim - 1)))
        v = basis @ z
        v /= np.linalg.norm(v)
        vectors.append(v)
    return np.stack(vectors)


def project_current(
    state: np.ndarray, target: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Minimum-change rank-one correction for one normalized direction."""
    residual = target @ direction - state @ direction
    return state + np.outer(residual, direction)


def truncate_rank(matrix: np.ndarray, rank: int | None) -> np.ndarray:
    if rank is None:
        return matrix.copy()
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    keep = min(rank, singular.size)
    if keep == 0:
        return np.zeros_like(matrix)
    return (u[:, :keep] * singular[:keep]) @ vt[:keep, :]


def sequential_compatible(
    target: np.ndarray,
    directions: np.ndarray,
    rank_cap: int | None,
    initial: np.ndarray | None = None,
) -> list[np.ndarray]:
    state = np.zeros_like(target) if initial is None else initial.copy()
    states: list[np.ndarray] = []
    for direction in directions:
        state = project_current(state, target, direction)
        if rank_cap is not None:
            state = truncate_rank(state, rank_cap)
        states.append(state.copy())
    return states


def compatible_offline(
    target: np.ndarray, directions: np.ndarray, rank_cap: int | None
) -> list[np.ndarray]:
    """Prefix least-squares solutions from all observations seen so far."""
    states: list[np.ndarray] = []
    observations = target @ directions.T
    for t in range(1, directions.shape[0] + 1):
        v = directions[:t].T
        y = observations[:, :t]
        least_squares = y @ np.linalg.pinv(v, rcond=PINV_RCOND)
        states.append(truncate_rank(least_squares, rank_cap))
    return states


def measurement_loss(
    state: np.ndarray, target: np.ndarray, directions: np.ndarray
) -> float:
    errors = (state - target) @ directions.T
    return float(0.5 * np.mean(np.sum(errors * errors, axis=0)))


def compatible_rows(
    config: CompatibleConfig, seed: int, regime: str
) -> list[dict[str, Any]]:
    target = make_truth(config, seed)
    directions = make_directions(config, seed, regime)
    observations = target @ directions.T
    seq = sequential_compatible(target, directions, config.true_rank)
    dense = sequential_compatible(target, directions, None)
    offline = compatible_offline(target, directions, config.true_rank)
    high = compatible_offline(target, directions, None)
    replay = offline
    zero = np.zeros_like(target)
    rows: list[dict[str, Any]] = []

    for t, state in enumerate(seq, start=1):
        prefix_directions = directions[:t]
        base_loss = measurement_loss(zero, target, prefix_directions)
        high_state = high[t - 1]
        offline_state = offline[t - 1]
        high_loss = measurement_loss(high_state, target, prefix_directions)
        offline_loss = measurement_loss(offline_state, target, prefix_directions)
        seq_loss = measurement_loss(state, target, prefix_directions)
        replay_loss = measurement_loss(replay[t - 1], target, prefix_directions)
        current_error = state @ directions[t - 1] - observations[:, t - 1]
        current_base = 0.5 * float(np.sum(observations[:, t - 1] ** 2))
        past_loss = (
            measurement_loss(state, target, directions[: t - 1]) if t > 1 else 0.0
        )
        past_base = (
            measurement_loss(zero, target, directions[: t - 1]) if t > 1 else 0.0
        )
        denominator = base_loss - high_loss
        eligible = denominator > EPS * max(base_loss, 1.0)
        dense_loss = measurement_loss(dense[t - 1], target, prefix_directions)
        current_gap = seq_loss - offline_loss
        addressable_raw = seq_loss - replay_loss
        rank_specific_raw = seq_loss - dense_loss
        adjacent_dot = (
            float(np.dot(directions[t - 2], directions[t - 1])) if t > 1 else None
        )
        rows.append(
            {
                "family": "compatible",
                "regime": regime,
                "seed": seed,
                "prefix": t,
                "base_loss": base_loss,
                "high_loss": high_loss,
                "offline_rank_loss": offline_loss,
                "seq_loss": seq_loss,
                "replay_loss": replay_loss,
                "dense_loss": dense_loss,
                "capacity_ratio_raw": _ratio_raw(offline_loss - high_loss, denominator),
                "total_online_gap_raw": _ratio_raw(current_gap, denominator),
                "addressable_gap_raw": _ratio_raw(addressable_raw, denominator),
                "rank_specific_gap_raw": _ratio_raw(rank_specific_raw, denominator),
                "capacity_ratio": _gap_ratio(offline_loss - high_loss, denominator),
                "total_online_gap": _gap_ratio(current_gap, denominator),
                "addressable_gap": _gap_ratio(addressable_raw, denominator),
                "rank_specific_gap": _gap_ratio(rank_specific_raw, denominator),
                "acquisition_loss": float(0.5 * np.sum(current_error**2)),
                "acquisition_ratio": _ratio(
                    float(0.5 * np.sum(current_error**2)), current_base
                ),
                "past_loss": past_loss,
                "past_base_loss": past_base,
                "past_retention_ratio": _ratio(past_loss, past_base),
                "rank_seq": int(np.linalg.matrix_rank(state, tol=1e-9)),
                "rank_dense": int(np.linalg.matrix_rank(dense[t - 1], tol=1e-9)),
                "rank_offline": int(np.linalg.matrix_rank(offline_state, tol=1e-9)),
                "eligible": bool(eligible),
                "direction_energy": float(np.sum(observations[:, t - 1] ** 2)),
                "direction_norm": float(np.linalg.norm(directions[t - 1])),
                "adjacent_direction_dot": adjacent_dot,
                "replay_independent": False,
                "addressability_applicable": False,
                "acquisition_matched_gap": None,
            }
        )
    return rows


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= EPS:
        return None
    return float(numerator / denominator)


def _ratio_raw(numerator: float, denominator: float) -> float | None:
    """Return a raw normalized value without numerical gap clamping."""
    return _ratio(numerator, denominator)


def _gap_ratio(numerator: float, denominator: float) -> float | None:
    """Clamp only machine-scale gap noise while retaining meaningful signs."""
    if denominator <= EPS:
        return None
    scale = max(abs(denominator), 1.0)
    adjusted = 0.0 if abs(numerator) <= NUMERICAL_GAP_TOL * scale else numerator
    return float(adjusted / denominator)


def _matrix_basis(
    config: MatrixConfig, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 300_003)
    left = _orthonormal(rng, config.dimension, config.dimension)
    right = _orthonormal(rng, config.dimension, config.dimension)
    base = np.zeros((config.dimension, config.dimension))
    base[: config.rank - 1, : config.rank - 1] = np.eye(config.rank - 1)
    transient = np.zeros_like(base)
    transient[config.rank - 1, config.rank - 1] = config.amplitude
    return left, right, base, transient


def make_matrix_targets(
    config: MatrixConfig, seed: int, regime: str
) -> list[np.ndarray]:
    if regime not in {"stationary", "canceling", "inadequate"}:
        raise ValueError(f"unknown matrix regime: {regime}")
    left, right, base, transient = _matrix_basis(config, seed)
    targets: list[np.ndarray] = []
    for t in range(config.windows):
        if regime == "stationary":
            coeff = 0.0
        elif regime == "canceling":
            coeff = 1.0 if t % 2 == 0 else -1.0
        else:
            coeff = 1.0
        diagonal = base + coeff * transient
        if regime == "inadequate":
            extra = np.zeros_like(diagonal)
            extra[config.rank, config.rank] = config.amplitude
            diagonal = diagonal + extra
        targets.append(left @ diagonal @ right.T)
    return targets


def matrix_rows(
    config: MatrixConfig, seed: int, regime: str
) -> list[dict[str, Any]]:
    targets = make_matrix_targets(config, seed, regime)
    zero = np.zeros_like(targets[0])
    rows: list[dict[str, Any]] = []
    for t, current in enumerate(targets, start=1):
        prefix = targets[:t]
        mean = np.mean(np.stack(prefix), axis=0)
        high = mean
        offline = truncate_rank(mean, config.rank)
        seq = truncate_rank(current, config.rank)
        dense = current.copy()
        base_loss = float(np.mean([0.5 * np.sum((zero - x) ** 2) for x in prefix]))
        high_loss = float(np.mean([0.5 * np.sum((high - x) ** 2) for x in prefix]))
        offline_loss = float(
            np.mean([0.5 * np.sum((offline - x) ** 2) for x in prefix])
        )
        seq_loss = float(np.mean([0.5 * np.sum((seq - x) ** 2) for x in prefix]))
        dense_loss = float(np.mean([0.5 * np.sum((dense - x) ** 2) for x in prefix]))
        current_loss = 0.5 * float(np.sum((seq - current) ** 2))
        denominator = base_loss - high_loss
        replay_loss = offline_loss
        current_gap = seq_loss - offline_loss
        addressable_raw = seq_loss - replay_loss
        rank_specific_raw = seq_loss - dense_loss
        past_loss = (
            float(np.mean([0.5 * np.sum((seq - x) ** 2) for x in prefix[:-1]]))
            if t > 1
            else 0.0
        )
        past_base = (
            float(np.mean([0.5 * np.sum(x**2) for x in prefix[:-1]]))
            if t > 1
            else 0.0
        )
        rows.append(
            {
                "family": "full_observation",
                "regime": regime,
                "seed": seed,
                "prefix": t,
                "base_loss": base_loss,
                "high_loss": high_loss,
                "offline_rank_loss": offline_loss,
                "seq_loss": seq_loss,
                "replay_loss": replay_loss,
                "dense_loss": dense_loss,
                "capacity_ratio_raw": _ratio_raw(offline_loss - high_loss, denominator),
                "total_online_gap_raw": _ratio_raw(current_gap, denominator),
                "addressable_gap_raw": _ratio_raw(addressable_raw, denominator),
                "rank_specific_gap_raw": _ratio_raw(rank_specific_raw, denominator),
                "capacity_ratio": _gap_ratio(offline_loss - high_loss, denominator),
                "total_online_gap": _gap_ratio(current_gap, denominator),
                "addressable_gap": _gap_ratio(addressable_raw, denominator),
                "rank_specific_gap": _gap_ratio(rank_specific_raw, denominator),
                # For rank-feasible targets, the current target is an
                # acquisition-matched zero-gap comparator.  In the
                # rank-inadequate control, that comparator exposes the real
                # rank-specific acquisition residual instead of pretending it
                # is zero.
                "acquisition_matched_loss": (
                    seq_loss if regime != "inadequate" else dense_loss
                ),
                "acquisition_matched_gap": (
                    0.0
                    if regime != "inadequate"
                    else _gap_ratio(rank_specific_raw, denominator)
                ),
                "acquisition_loss": current_loss,
                "acquisition_ratio": _ratio(current_loss, 0.5 * float(np.sum(current**2))),
                "past_loss": past_loss,
                "past_base_loss": past_base,
                "past_retention_ratio": _ratio(past_loss, past_base),
                "rank_seq": int(np.linalg.matrix_rank(seq, tol=1e-9)),
                "rank_dense": int(np.linalg.matrix_rank(dense, tol=1e-9)),
                "rank_offline": int(np.linalg.matrix_rank(offline, tol=1e-9)),
                "eligible": bool(denominator > EPS * max(base_loss, 1.0)),
                "direction_energy": float(np.sum(current**2)),
                "direction_norm": float(np.linalg.norm(current)),
                "adjacent_direction_dot": None,
                "replay_independent": False,
                "addressability_applicable": False,
            }
        )
    return rows


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["regime"]), int(row["seed"]))
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (family, regime, seed), values in sorted(grouped.items()):
        eligible = [x for x in values if x.get("eligible")]
        gap = [x["total_online_gap"] for x in eligible if x["total_online_gap"] is not None]
        addressable = [x["addressable_gap"] for x in eligible if x["addressable_gap"] is not None]
        rank_specific = [
            x["rank_specific_gap"]
            for x in eligible
            if x["rank_specific_gap"] is not None
        ]
        acquisition_matched = [
            x["acquisition_matched_gap"]
            for x in eligible
            if x.get("acquisition_matched_gap") is not None
        ]
        capacity = [x["capacity_ratio"] for x in eligible if x["capacity_ratio"] is not None]
        acquisition = [x["acquisition_ratio"] for x in eligible if x["acquisition_ratio"] is not None]
        summaries.append(
            {
                "family": family,
                "regime": regime,
                "seed": seed,
                "eligible_prefixes": len(eligible),
                # Prefixes are unit-spaced.  The normalized left-step area
                # equals this mean; retain the old *_auc key for consumers.
                "total_online_gap_prefix_mean": _mean(gap),
                "total_online_gap_auc": _mean(gap),
                "addressable_gap_prefix_mean": _mean(addressable),
                "addressable_gap_auc": _mean(addressable),
                "rank_specific_gap_prefix_mean": _mean(rank_specific),
                "rank_specific_gap_auc": _mean(rank_specific),
                "acquisition_matched_gap_prefix_mean": _mean(acquisition_matched),
                "capacity_ratio_median": _median(capacity),
                "capacity_ratio_max": max(capacity, default=None),
                "acquisition_ratio_median": _median(acquisition),
                "acquisition_ratio_max": max(acquisition, default=None),
                "rank_seq_max": max((x["rank_seq"] for x in values), default=0),
                "rank_dense_max": max((x["rank_dense"] for x in values), default=0),
            }
        )
    return {
        "metric_convention": {
            "prefix_summary": "mean over eligible unit-width prefixes",
            "total_online_gap_auc": "normalized left-step area; equal to prefix mean",
            "bootstrap_unit": "independent seeds unless explicitly marked paired",
            "numerical_gap_tolerance": NUMERICAL_GAP_TOL,
        },
        "per_seed": summaries,
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def bootstrap_mean_ci(
    values: list[float], seed: int = BOOTSTRAP_SEED, draws: int = BOOTSTRAP_DRAWS
) -> list[float] | None:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    samples = rng.choice(np.asarray(values, dtype=float), size=(draws, len(values)), replace=True)
    ci = np.quantile(np.mean(samples, axis=1), [0.025, 0.975])
    return [float(ci[0]), float(ci[1])]


def bootstrap_paired_ci(
    left: list[float],
    right: list[float],
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
) -> list[float] | None:
    """Bootstrap a paired seed-wise contrast, preserving seed matching."""
    if len(left) != len(right) or not left:
        return None
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(draws, len(differences)))
    samples = differences[indices]
    ci = np.quantile(np.mean(samples, axis=1), [0.025, 0.975])
    return [float(ci[0]), float(ci[1])]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    output_dir: Path, seeds: list[int], run_role: str = "unspecified"
) -> dict[str, Any]:
    started = time.time()
    compatible = CompatibleConfig()
    matrix = MatrixConfig()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for regime in ("orthogonal", "correlated"):
            rows.extend(compatible_rows(compatible, seed, regime))
        for regime in ("stationary", "canceling", "inadequate"):
            rows.extend(matrix_rows(matrix, seed, regime))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_prefix.csv"
    write_csv(csv_path, rows)
    summary = summarize(rows)
    summary["run_role"] = run_role
    summary["bootstrap_by_regime"] = {}
    for family in ("compatible", "full_observation"):
        for regime in ("orthogonal", "correlated", "stationary", "canceling", "inadequate"):
            values = [
                x["total_online_gap_auc"]
                for x in summary["per_seed"]
                if x["family"] == family and x["regime"] == regime and x["total_online_gap_auc"] is not None
            ]
            if values:
                summary["bootstrap_by_regime"][f"{family}:{regime}"] = {
                    "seed_values": values,
                    "mean": float(np.mean(values)),
                    "ci95": bootstrap_mean_ci(values),
                    "resampling_unit": "independent seeds",
                    "draws": BOOTSTRAP_DRAWS,
                }
    corr = _seed_metric(summary, "compatible", "correlated", "total_online_gap_auc")
    orth = _seed_metric(summary, "compatible", "orthogonal", "total_online_gap_auc")
    paired_seeds = sorted(set(corr) & set(orth))
    paired_left = [corr[seed] for seed in paired_seeds]
    paired_right = [orth[seed] for seed in paired_seeds]
    paired_diff = [left - right for left, right in zip(paired_left, paired_right)]
    summary["paired_bootstrap"] = {
        "compatible:correlated_minus_orthogonal": {
            "seeds": paired_seeds,
            "seed_differences": paired_diff,
            "mean": float(np.mean(paired_diff)) if paired_diff else None,
            "ci95": bootstrap_paired_ci(paired_left, paired_right),
            "resampling_unit": "paired seeds",
            "draws": BOOTSTRAP_DRAWS,
        }
    }
    summary["gates"] = evaluate_gates(rows, summary)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_jsonable), encoding="utf-8"
    )
    input_sha256, direction_stats = _input_fingerprint(compatible, matrix, seeds)
    protocol_path = _project_path("notes", "phase0_pilot_protocol.md")
    test_path = _project_path("experiments", "phase0_diagnostic", "test_phase0.py")
    manifest = {
        "stage": "phase0_diagnostic",
        "run_role": run_role,
        "started_unix": started,
        "finished_unix": time.time(),
        "host": platform.node(),
        "cwd": str(Path.cwd()),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "seeds": seeds,
        "compatible_config": asdict(compatible),
        "matrix_config": asdict(matrix),
        "source_sha256": _source_hash(),
        "protocol_sha256": _optional_hash(protocol_path),
        "test_sha256": _optional_hash(test_path),
        "input_sha256": input_sha256,
        "direction_stats": direction_stats,
        "numerical_settings": {
            "eps": EPS,
            "gap_tolerance": NUMERICAL_GAP_TOL,
            "pinv_rcond": PINV_RCOND,
            "rank_tolerance": 1e-9,
        },
        "git_commit": "NO_GIT",
        "command": [sys.executable, *sys.argv],
        "resource_ledger": {
            "optimizer_state_bytes": 0,
            "compatible_adapter_state_bytes": compatible.d_out * compatible.d_in * 8,
            "compatible_full_replay_bytes": compatible.windows
            * (compatible.d_in + compatible.d_out)
            * 8,
            "compatible_updates_per_seed": compatible.windows,
            "matrix_adapter_state_bytes": matrix.dimension * matrix.dimension * 8,
            "matrix_full_replay_bytes": matrix.windows * matrix.dimension * matrix.dimension * 8,
            "matrix_updates_per_seed": matrix.windows,
            "gpu_used": False,
        },
        "output_sha256": {
            "per_prefix.csv": _file_hash(csv_path),
            "summary.json": _file_hash(summary_path),
        },
        "rows": len(rows),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_jsonable), encoding="utf-8"
    )
    return {"manifest": manifest, "summary": summary, "csv": str(csv_path)}


def _source_hash() -> str:
    source = Path(__file__).read_bytes()
    return hashlib.sha256(source).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _direction_stats(directions: np.ndarray) -> dict[str, Any]:
    adjacent = np.sum(directions[:-1] * directions[1:], axis=1)
    gram = directions @ directions.T
    upper = gram[np.triu_indices_from(gram, k=1)]
    return {
        "count": int(directions.shape[0]),
        "norm_min": float(np.min(np.linalg.norm(directions, axis=1))),
        "norm_max": float(np.max(np.linalg.norm(directions, axis=1))),
        "adjacent_dot_min": float(np.min(adjacent)) if adjacent.size else None,
        "adjacent_dot_mean": float(np.mean(adjacent)) if adjacent.size else None,
        "adjacent_dot_max": float(np.max(adjacent)) if adjacent.size else None,
        "all_pair_dot_min": float(np.min(upper)) if upper.size else None,
        "all_pair_dot_mean": float(np.mean(upper)) if upper.size else None,
        "all_pair_dot_max": float(np.max(upper)) if upper.size else None,
    }


def _input_fingerprint(
    compatible: CompatibleConfig, matrix: MatrixConfig, seeds: list[int]
) -> tuple[str, dict[str, Any]]:
    """Hash generated inputs so a config/seed replay is independently checkable."""
    digest = hashlib.sha256()
    direction_stats: dict[str, list[dict[str, Any]]] = {
        "orthogonal": [],
        "correlated": [],
    }
    for seed in seeds:
        for regime in ("orthogonal", "correlated"):
            target = np.ascontiguousarray(make_truth(compatible, seed), dtype=np.float64)
            directions = np.ascontiguousarray(
                make_directions(compatible, seed, regime), dtype=np.float64
            )
            digest.update(f"compatible:{seed}:{regime}:target".encode("ascii"))
            digest.update(target.tobytes(order="C"))
            digest.update(f"compatible:{seed}:{regime}:directions".encode("ascii"))
            digest.update(directions.tobytes(order="C"))
            stats = _direction_stats(directions)
            stats.update({"seed": seed})
            direction_stats[regime].append(stats)
        for regime in ("stationary", "canceling", "inadequate"):
            targets = make_matrix_targets(matrix, seed, regime)
            digest.update(f"matrix:{seed}:{regime}".encode("ascii"))
            for index, target in enumerate(targets):
                digest.update(f":{index}".encode("ascii"))
                digest.update(np.ascontiguousarray(target, dtype=np.float64).tobytes(order="C"))
    return digest.hexdigest(), direction_stats


def _project_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[2].joinpath(*parts)


def _optional_hash(path: Path) -> str | None:
    return _file_hash(path) if path.exists() else None


def _seed_metric(
    summary: dict[str, Any], family: str, regime: str, metric: str
) -> dict[int, float]:
    return {
        int(row["seed"]): float(row[metric])
        for row in summary["per_seed"]
        if row["family"] == family
        and row["regime"] == regime
        and row.get(metric) is not None
    }


def evaluate_gates(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    """Evaluate frozen diagnostic gates without turning them into method claims."""
    compatible = [x for x in rows if x["family"] == "compatible" and x["eligible"]]
    correlated = [x for x in compatible if x["regime"] == "correlated"]
    orthogonal = [x for x in compatible if x["regime"] == "orthogonal"]
    canceling = [
        x
        for x in rows
        if x["family"] == "full_observation"
        and x["regime"] == "canceling"
        and x["eligible"]
    ]
    inadequate = [
        x
        for x in rows
        if x["family"] == "full_observation"
        and x["regime"] == "inadequate"
        and x["eligible"]
    ]
    capacities = [x["capacity_ratio"] for x in compatible if x["capacity_ratio"] is not None]
    acquisitions = [
        x["acquisition_ratio"]
        for x in correlated
        if x["acquisition_ratio"] is not None
    ]
    corr_by_seed = _seed_metric(summary, "compatible", "correlated", "total_online_gap_auc")
    orth_by_seed = _seed_metric(summary, "compatible", "orthogonal", "total_online_gap_auc")
    paired_seeds = sorted(set(corr_by_seed) & set(orth_by_seed))
    paired_values = [corr_by_seed[s] - orth_by_seed[s] for s in paired_seeds]
    paired_ci = bootstrap_paired_ci(
        [corr_by_seed[s] for s in paired_seeds],
        [orth_by_seed[s] for s in paired_seeds],
    )
    orth_values = list(orth_by_seed.values())
    rank_specific = [
        abs(x["rank_specific_gap"])
        for x in compatible
        if x["rank_specific_gap"] is not None
    ]
    cancel_raw_positive = any(
        (x.get("total_online_gap_raw") or 0.0) > NUMERICAL_GAP_TOL for x in canceling
    )
    cancel_match_zero = all(
        abs(x.get("acquisition_matched_gap") or 0.0) <= NUMERICAL_GAP_TOL
        for x in canceling
    )
    addressability_note = (
        "not_independent_on_compatible_stream: replay equals rank-R offline oracle"
    )
    gates: dict[str, Any] = {
        "compatible_capacity": {
            "median": float(np.median(capacities)) if capacities else None,
            "max": max(capacities, default=None),
            "pass": bool(capacities)
            and float(np.median(capacities)) <= 0.10
            and max(capacities) <= 0.20,
        },
        "compatible_acquisition": {
            "fraction_at_or_below_0.05": (
                float(np.mean(np.asarray(acquisitions) <= 0.05)) if acquisitions else None
            ),
            "pass": bool(acquisitions)
            and float(np.mean(np.asarray(acquisitions) <= 0.05)) >= 0.95,
        },
        "compatible_paired_online_signal": {
            "seed_count": len(paired_values),
            "mean": float(np.mean(paired_values)) if paired_values else None,
            "ci95": paired_ci,
            "lower_bound": paired_ci[0] if paired_ci else None,
            "pass": bool(paired_ci) and paired_ci[0] >= 0.10,
            "resampling_unit": "paired seeds",
        },
        "orthogonal_control": {
            "mean": float(np.mean(orth_values)) if orth_values else None,
            "pass": bool(orth_values) and float(np.mean(orth_values)) <= 0.02,
        },
        "canceling_conflict_control": {
            "raw_total_gap_positive": cancel_raw_positive,
            "acquisition_matched_gap_zero": cancel_match_zero,
            "pass": cancel_raw_positive and cancel_match_zero,
        },
        "compatible_rank_specific_null": {
            "max_abs": max(rank_specific, default=None),
            "pass": bool(rank_specific)
            and max(rank_specific) <= 1e-8,
        },
        "inadequate_capacity_positive_control": {
            "median": (
                float(np.median([x["capacity_ratio"] for x in inadequate if x["capacity_ratio"] is not None]))
                if any(x["capacity_ratio"] is not None for x in inadequate)
                else None
            ),
            "pass": bool(inadequate)
            and float(
                np.median([x["capacity_ratio"] for x in inadequate if x["capacity_ratio"] is not None])
            )
            > 0.20,
        },
        "addressability": {
            "status": addressability_note,
            "pass": None,
        },
    }
    required = [
        value["pass"]
        for key, value in gates.items()
        if key != "addressability"
        and isinstance(value.get("pass"), bool)
    ]
    gates["all_required_pass"] = bool(required) and all(required)
    return gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--run-role", default="unspecified")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    result = run_experiment(args.output_dir, seeds, args.run_role)
    print(
        json.dumps(
            {
                "bootstrap_by_regime": result["summary"]["bootstrap_by_regime"],
                "paired_bootstrap": result["summary"]["paired_bootstrap"],
                "gates": result["summary"]["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
