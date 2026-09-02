"""Aggregation for the locked Phase-2I five-arm headroom panel.

The evaluator writes score fractions in ``[0, 1]``.  This module refuses to
aggregate incomplete arms and keeps the task-level lower-tail metric separate
from the example-level CVaR objective used during training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    from .metrics import normalized_retention
except ImportError:  # Allow dependency-light direct execution in tests.
    from metrics import normalized_retention


LOCKED_ARMS = (
    "fixed_er",
    "mean_rank_er",
    "oracle_dro_rank",
    "cvar_uniform",
    "cvar_rank",
)


@dataclass(frozen=True)
class ArmSummary:
    arm: str
    task_names: tuple[str, ...]
    base_scores: tuple[float, ...]
    immediate_scores: tuple[float, ...]
    final_scores: tuple[float, ...]
    acquisition_gains: tuple[float, ...]
    normalized_retention: tuple[float, ...]
    mean_final: float
    mean_bwt: float
    min_retention: float
    worst_two_retention: float
    lower_tail_retention: float


def _vector(name: str, values: Sequence[float], n: int | None = None) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    if out.ndim != 1 or out.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if n is not None and out.size != n:
        raise ValueError(f"{name} has {out.size} entries, expected {n}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values")
    if ((out < 0.0) | (out > 1.0)).any():
        raise ValueError(f"{name} must contain score fractions in [0, 1]")
    return out


def lower_tail_mean(values: Sequence[float], alpha: float = 0.2) -> float:
    """Mean of the worst ``alpha`` mass, with the exact fractional boundary."""

    x = np.sort(np.asarray(values, dtype=np.float64))
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("values must be a finite non-empty vector")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    mass = alpha * x.size
    full = int(np.floor(mass))
    frac = mass - full
    total = float(x[:full].sum())
    if frac > 0.0:
        total += frac * float(x[full])
    return total / mass


def summarize_arm(
    *,
    arm: str,
    task_names: Sequence[str],
    base_scores: Sequence[float],
    immediate_scores: Sequence[float],
    final_scores: Sequence[float],
    tail_alpha: float = 0.2,
) -> ArmSummary:
    if arm not in LOCKED_ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    names = tuple(task_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("task_names must be non-empty and unique")
    base = _vector("base_scores", base_scores, len(names))
    immediate = _vector("immediate_scores", immediate_scores, len(names))
    final = _vector("final_scores", final_scores, len(names))
    retention = normalized_retention(base, immediate, final)
    worst = np.sort(retention)
    return ArmSummary(
        arm=arm,
        task_names=names,
        base_scores=tuple(float(x) for x in base),
        immediate_scores=tuple(float(x) for x in immediate),
        final_scores=tuple(float(x) for x in final),
        acquisition_gains=tuple(float(x) for x in immediate - base),
        normalized_retention=tuple(float(x) for x in retention),
        mean_final=float(final.mean()),
        mean_bwt=float((final - immediate).mean()),
        min_retention=float(worst[0]),
        worst_two_retention=float(worst[: min(2, len(worst))].mean()),
        lower_tail_retention=lower_tail_mean(retention, tail_alpha),
    )


def aggregate_completed_run(run_dir: str | Path) -> dict[str, ArmSummary]:
    """Load the five arm result files only after every completion sentinel exists."""

    root = Path(run_dir)
    summaries: dict[str, ArmSummary] = {}
    missing: list[str] = []
    for arm in LOCKED_ARMS:
        arm_dir = root / arm
        result_path = arm_dir / "result.json"
        complete_path = arm_dir / "COMPLETED"
        if not result_path.is_file() or not complete_path.is_file():
            missing.append(arm)
            continue
        with result_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("status") != "completed":
            raise ValueError(f"{result_path} is not marked completed")
        summaries[arm] = summarize_arm(
            arm=arm,
            task_names=payload["task_names"],
            base_scores=payload["base_scores"],
            immediate_scores=payload["immediate_scores"],
            final_scores=payload["final_scores"],
            tail_alpha=float(payload.get("task_tail_alpha", 0.2)),
        )
    if missing:
        raise RuntimeError("incomplete five-arm run; missing: " + ", ".join(missing))
    return summaries


def write_summary(run_dir: str | Path, summaries: Mapping[str, ArmSummary]) -> Path:
    if tuple(summaries.keys()) != LOCKED_ARMS:
        raise ValueError("summaries must be supplied in the locked arm order")
    out = Path(run_dir) / "panel_summary.json"
    payload = {arm: asdict(summaries[arm]) for arm in LOCKED_ARMS}
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return out

