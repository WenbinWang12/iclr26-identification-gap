"""Self-Confidence Gate (SCG): Batch Calibration gated on its own correction size.

Frozen protocol: notes/phase2u_scg_protocol.md
SHA 892ef1a8bb68bf1d044da031ae092fca73122c8c7ba77d2c8bd79486669e4426

The rule is three lines (§1).  The gate statistic is ``|mean(s)|`` -- the magnitude
of the move BC is proposing -- not a property of the batch's spread.  Phase-2T
failed by gating on the latter; §A1.3-A1.4 there records why.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Development choice on seeds 1-3 pooled (210 batches); see protocol §2.
# Smallest grid value that is harmful-free on each development seed separately.
TAU_M = 1.4


def margin(Z: np.ndarray) -> np.ndarray:
    """``s = z_1 - z_0``; gauge invariant by construction."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[1] != 2:
        raise ValueError(f"SCG is defined for K_S == 2, got shape {Z.shape}")
    return Z[:, 1] - Z[:, 0]


def bc_offset(Z: np.ndarray) -> np.ndarray:
    """Batch Calibration (Zhou et al., ICLR 2024), offset side, unchanged."""
    return np.array([0.0, -float(margin(Z).mean())])


@dataclass(frozen=True)
class SCGResult:
    offset: np.ndarray
    applied: bool
    abs_m: float
    tau_m: float
    reason: str


def scg_offset(Z: np.ndarray, *, tau_m: float = TAU_M) -> SCGResult:
    """BC, applied only when its own proposed correction is large enough."""
    m = float(margin(Z).mean())
    if abs(m) >= tau_m:
        return SCGResult(np.array([0.0, -m]), True, abs(m), tau_m, "applied")
    # §5: exact zero, so argmax is bit-identical to the raw prediction.
    return SCGResult(np.zeros(2), False, abs(m), tau_m,
                     f"|mean(s)| {abs(m):.3f} < tau {tau_m}")


def scg_offset_normalised(Z: np.ndarray, *, tau: float) -> SCGResult:
    """U6's dimensionless variant: gate on ``|mean(s)| / sd(s)`` instead.

    Reported for comparison only.  It scored worse on development
    (rho with BC's gain +0.32 against +0.61), which is why §2 chose logit units
    despite the scale dependence that choice carries.
    """
    s = margin(Z)
    m = float(s.mean())
    sd = float(s.std(ddof=1)) if s.size > 1 else 0.0
    stat = abs(m) / sd if sd > 0 else 0.0
    if stat >= tau:
        return SCGResult(np.array([0.0, -m]), True, stat, tau, "applied")
    return SCGResult(np.zeros(2), False, stat, tau, f"|m|/sd {stat:.3f} < tau {tau}")


def soft_offset(Z: np.ndarray, *, tau: float) -> np.ndarray:
    """Soft-threshold shrinkage, kept so §2's rejection stays reproducible.

    Rejected on measurement: dominated by the hard gate on both mean and tail
    across all three development seeds.
    """
    m = float(margin(Z).mean())
    return np.array([0.0, -np.sign(m) * max(abs(m) - tau, 0.0)])


def oracle_offset(Z: np.ndarray, y: np.ndarray, *, grid: int = 401) -> np.ndarray:
    """Best single threshold by balanced accuracy.  Uses labels; headroom only."""
    s = margin(Z)
    y = np.asarray(y)
    lo, hi = float(s.min()), float(s.max())
    span = max(hi - lo, 1e-9)
    best_t, best_v = 0.0, -1.0
    for t in np.linspace(lo - 0.05 * span, hi + 0.05 * span, grid):
        pred = (s > t).astype(int)
        rec = [float((pred[y == k] == k).mean()) for k in (0, 1) if (y == k).any()]
        v = float(np.mean(rec)) if rec else -1.0
        if v > best_v:
            best_v, best_t = v, float(t)
    return np.array([0.0, -best_t])


def balanced_accuracy(Z: np.ndarray, y: np.ndarray,
                      offset: np.ndarray | None = None) -> float:
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y)
    if offset is not None:
        Z = Z + np.asarray(offset, dtype=np.float64)[None, :]
    pred = Z.argmax(axis=1)
    rec = [float((pred[y == k] == k).mean())
           for k in range(Z.shape[1]) if (y == k).any()]
    return float(np.mean(rec)) if rec else float("nan")
