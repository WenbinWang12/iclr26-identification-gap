"""Variance-Gated Calibration (VGC).

Frozen protocol: notes/phase2t_vgc_protocol.md
SHA 7697291743d6a64be7dcef29a70f3d2adcf8d102e6bc004628510a08bb87a849

The rule is three lines (protocol §1).  Everything else in this module exists to
make the anti-artefact checks of §5 mechanically testable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Development-set choice, seed 1 only, frozen before seeds 2/3 were read.
# Protocol §2 records the sweep it came from and names it a development parameter.
TAU_SD = 0.50


def margin(Z: np.ndarray) -> np.ndarray:
    """The gauge-invariant decision statistic ``s = z_1 - z_0``.

    Reads only the difference, so ``margin(Z) == margin(Z + c)`` exactly.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[1] != 2:
        raise ValueError(f"VGC is defined for K_S == 2, got shape {Z.shape}")
    return Z[:, 1] - Z[:, 0]


def bc_offset(Z: np.ndarray) -> np.ndarray:
    """Batch Calibration (Zhou et al., ICLR 2024) on the margin, offset side.

    ``argmax(z + b)`` with ``b = (0, -mean(s))`` is identical to BC's
    ``argmax_y[p(y|x) - p_hat(y)]`` for K = 2 up to the gauge; we keep the
    two-vector form so it composes with ``balanced_accuracy``.
    """
    return np.array([0.0, -float(margin(Z).mean())])


@dataclass(frozen=True)
class VGCResult:
    offset: np.ndarray
    applied: bool
    sd: float
    tau_sd: float
    reason: str


def vgc_offset(Z: np.ndarray, *, tau_sd: float = TAU_SD) -> VGCResult:
    """BC, gated on the spread of the decision statistic.

    Label-free by construction: ``Z`` is the only argument.
    """
    s = margin(Z)
    sd = float(s.std(ddof=1)) if s.size > 1 else 0.0
    if sd >= tau_sd:
        return VGCResult(bc_offset(Z), True, sd, tau_sd, "applied")
    # §5: abstention must be exactly zero, so argmax is bit-identical to raw.
    return VGCResult(np.zeros(2), False, sd, tau_sd, f"sd {sd:.3f} < tau {tau_sd}")


def oracle_offset(Z: np.ndarray, y: np.ndarray, *, grid: int = 401) -> np.ndarray:
    """Best single threshold on ``s`` by balanced accuracy, for headroom only.

    Uses labels and is never part of any deployable arm.
    """
    s = margin(Z)
    y = np.asarray(y)
    lo, hi = float(s.min()), float(s.max())
    span = max(hi - lo, 1e-9)
    best_t, best_v = 0.0, -1.0
    for t in np.linspace(lo - 0.05 * span, hi + 0.05 * span, grid):
        v = _bal_from_margin(s, y, float(t))
        if v > best_v:
            best_v, best_t = v, float(t)
    return np.array([0.0, -best_t])


def _bal_from_margin(s: np.ndarray, y: np.ndarray, thresh: float) -> float:
    pred = (s > thresh).astype(int)
    recalls = [float((pred[y == k] == k).mean()) for k in (0, 1) if (y == k).any()]
    return float(np.mean(recalls)) if recalls else float("nan")


def balanced_accuracy(Z: np.ndarray, y: np.ndarray,
                      offset: np.ndarray | None = None) -> float:
    """Mean per-class recall; same definition as phase2j/offsets.py."""
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y)
    if offset is not None:
        Z = Z + np.asarray(offset, dtype=np.float64)[None, :]
    pred = Z.argmax(axis=1)
    recalls = [float((pred[y == k] == k).mean())
               for k in range(Z.shape[1]) if (y == k).any()]
    return float(np.mean(recalls)) if recalls else float("nan")


def spearman(a, b) -> float:
    """Rank correlation with average ranks for ties; no scipy on the GPU box."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    # average ranks within tie groups
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for idx in np.flatnonzero(counts > 1):
        m = inv == idx
        ranks[m] = ranks[m].mean()
    return ranks


def ols_slope(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc = x - x.mean()
    den = float((xc ** 2).sum())
    return float((xc * (y - y.mean())).sum() / den) if den > 0 else float("nan")
