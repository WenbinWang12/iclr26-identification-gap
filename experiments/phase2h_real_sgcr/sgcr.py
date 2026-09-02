"""Phase-2H SGCR: stable activation cells + robust replay + audit gate.

STATUS: development scaffold.  The clustering, coherence-eligibility, robust
q-update, and audit-gate DECISION logic are pure numpy and unit-testable with
synthetic signatures / CE arrays (no model needed).  The orchestration that
computes real per-example Transformer cross-entropy is driven by run_development
which passes CE arrays in.

Implements notes/phase2h_real_sgcr_protocol.md:

- "Stable activation cells": spherical k-means, K_max=8, deterministic non-empty
  init at the warmup boundary; refresh every ten drift windows from train-store
  signatures only; align refreshed centers to persistent IDs by exact assignment;
  assign audit examples with train-derived centers without moving a center;
  robust-eligible iff train AND audit support and train-only coherence
  >= 0.60 * median_cell_coherence.
- "Actual nonlinear robust replay": q_g <- Normalize(q_g * exp(0.1 * L_g /
  mean_active_loss)) over eligible cells, retain the original total eligible
  mass, cap q_g / q0_g at 4 before renormalizing.
- "audit gate": select branch B only if all three CE conditions hold.

Signatures are L2-normalized HIDDEN-d vectors (model.signature), so cosine
similarity == dot product and spherical k-means == k-means on the sphere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

K_MAX = 8
REFRESH_EVERY = 10                 # drift windows
COHERENCE_FRACTION = 0.60          # eligible iff coherence >= 0.60 * median
ROBUST_ETA = 0.1                   # q-update step
ROBUST_CAP = 4.0                   # cap q_g / q0_g
AUDIT_MARGIN = 0.01                # CE slack in the audit gate


# --------------------------------------------------------------------------- #
# Spherical k-means (deterministic, non-empty)
# --------------------------------------------------------------------------- #
def _normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def spherical_kmeans(X: np.ndarray, k: int, rng: np.random.Generator,
                     n_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic non-empty spherical k-means.

    Init: k-means++-style farthest-point seeding under cosine distance, seeded by
    ``rng`` (so it is deterministic given the seed).  Empty clusters are re-seeded
    to the point farthest from its assigned center, guaranteeing non-empty cells.
    Returns (centers[k,d] unit-norm, labels[n]).
    """
    X = _normalize_rows(np.asarray(X, dtype=np.float64))
    n, d = X.shape
    if n < k:
        raise ValueError(f"need at least k={k} points, got {n}")

    # farthest-point (deterministic given rng) seeding
    first = int(rng.integers(0, n))
    centers = [X[first].copy()]
    sim_to_nearest = X @ centers[0]
    for _ in range(1, k):
        dist = 1.0 - sim_to_nearest              # cosine distance to nearest center
        nxt = int(np.argmax(dist))
        centers.append(X[nxt].copy())
        sim_to_nearest = np.maximum(sim_to_nearest, X @ centers[-1])
    C = _normalize_rows(np.stack(centers))

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        sims = X @ C.T                            # (n,k)
        new_labels = np.argmax(sims, axis=1)
        # recompute centers
        newC = np.zeros_like(C)
        for c in range(k):
            idx = np.where(new_labels == c)[0]
            if idx.size == 0:
                # re-seed empty cluster to the worst-fit point (non-empty guarantee)
                worst = int(np.argmin(sims[np.arange(n), new_labels]))
                newC[c] = X[worst]
                new_labels[worst] = c
            else:
                newC[c] = X[idx].sum(axis=0)
        newC = _normalize_rows(newC)
        if np.array_equal(new_labels, labels) and np.allclose(newC, C):
            C, labels = newC, new_labels
            break
        C, labels = newC, new_labels
    return C, labels


def assign_to_centers(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Hard assignment of rows of X to the nearest (cosine) center.  Used for
    audit examples with train-derived centers (does NOT move centers)."""
    X = _normalize_rows(np.asarray(X, dtype=np.float64))
    return np.argmax(X @ centers.T, axis=1)


def cell_coherence(X: np.ndarray, labels: np.ndarray, centers: np.ndarray,
                   k: int) -> np.ndarray:
    """Per-cell coherence = mean cosine similarity of a cell's (train) points to
    its center.  Cells with no points get coherence 0."""
    X = _normalize_rows(np.asarray(X, dtype=np.float64))
    coh = np.zeros(k, dtype=np.float64)
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size:
            coh[c] = float((X[idx] @ centers[c]).mean())
    return coh


# --------------------------------------------------------------------------- #
# Center alignment across refreshes (persistent IDs)
# --------------------------------------------------------------------------- #
def align_centers(old_centers: np.ndarray, new_centers: np.ndarray) -> np.ndarray:
    """Return a permutation ``perm`` such that ``new_centers[perm]`` best matches
    ``old_centers`` by an exact maximum-similarity assignment (Hungarian).  Keeps
    persistent cell IDs stable across refreshes."""
    from scipy.optimize import linear_sum_assignment
    sim = old_centers @ new_centers.T            # (k,k)
    row, col = linear_sum_assignment(-sim)       # maximize similarity
    perm = np.empty(len(col), dtype=np.int64)
    perm[row] = col
    return perm


# --------------------------------------------------------------------------- #
# Robust-eligibility
# --------------------------------------------------------------------------- #
def eligible_cells(train_coh: np.ndarray, train_support: np.ndarray,
                   audit_support: np.ndarray) -> np.ndarray:
    """Boolean mask of robust-eligible cells: has train AND audit support AND
    train-only coherence >= COHERENCE_FRACTION * median cell coherence (median
    over cells that have train support)."""
    has_support = (train_support > 0) & (audit_support > 0)
    supported = train_support > 0
    if not supported.any():
        return np.zeros_like(has_support)
    median_coh = float(np.median(train_coh[supported]))
    thresh = COHERENCE_FRACTION * median_coh
    return has_support & (train_coh >= thresh)


# --------------------------------------------------------------------------- #
# Robust q-update (protocol)
# --------------------------------------------------------------------------- #
def robust_q_update(q0: np.ndarray, cell_loss: np.ndarray,
                    eligible: np.ndarray) -> np.ndarray:
    """One multiplicative-weights step over eligible cells.

        q_g <- Normalize(q_g * exp(eta * L_g / mean_active_loss))

    starting from q = q0, retaining the ORIGINAL total eligible mass, capping
    q_g / q0_g at ROBUST_CAP, then renormalizing to that retained mass.  Cells
    that are not eligible keep their q0 mass unchanged.

    q0 is the empirical (train-arrival) frequency vector over cells; cell_loss is
    the per-cell mean cross-entropy of the branch-B loss probe; both length k.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    L = np.asarray(cell_loss, dtype=np.float64)
    elig = np.asarray(eligible, dtype=bool)
    k = q0.shape[0]

    q = q0.copy()
    if elig.sum() == 0:
        return q  # no eligible cell -> no correction

    active_mass = float(q0[elig].sum())
    mean_active_loss = float((q0[elig] * L[elig]).sum() / max(active_mass, 1e-12))
    if not np.isfinite(mean_active_loss) or mean_active_loss <= 0:
        return q0.copy()

    qg = q0[elig] * np.exp(ROBUST_ETA * L[elig] / mean_active_loss)
    # cap the multiplicative ratio q_g / q0_g at ROBUST_CAP
    ratio = qg / np.maximum(q0[elig], 1e-12)
    ratio = np.minimum(ratio, ROBUST_CAP)
    qg = q0[elig] * ratio
    # renormalize eligible mass back to the original active_mass
    s = qg.sum()
    if s <= 0 or not np.isfinite(s):
        return q0.copy()
    qg = qg * (active_mass / s)

    q[elig] = qg
    return q


# --------------------------------------------------------------------------- #
# Audit gate (branch selection)
# --------------------------------------------------------------------------- #
@dataclass
class AuditInputs:
    """CE summaries needed to choose branch A or B (protocol 'audit gate')."""
    hist_mean_ce_A: float          # frequency-weighted (q0) historical audit mean
    hist_mean_ce_B: float
    cur_audit_ce_A: float          # current-window audit mean
    cur_audit_ce_B: float
    worst_cell_ce_A: float         # worst eligible-cell audit CE
    worst_cell_ce_B: float
    n_eligible: int
    finite: bool = True
    valid_rank: bool = True
    has_audit_support: bool = True


def select_branch(a: AuditInputs) -> Tuple[str, Dict]:
    """Return ('B', reason) iff ALL three conditions hold AND the forcing
    conditions do not trigger a fallback; otherwise ('A', reason).

        hist_mean_CE(B)  <= hist_mean_CE(A)  + 0.01
        cur_audit_CE(B)  <= cur_audit_CE(A)  + 0.01
        worst_cell_CE(B) <  worst_cell_CE(A)

    Missing audit support, < 2 eligible cells, non-finite loss, or invalid rank
    forces branch A (protocol)."""
    reason: Dict = {}
    if (not a.finite) or (not a.valid_rank) or (not a.has_audit_support) \
            or a.n_eligible < 2:
        reason["forced_A"] = {
            "finite": a.finite, "valid_rank": a.valid_rank,
            "has_audit_support": a.has_audit_support, "n_eligible": a.n_eligible,
        }
        return "A", reason

    c1 = a.hist_mean_ce_B <= a.hist_mean_ce_A + AUDIT_MARGIN
    c2 = a.cur_audit_ce_B <= a.cur_audit_ce_A + AUDIT_MARGIN
    c3 = a.worst_cell_ce_B < a.worst_cell_ce_A
    reason.update({"c_hist": c1, "c_cur": c2, "c_worst": c3})
    return ("B", reason) if (c1 and c2 and c3) else ("A", reason)


# --------------------------------------------------------------------------- #
# Codebook state (centers + persistent ids + q0)
# --------------------------------------------------------------------------- #
@dataclass
class Codebook:
    """Holds cell centers (persistent IDs), the warmup mean, and train-arrival
    counts that define q0.  Refreshed from train-store signatures only."""
    centers: np.ndarray                       # (k, 128) unit-norm
    warmup_mean: np.ndarray                   # (128,)
    k: int = K_MAX
    train_arrivals: np.ndarray = field(default=None)  # (k,) counts -> q0

    def __post_init__(self):
        if self.train_arrivals is None:
            self.train_arrivals = np.zeros(self.k, dtype=np.int64)

    def q0(self) -> np.ndarray:
        """Empirical cell-frequency vector from train arrivals (uniform if none
        counted yet, to avoid a zero vector)."""
        tot = float(self.train_arrivals.sum())
        if tot <= 0:
            return np.full(self.k, 1.0 / self.k)
        return self.train_arrivals.astype(np.float64) / tot

    def add_arrivals(self, labels: np.ndarray) -> None:
        for c in labels:
            self.train_arrivals[int(c)] += 1

    def refresh(self, train_signatures: np.ndarray, rng: np.random.Generator) -> None:
        """Refresh centers from current train-store signatures and re-align to
        persistent IDs so cell identity is stable across refreshes."""
        newC, _ = spherical_kmeans(train_signatures, self.k, rng)
        perm = align_centers(self.centers, newC)
        self.centers = newC[perm]
