"""Verbalizer-offset fitting and the identification gap.

Implements the measurable quantities of
``notes/phase2j_offset_conflict_probe_protocol.md``:

* ``R_raw``  -- accuracy with no offset (official free-generation argmax proxy);
* ``R_orc``  -- accuracy with a per-task offset (needs task identity);
* ``R_shr``  -- accuracy with one offset shared by all seen tasks;
* ``Delta_id = R_orc - R_shr`` -- the identification gap.

Design notes that matter for correctness:

* An offset on a verbalizer set is identifiable only up to an additive
  constant, because adding a constant to every label logit leaves the argmax
  unchanged.  Every returned offset is therefore gauge-fixed to zero mean.  For
  a binary set this collapses the search to the single scalar
  ``b[1] - b[0]``, which is why the budgeted bound is a quantization rather
  than a low-rank statement (see the theory note).
* Offsets are fitted by direct search on *balanced* accuracy over a
  train-derived split, never on test, and never on the split used to report.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Mapping, Sequence

import numpy as np


def gauge_fix(offset: np.ndarray) -> np.ndarray:
    """Remove the additive-constant degree of freedom."""
    offset = np.asarray(offset, dtype=np.float64)
    return offset - offset.mean()


def balanced_accuracy(logits: np.ndarray, labels: np.ndarray,
                      offset: np.ndarray | None = None) -> float:
    """Mean per-class recall of ``argmax(logits + offset)``.

    ``logits`` is ``(n, K)`` restricted to the task's verbalizer tokens and
    ``labels`` holds class indices in ``[0, K)``.
    """
    if offset is not None:
        logits = logits + np.asarray(offset, dtype=np.float64)[None, :]
    pred = logits.argmax(axis=1)
    recalls = []
    for k in range(logits.shape[1]):
        mask = labels == k
        if mask.any():
            recalls.append(float((pred[mask] == k).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def natural_accuracy(logits: np.ndarray, labels: np.ndarray,
                     offset: np.ndarray | None = None) -> float:
    """Plain accuracy of ``argmax(logits + offset)``."""
    if offset is not None:
        logits = logits + np.asarray(offset, dtype=np.float64)[None, :]
    return float((logits.argmax(axis=1) == labels).mean())


def fit_offset(logits: np.ndarray, labels: np.ndarray, *,
               objective=balanced_accuracy,
               grid: int = 81, span: float = 8.0,
               refine_rounds: int = 3) -> np.ndarray:
    """Fit a gauge-fixed offset maximizing ``objective`` by coordinate search.

    For ``K == 2`` this is an exact 1-D scan over the scalar contrast, refined
    around the incumbent.  For ``K > 2`` it is a coordinate-wise refinement,
    which is a heuristic: the objective is piecewise constant, so we report it
    as a *fitted* offset and never as a certified optimum.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    K = logits.shape[1]
    best = np.zeros(K)
    best_val = objective(logits, labels, best)

    lo, hi = -span, span
    for rnd in range(refine_rounds):
        improved = False
        for k in range(K):
            cand_vals = np.linspace(lo, hi, grid)
            for v in cand_vals:
                trial = best.copy()
                trial[k] = v
                trial = gauge_fix(trial)
                val = objective(logits, labels, trial)
                if val > best_val + 1e-12:
                    best_val, best, improved = val, trial, True
        # shrink the search window around the incumbent for the next round
        width = (hi - lo) / 4.0
        lo, hi = -width, width
        if not improved and rnd >= 1:
            break
    return gauge_fix(best)


@dataclass(frozen=True)
class TaskLogits:
    """Verbalizer-restricted logits for one task on one split."""

    task: str
    verbalizer: tuple[str, ...]
    logits: np.ndarray  # (n, K)
    labels: np.ndarray  # (n,) class indices

    def __post_init__(self) -> None:
        if self.logits.ndim != 2:
            raise ValueError("logits must be 2-D (n, K)")
        if self.logits.shape[1] != len(self.verbalizer):
            raise ValueError("logits width must match verbalizer size")
        if self.logits.shape[0] != self.labels.shape[0]:
            raise ValueError("logits and labels disagree on n")


def fit_shared_offset(
    fit_tasks: Sequence[TaskLogits],
    *,
    objective=balanced_accuracy,
    grid: int = 81,
    span: float = 8.0,
) -> Mapping[str, float]:
    """Fit ONE offset over shared verbalizer *tokens*, minimizing worst-task loss.

    The offset is indexed by label string, not by position, so tasks that share
    tokens are genuinely coupled -- required because in Order-4 ``neutral``
    spans both the NLI and sentiment groups and RTE's label set is a subset of
    MNLI's (see the theory note's verified sharing table).

    Returns a mapping ``token -> offset``, gauge-fixed per task at scoring time.
    """
    tokens = sorted({t for task in fit_tasks for t in task.verbalizer})
    index = {tok: i for i, tok in enumerate(tokens)}
    shared = np.zeros(len(tokens))

    per_task_oracle = {
        task.task: objective(task.logits, task.labels,
                             fit_offset(task.logits, task.labels,
                                        objective=objective))
        for task in fit_tasks
    }

    def worst_loss(vec: np.ndarray) -> float:
        worst = 0.0
        for task in fit_tasks:
            sub = np.array([vec[index[tok]] for tok in task.verbalizer])
            val = objective(task.logits, task.labels, gauge_fix(sub))
            worst = max(worst, per_task_oracle[task.task] - val)
        return worst

    best_val = worst_loss(shared)
    lo, hi = -span, span
    for rnd in range(3):
        improved = False
        for i in range(len(tokens)):
            for v in np.linspace(lo, hi, grid):
                trial = shared.copy()
                trial[i] = v
                val = worst_loss(trial)
                if val < best_val - 1e-12:
                    best_val, shared, improved = val, trial, True
        width = (hi - lo) / 4.0
        lo, hi = -width, width
        if not improved and rnd >= 1:
            break
    return {tok: float(shared[i]) for tok, i in index.items()}


def identification_gap(
    eval_task: TaskLogits,
    *,
    per_task_offset: np.ndarray,
    shared_offset: Mapping[str, float],
    objective=balanced_accuracy,
) -> dict[str, float]:
    """Report R_raw / R_orc / R_shr and Delta_id for one task.

    ``per_task_offset`` and ``shared_offset`` must have been fitted on a split
    disjoint from ``eval_task`` -- this function does not check that, the
    caller's partitioning does.
    """
    sub = np.array([shared_offset.get(tok, 0.0) for tok in eval_task.verbalizer])
    r_raw = objective(eval_task.logits, eval_task.labels, None)
    r_orc = objective(eval_task.logits, eval_task.labels,
                      gauge_fix(per_task_offset))
    r_shr = objective(eval_task.logits, eval_task.labels, gauge_fix(sub))
    return {
        "R_raw": r_raw,
        "R_orc": r_orc,
        "R_shr": r_shr,
        "Delta_id": r_orc - r_shr,
        "shallow_gap": r_orc - r_raw,
    }


def conflict_statistics(per_task_offsets: Mapping[str, np.ndarray],
                        verbalizers: Mapping[str, Sequence[str]]) -> dict:
    """Spread and quantization radii of per-task optima on shared tokens.

    ``omega`` is the max pairwise distance between per-task optima restricted
    to their common tokens (the Prop-2 spread).  ``q_m`` is the worst-case
    m-quantization radius (the corrected Prop-3 object).
    """
    names = sorted(per_task_offsets)
    pairs = []
    for a, b in itertools.combinations(names, 2):
        common = sorted(set(verbalizers[a]) & set(verbalizers[b]))
        if len(common) < 2:
            continue
        va = gauge_fix(np.array([
            per_task_offsets[a][list(verbalizers[a]).index(t)] for t in common]))
        vb = gauge_fix(np.array([
            per_task_offsets[b][list(verbalizers[b]).index(t)] for t in common]))
        pairs.append((a, b, float(np.linalg.norm(va - vb))))

    omega = max((d for _, _, d in pairs), default=0.0)

    groups: dict[tuple[str, ...], list[str]] = {}
    for name in names:
        key = tuple(sorted(verbalizers[name]))
        groups.setdefault(key, []).append(name)

    quant: dict[str, dict[int, float]] = {}
    for key, members in groups.items():
        if len(members) < 2:
            continue
        X = np.stack([
            gauge_fix(np.array([
                per_task_offsets[m][list(verbalizers[m]).index(t)] for t in key]))
            for m in members
        ])
        radii = {}
        for m in range(1, min(4, len(members)) + 1):
            radii[m] = _worst_case_quantization_radius(X, m)
        quant["|".join(key)] = radii

    return {"omega": omega, "pairwise": pairs, "quantization_radius": quant}


def _worst_case_quantization_radius(X: np.ndarray, m: int) -> float:
    """min over m centres of the max distance from a point to its centre.

    Exact by enumeration when the number of points is small (which it is: at
    most 4 tasks share a verbalizer in Order-4), using the points themselves
    as candidate centres plus their pairwise midpoints.
    """
    n = X.shape[0]
    if m >= n:
        return 0.0
    cands = [X[i] for i in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        cands.append((X[i] + X[j]) / 2.0)
    best = float("inf")
    for combo in itertools.combinations(range(len(cands)), m):
        C = np.stack([cands[i] for i in combo])
        d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=-1).min(axis=1)
        best = min(best, float(d.max()))
    return best
