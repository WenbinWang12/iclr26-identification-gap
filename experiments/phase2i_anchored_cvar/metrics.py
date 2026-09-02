"""Metrics for the Phase-2I headroom probe.

The functions here deliberately separate three quantities that are easy to
conflate in a heterogeneous continual-learning benchmark:

* raw loss (intrinsic difficulty plus forgetting),
* acquired gain (what the learner actually learned at task arrival), and
* anchored regret (the fraction of that gain subsequently lost).

Only training-derived risk records may be passed to the optimization path.
Audit and test records must be evaluated by a separate caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnchoredRegret:
    """Per-example normalized forgetting and its acquisition eligibility."""

    regret: np.ndarray
    acquired_gain: np.ndarray
    eligible: np.ndarray

    @property
    def tail_values(self) -> np.ndarray:
        """Finite regret values that are allowed to enter a tail objective."""

        return self.regret[self.eligible]


@dataclass(frozen=True)
class HeadroomGate:
    """Outcome of the pre-registered five-arm feasibility decision."""

    passed: bool
    failures: tuple[str, ...]


def _as_finite_vector(name: str, values: np.ndarray | list[float]) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {out.shape}")
    if out.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains a non-finite value")
    return out


def anchored_regret(
    loss_pre: np.ndarray | list[float],
    loss_post: np.ndarray | list[float],
    loss_now: np.ndarray | list[float],
    *,
    min_acquired_gain: float = 0.05,
    epsilon: float = 1e-8,
    r_max: float = 2.0,
) -> AnchoredRegret:
    """Compute the fraction of acquired per-example loss reduction forgotten.

    ``loss_pre`` and ``loss_post`` are measured immediately before and after
    the example's arrival episode.  Examples with too little positive acquired
    gain are marked ineligible: their normalized value is not reliable enough
    for tail optimization.  Their raw losses should still be reported.
    """

    pre = _as_finite_vector("loss_pre", loss_pre)
    post = _as_finite_vector("loss_post", loss_post)
    now = _as_finite_vector("loss_now", loss_now)
    if pre.shape != post.shape or pre.shape != now.shape:
        raise ValueError(
            f"loss vectors must have identical shapes, got "
            f"{pre.shape}, {post.shape}, and {now.shape}"
        )
    if min_acquired_gain < 0:
        raise ValueError("min_acquired_gain must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if r_max <= 0:
        raise ValueError("r_max must be positive")

    acquired = pre - post
    eligible = acquired >= min_acquired_gain
    denominator = np.maximum(acquired, epsilon)
    forgotten = np.maximum(now - post, 0.0)
    regret = np.clip(forgotten / denominator, 0.0, r_max)
    # NaN makes an unmasked CVaR call fail closed rather than silently treating
    # a never-learned example as a perfectly retained example.
    regret = np.where(eligible, regret, np.nan)
    return AnchoredRegret(regret=regret, acquired_gain=acquired, eligible=eligible)


def empirical_cvar_weights(
    values: np.ndarray | list[float], alpha: float
) -> np.ndarray:
    """Return an optimizer of empirical upper-tail CVaR's capped-weight LP.

    The empirical distribution gives every observation mass ``1 / N``.  The
    returned vector maximizes ``q @ values`` subject to ``q >= 0``, ``sum q=1``
    and ``q_i <= 1 / (alpha N)``.  Fractional ``alpha * N`` is handled exactly.
    Ties are resolved by stable input order and do not change the objective.
    """

    x = _as_finite_vector("values", values)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")

    n = x.size
    cap = 1.0 / (alpha * n)
    order = np.argsort(-x, kind="stable")
    q = np.zeros(n, dtype=np.float64)
    remaining = 1.0
    for index in order:
        mass = min(cap, remaining)
        q[index] = mass
        remaining -= mass
        if remaining <= 32 * np.finfo(np.float64).eps:
            break
    # Remove accumulated floating-point error without violating the cap.
    q[order[0]] += 1.0 - float(q.sum())
    if q[order[0]] > cap + 1e-12:
        raise RuntimeError("internal CVaR weight construction exceeded its cap")
    return q


def empirical_cvar(values: np.ndarray | list[float], alpha: float) -> float:
    """Compute empirical upper-tail CVaR, where larger values are worse."""

    x = _as_finite_vector("values", values)
    return float(empirical_cvar_weights(x, alpha) @ x)


def normalized_retention(
    score_base: np.ndarray | list[float],
    score_immediate: np.ndarray | list[float],
    score_final: np.ndarray | list[float],
    *,
    denominator_floor: float = 0.05,
) -> np.ndarray:
    """Normalize final retained gain by each task's own acquired score gain."""

    base = _as_finite_vector("score_base", score_base)
    immediate = _as_finite_vector("score_immediate", score_immediate)
    final = _as_finite_vector("score_final", score_final)
    if base.shape != immediate.shape or base.shape != final.shape:
        raise ValueError("score vectors must have identical shapes")
    if denominator_floor <= 0:
        raise ValueError("denominator_floor must be positive")
    acquired = immediate - base
    return (final - base) / np.maximum(acquired, denominator_floor)


def assess_headroom_gate(
    *,
    acquisition_gains: np.ndarray | list[float],
    oracle_tail_gain: float,
    oracle_mean_delta: float,
    groupfree_tail_gain: float,
    rank_exchange_tail_gain: float,
    seed_tail_deltas: np.ndarray | list[float],
    step_matched_gain: float,
    flop_matched_gain: float,
    beats_strong_reference: bool,
) -> HeadroomGate:
    """Apply the locked Phase-2I GO/KILL criteria in score-fraction units.

    A gain of ``0.05`` means five percentage points.  ``oracle_mean_delta`` is
    oracle minus reference, so negative values are a mean-performance loss.
    ``rank_exchange_tail_gain`` compares the full method with the same CVaR
    learner under uniform layer rank.  Step/FLOP gains compare the full method
    with the strongest correspondingly matched reference.
    """

    acquired = _as_finite_vector("acquisition_gains", acquisition_gains)
    seed_delta = _as_finite_vector("seed_tail_deltas", seed_tail_deltas)
    scalars = {
        "oracle_tail_gain": oracle_tail_gain,
        "oracle_mean_delta": oracle_mean_delta,
        "groupfree_tail_gain": groupfree_tail_gain,
        "rank_exchange_tail_gain": rank_exchange_tail_gain,
        "step_matched_gain": step_matched_gain,
        "flop_matched_gain": flop_matched_gain,
    }
    if not all(np.isfinite(v) for v in scalars.values()):
        raise ValueError("headroom gate contains a non-finite scalar")

    failures: list[str] = []
    if float(acquired.min()) < 0.05:
        failures.append("a primary task has <5pp acquisition gain")
    if oracle_tail_gain < 0.05:
        failures.append("task-ID oracle has <5pp tail headroom")
    if oracle_mean_delta < -0.01:
        failures.append("task-ID oracle loses >1pp mean performance")

    required_groupfree = max(0.03, 0.70 * max(oracle_tail_gain, 0.0))
    if groupfree_tail_gain < required_groupfree:
        failures.append(
            "group-free CVaR has <3pp gain or recovers <70% of oracle gain"
        )
    if rank_exchange_tail_gain < 0.02:
        failures.append("rank exchange adds <2pp over CVaR with uniform rank")
    if float(seed_delta.min()) < -0.03:
        failures.append("at least one seed has a >3pp tail catastrophe")
    if step_matched_gain <= 0.0:
        failures.append("gain does not survive update-step matching")
    if flop_matched_gain <= 0.0:
        failures.append("gain does not survive token-FLOP matching")
    if not beats_strong_reference:
        failures.append("full method does not beat SLAO+ER/mean-adaptive reference")
    return HeadroomGate(passed=not failures, failures=tuple(failures))
