"""Pure-numpy tests for Phase-2I anchored regret and empirical CVaR."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import (  # noqa: E402
    assess_headroom_gate,
    anchored_regret,
    empirical_cvar,
    empirical_cvar_weights,
    normalized_retention,
)


def test_anchor_does_not_chase_intrinsically_hard_example():
    # Example 0 remains high-loss but was never learned.  Example 1 was learned
    # and then lost all its acquired gain.  Raw loss ranks 0 worse; regret must
    # correctly rank 1 worse and mark 0 ineligible.
    out = anchored_regret(
        loss_pre=[5.00, 1.00],
        loss_post=[4.98, 0.20],
        loss_now=[5.20, 1.00],
        min_acquired_gain=0.05,
    )
    assert out.eligible.tolist() == [False, True]
    assert np.isnan(out.regret[0])
    assert np.allclose(out.tail_values, [1.0])


def test_anchor_clips_recovery_and_excess_forgetting():
    out = anchored_regret(
        loss_pre=[1.0, 1.0],
        loss_post=[0.5, 0.5],
        loss_now=[0.2, 2.0],
        r_max=2.0,
    )
    assert np.allclose(out.regret, [0.0, 2.0])


def test_cvar_fractional_tail_and_weight_cap():
    x = np.array([1.0, 2.0, 3.0, 10.0])
    # alpha*N = 1.5: take all of 10 and half of 3, then divide by 1.5.
    expected = (10.0 + 0.5 * 3.0) / 1.5
    q = empirical_cvar_weights(x, alpha=0.375)
    assert np.isclose(q.sum(), 1.0)
    assert q.max() <= 1.0 / (0.375 * len(x)) + 1e-12
    assert np.isclose(empirical_cvar(x, alpha=0.375), expected)


def test_cvar_bounds_every_large_empirical_subgroup_mean():
    rng = np.random.default_rng(7)
    x = rng.normal(size=100)
    alpha = 0.2
    tail = empirical_cvar(x, alpha)
    for _ in range(200):
        size = int(rng.integers(20, 101))
        group = rng.choice(len(x), size=size, replace=False)
        assert float(x[group].mean()) <= tail + 1e-12


def test_normalized_retention_uses_task_specific_acquisition():
    got = normalized_retention(
        score_base=[0.20, 0.70],
        score_immediate=[0.80, 0.80],
        score_final=[0.50, 0.75],
    )
    assert np.allclose(got, [0.5, 0.5])


def test_invalid_inputs_fail_closed():
    for alpha in (0.0, -0.1, 1.1):
        try:
            empirical_cvar([1.0, 2.0], alpha)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid alpha {alpha} was accepted")
    try:
        anchored_regret([1.0], [0.5, 0.4], [0.8])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched anchor vectors were accepted")


def test_locked_headroom_gate_passes_only_joint_positive_result():
    passed = assess_headroom_gate(
        acquisition_gains=[0.10, 0.08, 0.12],
        oracle_tail_gain=0.06,
        oracle_mean_delta=-0.005,
        groupfree_tail_gain=0.045,
        rank_exchange_tail_gain=0.025,
        seed_tail_deltas=[0.03, 0.04, 0.02],
        step_matched_gain=0.02,
        flop_matched_gain=0.01,
        beats_strong_reference=True,
    )
    assert passed.passed and not passed.failures

    failed = assess_headroom_gate(
        acquisition_gains=[0.10, 0.08, 0.12],
        oracle_tail_gain=0.06,
        oracle_mean_delta=-0.005,
        groupfree_tail_gain=0.045,
        rank_exchange_tail_gain=0.005,
        seed_tail_deltas=[0.03, -0.04, 0.02],
        step_matched_gain=0.02,
        flop_matched_gain=-0.001,
        beats_strong_reference=False,
    )
    assert not failed.passed
    assert len(failed.failures) == 4


if __name__ == "__main__":
    test_anchor_does_not_chase_intrinsically_hard_example()
    test_anchor_clips_recovery_and_excess_forgetting()
    test_cvar_fractional_tail_and_weight_cap()
    test_cvar_bounds_every_large_empirical_subgroup_mean()
    test_normalized_retention_uses_task_specific_acquisition()
    test_invalid_inputs_fail_closed()
    test_locked_headroom_gate_passes_only_joint_positive_result()
    print("test_metrics OK")
