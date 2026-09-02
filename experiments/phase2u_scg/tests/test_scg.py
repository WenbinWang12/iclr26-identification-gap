"""Each test is named for the failure it prevents.

These check mechanics and invariants only.  Phase-2S proved that a synthetic set
built from the method's own assumption proves nothing (§A4.4), and Phase-2T proved
that a statistic which separates in the median can still fail in the tail (§A1.4).
Whether SCG works is decided by seeds 11-15, not here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase2u_scg.scg import (  # noqa: E402
    TAU_M, balanced_accuracy, bc_offset, margin, oracle_offset, scg_offset,
    scg_offset_normalised, soft_offset,
)


def _batch(n=64, shift=0.0, sd=1.0, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.normal(shift, sd, n)
    y = (s > shift).astype(int)
    return np.stack([np.zeros(n), s], axis=1), y


def test_tau_m_is_the_frozen_development_value():
    """Guards against a later edit quietly retuning the frozen threshold."""
    assert TAU_M == 1.4


def test_margin_rejects_more_than_two_classes_rather_than_truncating():
    with pytest.raises(ValueError, match="K_S == 2"):
        scg_offset(np.zeros((10, 3)))
    with pytest.raises(ValueError):
        margin(np.zeros((5, 1)))


def test_margin_is_bitwise_gauge_invariant_for_exact_shifts():
    Z = np.stack([np.zeros(4), np.array([-3.0, 0.0, 1.0, 5.0])], axis=1)
    for c in (-8.0, 0.0, 16.0):
        assert np.array_equal(margin(Z + c), margin(Z))


def test_the_gate_decision_does_not_move_with_the_gauge():
    """A gate reading absolute logits would flip as soon as c crossed tau."""
    Z, _ = _batch(shift=2.0, seed=1)
    base = scg_offset(Z)
    for c in (-50.0, 0.0, 50.0, 1e4):
        r = scg_offset(Z + c)
        assert r.applied == base.applied
        assert r.abs_m == pytest.approx(base.abs_m, abs=1e-9)
        assert r.offset[1] == pytest.approx(base.offset[1], abs=1e-9)


def test_abstention_returns_exactly_zero_not_approximately_zero():
    Z, _ = _batch(shift=0.0, seed=2)
    r = scg_offset(Z)
    assert not r.applied
    assert r.offset.tolist() == [0.0, 0.0]


def test_abstention_leaves_predictions_and_score_bit_identical():
    Z, y = _batch(shift=0.0, seed=3)
    r = scg_offset(Z)
    assert not r.applied
    assert np.array_equal((Z + r.offset).argmax(1), Z.argmax(1))
    assert balanced_accuracy(Z, y, r.offset) == balanced_accuracy(Z, y)


def test_a_large_correction_is_applied_and_a_small_one_is_not():
    big, _ = _batch(shift=3.0, sd=0.5, seed=4)
    small, _ = _batch(shift=0.05, sd=0.5, seed=4)
    assert scg_offset(big).applied
    assert not scg_offset(small).applied


def test_the_gate_is_on_magnitude_so_it_is_sign_symmetric():
    """Gating on ``mean`` rather than ``|mean|`` would only protect one direction."""
    Z, _ = _batch(shift=2.5, sd=0.5, seed=5)
    flipped = Z[:, ::-1].copy()          # swaps the two classes, negating s
    a, b = scg_offset(Z), scg_offset(flipped)
    assert a.applied and b.applied
    assert a.abs_m == pytest.approx(b.abs_m)
    assert a.offset[1] == pytest.approx(-b.offset[1])


def test_when_applied_scg_is_exactly_bc():
    """SCG must not become a different estimator; it only chooses when to act."""
    Z, _ = _batch(shift=3.0, sd=0.5, seed=6)
    r = scg_offset(Z)
    assert r.applied
    assert np.allclose(r.offset, bc_offset(Z))


def test_bc_offset_is_the_negative_mean_margin():
    Z = np.stack([np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0])], axis=1)
    assert np.allclose(bc_offset(Z), [0.0, -2.5])


def test_the_gate_reads_no_labels():
    import inspect
    assert list(inspect.signature(scg_offset).parameters) == ["Z", "tau_m"]
    Z, _ = _batch(shift=2.0, seed=7)
    assert scg_offset(Z).offset.tolist() == \
        pytest.approx(scg_offset(Z[::-1]).offset.tolist())


def test_the_threshold_is_inclusive_at_the_boundary():
    """An exclusive test would shift every cell of the frozen sweep by one.

    Uses an exactly-representable value: ``mean([1.4, 1.4, 1.4])`` evaluates to
    1.3999999999999997 and would abstain, which is a property of the mean's
    rounding rather than of the comparison.  Nothing in the sweep depends on a
    batch landing exactly on tau, so this is the right way to pin the operator.
    """
    Z = np.stack([np.zeros(4), np.full(4, 1.5)], axis=1)
    assert scg_offset(Z, tau_m=1.5).applied
    assert not scg_offset(Z, tau_m=1.5 + 1e-9).applied


def test_single_example_batch_is_handled_without_error():
    Z = np.array([[0.0, 5.0]])
    assert scg_offset(Z).applied            # |mean| = 5 >= 1.4
    assert not scg_offset(np.array([[0.0, 0.1]])).applied


def test_normalised_variant_divides_by_sd_and_is_a_different_statistic():
    """U6 exists because these two disagree; a test that they agreed would be wrong."""
    Z, _ = _batch(shift=1.0, sd=4.0, seed=8)
    plain, norm = scg_offset(Z), scg_offset_normalised(Z, tau=1.0)
    assert plain.abs_m != pytest.approx(norm.abs_m)
    assert norm.abs_m == pytest.approx(abs(margin(Z).mean()) / margin(Z).std(ddof=1))


def test_normalised_variant_abstains_when_sd_is_zero_instead_of_dividing():
    Z = np.stack([np.zeros(4), np.full(4, 3.0)], axis=1)
    r = scg_offset_normalised(Z, tau=1.0)
    assert not r.applied and r.abs_m == 0.0


def test_soft_threshold_shrinks_toward_zero_and_keeps_the_sign():
    Z, _ = _batch(shift=2.0, sd=0.3, seed=9)
    m = float(margin(Z).mean())
    off = soft_offset(Z, tau=0.5)
    assert np.sign(off[1]) == np.sign(-m)
    assert abs(off[1]) == pytest.approx(abs(m) - 0.5)


def test_soft_threshold_is_exactly_zero_below_tau_so_it_is_not_a_no_op_gate():
    Z, _ = _batch(shift=0.05, sd=0.2, seed=10)
    assert soft_offset(Z, tau=5.0).tolist() == [0.0, -0.0] or \
        soft_offset(Z, tau=5.0)[1] == 0.0


def test_soft_threshold_reduces_to_bc_at_tau_zero():
    Z, _ = _batch(shift=1.0, seed=11)
    assert np.allclose(soft_offset(Z, tau=0.0), bc_offset(Z))


def test_oracle_never_loses_to_raw_and_bounds_both_arms():
    for seed in range(5):
        Z, y = _batch(shift=1.0, seed=seed)
        orc = balanced_accuracy(Z, y, oracle_offset(Z, y))
        assert orc >= balanced_accuracy(Z, y) - 1e-12
        assert orc >= balanced_accuracy(Z, y, bc_offset(Z)) - 1e-12
        assert orc >= balanced_accuracy(Z, y, scg_offset(Z).offset) - 1e-12


def test_balanced_accuracy_ignores_absent_classes_rather_than_scoring_them_zero():
    Z = np.stack([np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0])], axis=1)
    assert balanced_accuracy(Z, np.ones(4, dtype=int)) == pytest.approx(1.0)
