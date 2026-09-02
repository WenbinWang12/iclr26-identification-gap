"""Each test is named for the failure it prevents.

Phase-2S's lesson (§A4.4) was that a synthetic set built from the same generative
assumption the method makes proves nothing.  So these tests check *invariants and
mechanics* only -- gauge invariance, exact abstention, label-freeness, tie handling.
They deliberately make no claim about whether VGC helps; that is what the held-out
seeds are for.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase2t_vgc.vgc import (  # noqa: E402
    TAU_SD, balanced_accuracy, bc_offset, margin, ols_slope, oracle_offset,
    spearman, vgc_offset,
)


def _batch(n=64, sep=2.0, sd=1.0, ratio=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n1 = int(round(n * ratio))
    y = np.array([0] * (n - n1) + [1] * n1)
    s = np.where(y == 1, sep / 2, -sep / 2) + rng.normal(0, sd, n)
    return np.stack([np.zeros(n), s], axis=1), y


def test_margin_is_the_logit_difference():
    Z = np.array([[1.0, 3.0], [2.0, -1.0]])
    assert np.allclose(margin(Z), [2.0, -3.0])


def test_margin_is_gauge_invariant_bitwise_when_the_shift_is_exact():
    """Small integers survive addition without rounding, so this is bit-exact.
    On general floats ``Z + c`` itself rounds -- that is representation error in
    the *input*, not gauge dependence in the method, and the next test bounds it.
    """
    Z = np.stack([np.zeros(4), np.array([-3.0, 0.0, 1.0, 5.0])], axis=1)
    for c in (-8.0, 0.0, 16.0):
        assert np.array_equal(margin(Z + c), margin(Z))


def test_the_offset_is_gauge_invariant_up_to_float_addition_error():
    """``Z + c`` itself rounds; the method reads only differences, so the residual
    is representation error (measured max 1.2e-11 over 200x4 shifts), not a
    dependence on the gauge.  A method that read absolute logits would move by ~c.
    """
    Z, _ = _batch(seed=2)
    base = vgc_offset(Z)
    for c in (-4.0, 9.0, 1e3, 1e6):
        shifted = vgc_offset(Z + c)
        assert shifted.applied == base.applied
        assert shifted.offset[1] == pytest.approx(base.offset[1], abs=1e-9)
        assert shifted.sd == pytest.approx(base.sd, abs=1e-9)


def test_more_than_two_classes_is_rejected_rather_than_silently_handled():
    with pytest.raises(ValueError, match="K_S == 2"):
        vgc_offset(np.zeros((10, 3)))
    with pytest.raises(ValueError):
        margin(np.zeros((10, 4)))


def test_sd_of_the_margin_counts_separation_not_only_noise():
    """Caught by my own helper: ``sd(s)`` on a well-separated batch is large even
    when the within-class noise is tiny, because separation inflates the spread.
    So the gate is a *total*-spread test, not a noise test, and a low-sd batch has
    to be built by shrinking sep as well.  This is exactly why §2 flags the
    task-identity confound.
    """
    tiny_noise_wide_sep, _ = _batch(sd=0.05, sep=2.0, seed=3)
    assert vgc_offset(tiny_noise_wide_sep).sd > 0.9
    assert vgc_offset(tiny_noise_wide_sep).applied


def test_abstention_returns_exactly_zero_not_approximately_zero():
    """If abstention returned 1e-18 the argmax could still flip on a tie."""
    Z, _ = _batch(sd=0.05, sep=0.0, seed=3)
    res = vgc_offset(Z)
    assert not res.applied
    assert res.offset.dtype == np.float64
    assert res.offset.tolist() == [0.0, 0.0]


def test_abstention_prediction_is_bit_identical_to_raw():
    Z, y = _batch(sd=0.05, sep=0.0, seed=4)
    res = vgc_offset(Z)
    assert not res.applied
    assert np.array_equal((Z + res.offset).argmax(1), Z.argmax(1))
    assert balanced_accuracy(Z, y, res.offset) == balanced_accuracy(Z, y)


def test_low_variance_batches_abstain_and_high_variance_batches_apply():
    """The gate must actually be a function of sd, not of anything else."""
    lo, _ = _batch(sd=0.10, sep=0.0, seed=5)
    hi, _ = _batch(sd=1.20, sep=0.0, seed=5)
    assert not vgc_offset(lo).applied
    assert vgc_offset(hi).applied


def test_the_gate_reads_no_labels():
    """Signature-level guarantee: a label-dependent gate is not deployable."""
    import inspect
    params = list(inspect.signature(vgc_offset).parameters)
    assert params == ["Z", "tau_sd"]
    Z, y = _batch(seed=6)
    a = vgc_offset(Z)
    b = vgc_offset(Z[np.argsort(-y)])  # relabelling/reordering rows
    assert a.applied == b.applied
    assert a.offset.tolist() == pytest.approx(b.offset.tolist())


def test_vgc_equals_bc_whenever_it_applies():
    """VGC must not quietly become a different estimator."""
    Z, _ = _batch(sd=1.5, seed=7)
    res = vgc_offset(Z)
    assert res.applied
    assert np.allclose(res.offset, bc_offset(Z))


def test_bc_offset_is_the_negative_mean_margin():
    """Phase-2S burned a test on the sign convention; pin it here."""
    Z = np.stack([np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0])], axis=1)
    assert np.allclose(bc_offset(Z), [0.0, -2.5])


def test_sd_uses_the_sample_convention_so_it_does_not_depend_on_n():
    Z, _ = _batch(n=200, sd=1.0, sep=0.0, seed=8)
    assert vgc_offset(Z).sd == pytest.approx(float(margin(Z).std(ddof=1)), rel=1e-12)


def test_single_example_batch_abstains_instead_of_dividing_by_zero():
    Z = np.array([[0.0, 5.0]])
    res = vgc_offset(Z)
    assert not res.applied and res.sd == 0.0


def test_threshold_is_inclusive_at_the_boundary():
    """An exclusive comparison would make TAU_SD's reported sweep off by one cell."""
    s = np.array([-1.0, 1.0])           # ddof=1 sd is exactly sqrt(2)
    Z = np.stack([np.zeros(2), s], axis=1)
    assert vgc_offset(Z, tau_sd=float(np.sqrt(2.0))).applied


def test_tau_sd_is_the_frozen_development_value():
    """Guards against a later edit quietly retuning the frozen threshold."""
    assert TAU_SD == 0.50


def test_oracle_never_loses_to_raw():
    for seed in range(5):
        Z, y = _batch(seed=seed, ratio=0.3)
        assert balanced_accuracy(Z, y, oracle_offset(Z, y)) >= balanced_accuracy(Z, y) - 1e-12


def test_oracle_is_an_upper_bound_on_bc_and_vgc():
    for seed in range(5):
        Z, y = _batch(seed=seed, ratio=0.8)
        orc = balanced_accuracy(Z, y, oracle_offset(Z, y))
        assert orc >= balanced_accuracy(Z, y, bc_offset(Z)) - 1e-12
        assert orc >= balanced_accuracy(Z, y, vgc_offset(Z).offset) - 1e-12


def test_spearman_matches_pearson_on_ranks_for_a_monotone_map():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(x, np.exp(x)) == pytest.approx(1.0)
    assert spearman(x, -np.exp(x)) == pytest.approx(-1.0)


def test_spearman_averages_tied_ranks_instead_of_breaking_by_input_order():
    a = np.array([1.0, 1.0, 2.0, 3.0])
    assert spearman(a, np.array([5.0, 5.0, 6.0, 7.0])) == pytest.approx(1.0)
    assert spearman(a, np.array([5.0, 5.0, 6.0, 7.0])) == \
        pytest.approx(spearman(a[::-1], np.array([7.0, 6.0, 5.0, 5.0])))


def test_spearman_is_nan_below_three_points_rather_than_a_spurious_one():
    assert np.isnan(spearman([1.0, 2.0], [3.0, 4.0]))


def test_ols_slope_recovers_a_known_line():
    x = np.linspace(0, 1, 11)
    assert ols_slope(x, 3.0 * x - 1.0) == pytest.approx(3.0)


def test_ols_slope_is_nan_on_a_constant_regressor():
    assert np.isnan(ols_slope(np.ones(5), np.arange(5.0)))
