"""Regression tests for Phase-2S SIO.

Every test here corresponds to a bug or a false gate decision that actually
happened during development, named so the failure it prevents is readable from the
test name. Protocol §6 requires the parameter-recovery test to pass before any real
logits are touched.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.phase2s_sio.sio import (
    N_MIN,
    SIOResult,
    _em2,
    _tau,
    bc_offset,
    marginal_slope,
    sio_offset,
)


def make_logits(n0, n1, *, sep=1.2, bias=0.6, seed=0, sds=(0.8, 0.8)):
    """Two-class logits with a known injected bias on coordinate 1."""
    rng = np.random.default_rng(seed)
    parts, labels = [], []
    for c, n in enumerate((n0, n1)):
        if n == 0:
            continue
        z = np.zeros((n, 2))
        z[:, c] = sep
        z += rng.normal(0.0, sds[c], size=(n, 2))
        z[:, 1] += bias
        parts.append(z)
        labels.append(np.full(n, c))
    return np.vstack(parts), np.concatenate(labels)


def balanced_accuracy(Z, y, offset=None):
    pred = (Z if offset is None else Z + offset).argmax(axis=1)
    return float(np.mean([np.mean(pred[y == c] == c)
                          for c in (0, 1) if (y == c).any()]))


# --- the dimensional bug that invalidated two gate designs -------------------

def test_tied_em_recovers_known_sigma():
    """Protocol §6's precondition. `sigma^2 = sum_k sum_i R_ik (s-mu_k)^2 / n`.

    Normalising by n_k first and then dividing by n again shrank sigma to 0.06
    against a truth of 1.131, which inflated snr to 42.
    """
    for n0, n1 in ((200, 200), (280, 120), (360, 40)):
        Z, y = make_logits(n0, n1, seed=0)
        mu, sd, _ = _em2(Z[:, 1] - Z[:, 0], tied=True)
        assert sd[0] == pytest.approx(sd[1])
        assert abs(sd[0] - 1.131) / 1.131 < 0.10, f"sigma={sd[0]} at {n0}:{n1}"
        assert abs(mu[0] - (-0.6)) < 0.4
        assert abs(mu[1] - 1.8) < 0.5


def test_tied_em_sigma_is_not_shrunk_by_a_second_division_by_n():
    """Directly pins the wrong formula out: the buggy value was ~1/n of the truth."""
    Z, _ = make_logits(200, 200, seed=1)
    _, sd, _ = _em2(Z[:, 1] - Z[:, 0], tied=True)
    assert sd[0] > 0.5, "sigma collapsed -- the tied M-step is normalising twice"


def test_free_em_degenerates_under_skew_which_is_why_it_only_vetoes():
    """Documents why the offset comes from the tied fit even though free fits better.

    At 90:10 the free fit puts mu_1 far below the truth (1.88) while inflating that
    component's sd. If a future change makes the free fit supply tau, this fails.
    """
    Z, _ = make_logits(360, 40, seed=0)
    s = Z[:, 1] - Z[:, 0]
    fmu, fsd, _ = _em2(s, tied=False)
    tmu, _, _ = _em2(s, tied=True)
    assert abs(fmu[1] - 1.8) > abs(tmu[1] - 1.8)


# --- marginal invariance: the mechanism claim -------------------------------

def test_tau_discards_the_mixture_weights():
    """Locations are class-conditional, weights are the marginal. tau reads only mu.

    Same class conditionals, very different mixes: tau must stay near the true bias
    while BC's correction moves.
    """
    taus, bcs = [], []
    for n0, n1 in ((200, 200), (280, 120), (340, 60)):
        Z, _ = make_logits(n0, n1, seed=7)
        taus.append(_tau(Z[:, 1] - Z[:, 0]))
        b = bc_offset(Z)
        bcs.append(b[1] - b[0])
    assert max(taus) - min(taus) < 0.45, f"tau moved with the mix: {taus}"
    assert abs(bcs[0] - bcs[-1]) > 0.15, f"BC did not move; test lost its contrast: {bcs}"


def test_bc_correction_slope_against_the_mix_is_systematic():
    """The defect being fixed: BC's slope is large and monotone in the mix.

    Sign convention: `bc_offset` returns the offset to be *added* to the logits, so
    its slope is the negation of the slope of the batch mean itself. The test asserts
    magnitude and monotonicity, which is what "systematic" means, rather than a sign
    that depends on that convention.
    """
    ratios = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    bcs = []
    for r in ratios:
        n = 400
        k = int(n * r)
        Z, _ = make_logits(k, n - k, seed=11)
        b = bc_offset(Z)
        bcs.append(b[1] - b[0])
    assert abs(marginal_slope(np.array(bcs), ratios)) > 0.4
    diffs = np.diff(bcs)
    assert np.all(diffs > 0) or np.all(diffs < 0), f"not monotone: {bcs}"


def test_marginal_slope_is_nan_without_variation_in_the_ratio():
    assert np.isnan(marginal_slope(np.array([1.0, 2.0]), np.array([0.5, 0.5])))


# --- the gate: each term, and the failure that forced it --------------------

def test_a_single_class_batch_abstains():
    """BC loses 10.94 pp here; abstention is the point."""
    for n0, n1 in ((128, 0), (0, 128)):
        Z, _ = make_logits(n0, n1, seed=1)
        res = sio_offset(Z)
        assert not res.applied
        assert res.reason in {"snr", "w_min"}


def test_an_unseparated_batch_abstains():
    Z, _ = make_logits(200, 200, sep=0.1, seed=2)
    assert not sio_offset(Z).applied


def test_a_small_batch_abstains_even_when_it_looks_separated():
    """n=16 showed snr 3.88 and oracle headroom of 9.09 pp; it still must abstain."""
    Z, _ = make_logits(11, 5, seed=5)
    res = sio_offset(Z)
    assert not res.applied and res.reason == "n_min"
    assert len(Z) < N_MIN


def test_unequal_class_conditional_variance_abstains_via_rho():
    """The most dangerous failure: it passed at snr=4.76 -- a *falsely high* snr --
    and lost 6.43 pp. A misspecified model produced false confidence."""
    for sds in ((0.4, 1.6), (1.6, 0.4)):
        Z, _ = make_logits(280, 120, sds=sds, seed=3)
        res = sio_offset(Z)
        assert not res.applied, f"sds={sds} snr={res.snr:.2f}"
        assert res.reason == "rho_misspecified"
        assert res.rho > res.rho_crit


def test_rho_critical_value_is_calibrated_not_constant():
    """A constant threshold cannot separate the two groups -- their rho ranges overlap.

    Skewed-but-tied batches must have a *higher* rho_crit than balanced ones, since
    rho is upward-biased under skew.
    """
    crits = []
    for r in (0.5, 0.9):
        n = 400
        k = int(n * r)
        Z, _ = make_logits(k, n - k, seed=0)
        crits.append(sio_offset(Z).rho_crit)
    assert np.isfinite(crits).all()
    assert crits[1] > crits[0], f"rho_crit did not adapt to skew: {crits}"


def test_a_batch_with_no_injected_bias_abstains_via_the_ci():
    """bias=0 at 90:10 passed a hand-picked |tau|/sigma >= 0.15 and lost 2.78 pp."""
    for r in (0.5, 0.7, 0.9):
        n = 400
        k = int(n * r)
        Z, _ = make_logits(k, n - k, bias=0.0, seed=4)
        res = sio_offset(Z)
        assert not res.applied, f"ratio={r} tau={res.tau:.3f}"
        assert res.reason == "tau_ci_contains_zero"
        assert res.ci[0] <= 0 <= res.ci[1]


def test_extreme_skew_abstains_on_most_seeds():
    """Pins §A2.1: at 90:10 the apply rate is ~1/20, not 20/20.

    §3's single-seed +6.94 pp was one lucky seed. If a future change makes this cell
    apply broadly, the CI gate has been loosened and the frozen thresholds violated.
    """
    applied = 0
    for s in range(6):
        Z, _ = make_logits(360, 40, seed=100 + s)
        applied += sio_offset(Z, seed=s).applied
    assert applied <= 1, f"{applied}/6 applied at 90:10 -- gate was loosened"


def test_the_middle_of_the_range_does_apply_and_beats_bc():
    """The claim that survives §A2.1: flat where BC decays, from 50:50 to 80:20."""
    for r in (0.5, 0.7, 0.8):
        n = 400
        k = int(n * r)
        Z, y = make_logits(k, n - k, seed=0)
        res = sio_offset(Z)
        assert res.applied, f"ratio={r} abstained: {res.reason}"
        raw = balanced_accuracy(Z, y)
        assert balanced_accuracy(Z, y, res.offset) >= raw


def test_sio_beats_bc_where_bc_has_inverted():
    """At 80:20 BC is below raw; SIO must be above it."""
    Z, y = make_logits(320, 80, seed=0)
    raw = balanced_accuracy(Z, y)
    bc = balanced_accuracy(Z, y, bc_offset(Z))
    sio = balanced_accuracy(Z, y, sio_offset(Z).offset)
    assert bc < raw, "BC did not invert; the test lost its point"
    assert sio > bc


# --- mechanical invariants required by protocol §6 --------------------------

def test_abstention_returns_exactly_zero_not_approximately_zero():
    Z, y = make_logits(200, 200, sep=0.1, seed=2)
    res = sio_offset(Z)
    assert not res.applied
    assert np.array_equal(res.offset, np.zeros(2))
    assert balanced_accuracy(Z, y, res.offset) == balanced_accuracy(Z, y)


def test_the_offset_is_gauge_invariant():
    Z, _ = make_logits(280, 120, seed=0)
    a = sio_offset(Z)
    b = sio_offset(Z + 3.7)
    assert a.applied and b.applied
    assert a.tau == pytest.approx(b.tau, abs=1e-9)
    np.testing.assert_allclose(a.offset, b.offset, atol=1e-9)


def test_the_offset_is_gauge_fixed_on_the_first_coordinate():
    Z, _ = make_logits(280, 120, seed=0)
    res = sio_offset(Z)
    assert res.applied and res.offset[0] == 0.0


def test_more_than_two_classes_is_rejected_rather_than_silently_handled():
    """Protocol §4 S6: SIO is defined for K_S=2 only."""
    with pytest.raises(ValueError, match="K_S=2"):
        sio_offset(np.zeros((60, 3)))


def test_the_result_is_immutable_so_a_scoring_pass_cannot_edit_the_gate():
    Z, _ = make_logits(280, 120, seed=0)
    res = sio_offset(Z)
    assert isinstance(res, SIOResult)
    with pytest.raises(Exception):
        res.applied = False  # type: ignore[misc]


def test_the_same_batch_and_seed_give_the_same_decision():
    Z, _ = make_logits(280, 120, seed=0)
    a, b = sio_offset(Z, seed=3), sio_offset(Z, seed=3)
    assert (a.applied, a.tau, a.rho_crit) == (b.applied, b.tau, b.rho_crit)
