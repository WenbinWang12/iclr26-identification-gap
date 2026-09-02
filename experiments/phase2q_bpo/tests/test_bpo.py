"""Tests for Batch Prior Offset.

Two of these are not ordinary unit tests but *protocol assertions*: §6 of
`notes/phase2q_batch_prior_offset_protocol.md` requires that "BPO uses no
labels" and "BPO stores no state" be checked by test rather than claimed in
prose.  `test_signature_has_no_labels` and `test_stores_nothing` are those
checks, and they are meant to fail loudly if someone later "improves" BPO by
feeding it supervision.

Three more deliberately pin *negative* facts, so that a later reader cannot
mistake the surrogate for the objective:
`test_uniform_target_is_not_accuracy`, `test_can_be_worse_than_no_offset`, and
`test_imbalanced_mix_changes_the_answer`.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from experiments.phase2j_offset_conflict.offsets import (
    balanced_accuracy, fit_offset, gauge_fix)
from experiments.phase2q_bpo.bpo import (
    fit_batch_prior_offset, predicted_histogram, stored_floats,
    subsample_imbalanced, uniformity_objective)


# ---------------------------------------------------------------- protocol §6

def test_signature_has_no_labels():
    """§6: no labels in the offset computation, asserted on the signature."""
    params = set(inspect.signature(fit_batch_prior_offset).parameters)
    assert "labels" not in params
    assert "y" not in params
    assert "targets" not in params
    # Only `logits` is positional; everything else is keyword-only tuning.
    positional = [n for n, p in
                  inspect.signature(fit_batch_prior_offset).parameters.items()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    assert positional == ["logits"]


def test_stores_nothing():
    """§6: zero stored floats, by construction."""
    assert stored_floats() == 0


def test_offset_is_gauge_fixed():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(64, 3)) + np.array([1.5, 0.0, -1.0])
    b = fit_batch_prior_offset(z)
    assert abs(float(b.mean())) < 1e-12


# ------------------------------------------------------------------ mechanics

def test_histogram_sums_to_one():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(32, 4))
    h = predicted_histogram(z)
    assert abs(h.sum() - 1.0) < 1e-12
    assert len(h) == 4


def test_removes_a_collapsed_class():
    """The failure mode BPO exists for: a bias so large one class never wins."""
    rng = np.random.default_rng(2)
    z = rng.normal(scale=0.5, size=(200, 2))
    z[:, 0] += 6.0                      # class 1 can never win
    assert predicted_histogram(z)[1] == 0.0
    b = fit_batch_prior_offset(z)
    h = predicted_histogram(z, b)
    assert h[1] > 0.3                   # class 1 is back in play
    assert uniformity_objective(z, b) > uniformity_objective(z, None)


def test_already_uniform_batch_is_left_alone():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(400, 2))
    before = uniformity_objective(z, None)
    b = fit_batch_prior_offset(z)
    after = uniformity_objective(z, b)
    assert after >= before - 1e-12
    assert abs(float(np.abs(b).max())) < 1.0    # no large gratuitous shift


def test_empty_batch_returns_zero_offset():
    b = fit_batch_prior_offset(np.zeros((0, 3)))
    assert b.shape == (3,)
    assert np.allclose(b, 0.0)


def test_rejects_non_matrix_logits():
    with pytest.raises(ValueError):
        fit_batch_prior_offset(np.zeros(5))


def test_objective_maximum_is_zero():
    """Exact uniformity scores 0; nothing can beat it."""
    z = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert abs(uniformity_objective(z, np.zeros(2))) < 1e-12


# ------------------------------------------------- negative facts, pinned down

def test_uniform_target_is_not_accuracy():
    """The surrogate and the objective genuinely differ.

    Constructed so that matching uniform costs balanced accuracy: the batch is
    truly imbalanced, so forcing a uniform *prediction* histogram must
    misclassify part of the majority class.  This is the gap criterion Q4
    measures, and it exists in the code, not just in the protocol's prose.
    """
    rng = np.random.default_rng(4)
    n0, n1 = 180, 20
    z = np.concatenate([
        np.stack([rng.normal(2.0, 0.5, n0), rng.normal(0.0, 0.5, n0)], axis=1),
        np.stack([rng.normal(0.0, 0.5, n1), rng.normal(2.0, 0.5, n1)], axis=1),
    ])
    y = np.array([0] * n0 + [1] * n1)

    b_bpo = fit_batch_prior_offset(z)
    b_sup = fit_offset(z, y)
    acc_bpo = balanced_accuracy(z, y, b_bpo)
    acc_sup = balanced_accuracy(z, y, b_sup)
    assert acc_sup > acc_bpo, "supervised fit must dominate on a skewed batch"
    # And the BPO histogram really is near-uniform despite the 9:1 truth.
    assert abs(predicted_histogram(z, b_bpo)[0] - 0.5) < 0.1


def test_can_be_worse_than_no_offset():
    """BPO is not guaranteed no-harm, which is why Q2 is a frozen criterion.

    Same skewed construction: the unmodified logits already classify nearly
    everything correctly, and forcing uniformity actively breaks that.
    """
    rng = np.random.default_rng(5)
    n0, n1 = 190, 10
    z = np.concatenate([
        np.stack([rng.normal(3.0, 0.4, n0), rng.normal(0.0, 0.4, n0)], axis=1),
        np.stack([rng.normal(0.0, 0.4, n1), rng.normal(3.0, 0.4, n1)], axis=1),
    ])
    y = np.array([0] * n0 + [1] * n1)
    b = fit_batch_prior_offset(z)
    assert balanced_accuracy(z, y, b) < balanced_accuracy(z, y, None) + 1e-9


def test_imbalanced_mix_changes_the_answer():
    """Q4's instrument must actually bite, else the criterion is vacuous."""
    rng = np.random.default_rng(6)
    z = np.concatenate([
        np.stack([rng.normal(1.5, 0.6, 128), rng.normal(0.0, 0.6, 128)], axis=1),
        np.stack([rng.normal(0.0, 0.6, 128), rng.normal(1.5, 0.6, 128)], axis=1),
    ])
    y = np.array([0] * 128 + [1] * 128)

    idx = subsample_imbalanced(y, 0.9, seed=7)
    share = float((y[idx] == 0).mean())
    assert 0.85 < share < 0.95, "the mix must really be 90:10, got %.3f" % share
    b_bal = fit_batch_prior_offset(z)
    b_skew = fit_batch_prior_offset(z[idx])
    assert not np.allclose(b_bal, b_skew, atol=1e-6)


def test_subsample_ratios_and_guards():
    y = np.array([0] * 64 + [1] * 64)
    for ratio in (0.5, 0.7, 0.9):
        idx = subsample_imbalanced(y, ratio, seed=1)
        share = float((y[idx] == 0).mean())
        assert abs(share - ratio) < 0.06, (ratio, share)
        assert len(set(idx.tolist())) == len(idx), "no replacement"
    with pytest.raises(ValueError):
        subsample_imbalanced(y, 0.0, seed=1)
    with pytest.raises(ValueError):
        subsample_imbalanced(np.array([0] * 8), 0.5, seed=1)


def test_multiclass_recovers_all_classes():
    rng = np.random.default_rng(8)
    K = 4
    z = rng.normal(scale=0.5, size=(400, K))
    z[:, 0] += 5.0                      # only class 0 ever wins
    assert (predicted_histogram(z) > 0).sum() == 1
    b = fit_batch_prior_offset(z)
    assert (predicted_histogram(z, b) > 0).sum() >= 3
    assert abs(float(gauge_fix(b).mean())) < 1e-12