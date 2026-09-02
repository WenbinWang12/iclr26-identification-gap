"""Tests for the offset-conflict measurement primitives."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.phase2j_offset_conflict.offsets import (
    TaskLogits,
    balanced_accuracy,
    conflict_statistics,
    fit_offset,
    fit_shared_offset,
    gauge_fix,
    identification_gap,
    natural_accuracy,
    _worst_case_quantization_radius,
)


def _binary_task(name, mu_pos, mu_neg, n=400, sd=1.0, seed=0):
    rng = np.random.default_rng(seed)
    npos = n // 2
    m_pos = rng.normal(mu_pos, sd, npos)
    m_neg = rng.normal(mu_neg, sd, n - npos)
    # logits[:,1] - logits[:,0] == margin
    logits = np.zeros((n, 2))
    logits[:npos, 1] = m_pos
    logits[npos:, 1] = m_neg
    labels = np.concatenate([np.ones(npos, int), np.zeros(n - npos, int)])
    return TaskLogits(name, ("false", "true"), logits, labels)


def test_gauge_fix_is_zero_mean_and_argmax_invariant():
    off = np.array([1.0, 3.0, -2.0])
    g = gauge_fix(off)
    assert abs(g.mean()) < 1e-12
    logits = np.random.default_rng(0).normal(size=(50, 3))
    assert np.array_equal((logits + off).argmax(1), (logits + g).argmax(1))


def test_adding_constant_does_not_change_accuracy():
    t = _binary_task("a", 1.5, -1.5)
    base = balanced_accuracy(t.logits, t.labels, np.zeros(2))
    shifted = balanced_accuracy(t.logits, t.labels, np.array([2.0, 2.0]))
    assert base == pytest.approx(shifted)


def test_fitted_offset_never_hurts_and_is_gauge_fixed():
    # a badly biased task: both classes present but boundary far off
    t = _binary_task("biased", 3.0, 0.5)
    off = fit_offset(t.logits, t.labels)
    assert abs(off.mean()) < 1e-12
    before = balanced_accuracy(t.logits, t.labels, None)
    after = balanced_accuracy(t.logits, t.labels, off)
    assert after >= before - 1e-12
    assert after > before  # this task genuinely needs a shift


def test_offset_recovers_known_bias():
    # margins shifted by +2 for both classes => optimal contrast is about -2
    rng = np.random.default_rng(3)
    n = 600
    m = np.concatenate([rng.normal(2.0 + 1.5, 0.7, n // 2),
                        rng.normal(2.0 - 1.5, 0.7, n // 2)])
    logits = np.zeros((n, 2)); logits[:, 1] = m
    labels = np.concatenate([np.ones(n // 2, int), np.zeros(n // 2, int)])
    off = fit_offset(logits, labels)
    contrast = off[1] - off[0]
    assert contrast < -1.0  # pushes back toward the true boundary


def test_identification_gap_zero_when_tasks_agree():
    a = _binary_task("a", 1.5, -1.5, seed=1)
    b = _binary_task("b", 1.5, -1.5, seed=2)
    shared = fit_shared_offset([a, b])
    for t in (a, b):
        own = fit_offset(t.logits, t.labels)
        rep = identification_gap(t, per_task_offset=own, shared_offset=shared)
        assert rep["Delta_id"] < 0.02  # no conflict -> shared offset suffices


def test_identification_gap_positive_under_conflict():
    # one task wants a positive contrast, the other a negative one
    a = _binary_task("wants_neg", 0.2, -3.0, seed=4)
    b = _binary_task("wants_pos", 3.0, -0.2, seed=5)
    shared = fit_shared_offset([a, b])
    gaps = []
    for t in (a, b):
        own = fit_offset(t.logits, t.labels)
        gaps.append(identification_gap(
            t, per_task_offset=own, shared_offset=shared)["Delta_id"])
    assert max(gaps) > 0.0
    assert all(g >= -1e-9 for g in gaps)  # R_orc >= R_shr always


def test_disjoint_verbalizers_give_no_conflict():
    """Prop 1: with disjoint label sets one shared vector serves both tasks."""
    a = _binary_task("a", 0.2, -3.0, seed=6)
    b = _binary_task("b", 3.0, -0.2, seed=7)
    b = TaskLogits("b", ("no", "yes"), b.logits, b.labels)  # disjoint tokens
    shared = fit_shared_offset([a, b])
    for t in (a, b):
        own = fit_offset(t.logits, t.labels)
        rep = identification_gap(t, per_task_offset=own, shared_offset=shared)
        assert rep["Delta_id"] < 0.03


def test_r_orc_at_least_r_shr_at_least_nothing():
    a = _binary_task("a", 0.5, -2.5, seed=8)
    b = _binary_task("b", 2.5, -0.5, seed=9)
    shared = fit_shared_offset([a, b])
    for t in (a, b):
        own = fit_offset(t.logits, t.labels)
        rep = identification_gap(t, per_task_offset=own, shared_offset=shared)
        assert rep["R_orc"] >= rep["R_shr"] - 1e-9
        assert rep["R_orc"] >= rep["R_raw"] - 1e-9


def test_quantization_radius_monotone_and_zero_when_enough_centres():
    X = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0]])
    r1 = _worst_case_quantization_radius(X, 1)
    r2 = _worst_case_quantization_radius(X, 2)
    r4 = _worst_case_quantization_radius(X, 4)
    assert r1 > r2 > 0.0
    assert r4 == 0.0


def test_conflict_statistics_reports_shared_group():
    a = _binary_task("QQP", 0.2, -3.0, seed=10)
    b = _binary_task("BoolQA", 3.0, -0.2, seed=11)
    offs = {t.task: fit_offset(t.logits, t.labels) for t in (a, b)}
    verbs = {t.task: t.verbalizer for t in (a, b)}
    stats = conflict_statistics(offs, verbs)
    assert stats["omega"] > 0.0
    assert "false|true" in stats["quantization_radius"]


def test_natural_and_balanced_differ_under_imbalance():
    rng = np.random.default_rng(12)
    n = 500
    npos = 450  # heavily imbalanced
    logits = np.zeros((n, 2))
    logits[:npos, 1] = rng.normal(1.0, 1.0, npos)
    logits[npos:, 1] = rng.normal(0.5, 1.0, n - npos)
    labels = np.concatenate([np.ones(npos, int), np.zeros(n - npos, int)])
    nat = natural_accuracy(logits, labels)
    bal = balanced_accuracy(logits, labels)
    assert nat != pytest.approx(bal)
