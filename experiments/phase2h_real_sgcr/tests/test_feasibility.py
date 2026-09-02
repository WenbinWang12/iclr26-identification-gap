"""Phase-2H feasibility-gate tests: probe balacc, AMI/purity, rare prec/rec, and
F0-F3 pass/fail branches at the exact protocol thresholds.

Pure numpy/scipy (no scikit-learn, per protocol dependency discipline).
Run: python tests/test_feasibility.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import feasibility as F


def _separable_signatures(k=6, d=128, per=120, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, d))
    sig, src = [], []
    for c in range(k):
        sig.append(centers[c] + 0.5 * rng.standard_normal((per, d)))
        src += [c] * per
    sig = np.vstack(sig)
    sig = sig / np.linalg.norm(sig, axis=1, keepdims=True)
    return sig, np.array(src)


def test_balanced_accuracy():
    yt = np.array([0, 0, 1, 1, 2, 2])
    assert F.balanced_accuracy(yt, yt, 3) == 1.0
    assert F.balanced_accuracy(yt, np.zeros_like(yt), 3) < 0.5


def test_probe_separates_then_chance():
    sig, src = _separable_signatures()
    assert F.linear_probe_balacc(sig, src, seed=0) > 0.75
    rng = np.random.default_rng(1)
    assert F.linear_probe_balacc(sig, rng.permutation(src), seed=0) < 0.35


def test_ami_purity():
    sig, src = _separable_signatures()
    ami, pur = F.kmeans_ami_purity(src.copy(), src)
    assert ami > 0.95 and pur > 0.99
    rng = np.random.default_rng(2)
    ami_r, _ = F.kmeans_ami_purity(rng.integers(0, 8, size=len(src)), src)
    assert ami_r < 0.1


def test_rare_prec_rec():
    _, src = _separable_signatures()
    p, r = F.rare_source_cell_prec_rec(src.copy(), src, rare_source=5)
    assert p >= 0.5 and r >= 0.5


def test_F0():
    ok = dict(single_polarity_map=True, splits_disjoint=True,
              no_source_in_learner=True, chronology_ok=True, role_disjoint=True)
    assert F.gate_F0(ok).passed
    bad = dict(ok, splits_disjoint=False)
    assert not F.gate_F0(bad).passed


def test_F1():
    sig, src = _separable_signatures()
    assert F.gate_F1(sig, src, src.copy(), rare_source=5, seed=0).passed


def test_F2():
    assert F.gate_F2([0.70, 0.72, 0.71], [0.66, 0.67, 0.66],
                     [0.80, 0.81, 0.80], [0.805, 0.812, 0.803]).passed
    assert not F.gate_F2([0.67, 0.67, 0.67], [0.66, 0.67, 0.66],
                         [0.80] * 3, [0.80] * 3).passed


def test_F3():
    assert F.gate_F3(0.75, 0.82, 0.70, 0.5, 1.7, 0.05, True, 4, True).passed
    assert not F.gate_F3(0.75, 0.82, 0.60, 0.5, 1.7, 0.05, True, 4, True).passed  # worst<0.65
    assert not F.gate_F3(0.75, 0.82, 0.70, 0.5, 1.7, 0.05, True, 5, True).passed  # rank>4
    assert not F.gate_F3(0.75, 0.82, 0.70, 0.5, 0.0, 0.05, True, 4, True).passed  # lora no effect


if __name__ == "__main__":
    for fn in [test_balanced_accuracy, test_probe_separates_then_chance,
               test_ami_purity, test_rare_prec_rec, test_F0, test_F1,
               test_F2, test_F3]:
        fn()
    print("test_feasibility OK")
