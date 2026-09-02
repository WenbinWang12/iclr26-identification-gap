"""Phase-2H SGCR logic tests: spherical k-means, coherence/eligibility, robust
q-update (cap + mass retention), audit gate, codebook refresh alignment.

Pure numpy + scipy; no model, no network.  Run: python tests/test_sgcr.py
"""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgcr as S


def _blobs(k=4, d=8, per=60, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.eye(k, d)
    X, y = [], []
    for c in range(k):
        X.append(centers[c] + 0.05 * rng.standard_normal((per, d)))
        y += [c] * per
    return np.vstack(X), np.array(y)


def test_kmeans_purity_and_nonempty():
    X, y = _blobs()
    rng = np.random.default_rng(1)
    C, lab = S.spherical_kmeans(X, 4, rng)
    pur = sum(max(Counter(y[lab == c]).values()) for c in range(4)) / len(y)
    assert pur > 0.95, pur
    _, lab8 = S.spherical_kmeans(X, 8, rng)
    assert all((lab8 == c).sum() > 0 for c in range(8)), "empty cluster"


def test_eligibility_needs_train_and_audit_support():
    X, y = _blobs()
    rng = np.random.default_rng(1)
    C, lab = S.spherical_kmeans(X, 4, rng)
    coh = S.cell_coherence(X, lab, C, 4)
    tr = np.array([(lab == c).sum() for c in range(4)])
    assert S.eligible_cells(coh, tr, tr.copy())[1]
    au = tr.copy(); au[1] = 0
    assert not S.eligible_cells(coh, tr, au)[1]


def test_robust_q_update_cap_and_mass():
    q0 = np.array([0.4, 0.3, 0.2, 0.1])
    loss = np.array([1.0, 1.0, 1.0, 9.0])
    elig = np.ones(4, dtype=bool)
    q = S.robust_q_update(q0, loss, elig)
    assert abs(q.sum() - 1.0) < 1e-9
    assert q[3] > q0[3]
    assert q[3] / q0[3] <= S.ROBUST_CAP + 1e-9
    # no eligible cell -> unchanged
    assert np.allclose(S.robust_q_update(q0, loss, np.zeros(4, bool)), q0)


def test_audit_gate():
    B = S.AuditInputs(1.0, 1.0, 1.0, 0.9, 2.0, 1.5, 3)
    assert S.select_branch(B)[0] == "B"
    bad = S.AuditInputs(1.0, 1.0, 1.0, 0.9, 2.0, 2.5, 3)   # worst not improved
    assert S.select_branch(bad)[0] == "A"
    forced = S.AuditInputs(1, 1, 1, 1, 1, 0.5, 1)          # <2 eligible
    assert S.select_branch(forced)[0] == "A"
    nonfin = S.AuditInputs(1, 0.5, 1, 0.5, 1, 0.5, 3, finite=False)
    assert S.select_branch(nonfin)[0] == "A"


def test_codebook_refresh_alignment():
    X, _ = _blobs()
    rng = np.random.default_rng(1)
    C, _ = S.spherical_kmeans(X, 4, rng)
    cb = S.Codebook(centers=C, warmup_mean=np.zeros(8), k=4)
    before = cb.centers.copy()
    cb.refresh(X, rng)
    sims = np.sum(before * cb.centers, axis=1)
    assert (sims > 0.9).all(), sims


if __name__ == "__main__":
    test_kmeans_purity_and_nonempty()
    test_eligibility_needs_train_and_audit_support()
    test_robust_q_update_cap_and_mass()
    test_audit_gate()
    test_codebook_refresh_alignment()
    print("test_sgcr OK")
