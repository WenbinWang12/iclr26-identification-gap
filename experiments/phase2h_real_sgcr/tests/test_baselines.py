"""Phase-2H baseline buffer tests: reservoir capacity, persistent-loss hard
mixture, oracle inverse-frequency equalization, arm wiring / step budgets.

Pure numpy; no model, no network.  Run: python tests/test_baselines.py
"""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import baselines as B


def _rec(i):
    return {"input_ids": np.zeros(8, np.int32), "attention_mask": np.ones(8, np.uint8),
            "label": np.int8(i % 2), "example_id": np.uint64(i)}


def test_reservoir_capacity_and_sample():
    rng = np.random.default_rng(0)
    r = B.UniformReservoir(50)
    r.offer([_rec(i) for i in range(500)], rng)
    assert len(r) == 50
    assert len(r.sample_replay(16, rng)) == 16


def test_persistent_loss_hard_mixture():
    rng = np.random.default_rng(0)
    pl = B.PersistentLossBuffer(100)
    pl.offer([_rec(i) for i in range(100)], [float(i) for i in range(100)], rng)
    cnt, N = 0, 4000
    for _ in range(N // 16):
        for x in pl.sample_replay(16, rng):
            if int(x["example_id"]) >= 80:
                cnt += 1
    frac = cnt / ((N // 16) * 16)
    assert frac > 0.3, f"hard mixture not over-sampling high-loss: {frac}"


def test_oracle_inverse_frequency():
    rng = np.random.default_rng(0)
    osrc = B.OracleGroupStore(120, n_sources=3, min_per_source=6)
    r2 = np.random.default_rng(1)
    srcs, recs = [], []
    for i in range(3000):
        u = r2.random()
        s = 0 if u < 0.9 else (1 if u < 0.99 else 2)
        srcs.append(s); recs.append(_rec(i))
    osrc.offer(recs, srcs, rng)
    id2src = {int(r["example_id"]): s for r, s in zip(recs, srcs)}
    c = Counter()
    for _ in range(300):
        for x in osrc.sample_replay(16, rng):
            c[id2src[int(x["example_id"])]] += 1
    tot = sum(c.values())
    assert c[2] / tot > 0.2, f"rare source not equalized: {c[2] / tot}"


def test_arm_wiring_and_budgets():
    arms = B.make_arms({"er": 200, "cell": 100, "ploss": 150, "oracle": 120},
                       n_sources=6)
    assert arms["er_flop"].n_deploy_steps == 16
    assert arms["er_step"].n_deploy_steps == 12
    assert arms["cell_q0_flop"].n_deploy_steps == 16
    assert arms["sequential"].buffer is None
    assert arms["oracle"].is_oracle


if __name__ == "__main__":
    test_reservoir_capacity_and_sample()
    test_persistent_loss_hard_mixture()
    test_oracle_inverse_frequency()
    test_arm_wiring_and_budgets()
    print("test_baselines OK")
