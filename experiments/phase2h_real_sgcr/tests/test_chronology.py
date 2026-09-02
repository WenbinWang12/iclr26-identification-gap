"""Phase-2H F0 chronology + role-separation test on a synthetic stream.

Runnable as pytest OR as ``python tests/test_chronology.py``.
Imports numpy + stdlib only; no torch, no transformers, no network.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buffers import MinPlusSharedStore, role_of

SALT = "phase2h-protocol-salt-v1"


def _record(example_id: int, cell_id: int, priority: float, L: int = 8) -> dict:
    return {
        "input_ids": np.zeros(L, dtype=np.int32),
        "attention_mask": np.ones(L, dtype=np.uint8),
        "label": np.int8(example_id % 2),
        "priority": priority,
        "example_id": np.uint64(example_id),
        "cell_id": np.uint8(cell_id),
    }


def test_role_deterministic_and_balanced():
    for eid in (0, 7, 12345, 99999):
        assert role_of(eid, SALT) == role_of(eid, SALT)
    n = 10_000
    train = sum(1 for eid in range(n) if role_of(eid, SALT) == "train")
    frac = train / n
    assert 0.45 <= frac <= 0.55, f"role split not ~50/50: train frac={frac:.3f}"


def test_train_store_never_returns_audit_role():
    """Simulate disjoint stores: only train-role records are offered to the
    train store, so a train-only sample path can never surface an audit record."""
    store = MinPlusSharedStore(capacity=24, k_cells=2, min_per_cell=6)
    for eid in range(400):
        if role_of(eid, SALT) != "train":
            continue
        store.offer(_record(eid, cell_id=eid % 2, priority=float(eid % 7)))
    rng = np.random.default_rng(0)
    drawn = store.sample(64, rng)
    assert drawn, "expected a non-empty train sample"
    for r in drawn:
        assert role_of(int(r["example_id"]), SALT) == "train", (
            f"train-only sample returned audit-role id {int(r['example_id'])}"
        )


def test_replay_only_from_past_windows():
    """At window t, replay may contain only records from windows < t
    (protocol "Stream construction and chronology", rule 1 and step 6)."""
    n_windows = 12
    per_window = 96
    store = MinPlusSharedStore(capacity=64, k_cells=4, min_per_cell=6)
    rng = np.random.default_rng(1)
    # example_id encodes its window: eid = window * per_window + offset.
    for t in range(n_windows):
        # 1. replay drawn BEFORE offering current window -> only windows < t.
        replay = store.sample(16, rng)
        for r in replay:
            win = int(r["example_id"]) // per_window
            assert win < t, f"replay at window {t} leaked window {win}"
        # 6. only now offer current-window records to the store.
        for off in range(per_window):
            eid = t * per_window + off
            if role_of(eid, SALT) != "train":
                continue
            store.offer(_record(eid, cell_id=off % 4, priority=rng.random()))


def run_all():
    test_role_deterministic_and_balanced()
    test_train_store_never_returns_audit_role()
    test_replay_only_from_past_windows()
    print("OK")


if __name__ == "__main__":
    run_all()
