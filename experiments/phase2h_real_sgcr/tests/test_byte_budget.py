"""Phase-2H fixed historical-byte-budget test.

Runnable as pytest OR as ``python tests/test_byte_budget.py``.
Imports numpy + stdlib only; no torch, no transformers, no network.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buffers import (
    MinPlusSharedStore,
    capacity_from_envelope,
    history_envelope_bytes,
)


def _record(example_id: int, cell_id: int, k_cells: int, L: int) -> dict:
    return {
        "input_ids": np.zeros(L, dtype=np.int32),
        "attention_mask": np.ones(L, dtype=np.uint8),
        "label": np.int8(0),
        "priority": float(example_id % 11),
        "example_id": np.uint64(example_id),
        "cell_id": np.uint8(cell_id % k_cells),
    }


def test_envelope_matches_protocol():
    # protocol: 256 canonical non-cell records at L=64 -> 86,272 bytes.
    assert history_envelope_bytes(64, 256) == 86_272


def test_bytes_never_exceed_derived_capacity():
    for L in (16, 32, 64, 96):
        for k_cells in (2, 4, 8):
            M = history_envelope_bytes(64, 256)
            try:
                cap = capacity_from_envelope(M, L, k_cells)
            except ValueError:
                # this (L, k_cells) cannot host 6/cell under the envelope; skip.
                continue
            store = MinPlusSharedStore(capacity=cap, k_cells=k_cells, min_per_cell=6)
            rng = np.random.default_rng(7)
            # offer far more than capacity to force eviction pressure.
            for i in range(cap * 5 + 50):
                store.offer(_record(i, cell_id=int(rng.integers(k_cells)),
                                    k_cells=k_cells, L=L))
                assert len(store) <= cap
            # the per-store byte budget = half the envelope minus controller state.
            per_store_budget = M // 2
            assert store.bytes_used() <= per_store_budget, (
                f"L={L} k={k_cells}: bytes_used={store.bytes_used()} "
                f"> per_store_budget={per_store_budget}"
            )


def test_min_per_cell_honored_after_pressure():
    L, k_cells = 32, 4
    M = history_envelope_bytes(64, 256)
    cap = capacity_from_envelope(M, L, k_cells)
    store = MinPlusSharedStore(capacity=cap, k_cells=k_cells, min_per_cell=6)
    rng = np.random.default_rng(3)
    for i in range(cap * 6):
        store.offer(_record(i, cell_id=int(rng.integers(k_cells)),
                            k_cells=k_cells, L=L))
    cells = np.array([int(r["cell_id"]) for r in store._records])
    for c in range(k_cells):
        assert (cells == c).sum() >= 6, f"cell {c} under its minimum of 6"


def test_capacity_raises_when_six_per_cell_cannot_fit():
    # A tiny envelope cannot host 6 records/cell -> must raise, not silently grow.
    raised = False
    try:
        capacity_from_envelope(2_000, L=64, k_cells=8)
    except ValueError:
        raised = True
    assert raised, "expected capacity_from_envelope to raise on infeasible 6/cell"

    # Also raises when controller state alone exceeds the envelope.
    raised = False
    try:
        capacity_from_envelope(100, L=64, k_cells=8)
    except ValueError:
        raised = True
    assert raised


def run_all():
    test_envelope_matches_protocol()
    test_bytes_never_exceed_derived_capacity()
    test_min_per_cell_honored_after_pressure()
    test_capacity_raises_when_six_per_cell_cannot_fit()
    print("OK")


if __name__ == "__main__":
    run_all()
