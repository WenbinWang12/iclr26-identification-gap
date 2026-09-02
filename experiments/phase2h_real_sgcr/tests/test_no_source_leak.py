"""Phase-2H F0 integrity test: no source/category leak, clean LoRA attach.

Runnable as pytest OR as ``python tests/test_no_source_leak.py``.
Imports numpy + torch + stdlib only; no transformers, no network.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buffers import MinPlusSharedStore, _FORBIDDEN_KEYS, role_of
from lora_bert import attach_lora_to_bert, integrity_report


def _make_record(example_id: int, cell_id: int = 0, L: int = 8) -> dict:
    return {
        "input_ids": np.arange(L, dtype=np.int32),
        "attention_mask": np.ones(L, dtype=np.uint8),
        "label": np.int8(1),
        "priority": 0.5,
        "example_id": np.uint64(example_id),
        "cell_id": np.uint8(cell_id),
    }


def test_store_rejects_forbidden_keys():
    store = MinPlusSharedStore(capacity=12, k_cells=2, min_per_cell=6)
    for bad in _FORBIDDEN_KEYS:
        rec = _make_record(1)
        rec[bad] = "electronics"
        try:
            store.offer(rec)
        except ValueError:
            continue
        raise AssertionError(f"store accepted forbidden key '{bad}'")


def test_records_carry_only_schema_keys():
    store = MinPlusSharedStore(capacity=12, k_cells=2, min_per_cell=6)
    for i in range(20):
        store.offer(_make_record(i, cell_id=i % 2))
    allowed = set(MinPlusSharedStore._REQUIRED_KEYS)
    for r in store._records:
        keys = set(r.keys())
        assert keys == allowed, f"unexpected record keys: {keys ^ allowed}"
        for bad in _FORBIDDEN_KEYS:
            assert bad not in keys


class _FakeSelfAttention(nn.Module):
    """Minimal module whose class name ends with 'SelfAttention', exposing
    nn.Linear query/key/value like a real BERT attention block."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)


class _TwoLayerFakeBert(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.layer0 = _FakeSelfAttention(hidden)
        self.layer1 = _FakeSelfAttention(hidden)
        self.pooler = nn.Linear(hidden, 2)  # a non-attention module, must stay frozen


def test_attach_lora_clean_and_low_rank():
    model = _TwoLayerFakeBert(hidden=16)
    wrapped = attach_lora_to_bert(model, rank=4, alpha=8.0)
    # query + value of two SelfAttention modules -> four wraps.
    assert len(wrapped) == 4, wrapped
    rep = integrity_report(model)
    assert rep["stray_trainable"] == [], rep["stray_trainable"]
    assert rep["max_rank"] <= 4, rep["max_rank"]
    # key projections and the pooler must remain frozen.
    trainable = set(rep["trainable_names"])
    for n in trainable:
        assert n.endswith("lora_A") or n.endswith("lora_B"), n


def test_materialize_record_schema_has_no_source():
    """A learner-facing record synthesized by prepare_marc must never carry a
    source/category key; assert_no_source_leak must catch it if one is added."""
    import prepare_marc as pm
    rec = {
        "input_ids": np.zeros(64, dtype=np.int32),
        "attention_mask": np.ones(64, dtype=np.uint8),
        "label": np.int8(1),
        "example_id": np.uint64(7),
    }
    pm.assert_no_source_leak(rec)  # clean
    for bad in ("source", "category", "product_category", "source_index"):
        polluted = dict(rec, **{bad: 3})
        try:
            pm.assert_no_source_leak(polluted)
        except ValueError:
            continue
        raise AssertionError(f"assert_no_source_leak missed '{bad}'")


def test_split_is_pure_function_of_example_id():
    """Cross-split leakage is impossible by construction: split depends only on
    the immutable example_id, so the same record can never land in two splits."""
    import prepare_marc as pm
    for eid in (1, 2, 99, 123456789, 2**63 + 5):
        assert pm.split_of(eid) == pm.split_of(eid)
        assert pm.split_of(eid) in ("train", "validation", "test")


def run_all():
    test_store_rejects_forbidden_keys()
    test_records_carry_only_schema_keys()
    test_attach_lora_clean_and_low_rank()
    test_materialize_record_schema_has_no_source()
    test_split_is_pure_function_of_example_id()
    # sanity: role split is deterministic
    assert role_of(123, "salt") == role_of(123, "salt")
    print("OK")


if __name__ == "__main__":
    run_all()
