"""Phase-2H stream construction tests: determinism, RNG-order hash stability,
warmup/drift structure, min-2-per-source, phase boundaries, no example reuse.

Pure numpy; no network, no torch.  Run: python tests/test_stream.py
"""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stream as st


def _make_ids(per_pool=4000):
    ids, base = [], 0
    for _s in range(st.N_SOURCES):
        pols = []
        for _p in (0, 1):
            pols.append(list(range(base, base + per_pool)))
            base += per_pool
        ids.append(pols)
    return ids


def test_determinism_and_seed_sensitivity():
    fp_a = st.stream_fingerprint(st.build_from_ids(_make_ids(), seed=12345))
    fp_b = st.stream_fingerprint(st.build_from_ids(_make_ids(), seed=12345))
    fp_c = st.stream_fingerprint(st.build_from_ids(_make_ids(), seed=999))
    assert fp_a == fp_b, "same seed must reproduce the schedule byte-identically"
    assert fp_a != fp_c, "different seed must change the schedule"


def test_warmup_structure():
    plans = st.build_from_ids(_make_ids(), seed=7)
    assert len(plans) == st.N_WINDOWS == 60
    for wp in plans[: st.N_WARMUP_WINDOWS]:
        assert wp.phase == "warmup"
        assert len(wp.records) == st.WINDOW_SIZE
        cnt = wp.realized_source_counts()
        assert all(cnt[s] == st.WARMUP_PER_SOURCE for s in range(st.N_SOURCES)), cnt
        pc = Counter((r.source_index, r.polarity) for r in wp.records)
        assert all(v == 8 for v in pc.values()), "warmup: 8 per (source,polarity)"


def test_drift_structure_and_min2():
    plans = st.build_from_ids(_make_ids(), seed=7)
    for wp in plans[st.N_WARMUP_WINDOWS:]:
        assert wp.phase in ("A", "B", "C")
        assert len(wp.records) == st.WINDOW_SIZE
        m = 2 + (wp.window % 3)
        assert len(wp.selected_sources) == m
        cnt = wp.realized_source_counts()
        assert set(cnt) <= set(wp.selected_sources)
        assert all(c >= st.MIN_PER_SELECTED_SOURCE for c in cnt.values()), cnt


def test_phase_boundaries():
    plans = st.build_from_ids(_make_ids(), seed=7)
    assert plans[6].phase == "A" and plans[23].phase == "A"
    assert plans[24].phase == "B" and plans[41].phase == "B"
    assert plans[42].phase == "C" and plans[59].phase == "C"


def test_no_example_reuse():
    plans = st.build_from_ids(_make_ids(), seed=7)
    all_ids = [r.example_id for wp in plans for r in wp.records]
    assert len(all_ids) == len(set(all_ids)), "an example was scheduled twice"


if __name__ == "__main__":
    test_determinism_and_seed_sensitivity()
    test_warmup_structure()
    test_drift_structure_and_min2()
    test_phase_boundaries()
    test_no_example_reuse()
    print("test_stream OK")
