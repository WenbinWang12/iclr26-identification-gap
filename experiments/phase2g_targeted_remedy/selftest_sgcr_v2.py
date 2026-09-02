"""Lightweight invariant tests for :mod:`explore_sgcr_v2`.

The test uses one prefix of an existing development world.  Latent source ids
are used only by the benchmark generator and are never passed to the learner.
Run from the repository root with::

    python experiments/phase2g_targeted_remedy/selftest_sgcr_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import explore_gcdr as E  # noqa: E402
import explore_sgcr_v2 as S  # noqa: E402


PREFIX_WINDOWS = 24
RUN_KWARGS = {
    "warmup_windows": 16,
    "refresh_every": PREFIX_WINDOWS,
    "rounds": 0,
    "blend_grid": (),
    "return_debug": True,
}


def _dev_batches():
    """Materialize X/Y only; no latent source metadata reaches SGCR."""
    cfg = E._cfg()
    teacher_seed = E.DEV_TEACHERS[0]
    stream_seed = E.DEV_STREAMS[0]
    teacher = E.BM.build_teacher(E.REGIME, cfg, teacher_seed)
    windows, _ = E.BM.make_stream(cfg, stream_seed)
    batches, _, _, _ = E.BM.materialize(
        teacher, cfg, windows[:PREFIX_WINDOWS], stream_seed
    )
    return cfg, batches


def _effective_rank(model, relative_tolerance=1e-8):
    singular_values = np.linalg.svd(model, compute_uv=False)
    if singular_values[0] == 0.0:
        return 0
    return int(np.sum(singular_values > relative_tolerance * singular_values[0]))


def _assert_common_invariants(cfg, model, stats):
    train_ids = set(stats["debug_train_ids"])
    audit_ids = set(stats["debug_audit_ids"])
    assert train_ids.isdisjoint(audit_ids), "train/audit IDs overlap"

    assert stats["persistent_numeric_floats"] <= stats["numeric_budget_floats"]
    assert stats["actual_train_entries"] <= stats["train_capacity"]
    assert stats["actual_audit_entries"] <= stats["audit_capacity"]
    actual_occupancy = (
        stats["actual_train_entries"] + stats["actual_audit_entries"]
    )
    total_capacity = stats["train_capacity"] + stats["audit_capacity"]
    assert actual_occupancy <= total_capacity

    assert np.isfinite(model).all(), "learner returned a non-finite model"
    assert _effective_rank(model) <= cfg["R"], "returned model exceeds LoRA rank"
    frequencies = np.asarray(stats["arrival_frequencies"])
    assert np.isfinite(frequencies).all()
    np.testing.assert_allclose(frequencies.sum(), 1.0, atol=1e-12)


def _run_and_capture_frequency_baseline(cfg, batches):
    """Record the independently recomputed last-fit frequency baseline."""
    original_select = S._select_same_pool
    captures = []

    def recording_select(train_grouped, audit_grouped, frequencies,
                         robust_mask, rank, ridge, rounds, eta, mean_margin,
                         min_audit_gain, blend_grid):
        q0 = np.asarray(frequencies, dtype=float)
        q0 /= q0.sum()
        expected = S._fit_groups(train_grouped, q0, rank, ridge)
        result = original_select(
            train_grouped, audit_grouped, frequencies, robust_mask, rank,
            ridge, rounds, eta, mean_margin, min_audit_gain, blend_grid
        )
        captures.append((expected, q0))
        return result

    S._select_same_pool = recording_select
    try:
        model, stats = S.run_sgcr_v2(cfg, batches, E.RUN_SEED, **RUN_KWARGS)
    finally:
        S._select_same_pool = original_select

    # refresh_every equals the prefix length, hence exactly one final fit.
    assert len(captures) == 1
    expected_model, expected_weights = captures[0]
    assert stats["selected_candidate"] == "frequency_baseline"
    assert stats["selected_round"] == 0
    assert stats["feasible_candidates"] == 1
    np.testing.assert_allclose(stats["selected_weights"], expected_weights)
    np.testing.assert_allclose(model, expected_model, rtol=1e-11, atol=1e-11)
    return model, stats


def main():
    cfg, batches = _dev_batches()

    model, stats = _run_and_capture_frequency_baseline(cfg, batches)
    _assert_common_invariants(cfg, model, stats)
    assert stats["max_groups"] == 2 * cfg["R"]
    print("PASS: disjoint roles, persistent budget, occupancy, rank, baseline")

    # K is planted-world metadata.  Once X/Y batches exist, changing K must not
    # alter the learner's default resolution budget or any deterministic output.
    cfg_changed_k = dict(cfg)
    cfg_changed_k["K"] = cfg["K"] + 37
    model_changed_k, stats_changed_k = S.run_sgcr_v2(
        cfg_changed_k, batches, E.RUN_SEED, **RUN_KWARGS
    )
    _assert_common_invariants(cfg_changed_k, model_changed_k, stats_changed_k)
    assert stats_changed_k["max_groups"] == 2 * cfg["R"]
    np.testing.assert_allclose(model_changed_k, model, rtol=0.0, atol=0.0)
    assert stats_changed_k["debug_train_ids"] == stats["debug_train_ids"]
    assert stats_changed_k["debug_audit_ids"] == stats["debug_audit_ids"]
    np.testing.assert_allclose(
        stats_changed_k["arrival_frequencies"], stats["arrival_frequencies"],
        rtol=0.0, atol=0.0
    )
    print("PASS: default max_groups is independent of cfg['K']")

    # Traversal order changes which examples receive particular random
    # reservoir keys, so bitwise model equality is intentionally not required.
    # The distributional-mode conclusion must still be a valid frequency
    # baseline, and all structural invariants must survive without source ids.
    permutation_rng = np.random.default_rng(73021)
    permuted_batches = []
    for X, Y in batches:
        permutation = permutation_rng.permutation(len(X))
        permuted_batches.append((X[permutation], Y[permutation]))
    permuted_model, permuted_stats = S.run_sgcr_v2(
        cfg, permuted_batches, E.RUN_SEED, **RUN_KWARGS
    )
    _assert_common_invariants(cfg, permuted_model, permuted_stats)
    assert permuted_stats["selected_candidate"] == "frequency_baseline"
    assert permuted_stats["selected_round"] == 0
    print("PASS: within-window row permutation smoke/distribution invariants")

    # K_max is an upper bound: fewer observable modes must safely collapse to a
    # train-only frequency baseline instead of producing empty-center NaNs.
    degenerate_cfg = {
        "R": 2, "buf_u": 4, "buf_g": 12, "batch": 16, "ridge": 1e-6
    }
    one_x = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (16, 1))
    one_mode = [(one_x.copy(), one_x.copy()) for _ in range(24)]
    one_model, one_stats = S.run_sgcr_v2(
        degenerate_cfg, one_mode, E.RUN_SEED,
        warmup_windows=4, refresh_every=4
    )
    assert np.isfinite(one_model).all()
    assert _effective_rank(one_model) <= degenerate_cfg["R"]
    assert one_stats["selected_candidate"].startswith("frequency_baseline")
    assert one_stats["selected_round_zero_rate"] == 1.0
    print("PASS: K_max upper-bound / duplicate-mode safe fallback")

    print(
        "ALL PASS: occupancy=%d/%d, persistent_floats=%d/%d, rank=%d"
        % (
            stats["actual_train_entries"] + stats["actual_audit_entries"],
            stats["train_capacity"] + stats["audit_capacity"],
            stats["persistent_numeric_floats"],
            stats["numeric_budget_floats"],
            _effective_rank(model),
        )
    )


if __name__ == "__main__":
    main()
