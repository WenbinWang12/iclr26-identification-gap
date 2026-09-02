from __future__ import annotations

import tempfile
import unittest
import importlib.util
import json
from pathlib import Path
from unittest import mock

import numpy as np

from .artifacts import canonical_array_bytes, hash_arrays, hash_json, write_json
from .core_numpy import (
    BridgeConfig,
    PROTOCOL_SHA256,
    RANKS,
    REGIMES,
    alpha_values,
    assert_protocol_hash,
    bootstrap_gap_fraction_replicates,
    bootstrap_index_matrices,
    bootstrap_summary,
    build_prefix_oracle,
    build_seed_dataset,
    make_base_draws,
    make_factor_initializations,
    make_seed_streams,
    mode_count,
    mode_schedule,
    population_risk,
    q_min_keep,
    run_all_sanity,
    sanity_contraction_bound,
    weighted_modal_svd,
)
from .replay import B_MAX, REPLAY_BYTES, ClockBalancedReplay, ReservoirReplay, ReplayState, clock_quotas
from .phase2_gpu_bridge import _derive_contrasts, _post_run_gates, _split_manifest_entry, main, run_numpy_smoke


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "notes" / "phase2_gpu_bridge_protocol.md"


class Phase2BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = BridgeConfig()
        cls.base = make_base_draws(cls.config)

    def test_protocol_hash_and_fixed_bytes(self):
        self.assertEqual(assert_protocol_hash(PROTOCOL_PATH), PROTOCOL_SHA256)
        self.assertEqual(self.config.fixed_replay_bytes, 93488)
        self.assertEqual(REPLAY_BYTES, 93488)
        self.assertTrue(self.config.is_protocol_locked)

    def test_base_draws_are_deterministic_and_orthonormal(self):
        second = make_base_draws(self.config)
        self.assertEqual(self.base.sha256(), second.sha256())
        for name in ("compatible_U", "compatible_V", "stress_U", "stress_V"):
            matrix = getattr(self.base, name)
            self.assertLessEqual(float(np.linalg.norm(matrix.T @ matrix - np.eye(matrix.shape[1]))), 1e-10)
            for column in range(matrix.shape[1]):
                row = int(np.argmax(np.abs(matrix[:, column])))
                self.assertGreaterEqual(float(matrix[row, column]), 0.0)

    def test_seed_dataset_counts_pi_and_split_namespaces(self):
        dataset = build_seed_dataset(self.base, 11, self.config)
        self.assertEqual(len(dataset.cells), 6)
        for (regime, rank), windows in dataset.cells.items():
            for window in windows:
                self.assertEqual(window.train.count, 1024)
                self.assertEqual(window.eval.count, 2048)
                self.assertEqual(np.unique(window.pi_t).size, 1024)
                self.assertTrue(np.array_equal(np.sort(window.pi_t), np.arange(1024)))
                self.assertTrue(np.array_equal(np.unique(window.train.global_keys[:, 3]), np.array([0])))
                self.assertTrue(np.array_equal(np.unique(window.eval.global_keys[:, 3]), np.array([1])))
                self.assertTrue(np.all(window.train.global_keys[:, 3] == 0))
                self.assertTrue(np.all(window.eval.global_keys[:, 3] == 1))
                train_counts = np.bincount(window.train.mode_ids, minlength=mode_count(regime, rank))
                eval_counts = np.bincount(window.eval.mode_ids, minlength=mode_count(regime, rank))
                current, previous = mode_schedule(regime, rank, window.window)
                if window.window == 1:
                    self.assertEqual(int(train_counts[current]), 1024)
                    self.assertEqual(int(eval_counts[current]), 2048)
                else:
                    self.assertEqual(int(train_counts[current]), 768)
                    self.assertEqual(int(train_counts[int(previous)]), 256)
                    self.assertEqual(int(eval_counts[current]), 1536)
                    self.assertEqual(int(eval_counts[int(previous)]), 512)
                for block in (window.train, window.eval):
                    for mode in np.unique(block.mode_ids):
                        signs = block.signs[block.mode_ids == mode]
                        self.assertEqual(int(np.count_nonzero(signs == 1)), signs.size // 2)
                        self.assertEqual(int(np.count_nonzero(signs == -1)), signs.size // 2)
        manifest = _split_manifest_entry(dataset, self.base, self.config)
        self.assertTrue(manifest["deterministic_rebuild_passed"])
        self.assertTrue(manifest["base_finite_passed"])
        self.assertTrue(manifest["basis_orthogonality_passed"])
        for cell in manifest["cells"].values():
            for item in cell["integrity"]:
                self.assertTrue(item["pi_matches_train_storage_permutation"])
                self.assertTrue(item["pi_is_full_permutation"])
                for block in (item["train"], item["eval"]):
                    self.assertTrue(all(value for key, value in block.items() if key.endswith("_passed")))
                    self.assertTrue(block["mode_counts_match"])

    def test_oracles_capacity_and_ranks(self):
        for regime in REGIMES:
            for rank in RANKS:
                for window in range(1, self.config.T + 1):
                    oracle = build_prefix_oracle(self.base, regime, rank, window)
                    self.assertLessEqual(len(oracle.selected), rank)
                    self.assertAlmostEqual(oracle.loss_rank, oracle.capacity_tail, delta=1e-10)
                    self.assertGreater(oracle.D_t, 1e-8)
                    self.assertGreater(oracle.D_current, 1e-8)
                    if regime == "compatible":
                        self.assertLessEqual(oracle.capacity_tail / oracle.D_t, 1e-10)
                    direct = population_risk(oracle.delta_rank, *self.base.basis(regime, rank), oracle.alpha, oracle.q)
                    self.assertAlmostEqual(direct, oracle.loss_rank, delta=1e-10)
                    svd_delta, svd_selected, _ = weighted_modal_svd(
                        *self.base.basis(regime, rank), oracle.alpha, oracle.q, rank
                    )
                    self.assertEqual(svd_selected, oracle.selected)
                    self.assertAlmostEqual(
                        population_risk(svd_delta, *self.base.basis(regime, rank), oracle.alpha, oracle.q),
                        oracle.loss_rank,
                        delta=1e-10,
                    )

    def test_weighted_modal_svd_uses_numerical_spectrum_and_canonical_ties(self):
        U = np.eye(32, 3, dtype=np.float64)
        V = np.eye(32, 3, dtype=np.float64)
        alpha = np.ones(3, dtype=np.float64)
        weights = np.ones(3, dtype=np.float64)
        _, selected, singular_values = weighted_modal_svd(U, V, alpha, weights, 1)
        self.assertEqual(selected, (0,))
        self.assertTrue(np.allclose(singular_values[:3], 1.0))

        original_svd = np.linalg.svd

        def corrupted_svd(*args, **kwargs):
            left, values, right = original_svd(*args, **kwargs)
            values = values.copy()
            values[0] *= 0.9
            return left, values, right

        with mock.patch("experiments.phase2_gpu_bridge.core_numpy.np.linalg.svd", side_effect=corrupted_svd):
            with self.assertRaises(AssertionError):
                weighted_modal_svd(U, V, alpha, weights, 1)

    def test_sanity_support_and_contraction(self):
        sanity = run_all_sanity(self.base, self.config)
        self.assertEqual(len(sanity), 72)
        self.assertAlmostEqual(q_min_keep(sanity), 1.0 / 12.0, delta=0.0)
        self.assertLess(sanity_contraction_bound(q_min_keep(sanity)), 1.2e-19)
        for result in sanity.values():
            self.assertTrue(all(selected == result.oracle.selected for selected in result.pre_selected))
            self.assertTrue(all(support == result.oracle.selected for support in result.post_support))
            self.assertTrue(all(residual <= tau for residual, tau in zip(result.residual_norms, result.taus)))
            self.assertLessEqual(result.final_normalized_error, 1e-4)

    def test_bootstrap_shapes_and_reuse_contract(self):
        all_indices, valid_indices = bootstrap_index_matrices(self.config, 16)
        self.assertEqual(all_indices.shape, (10000, 20))
        self.assertEqual(valid_indices.shape, (10000, 16))
        self.assertEqual(all_indices.dtype, np.int64)
        values = np.linspace(0.1, 0.9, 16)
        reps = bootstrap_gap_fraction_replicates(values, valid_indices)
        self.assertEqual(reps.shape, (10000,))
        summary = bootstrap_summary({"metric": np.arange(20.0)}, values, self.config)
        self.assertEqual(summary["all_seed_index_shape"], [10000, 20])
        self.assertEqual(summary["valid_gap_index_shape"], [10000, 16])
        self.assertEqual(summary["statistics"]["gap_fraction"]["valid_count"], 16)

    def test_replay_state_quota_and_sentinels(self):
        state = ReplayState()
        self.assertEqual(state.n_occupied, 0)
        self.assertTrue(np.all(state.priority_slots == 1.0))
        self.assertTrue(np.all(state.x_slots == 0.0))
        self.assertEqual(state.x_slots.nbytes + state.y_slots.nbytes + state.priority_slots.nbytes + state.occupied.nbytes + state.window_counts.nbytes + state.window_offsets.nbytes + state.write_ptr.nbytes + state.eligible_count.nbytes, 93488)
        self.assertEqual([int(clock_quotas(h).sum()) for h in range(1, 12)], [352] * 11)

    def test_replay_transitions_and_rng_boundary(self):
        x = np.arange(1024 * 32, dtype=np.float32).reshape(1024, 32)
        y = -x
        clock = ClockBalancedReplay()
        reservoir = ReservoirReplay()
        clock_rng = np.random.default_rng(3)
        reservoir_rng = np.random.default_rng(4)
        clock.insert_completed_window(1, x, y, clock_rng)
        reservoir.insert_completed_window(1, x, y, reservoir_rng)
        self.assertEqual(clock.state.n_occupied, B_MAX)
        self.assertEqual(reservoir.state.n_occupied, B_MAX)
        self.assertEqual(int(clock.state.window_counts[0]), B_MAX)
        self.assertTrue(np.all(reservoir.state.window_counts == 0))
        for window in range(2, 12):
            clock.insert_completed_window(window, x, y, np.random.default_rng(window))
            reservoir.insert_completed_window(window, x, y, np.random.default_rng(window))
        self.assertEqual(clock.state.n_occupied, B_MAX)
        self.assertEqual(reservoir.state.n_occupied, B_MAX)
        self.assertEqual(int(clock.state.eligible_count[0]), 11264)
        self.assertEqual(int(reservoir.state.eligible_count[0]), 11264)
        self.assertEqual(clock.sample_indices(np.random.default_rng(9), 64).shape, (64,))
        self.assertEqual(reservoir.sample_indices(np.random.default_rng(9), 64).shape, (64,))
        self.assertEqual(clock.persistent_sidecar_records, 0)
        self.assertEqual(reservoir.persistent_sidecar_records, 0)
        self.assertEqual(clock.persistent_sidecar_bytes, 0)
        self.assertEqual(reservoir.persistent_sidecar_bytes, 0)
        self.assertEqual(clock.transient_records_peak, 1024)
        self.assertEqual(reservoir.transient_records_peak, B_MAX + 1024)
        self.assertGreater(clock.transient_python_bytes_peak, 0)
        self.assertGreater(reservoir.transient_numpy_bytes_peak, 0)
        self.assertGreater(clock.comparison_count, 0)
        self.assertGreater(reservoir.comparison_count, 0)

    def test_replay_matches_naive_reference_and_exact_priority_ties(self):
        def block(window):
            x = np.zeros((1024, 32), dtype=np.float32)
            x[:, 0] = window * 10000 + np.arange(1024)
            return x, -x

        for controller_type, seed in ((ClockBalancedReplay, 17), (ReservoirReplay, 19)):
            controller = controller_type()
            controller_rng = np.random.default_rng(seed)
            reference_rng = np.random.default_rng(seed)
            records = []
            for window in range(1, 12):
                x, y = block(window)
                priorities = reference_rng.random(1024, dtype=np.float64)
                records.extend(
                    (float(priorities[index]), (0, 0, window, 0, index), float(x[index, 0]))
                    for index in range(1024)
                )
                controller.insert_completed_window(window, x, y, controller_rng)
                if controller_type is ClockBalancedReplay:
                    selected = []
                    for history_window, quota in enumerate(clock_quotas(window), start=1):
                        selected.extend(
                            sorted(
                                (item for item in records if item[1][2] == history_window),
                                key=lambda item: (item[0], item[1]),
                            )[: int(quota)]
                        )
                else:
                    selected = sorted(records, key=lambda item: (item[0], item[1]))[:B_MAX]
                    records = selected
                self.assertTrue(
                    np.array_equal(
                        controller.state.priority_slots,
                        np.asarray([item[0] for item in selected], dtype=np.float64),
                    )
                )
                self.assertTrue(
                    np.array_equal(
                        controller.state.x_slots[:, 0],
                        np.asarray([item[2] for item in selected], dtype=np.float32),
                    )
                )

        class FixedPriorities:
            def __init__(self, values):
                self.values = list(values)

            def random(self, size, dtype):
                value = self.values.pop(0)
                self_outer.assertEqual(value.shape, (size,))
                self_outer.assertEqual(value.dtype, dtype)
                return value.copy()

        self_outer = self
        first = np.linspace(0.0, 1.0, 1024, endpoint=False, dtype=np.float64)
        first[1] = first[0]
        second = np.linspace(0.5, 1.0, 1024, endpoint=False, dtype=np.float64)
        second[0] = first[0]
        reservoir = ReservoirReplay()
        x1, y1 = block(1)
        x2, y2 = block(2)
        fixed = FixedPriorities((first, second))
        reservoir.insert_completed_window(1, x1, y1, fixed)
        transition = reservoir.insert_completed_window(2, x2, y2, fixed)
        self.assertEqual(reservoir.state.x_slots[:3, 0].tolist(), [10000.0, 10001.0, 20000.0])
        self.assertGreaterEqual(transition.priority_tie_count, 1)

    def test_factor_initializations_and_paired_contrasts(self):
        first = make_factor_initializations(make_seed_streams(11, self.config), self.config)
        second = make_factor_initializations(make_seed_streams(11, self.config), self.config)
        self.assertEqual(set(first), {(regime, rank) for regime in REGIMES for rank in RANKS})
        for cell, init in first.items():
            self.assertEqual(init.sha256(), second[cell].sha256())
            self.assertEqual(init.A32.shape, (cell[1], 32))
            self.assertEqual(init.B32.shape, (32, cell[1]))
            self.assertLessEqual(float(np.linalg.norm(init.A64 @ init.A64.T - np.eye(cell[1]))), 1e-10)
            self.assertTrue(np.all(init.B32 == 0.0))

        rows = [
            {"seed": 1, "regime": "compatible", "R": 4, "window": 12, "arm": "sequential_dense_rank_R", "D_t": 2.0, "gap": 1.0},
            {"seed": 1, "regime": "compatible", "R": 4, "window": 12, "arm": "clock_balanced_replay_rank_R", "D_t": 2.0, "gap": 0.25},
            {"seed": 1, "regime": "compatible", "R": 4, "window": 12, "arm": "reservoir_random_replay_rank_R", "D_t": 2.0, "gap": 0.5},
        ]
        _derive_contrasts(rows)
        for row in rows:
            self.assertAlmostEqual(row["bounded_history_reduction"], 0.375)
            self.assertAlmostEqual(row["selection_gain"], 0.125)
            self.assertAlmostEqual(row["gap_fraction"], 0.75)

    def test_primary_gate_bootstrap_contract(self):
        rows = []
        gaps = {
            "sequential_dense_rank_R": 1.0,
            "clock_balanced_replay_rank_R": 0.2,
            "reservoir_random_replay_rank_R": 0.4,
        }
        for seed in self.config.primary_seeds:
            for window in range(1, self.config.T + 1):
                for arm, gap in gaps.items():
                    rows.append(
                        {
                            "seed": seed,
                            "regime": "compatible",
                            "R": 4,
                            "window": window,
                            "arm": arm,
                            "gap": gap,
                            "D_t": 1.0,
                            "online_ratio": 0.2 if arm == "sequential_dense_rank_R" else gap,
                            "current_acquisition_ratio": 0.01,
                            "bounded_history_reduction": 0.8,
                            "selection_gain": 0.2,
                        }
                    )
        gates = _post_run_gates(rows, [], self.config, "primary")
        self.assertTrue(gates["G2_phenomenon"]["passed"])
        self.assertTrue(gates["G3_bounded_history"]["passed"])
        self.assertEqual(gates["G3_bounded_history"]["valid_positive_gap_count"], 20)
        self.assertAlmostEqual(gates["G3_bounded_history"]["gap_fraction_median"], 0.8)

    def test_numpy_smoke_rejects_primary_role(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_numpy_smoke(Path(directory) / "bad", "primary")

    def test_cli_failure_retains_strict_failure_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed"
            exit_code = main(
                [
                    "--mode",
                    "smoke",
                    "--run-role",
                    "primary",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 2)
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(failure["execution_pass"])
            self.assertTrue(failure["positive_claim_blocked"])
            self.assertFalse(manifest["complete"])
            self.assertEqual(manifest["failure_sha256"], manifest["partial_output_hashes"]["failure.json"])

            occupied = Path(directory) / "occupied"
            occupied.mkdir()
            (occupied / "keep.txt").write_text("keep", encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "--mode",
                        "smoke",
                        "--run-role",
                        "local_cpu_smoke",
                        "--output-dir",
                        str(occupied),
                    ]
                ),
                2,
            )
            self.assertEqual(sorted(path.name for path in occupied.iterdir()), ["keep.txt"])

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is unavailable")
    def test_torch_model_shapes_and_replay_trace(self):
        from .torch_runner import import_torch, make_model, make_optimizer, train_window

        torch = import_torch()
        init = make_factor_initializations(make_seed_streams(11, self.config), self.config)[("compatible", 4)]
        model = make_model(torch, self.base.W0.astype(np.float32), init.A32)
        self.assertEqual(tuple(model.B.shape), (32, 4))
        self.assertEqual(tuple(model.A.shape), (4, 32))
        x = np.random.default_rng(5).standard_normal((1024, 32)).astype(np.float32)
        y = np.asarray(x @ self.base.W0.astype(np.float32).T, dtype=np.float32)
        pi = np.arange(1024, dtype=np.int64)
        optimizer = make_optimizer(torch, model)
        trace = []
        train_window(
            torch,
            model,
            optimizer,
            x_current=x,
            y_current=y,
            pi_t=pi,
            replay_state=None,
            replay_rng=None,
            device=torch.device("cpu"),
            updates=1,
            trace=trace,
        )
        self.assertEqual(trace[0].replay_count, 0)
        controller = ClockBalancedReplay()
        controller.insert_completed_window(1, x, y, np.random.default_rng(7))
        replay_trace = []
        train_window(
            torch,
            model,
            optimizer,
            x_current=x,
            y_current=y,
            pi_t=pi,
            replay_state=controller,
            replay_rng=np.random.default_rng(8),
            device=torch.device("cpu"),
            updates=1,
            trace=replay_trace,
        )
        self.assertEqual(replay_trace[0].replay_count, 64)
        self.assertEqual(replay_trace[0].replay_unique_count, 64)
        self.assertEqual(replay_trace[0].replay_duplicate_count, 0)

    def test_hash_helpers_and_lazy_torch_boundary(self):
        first = hash_arrays((("x", np.arange(8, dtype=np.float64)),))
        second = hash_arrays((("x", np.arange(8, dtype=np.float64).reshape(2, 4)),))
        self.assertNotEqual(first, second)
        self.assertEqual(hash_json({"b": 2, "a": 1}), hash_json({"a": 1, "b": 2}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.json"
            digest = write_json(path, {"b": 2, "a": 1})
            self.assertEqual(len(digest), 64)
            self.assertTrue(path.is_file())
            with self.assertRaises(ValueError):
                write_json(Path(directory) / "bad.json", {"value": float("nan")})
            self.assertFalse((Path(directory) / "bad.json").exists())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1, "b": 2})
        from . import torch_runner

        self.assertTrue(callable(torch_runner.configure_environment))


if __name__ == "__main__":
    unittest.main()
