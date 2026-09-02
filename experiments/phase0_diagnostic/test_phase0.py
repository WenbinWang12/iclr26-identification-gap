"""Unit and falsification checks for the exact Phase-0 diagnostic."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from phase0_diagnostic import (
    CompatibleConfig,
    MatrixConfig,
    compatible_offline,
    evaluate_gates,
    _gap_ratio,
    _input_fingerprint,
    _ratio_raw,
    _source_hash,
    bootstrap_mean_ci,
    bootstrap_paired_ci,
    make_directions,
    make_matrix_targets,
    make_truth,
    matrix_rows,
    project_current,
    sequential_compatible,
    summarize,
    run_experiment,
)


class Phase0Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CompatibleConfig(windows=18)

    def test_current_acquisition_is_exact_from_zero(self) -> None:
        target = make_truth(self.config, 0)
        directions = make_directions(self.config, 0, "correlated")
        states = sequential_compatible(target, directions, self.config.true_rank)
        for state, direction in zip(states, directions):
            np.testing.assert_allclose(state @ direction, target @ direction, atol=1e-10)

    def test_rank_is_preserved_only_under_the_documented_initialization(self) -> None:
        target = make_truth(self.config, 1)
        directions = make_directions(self.config, 1, "correlated")
        states = sequential_compatible(target, directions, self.config.true_rank)
        self.assertLessEqual(
            max(np.linalg.matrix_rank(state, tol=1e-9) for state in states),
            self.config.true_rank,
        )
        # An arbitrary rank-R initial state can leave the rank-R manifold after
        # the exact current projection; this guards against an invalid theorem.
        exact_target = np.array([[1.0, 0.0], [0.0, 0.0]])
        initial = np.array([[0.0, 0.0], [0.0, 1.0]])
        direction = np.array([1.0, 1.0]) / np.sqrt(2.0)
        unconstrained = project_current(initial, exact_target, direction)
        self.assertEqual(np.linalg.matrix_rank(unconstrained, tol=1e-9), 2)

    def test_orthogonal_control_has_no_interference(self) -> None:
        target = make_truth(self.config, 2)
        directions = make_directions(self.config, 2, "orthogonal")
        states = sequential_compatible(target, directions, self.config.true_rank)
        for t, state in enumerate(states, start=1):
            old = directions[:t]
            residual = (state - target) @ old.T
            self.assertLess(float(np.max(np.abs(residual))), 1e-9)

    def test_prefix_oracle_is_exact_and_dense_rank_is_preserved(self) -> None:
        target = make_truth(self.config, 3)
        for regime in ("orthogonal", "correlated"):
            directions = make_directions(self.config, 3, regime)
            states = compatible_offline(target, directions, self.config.true_rank)
            for t, state in enumerate(states, start=1):
                residual = (state - target) @ directions[:t].T
                self.assertLess(float(np.max(np.abs(residual))), 1e-9)
            dense = sequential_compatible(target, directions, None)
            self.assertLessEqual(
                max(np.linalg.matrix_rank(state, tol=1e-9) for state in dense),
                self.config.true_rank,
            )

    def test_future_directions_do_not_change_earlier_offline_prefixes(self) -> None:
        target = make_truth(self.config, 6)
        directions = make_directions(self.config, 6, "correlated")
        perturbed = directions.copy()
        perturbed[6:] *= -1.0
        original = compatible_offline(target, directions, self.config.true_rank)
        changed = compatible_offline(target, perturbed, self.config.true_rank)
        for left, right in zip(original[:6], changed[:6]):
            np.testing.assert_allclose(left, right, atol=1e-12)

    def test_past_retention_is_reported_after_the_first_prefix(self) -> None:
        from phase0_diagnostic import compatible_rows

        rows = compatible_rows(self.config, 7, "correlated")
        self.assertTrue(all(row["past_retention_ratio"] is not None for row in rows[1:]))

    def test_matrix_controls_have_expected_rank_and_conflict(self) -> None:
        cancel = make_matrix_targets(
            MatrixConfig(windows=6), 0, "canceling"
        )
        self.assertTrue(all(np.linalg.matrix_rank(x, tol=1e-9) <= 2 for x in cancel))
        rows = matrix_rows(
            MatrixConfig(windows=6), 0, "canceling"
        )
        even = [row for row in rows if row["prefix"] % 2 == 0]
        self.assertTrue(any((row["total_online_gap"] or 0.0) > 0.0 for row in even))
        self.assertTrue(
            all(abs(row["acquisition_matched_gap"] or 0.0) <= 1e-10 for row in even)
        )
        self.assertTrue(any((row["addressable_gap"] or 0.0) > 0.0 for row in even))

    def test_rank_inadequate_control_is_actually_capacity_bound(self) -> None:
        config = MatrixConfig(windows=6)
        rows = matrix_rows(config, 0, "inadequate")
        self.assertTrue(all(row["rank_seq"] <= config.rank for row in rows))
        self.assertTrue(any((row["rank_specific_gap"] or 0.0) > 0.20 for row in rows))
        self.assertTrue(
            any((row["acquisition_matched_gap"] or 0.0) > 0.20 for row in rows)
        )
        ratios = [row["capacity_ratio"] for row in rows if row["capacity_ratio"] is not None]
        self.assertGreater(float(np.median(ratios)), 0.20)

    def test_compatible_replay_is_not_an_independent_addressability_test(self) -> None:
        target = make_truth(self.config, 4)
        directions = make_directions(self.config, 4, "correlated")
        rows = []
        from phase0_diagnostic import compatible_rows

        rows.extend(compatible_rows(self.config, 4, "correlated"))
        self.assertTrue(all(row["addressability_applicable"] is False for row in rows))
        for row in rows:
            if row["eligible"]:
                self.assertAlmostEqual(
                    row["addressable_gap"], row["total_online_gap"], places=9
                )

    def test_eligibility_guard_preserves_raw_losses(self) -> None:
        config = CompatibleConfig(d_out=2, d_in=2, true_rank=1, latent_dim=2, windows=2)
        rows = []
        from phase0_diagnostic import compatible_rows

        rows.extend(compatible_rows(config, 0, "orthogonal"))
        self.assertTrue(all(row["base_loss"] >= 0.0 for row in rows))
        self.assertTrue(all("total_online_gap_raw" in row for row in rows))

    def test_frozen_gate_evaluator_reports_capacity_and_conflict_controls(self) -> None:
        from phase0_diagnostic import compatible_rows

        rows = []
        for seed in range(5):
            rows.extend(compatible_rows(self.config, seed, "orthogonal"))
            rows.extend(compatible_rows(self.config, seed, "correlated"))
            rows.extend(matrix_rows(MatrixConfig(windows=6), seed, "canceling"))
            rows.extend(matrix_rows(MatrixConfig(windows=6), seed, "inadequate"))
        summary = summarize(rows)
        gates = evaluate_gates(rows, summary)
        self.assertTrue(gates["canceling_conflict_control"]["pass"])
        self.assertTrue(gates["inadequate_capacity_positive_control"]["pass"])
        self.assertIsNone(gates["addressability"]["pass"])

    def test_bootstrap_is_deterministic_and_seed_level(self) -> None:
        values = [0.1, 0.4, 0.8]
        self.assertEqual(bootstrap_mean_ci(values), bootstrap_mean_ci(values))
        left = [0.2, 0.5, 0.9]
        right = [0.1, 0.2, 0.4]
        shifted = [value + 10.0 for value in right]
        np.testing.assert_allclose(
            bootstrap_paired_ci(left, right),
            bootstrap_paired_ci([x + 10.0 for x in left], shifted),
            atol=1e-12,
        )

    def test_gap_clamping_keeps_raw_value(self) -> None:
        raw = _ratio_raw(-1e-12, 1.0)
        clamped = _gap_ratio(-1e-12, 1.0)
        self.assertNotEqual(raw, 0.0)
        self.assertEqual(clamped, 0.0)

    def test_input_and_source_fingerprints_are_stable(self) -> None:
        self.assertEqual(_source_hash(), _source_hash())
        config = CompatibleConfig(windows=4)
        matrix = MatrixConfig(windows=3)
        self.assertEqual(
            _input_fingerprint(config, matrix, [0, 1]),
            _input_fingerprint(config, matrix, [0, 1]),
        )

    def test_inadequate_targets_have_rank_r_plus_one(self) -> None:
        targets = make_matrix_targets(MatrixConfig(windows=3), 0, "inadequate")
        self.assertTrue(
            all(np.linalg.matrix_rank(target, tol=1e-9) == 3 for target in targets)
        )

    def test_small_runs_are_byte_reproducible(self) -> None:
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            run_experiment(Path(first), [0, 1], "test_reproducibility")
            run_experiment(Path(second), [0, 1], "test_reproducibility")
            self.assertEqual(
                (Path(first) / "per_prefix.csv").read_bytes(),
                (Path(second) / "per_prefix.csv").read_bytes(),
            )
            self.assertEqual(
                (Path(first) / "summary.json").read_bytes(),
                (Path(second) / "summary.json").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
