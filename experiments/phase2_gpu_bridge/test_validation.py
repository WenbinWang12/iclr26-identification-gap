"""Focused tests for the read-only Phase-2 reconciliation validator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import artifacts
from .core_numpy import BridgeConfig, PRIMARY_SEEDS
from .evidence import (
    EvidenceError,
    _assert_marker_disjoint,
    _assert_validation_report_bindings,
    _snapshot_local_validation_report,
)
from .phase2_gpu_bridge import _config_payload, main
from .validation import (
    CSVTable,
    IssueCollector,
    RunBundle,
    StrictJSONError,
    _index_transitions,
    _load_json_strict,
    _validate_csv_row_types,
    _validate_gate_leaf_types,
    _validate_gate_schema,
    _validate_inline_json,
    _validate_resource_identity,
    _validate_run_schema,
    reconcile_runs,
)


class ValidationTests(unittest.TestCase):
    def test_strict_json_rejects_nonfinite_overflow_and_duplicate_keys(self):
        cases = (
            '{"value":NaN}',
            '{"value":Infinity}',
            '{"value":1e999}',
            '{"value":1,"value":2}',
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, payload in enumerate(cases):
                path = root / f"bad_{index}.json"
                path.write_text(payload, encoding="utf-8")
                with self.subTest(payload=payload), self.assertRaises(StrictJSONError):
                    _load_json_strict(path)

    def test_strict_json_rejects_integer_limit_and_recursion_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            huge_int = root / "huge_int.json"
            huge_int.write_text('{"value":' + ("9" * 5000) + "}", encoding="utf-8")
            nested = root / "nested.json"
            nested.write_text("[" * 2000 + "0" + "]" * 2000, encoding="utf-8")
            for path in (huge_int, nested):
                with self.subTest(path=path.name), self.assertRaises(StrictJSONError):
                    _load_json_strict(path)

    def test_inline_json_is_strict_and_returns_the_parsed_value(self):
        self.assertEqual(_validate_inline_json('{"value":1}', "inline"), {"value": 1})
        for payload in ('{"value":NaN}', '{"value":1,"value":2}'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                _validate_inline_json(payload, "inline")

    def test_malformed_transition_keys_become_issues(self):
        base = {
            "seed": PRIMARY_SEEDS[0],
            "regime": "compatible",
            "R": 2,
            "arm": "clock_balanced_replay_rank_R",
            "completed_window": 1,
            "history_windows": 1,
            "capacity": 352,
            "eligible_count": 1024,
            "occupied_count": 352,
            "unique_count": 352,
            "duplicate_count": 0,
            "replacement_count": 0,
            "priority_draws": 1024,
            "selection_comparisons": 1,
            "priority_tie_count": 0,
            "controller_cpu_seconds": 0.0,
            "selected_priority_order_sha256": "a" * 64,
        }
        for bad_seed in ([], float(PRIMARY_SEEDS[0])):
            collector = IssueCollector()
            row = {**base, "seed": bad_seed}
            indexed = _index_transitions([row], "primary", collector)
            with self.subTest(seed=bad_seed):
                self.assertEqual(indexed, {})
                self.assertIn("invalid_transition_integer", {issue.code for issue in collector.issues})

    def test_malformed_runtime_device_indices_become_issues(self):
        deterministic_flags = {
            "deterministic_algorithms": None,
            "cudnn_deterministic": None,
            "cudnn_benchmark": None,
            "cuda_matmul_allow_tf32": None,
            "cudnn_allow_tf32": None,
            "float32_matmul_precision": None,
            "environment_controls": None,
        }
        key = (PRIMARY_SEEDS[0], "compatible", 2, "sequential_dense_rank_R")
        for row_index, runtime_index in (("0", "bad"), ("bad", 0)):
            runtime_devices = [{"index": runtime_index, "name": "A100", "driver_version": "test"}]
            manifest = {
                "source_hashes": {},
                "data_sha256_by_seed": {},
                "split_sha256_by_seed": {},
                "runtime": {"nvidia_smi_devices": runtime_devices},
            }
            row = {
                "source_hashes": "{}",
                "nvidia_smi_devices": json.dumps(runtime_devices),
                "deterministic_flags": json.dumps(deterministic_flags),
                "device": "cuda:0",
                "device_index": row_index,
            }
            bundle = RunBundle(
                Path("."),
                "primary",
                "primary",
                manifest,
                {"config.json": {"device": "cuda:0"}, "traces.json": {"factor_initializations": {}}},
            )
            collector = IssueCollector()
            _validate_resource_identity(bundle, CSVTable("resource_ledger.csv", tuple(row), {key: row}), collector)
            with self.subTest(row_index=row_index, runtime_index=runtime_index):
                self.assertGreater(collector.count(), 0)
                self.assertTrue(
                    {"resource_device_index_invalid", "runtime_device_index_invalid"}
                    & {issue.code for issue in collector.issues}
                )

    def test_csv_type_validation_rejects_nonfinite_runtime_values(self):
        collector = IssueCollector()
        _validate_csv_row_types("resource_ledger.csv", {"wall_seconds": "NaN"}, "row", collector)
        self.assertIn("invalid_csv_field_type", {issue.code for issue in collector.issues})

    def test_gate_count_status_and_boolean_types_are_not_inferred(self):
        collector = IssueCollector()
        _validate_gate_leaf_types(
            {
                "G4_resource_and_reproducibility": {
                    "transition_rows": "2640",
                    "status": "anything",
                    "single_run_passed": "true",
                }
            },
            "primary",
            collector,
            (),
        )
        self.assertTrue(
            {"invalid_gate_count", "invalid_gate_status", "invalid_gate_boolean"}
            <= {issue.code for issue in collector.issues}
        )

    def test_gate_numeric_metrics_must_be_finite_numbers(self):
        collector = IssueCollector()
        _validate_gate_leaf_types(
            {"G1_oracle_capacity": {"contraction_bound": "bogus", "q_min_keep": "NaN"}},
            "primary",
            collector,
            (),
        )
        self.assertEqual(collector.count(check="run_schema"), 2)
        self.assertTrue(all(issue.code == "invalid_gate_numeric" for issue in collector.issues))
        huge = IssueCollector()
        _validate_gate_leaf_types(
            {"G1_oracle_capacity": {"contraction_bound": int("9" * 4000)}},
            "primary",
            huge,
            (),
        )
        self.assertEqual(huge.count(check="run_schema"), 1)
        self.assertEqual(huge.issues[0].code, "invalid_gate_numeric")

    def test_gate_pass_flags_must_match_resource_subgates(self):
        gates = {
            "G0_numpy_identity": {},
            "G1_oracle_capacity": {},
            "G2_phenomenon": {"status": "not_evaluated"},
            "G3_bounded_history": {"status": "not_evaluated"},
            "G4_resource_and_reproducibility": {
                "passed": False,
                "single_run_passed": True,
                "transition_passed": False,
                "occupancy_passed": True,
                "compute_passed": True,
                "optimizer_contract_passed": True,
                "expected_optimizer_contract_sha256": "a" * 64,
                "replay_resource_passed": True,
                "matched_arm_passed": True,
                "accounting_complete_passed": True,
                "transition_rows": 2640,
                "expected_transition_rows": 2640,
                "confirmation_required_for_citation": True,
                "confirmation_status": "pending",
                "post_result_four_provider_audit_status": "pending",
                "status": "pending_reconciliation",
                "peak_gpu_measurement_scope": "arm_cell_isolated_full_12_window_training_from_pre_model_zero_allocation_baseline; inference_independently_reset_from_post_training_model_resident_baseline",
            },
        }
        bundle = RunBundle(Path("primary"), "primary", "primary", {}, {"gates.json": gates})
        collector = IssueCollector()
        _validate_gate_schema(bundle, collector)
        self.assertIn("inconsistent_g4_single_run_passed", {issue.code for issue in collector.issues})

    def test_gate_pass_flags_must_match_g2_and_g3_statistics(self):
        gates = {
            "G0_numpy_identity": {},
            "G1_oracle_capacity": {},
            "G2_phenomenon": {
                "status": "evaluated",
                "acquisition": {
                    arm: {"count": 0, "total": 240, "threshold": 228, "status": "primary_only"}
                    for arm in ("sequential_dense_rank_R", "clock_balanced_replay_rank_R", "reservoir_random_replay_rank_R")
                },
                "acquisition_ok": {
                    arm: False
                    for arm in ("sequential_dense_rank_R", "clock_balanced_replay_rank_R", "reservoir_random_replay_rank_R")
                },
                "paired_final_rows_complete": True,
                "sequential_online_ratio_lower_bound": 0.2,
                "sequential_online_ratio_threshold": 0.1,
                "passed": True,
            },
            "G3_bounded_history": {
                "status": "evaluated",
                "paired_final_rows_complete": True,
                "valid_positive_gap_count": 0,
                "indeterminate_count": 0,
                "nonpositive_count": 16,
                "minimum_valid_count": 16,
                "gap_fraction_median": 0.6,
                "risk_consistency_failures": 1,
                "bootstrap": {"statistics": {"bounded_history_reduction_T": {"lower_bound_05": 0.1}}},
                "passed": True,
            },
            "G4_resource_and_reproducibility": {},
        }
        bundle = RunBundle(Path("primary"), "primary", "primary", {}, {"gates.json": gates})
        collector = IssueCollector()
        _validate_gate_schema(bundle, collector)
        codes = {issue.code for issue in collector.issues}
        self.assertIn("inconsistent_g2_passed", codes)
        self.assertIn("inconsistent_g3_passed", codes)

    def test_run_schema_binds_frozen_config_to_bridge_config(self):
        frozen = BridgeConfig().payload()
        frozen["lr"] = 0.123
        config = {
            "kind": "phase2_frozen_experiment_config",
            "schema_version": 1,
            "seeds": list(PRIMARY_SEEDS),
            "protocol_sha256": BridgeConfig().protocol_sha256,
            "frozen": frozen,
        }
        bundle = RunBundle(
            Path("primary"),
            "primary",
            "primary",
            {"frozen_config_sha256": artifacts.hash_json(frozen)},
            {"config.json": config},
        )
        collector = IssueCollector()
        _validate_run_schema(bundle, collector)
        codes = {issue.code for issue in collector.issues}
        self.assertIn("noncanonical_frozen_config", codes)
        self.assertIn("noncanonical_frozen_config_hash", codes)

    def test_shared_config_payload_has_no_run_role(self):
        payload = _config_payload(BridgeConfig(), PRIMARY_SEEDS, "cuda:0")
        self.assertNotIn("run_role", payload)
        self.assertEqual(payload, _config_payload(BridgeConfig(), PRIMARY_SEEDS, "cuda:0"))

    def test_freeze_cli_reports_missing_evidence_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "freeze.json"
            exit_code = main(["--mode", "freeze", "--freeze-marker", str(marker)])
            self.assertEqual(exit_code, 2)
            self.assertFalse(marker.exists())

    def test_freeze_marker_cannot_overwrite_or_nest_bound_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_dir = root / "evidence"
            evidence_dir.mkdir()
            existing = evidence_dir / "manifest.json"
            existing.write_text("{}", encoding="utf-8")
            with self.assertRaises(EvidenceError):
                _assert_marker_disjoint(existing, [evidence_dir])
            nested = evidence_dir / "nested" / "marker.json"
            with self.assertRaises(EvidenceError):
                _assert_marker_disjoint(nested, [evidence_dir])

    def test_local_validation_report_binds_runtime_and_exact_rebuild_schema(self):
        digest = "a" * 64
        source_hashes = {"source.py": digest}
        runtime = {"python": "3.12", "numpy": "2.4.6"}
        manifest = {
            "run_role": "local_cpu_smoke",
            "base_sha256": digest,
            "data_sha256": digest,
            "split_sha256": digest,
            "runtime": runtime,
        }
        report = {
            "schema_version": 1,
            "kind": "phase2_local_cpu_smoke_validation",
            "validator_version": "phase2-evidence-v2",
            "validation_mode": "same_host_exact_rebuild",
            "hash_comparison_policy": "local_generated_arrays=provenance_only_cross_host",
            "claim_boundary": "provenance_only_local_cpu_smoke",
            "protocol_sha256": digest,
            "config_sha256": digest,
            "source_hashes": source_hashes,
            "run": {},
            "run_role": "local_cpu_smoke",
            "expected_seeds": [11],
            "exact_rebuild": {
                "base_sha256": digest,
                "data_sha256": digest,
                "split_sha256": digest,
            },
            "runtime": runtime,
            "checks": {
                "artifact_binding": True,
                "config_and_protocol": True,
                "split_semantics": True,
                "g0_numpy_identity": True,
                "g1_oracle_capacity": True,
                "source_snapshot": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "report.json"

            def validate(payload, manifest_value=manifest):
                artifacts.write_json(path, payload)
                with mock.patch(
                    "experiments.phase2_gpu_bridge.evidence._check_run_binding",
                    return_value=({}, manifest_value, root / "run"),
                ):
                    return _snapshot_local_validation_report(
                        root,
                        path,
                        protocol_hash=digest,
                        config_hash=digest,
                        source_hashes=source_hashes,
                    )

            self.assertEqual(validate(report)["path"], "report.json")
            with self.assertRaisesRegex(EvidenceError, "runtime does not match manifest"):
                validate({**report, "runtime": {"forged": "unrelated"}})
            with self.assertRaisesRegex(EvidenceError, "runtime is empty"):
                empty_runtime_manifest = {**manifest, "runtime": {}}
                validate({**report, "runtime": {}}, empty_runtime_manifest)
            with self.assertRaisesRegex(EvidenceError, "exact rebuild schema mismatch"):
                validate(
                    {
                        **report,
                        "exact_rebuild": {**report["exact_rebuild"], "unbound": digest},
                    }
                )

    def test_validation_reports_must_bind_marker_sibling_evidence(self):
        local_run = {"manifest": "runs/local/manifest.json"}
        remote_run = {"manifest": "runs/remote/manifest.json"}
        calibration_run = {"manifest": "runs/calibration/manifest.json"}
        remote_test = {"path": "tests/result.json", "sha256": "a" * 64}
        local_report = {"run": local_run, "path": "reports/local.json", "sha256": "c" * 64}
        remote_report = {
            "runs": {
                "remote_gpu_smoke": remote_run,
                "non_citable_calibration": calibration_run,
            },
            "remote_test": remote_test,
            "local_cpu_smoke_validation": {
                "path": local_report["path"],
                "sha256": local_report["sha256"],
            },
        }
        _assert_validation_report_bindings(
            local_report,
            local_run,
            remote_report,
            remote_run,
            calibration_run,
            remote_test,
        )
        with self.assertRaisesRegex(EvidenceError, "local validation report"):
            _assert_validation_report_bindings(
                {"run": {"manifest": "runs/other/manifest.json"}},
                local_run,
                remote_report,
                remote_run,
                calibration_run,
                remote_test,
            )
        with self.assertRaisesRegex(EvidenceError, "remote smoke run"):
            _assert_validation_report_bindings(
                local_report,
                local_run,
                {**remote_report, "runs": {**remote_report["runs"], "remote_gpu_smoke": local_run}},
                remote_run,
                calibration_run,
                remote_test,
            )
        with self.assertRaisesRegex(EvidenceError, "remote test result"):
            _assert_validation_report_bindings(
                local_report,
                local_run,
                {**remote_report, "remote_test": {"path": "tests/forged.json", "sha256": "b" * 64}},
                remote_run,
                calibration_run,
                remote_test,
            )
        with self.assertRaisesRegex(EvidenceError, "local validation report"):
            _assert_validation_report_bindings(
                local_report,
                local_run,
                {**remote_report, "local_cpu_smoke_validation": {"path": "reports/other.json", "sha256": "d" * 64}},
                remote_run,
                calibration_run,
                remote_test,
            )

    def test_invalid_inputs_produce_deterministic_revise_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            confirmation = root / "confirmation"
            output = root / "reconciliation"
            primary.mkdir()
            confirmation.mkdir()
            primary_manifest = b'{"run_role":"primary"}\n'
            confirmation_manifest = b'{"run_role":"confirmation"}\n'
            (primary / "manifest.json").write_bytes(primary_manifest)
            (confirmation / "manifest.json").write_bytes(confirmation_manifest)

            result = reconcile_runs(primary, confirmation, output)

            self.assertEqual(result["decision"], "REVISE")
            self.assertFalse(result["positive_claim_authorized"])
            self.assertGreater(result["issue_count"], 0)
            self.assertEqual((primary / "manifest.json").read_bytes(), primary_manifest)
            self.assertEqual((confirmation / "manifest.json").read_bytes(), confirmation_manifest)
            written = json.loads((output / "reconciliation.json").read_text(encoding="utf-8"))
            self.assertEqual(written, result)

    def test_output_cannot_be_nested_below_an_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            confirmation = root / "confirmation"
            primary.mkdir()
            confirmation.mkdir()
            with self.assertRaises(ValueError):
                reconcile_runs(primary, confirmation, primary / "output")


if __name__ == "__main__":
    unittest.main()
