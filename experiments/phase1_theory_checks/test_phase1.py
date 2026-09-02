import json
import shutil
import tempfile
import unittest
from pathlib import Path

from phase1_theory_checks import (
    check_block_allocation,
    check_decomposition_and_spectral,
    check_safe_space,
    check_sequential,
    check_taylor,
    run_all_checks,
    write_run,
)


def _assert_case_passes(testcase: unittest.TestCase, result) -> None:
    summary, _ = result
    failed = [gate["name"] for gate in summary["gates"] if not gate["passed"]]
    testcase.assertEqual(failed, [], msg=f"failed gates: {failed}")


class Phase1TheoryTests(unittest.TestCase):
    def test_taylor_and_stationarity(self):
        _assert_case_passes(self, check_taylor())

    def test_sequential_and_innovations(self):
        _assert_case_passes(self, check_sequential())

    def test_decomposition_and_spectral_capacity(self):
        _assert_case_passes(self, check_decomposition_and_spectral())

    def test_safe_space_and_price(self):
        _assert_case_passes(self, check_safe_space())

    def test_block_allocation_and_counterexample(self):
        _assert_case_passes(self, check_block_allocation())

    def test_all_required_gates(self):
        result = run_all_checks()
        self.assertTrue(result["all_required_pass"])
        self.assertEqual(result["failed_gate_count"], 0)
        self.assertEqual(
            [case["case"] for case in result["cases"]],
            [
                "taylor_stationarity",
                "sequential_innovations",
                "decomposition_spectral_capacity",
                "safe_space_and_price",
                "block_allocation",
            ],
        )
        for gate in result["gates"]:
            self.assertIn(gate["relation"], {"eq", "le", "ge"})
            self.assertEqual(gate["violation"], 0.0, msg=gate["name"])
            if gate["relation"] != "eq":
                self.assertIsNone(gate["abs_error"], msg=gate["name"])

    def test_artifact_schema_and_hash_inputs(self):
        temporary_root = Path(tempfile.mkdtemp(prefix="phase1-theory-"))
        try:
            output = write_run(temporary_root / "run", "unit_test")
            run_dir = Path(output["output_dir"])
            for name in ("per_case.csv", "traces.json", "summary.json", "manifest.json"):
                self.assertTrue((run_dir / name).is_file())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["all_required_pass"])
            self.assertFalse(manifest["resource_ledger"]["gpu_used"])
            self.assertEqual(manifest["resource_ledger"]["optimizer_state_bytes"], 0)
            self.assertEqual(len(manifest["source_sha256"]), 64)
            self.assertEqual(len(manifest["protocol_sha256"]), 64)
            self.assertEqual(len(manifest["test_sha256"]), 64)
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
