"""Tests for five-arm result aggregation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panel_report import (  # noqa: E402
    LOCKED_ARMS,
    aggregate_completed_run,
    lower_tail_mean,
    summarize_arm,
)


def test_fractional_lower_tail():
    # alpha*N=1.5: all of 1 and half of 3.
    assert np.isclose(lower_tail_mean([1.0, 3.0, 7.0, 9.0], 0.375), 2.5 / 1.5)


def test_arm_summary_keeps_tail_and_mean_distinct():
    out = summarize_arm(
        arm="fixed_er",
        task_names=["a", "b", "c"],
        base_scores=[0.2, 0.2, 0.2],
        immediate_scores=[0.8, 0.7, 0.6],
        final_scores=[0.5, 0.7, 0.4],
    )
    assert np.isclose(out.mean_final, (0.5 + 0.7 + 0.4) / 3)
    assert np.isclose(out.mean_bwt, (-0.3 + 0.0 - 0.2) / 3)
    assert np.isclose(out.min_retention, 0.5)


def test_incomplete_panel_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for arm in LOCKED_ARMS[:-1]:
            arm_dir = root / arm
            arm_dir.mkdir()
            payload = {
                "status": "completed",
                "task_names": ["x"],
                "base_scores": [0.0],
                "immediate_scores": [1.0],
                "final_scores": [1.0],
            }
            (arm_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
            (arm_dir / "COMPLETED").touch()
        try:
            aggregate_completed_run(root)
        except RuntimeError as exc:
            assert LOCKED_ARMS[-1] in str(exc)
        else:
            raise AssertionError("an incomplete panel was aggregated")


if __name__ == "__main__":
    test_fractional_lower_tail()
    test_arm_summary_keeps_tail_and_mean_distinct()
    test_incomplete_panel_fails_closed()
    print("test_panel_report OK")
