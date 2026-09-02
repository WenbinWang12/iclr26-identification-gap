"""Tests for the Phase-2T scorer.

The point of these is that ``decide`` is a *mechanical* function of the criteria:
every one of §4's five outcomes must be reachable, and no combination may fall
through silently.  Phase-2S needed an amendment because two frozen criteria were
scoreable-but-wrong; here the criteria are exercised against constructed rows
before any real data is loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase2t_vgc.decide_2t import (  # noqa: E402
    PROTOCOL_SHA, decide, parse_name, t1, t2, t3, t4, t5, t6,
    task_identity_overlap,
)
from phase2t_vgc.vgc import TAU_SD  # noqa: E402


def row(task, *, raw=0.50, bc=None, vgc=None, sd=1.0, applied=None, seed="2"):
    bc = raw + 0.06 if bc is None else bc
    applied = (sd >= TAU_SD) if applied is None else applied
    vgc = (bc if applied else raw) if vgc is None else vgc
    return {"task": task, "seed": seed, "stage": "1", "file": "f.npz", "n": 64,
            "sd": sd, "applied": applied, "raw": raw, "bc": bc, "vgc": vgc,
            "oracle": max(raw, bc, vgc) + 0.02}


def good_rows(n_tasks=5, per=8):
    """Rows where the mechanism holds: within each task, higher sd -> bigger gain."""
    rows = []
    for t in range(n_tasks):
        for i in range(per):
            sd = 0.20 + 0.16 * i                       # 0.20 .. 1.32
            gain = -0.02 + 0.05 * i                    # negative at low sd
            rows.append(row(f"T{t}", raw=0.50, bc=0.50 + gain, sd=sd))
    return rows


def test_protocol_sha_is_the_frozen_one():
    assert PROTOCOL_SHA == \
        "7697291743d6a64be7dcef29a70f3d2adcf8d102e6bc004628510a08bb87a849"


def test_parse_name_keeps_task_names_containing_underscores():
    assert parse_name("seed2_stage7_SST-2.npz") == ("2", "7", "SST-2")
    assert parse_name("seed3_stage11_my_task.npz") == ("3", "11", "my_task")
    with pytest.raises(ValueError):
        parse_name("logits.npz")


def test_t1_compares_vgc_against_bc_on_the_same_batches():
    rows = [row("A", raw=0.50, bc=0.30, sd=0.20),      # BC harmful, gate abstains
            row("A", raw=0.50, bc=0.60, sd=1.00)]
    r = t1(rows)
    assert r["worst_bc"] == pytest.approx(-20.0)
    assert r["worst_vgc"] == pytest.approx(0.0)
    assert r["harmful_bc"] == 1 and r["harmful_vgc"] == 0
    assert r["pass"]


def test_t1_fails_when_the_tail_survives_the_gate():
    rows = [row("A", raw=0.50, bc=0.40, sd=1.20)]      # harmful AND high sd
    assert not t1(rows)["pass"]


def test_t2_needs_both_the_point_estimate_and_the_ci_lower_bound():
    """A median above 1 pp with a CI touching 0 must not pass."""
    rows = [row("A", bc=0.56, sd=1.0), row("B", bc=0.56, sd=1.0),
            row("C", bc=0.44, sd=1.0), row("D", bc=0.58, sd=1.0),
            row("E", bc=0.57, sd=1.0)]
    r = t2(rows)
    assert r["point"] > 0
    assert r["pass"] == (r["point"] >= 1.0 and r["lo"] > 0.0)


def test_t2_fails_when_the_gate_abstains_everywhere():
    rows = [row(f"T{i}", sd=0.10) for i in range(5)]
    r = t2(rows)
    assert r["point"] == pytest.approx(0.0)
    assert not r["pass"]


def test_t3_is_negative_because_gating_costs_mean_gain():
    """The sign is predeclared; a positive T3 would mean VGC beats BC outright."""
    rows = [row(f"T{i}", raw=0.50, bc=0.60, sd=0.20) for i in range(5)]
    r = t3(rows)
    assert r["point"] < 0
    assert not r["pass"]                      # -10 pp is below the -2 pp floor


def test_t3_passes_when_the_gate_only_drops_batches_bc_was_losing():
    rows = [row(f"T{i}", raw=0.50, bc=0.49, sd=0.20) for i in range(5)]
    assert t3(rows)["pass"]


def test_t4_rejects_a_vacuous_gate_in_both_directions():
    assert not t4([row("A", sd=0.1) for _ in range(10)])["pass"]        # 0.0
    assert not t4([row("A", sd=2.0) for _ in range(10)])["pass"]        # 1.0
    mixed = [row("A", sd=2.0)] * 5 + [row("A", sd=0.1)] * 5
    assert t4(mixed)["pass"] and t4(mixed)["apply_rate"] == 0.5


def test_t5_passes_only_when_every_qualifying_task_is_positive():
    r = t5(good_rows())
    assert r["n_tasks"] == 5 and r["n_positive"] == 5 and r["pass"]


def test_t5_fails_if_one_task_reverses_even_with_a_positive_median():
    rows = good_rows(n_tasks=5)
    flipped = [dict(x, bc=x["raw"] + (0.30 - (x["sd"] - 0.20)))
               for x in rows if x["task"] == "T0"]
    rows = [x for x in rows if x["task"] != "T0"] + flipped
    r = t5(rows)
    assert r["median"] > 0
    assert r["per_task"]["T0"] < 0
    assert not r["pass"]


def test_t5_ignores_tasks_with_too_few_batches_instead_of_using_noisy_rhos():
    rows = good_rows(n_tasks=2) + [row("TINY", sd=0.3), row("TINY", sd=1.4)]
    assert "TINY" not in t5(rows)["per_task"]


def test_t5_is_not_a_pass_when_no_task_qualifies():
    r = t5([row("A", sd=0.3), row("B", sd=1.2)])
    assert not r["pass"] and r["median"] is None


def test_t6_reports_the_fraction_of_bcs_tail_the_gate_can_see():
    rows = [row("QQP", raw=0.50, bc=0.35, sd=0.33),    # harmful, low sd  -> caught
            row("RTE", raw=0.50, bc=0.35, sd=1.10),    # harmful, high sd -> missed
            row("RTE", raw=0.50, bc=0.60, sd=1.10)]
    r = t6(rows)
    assert r["n_harmful"] == 2 and r["n_caught"] == 1
    assert r["caught_fraction"] == pytest.approx(0.5)
    assert r["harmful_tasks"] == ["QQP", "RTE"]


def test_t6_does_not_divide_by_zero_when_bc_never_hurts():
    assert t6([row("A", sd=1.0)])["caught_fraction"] is None


def test_task_identity_flags_an_all_or_nothing_gate():
    rows = [row("HI", sd=1.2) for _ in range(4)] + [row("LO", sd=0.2) for _ in range(4)]
    ti = task_identity_overlap(rows)
    assert ti["apply_rate_by_task"] == {"HI": 1.0, "LO": 0.0}
    assert ti["is_task_filter"]


def test_task_identity_does_not_flag_within_task_variation():
    rows = [row("MIX", sd=1.2), row("MIX", sd=0.2), row("OTHER", sd=1.2)]
    assert not task_identity_overlap(rows)["is_task_filter"]


def _res(**kw):
    base = {"T1": {"pass": True}, "T2": {"pass": True}, "T3": {"pass": True},
            "T4": {"pass": True}, "T5": {"pass": True},
            "task_identity": {"is_task_filter": False}, "dev_T1_pass": True}
    base.update(kw)
    return base


def test_outcome_1_when_everything_passes():
    assert decide(_res()) == "OUTCOME_1_working_fix_to_a_documented_defect"


def test_outcome_2_when_the_tail_is_fixed_but_the_mechanism_is_not_shown():
    assert decide(_res(T5={"pass": False})) == \
        "OUTCOME_2_works_but_the_selector_is_unexplained"


def test_outcome_4_takes_priority_over_2_when_the_gate_is_a_task_filter():
    assert decide(_res(T5={"pass": False},
                       task_identity={"is_task_filter": True})) == \
        "OUTCOME_4_task_filter_not_a_calibration_gate"


def test_outcome_3_when_seed_1_passed_t1_and_the_heldout_seeds_do_not():
    assert decide(_res(T1={"pass": False})) == \
        "OUTCOME_3_threshold_was_fitted_to_seed_1"


def test_outcome_5_takes_priority_because_no_gain_makes_the_tail_moot():
    assert decide(_res(T2={"pass": False}, T1={"pass": False})) == \
        "OUTCOME_5_gate_removes_the_gain_with_the_tail"


def test_a_combination_outside_the_five_outcomes_is_reported_as_a_protocol_gap():
    """T1 fails, seed 1 also failed T1: not outcome 3, not any other branch."""
    assert decide(_res(T1={"pass": False}, dev_T1_pass=False)) == \
        "PROTOCOL_GAP_record_in_amendment"


def test_t3_failure_alone_does_not_silently_become_outcome_1():
    out = decide(_res(T3={"pass": False}))
    assert out != "OUTCOME_1_working_fix_to_a_documented_defect"
    assert out == "PROTOCOL_GAP_record_in_amendment"
