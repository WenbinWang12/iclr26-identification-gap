"""Tests for the Phase-2U scorer.

``decide`` must be a mechanical function of U1-U6, every §4 outcome must be
reachable, and no combination may fall through unnoticed.  Phase-2T needed an
amendment because T3 was too loose and T5 too strict once the median was 0; those
two lessons are pinned here as explicit tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase2u_scg.decide_2u import (  # noqa: E402
    PROTOCOL_SHA, decide, parse_name, u1, u2, u3, u4, u5, u6, u7,
)
from phase2u_scg.scg import TAU_M  # noqa: E402


def row(task, *, seed="11", raw=0.50, bc=None, sd=1.0, abs_m=2.0, applied=None):
    bc = raw + 0.05 if bc is None else bc
    applied = (abs_m >= TAU_M) if applied is None else applied
    return {"task": task, "seed": seed, "stage": "9", "n": 128,
            "abs_m": abs_m, "sd": sd, "applied": applied, "raw": raw, "bc": bc,
            "scg": bc if applied else raw, "oracle": max(raw, bc) + 0.02}


def test_protocol_sha_is_the_frozen_one():
    assert PROTOCOL_SHA == \
        "892ef1a8bb68bf1d044da031ae092fca73122c8c7ba77d2c8bd79486669e4426"


def test_parse_name_keeps_task_names_with_underscores_and_two_digit_seeds():
    assert parse_name("seed11_stage15_SST-2.npz") == ("11", "15", "SST-2")
    assert parse_name("seed13_stage2_my_task.npz") == ("13", "2", "my_task")
    with pytest.raises(ValueError):
        parse_name("seedX_stage1_A.npz")


def test_u1_fails_on_a_single_bad_seed_even_when_pooled_looks_fine():
    """This is the exact way 2T's threshold broke; U1 must catch it."""
    rows = [row("A", seed=s) for s in ("11", "12", "13", "14")] * 5
    rows.append(row("A", seed="15", raw=0.50, bc=0.40, abs_m=2.0))  # -10 pp
    r = u1(rows)
    assert r["worst_by_seed"]["15"] == pytest.approx(-10.0)
    assert not r["pass"]


def test_u1_passes_when_every_seed_is_within_the_floor():
    rows = [row("A", seed=s, bc=0.56) for s in ("11", "12", "13", "14", "15")]
    r = u1(rows)
    assert r["pass"] and r["worst_scg"] > 0


def test_u1_credits_abstention_only_because_abstaining_scores_raw():
    rows = [row("A", raw=0.50, bc=0.30, abs_m=0.1)]      # gate abstains
    r = u1(rows)
    assert r["worst_bc"] == pytest.approx(-20.0)
    assert r["worst_scg"] == pytest.approx(0.0) and r["pass"]


def test_u2_is_scored_on_the_mean_not_the_median():
    """2T's T2 used the median and was mathematically unpassable at a 0 median."""
    rows = [row(f"T{i}", raw=0.50, bc=0.70, abs_m=2.0) for i in range(3)]
    rows += [row(f"Z{i}", raw=0.50, bc=0.50, abs_m=0.1) for i in range(4)]
    r = u2(rows)
    assert r["batch_mean"] > 0
    med = np.median([100 * (x["scg"] - x["raw"]) for x in rows])
    assert med == 0.0                     # median is 0 ...
    assert r["pass"]                      # ... and U2 still passes


def test_u2_fails_when_the_gate_never_fires():
    rows = [row(f"T{i}", abs_m=0.1) for i in range(5)]
    r = u2(rows)
    assert r["point"] == pytest.approx(0.0) and not r["pass"]


def test_u3_tolerates_the_expected_loss_against_bc_but_not_an_unbounded_one():
    small = [row(f"T{i}", raw=0.50, bc=0.53, abs_m=0.1) for i in range(5)]
    assert u3(small)["pass"]                            # -3 pp, within -3.5
    big = [row(f"T{i}", raw=0.50, bc=0.60, abs_m=0.1) for i in range(5)]
    assert not u3(big)["pass"]                          # -10 pp


def test_u4_band_rejects_both_a_no_op_and_a_bc_clone():
    assert not u4([row("A", abs_m=0.1)] * 10)["pass"]           # 0.0
    assert not u4([row("A", abs_m=9.0)] * 10)["pass"]           # 1.0
    mixed = [row("A", abs_m=9.0)] * 2 + [row("A", abs_m=0.1)] * 8
    assert u4(mixed)["pass"] and u4(mixed)["apply_rate"] == 0.2


def test_u5_requires_magnitude_not_merely_a_positive_sign():
    """2T's T5 passed anything above 0; a rho of +0.05 is not a mechanism."""
    # Spearman is rank-based, so a tiny but perfectly ordered gain still gives a
    # high rho.  A weak rho needs the ORDER to be mostly wrong, not the magnitude
    # to be small -- which is itself the reason U5 is stated on rho and not on a
    # slope in accuracy units.
    order = [0, 3, 4, 5, 2, 1]            # rho = +0.086 against i, verified
    rows = []
    for t in range(5):
        for i in range(6):
            rows.append(row(f"T{t}", abs_m=0.3 + 0.3 * i,
                            bc=0.50 + 0.01 * order[i]))
    r = u5(rows)
    assert r["n_positive"] == 5           # sign is right ...
    assert 0.0 < r["median"] < 0.25       # ... but the strength is not there
    assert not r["pass"]


def test_u5_passes_with_a_strong_within_task_relationship():
    rows = []
    for t in range(5):
        for i in range(6):
            rows.append(row(f"T{t}", abs_m=0.3 + 0.3 * i, bc=0.48 + 0.02 * i))
    r = u5(rows)
    assert r["median"] > 0.25 and r["n_positive"] == 5 and r["pass"]


def test_u5_tolerates_one_negative_task_unlike_2ts_all_positive_rule():
    rows = []
    for t in range(5):
        for i in range(6):
            gain = 0.02 * i if t > 0 else -0.02 * i
            rows.append(row(f"T{t}", abs_m=0.3 + 0.3 * i, bc=0.50 + gain))
    r = u5(rows)
    assert r["per_task"]["T0"] < 0
    assert r["n_positive"] == 4 and r["pass"]


def test_u5_skips_tasks_with_too_few_batches():
    rows = [row("BIG", abs_m=0.3 + 0.3 * i, bc=0.48 + 0.02 * i) for i in range(6)]
    rows += [row("TINY", abs_m=1.0), row("TINY", abs_m=2.0)]
    assert "TINY" not in u5(rows)["per_task"]


def test_u6_reports_the_best_harmful_free_threshold_for_both_statistics():
    rows = [row("A", abs_m=0.3, sd=1.0, bc=0.40),          # harmful if applied
            row("A", abs_m=2.0, sd=1.0, bc=0.60),
            row("A", abs_m=2.4, sd=4.0, bc=0.62)]
    r = u6(rows, list(rows))
    assert r["heldout_best_absm"]["harmful"] == 0
    assert r["heldout_best_absm"]["mean"] > 0
    assert r["heldout_best_norm"] is not None


def test_u6_returns_none_when_no_threshold_is_harmful_free():
    rows = [row("A", abs_m=9.0, sd=1.0, bc=0.30)]          # harmful at every tau
    r = u6(rows, list(rows))
    assert r["heldout_best_absm"] is None


def test_u7_flags_an_all_or_nothing_gate():
    rows = [row("HI", abs_m=9.0)] * 3 + [row("LO", abs_m=0.1)] * 3
    assert u7(rows)["is_task_filter"]
    assert not u7(rows + [row("HI", abs_m=0.1)])["is_task_filter"]


def _res(**kw):
    base = {"U1": {"pass": True}, "U2": {"pass": True}, "U3": {"pass": True},
            "U4": {"pass": True}, "U5": {"pass": True},
            "U6": {"heldout_best_absm": {"mean": 4.0},
                   "heldout_best_norm": {"mean": 2.0}},
            "U7": {"is_task_filter": False}}
    base.update(kw)
    return base


def test_outcome_1_when_everything_passes():
    assert decide(_res()) == "OUTCOME_1_working_and_mechanistically_explained"


def test_outcome_2_when_the_mechanism_criterion_fails():
    assert decide(_res(U5={"pass": False})) == \
        "OUTCOME_2_works_but_the_mechanism_is_not_established"


def test_outcome_3_when_the_gate_is_safe_but_gains_nothing():
    assert decide(_res(U2={"pass": False})) == \
        "OUTCOME_3_gate_is_safe_and_pointless"


def test_outcome_4_takes_priority_because_a_failed_tail_voids_the_premise():
    assert decide(_res(U1={"pass": False}, U2={"pass": False})) == \
        "OUTCOME_4_threshold_did_not_transfer_stop_pursuing_gated_bc"


def test_outcome_5_when_the_normalised_statistic_is_at_least_as_good():
    assert decide(_res(U6={"heldout_best_absm": {"mean": 3.0},
                          "heldout_best_norm": {"mean": 3.0}})) == \
        "OUTCOME_5_prefer_the_normalised_statistic"


def test_u3_failure_alone_does_not_silently_become_outcome_1():
    out = decide(_res(U3={"pass": False}))
    assert out != "OUTCOME_1_working_and_mechanistically_explained"
    assert out == "PROTOCOL_GAP_record_in_amendment"


def test_u4_failure_alone_is_a_protocol_gap_not_a_pass():
    assert decide(_res(U4={"pass": False})) == "PROTOCOL_GAP_record_in_amendment"


def test_missing_u6_data_does_not_crash_the_branch():
    assert decide(_res(U6={"heldout_best_absm": None,
                          "heldout_best_norm": None})) == \
        "OUTCOME_1_working_and_mechanistically_explained"
