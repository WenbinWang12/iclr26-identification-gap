"""Tests for the Phase-2S judgement, written before the real run returned.

The point of these is that each criterion can be shown to FIRE and to NOT FIRE on
constructed data, so a real-data verdict cannot be an artefact of a criterion that
never discriminates. The two defective criteria (§A2.3, §A2.4) get tests showing
exactly how they fail, since that failure is a recorded finding.
"""
from __future__ import annotations

import numpy as np

from experiments.phase2s_sio.decide_2s import (
    applicable,
    s1,
    s2,
    s3,
    s3b,
    s4,
    s5,
    s5b,
    s6,
)


def mk(task, *, r_raw, r_sio, r_bc, applied=True, K_S=2, imbal=None,
       slope_inputs=None, pos=1):
    """One (task, stage) record in the shape run_qoc.py writes."""
    sio = {"applicable": K_S == 2, "K_S": K_S}
    if K_S == 2:
        sio.update({"R_sio": r_sio, "R_bc": r_bc, "applied": applied,
                    "reason": "applied" if applied else "snr",
                    "tau": 0.5, "snr": 2.0, "rho": 1.2, "rho_crit": 1.8,
                    "ci": [0.2, 0.8]})
        sio["imbalanced"] = imbal or {}
        sio["slope_inputs"] = slope_inputs or []
    return (1, pos, task, {"R_raw": r_raw, "sio": sio})


def rows_where_sio_wins(n_tasks=6, gain=0.03):
    """SIO 3 pp above raw and above BC, on several distinct task clusters."""
    return [mk(f"T{i}", r_raw=0.60, r_sio=0.60 + gain, r_bc=0.60 + gain / 3)
            for i in range(n_tasks)]


# --- S1 -------------------------------------------------------------------

def test_s1_fires_when_sio_clearly_beats_raw():
    res = s1(rows_where_sio_wins())
    assert res["verdict"] is True
    assert res["median"] > 1.0 and res["ci"][0] > 0


def test_s1_does_not_fire_on_a_tiny_gain_below_the_threshold():
    """0.3 pp is real but below the frozen 1.0 pp bar."""
    res = s1(rows_where_sio_wins(gain=0.003))
    assert res["verdict"] is False


def test_s1_does_not_fire_when_the_ci_straddles_zero():
    rows = [mk("A", r_raw=0.6, r_sio=0.70, r_bc=0.6),
            mk("B", r_raw=0.6, r_sio=0.45, r_bc=0.6),
            mk("C", r_raw=0.6, r_sio=0.72, r_bc=0.6),
            mk("D", r_raw=0.6, r_sio=0.44, r_bc=0.6)]
    assert s1(rows)["verdict"] is False


# --- S2 -------------------------------------------------------------------

def test_s2_allows_a_tie_with_bc_on_balanced_batches():
    """S2 only requires 'does not lose' where BC is already sound."""
    rows = [mk(f"T{i}", r_raw=0.6, r_sio=0.65, r_bc=0.65) for i in range(5)]
    res = s2(rows)
    assert res["median"] == 0.0
    assert res["verdict"] is True


def test_s2_fails_when_sio_loses_to_bc():
    rows = [mk(f"T{i}", r_raw=0.6, r_sio=0.62, r_bc=0.67) for i in range(5)]
    assert s2(rows)["verdict"] is False


# --- S3 and the S3b defect ------------------------------------------------

def _imbal(ratio, *, r_sio, r_bc, r_raw, applied):
    return {ratio: {"R_sio": r_sio, "R_bc": r_bc, "R_raw": r_raw,
                    "applied": applied, "reason": "applied" if applied else "snr",
                    "tau": 0.5, "n": 128, "class0_share": float(ratio)}}


def test_s3_fires_when_sio_genuinely_beats_bc_under_skew():
    rows = [mk(f"T{i}", r_raw=0.6, r_sio=0.63, r_bc=0.61,
               imbal=_imbal("0.9", r_sio=0.64, r_bc=0.58, r_raw=0.60,
                            applied=True))
            for i in range(6)]
    assert s3(rows, "0.9")["verdict"] is True


def test_s3_is_fooled_by_abstention_which_is_why_s3b_exists():
    """§A2.3, the recorded defect: every batch ABSTAINS, so SIO == raw. BC loses
    2 pp, so `SIO - BC` is +2 pp and S3's letter passes -- while SIO did nothing."""
    rows = [mk(f"T{i}", r_raw=0.6, r_sio=0.60, r_bc=0.58,
               imbal=_imbal("0.9", r_sio=0.60, r_bc=0.58, r_raw=0.60,
                            applied=False))
            for i in range(6)]
    assert s3(rows, "0.9")["verdict"] is True, "S3 should be fooled here"
    b = s3b(rows, "0.9")
    assert b["apply_rate"] == 0.0
    assert b["verdict"] is False, "S3b must not be fooled"
    assert b["conclusion_wording"] == "SIO abstains on most skewed batches"


def test_s3b_fires_only_when_the_gate_acts_on_most_batches():
    rows = [mk(f"T{i}", r_raw=0.6, r_sio=0.64, r_bc=0.58,
               imbal=_imbal("0.9", r_sio=0.64, r_bc=0.58, r_raw=0.60,
                            applied=True))
            for i in range(6)]
    b = s3b(rows, "0.9")
    assert b["apply_rate"] == 1.0 and b["verdict"] is True


def test_s3b_reports_a_rate_between_the_extremes():
    rows = ([mk(f"A{i}", r_raw=0.6, r_sio=0.64, r_bc=0.58,
                imbal=_imbal("0.7", r_sio=0.64, r_bc=0.58, r_raw=0.60,
                             applied=True)) for i in range(3)]
            + [mk(f"B{i}", r_raw=0.6, r_sio=0.60, r_bc=0.58,
                  imbal=_imbal("0.7", r_sio=0.60, r_bc=0.58, r_raw=0.60,
                               applied=False)) for i in range(3)])
    assert s3b(rows, "0.7")["apply_rate"] == 0.5


# --- S4 -------------------------------------------------------------------

def test_s4_catches_an_abstention_that_is_not_exactly_raw():
    """The protocol demands bit-for-bit equality, not approximate."""
    rows = [mk("A", r_raw=0.6, r_sio=0.6000001, r_bc=0.6, applied=False)]
    res = s4(rows)
    assert res["violations"] and res["verdict"] is False


def test_s4_accepts_exact_abstention_but_rejects_abstaining_always():
    rows = [mk(f"T{i}", r_raw=0.6, r_sio=0.6, r_bc=0.6, applied=False)
            for i in range(5)]
    res = s4(rows)
    assert not res["violations"]
    assert res["abstain_rate_balanced"] == 1.0
    assert res["verdict"] is False, "a rule that always abstains is vacuous"


def test_s4_passes_when_the_gate_mostly_acts_and_abstains_exactly():
    rows = ([mk(f"A{i}", r_raw=0.6, r_sio=0.64, r_bc=0.6) for i in range(4)]
            + [mk("B", r_raw=0.6, r_sio=0.6, r_bc=0.6, applied=False)])
    assert s4(rows)["verdict"] is True


# --- S5 (frozen, defective) vs S5b (slope) --------------------------------

def _slopes(taus, bcs, ratios=(0.5, 0.7, 0.9)):
    return [{"ratio": r, "tau": t, "bc": b}
            for r, t, b in zip(ratios, taus, bcs)]


def test_s5_fails_even_when_the_mechanism_holds():
    """§A2.4: tau scatters around a constant (noise) while BC marches (bias), and
    their sds are comparable -- so the sd-ratio test fails on data that supports
    the mechanism. This is the recorded criterion defect.

    Each cluster gets a *different* tau pattern, so tau's slope varies across tasks
    with no consistent sign -- which is what "no systematic dependence" looks like.
    Giving every cluster identical values would collapse the bootstrap CI to zero
    width and make the test measure the fixture instead of the criterion.
    """
    taus = [[0.57, 0.67, 0.52], [0.62, 0.55, 0.66], [0.59, 0.71, 0.54],
            [0.65, 0.53, 0.63], [0.55, 0.64, 0.58]]
    bcs = [0.21, 0.04, -0.12]
    rows = [mk(f"T{i}", r_raw=0.6, r_sio=0.64, r_bc=0.6,
               slope_inputs=_slopes(taus[i], bcs)) for i in range(5)]
    assert s5(rows)["verdict"] is False
    b = s5b(rows)
    assert b["bc_ci_excludes_zero"] is True
    assert b["sio_ci_contains_zero"] is True
    assert b["verdict"] is True, "the slope form sees the mechanism S5 misses"


def test_s5b_fails_when_sio_itself_drifts_with_the_mix():
    """If tau tracked the marginal, SIO would have BC's defect too."""
    rows = [mk(f"T{i}", r_raw=0.6, r_sio=0.64, r_bc=0.6,
               slope_inputs=_slopes([0.9, 0.5, 0.1], [0.21, 0.04, -0.12]))
            for i in range(5)]
    res = s5b(rows)
    assert res["sio_ci_contains_zero"] is False
    assert res["verdict"] is False


def test_s5b_fails_when_bc_shows_no_drift_so_there_is_no_defect_to_fix():
    rows = [mk(f"T{i}", r_raw=0.6, r_sio=0.64, r_bc=0.6,
               slope_inputs=_slopes([0.57, 0.67, 0.52], [0.10, 0.10, 0.10]))
            for i in range(5)]
    assert s5b(rows)["verdict"] is False


# --- S6 coverage ----------------------------------------------------------

def test_s6_counts_only_binary_scopes_and_reports_the_histogram():
    rows = ([mk(f"B{i}", r_raw=0.6, r_sio=0.64, r_bc=0.6) for i in range(3)]
            + [mk(f"M{i}", r_raw=0.6, r_sio=0, r_bc=0, K_S=3) for i in range(7)])
    res = s6(rows, applicable(rows))
    assert res["n_task_stage_pairs"] == 10
    assert res["n_with_K_S_2"] == 3
    assert res["coverage"] == 0.3
    assert res["K_S_histogram"] == {"2": 3, "3": 7}


def test_s6_fails_when_nothing_is_scorable():
    rows = [mk(f"M{i}", r_raw=0.6, r_sio=0, r_bc=0, K_S=5) for i in range(4)]
    res = s6(rows, applicable(rows))
    assert res["n_with_K_S_2"] == 0 and res["verdict"] is False


def test_applicable_drops_non_binary_scopes():
    rows = [mk("A", r_raw=0.6, r_sio=0.64, r_bc=0.6),
            mk("B", r_raw=0.6, r_sio=0, r_bc=0, K_S=14)]
    assert [r[2] for r in applicable(rows)] == ["A"]
