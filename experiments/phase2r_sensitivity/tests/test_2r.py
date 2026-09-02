"""Phase-2R 的反伪影测试，对应冻结协议 §6 逐条。

协议 SHA 038c238b22615e4fddbc4629feb9c8a9bcc9f70af7a5c0dcb5a6702843d530df。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.phase2r_sensitivity.decide_2r import (  # noqa: E402
    ols_slope, r3, r4, r5, slope_ci,
)
from experiments.phase2r_sensitivity.panel import build_panel, live_members  # noqa: E402


def _stage(pos, trained, tasks, radii):
    return {"position": pos, "trained_task": trained, "n_steps": 1,
            "tasks": tasks, "scope_radii": {"1": radii}}


def _task(scope, *, scope_size, orc=0.8, shr_scoped=0.8, shr_global=0.7,
          raw=0.6, sso1=None, scorable=True):
    d = {"scope": scope, "scope_size": scope_size, "scorable": scorable,
         "R_orc": orc, "R_shr_scoped": shr_scoped, "R_shr_global": shr_global,
         "R_raw": raw}
    if sso1 is not None:
        d["R_sso1"] = sso1
    return d


def _write(tmp_path, stages, singletons):
    p = tmp_path / "run.json"
    p.write_text(json.dumps({
        "args": {"seed": 1}, "singleton_scopes": singletons, "stages": stages,
    }), encoding="utf-8")
    return p


def test_m_live_comes_from_scope_radii_not_scope_size(tmp_path):
    """协议 §2 记录的陷阱：scope_size 是最终规模，不是活跃规模。

    fixture 里 scope_size=4 而该 stage 只有 1 个活跃成员，分析必须取到 1。
    """

    stages = [_stage(1, "WiC", {"WiC": _task("False|True", scope_size=4)},
                     {"False|True": {"members": ["WiC"], "n_centres": 1, "q_m": 0.0}})]
    panel = build_panel(_write(tmp_path, stages, []))
    assert len(panel) == 1
    assert panel[0].m_live == 1, "m 必须来自 scope_radii.members，不是 scope_size"
    assert panel[0].m_live != 4


def test_m_live_tracks_growth_across_stages(tmp_path):
    stages = [
        _stage(1, "WiC", {"WiC": _task("False|True", scope_size=4)},
               {"False|True": {"members": ["WiC"], "n_centres": 1, "q_m": 0.0}}),
        _stage(2, "QQP", {"WiC": _task("False|True", scope_size=4),
                          "QQP": _task("False|True", scope_size=4)},
               {"False|True": {"members": ["WiC", "QQP"], "n_centres": 1, "q_m": 0.2}}),
    ]
    panel = build_panel(_write(tmp_path, stages, []))
    assert sorted({o.m_live for o in panel}) == [1, 2]


def test_age_is_derived_from_trained_task_and_never_negative(tmp_path):
    stages = [
        _stage(1, "A", {"A": _task("sA", scope_size=1)},
               {"sA": {"members": ["A"], "n_centres": 1, "q_m": 0.0}}),
        _stage(2, "B", {"A": _task("sA", scope_size=1),
                        "B": _task("sB", scope_size=1)},
               {"sA": {"members": ["A"], "n_centres": 1, "q_m": 0.0},
                "sB": {"members": ["B"], "n_centres": 1, "q_m": 0.0}}),
    ]
    panel = build_panel(_write(tmp_path, stages, ["sA", "sB"]))
    ages = {(o.task, o.stage): o.age for o in panel}
    assert ages[("A", 1)] == 0 and ages[("A", 2)] == 1 and ages[("B", 2)] == 0
    assert all(o.age >= 0 for o in panel)


def test_a_task_scored_before_it_was_trained_is_rejected(tmp_path):
    """负 age 是接线错误，不能静默算进回归。"""

    stages = [
        _stage(1, "A", {"A": _task("sA", scope_size=1),
                        "B": _task("sB", scope_size=1)},
               {"sA": {"members": ["A"], "n_centres": 1, "q_m": 0.0},
                "sB": {"members": ["B"], "n_centres": 1, "q_m": 0.0}}),
        _stage(2, "B", {"B": _task("sB", scope_size=1)},
               {"sB": {"members": ["B"], "n_centres": 1, "q_m": 0.0}}),
    ]
    with pytest.raises(ValueError, match="negative age"):
        build_panel(_write(tmp_path, stages, ["sA", "sB"]))


def test_singleton_flag_comes_from_the_run_not_the_label_string(tmp_path):
    """§6：单例判定取自 run 自身的 singleton_scopes。

    fixture 里 'False|True' 只有一个成员，但 run 未把它列为单例；
    分析必须尊重 run 的声明而不是自己数成员。
    """

    stages = [_stage(1, "WiC", {"WiC": _task("False|True", scope_size=4)},
                     {"False|True": {"members": ["WiC"], "n_centres": 1, "q_m": 0.0}})]
    panel = build_panel(_write(tmp_path, stages, []))
    assert panel[0].is_singleton is False


def test_non_scorable_tasks_are_excluded(tmp_path):
    stages = [_stage(1, "A", {"A": _task("sA", scope_size=1),
                              "Yelp": _task("sY", scope_size=2, scorable=False)},
                     {"sA": {"members": ["A"], "n_centres": 1, "q_m": 0.0}})]
    panel = build_panel(_write(tmp_path, stages, ["sA"]))
    assert [o.task for o in panel] == ["A"]


def test_live_members_returns_empty_for_absent_scope():
    assert live_members({"scope_radii": {"1": {}}}, "nope") == []


def test_ols_slope_recovers_a_known_line():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [3.0, 5.0, 7.0, 9.0]          # y = 2x + 1
    assert ols_slope(xs, ys) == pytest.approx(2.0)


def test_ols_slope_is_nan_when_x_has_no_variation():
    """x 无变异时返回 nan，而不是 0 —— 0 会伪装成'测到了无效应'。"""

    assert np.isnan(ols_slope([2.0, 2.0, 2.0], [1.0, 5.0, 9.0]))


def test_slope_ci_with_a_single_cluster_reports_nan_ci():
    """簇数 < 2 时 CI 不可得，必须是 nan，不能悄悄退化成点估计。"""

    res = slope_ci({"only": [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]})
    assert res["n_clusters"] == 1
    assert np.isnan(res["ci"][0]) and np.isnan(res["ci"][1])
    assert res["ci_usable"] is False


def test_a_degenerate_cluster_does_not_nan_out_the_whole_ci():
    """回归测试：单簇 x 无变异曾让 percentile 把整个 CI 变成 nan。

    'flat' 只有 x=0 一个取值。当某次重采样恰好只抽到它时斜率无定义；那一次
    必须被丢弃并计数，而不是让 CI 整体作废。
    """

    pairs = {
        "flat": [(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)],
        "rising": [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
        "also": [(0.0, 0.5), (1.0, 1.4), (2.0, 2.6), (3.0, 3.3)],
    }
    res = slope_ci(pairs)
    assert not np.isnan(res["ci"][0]) and not np.isnan(res["ci"][1])
    assert res["n_degenerate_draws"] > 0, "该 fixture 必须触发退化抽样"
    # 3 簇时全抽中 flat 的概率是 (1/3)^3 ≈ 3.7%，超过 1% 界，故 CI 判为不可用。
    assert res["frac_degenerate"] > 0.01 and res["ci_usable"] is False


def test_the_one_percent_rule_marks_a_rare_degeneracy_usable():
    """簇多时退化极罕见，CI 应判为可用。5 簇 → (1/5)^5 ≈ 0.03%，与实测同量级。"""

    pairs = {"flat": [(0.0, 1.0)]}
    for k in "abcd":
        pairs[k] = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    res = slope_ci(pairs)
    assert res["n_degenerate_draws"] > 0
    assert res["frac_degenerate"] <= 0.01 and res["ci_usable"] is True


def test_degenerate_draws_are_dropped_not_scored_as_zero():
    """丢弃 ≠ 记作 0。若记作 0，CI 会被人为拉向 0。

    构造一个斜率恒正的设计，使 CI 下界严格为正；若退化抽样被当成 0，
    下界会被压到 <= 0。
    """

    pairs = {
        "flat": [(0.0, 5.0)],
        "a": [(0.0, 0.0), (1.0, 10.0), (2.0, 20.0)],
        "b": [(0.0, 1.0), (1.0, 11.0), (2.0, 21.0)],
        "c": [(0.0, 2.0), (1.0, 12.0), (2.0, 22.0)],
    }
    res = slope_ci(pairs)
    assert res["n_degenerate_draws"] > 0
    assert res["ci"][0] > 0.0, "退化抽样若记作斜率 0，此下界会 <= 0"


def test_ci_unusable_blocks_a_pass_verdict():
    """ci_usable=False 时不得判通过，即使点估计与 CI 看起来都为正。"""

    from experiments.phase2r_sensitivity.decide_2r import _verdict
    good = {"slope_pp_per_unit": 1.0, "ci": [0.5, 1.5], "ci_usable": True}
    bad = {"slope_pp_per_unit": 1.0, "ci": [0.5, 1.5], "ci_usable": False}
    assert _verdict(good, positive=True) is True
    assert _verdict(bad, positive=True) is False


def test_r5_flags_a_nonzero_scoped_gap_on_a_singleton(tmp_path):
    stages = [_stage(1, "A", {"A": _task("sA", scope_size=1, orc=0.8,
                                         shr_scoped=0.75)},
                     {"sA": {"members": ["A"], "n_centres": 1, "q_m": 0.0}})]
    out = r5(build_panel(_write(tmp_path, stages, ["sA"])))
    assert out["pass"] is False and len(out["violations"]) == 1


def test_r5_passes_when_scoped_equals_oracle_on_singletons(tmp_path):
    stages = [_stage(1, "A", {"A": _task("sA", scope_size=1, orc=0.8,
                                         shr_scoped=0.8)},
                     {"sA": {"members": ["A"], "n_centres": 1, "q_m": 0.0}})]
    out = r5(build_panel(_write(tmp_path, stages, ["sA"])))
    assert out["pass"] is True and out["n_singleton_obs"] == 1


def test_r3_and_r4_read_different_regressors(tmp_path):
    """R3 对 age、R4 对 pos。构造两任务，使二者斜率符号相反。"""

    def blk(pos, trained, rows, radii):
        return _stage(pos, trained, rows, radii)

    rad = {"sA": {"members": ["A"], "n_centres": 1, "q_m": 0.0},
           "sB": {"members": ["B"], "n_centres": 1, "q_m": 0.0}}
    # A 训于 p1，B 训于 p2。stale 随 age 上升，但晚训的 B 整体 stale 更低。
    stages = [
        blk(1, "A", {"A": _task("sA", scope_size=1, orc=0.8, shr_scoped=0.8,
                                sso1=0.80)}, rad),
        blk(2, "B", {"A": _task("sA", scope_size=1, orc=0.8, shr_scoped=0.8,
                                sso1=0.70),
                     "B": _task("sB", scope_size=1, orc=0.8, shr_scoped=0.8,
                                sso1=0.80)}, rad),
        blk(3, "C", {"A": _task("sA", scope_size=1, orc=0.8, shr_scoped=0.8,
                                sso1=0.60),
                     "B": _task("sB", scope_size=1, orc=0.8, shr_scoped=0.8,
                                sso1=0.78),
                     "C": _task("sC", scope_size=1, orc=0.8, shr_scoped=0.8,
                                sso1=0.80)},
            dict(rad, sC={"members": ["C"], "n_centres": 1, "q_m": 0.0})),
    ]
    panel = build_panel(_write(tmp_path, stages, ["sA", "sB", "sC"]))
    a, p = r3(panel), r4(panel)
    assert a["slope_pp_per_unit"] > 0.0, "stale 随 age 上升"
    assert p["slope_pp_per_unit"] < 0.0, "晚训任务 stale 更低"
    assert a["n_obs"] == p["n_obs"] == 6


def test_r3_ignores_multi_task_scopes(tmp_path):
    """R3 只在单例上有意义（Prop 1），多任务 scope 必须不进回归。"""

    stages = [_stage(1, "W", {"W": _task("False|True", scope_size=4, sso1=0.5)},
                     {"False|True": {"members": ["W"], "n_centres": 1, "q_m": 0.0}})]
    panel = build_panel(_write(tmp_path, stages, []))
    assert r3(panel)["n_obs"] == 0


def test_stale_is_none_without_sso1(tmp_path):
    """非 2P 的臂没有 R_sso1，stale 不可得；不能当 0 处理。"""

    stages = [_stage(1, "A", {"A": _task("sA", scope_size=1)},
                     {"sA": {"members": ["A"], "n_centres": 1, "q_m": 0.0}})]
    panel = build_panel(_write(tmp_path, stages, ["sA"]))
    assert panel[0].stale is None
    assert r3(panel)["n_obs"] == 0
