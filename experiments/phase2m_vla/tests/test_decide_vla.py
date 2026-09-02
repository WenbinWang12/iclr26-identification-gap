"""`decide_vla.py` 的测试。

评分代码是在结果产生之后写的，所以它比平时更需要测试：任何一处符号错误都会
把结论翻个面。重点测三类事——判据的方向（哪边算 pass）、配对是否真按任务配、
以及 §2 要求的"peak 必须并列上报"没有被聚合掉。
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.phase2j_offset_conflict.decide import cluster_bootstrap_ci
from experiments.phase2m_vla import decide_vla as dv


def _rec(*, acc, scoped, peak, radii1=None, radii2=None, scope_size=2):
    """造一条 record：n 个任务，一个最终阶段带 ACC/ACC_scoped，每任务一个 peak 阶段。"""
    names = sorted(acc)
    stages = []
    for i, n in enumerate(names):
        stages.append({
            "position": i, "n_steps": 38, "trained_task": n,
            "scope_radii": {},
            "tasks": {n: {"scorable": True, "R_raw": peak[n],
                          "R_shr_global": acc[n], "R_shr_scoped": scoped[n],
                          "R_orc": scoped[n], "scope_size": scope_size,
                          "Delta_id_global": 0.0, "scope": "s", "qoc": {},
                          "scope_recoverable": 0.0}},
        })
    stages.append({
        "position": len(names), "n_steps": 38, "trained_task": "last",
        "scope_radii": {"1": radii1 or {}, "2": radii2 or {}},
        "tasks": {n: {"scorable": True, "R_raw": acc[n],
                      "R_shr_global": acc[n], "R_shr_scoped": scoped[n],
                      "R_orc": scoped[n], "scope_size": scope_size,
                      "Delta_id_global": 0.0, "scope": "s", "qoc": {},
                      "scope_recoverable": 0.0}
                  for n in names},
    })
    return {"args": {}, "stages": stages}


def _pair(delta_acc=0.05, delta_scoped=0.05, peak_treat=0.74, peak_base=0.7381):
    names = ["t%d" % i for i in range(6)]
    base_acc = {n: 0.65 for n in names}
    treat_acc = {n: 0.65 + delta_acc for n in names}
    base_sc = {n: 0.70 for n in names}
    treat_sc = {n: 0.70 + delta_scoped for n in names}
    treat = _rec(acc=treat_acc, scoped=treat_sc, peak={n: peak_treat for n in names})
    base = _rec(acc=base_acc, scoped=base_sc, peak={n: peak_base for n in names})
    return [treat], [base]


def test_bootstrap_is_the_shared_phase2j_function():
    assert dv.cluster_bootstrap_ci is cluster_bootstrap_ci


def test_point_estimate_is_a_mean_not_a_median():
    """§2 把 ACC 定义为对任务取平均；用中位数会静默改掉主指标的定义。"""
    names = ["t%d" % i for i in range(6)]
    acc = {n: 0.60 for n in names}
    acc["t0"] = 0.90  # 均值与中位数在此处不同
    treat = [_rec(acc=acc, scoped=acc, peak={n: 0.74 for n in names})]
    base = [_rec(acc={n: 0.60 for n in names}, scoped={n: 0.60 for n in names},
                 peak={n: 0.7381 for n in names})]
    out = dv.evaluate(treat, base, m="4", router="prototype")
    assert out["criterion_1"]["point"] == pytest.approx(0.30 / 6)


def test_criterion_1_passes_only_when_ci_excludes_zero_on_the_positive_side():
    treat, base = _pair(delta_acc=0.05)
    out = dv.evaluate(treat, base, m="4", router="prototype")
    assert out["criterion_1"]["point"] == pytest.approx(0.05)
    assert out["criterion_1"]["passed"] is True

    treat, base = _pair(delta_acc=-0.05)
    out = dv.evaluate(treat, base, m="4", router="prototype")
    assert out["criterion_1"]["passed"] is False


def test_zero_difference_does_not_pass():
    treat, base = _pair(delta_acc=0.0, delta_scoped=0.0)
    out = dv.evaluate(treat, base, m="4", router="prototype")
    assert out["criterion_1"]["passed"] is False
    assert out["criterion_2"]["passed"] is False


def test_criterion_3_plasticity_guard_direction():
    """peak 掉超过 2pp 才算 fail；peak 更高绝不能算 fail。"""
    treat, base = _pair(peak_treat=0.7381 - 0.03)
    out = dv.evaluate(treat, base, m="4", router="prototype")
    assert out["criterion_3"]["passed"] is False
    assert out["criterion_3"]["drop_pp_vs_reference"] == pytest.approx(3.0)

    treat, base = _pair(peak_treat=0.7381 - 0.01)
    assert dv.evaluate(treat, base, m="4",
                       router="prototype")["criterion_3"]["passed"] is True

    treat, base = _pair(peak_treat=0.90)
    assert dv.evaluate(treat, base, m="4",
                       router="prototype")["criterion_3"]["passed"] is True


def test_criterion_4_requires_both_radii_to_decrease():
    names = ["t%d" % i for i in range(6)]
    common = dict(acc={n: 0.65 for n in names}, scoped={n: 0.70 for n in names},
                  peak={n: 0.74 for n in names})
    r_small = {"A|B": {"q_m": 0.10, "members": ["x"], "n_centres": 1}}
    r_big = {"A|B": {"q_m": 0.30, "members": ["x"], "n_centres": 1}}

    # 两个都降 -> pass
    treat = [_rec(radii1=r_small, radii2=r_small, **common)]
    base = [_rec(radii1=r_big, radii2=r_big, **common)]
    out = dv.evaluate(treat, base, m="4", router="prototype")
    assert out["criterion_4"]["passed"] is True
    assert out["criterion_4"]["q_1"]["decreased"] is True

    # q_1 降、q_2 升 -> fail（§3 要求两个都降）
    treat = [_rec(radii1=r_small, radii2=r_big, **common)]
    base = [_rec(radii1=r_big, radii2=r_small, **common)]
    out = dv.evaluate(treat, base, m="4", router="prototype")
    assert out["criterion_4"]["passed"] is False


def test_criterion_5_flags_nonzero_singleton_gap():
    names = ["t%d" % i for i in range(6)]
    rec = _rec(acc={n: 0.65 for n in names}, scoped={n: 0.70 for n in names},
               peak={n: 0.74 for n in names}, scope_size=1)
    assert dv.prop1_violations([rec])["violations"] == 0
    # 把一个单任务域的 R_orc 挪开 -> 必须被抓到
    rec["stages"][-1]["tasks"]["t0"]["R_orc"] = 0.75
    got = dv.prop1_violations([rec])
    assert got["violations"] == 1
    assert got["worst_abs_gap"] == pytest.approx(0.05)


def test_multi_task_scopes_are_not_checked_by_prop1():
    """Prop 1 只对单任务域断言恒等；多任务域有非零 Δ 是预期的。"""
    names = ["t%d" % i for i in range(6)]
    rec = _rec(acc={n: 0.65 for n in names}, scoped={n: 0.70 for n in names},
               peak={n: 0.74 for n in names}, scope_size=3)
    rec["stages"][-1]["tasks"]["t0"]["R_orc"] = 0.99
    assert dv.prop1_violations([rec])["checked"] == 0
    assert dv.prop1_violations([rec])["violations"] == 0


def test_mismatched_task_sets_are_refused_not_silently_intersected():
    treat, base = _pair()
    del treat[0]["stages"][-1]["tasks"]["t0"]
    with pytest.raises(ValueError, match="task sets differ"):
        dv.evaluate(treat, base, m="4", router="prototype")


def test_final_stage_is_chosen_by_position_not_list_order():
    names = ["t%d" % i for i in range(3)]
    rec = _rec(acc={n: 0.65 for n in names}, scoped={n: 0.70 for n in names},
               peak={n: 0.74 for n in names})
    last = rec["stages"][-1]
    rec["stages"] = [last] + rec["stages"][:-1]  # 打乱顺序
    assert dv._final_stage(rec) is last


def test_training_gate_uses_peaks_and_a_strict_threshold():
    names = ["t%d" % i for i in range(6)]
    rec = _rec(acc={n: 0.65 for n in names}, scoped={n: 0.70 for n in names},
               peak={n: 0.60 for n in names})
    assert dv.training_gate([rec])["passed"] is False  # 0.60 不 > 0.60
    rec2 = _rec(acc={n: 0.65 for n in names}, scoped={n: 0.70 for n in names},
                peak={n: 0.61 for n in names})
    assert dv.training_gate([rec2])["passed"] is True


def test_peak_is_reported_for_both_arms_alongside_the_levels():
    """§2：peak 必须与水平并列，不能被聚合掉。"""
    treat, base = _pair(peak_treat=0.80, peak_base=0.7381)
    out = dv.evaluate(treat, base, m="4", router="prototype")
    assert out["criterion_3"]["peak_treat"] == pytest.approx(0.80)
    assert out["criterion_3"]["peak_base"] == pytest.approx(0.7381)
    assert "ACC_treat" in out and "ACC_base" in out


def test_per_task_diffs_are_reported_in_pp():
    treat, base = _pair(delta_acc=0.05)
    out = dv.evaluate(treat, base, m="4", router="prototype")
    assert out["per_task"]["t0"]["diff_pp"] == pytest.approx(5.0)


def test_criterion_4_reports_when_no_scope_is_shared():
    """域交集为空时必须显式说"无法评估"，不能给个 NaN 就当通过。"""
    treat, base = _pair()
    out = dv.evaluate(treat, base, m="4", router="prototype")
    assert out["criterion_4"]["evaluable"] is False
    assert out["criterion_4"]["passed"] is False
