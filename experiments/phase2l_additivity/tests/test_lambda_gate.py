"""协议 §A2.1 的修正闸门测试。

两条核心不变量：不许在留出划分上选 λ；**可塑性必须抵消**（这是第一版
闸门的缺陷，见协议 §A2）。
"""

from __future__ import annotations

import inspect

import pytest

from experiments.phase2l_additivity.lambda_gate import retention_score, select


def _rec(method, lam, *, peak, final, n_stages=6):
    """造最小 record。

    `peak[t]` 写在 t 作为 trained_task 的那个 stage 上，`final[t]` 写在最后
    一个 stage 上——这正是修正闸门需要的两个时点。
    """
    order = ["T%d" % i for i in range(1, n_stages + 1)]
    stages = []
    for i, name in enumerate(order, start=1):
        tasks = {}
        for seen in order[:i]:
            entry = {"scorable": True, "R_raw": 0.5}
            if i == n_stages and seen in final:
                entry["R_update_raw"] = final[seen]
            elif seen == name and seen in peak:
                entry["R_update_raw"] = peak[seen]
            tasks[seen] = entry
        stages.append({"position": i, "trained_task": name, "tasks": tasks})
    return {"args": {"cl_method": method, "olora_lambda": lam}, "stages": stages}


def _flat(value, n=5):
    return {"T%d" % i: value for i in range(1, n + 1)}


def test_scores_retention_as_final_minus_own_peak():
    got = retention_score(_rec("olora", 0.5, peak=_flat(0.90), final=_flat(0.70)))
    assert set(got["per_task"]) == {"T1", "T2", "T3", "T4", "T5"}
    assert got["mean"] == pytest.approx(-0.20)


def test_plasticity_cancels_exactly():
    """这是 §A2 缺陷的回归测试。

    两臂保持完全相同（都掉 10pp），但 B 臂每个任务的 peak 高 20pp。
    旧闸门会选 B（final 更高），修正闸门必须判两者相等。
    """
    a = retention_score(_rec("olora", 0.1, peak=_flat(0.60), final=_flat(0.50)))
    b = retention_score(_rec("olora", 0.5, peak=_flat(0.80), final=_flat(0.70)))
    assert a["mean"] == pytest.approx(b["mean"])
    assert a["mean"] == pytest.approx(-0.10)


def test_higher_plasticity_with_worse_retention_loses():
    """第一次确认跑的真实形状：peak 高但遗忘更多，必须被判为更差。"""
    plastic = _rec("olora", 1.0, peak=_flat(0.77), final=_flat(0.65))   # −12pp
    steady = _rec("olora", 0.1, peak=_flat(0.74), final=_flat(0.66))    # −8pp
    out = select({"plastic": plastic, "steady": steady})
    assert out["selected_lambda"] == 0.1
    assert out["selected_mean"] == pytest.approx(-0.08)


def test_missing_peak_measurement_raises():
    """只有 final 没有 peak 时必须报错，不许退回旧的混合口径。

    实际抛的是 KeyError（指名道姓说缺的是 peak 那一次测量），比
    "both peak and final" 这种笼统的 ValueError 更有用，所以钉前者。
    """
    rec = _rec("olora", 0.5, peak={}, final=_flat(0.70))
    with pytest.raises(KeyError, match="at its own peak"):
        retention_score(rec)


def test_missing_update_key_raises_rather_than_falling_back_to_audit():
    rec = _rec("olora", 0.5, peak=_flat(0.9), final={})
    with pytest.raises(KeyError, match="R_update_raw missing"):
        retention_score(rec)


def test_selects_least_negative_retention():
    cands = {
        "l0.1": _rec("olora", 0.1, peak=_flat(0.80), final=_flat(0.60)),
        "l0.5": _rec("olora", 0.5, peak=_flat(0.80), final=_flat(0.72)),
        "l1.0": _rec("olora", 1.0, peak=_flat(0.80), final=_flat(0.65)),
    }
    out = select(cands)
    assert out["selected_lambda"] == 0.5
    assert out["selected_mean"] == pytest.approx(-0.08)


def test_ties_take_the_smallest_lambda():
    cands = {
        "a": _rec("olora", 1.0, peak=_flat(0.8), final=_flat(0.7)),
        "b": _rec("olora", 0.1, peak=_flat(0.8), final=_flat(0.7)),
    }
    assert select(cands)["selected_lambda"] == 0.1


def test_reports_when_no_lambda_beats_the_baseline():
    """协议 §A2.2：没有 λ 减少遗忘时如实记录，不扩网格、不换方法。"""
    cands = {
        "none": _rec("none", 0.0, peak=_flat(0.80), final=_flat(0.75)),
        "l0.5": _rec("olora", 0.5, peak=_flat(0.80), final=_flat(0.60)),
    }
    out = select(cands)
    assert out["beats_baseline"] is False
    assert "expected-to-fail" in out["note"]
    assert out["selected_lambda"] == 0.5


def test_baseline_is_compared_but_never_selected():
    cands = {
        "none": _rec("none", 0.0, peak=_flat(0.8), final=_flat(0.79)),
        "l0.5": _rec("olora", 0.5, peak=_flat(0.8), final=_flat(0.70)),
    }
    out = select(cands)
    assert out["selected_label"] == "l0.5"
    assert out["baseline_mean"] == pytest.approx(-0.01)


def test_requires_at_least_one_olora_arm():
    with pytest.raises(ValueError, match="no olora arms"):
        select({"none": _rec("none", 0.0, peak=_flat(0.5), final=_flat(0.5))})


def test_gate_reads_update_split_only():
    from experiments.phase2l_additivity import lambda_gate

    body = inspect.getsource(lambda_gate.retention_score)
    assert "R_update_raw" in body
    for forbidden in ("R_orc", "R_shr", "audit", "test.json"):
        assert forbidden not in body, forbidden


def test_run_qoc_records_update_at_peak_and_final():
    """两个时点都要取，否则可塑性抵消不掉。"""
    from experiments.phase2k_qoc import run_qoc

    source = inspect.getsource(run_qoc.main)
    assert '"--gate-update-eval", action="store_true"' in source
    assert "position == last_position or old.name == task.name" in source
