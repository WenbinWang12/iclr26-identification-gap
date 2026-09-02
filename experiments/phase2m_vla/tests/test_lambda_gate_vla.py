"""Phase-2M λ_a 闸门的测试。

重点不是"函数会不会跑"，而是三件会影响结论可信度的事：
1. 打分函数和 Phase-2L 用的是**同一个对象**（不是抄的副本）；
2. λ 从 `vla_lambda` 读，不是从 `olora_lambda`（读错会让所有 arm 的 λ 都变 0）；
3. endpoint 判定会如实报告，因为协议 §4 禁止为此扩栅格。
"""

from __future__ import annotations

import pytest

from experiments.phase2l_additivity import lambda_gate as gate_2l
from experiments.phase2m_vla import lambda_gate_vla as gate_2m


def _record(method, lam, peaks, finals, *, lam_key=None):
    """造一个最小 record：5 个早期任务，每个在自己的 stage 上有 peak，最后一个
    stage 上有 final。stage 数 = 6，和真实闸门跑一致。"""
    names = sorted(peaks)
    args = {"cl_method": method}
    if lam_key is not None:
        args[lam_key] = lam
    stages = []
    for n in names:
        stages.append({"trained_task": n,
                       "tasks": {n: {"scorable": True, "R_update_raw": peaks[n]}}})
    stages.append({"trained_task": "sixth",
                   "tasks": {n: {"scorable": True, "R_update_raw": finals[n]}
                             for n in names}})
    return {"args": args, "stages": stages}


def _arms():
    peaks = {"t%d" % i: 0.80 for i in range(5)}
    return {
        "none": _record("none", 0.0, peaks, {"t%d" % i: 0.70 for i in range(5)}),
        # a05 保持最好：final 掉得最少
        "a05": _record("vla", 0.5, peaks, {"t%d" % i: 0.76 for i in range(5)},
                       lam_key="vla_lambda"),
        "a20": _record("vla", 2.0, peaks, {"t%d" % i: 0.74 for i in range(5)},
                       lam_key="vla_lambda"),
        "a80": _record("vla", 8.0, peaks, {"t%d" % i: 0.66 for i in range(5)},
                       lam_key="vla_lambda"),
    }


def test_scoring_function_is_the_same_object_as_phase2l():
    """复用而非复制：两期必须共享一把尺子，否则"同一个 gate"的说法不成立。"""
    assert gate_2m.retention_score is gate_2l.retention_score


def test_selects_the_arm_with_the_least_drop_from_its_own_peak():
    out = gate_2m.select(_arms())
    assert out["selected_label"] == "a05"
    assert out["selected_lambda"] == 0.5
    assert out["selected_mean"] == pytest.approx(0.76 - 0.80)
    assert out["baseline_mean"] == pytest.approx(0.70 - 0.80)
    assert out["beats_baseline"] is True


def test_lambda_is_read_from_vla_lambda_not_olora_lambda():
    """若读错键，三个 vla arm 的 λ 都会塌成 0.0，栅格就消失了。"""
    out = gate_2m.select(_arms())
    assert sorted(v["lambda"] for v in out["table"].values()
                  if v["cl_method"] == "vla") == [0.5, 2.0, 8.0]


def test_phase2l_gate_would_have_rejected_these_arms():
    """记录我们为什么新写一个文件而不是复用 Phase-2L 的 select()。"""
    with pytest.raises(ValueError, match="no olora arms"):
        gate_2l.select(_arms())


def test_smallest_positive_lambda_is_not_an_open_endpoint():
    """λ_a=0 就是 `none` arm（等价性已验证），所以栅格下界是封闭的。
    选中最小的正 λ 不构成"选择落在未探索的边界上"这个隐忧。"""
    out = gate_2m.select(_arms())
    assert out["selected_lambda"] == 0.5
    assert out["selection_at_grid_endpoint"] is False
    assert out["endpoint_still_improving"] is False


def test_upper_edge_is_the_only_open_endpoint():
    peaks = {"t%d" % i: 0.80 for i in range(5)}
    arms = {
        "a05": _record("vla", 0.5, peaks, {"t%d" % i: 0.70 for i in range(5)},
                       lam_key="vla_lambda"),
        "a80": _record("vla", 8.0, peaks, {"t%d" % i: 0.76 for i in range(5)},
                       lam_key="vla_lambda"),
    }
    out = gate_2m.select(arms)
    assert out["selected_lambda"] == 8.0
    assert out["selection_at_grid_endpoint"] is True


def test_endpoint_still_improving_is_reported():
    """选择落在边界且仍单调向外——协议要求如实记录，不准扩栅格。"""
    peaks = {"t%d" % i: 0.80 for i in range(5)}
    arms = {
        "none": _record("none", 0.0, peaks, {"t%d" % i: 0.70 for i in range(5)}),
        "a05": _record("vla", 0.5, peaks, {"t%d" % i: 0.72 for i in range(5)},
                       lam_key="vla_lambda"),
        "a20": _record("vla", 2.0, peaks, {"t%d" % i: 0.74 for i in range(5)},
                       lam_key="vla_lambda"),
        "a80": _record("vla", 8.0, peaks, {"t%d" % i: 0.78 for i in range(5)},
                       lam_key="vla_lambda"),
    }
    out = gate_2m.select(arms)
    assert out["selected_lambda"] == 8.0
    assert out["selection_at_grid_endpoint"] is True
    assert out["endpoint_still_improving"] is True


def test_ties_break_toward_the_smaller_lambda():
    peaks = {"t%d" % i: 0.80 for i in range(5)}
    arms = {
        "a05": _record("vla", 0.5, peaks, {"t%d" % i: 0.75 for i in range(5)},
                       lam_key="vla_lambda"),
        "a20": _record("vla", 2.0, peaks, {"t%d" % i: 0.75 for i in range(5)},
                       lam_key="vla_lambda"),
    }
    assert gate_2m.select(arms)["selected_lambda"] == 0.5


def test_requires_at_least_one_arm_of_the_tuned_method():
    peaks = {"t%d" % i: 0.80 for i in range(5)}
    only_none = {"none": _record("none", 0.0, peaks,
                                 {"t%d" % i: 0.70 for i in range(5)})}
    with pytest.raises(ValueError, match="no vla arms"):
        gate_2m.select(only_none)


def test_mixing_two_cl_methods_in_one_grid_is_refused():
    """olora 和 vla 是两种干预，混在一个栅格里选出来的 λ 没有意义。"""
    arms = _arms()
    arms["olora"] = _record("olora", 1.0, {"t%d" % i: 0.80 for i in range(5)},
                            {"t%d" % i: 0.79 for i in range(5)},
                            lam_key="olora_lambda")
    with pytest.raises(ValueError, match="mixing methods"):
        gate_2m.select(arms)


def test_unknown_method_is_refused_before_scoring():
    with pytest.raises(ValueError, match="register its lambda key"):
        gate_2m.select(_arms(), method="psr")


def test_missing_update_eval_points_at_the_missing_flag():
    """没跑 --gate-update-eval 时，报错必须指向那个 flag，而不是让人去猜。"""
    rec = _record("vla", 0.5, {"t0": 0.8}, {"t0": 0.7}, lam_key="vla_lambda")
    for stage in rec["stages"]:
        for entry in stage["tasks"].values():
            entry.pop("R_update_raw")
    with pytest.raises(KeyError, match="gate-update-eval"):
        gate_2m.select({"a05": rec})
