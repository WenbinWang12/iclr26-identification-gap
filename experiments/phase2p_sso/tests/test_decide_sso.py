"""判决脚本的检查：阈值必须逐字来自冻结协议 §3，结局分支必须只由 P1/P2/P5 决定。

这些测试用合成的 run JSON，因为要检的是**判据逻辑**，不是模型。Phase-2M 的
教训是判决脚本里一个反向的不等号能把 FAIL 读成 PASS，而那时没有任何测试拦得住。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments" / "phase2p_sso" / "decide_sso.py"
SRC = SCRIPT.read_text(encoding="utf-8")


def _run_json(tmp_path: Path, name: str, per_task: dict[str, dict]) -> Path:
    """造一个只有末阶段的 run JSON。"""
    payload = {"stages": [{"trained_task": "last", "tasks": per_task}]}
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _task(raw, glob, sso1, sso4, scoped, orc, scope_size=2, scorable=True):
    return {"R_raw": raw, "R_shr_global": glob, "R_sso1": sso1, "R_sso4": sso4,
            "R_shr_scoped": scoped, "R_orc": orc,
            "scope_size": scope_size, "scorable": scorable}


def _decide(tmp_path: Path, seeds: list[dict]) -> dict:
    paths = [str(_run_json(tmp_path, "s%d.json" % i, t)) for i, t in enumerate(seeds)]
    out = tmp_path / "decide.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--runs", *paths, "--out", str(out)],
        cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(out.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 阈值来源

def test_thresholds_are_the_frozen_ones():
    """§3：P2 ≤ 1pp，P3 > 2pp 为失败，门槛中位数 0.60。"""
    assert "P2_MAX_PP = 1.0" in SRC
    assert "P3_MAX_LOSS_PP = 2.0" in SRC
    assert "GATE_MEDIAN_RAW = 0.60" in SRC


def test_bootstrap_settings_match_the_protocol():
    """§3：任务作簇、10 000 draws、统计量取均值。"""
    assert "draws=10000" in SRC and "statistic=np.mean" in SRC


def test_p5_is_not_thresholded():
    """§3 明写 P5「Reported, not thresholded」，所以它没有 pass 字段。"""
    block = SRC.split('out["P5"] = {')[1].split("}")[0]
    assert '"pass"' not in block


# ------------------------------------------------------------------ 判据逻辑

def test_p1_passes_only_when_the_ci_is_strictly_positive(tmp_path):
    """SSO 一致高出 4pp：P1 必须 PASS。"""
    seeds = []
    for shift in (0.0, 0.01, -0.01):
        seeds.append({
            "A": _task(0.60, 0.62, 0.66 + shift, 0.66, 0.67, 0.67),
            "B": _task(0.55, 0.58, 0.62 + shift, 0.62, 0.63, 0.63),
            "C": _task(0.70, 0.71, 0.75 + shift, 0.75, 0.76, 0.76),
        })
    got = _decide(tmp_path, seeds)
    assert got["P1"]["pass"] is True
    assert got["P1"]["delta_pp"] == pytest.approx(4.0, abs=0.7)
    assert got["P1"]["ci"][0] > 0.0


def test_p1_fails_when_tasks_disagree_in_sign(tmp_path):
    """两升两降：点估计可能为正，但 CI 跨 0，P1 必须 FAIL。"""
    seeds = [{
        "A": _task(0.60, 0.60, 0.75, 0.75, 0.75, 0.75),
        "B": _task(0.60, 0.60, 0.45, 0.45, 0.45, 0.45),
        "C": _task(0.60, 0.60, 0.72, 0.72, 0.72, 0.72),
        "D": _task(0.60, 0.60, 0.48, 0.48, 0.48, 0.48),
    } for _ in range(3)]
    got = _decide(tmp_path, seeds)
    assert got["P1"]["pass"] is False
    assert got["P1"]["ci"][0] < 0.0 < got["P1"]["ci"][1]


def test_p2_uses_the_point_estimate_not_the_ci(tmp_path):
    """§3 的 P2 是「≤ 1pp at the point estimate」，不是区间判据。"""
    seeds = [{
        "A": _task(0.60, 0.60, 0.66, 0.66, 0.665, 0.67),
        "B": _task(0.55, 0.55, 0.62, 0.62, 0.625, 0.63),
    } for _ in range(3)]
    got = _decide(tmp_path, seeds)
    assert got["P2"]["delta_pp"] == pytest.approx(0.5, abs=1e-6)
    assert got["P2"]["pass"] is True


def test_p2_fails_when_rehearsal_buys_more_than_a_point(tmp_path):
    seeds = [{"A": _task(0.60, 0.60, 0.62, 0.62, 0.66, 0.66)} for _ in range(3)]
    got = _decide(tmp_path, seeds)
    assert got["P2"]["delta_pp"] == pytest.approx(4.0, abs=1e-6)
    assert got["P2"]["pass"] is False


def test_p3_reports_the_worst_task_and_fails_past_two_points(tmp_path):
    seeds = [{
        "A": _task(0.60, 0.60, 0.66, 0.66, 0.66, 0.66),
        "B": _task(0.70, 0.70, 0.66, 0.66, 0.66, 0.66),   # −4pp vs raw
    } for _ in range(3)]
    got = _decide(tmp_path, seeds)
    assert got["P3"]["worst_task"] == "B"
    assert got["P3"]["worst_delta_pp"] == pytest.approx(-4.0, abs=1e-6)
    assert got["P3"]["pass"] is False


def test_p3_passes_when_the_worst_loss_is_within_tolerance(tmp_path):
    seeds = [{
        "A": _task(0.60, 0.60, 0.66, 0.66, 0.66, 0.66),
        "B": _task(0.70, 0.70, 0.69, 0.69, 0.69, 0.69),   # −1pp
    } for _ in range(3)]
    assert _decide(tmp_path, seeds)["P3"]["pass"] is True


def test_p5_measures_singleton_staleness_and_counts_exact_zeros(tmp_path):
    """单例 scope 上 SSO(m=1) 就是那个（陈旧的）per-task 偏移。"""
    seeds = [{
        "S1": _task(0.60, 0.60, 0.66, 0.66, 0.66, 0.66, scope_size=1),  # gap 0
        "S2": _task(0.60, 0.60, 0.60, 0.60, 0.66, 0.66, scope_size=1),  # gap 6pp
        "M1": _task(0.60, 0.60, 0.63, 0.63, 0.64, 0.70, scope_size=2),  # 不计入
    } for _ in range(3)]
    got = _decide(tmp_path, seeds)
    assert got["singleton_tasks"] == ["S1", "S2"]
    assert got["P5"]["n_measurements"] == 6
    assert got["P5"]["exactly_zero"] == 3
    assert got["P5"]["singleton_abs_gap_pp_mean"] == pytest.approx(3.0, abs=1e-6)
    assert got["P5"]["singleton_abs_gap_pp_max"] == pytest.approx(6.0, abs=1e-6)


def test_gate_uses_the_median_of_raw(tmp_path):
    seeds = [{
        "A": _task(0.50, 0.60, 0.62, 0.62, 0.62, 0.62),
        "B": _task(0.52, 0.60, 0.62, 0.62, 0.62, 0.62),
        "C": _task(0.90, 0.60, 0.62, 0.62, 0.62, 0.62),
    } for _ in range(3)]
    got = _decide(tmp_path, seeds)
    assert got["gate"]["median_R_raw"] == pytest.approx(0.52, abs=1e-9)
    assert got["gate"]["pass"] is False


def test_non_scorable_tasks_are_dropped(tmp_path):
    seeds = [{
        "A": _task(0.60, 0.60, 0.66, 0.66, 0.66, 0.66),
        "CB": _task(0.10, 0.10, 0.10, 0.10, 0.10, 0.10, scorable=False),
    } for _ in range(3)]
    got = _decide(tmp_path, seeds)
    assert got["scorable_per_seed"] == [1]
    assert "CB" not in got["P3"]["per_task_delta_pp"]


# ------------------------------------------------------------------ 结局分支

def test_outcome_1_fires_when_p1_and_p2_pass(tmp_path):
    seeds = [{
        "A": _task(0.60, 0.60, 0.66, 0.66, 0.665, 0.67),
        "B": _task(0.55, 0.55, 0.61, 0.61, 0.615, 0.62),
        "C": _task(0.70, 0.70, 0.76, 0.76, 0.765, 0.77),
    } for _ in range(3)]
    got = _decide(tmp_path, seeds)
    assert (got["P1"]["pass"], got["P2"]["pass"]) == (True, True)
    assert got["outcome_fired"] == 1


def test_outcome_2_fires_when_p1_passes_but_rehearsal_buys_a_lot(tmp_path):
    seeds = [{
        "A": _task(0.60, 0.60, 0.66, 0.66, 0.72, 0.72),
        "B": _task(0.55, 0.55, 0.61, 0.61, 0.67, 0.67),
        "C": _task(0.70, 0.70, 0.76, 0.76, 0.82, 0.82),
    } for _ in range(3)]
    got = _decide(tmp_path, seeds)
    assert got["P1"]["pass"] is True and got["P2"]["pass"] is False
    assert got["outcome_fired"] == 2


def test_outcome_3_fires_when_p1_fails_with_large_singleton_staleness(tmp_path):
    seeds = [{
        "S1": _task(0.60, 0.66, 0.58, 0.58, 0.66, 0.68, scope_size=1),
        "S2": _task(0.60, 0.66, 0.57, 0.57, 0.66, 0.67, scope_size=1),
    } for _ in range(3)]
    got = _decide(tmp_path, seeds)
    assert got["P1"]["pass"] is False
    assert got["P5"]["singleton_abs_gap_pp_mean"] > 2.0
    assert got["outcome_fired"] == 3


def test_outcome_4_fires_when_p1_fails_with_small_singleton_staleness(tmp_path):
    seeds = [{
        "S1": _task(0.60, 0.66, 0.655, 0.655, 0.66, 0.66, scope_size=1),
        "S2": _task(0.60, 0.66, 0.657, 0.657, 0.66, 0.66, scope_size=1),
    } for _ in range(3)]
    got = _decide(tmp_path, seeds)
    assert got["P1"]["pass"] is False
    assert got["P5"]["singleton_abs_gap_pp_mean"] < 2.0
    assert got["outcome_fired"] == 4


def test_outcome_branch_reads_only_p1_p2_p5(tmp_path):
    """结局分支不许偷偷把 P3/P4 也算进去——§4 只由 P1/P2/P5 定义。"""
    block = SRC.split("# §4 的四个预先声明结果")[1]
    assert 'out["P3"]["pass"]' not in block and 'out["P4"]["pass"]' not in block
