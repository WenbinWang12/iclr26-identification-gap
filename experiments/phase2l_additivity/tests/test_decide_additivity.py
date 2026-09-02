"""判决脚本对 §3 判据 A-D 的检查。

重点是 §3 那条「C 失败则 A、B 不可解释」的从属关系：它必须由代码强制，
而不是留给读表的人自觉。§A2 犯过的错是把 A 的 PASS 单独拿出来说话，
协议靠一句话拦住；这里靠一个断言拦住。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments" / "phase2l_additivity" / "decide_additivity.py"
SRC = SCRIPT.read_text(encoding="utf-8")


def _task(raw, scoped, orc, book, scope_size=2, scorable=True):
    return {"R_raw": raw, "R_shr_scoped": scoped, "R_shr_global": raw, "R_orc": orc,
            "scope_size": scope_size, "scorable": scorable,
            "qoc": {"4": {"prototype": {"accuracy": book}}}}


def _run(tmp_path: Path, name: str, stages: list[dict], lam=0.5, method="olora") -> Path:
    payload = {"args": {"olora_lambda": lam, "cl_method": method}, "stages": stages}
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _stage(pos, trained, tasks):
    return {"position": pos, "trained_task": trained, "tasks": tasks, "n_steps": 10,
            "scope_radii": {}}


def _decide(tmp_path, base_stages, none_stages, tag="x"):
    b = _run(tmp_path, "base_%s.json" % tag, base_stages)
    n = _run(tmp_path, "none_%s.json" % tag, none_stages, lam=0.0, method="none")
    out = tmp_path / ("decide_%s.json" % tag)
    proc = subprocess.run([sys.executable, str(SCRIPT), "--base", str(b), "--none", str(n),
                           "--m", "4", "--out", str(out)],
                          cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(out.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 定义来源

def test_constants_are_the_frozen_ones():
    """§3：C 的分层 8pp，B 的 Phase-2K 参考 0.0630，门槛 0.60。"""
    assert "DEEP_STRATUM_PP = 0.08" in SRC
    assert "PHASE2K_SHALLOW_HEAD = 0.0630" in SRC
    assert "GATE_MEDIAN_RAW = 0.60" in SRC


def test_A_uses_the_median_not_the_mean():
    """§3 的 A 是「Median shallow_head」。"""
    block = SRC.split('out["A"] = {')[0].split("a_pt, a_lo, a_hi")[1]
    assert "statistic=np.median" in block


def test_shallow_head_takes_the_max_of_scoped_and_book():
    assert "np.nanmax([scoped, book_acc])" in SRC


# ------------------------------------------------------------------ 从属关系

def test_C_failing_forces_outcome_3_even_when_A_passes(tmp_path):
    """§3：C 失败 ⇒ A、B 不可解释 ⇒ 结局 3。这是 §A2 踩过的那一脚。"""
    # BASE 与 none 的 R_raw 完全相同 ⇒ deep_repair ≡ 0 ⇒ C 必失败。
    # 同时 shallow_head 恒为 +0.10 ⇒ A 本身会「通过」。
    base, none = [], []
    for pos, trained in enumerate(["T0", "T1", "T2", "T3"], start=1):
        tasks = {"A": _task(0.60, 0.70, 0.72, 0.65), "B": _task(0.55, 0.65, 0.67, 0.60)}
        tasks[trained] = _task(0.90, 0.90, 0.90, 0.90)
        base.append(_stage(pos, trained, dict(tasks)))
        none.append(_stage(pos, trained, dict(tasks)))
    got = _decide(tmp_path, base, none, "c_fail")
    assert got["A"]["passed"] is True          # A 自身为真
    assert got["C"]["passed"] is False
    assert got["outcome"] == 3                 # 但结局必须是 3
    assert "uninterpretable" in got["verdict"]


def _stream(levels: dict[str, tuple[float, float, float, float]], n_tasks: int = 3,
            peak: float = 0.90) -> list[dict]:
    """造一条真实形状的流：每个任务先在自己那一阶段被训练（记下峰值 `peak`），
    之后每一阶段作为旧任务被重新评估。

    C 的分层需要「峰值 − 当前」> 8pp，而峰值只在任务被训练的那一阶段记录，
    所以任何不训练该任务的 fixture 都会把它排除在 C 之外 —— 这正是本文件
    最初两个失败的原因，也是真实数据里不会发生的情形。
    """
    names = sorted(levels)[:n_tasks]
    stages = []
    for pos, trained in enumerate(names, start=1):
        tasks = {trained: _task(peak, peak, peak, peak)}
        for name in names[:pos - 1]:               # 已见的旧任务
            tasks[name] = _task(*levels[name])
        stages.append(_stage(pos, trained, tasks))
    # 再加一个尾阶段，让所有任务都作为旧任务出现过
    tail = {name: _task(*levels[name]) for name in names}
    tail["Z"] = _task(peak, peak, peak, peak)
    stages.append(_stage(len(names) + 1, "Z", tail))
    return stages


def test_outcome_1_needs_C_and_A_and_B(tmp_path):
    """C 真的修好了深层，且 shallow_head 仍在 ⇒ 结局 1（可加性）。"""
    # none: 峰值 0.90 → 0.60，遗忘 30pp，进 C 的分层。
    none = _stream({"A": (0.60, 0.70, 0.72, 0.65), "B": (0.58, 0.68, 0.70, 0.63),
                    "C": (0.62, 0.72, 0.74, 0.67)})
    # base: 正则把每个修回 +10pp，臂内 shallow_head 仍是 +10pp。
    base = _stream({"A": (0.70, 0.80, 0.82, 0.75), "B": (0.68, 0.78, 0.80, 0.73),
                    "C": (0.72, 0.82, 0.84, 0.77)})
    got = _decide(tmp_path, base, none, "o1")
    assert got["C"]["passed"] is True and got["C"]["median"] == pytest.approx(0.10, abs=1e-9)
    assert got["A"]["passed"] is True
    assert got["B"]["passed"] is True
    assert got["outcome"] == 1


def test_B_detects_cannibalisation(tmp_path):
    """BASE 把 shallow_head 吃掉 ⇒ B 的 CI 整体 < 0 ⇒ 不是结局 1。"""
    none = _stream({"A": (0.60, 0.75, 0.76, 0.70), "B": (0.58, 0.73, 0.74, 0.68),
                    "C": (0.62, 0.77, 0.78, 0.72)})
    base = _stream({"A": (0.70, 0.705, 0.71, 0.70), "B": (0.68, 0.685, 0.69, 0.68),
                    "C": (0.72, 0.725, 0.73, 0.72)})
    got = _decide(tmp_path, base, none, "cann")
    assert got["C"]["passed"] is True
    assert got["B"]["cannibalised"] is True
    assert got["outcome"] != 1


def test_c_stratum_requires_a_recorded_peak(tmp_path):
    """从未被训练过的任务没有峰值，因此不进 C 的分层 —— 记录这个边界行为，
    因为它是本文件两个测试最初失败的原因，不是脚本的缺陷。"""
    base, none = [], []
    for pos, trained in enumerate(["T0", "T1"], start=1):
        t = {"Never": _task(0.60, 0.70, 0.72, 0.65)}
        t[trained] = _task(0.90, 0.90, 0.90, 0.90)
        base.append(_stage(pos, trained, dict(t))); none.append(_stage(pos, trained, dict(t)))
    got = _decide(tmp_path, base, none, "nopeak")
    assert got["C"]["n"] == 0
    assert got["outcome"] == 3


def test_D_flags_a_prop1_violation_on_singletons(tmp_path):
    """单例 scope 上 R_orc 必须等于 R_shr_scoped，不等就是接线坏了。"""
    base, none = [], []
    for pos, trained in enumerate(["T0", "T1"], start=1):
        t = {"S": _task(0.60, 0.70, 0.77, 0.65, scope_size=1)}   # 差 7pp ⇒ 违反
        t[trained] = _task(0.90, 0.90, 0.90, 0.90)
        base.append(_stage(pos, trained, dict(t))); none.append(_stage(pos, trained, dict(t)))
    got = _decide(tmp_path, base, none, "d")
    assert got["D"]["passed"] is False
    assert got["D"]["violations"] >= 1
    assert got["D"]["worst_abs_gap"] == pytest.approx(0.07, abs=1e-9)


def test_D_passes_when_singletons_are_exact(tmp_path):
    base, none = [], []
    for pos, trained in enumerate(["T0", "T1"], start=1):
        t = {"S": _task(0.60, 0.70, 0.70, 0.65, scope_size=1)}
        t[trained] = _task(0.90, 0.90, 0.90, 0.90)
        base.append(_stage(pos, trained, dict(t))); none.append(_stage(pos, trained, dict(t)))
    got = _decide(tmp_path, base, none, "dok")
    assert got["D"]["passed"] is True and got["D"]["violations"] == 0


def test_trained_task_is_excluded_from_observations(tmp_path):
    """判据问的是**已见旧任务**上的遗忘，当前训练任务不进观测。"""
    base, none = [], []
    for pos, trained in enumerate(["A", "B"], start=1):
        t = {"A": _task(0.60, 0.70, 0.72, 0.65), "B": _task(0.55, 0.65, 0.67, 0.60)}
        base.append(_stage(pos, trained, dict(t))); none.append(_stage(pos, trained, dict(t)))
    got = _decide(tmp_path, base, none, "excl")
    # 2 stages x 2 tasks = 4 格，减去 2 个当前训练任务 = 2
    assert got["A"]["n"] == 2


def test_non_scorable_dropped_and_gate_uses_final_stage(tmp_path):
    base, none = [], []
    for pos, trained in enumerate(["T0", "T1"], start=1):
        t = {"A": _task(0.62, 0.70, 0.72, 0.65),
             "CB": _task(0.05, 0.05, 0.05, 0.05, scorable=False)}
        t[trained] = _task(0.90, 0.90, 0.90, 0.90)
        base.append(_stage(pos, trained, dict(t))); none.append(_stage(pos, trained, dict(t)))
    got = _decide(tmp_path, base, none, "gate")
    assert got["training_gate"]["n_obs"] == 2           # A 与 T1，CB 被丢
    assert got["training_gate"]["median_R_raw"] == pytest.approx(0.76, abs=1e-9)
    assert got["training_gate"]["passed"] is True
