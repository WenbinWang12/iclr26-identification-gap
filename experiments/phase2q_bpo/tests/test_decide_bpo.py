"""判决脚本的测试。合成记录喂进去,检查每条判据翻译得对。

重点在**能失败**:每条判据都各有一个构造好的失败样例。Phase-2M 的教训是判决
脚本自己会有 bug(那次是 `R_shr_scoped` 当成对手而不是天花板),所以这里把
「天花板不参与 Q1」也钉成一条测试。
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def _task(name, *, raw=0.70, glob=0.71, bpo=0.75, scoped=0.755, orc=None,
          scope_size=1, imbal=None, by_batch=None, scope_recoverable=0.02):
    # 默认 orc == scoped:这些 fixture 默认是**单任务域**,而 Prop 1 要求单任务域上
    # R_orc == R_shr_scoped。默认 scope_recoverable 故意**非零**,这样任何把 Q6 读回
    # `scope_recoverable` 的回归都会立刻让 Q6 假失败。见 §A1.4。
    if orc is None:
        orc = scoped
    block = {
        "scorable": True, "scope_size": scope_size,
        "scope_recoverable": scope_recoverable,
        "R_raw": raw, "R_shr_global": glob, "R_shr_scoped": scoped, "R_orc": orc,
        "bpo": {
            "R_bpo": bpo,
            "offset": [0.1, -0.1], "hist": [0.5, 0.5], "hist_no_offset": [1.0, 0.0],
            "R_bpo_by_batch": by_batch or {"16": bpo - 0.01, "32": bpo, "64": bpo},
            "imbalanced": imbal if imbal is not None else {
                r: {"R_bpo": bpo - 0.005, "R_raw": raw, "R_shr_global": glob,
                    "n": 100, "class0_share": float(r)}
                for r in ("0.5", "0.7", "0.9")
            },
        },
    }
    return name, block


def _write(tmp_path, tasks, *, seed=1):
    payload = {"stages": [{"position": 15, "trained_task": "Yahoo",
                           "tasks": dict(tasks)}]}
    p = tmp_path / ("qoc_seed%d.json" % seed)
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _run(tmp_path, paths):
    out = tmp_path / "decide.json"
    r = subprocess.run(
        [sys.executable, "-m", "experiments.phase2q_bpo.decide_bpo",
         "--runs", *[str(p) for p in paths], "--out", str(out)],
        cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(out.read_text(encoding="utf-8"))


def _three_seeds(tmp_path, tasks):
    return [_write(tmp_path / str(s), tasks, seed=s) for s in (1, 2, 3)
            if (tmp_path / str(s)).mkdir(parents=True, exist_ok=True) or True]


# ------------------------------------------------------------------ happy path

def test_all_criteria_pass_on_a_clean_positive(tmp_path):
    tasks = [_task("A"), _task("B"), _task("C"), _task("D")]
    d = _run(tmp_path, _three_seeds(tmp_path, tasks))
    assert d["gate"]["pass"]
    assert d["Q1"]["pass"] and d["Q1"]["delta_pp"] > 0
    assert d["Q2"]["pass"]
    assert d["Q3"]["pass"]
    assert d["Q4"]["pass"]
    assert d["Q6"]["pass"]
    assert d["outcome_fired"] == 1


# ------------------------------------------------------- each criterion can fail

def test_q1_fails_when_bpo_below_global(tmp_path):
    tasks = [_task(n, bpo=0.68, glob=0.71) for n in "ABCD"]
    d = _run(tmp_path, _three_seeds(tmp_path, tasks))
    assert not d["Q1"]["pass"]
    assert d["outcome_fired"] == 4


def test_q2_fails_on_one_destroyed_task(tmp_path):
    """均值上升但单任务被毁 —— 正是 Phase-2P 的 SSO 的形状。"""
    # 8 个好任务 + 1 个被毁的。任务数要够,否则簇 bootstrap 的区间宽到跨 0,
    # Q1 会因为样例太小而失败,测不到本意 —— 第一版就是那样。
    tasks = [_task(n, bpo=0.80) for n in "ABCDEFGH"]
    tasks.append(_task("Z", raw=0.70, bpo=0.60))      # −10 pp
    d = _run(tmp_path, _three_seeds(tmp_path, tasks))
    assert d["Q1"]["pass"]
    assert not d["Q2"]["pass"]
    assert d["Q2"]["worst_task"] == "Z"
    assert d["Q2"]["worst_delta_pp"] == pytest.approx(-10.0, abs=1e-6)
    assert d["outcome_fired"] == 5


def test_q3_fails_when_ceiling_gap_exceeds_one_pp(tmp_path):
    tasks = [_task(n, bpo=0.72, scoped=0.76, glob=0.71) for n in "ABCD"]
    d = _run(tmp_path, _three_seeds(tmp_path, tasks))
    assert d["Q1"]["pass"] and d["Q2"]["pass"]
    assert not d["Q3"]["pass"]
    assert d["Q3"]["delta_pp"] > 1.0
    assert d["outcome_fired"] == 2


def test_q4_fails_when_the_gain_vanishes_on_imbalanced_mixes(tmp_path):
    """协议 §4 结局 3:Q1 过而 Q4 塌 —— 增益是平衡 audit 的假象。"""
    imbal = {"0.5": {"R_bpo": 0.75, "R_raw": 0.70, "R_shr_global": 0.71,
                     "n": 100, "class0_share": 0.5},
             "0.7": {"R_bpo": 0.68, "R_raw": 0.70, "R_shr_global": 0.71,
                     "n": 90, "class0_share": 0.7},
             "0.9": {"R_bpo": 0.60, "R_raw": 0.70, "R_shr_global": 0.71,
                     "n": 70, "class0_share": 0.9}}
    tasks = [_task(n, imbal=imbal) for n in "ABCD"]
    d = _run(tmp_path, _three_seeds(tmp_path, tasks))
    assert d["Q1"]["pass"]
    assert not d["Q4"]["pass"]
    assert d["Q4"]["by_ratio"]["0.7"]["bpo_minus_global_pp"] < 0
    assert d["outcome_fired"] == 3


def test_q4_only_thresholds_the_seventy_thirty_ratio(tmp_path):
    """90:10 报告但不设阈:协议 §3 明说。"""
    imbal = {"0.5": {"R_bpo": 0.75, "R_raw": 0.70, "R_shr_global": 0.71,
                     "n": 100, "class0_share": 0.5},
             "0.7": {"R_bpo": 0.74, "R_raw": 0.70, "R_shr_global": 0.71,
                     "n": 90, "class0_share": 0.7},
             "0.9": {"R_bpo": 0.50, "R_raw": 0.70, "R_shr_global": 0.71,
                     "n": 70, "class0_share": 0.9}}
    tasks = [_task(n, imbal=imbal) for n in "ABCD"]
    d = _run(tmp_path, _three_seeds(tmp_path, tasks))
    assert d["Q4"]["pass"], "0.9 must not gate the verdict"
    assert d["Q4"]["by_ratio"]["0.9"]["bpo_minus_global_pp"] < 0


def test_q6_catches_a_prop1_violation(tmp_path):
    # 单任务域上 R_orc 必须等于 R_shr_scoped;掰开它就是一次真的 Prop 1 违反。
    name, block = _task("A", scoped=0.755, orc=0.77)
    tasks = [(name, block), _task("B"), _task("C")]
    d = _run(tmp_path, _three_seeds(tmp_path, tasks))
    assert not d["Q6"]["pass"]
    assert d["Q6"]["violations"]
    assert abs(d["Q6"]["violations"][0][1] - 0.015) < 1e-9


def test_q6_reads_delta_id_scoped_not_scope_recoverable(tmp_path):
    """回归钉:Q6 必须读 R_orc − R_shr_scoped,**不能**读 scope_recoverable。

    这是 §A1.4 记录的缺陷。原实现读 scope_recoverable(= R_shr_scoped − R_shr_global),
    在单任务域上没有理由为 0,于是 Q6 在真实记录上 13/15 假失败,而当时的 fixture 把
    scope_recoverable 钉成 0.0,所以测试跟着一起错。这里两个方向都钉住。
    """
    # 方向一:scope_recoverable 很大但 Prop 1 成立 —— Q6 必须 PASS。
    tasks = [_task(n, scope_recoverable=0.09) for n in "ABC"]
    d = _run(tmp_path / "ok", _three_seeds(tmp_path / "ok", tasks))
    assert d["Q6"]["pass"], "Q6 误读了 scope_recoverable"
    assert d["Q6"]["n_singleton_obs"] == 9      # 3 任务 × 3 seed

    # 方向二:scope_recoverable 恰为 0 但 Prop 1 被违反 —— Q6 必须 FAIL。
    name, block = _task("A", scoped=0.755, orc=0.78, scope_recoverable=0.0)
    tasks2 = [(name, block), _task("B"), _task("C")]
    d2 = _run(tmp_path / "bad", _three_seeds(tmp_path / "bad", tasks2))
    assert not d2["Q6"]["pass"], "Q6 漏掉了真的 Prop 1 违反"
    assert d2["Q6"]["field"] == "R_orc - R_shr_scoped"


def test_q6_ignores_non_singleton_scopes(tmp_path):
    # 多任务域上 R_orc − R_shr_scoped 本来就该非零(这正是 Δ_id_scoped 的意义),
    # Q6 只对单任务域断言。
    tasks = [_task(n, scope_size=3, scoped=0.755, orc=0.79) for n in "ABC"]
    d = _run(tmp_path, _three_seeds(tmp_path, tasks))
    assert d["Q6"]["pass"]
    assert d["Q6"]["n_singleton_obs"] == 0


def test_gate_fails_on_undertrained_run(tmp_path):
    tasks = [_task(n, raw=0.50, glob=0.51, bpo=0.55, scoped=0.555, orc=0.56)
             for n in "ABCD"]
    d = _run(tmp_path, _three_seeds(tmp_path, tasks))
    assert not d["gate"]["pass"]


# ------------------------------------------------ the ceiling is not a rival

def test_scoped_ceiling_never_enters_q1(tmp_path):
    """把天花板抬到 0.99,Q1 一个数字都不许动 —— Phase-2M 判决脚本的旧 bug。"""
    a = [_task(n) for n in "ABCD"]
    b = [_task(n, scoped=0.99) for n in "ABCD"]
    da = _run(tmp_path / "a", _three_seeds(tmp_path / "a", a))
    db = _run(tmp_path / "b", _three_seeds(tmp_path / "b", b))
    assert da["Q1"]["delta_pp"] == pytest.approx(db["Q1"]["delta_pp"])
    assert da["Q2"]["delta_pp"] == pytest.approx(db["Q2"]["delta_pp"])
    assert db["Q3"]["pass"] is False       # 天花板差变大只影响 Q3


def test_missing_bpo_records_is_a_hard_error(tmp_path):
    """没传 --score-bpo 就该报错,而不是静默给出一份没有 BPO 的判决。"""
    name, block = _task("A")
    del block["bpo"]
    p = _write(tmp_path, [(name, block)])
    r = subprocess.run(
        [sys.executable, "-m", "experiments.phase2q_bpo.decide_bpo",
         "--runs", str(p), "--out", str(tmp_path / "o.json")],
        cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode != 0
    assert "score-bpo" in (r.stderr + r.stdout)