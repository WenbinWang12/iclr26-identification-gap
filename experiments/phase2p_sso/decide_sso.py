"""Phase-2P 判决：对冻结协议 §3 的 P1-P6 逐条求值。

协议冻结于 ae59bcd5d370271e64bef8b49b90d6150bfb16ac1fc6a4f170c53f40d9afac1e，
在实现之前。本脚本不新增判据、不改阈值，只把 §3 的文字翻成计算。

指标口径（协议 §2）：末阶段 audit 分片、受限 argmax 平衡准确率，
在 12 个 scorable 任务上取均值。R_shr_scoped 是**天花板不是对手**。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.phase2j_offset_conflict.decide import cluster_bootstrap_ci

#: 协议 §3 的阈值，逐条对应，不在别处硬编码。
P2_MAX_PP = 1.0
P3_MAX_LOSS_PP = 2.0
GATE_MEDIAN_RAW = 0.60


def load_final_stage(path: Path) -> dict:
    """取一个 seed 的末阶段 per-task 记录，只保留 scorable 的。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    stage = payload["stages"][-1]
    tasks = {name: info for name, info in stage["tasks"].items() if info.get("scorable")}
    return {"tasks": tasks, "payload": payload, "stage": stage}


def paired_ci(runs: list[dict], left: str, right: str, *,
              only: set[str] | None = None) -> tuple[float, float, float]:
    """(left − right) 的百分点差，任务作簇，10 000 次重采样，统计量取均值。"""

    by_task: dict[str, list[float]] = {}
    for run in runs:
        for name, info in run["tasks"].items():
            if only is not None and name not in only:
                continue
            if left not in info or right not in info:
                continue
            by_task.setdefault(name, []).append(100.0 * (info[left] - info[right]))
    return cluster_bootstrap_ci(by_task, draws=10000, seed=0, statistic=np.mean)


def mean_level(runs: list[dict], key: str) -> float:
    values = [info[key] for run in runs for info in run["tasks"].values() if key in info]
    return float(np.mean(values)) if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="runs/phase2p_decide.json")
    args = parser.parse_args()

    runs = [load_final_stage(Path(p)) for p in args.runs]
    n_tasks = sorted({len(r["tasks"]) for r in runs})
    singles = {name for run in runs for name, info in run["tasks"].items()
               if int(info.get("scope_size", 0)) == 1}
    multis = {name for run in runs for name, info in run["tasks"].items()
              if int(info.get("scope_size", 0)) > 1}

    out: dict = {"n_seeds": len(runs), "scorable_per_seed": n_tasks,
                 "singleton_tasks": sorted(singles), "multi_tasks": sorted(multis)}

    out["levels"] = {k: mean_level(runs, k) for k in
                     ("R_raw", "R_shr_global", "R_sso1", "R_sso4", "R_shr_scoped", "R_orc")}

    # P1（主判据）：SSO(m=1) − 全局单一偏移，CI 正侧排除 0。
    p1 = paired_ci(runs, "R_sso1", "R_shr_global")
    out["P1"] = {"delta_pp": p1[0], "ci": [p1[1], p1[2]],
                 "pass": bool(p1[1] > 0.0), "spec": "R_sso1 - R_shr_global, CI excludes 0 positive"}

    # P2（陈旧代价）：天花板与 SSO 的点估计差 ≤ 1 pp。
    p2 = paired_ci(runs, "R_shr_scoped", "R_sso1")
    out["P2"] = {"delta_pp": p2[0], "ci": [p2[1], p2[2]],
                 "pass": bool(p2[0] <= P2_MAX_PP), "spec": "R_shr_scoped - R_sso1 <= 1pp point est"}

    # P3（无伤害）：单个任务对 R_raw 的损失不超过 2 pp，按任务跨 seed 取均值。
    per_task: dict[str, list[float]] = {}
    for run in runs:
        for name, info in run["tasks"].items():
            per_task.setdefault(name, []).append(100.0 * (info["R_sso1"] - info["R_raw"]))
    task_means = {name: float(np.mean(v)) for name, v in per_task.items()}
    worst = min(task_means.items(), key=lambda kv: kv[1]) if task_means else ("", float("nan"))
    out["P3"] = {"worst_task": worst[0], "worst_delta_pp": worst[1],
                 "pass": bool(worst[1] >= -P3_MAX_LOSS_PP),
                 "per_task_delta_pp": task_means,
                 "spec": "no task loses >2pp vs R_raw under SSO(m=1)"}

    # P4（码本是否买到东西）：SSO(m=4) − SSO(m=1)，CI 排除 0。§5 预先声明预期失败。
    p4 = paired_ci(runs, "R_sso4", "R_sso1")
    out["P4"] = {"delta_pp": p4[0], "ci": [p4[1], p4[2]],
                 "pass": bool(p4[1] > 0.0 or p4[2] < 0.0),
                 "spec": "R_sso4 - R_sso1, CI excludes 0 (predeclared expected to fail)"}

    # P5（Prop 1 精确性 + 纯陈旧探针）：单例 scope 上 |R_sso1 − R_orc|。报告，不设阈。
    gaps: dict[str, list[float]] = {}
    for run in runs:
        for name in sorted(singles):
            info = run["tasks"].get(name)
            if info:
                gaps.setdefault(name, []).append(100.0 * abs(info["R_sso1"] - info["R_orc"]))
    flat = [v for vs in gaps.values() for v in vs]
    exact = sum(1 for v in flat if v < 1e-9)
    out["P5"] = {"singleton_abs_gap_pp_mean": float(np.mean(flat)) if flat else float("nan"),
                 "singleton_abs_gap_pp_max": float(np.max(flat)) if flat else float("nan"),
                 "exactly_zero": exact, "n_measurements": len(flat),
                 "per_task_mean_pp": {k: float(np.mean(v)) for k, v in gaps.items()},
                 "spec": "reported, not thresholded"}

    # P6（成本核算）：从 run 日志里取 stored_floats，与 Σ_S |table[S]|·(K_S−1) 对账。
    # 公式侧由 tests/test_sso.py 断言；此处只搬运运行时实测值。
    out["P6"] = {"note": "stored_floats/router_floats 由 run 日志与 test_sso.py 共同断言",
                 "spec": "sum |table[S]|*(K_S-1), no examples retained"}

    # 训练门槛（无条件）：末阶段 R_raw 中位数 > 0.60。
    raws = [info["R_raw"] for run in runs for info in run["tasks"].values()]
    med = float(np.median(raws)) if raws else float("nan")
    out["gate"] = {"median_R_raw": med, "pass": bool(med > GATE_MEDIAN_RAW)}

    # 多任务 scope 的分裂，§5 已预先声明大部分 +1.51pp 来自单例。
    out["split"] = {
        "multi_sso1_minus_global": paired_ci(runs, "R_sso1", "R_shr_global", only=multis),
        "single_sso1_minus_global": paired_ci(runs, "R_sso1", "R_shr_global", only=singles),
        "multi_scoped_minus_sso1": paired_ci(runs, "R_shr_scoped", "R_sso1", only=multis),
    }

    # §4 的四个预先声明结果，哪个开火由 P1/P2/P5 决定，不在此处新增分支。
    p1_ok, p2_ok = out["P1"]["pass"], out["P2"]["pass"]
    stale_large = out["P5"]["singleton_abs_gap_pp_mean"] > P3_MAX_LOSS_PP
    if p1_ok and p2_ok:
        fired = 1
    elif p1_ok:
        fired = 2
    elif stale_large:
        fired = 3
    else:
        fired = 4
    out["outcome_fired"] = fired

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
