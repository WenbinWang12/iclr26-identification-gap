"""Phase-2Q 判决：对冻结协议 §3 的 Q1-Q6 逐条求值。

协议冻结于 52b85c5c0464e1ad21982bde48b4c698689aa9d5883f1e620e37839909da73a6,
在实现之前。本脚本不新增判据、不改阈值,只把 §3 的文字翻成计算。

指标口径(协议 §2):末阶段 audit 分片、受限 argmax 平衡准确率,在 12 个 scorable
任务上取均值。`R_shr_scoped` 是**天花板不是对手**;`R_orc` 只作上界。

Q4 的算法与别处不同,值得说清:不平衡混合上的 `R_bpo − R_shr_global` 要在
**该子集**上比,因为查询混合就是它。所以 Q4 不复用 `paired_ci` 的键查找,而是
从 `bpo.imbalanced[ratio]` 里取三个同片数字。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.phase2j_offset_conflict.decide import cluster_bootstrap_ci

#: 协议 §3 的阈值,逐条对应,不在别处硬编码。
Q2_MAX_LOSS_PP = 2.0
Q3_MAX_PP = 1.0
Q4_THRESHOLD_RATIO = "0.7"      # 只有 70:30 设阈值;90:10 报告不设阈
GATE_MEDIAN_RAW = 0.60


def load_final_stage(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stage = payload["stages"][-1]
    tasks = {n: i for n, i in stage["tasks"].items() if i.get("scorable")}
    missing = [n for n, i in tasks.items() if "bpo" not in i]
    if missing:
        raise SystemExit("run %s lacks bpo records for %s — was --score-bpo passed?"
                         % (path, sorted(missing)))
    return {"tasks": tasks, "payload": payload, "stage": stage, "path": str(path)}


def _get(info: dict, key: str) -> float | None:
    """取一个层级值。BPO 的键在 info['bpo'] 里,其余在 info 顶层。"""
    if key == "R_bpo":
        return info["bpo"]["R_bpo"]
    return info.get(key)


def paired_ci(runs: list[dict], left: str, right: str, *,
              only: set[str] | None = None) -> tuple[float, float, float]:
    """(left − right) 的百分点差,任务作簇,10 000 次重采样,统计量取均值。"""
    by_task: dict[str, list[float]] = {}
    for run in runs:
        for name, info in run["tasks"].items():
            if only is not None and name not in only:
                continue
            a, b = _get(info, left), _get(info, right)
            if a is None or b is None:
                continue
            by_task.setdefault(name, []).append(100.0 * (a - b))
    return cluster_bootstrap_ci(by_task, draws=10000, seed=0, statistic=np.mean)


def imbalanced_ci(runs: list[dict], ratio: str, left: str,
                  right: str) -> tuple[float, float, float] | None:
    """Q4:在不平衡子集**自身**上比 left − right。"""
    by_task: dict[str, list[float]] = {}
    for run in runs:
        for name, info in run["tasks"].items():
            block = info["bpo"].get("imbalanced", {}).get(ratio)
            if not block or left not in block or right not in block:
                continue
            by_task.setdefault(name, []).append(100.0 * (block[left] - block[right]))
    if not by_task:
        return None
    return cluster_bootstrap_ci(by_task, draws=10000, seed=0, statistic=np.mean)


def mean_level(runs: list[dict], key: str) -> float:
    vals = [v for run in runs for info in run["tasks"].values()
            if (v := _get(info, key)) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="runs/phase2q_decide.json")
    args = parser.parse_args()

    runs = [load_final_stage(Path(p)) for p in args.runs]
    out: dict = {"n_seeds": len(runs), "runs": [r["path"] for r in runs]}

    out["levels"] = {k: mean_level(runs, k) for k in
                     ("R_raw", "R_shr_global", "R_bpo", "R_shr_scoped", "R_orc")}

    # ---- 训练闸门,无条件 ----
    raws = [info["R_raw"] for run in runs for info in run["tasks"].values()]
    out["gate"] = {"median_R_raw": float(np.median(raws)),
                   "pass": bool(np.median(raws) > GATE_MEDIAN_RAW)}

    # ---- Q1 主判据 ----
    d, lo, hi = paired_ci(runs, "R_bpo", "R_shr_global")
    out["Q1"] = {"delta_pp": d, "ci": [lo, hi], "pass": bool(lo > 0.0),
                 "spec": "R_bpo - R_shr_global, CI excludes 0 positive"}

    # ---- Q2 无伤害:均值不低于 R_raw,且没有单任务掉超过 2pp ----
    d2, lo2, hi2 = paired_ci(runs, "R_bpo", "R_raw")
    per_task: dict[str, list[float]] = {}
    for run in runs:
        for name, info in run["tasks"].items():
            per_task.setdefault(name, []).append(
                100.0 * (info["bpo"]["R_bpo"] - info["R_raw"]))
    per_task_mean = {n: float(np.mean(v)) for n, v in per_task.items()}
    worst = min(per_task_mean.items(), key=lambda kv: kv[1]) if per_task_mean else None
    out["Q2"] = {
        "delta_pp": d2, "ci": [lo2, hi2],
        "per_task_delta_pp": per_task_mean,
        "worst_task": worst[0] if worst else None,
        "worst_delta_pp": worst[1] if worst else None,
        "pass": bool(d2 >= 0.0 and worst and worst[1] >= -Q2_MAX_LOSS_PP),
        "spec": "R_bpo >= R_raw at point est AND no task loses >2pp",
    }

    # ---- Q3 天花板差 ----
    d3, lo3, hi3 = paired_ci(runs, "R_shr_scoped", "R_bpo")
    out["Q3"] = {"delta_pp": d3, "ci": [lo3, hi3],
                 "pass": bool(d3 <= Q3_MAX_PP),
                 "spec": "R_shr_scoped - R_bpo <= 1pp point est"}

    # ---- Q4 Q1 是不是平衡 audit 的假象 ----
    q4: dict = {"spec": "imbalanced mixes; only 0.7 is thresholded",
                "threshold_ratio": Q4_THRESHOLD_RATIO, "by_ratio": {}}
    for ratio in ("0.5", "0.7", "0.9"):
        res = imbalanced_ci(runs, ratio, "R_bpo", "R_shr_global")
        vs_raw = imbalanced_ci(runs, ratio, "R_bpo", "R_raw")
        if res is None:
            continue
        q4["by_ratio"][ratio] = {
            "bpo_minus_global_pp": res[0], "ci": [res[1], res[2]],
            "bpo_minus_raw_pp": vs_raw[0] if vs_raw else None,
        }
    thr = q4["by_ratio"].get(Q4_THRESHOLD_RATIO)
    q4["pass"] = bool(thr and thr["bpo_minus_global_pp"] > 0.0)
    out["Q4"] = q4

    # ---- Q5 批大小敏感性,报告不设阈 ----
    by_bs: dict[str, list[float]] = {}
    for run in runs:
        for info in run["tasks"].values():
            for bs, acc in info["bpo"].get("R_bpo_by_batch", {}).items():
                by_bs.setdefault(bs, []).append(100.0 * (acc - info["R_shr_global"]))
    out["Q5"] = {"minus_global_pp_by_batch":
                 {bs: float(np.mean(v)) for bs, v in sorted(by_bs.items())},
                 "full_batch_pp": out["Q1"]["delta_pp"],
                 "spec": "reported, not thresholded"}

    # ---- Q6 Prop 1 必须仍然成立 ----
    singles = [(n, info) for run in runs for n, info in run["tasks"].items()
               if info.get("scope_size") == 1]
    # 协议 §3 Q6 点名的量是 `Δ_id_scoped`,而记录里**没有**这个字段。
    # 定义见正文 eq:delta-id:Δ_id = R_orc − R_shr,所以 scoped 版就是
    # R_orc − R_shr_scoped;单任务域上 Prop 1 要求它恒为 0。
    # phase2k_qoc/decide.py:109 早在 2Q 之前就是这么构造这个 key 的,这里对齐它。
    # 注意**不要**读 `scope_recoverable`(= R_shr_scoped − R_shr_global):那是另一个量,
    # 单任务域上没有任何理由为 0,论文正文报告的 +1.51pp 增益大部分正来自它。
    # 详见 notes/phase2q_batch_prior_offset_protocol.md §A1.4 与 §A1.6。
    viol = [(n, info["R_orc"] - info["R_shr_scoped"]) for n, info in singles
            if abs(info["R_orc"] - info["R_shr_scoped"]) > 1e-12]
    out["Q6"] = {"n_singleton_obs": len(singles), "violations": viol,
                 "pass": not viol,
                 "field": "R_orc - R_shr_scoped",
                 "spec": "Delta_id_scoped == 0 exactly on singleton scopes"}

    # ---- 预先声明的结局,§4 ----
    q1, q2, q3, q4p = (out["Q1"]["pass"], out["Q2"]["pass"],
                       out["Q3"]["pass"], out["Q4"]["pass"])
    if q1 and q2 and q3 and q4p:
        fired = 1
    elif q1 and q2 and not q3:
        fired = 2
    elif q1 and not q4p:
        fired = 3
    elif not q1:
        fired = 4
    else:
        fired = 5
    out["outcome_fired"] = fired

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True),
                              encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())