"""Phase-2M §3 的预注册判据。

**为什么不能直接用 `phase2k_qoc/decide.py`**：2K 的判据打的是 *scoped gap*
（`R_qoc − R_shr_scoped` 之类的域内量），2M §3 打的是**水平**（ACC / ACC_scoped）
以及它相对 `none` 基线的差。是两组不同的量，不是同一个脚本换个路径的事。

但自助法必须是同一套：本文件 `from experiments.phase2j_offset_conflict.decide
import cluster_bootstrap_ci`，簇 = 任务，10000 次重采样，和 2J/2K/2L 完全一致。
点估计统计量在这里显式取**均值**（`statistic=np.mean`），因为 §2 把 ACC 定义为
"对 15 个任务取平均"；2K 判据默认的中位数是为它自己那些 gap 量选的。

判据编号与 `notes/phase2m_vla_protocol.md` §3 一一对应，本脚本在确认跑结果
产生**之后**才写（协议冻结在前，评分代码在后），所以它不允许有任何阈值选择：
所有阈值都从 §3 抄下来，写成模块级常量。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.phase2j_offset_conflict.decide import cluster_bootstrap_ci, spearman

#: §3 判据 3 的可塑性护栏：`none` 的 peak，与允许的最大跌幅。
NONE_PEAK = 0.7381
PLASTICITY_TOLERANCE_PP = 2.0
#: §3 无条件训练闸门。
GATE_MEDIAN_R_RAW = 0.60
DRAWS = 10000


def load(runs: Path, seeds) -> list[dict]:
    records = []
    for seed in seeds:
        path = runs / ("qoc_seed%d.json" % seed)
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise SystemExit("no records under %s" % runs)
    return records


def _final_stage(record: dict) -> dict:
    """最后一个阶段 = position 最大的那个，而不是列表末尾。

    列表顺序目前恰好一致，但依赖它是脆的：只要有人加了中途 checkpoint 的
    写入，末尾就不再是"训完全部任务之后"。
    """
    return max(record["stages"], key=lambda s: int(s["position"]))


def accuracy_by_task(records, *, m: str, router: str) -> tuple[dict, dict]:
    """返回 (ACC 的按任务簇, ACC_scoped 的按任务簇)。

    ACC = 最终阶段每个可打分任务的 `R_shr_global`：共享输出层、不加偏置的
    受限 argmax 平衡准确率，即"部署时真实拿到的水平"。
    ACC_scoped = 同一阶段的 `R_shr_scoped`（域索引偏置，free 那一层）。

    刻意**不**用 `R_orc`：它在 `risk` 上拟合、在 `audit` 上打分，是个有噪声的
    参考量而非上界（Phase-2K 发现 12.8% 的 scoped gap 观测为负就是这个原因）。
    """
    acc: dict[str, list[float]] = {}
    acc_scoped: dict[str, list[float]] = {}
    for record in records:
        stage = _final_stage(record)
        for name, entry in stage["tasks"].items():
            if not entry.get("scorable"):
                continue
            acc.setdefault(name, []).append(float(entry["R_shr_global"]))
            acc_scoped.setdefault(name, []).append(float(entry["R_shr_scoped"]))
    return acc, acc_scoped


def paired_diff_by_task(treat: dict, base: dict) -> dict:
    """按任务配对求差。只保留两边都有的任务，缺任务要说出来而不是静默丢弃。"""
    missing = sorted(set(treat) ^ set(base))
    if missing:
        raise ValueError("task sets differ between arms: %s" % missing)
    out = {}
    for name in sorted(treat):
        a, b = treat[name], base[name]
        if len(a) != len(b):
            raise ValueError("unequal seed count for %s: %d vs %d"
                             % (name, len(a), len(b)))
        out[name] = [x - y for x, y in zip(a, b)]
    return out


def peak_accuracy(records) -> dict:
    """每个任务刚训完那一刻的 `R_raw`，按任务分簇。§2 要求 peak 始终并列上报。"""
    peaks: dict[str, list[float]] = {}
    for record in records:
        for stage in record["stages"]:
            name = stage["trained_task"]
            entry = stage["tasks"].get(name)
            if entry is not None and entry.get("scorable"):
                peaks.setdefault(name, []).append(float(entry["R_raw"]))
    return peaks


def training_gate(records) -> dict:
    per_task = peak_accuracy(records)
    flat = [v for vs in per_task.values() for v in vs]
    median = float(np.median(flat)) if flat else float("nan")
    return {"median_R_raw": median, "threshold": GATE_MEDIAN_R_RAW,
            "passed": bool(median > GATE_MEDIAN_R_RAW), "n_obs": len(flat)}


def radii(records, *, m: str) -> dict:
    """每个 (scope, seed) 的 q_m，按 scope 分簇。判据 4。

    只取最终阶段：这是 §3 判据 4 说的"每个冲突域的半径"，而中途阶段的域构成
    还在变，混在一起会把域数量的变化读成半径的变化。
    """
    out: dict[str, list[float]] = {}
    for record in records:
        stage = _final_stage(record)
        for scope, info in stage["scope_radii"].get(m, {}).items():
            out.setdefault(scope, []).append(float(info["q_m"]))
    return out


def prop1_violations(records) -> dict:
    """判据 5：单任务域的 Δ_id_scoped 必须恒为 0。

    `R_shr_scoped == R_orc` 就是 Δ_id_scoped == 0。单任务域里域索引偏置能
    完全实现该任务的最优偏置，所以这是恒等式而非近似——任何非零都是接线错误。
    """
    checked = violations = 0
    worst = 0.0
    for record in records:
        for stage in record["stages"]:
            for entry in stage["tasks"].values():
                if not entry.get("scorable") or int(entry["scope_size"]) != 1:
                    continue
                checked += 1
                gap = abs(float(entry["R_orc"]) - float(entry["R_shr_scoped"]))
                worst = max(worst, gap)
                if gap > 1e-9:
                    violations += 1
    return {"checked": checked, "violations": violations, "worst_abs_gap": worst}


def evaluate(treat_records, base_records, *, m: str, router: str) -> dict:
    t_acc, t_scoped = accuracy_by_task(treat_records, m=m, router=router)
    b_acc, b_scoped = accuracy_by_task(base_records, m=m, router=router)

    d_acc = paired_diff_by_task(t_acc, b_acc)
    d_scoped = paired_diff_by_task(t_scoped, b_scoped)

    c1 = cluster_bootstrap_ci(d_acc, draws=DRAWS, statistic=np.mean)
    c2 = cluster_bootstrap_ci(d_scoped, draws=DRAWS, statistic=np.mean)

    t_peak, b_peak = peak_accuracy(treat_records), peak_accuracy(base_records)
    t_peak_mean = float(np.mean([v for vs in t_peak.values() for v in vs]))
    b_peak_mean = float(np.mean([v for vs in b_peak.values() for v in vs]))

    def level(by_task):
        return float(np.mean([v for vs in by_task.values() for v in vs]))

    result = {
        "m": m, "router": router,
        "n_seeds_treat": len(treat_records), "n_seeds_base": len(base_records),
        "ACC_treat": level(t_acc), "ACC_base": level(b_acc),
        "ACC_scoped_treat": level(t_scoped), "ACC_scoped_base": level(b_scoped),
        "criterion_1": {"point": c1[0], "lo": c1[1], "hi": c1[2],
                        "passed": bool(c1[1] > 0.0)},
        "criterion_2": {"point": c2[0], "lo": c2[1], "hi": c2[2],
                        "passed": bool(c2[1] > 0.0)},
        "criterion_3": {"peak_treat": t_peak_mean, "peak_base": b_peak_mean,
                        "peak_reference": NONE_PEAK,
                        "drop_pp_vs_reference": 100.0 * (NONE_PEAK - t_peak_mean),
                        "tolerance_pp": PLASTICITY_TOLERANCE_PP,
                        "passed": bool(100.0 * (NONE_PEAK - t_peak_mean)
                                       <= PLASTICITY_TOLERANCE_PP)},
        "criterion_5": prop1_violations(treat_records),
        "training_gate": training_gate(treat_records),
        "per_task": {n: {"treat": float(np.mean(t_acc[n])),
                         "base": float(np.mean(b_acc[n])),
                         "diff_pp": 100.0 * float(np.mean(d_acc[n]))}
                     for n in sorted(d_acc)},
    }
    result["criterion_5"]["passed"] = result["criterion_5"]["violations"] == 0

    # 判据 4：q_1 与 q_2 都必须下降。域集合两边可能不完全相同（域是由任务的
    # verbalizer 决定的，和 arm 无关，但可打分性会变），只比交集，并报告差集。
    c4 = {}
    for mm in ("1", "2"):
        t_r, b_r = radii(treat_records, m=mm), radii(base_records, m=mm)
        shared = sorted(set(t_r) & set(b_r))
        diffs = {s: [np.mean(t_r[s]) - np.mean(b_r[s])] for s in shared}
        pt, lo, hi = cluster_bootstrap_ci(diffs, draws=DRAWS, statistic=np.mean)
        # 域交集为空时不要让 np.mean 静默给出 NaN：判据 4 无法评估这件事本身
        # 就是要上报的信息（两个 arm 的可打分域完全不重叠）。
        nan = float("nan")
        c4["q_%s" % mm] = {
            "treat_mean": float(np.mean([v for s in shared for v in t_r[s]]))
                          if shared else nan,
            "base_mean": float(np.mean([v for s in shared for v in b_r[s]]))
                         if shared else nan,
            "diff_point": pt, "lo": lo, "hi": hi,
            "n_scopes": len(shared),
            "evaluable": bool(shared),
            "only_in_treat": sorted(set(t_r) - set(b_r)),
            "only_in_base": sorted(set(b_r) - set(t_r)),
            "decreased": bool(shared and pt < 0.0),
            "decreased_significantly": bool(shared and hi < 0.0),
        }
    c4["passed"] = bool(c4["q_1"]["decreased"] and c4["q_2"]["decreased"])
    c4["evaluable"] = bool(c4["q_1"]["evaluable"] and c4["q_2"]["evaluable"])
    result["criterion_4"] = c4
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--treat", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--m", default="4")
    parser.add_argument("--router", default="prototype")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    result = evaluate(load(Path(args.treat), seeds), load(Path(args.base), seeds),
                      m=args.m, router=args.router)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
