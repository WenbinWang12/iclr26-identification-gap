"""对 QOC 结果套用 `notes/method_qoc_v1.md` §6 的预注册判据。

正面结果需要三条：

1. 域划分有效且 Prop 1 的预测成立 —— `R_shr_scoped ≥ R_shr_global` 中位数成立，
   且所有单任务域的 Δ_id_scoped 恒为 0（这是可证伪预测，不是拟合）。
2. 码本买到了域划分之外的东西 —— `R_qoc(4, prototype) − R_shr_scoped` 的
   聚类自助 95% CI 不含 0。
3. 最差任务损失对 m ∈ {1,2,4} 单调不增，且与该域实测 q_m 的排序一致。

另外把残差拆成量化损失与路由损失（§5）：只有前者才是 q_m 预测的容量陈述。
判据 (2) 用 `prototype`；`confidence` 与 `oracle` 一并报告，oracle 只作上界。

本脚本在任何 QOC 结果产生**之前**写成。阈值与 §6 一致，无其他可调项。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.phase2j_offset_conflict.decide import (
    cluster_bootstrap_ci,
    spearman,
)


def load(runs: Path, seeds) -> list[dict]:
    records = []
    for seed in seeds:
        path = runs / ("qoc_seed%d.json" % seed)
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise SystemExit("no QOC records found under %s" % runs)
    return records


def training_gate(records, *, threshold: float = 0.60) -> dict:
    """预注册训练闸门（方法笔记 §M4.4），必须无条件上报。

    量：每个任务**刚训练完那一刻**的原始受限 argmax 平衡准确率（`R_raw` 取自
    `stage["trained_task"] == name` 的那个阶段），对可打分任务取中位数。

    存在的理由：M3 那批跑里 `--epochs` 是死参数，全流只有 74 个梯度步，
    WiC/COPA/QQP 的原始准确率贴着 0.50。模型没学会任务，"遗忘"就没有语义，
    判据 2 也就没有可恢复的东西。所以判据的成败必须**附带**这个闸门一起读：
    闸门不过时，判据结果的证据力不比 M3 更强，得照实说，不能把它当成
    "在收敛模型上仍然失败"来引用。
    """

    per_task: dict[str, list[float]] = {}
    steps: list[int] = []
    for record in records:
        for stage in record["stages"]:
            steps.append(int(stage.get("n_steps") or 0))
            name = stage["trained_task"]
            entry = stage["tasks"].get(name)
            if entry is None or not entry.get("scorable"):
                continue
            value = entry.get("R_raw")
            if value is not None:
                per_task.setdefault(name, []).append(float(value))

    medians = {k: float(np.median(v)) for k, v in sorted(per_task.items())}
    overall = float(np.median(list(medians.values()))) if medians else float("nan")
    at_chance = sorted(k for k, v in medians.items() if v < 0.55)
    return {
        "PREREGISTERED_M4_4": True,
        "quantity": "median over scorable tasks of post-training R_raw (balanced)",
        "threshold": threshold,
        "median_post_training_raw": round(overall, 4),
        "pass": bool(overall > threshold),
        "per_task_median": {k: round(v, 4) for k, v in medians.items()},
        "tasks_still_near_chance_below_0p55": at_chance,
        "total_gradient_steps_per_seed": (
            int(np.median([sum(steps[i::len(records)]) for i in range(len(records))]))
            if records and steps else None
        ),
        "note": ("闸门不过 => 判据 2 的结果不得被引用为「收敛模型上仍失败」；"
                 "它的证据力与 M3 相同。"),
    }


def collect(records, *, m: str, router: str):
    """按任务收集各阶段测量，排除刚训练完的任务（那不是遗忘）。

    返回 {task: [ (scope_recoverable, codebook_gain, quantization, routing) ]}。
    """
    out: dict[str, list[dict]] = {}
    for record in records:
        for stage in record["stages"]:
            for name, entry in stage["tasks"].items():
                if name == stage["trained_task"]:
                    continue
                if not entry["scorable"] or m not in entry.get("qoc", {}):
                    continue
                block = entry["qoc"][m]
                out.setdefault(name, []).append({
                    "scope_recoverable": entry["scope_recoverable"],
                    "codebook_gain": block[router]["accuracy"] - entry["R_shr_scoped"],
                    "quantization_loss": entry["R_orc"] - block["oracle"]["accuracy"],
                    "routing_loss": (block["oracle"]["accuracy"]
                                     - block[router]["accuracy"]),
                    "Delta_id_global": entry["Delta_id_global"],
                    "Delta_id_scoped": entry["R_orc"] - entry["R_shr_scoped"],
                    "scope": entry["scope"],
                    "scope_size": entry["scope_size"],
                    "q_m": block[router]["radius_q_m"],
                    "route_match": block[router]["route_matches_oracle"],
                })
    return out


def criterion_1(records, by_task) -> dict:
    """域划分免费增益 + Prop 1 在单任务域上的可证伪预测。"""
    gains = {task: [row["scope_recoverable"] for row in rows]
             for task, rows in by_task.items()}
    point, lo, hi = cluster_bootstrap_ci(gains)
    # 单任务域：域内只有它自己，域内共享偏移就等于它自己的最优偏移，
    # 所以 Δ_id_scoped 必须恒为 0。任何非零都说明实现有 bug 或 Prop 1 是假的。
    violations = []
    for task, rows in by_task.items():
        for row in rows:
            if row["scope_size"] == 1 and abs(row["Delta_id_scoped"]) > 1e-9:
                violations.append({"task": task, "scope": row["scope"],
                                   "Delta_id_scoped": row["Delta_id_scoped"]})
    singleton_seen = sorted({row["scope"] for rows in by_task.values()
                             for row in rows if row["scope_size"] == 1})
    return {
        "median_scope_recoverable": point,
        "ci95": [lo, hi],
        "singleton_scopes_observed": singleton_seen,
        "prop1_violations": violations[:10],
        "n_prop1_violations": len(violations),
        "pass": bool(point >= 0.0 and not violations),
    }


def criterion_2(by_task, *, router: str) -> dict:
    """码本增益的聚类自助 CI 必须不含 0。"""
    gains = {task: [row["codebook_gain"] for row in rows]
             for task, rows in by_task.items()}
    point, lo, hi = cluster_bootstrap_ci(gains)
    return {"router": router, "median_codebook_gain": point, "ci95": [lo, hi],
            "n_tasks": len(gains), "pass": bool(lo > 0.0)}


def criterion_3(records, *, router: str) -> dict:
    """最差任务损失对 m 单调，且与 q_m 排序一致。"""
    ms = ["1", "2", "4"]
    worst: dict[str, list[float]] = {m: [] for m in ms}
    radii: dict[str, list[float]] = {m: [] for m in ms}
    for record in records:
        for stage in record["stages"]:
            per_scope: dict[str, dict[str, list[float]]] = {}
            for name, entry in stage["tasks"].items():
                if name == stage["trained_task"] or not entry["scorable"]:
                    continue
                for m in ms:
                    if m not in entry.get("qoc", {}):
                        continue
                    loss = entry["R_orc"] - entry["qoc"][m][router]["accuracy"]
                    slot = per_scope.setdefault(entry["scope"], {})
                    slot.setdefault(m, []).append(loss)
            for scope, blocks in per_scope.items():
                for m, losses in blocks.items():
                    worst[m].append(max(losses))
                    radii_map = stage["scope_radii"].get(m, {}).get(scope)
                    if radii_map is not None:
                        radii[m].append(radii_map["q_m"])
    medians = {m: (float(np.median(v)) if v else float("nan"))
               for m, v in worst.items()}
    monotone = all(
        medians[b] <= medians[a] + 1e-9
        for a, b in zip(ms, ms[1:])
        if not (np.isnan(medians[a]) or np.isnan(medians[b]))
    )
    # q_m 与最差损失在同一批 (阶段, 域) 上配对，检验排序一致性。
    flat_q, flat_loss = [], []
    for m in ms:
        n = min(len(radii[m]), len(worst[m]))
        flat_q.extend(radii[m][:n])
        flat_loss.extend(worst[m][:n])
    rho = spearman(np.array(flat_q), np.array(flat_loss)) if len(flat_q) >= 3 else float("nan")
    return {"router": router, "median_worst_loss_by_m": medians,
            "monotone_in_m": bool(monotone),
            "spearman_q_m_vs_worst_loss": rho, "n_points": len(flat_q),
            "pass": bool(monotone)}


def criterion_2_conflict_scopes_only(by_task, *, router: str) -> dict:
    """**事后分析，不是预注册判据。**只在多成员域上算码本增益。

    为什么需要它：冻结的判据 2 对**所有**可打分任务取中位数，但 Order-4 里
    大多数任务落在单任务域，那里 m=1 时 q_m 已经是 0，码本在数学上**不可能**
    有增益。于是中位数被一堆结构性的 0 稀释，判据 2 检验的其实是「多成员域
    在可打分任务里占多数吗」，而不是「码本买到东西了吗」。这是我冻结判据时
    的规格错误。

    处理方式：判据 2 照原样报告为失败，另附此项并**明确标注为事后分析**。
    绝不把它当作预注册判据通过。
    """
    gains = {task: [row["codebook_gain"] for row in rows]
             for task, rows in by_task.items()
             if any(row["scope_size"] >= 2 for row in rows)}
    if not gains:
        return {"post_hoc": True, "n_tasks": 0, "note": "no multi-member scope tasks"}
    point, lo, hi = cluster_bootstrap_ci(gains)
    return {"post_hoc_NOT_PREREGISTERED": True, "router": router,
            "median_codebook_gain": point, "ci95": [lo, hi],
            "n_tasks": len(gains), "tasks": sorted(gains),
            "ci_excludes_zero": bool(lo > 0.0)}


def decompose(by_task) -> dict:
    """§5 的残差拆分：量化损失 vs 路由损失。"""
    def med(key):
        vals = [row[key] for rows in by_task.values() for row in rows]
        return float(np.median(vals)) if vals else float("nan")
    return {"median_quantization_loss": med("quantization_loss"),
            "median_routing_loss": med("routing_loss"),
            "median_Delta_id_global": med("Delta_id_global"),
            "median_Delta_id_scoped": med("Delta_id_scoped"),
            "median_route_match_oracle": med("route_match")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="/mnt/data/wenbin/iclr26/runs/phase2k_qoc_full")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--m", default="4")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    records = load(Path(args.runs), seeds)

    verdict = {"seeds": [r["args"]["seed"] for r in records], "m": args.m}
    verdict["training_gate"] = training_gate(records)
    for router in ("prototype", "batch_margin", "confidence"):
        by_task = collect(records, m=args.m, router=router)
        block = {
            "criterion_1_scope_is_free_and_prop1_holds": criterion_1(records, by_task),
            "criterion_2_codebook_buys_more": criterion_2(by_task, router=router),
            "post_hoc_conflict_scopes_only": criterion_2_conflict_scopes_only(
                by_task, router=router),
            "criterion_3_monotone_in_m": criterion_3(records, router=router),
            "decomposition": decompose(by_task),
        }
        block["all_pass"] = all(block[k]["pass"] for k in block
                                if k.startswith("criterion"))
        verdict[router] = block

    # oracle 只作上界参照，单独列出，绝不作为方法结果。
    oracle_by_task = collect(records, m=args.m, router="oracle")
    verdict["oracle_upper_reference_not_a_method"] = criterion_2(
        oracle_by_task, router="oracle")

    # 泄漏量化：官方提示词字面含 `Dataset:WiC`，读输入的路由能直接读出任务
    # 身份。这一项**不是方法结果**，只用来说明泄漏幅度有多大。
    leaky_by_task = collect(records, m=args.m, router="prototype_official_leaky")
    verdict["prototype_official_leaky_NOT_EVIDENCE"] = criterion_2(
        leaky_by_task, router="prototype_official_leaky")

    primary = verdict["prototype"]
    if primary["criterion_1_scope_is_free_and_prop1_holds"]["pass"] and \
            primary["criterion_2_codebook_buys_more"]["pass"]:
        verdict["decision"] = "POSITIVE"
    elif primary["criterion_1_scope_is_free_and_prop1_holds"]["pass"]:
        verdict["decision"] = "NEGATIVE_SCOPE_ONLY"
        verdict["rationale"] = (
            "域划分免费买到了增益，但码本没有；识别间隙主要是跨域记账，"
            "与 Phase-2J 判据 3（ω 预测保持损失）需要显式调和。")
    else:
        verdict["decision"] = "KILL"

    # 闸门不过时，判决字符串本身就带上标记，避免下游只读 decision 字段而
    # 漏掉前提没成立这件事。
    gate = verdict["training_gate"]
    if not gate["pass"]:
        verdict["decision_qualifier"] = (
            "UNDERTRAINED_PREMISE_NOT_MET: median post-training raw accuracy "
            "%.4f <= %.2f；判据结果的证据力不强于 M3。"
            % (gate["median_post_training_raw"], gate["threshold"])
        )

    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    (Path(args.runs) / "verdict_qoc.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    print("TRAINING GATE %s (median post-training raw %.4f, threshold %.2f)"
          % ("PASS" if gate["pass"] else "FAIL",
             gate["median_post_training_raw"], gate["threshold"]))
    if gate["tasks_still_near_chance_below_0p55"]:
        print("still near chance: %s"
              % ", ".join(gate["tasks_still_near_chance_below_0p55"]))
    print("DECISION %s" % verdict["decision"])
    if "decision_qualifier" in verdict:
        print(verdict["decision_qualifier"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
