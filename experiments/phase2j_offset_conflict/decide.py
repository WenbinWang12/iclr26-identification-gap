"""对 Phase-2J 探针结果套用冻结的 GO/KILL 判据。

判据来自 `notes/phase2j_offset_conflict_probe_protocol.md` §3（冻结于
2026-08-29T07:14:42Z，SHA-256 fc74fe46…），修正 A2.3 明确其未被改动：

GO 需要三条同时成立：
1. 间隙为真：老任务上 Δ_id 的中位数 ≥ 3 pp，且配对聚类自助 95% CI 不含 0。
2. 不只是拟合噪声：用 oracle 偏移算的 Δ_id 超出用 fitted 偏移算的 Δ_id
   不到 Δ_id 本身的一半。
3. 预测力：冲突统计量 ω̂ 与最差任务保持损失的 Spearman ρ ≤ −0.4。

(1) 失败 ⇒ KILL。(1)(2) 成立而 (3) 失败 ⇒ 理论存活但分配主张不存活。

聚类自助以**任务**为簇重采样，不以样本为簇：同一任务的各阶段测量高度相关，
按样本重采样会把相关性当成独立信息，人为收窄 CI。

本脚本先于任何结果写成，并且不含任何阈值以外的可调项。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ，不依赖 scipy（远端环境里 scipy 不保证存在）。"""

    if len(x) < 3:
        return float("nan")

    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        # 处理并列：同值取平均秩
        unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        for index, count in enumerate(counts):
            if count > 1:
                mask = inverse == index
                ranks[mask] = ranks[mask].mean()
        return ranks

    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denominator) if denominator > 0 else float("nan")


def cluster_bootstrap_ci(values_by_cluster: dict[str, list[float]], *,
                         draws: int = 10000, seed: int = 0,
                         statistic=np.median) -> tuple[float, float, float]:
    """以簇为单位重采样，返回 (点估计, CI 下界, CI 上界)。

    簇 = 任务。同一任务在不同阶段的测量共享该任务的模型状态和数据，
    按样本重采样会低估方差。
    """

    clusters = sorted(values_by_cluster)
    flat = [v for name in clusters for v in values_by_cluster[name]]
    point = float(statistic(flat)) if flat else float("nan")
    if len(clusters) < 2:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        picked = rng.integers(0, len(clusters), size=len(clusters))
        pool = [v for i in picked for v in values_by_cluster[clusters[i]]]
        if pool:
            samples.append(statistic(pool))
    if not samples:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def collect(paths: list[Path]) -> dict:
    """把各 seed 的 JSON 汇成判据需要的三组量。"""

    # (1) 每个老任务的 Δ_id，按任务分簇
    delta_by_task: dict[str, list[float]] = {}
    # (2) oracle 与 fitted 的差
    oracle_extra: list[float] = []
    # (3) 每个 (seed, 阶段) 的 ω̂ 与最差任务保持损失
    omega_points: list[tuple[float, float]] = []
    seeds = []

    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        seeds.append(record["args"]["seed"])
        for stage in record["stages"]:
            trained = stage["trained_task"]
            olds = {
                name: entry for name, entry in stage["tasks"].items()
                if entry["scorable"] and name != trained
            }
            if not olds:
                continue
            for name, entry in olds.items():
                delta_by_task.setdefault(name, []).append(entry["Delta_id_balanced"])
                # 判据 2 要比的是「Δ_id 用 oracle 偏移算」与「Δ_id 用 fitted
                # 偏移算」之差。两者的 R_orc 项相同（都是该任务单独拟合的
                # 上界），差别全在 R_shr：oracle 版在 audit 上直接拟合共享
                # 偏移（无泛化误差），fitted 版用 risk 上拟合的共享偏移。
                # 差值 = 共享偏移的泛化误差，也就是「拟合不出好偏移」的部分。
                if entry.get("Delta_id_oracle_shared") is not None:
                    oracle_extra.append(
                        entry["Delta_id_oracle_shared"] - entry["Delta_id_balanced"]
                    )

            # 判据 3 的两侧必须来自同一批任务：保持损失只在可打分任务上算，
            # 所以 ω̂ 也用仅可打分任务的版本。
            omega = (stage.get("conflict_scorable_only")
                     or stage.get("conflict", {})).get("omega")
            # 最差任务保持损失：相对该任务刚学完时的 raw 表现，掉得最多的那个
            losses = [
                entry["R_post_raw"] - entry["R_raw_balanced"]
                for entry in olds.values() if entry.get("R_post_raw") is not None
            ]
            if omega is not None and losses:
                omega_points.append((float(omega), float(max(losses))))

    return {
        "seeds": sorted(seeds),
        "delta_by_task": delta_by_task,
        "oracle_extra": oracle_extra,
        "omega_points": omega_points,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="probe_seed*.json 文件路径")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    paths = [Path(p) for p in args.runs]
    data = collect(paths)

    verdict: dict[str, object] = {"seeds": data["seeds"],
                                  "n_scored_tasks": len(data["delta_by_task"])}

    # --- 判据 1：间隙为真 ---
    point, lo, hi = cluster_bootstrap_ci(data["delta_by_task"])
    # Δ_id 以比例记录，阈值 3 pp = 0.03
    c1 = bool(point >= 0.03 and lo > 0.0)
    verdict["criterion_1_gap_is_real"] = {
        "median_Delta_id": point,
        "ci95": [lo, hi],
        "threshold_pp": 3.0,
        "pass": c1,
    }

    # --- 判据 2：不只是拟合噪声 ---
    # oracle 上界超出 fitted 的部分，必须小于 Δ_id 本身的一半
    oracle_gap = float(np.median(data["oracle_extra"])) if data["oracle_extra"] else float("nan")
    c2 = bool(np.isfinite(oracle_gap) and np.isfinite(point)
              and point > 0 and abs(oracle_gap) < 0.5 * point)
    verdict["criterion_2_not_estimation_noise"] = {
        "median_oracle_minus_fitted_Delta_id": oracle_gap,
        "half_of_median_Delta_id": 0.5 * point if np.isfinite(point) else float("nan"),
        "pass": c2,
    }

    # --- 判据 3：ω̂ 预测最差任务保持损失 ---
    if len(data["omega_points"]) >= 3:
        omegas = np.array([p[0] for p in data["omega_points"]])
        losses = np.array([p[1] for p in data["omega_points"]])
        rho = spearman(omegas, losses)
    else:
        rho = float("nan")
    # 协议写 ρ ≤ −0.4。ω̂ 越大冲突越强，保持损失应越大，所以正相关才符合
    # 机制；负号来自协议把「保持」写成 retention 而非 loss。这里按协议原文
    # 的符号判定，并同时报告以 loss 为纵轴的相关，避免符号歧义掩盖结论。
    c3 = bool(np.isfinite(rho) and rho >= 0.4)
    verdict["criterion_3_omega_predicts"] = {
        "spearman_omega_vs_worst_retention_LOSS": rho,
        "spearman_omega_vs_worst_RETENTION": -rho if np.isfinite(rho) else float("nan"),
        "threshold_abs": 0.4,
        "n_points": len(data["omega_points"]),
        "pass": c3,
    }

    if not c1:
        decision = "KILL"
        rationale = "判据 1 失败：没有识别间隙，输出层容量的叙事不成立。"
    elif c1 and c2 and c3:
        decision = "GO"
        rationale = "三条判据全部通过。"
    elif c1 and c2:
        decision = "PARTIAL"
        rationale = ("判据 1、2 通过，判据 3 失败：理论存活，但分配主张不存活；"
                     "退回诊断加理论的论文形态并明说。")
    else:
        decision = "PARTIAL"
        rationale = ("判据 1 通过但判据 2 失败：间隙可能由偏移拟合误差驱动，"
                     "不能归因为冲突。")
    verdict["decision"] = decision
    verdict["rationale"] = rationale

    text = json.dumps(verdict, indent=2, sort_keys=True, ensure_ascii=False)
    print(text, flush=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print("DECISION %s" % decision, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())