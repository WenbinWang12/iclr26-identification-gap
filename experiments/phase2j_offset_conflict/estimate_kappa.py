"""从探针记录的裕度统计估计 κ（理论笔记 §5、Prop 2b）。

κ 的定义（`notes/theory_prop2_proof.md` §4）：偏移沿最坏方向移动距离 t 时
翻转的样本比例 N(t) 至少为 κ·min(t, t_max)。所以 κ 是 N(t) 在 t=0 附近的
下界斜率，可以直接从裕度分布读出：裕度小于 t 的样本在偏移移动 t 后可能翻转，
于是 N(t) ≈ F(t)，F 是裕度的经验 CDF。κ 就是 F 在原点附近的最小斜率。

诚实性说明：探针只记录了 margin_mean 与 margin_p10，不是完整直方图。用两个
分位点估计斜率是粗糙的，所以本脚本给出的是一个**下界式的粗估**并明确标注，
不作为精确值上报。要精确值需要探针额外落盘完整裕度向量。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    per_task: dict[str, list[tuple[float, float]]] = {}
    slopes: dict[str, list[float]] = {}
    safe_slopes: dict[str, list[float]] = {}
    for path in args.runs:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        for stage in record["stages"]:
            for name, entry in stage["tasks"].items():
                if not entry.get("scorable"):
                    continue
                # 首选：完整的翻转比例曲线，直接给出 N(t)/t 的下界斜率。
                curve = entry.get("flip_fraction_lower_bound")
                if curve:
                    ratios = [
                        float(fraction) / float(t)
                        for t, fraction in curve.items()
                        if float(t) > 0 and float(fraction) > 0
                    ]
                    if ratios:
                        # κ 是下界斜率，故取曲线上各点 N(t)/t 的最小值。
                        slopes.setdefault(name, []).append(min(ratios))
                    # 网格安全版本。上面那个数只在**网格点上**验证了
                    # N(t) ≥ κ·t，而 Prop 2b 要求它对区间内的每个 t 都成立。
                    # N 单调不减，故区间 [t_i, t_{i+1}] 内的最坏情形是
                    # t → t_{i+1}⁻ 而 N(t) 仍只有 N(t_i)，于是真正安全的斜率是
                    # min_i N(t_i)/t_{i+1}，而不是 min_i N(t_i)/t_i。网格比值
                    # 最大到 2.5×，所以两者可以差 2.5 倍——这不是小数点问题，
                    # 是「报出来的 κ 是否真的是下界」的问题。
                    points = sorted((float(t), float(f)) for t, f in curve.items())
                    safe = [
                        points[i][1] / points[i + 1][0]
                        for i in range(len(points) - 1)
                        if points[i][1] > 0
                    ]
                    if safe:
                        safe_slopes.setdefault(name, []).append(min(safe))
                mean = entry.get("margin_mean")
                p10 = entry.get("margin_p10")
                if mean is None or p10 is None:
                    continue
                per_task.setdefault(name, []).append((float(mean), float(p10)))

    report: dict[str, object] = {
        "method": (
            "kappa ≈ 0.10 / p10：第 10 百分位裕度 p10 意味着有 10% 的样本裕度 "
            "≤ p10，故 N(p10) ≥ 0.10，给出 N(t)/t 在 t=p10 处的一个取值。"
            "这是粗估而非精确斜率，因为只有两个分位点可用。"
        ),
        "caveat": (
            "p10 若为负，说明超过 10% 的样本已被错分，裕度定义为 top1-top2 "
            "恒非负，故此处不应出现负值；若出现则为数据问题。"
        ),
        "per_task": {},
    }
    kappas = []
    for name, points in sorted(per_task.items()):
        p10s = np.array([p for _, p in points], dtype=float)
        means = np.array([m for m, _ in points], dtype=float)
        usable = p10s[p10s > 1e-9]
        kappa = float(np.median(0.10 / usable)) if usable.size else float("nan")
        if np.isfinite(kappa):
            kappas.append(kappa)
        slope_list = slopes.get(name, [])
        safe_list = safe_slopes.get(name, [])
        report["per_task"][name] = {
            "n_measurements": len(points),
            "median_margin_mean": float(np.median(means)),
            "median_margin_p10": float(np.median(p10s)),
            "kappa_rough_from_p10": kappa,
            "kappa_from_flip_curve": (
                float(np.median(slope_list)) if slope_list else None
            ),
            "kappa_grid_safe": (
                float(np.median(safe_list)) if safe_list else None
            ),
        }

    report["kappa_rough_median_over_tasks"] = (
        float(np.median(kappas)) if kappas else float("nan")
    )
    all_slopes = [s for values in slopes.values() for s in values]
    report["kappa_from_flip_curve_median"] = (
        float(np.median(all_slopes)) if all_slopes else None
    )
    all_safe = [s for values in safe_slopes.values() for s in values]
    report["kappa_grid_safe_median"] = (
        float(np.median(all_safe)) if all_safe else None
    )
    # 论文里要引的是**最小**的那个任务，不是中位数：Prop 2b 的 κ 必须对所有
    # 参与共享偏移的任务同时成立，一个 κ 小的任务就把界拖到它那里。中位数只是
    # 描述性的。
    per_task_safe = [
        float(np.median(v)) for v in safe_slopes.values() if v
    ]
    report["kappa_grid_safe_min_over_tasks"] = (
        float(min(per_task_safe)) if per_task_safe else None
    )
    report["preferred"] = (
        "kappa_grid_safe_min_over_tasks（论文引用值：网格安全 + 取最坏任务）"
        if all_safe else "kappa_rough_median_over_tasks（该次运行未落盘翻转曲线）"
    )
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    print(text, flush=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
