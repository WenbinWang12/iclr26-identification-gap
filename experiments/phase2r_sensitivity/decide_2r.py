"""Phase-2R 判决：对冻结协议 §3 的 R1-R6 逐条求值。

协议冻结于 038c238b22615e4fddbc4629feb9c8a9bcc9f70af7a5c0dcb5a6702843d530df，
在实现之前，SHA 见 notes/phase2r_sha.txt。本脚本不新增判据、不改阈值。

统计口径（协议 §3）：3 seeds，cluster bootstrap，**任务为 cluster**，
10 000 draws。斜率的重采样复用 phase2j 的 cluster_bootstrap_ci，不另写一份统计。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.phase2j_offset_conflict.decide import cluster_bootstrap_ci  # noqa: E402
from experiments.phase2r_sensitivity.panel import Obs, build_panel  # noqa: E402

DRAWS = 10000
SEED = 0


def ols_slope(xs, ys) -> float:
    """一元最小二乘斜率。x 无变异时返回 nan（而不是 0，那会伪装成'无效应'）。"""

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 2:
        return float("nan")
    xc = x - x.mean()
    denom = float(xc @ xc)
    if denom <= 0.0:
        return float("nan")
    return float((xc @ (y - y.mean())) / denom)


def slope_ci(pairs_by_task: dict[str, list[tuple[float, float]]]) -> dict:
    """以任务为簇重采样，统计量 = 汇集后的 OLS 斜率。

    簇结构与全文其他判据一致（任务为簇，10 000 draws，seed 0）：传给统计量的
    "值" 是 (x, y) 对本身，statistic 在池上跑 OLS，而不是对每任务斜率再取平均。

    **退化抽样的处理，及其时序，如实记录。** 协议 §3 只写了"cluster bootstrap，
    任务为 cluster，10 000 draws"，未规定当某次重采样的池内 x 无变异时怎么办
    （斜率无定义）。首次运行时确实发生了：全 5 簇都抽中 Yahoo（它只有 age=0 一
    个 x 值）的抽样，斜率为 nan，`np.percentile` 于是把整个 CI 变成 nan。

    这里的选择是**丢弃退化抽样并报告其比例**，而不是把它记作斜率 0（那会把
    "无定义"伪装成"无效应"，人为把 CI 往 0 拉）。此决定作出于第一次运行之后、
    知道点估计之后，因此**不是**冻结内容；`n_degenerate_draws` 与
    `frac_degenerate` 随每个判据一同输出，读者可自行核验其影响是否可忽略。
    若该比例不可忽略（我们取 1% 为界），则 CI 不可用，见返回的 `ci_usable`。
    """

    clusters = sorted(pairs_by_task)
    flat = [v for name in clusters for v in pairs_by_task[name]]
    arr = np.asarray(flat, dtype=float) if flat else np.zeros((0, 2))
    point = ols_slope(arr[:, 0], arr[:, 1]) if arr.shape[0] >= 2 else float("nan")
    n_obs = len(flat)
    base = {"slope_pp_per_unit": point, "n_clusters": len(clusters), "n_obs": n_obs}

    if len(clusters) < 2:
        return dict(base, ci=[float("nan"), float("nan")], n_degenerate_draws=0,
                    frac_degenerate=float("nan"), ci_usable=False,
                    ci_note="fewer than 2 clusters; a cluster bootstrap CI is undefined")

    rng = np.random.default_rng(SEED)
    samples: list[float] = []
    degenerate = 0
    for _ in range(DRAWS):
        picked = rng.integers(0, len(clusters), size=len(clusters))
        pool = [v for i in picked for v in pairs_by_task[clusters[i]]]
        a = np.asarray(pool, dtype=float)
        s = ols_slope(a[:, 0], a[:, 1]) if a.shape[0] >= 2 else float("nan")
        if np.isnan(s):
            degenerate += 1
        else:
            samples.append(s)

    frac = degenerate / DRAWS
    if not samples:
        return dict(base, ci=[float("nan"), float("nan")],
                    n_degenerate_draws=degenerate, frac_degenerate=frac,
                    ci_usable=False, ci_note="every draw was degenerate")
    return dict(base, ci=[float(np.percentile(samples, 2.5)),
                          float(np.percentile(samples, 97.5))],
                n_degenerate_draws=degenerate, frac_degenerate=frac,
                n_valid_draws=len(samples), ci_usable=bool(frac <= 0.01),
                ci_note="degenerate draws (pooled x had no variation) dropped, "
                        "not counted as slope 0; decision post-dates first run")


def _verdict(res: dict, *, positive: bool) -> bool:
    """CI 排除 0；positive=True 时还要求为正。nan 一律不算通过。"""

    lo, hi = res["ci"]
    if any(np.isnan(v) for v in (res["slope_pp_per_unit"], lo, hi)):
        return False
    if not res.get("ci_usable", True):
        return False
    if positive:
        return lo > 0.0
    return lo > 0.0 or hi < 0.0


def r1_r2(panel: list[Obs], field: str) -> dict:
    """R1/R2：多任务 scope 上 Δ_id 对 m_live 的回归，单位 pp / 每多一个成员。"""

    pairs: dict[str, list[tuple[float, float]]] = {}
    for o in panel:
        if o.is_singleton:
            continue
        y = 100.0 * (o.delta_id_global if field == "global" else o.delta_id_scoped)
        pairs.setdefault(o.task, []).append((float(o.m_live), y))
    return slope_ci(pairs)


def r3(panel: list[Obs]) -> dict:
    """R3：单例 scope 上 stale 对 age 的回归，单位 pp / 每多一个后继任务。"""

    pairs: dict[str, list[tuple[float, float]]] = {}
    for o in panel:
        if not o.is_singleton or o.stale is None:
            continue
        pairs.setdefault(o.task, []).append((float(o.age), 100.0 * o.stale))
    return slope_ci(pairs)


def r4(panel: list[Obs]) -> dict:
    """R4：同一 y 换成对 pos(g) 回归。R4 通过 = pos 的斜率 CI 含 0。

    协议 §3：设计上 age 与 pos 强负相关，所以 R3 的斜率可能只是"早训的任务
    本来就更差"。这里把混淆量化，并附上实际的 corr(age, pos)。
    """

    pairs: dict[str, list[tuple[float, float]]] = {}
    ages, poss = [], []
    for o in panel:
        if not o.is_singleton or o.stale is None:
            continue
        pairs.setdefault(o.task, []).append((float(o.pos), 100.0 * o.stale))
        ages.append(o.age)
        poss.append(o.pos)
    res = slope_ci(pairs)
    if len(ages) >= 2 and np.std(ages) > 0 and np.std(poss) > 0:
        res["corr_age_pos"] = float(np.corrcoef(ages, poss)[0, 1])
    else:
        res["corr_age_pos"] = float("nan")
    return res


def r5(panel: list[Obs]) -> dict:
    """R5：单例 scope 上 Δ_id_scoped ≡ 0 精确成立。与 2K/2M/2P/2Q 同口径。"""

    viol = [{"task": o.task, "seed": o.seed, "stage": o.stage,
             "delta_id_scoped": o.delta_id_scoped}
            for o in panel if o.is_singleton and abs(o.delta_id_scoped) > 1e-12]
    n = sum(1 for o in panel if o.is_singleton)
    return {"n_singleton_obs": n, "violations": viol, "pass": not viol,
            "field": "R_orc - R_shr_scoped",
            "spec": "Delta_id_scoped == 0 exactly on singleton scopes"}


def design_coverage(panel: list[Obs]) -> dict:
    """协议 §4 的四条功效限制所依赖的设计信息，与结果量分开输出。"""

    by_age: dict[int, int] = {}
    for o in panel:
        if o.is_singleton and o.stale is not None:
            by_age[o.age] = by_age.get(o.age, 0) + 1
    m_seen: dict[str, set[int]] = {}
    for o in panel:
        if not o.is_singleton:
            m_seen.setdefault(o.scope, set()).add(o.m_live)
    return {
        "singleton_obs_by_age": {str(k): v for k, v in sorted(by_age.items())},
        "singleton_tasks": sorted({o.task for o in panel
                                   if o.is_singleton and o.stale is not None}),
        "train_position": {o.task: o.pos for o in panel},
        "m_live_values_by_multi_scope": {k: sorted(v) for k, v in m_seen.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                       help="存量 run 的 JSON 路径，只读")
    parser.add_argument("--label", default="unnamed", help="臂名，用于 R6")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    panel: list[Obs] = []
    for i, p in enumerate(args.runs, start=1):
        panel.extend(build_panel(Path(p), seed=i))

    res_r5 = r5(panel)
    out = {
        "protocol_sha": "038c238b22615e4fddbc4629feb9c8a9bcc9f70af7a5c0dcb5a6702843d530df",
        "label": args.label,
        "runs": list(args.runs),
        "n_obs": len(panel),
        "design": design_coverage(panel),
        "R5": res_r5,
    }

    if not res_r5["pass"]:
        # 协议 §5 outcome 5：R5 失败则全部作废，不报告其他判据。
        out["outcome"] = 5
        out["note"] = "R5 failed; every other number is void per protocol §3 R5"
    else:
        g = r1_r2(panel, "global")
        s = r1_r2(panel, "scoped")
        a = r3(panel)
        p = r4(panel)
        g["pass"] = _verdict(g, positive=True)
        s["significant"] = _verdict(s, positive=True)
        a["pass"] = _verdict(a, positive=True)
        p["pass"] = not _verdict(p, positive=False)   # R4 通过 = CI 含 0
        out["R1"] = dict(g, spec="Delta_id_global vs m_live on multi-task scopes; "
                                 "pass = slope CI excludes 0 on the positive side")
        out["R2"] = dict(s, spec="Delta_id_scoped vs m_live; predeclared expectation: "
                                 "not significant")
        out["R3"] = dict(a, spec="|R_sso1-R_orc| vs age on singleton scopes; "
                                 "pass = slope CI excludes 0 on the positive side")
        out["R4"] = dict(p, spec="same y vs pos(g); pass = pos slope CI CONTAINS 0")

        if a["pass"] and not p["pass"]:
            out["outcome"] = 2
        elif s["significant"]:
            out["outcome"] = 3
        elif g["pass"] and a["pass"] and p["pass"]:
            out["outcome"] = 1
        else:
            out["outcome"] = None
            out["note"] = ("no predeclared outcome fires verbatim; report each "
                           "criterion under its own name per §3")

    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
