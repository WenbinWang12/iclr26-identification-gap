"""Phase-2L 判决：把冻结协议 §3 的判据 A-D 写成脚本。

**为什么这个文件现在才存在。**  §A2 的 λ=1.0 判决是内联算的，没有落盘的实现，
所以那些数字无法被重算。λ=0.5 的确认运行(§A3 用修正门槛选出的臂)在
2026-08-31 00:48 就跑完了，却从未判决。本脚本补上实现，并对两次运行都求值，
使 §A2 的数字可被复算而不是只能被引用。

判据定义逐字来自 §3，不新增、不改阈值：

    deep_repair  = R_raw[BASE] − R_raw[none]              每个 (seen task, stage)
    shallow_head = max(R_shr_scoped, R_book(m)) − R_raw   臂内

A：BASE 的 shallow_head 中位数，任务作簇的 95% CI 排除 0。
B：shallow_head[BASE] 不显著低于 shallow_head[none]，报告配对差及 CI。
C：deep_repair 中位数 > 0 且 CI 排除 0，**限定在 > 8pp 遗忘层**。C 失败则 A、B
   不可解释。
D：单例 scope 上 Delta_id_scoped ≡ 0 精确成立。
门槛：末阶段后 R_raw 中位数 > 0.60。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.phase2j_offset_conflict.decide import cluster_bootstrap_ci

#: §3 的常数。C 的分层阈值 8pp、B 的 Phase-2K 参考值 0.0630、门槛 0.60。
DEEP_STRATUM_PP = 0.08
PHASE2K_SHALLOW_HEAD = 0.0630
GATE_MEDIAN_RAW = 0.60


def observations(run_paths: list[Path], m: int) -> dict:
    """按 (seed, stage position, task) 收 R_raw / shallow_head / peak。

    只收 scorable 且**非当前训练任务**的观测：判据问的是已见旧任务上的遗忘。
    """

    per_seed: list[dict] = []
    for path in run_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows: dict[tuple[int, str], dict] = {}
        peak: dict[str, float] = {}
        for stage in payload["stages"]:
            pos = int(stage["position"])
            trained = stage["trained_task"]
            for name, info in stage["tasks"].items():
                if not info.get("scorable"):
                    continue
                raw = float(info["R_raw"])
                if name == trained:
                    peak[name] = max(peak.get(name, raw), raw)
                    continue
                book = info.get("qoc", {}).get(str(m), {}).get("prototype", {})
                book_acc = float(book.get("accuracy", float("nan")))
                scoped = float(info["R_shr_scoped"])
                best = np.nanmax([scoped, book_acc])
                rows[(pos, name)] = {
                    "R_raw": raw,
                    "shallow_head": float(best) - raw,
                    "peak": peak.get(name, float("nan")),
                    "scope_size": int(info.get("scope_size", 0)),
                    "Delta_id_scoped": float(info["R_orc"]) - scoped,
                }
        per_seed.append({"rows": rows, "final": payload["stages"][-1], "args": payload["args"]})
    return {"per_seed": per_seed}


def by_task(values: list[tuple[str, float]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for name, value in values:
        if np.isfinite(value):
            out.setdefault(name, []).append(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", nargs="+", required=True, help="BASE 臂（含正则）的 seed JSON")
    parser.add_argument("--none", nargs="+", required=True, dest="none_arm", help="none 臂的 seed JSON")
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = observations([Path(p) for p in args.base], args.m)
    none = observations([Path(p) for p in args.none_arm], args.m)
    out: dict = {"m": args.m, "n_seeds_base": len(args.base), "n_seeds_none": len(args.none_arm),
                 "lambda": base["per_seed"][0]["args"].get("olora_lambda"),
                 "cl_method_base": base["per_seed"][0]["args"].get("cl_method")}

    # A：BASE 的 shallow_head 中位数，CI 排除 0。
    a_vals = [(name, r["shallow_head"]) for s in base["per_seed"]
              for (pos, name), r in s["rows"].items()]
    a_pt, a_lo, a_hi = cluster_bootstrap_ci(by_task(a_vals), draws=10000, seed=0,
                                            statistic=np.median)
    out["A"] = {"median": a_pt, "ci": [a_lo, a_hi], "n": len(a_vals),
                "passed": bool(a_lo > 0.0),
                "spec": "median shallow_head[BASE], CI excludes 0"}

    # B：配对差。配对在 (stage position, task) 上，跨 seed 取该格的均值后再配对，
    # 因为两臂的 seed 是同一组随机种子但不同轨迹，逐 seed 配对会引入伪配对。
    def cell_mean(pack: dict, key: str) -> dict[tuple[int, str], float]:
        acc: dict[tuple[int, str], list[float]] = {}
        for s in pack["per_seed"]:
            for cell, r in s["rows"].items():
                if np.isfinite(r[key]):
                    acc.setdefault(cell, []).append(r[key])
        return {c: float(np.mean(v)) for c, v in acc.items()}

    b_base, b_none = cell_mean(base, "shallow_head"), cell_mean(none, "shallow_head")
    shared = sorted(set(b_base) & set(b_none))
    b_vals = [(name, b_base[(pos, name)] - b_none[(pos, name)]) for pos, name in shared]
    b_pt, b_lo, b_hi = cluster_bootstrap_ci(by_task(b_vals), draws=10000, seed=0,
                                            statistic=np.median)
    out["B"] = {"paired_median": b_pt, "ci": [b_lo, b_hi], "n": len(b_vals),
                "cannibalised": bool(b_hi < 0.0),
                "passed": bool(not (b_hi < 0.0)),
                "phase2k_reference": PHASE2K_SHALLOW_HEAD,
                "spec": "shallow_head[BASE] not significantly below shallow_head[none]"}

    # C：deep_repair，限定在 none 臂遗忘 > 8pp 的层。遗忘 = 该任务峰值 − 当前 R_raw。
    r_base, r_none = cell_mean(base, "R_raw"), cell_mean(none, "R_raw")
    peak_none = {}
    for s in none["per_seed"]:
        for (pos, name), r in s["rows"].items():
            if np.isfinite(r["peak"]):
                peak_none.setdefault(name, []).append(r["peak"])
    peak_none = {k: float(np.mean(v)) for k, v in peak_none.items()}
    c_vals = []
    for pos, name in sorted(set(r_base) & set(r_none)):
        forgetting = peak_none.get(name, float("nan")) - r_none[(pos, name)]
        if np.isfinite(forgetting) and forgetting > DEEP_STRATUM_PP:
            c_vals.append((name, r_base[(pos, name)] - r_none[(pos, name)]))
    # §A3 第 3 条预先声明「MNLI 在每个臂下都遗忘 ~0.20，正则几乎碰不到它」。
    # 逐任务列出 deep_repair 与 none 臂遗忘量，让那条预测可被核对而非只被引用。
    by_name: dict[str, list[float]] = {}
    for name, value in c_vals:
        by_name.setdefault(name, []).append(value)
    out["C_per_task"] = {
        name: {"deep_repair_pp": 100.0 * float(np.mean(v)), "n": len(v),
               "none_forgetting_pp": 100.0 * float(peak_none.get(name, float("nan"))
                                                   - np.mean([r_none[(p, t)] for (p, t) in r_none
                                                              if t == name]))}
        for name, v in sorted(by_name.items())}

    c_pt, c_lo, c_hi = cluster_bootstrap_ci(by_task(c_vals), draws=10000, seed=0,
                                            statistic=np.median)
    out["C"] = {"median": c_pt, "ci": [c_lo, c_hi], "n": len(c_vals),
                "passed": bool(c_lo > 0.0),
                "stratum_pp": DEEP_STRATUM_PP,
                "spec": "median deep_repair > 0 with CI excluding 0, >8pp forgetting stratum"}

    # D：单例 scope 上 Delta_id_scoped 必须精确为 0。
    checked = violations = 0
    worst = 0.0
    for pack in (base, none):
        for s in pack["per_seed"]:
            for (pos, name), r in s["rows"].items():
                if r["scope_size"] == 1:
                    checked += 1
                    gap = abs(r["Delta_id_scoped"])
                    worst = max(worst, gap)
                    if gap > 1e-9:
                        violations += 1
    out["D"] = {"checked": checked, "violations": violations, "worst_abs_gap": worst,
                "passed": bool(violations == 0),
                "spec": "singleton scopes: Delta_id_scoped == 0 exactly"}

    # 门槛：末阶段 R_raw 中位数。
    finals = [float(i["R_raw"]) for s in base["per_seed"]
              for i in s["final"]["tasks"].values() if i.get("scorable")]
    med = float(np.median(finals)) if finals else float("nan")
    out["training_gate"] = {"median_R_raw": med, "n_obs": len(finals),
                            "threshold": GATE_MEDIAN_RAW, "passed": bool(med > GATE_MEDIAN_RAW)}

    # §5 的四个预先声明结局。C 失败 ⇒ 结局 3，A、B 不可解释。
    if not out["C"]["passed"]:
        out["outcome"] = 3
        out["verdict"] = ("outcome 3 — inconclusive on additivity: C failed, so A and B "
                          "are uninterpretable per §3")
    elif out["A"]["passed"] and out["B"]["passed"]:
        out["outcome"] = 1
        out["verdict"] = ("outcome 1 — additive: the shallow head survives a "
                          "representation-level fix")
    elif out["A"]["passed"]:
        out["outcome"] = 2
        out["verdict"] = "outcome 2 — A holds but B shows cannibalisation"
    else:
        out["outcome"] = 4
        out["verdict"] = "outcome 4 — C acted and the shallow head did not survive"

    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
