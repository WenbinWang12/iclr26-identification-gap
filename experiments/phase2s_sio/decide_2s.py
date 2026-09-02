"""Phase-2S judgement: score S1-S6 (protocol §4) plus S3b/S5b (§A2.3, §A2.4).

Written and committed BEFORE the real-data run returned, so no threshold here can
have been chosen against an Order-4 number. The protocol froze the thresholds; this
file only implements them.

Statistics are the same ones every prior phase used: `cluster_bootstrap_ci` with
clusters = tasks, 10000 draws, seed 0, median statistic. Nothing new is introduced
at judgement time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.phase2j_offset_conflict.decide import cluster_bootstrap_ci
from experiments.phase2s_sio.sio import marginal_slope

PROTOCOL_SHA = "21e8963e3ca774f93cc13bf51b438027ec9f9cf7b30f0d23869f2a1cbe52e604"
AMENDMENT_SHA = "63015b1e4c7e290e51071b84a6b7196baae082ec578f099368d9ddee04b90d5d"


def load_records(paths):
    """Yield (seed, stage_position, task, info) for every scored (task, stage)."""
    for p in paths:
        rec = json.loads(Path(p).read_text(encoding="utf-8"))
        seed = rec.get("args", {}).get("seed")
        for stage in rec["stages"]:
            for task, info in stage["tasks"].items():
                yield seed, stage["position"], task, info


def by_cluster(rows, value_fn):
    """Group values by task name -- tasks are the clusters, as in every prior phase."""
    out: dict[str, list[float]] = {}
    for _seed, _pos, task, info in rows:
        v = value_fn(info)
        if v is None or not np.isfinite(v):
            continue
        out.setdefault(task, []).append(float(v))
    return out


def _ci(clusters):
    if not clusters:
        return {"median": None, "ci": None, "n_clusters": 0, "n": 0}
    flat = [v for vs in clusters.values() for v in vs]
    # 返回三元组 (point, lo, hi)；点估计用它自己的，不用我另算的 median，
    # 免得两个统计量在报告里并存而读者不知道哪个进了判据。
    point, lo, hi = cluster_bootstrap_ci(clusters, draws=10000, seed=0,
                                        statistic=np.median)
    return {"median": float(point), "ci": [float(lo), float(hi)],
            "n_clusters": len(clusters), "n": len(flat)}


def _pass(res, *, threshold, side):
    """side='above': median >= threshold and CI strictly above 0.
       side='not_below': median >= threshold and CI lower bound >= 0."""
    if res["median"] is None or res["ci"] is None:
        return False
    if res["median"] < threshold:
        return False
    lo = res["ci"][0]
    return lo > 0.0 if side == "above" else lo >= 0.0


def applicable(rows):
    return [r for r in rows if (r[3].get("sio") or {}).get("applicable")]


def s1(rows):
    """SIO - raw >= +1.0 pp, CI strictly above 0."""
    cl = by_cluster(rows, lambda i: 100.0 * (i["sio"]["R_sio"] - i["R_raw"]))
    res = _ci(cl)
    res["threshold_pp"] = 1.0
    res["verdict"] = _pass(res, threshold=1.0, side="above")
    return res


def s2(rows):
    """Balanced audit batches: SIO - BC >= 0, CI not below 0."""
    cl = by_cluster(rows, lambda i: 100.0 * (i["sio"]["R_sio"] - i["sio"]["R_bc"]))
    res = _ci(cl)
    res["threshold_pp"] = 0.0
    res["verdict"] = _pass(res, threshold=0.0, side="not_below")
    return res


def s3(rows, ratio):
    """Skewed mixes: SIO - BC >= +1.0 AND SIO - raw >= 0, both by CI."""
    def d_bc(i):
        e = i["sio"].get("imbalanced", {}).get(ratio)
        return None if e is None else 100.0 * (e["R_sio"] - e["R_bc"])

    def d_raw(i):
        e = i["sio"].get("imbalanced", {}).get(ratio)
        return None if e is None else 100.0 * (e["R_sio"] - e["R_raw"])

    vs_bc, vs_raw = _ci(by_cluster(rows, d_bc)), _ci(by_cluster(rows, d_raw))
    return {"ratio": ratio, "vs_bc": vs_bc, "vs_raw": vs_raw,
            "verdict": (_pass(vs_bc, threshold=1.0, side="above")
                        and _pass(vs_raw, threshold=0.0, side="not_below"))}


def s3b(rows, ratio):
    """§A2.3: S3 may pass only on batches where the gate APPLIES.

    Abstention returns raw, so on a mix where BC loses, `SIO - BC` goes positive
    without SIO doing anything. S3 cannot tell "fixed the defect" from "declined
    it"; this can.
    """
    applied_rows, flags = [], []
    for r in rows:
        e = r[3]["sio"].get("imbalanced", {}).get(ratio)
        if e is None:
            continue
        flags.append(bool(e["applied"]))
        if e["applied"]:
            applied_rows.append(r)

    def d_bc(i):
        e = i["sio"]["imbalanced"][ratio]
        return 100.0 * (e["R_sio"] - e["R_bc"]) if e["applied"] else None

    def d_raw(i):
        e = i["sio"]["imbalanced"][ratio]
        return 100.0 * (e["R_sio"] - e["R_raw"]) if e["applied"] else None

    rate = float(np.mean(flags)) if flags else 0.0
    vs_bc = _ci(by_cluster(applied_rows, d_bc))
    vs_raw = _ci(by_cluster(applied_rows, d_raw))
    return {"ratio": ratio, "apply_rate": rate, "n_batches": len(flags),
            "vs_bc_applied_only": vs_bc, "vs_raw_applied_only": vs_raw,
            "conclusion_wording": ("SIO abstains on most skewed batches"
                                   if rate < 0.5 else "SIO acts on most batches"),
            "verdict": (rate >= 0.5
                        and _pass(vs_bc, threshold=1.0, side="above")
                        and _pass(vs_raw, threshold=0.0, side="not_below"))}


def s4(rows):
    """Gate protects: abstention is exactly raw, and it does not abstain always."""
    exact, viol, bal_flags = 0, [], []
    for _s, _p, task, info in rows:
        sio = info["sio"]
        bal_flags.append(bool(sio["applied"]))
        if not sio["applied"]:
            if sio["R_sio"] == info["R_raw"]:
                exact += 1
            else:
                viol.append({"task": task, "R_sio": sio["R_sio"],
                             "R_raw": info["R_raw"]})
    rate = float(np.mean(bal_flags)) if bal_flags else 0.0
    return {"abstentions": exact + len(viol), "exactly_raw": exact,
            "violations": viol, "abstain_rate_balanced": 1.0 - rate,
            "verdict": (not viol) and (1.0 - rate) <= 0.5}


def s5(rows):
    """S5 as frozen: sd(tau) <= sd(BC)/3 across mixes. Reported even though §A2.4
    shows it is the wrong statistic -- its own data is what shows that."""
    ratios, taus, bcs = [], [], []
    for _s, _p, _t, info in rows:
        for si in info["sio"].get("slope_inputs", []):
            if si.get("tau") is None:
                continue
            ratios.append(si["ratio"])
            taus.append(si["tau"])
            bcs.append(si["bc"])
    if len(taus) < 3:
        return {"verdict": False, "reason": "too few paired points"}
    sd_t, sd_b = float(np.std(taus)), float(np.std(bcs))
    return {"sd_tau": sd_t, "sd_bc": sd_b,
            "ratio": sd_t / sd_b if sd_b > 0 else None,
            "verdict": sd_b > 0 and sd_t <= sd_b / 3.0}


def s5b(rows):
    """§A2.4: the mechanism test. SIO's slope CI must contain 0; BC's must not."""
    per_task_tau: dict[str, tuple[list, list]] = {}
    per_task_bc: dict[str, tuple[list, list]] = {}
    for _s, _p, task, info in rows:
        for si in info["sio"].get("slope_inputs", []):
            if si.get("tau") is None:
                continue
            per_task_tau.setdefault(task, ([], []))
            per_task_tau[task][0].append(si["ratio"])
            per_task_tau[task][1].append(si["tau"])
            per_task_bc.setdefault(task, ([], []))
            per_task_bc[task][0].append(si["ratio"])
            per_task_bc[task][1].append(si["bc"])

    def slopes(d):
        out = {}
        for task, (xs, ys) in d.items():
            if len(set(xs)) < 2:
                continue
            s = marginal_slope(np.array(ys), np.array(xs))
            if np.isfinite(s):
                out[task] = [float(s)]
        return out

    st, sb = slopes(per_task_tau), slopes(per_task_bc)
    rt, rb = _ci(st), _ci(sb)
    sio_contains_0 = (rt["ci"] is not None and rt["ci"][0] <= 0 <= rt["ci"][1])
    bc_excludes_0 = (rb["ci"] is not None
                     and (rb["ci"][0] > 0 or rb["ci"][1] < 0))
    return {"sio_slope": rt, "bc_slope": rb,
            "sio_ci_contains_zero": sio_contains_0,
            "bc_ci_excludes_zero": bc_excludes_0,
            "verdict": sio_contains_0 and bc_excludes_0}


def s6(all_rows, app_rows):
    """Coverage: what fraction of (task, stage) pairs K_S=2 removes."""
    n_all, n_app = len(all_rows), len(app_rows)
    ks = {}
    for _s, _p, _t, info in all_rows:
        k = (info.get("sio") or {}).get("K_S")
        if k is not None:
            ks[str(k)] = ks.get(str(k), 0) + 1
    return {"n_task_stage_pairs": n_all, "n_with_K_S_2": n_app,
            "coverage": (n_app / n_all) if n_all else 0.0,
            "K_S_histogram": ks,
            "verdict": n_app > 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="qoc_seed*.json files")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    all_rows = list(load_records(a.runs))
    rows = applicable(all_rows)

    verdict = {
        "protocol_sha": PROTOCOL_SHA,
        "amendment_sha": AMENDMENT_SHA,
        "runs": list(a.runs),
        "S1_beats_doing_nothing": s1(rows),
        "S2_beats_BC_on_balanced": s2(rows),
        "S3_skewed": {r: s3(rows, r) for r in ("0.7", "0.9")},
        "S3b_skewed_applied_only": {r: s3b(rows, r) for r in ("0.7", "0.9")},
        "S4_gate_protects": s4(rows),
        "S5_frozen_sd_form": s5(rows),
        "S5b_slope_form": s5b(rows),
        "S6_coverage": s6(all_rows, rows),
    }

    # Predeclared outcomes, protocol §5. Evaluated mechanically, in order.
    p1 = verdict["S1_beats_doing_nothing"]["verdict"]
    p2 = verdict["S2_beats_BC_on_balanced"]["verdict"]
    p3 = any(v["verdict"] for v in verdict["S3_skewed"].values())
    p3b = any(v["verdict"] for v in verdict["S3b_skewed_applied_only"].values())
    p4 = verdict["S4_gate_protects"]["verdict"]
    p5b = verdict["S5b_slope_form"]["verdict"]

    if p1 and p2 and p3 and p3b and p4 and p5b:
        outcome = "1_working_rule_fixing_a_published_defect"
    elif p1 and p3 and not p5b:
        outcome = "2_gain_real_mechanism_unconfirmed"
    elif p1 and not p3:
        outcome = "3_another_balanced_split_artefact"
    elif not p1 and p4:
        outcome = "4_gate_works_rule_adds_nothing"
    elif verdict["S6_coverage"]["n_with_K_S_2"] == 0:
        outcome = "5_underpowered_no_K_S_2_coverage"
    else:
        outcome = "PROTOCOL_GAP_record_in_amendment"
    verdict["outcome"] = outcome
    verdict["outcome_note"] = (
        "S3b is required for outcome 1 even though §4 as frozen did not list it: "
        "§A2.3 added it because S3 alone scores abstention as success."
    )

    text = json.dumps(verdict, indent=2, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
