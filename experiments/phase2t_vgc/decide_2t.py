"""Mechanical scorer for Phase-2T criteria T1-T6.

Written and committed BEFORE seeds 2 and 3 were read.  Outcome selection is a
mechanical branch over the criteria, exactly as in decide_2s.py, so the verdict is
not a judgement call made after seeing the numbers.

Usage:
    python decide_2t.py --dev DEVDIR --heldout DIR2 DIR3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase2j_offset_conflict.decide import cluster_bootstrap_ci  # noqa: E402
from phase2t_vgc.vgc import (  # noqa: E402
    TAU_SD, balanced_accuracy, bc_offset, margin, ols_slope, oracle_offset,
    spearman, vgc_offset,
)

PROTOCOL = "notes/phase2t_vgc_protocol.md"
PROTOCOL_SHA = "7697291743d6a64be7dcef29a70f3d2adcf8d102e6bc004628510a08bb87a849"

T1_WORST_FLOOR = -3.0     # pp
T2_MEDIAN_MIN = 1.0       # pp, CI must also exclude 0
T3_COST_FLOOR = -2.0      # pp vs plain BC
T4_APPLY_LO, T4_APPLY_HI = 0.3, 0.9
T5_MIN_BATCHES = 6


def load_batches(dirs) -> list[dict]:
    """Every stored K_S == 2 batch, with its task label for clustering."""
    out = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            Z = np.asarray(z["logits"], dtype=np.float64)
            if Z.ndim != 2 or Z.shape[1] != 2:
                continue
            y = np.asarray(z["labels"], dtype=int)
            if len(np.unique(y)) < 2:
                continue          # balanced accuracy is undefined on one class
            seed, stage, task = parse_name(f.name)
            out.append({"task": task, "seed": seed, "stage": stage,
                        "file": f.name, "Z": Z, "y": y})
    return out


def parse_name(name: str) -> tuple[str, str, str]:
    """``seed{S}_stage{P}_{TASK}.npz`` -> (S, P, TASK).

    The task is everything after the stage field, so a task name containing an
    underscore would not be silently truncated.
    """
    m = re.fullmatch(r"seed(\d+)_stage(\d+)_(.+)\.npz", name)
    if not m:
        raise ValueError(f"unrecognised dump filename: {name}")
    return m.group(1), m.group(2), m.group(3)


def score_batch(b: dict) -> dict:
    Z, y = b["Z"], b["y"]
    raw = balanced_accuracy(Z, y)
    res = vgc_offset(Z)
    return {
        "task": b["task"], "seed": b["seed"], "file": b["file"], "n": int(len(y)),
        "sd": res.sd, "applied": bool(res.applied),
        "raw": raw,
        "bc": balanced_accuracy(Z, y, bc_offset(Z)),
        "vgc": balanced_accuracy(Z, y, res.offset),
        "oracle": balanced_accuracy(Z, y, oracle_offset(Z, y)),
    }


def _pp(x) -> float:
    return 100.0 * float(x)


def by_cluster(rows, key) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(r["task"], []).append(key(r))
    return out


def _ci(values_by_cluster):
    point, lo, hi = cluster_bootstrap_ci(values_by_cluster, draws=10000, seed=0,
                                        statistic=np.median)
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def t1(rows) -> dict:
    """Worst per-batch VGC gain over raw, against BC's worst on the same batches."""
    vg = [_pp(r["vgc"] - r["raw"]) for r in rows]
    bg = [_pp(r["bc"] - r["raw"]) for r in rows]
    worst_v, worst_b = min(vg), min(bg)
    harm_v = sum(1 for v in vg if v < -1.0)
    harm_b = sum(1 for v in bg if v < -1.0)
    return {"criterion": "T1", "worst_vgc": worst_v, "worst_bc": worst_b,
            "harmful_vgc": harm_v, "harmful_bc": harm_b, "n": len(rows),
            "threshold": T1_WORST_FLOOR, "pass": worst_v >= T1_WORST_FLOOR}


def t2(rows) -> dict:
    ci = _ci(by_cluster(rows, lambda r: _pp(r["vgc"] - r["raw"])))
    return {"criterion": "T2", **ci, "threshold": T2_MEDIAN_MIN,
            "pass": ci["point"] >= T2_MEDIAN_MIN and ci["lo"] > 0.0}


def t3(rows) -> dict:
    ci = _ci(by_cluster(rows, lambda r: _pp(r["vgc"] - r["bc"])))
    return {"criterion": "T3", **ci, "threshold": T3_COST_FLOOR,
            "pass": ci["point"] >= T3_COST_FLOOR}


def t4(rows) -> dict:
    rate = float(np.mean([r["applied"] for r in rows])) if rows else 0.0
    return {"criterion": "T4", "apply_rate": rate,
            "band": [T4_APPLY_LO, T4_APPLY_HI],
            "pass": T4_APPLY_LO <= rate <= T4_APPLY_HI}


def t5(rows) -> dict:
    """Within-task Spearman(sd, BC gain).  The decisive mechanism criterion."""
    per_task = {}
    for task, grp in _group(rows).items():
        if len(grp) < T5_MIN_BATCHES:
            continue
        rho = spearman([r["sd"] for r in grp],
                       [r["bc"] - r["raw"] for r in grp])
        if not np.isnan(rho):
            per_task[task] = float(rho)
    if not per_task:
        return {"criterion": "T5", "per_task": {}, "median": None,
                "pass": False, "note": "no task had enough held-out batches"}
    med = float(np.median(list(per_task.values())))
    n_pos = sum(1 for v in per_task.values() if v > 0)
    return {"criterion": "T5", "per_task": per_task, "median": med,
            "n_positive": n_pos, "n_tasks": len(per_task),
            "min_batches": T5_MIN_BATCHES,
            "pass": med > 0.0 and n_pos == len(per_task)}


def t6(rows) -> dict:
    """How much of BC's tail the gate can even see.  Reported, no threshold."""
    harmful = [r for r in rows if _pp(r["bc"] - r["raw"]) < -1.0]
    if not harmful:
        return {"criterion": "T6", "n_harmful": 0, "caught_fraction": None,
                "note": "BC was never harmful on held-out batches"}
    caught = sum(1 for r in harmful if r["sd"] < TAU_SD)
    return {"criterion": "T6", "n_harmful": len(harmful), "n_caught": caught,
            "caught_fraction": caught / len(harmful),
            "harmful_tasks": sorted({r["task"] for r in harmful})}


def _group(rows) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["task"], []).append(r)
    return out


def task_identity_overlap(rows) -> dict:
    """§2's confound check: does the gate's decision reduce to the task name?"""
    per_task = {}
    for task, grp in _group(rows).items():
        per_task[task] = float(np.mean([r["applied"] for r in grp]))
    pure = sum(1 for v in per_task.values() if v in (0.0, 1.0))
    return {"apply_rate_by_task": per_task, "n_tasks": len(per_task),
            "n_all_or_nothing": pure,
            "is_task_filter": pure == len(per_task)}


def decide(res: dict) -> str:
    """Mechanical branch over §4's five predeclared outcomes."""
    T = {k: res[k] for k in ("T1", "T2", "T3", "T4", "T5")}
    if not T["T2"]["pass"]:
        return "OUTCOME_5_gate_removes_the_gain_with_the_tail"
    if T["T1"]["pass"] and T["T4"]["pass"] and T["T5"]["pass"] and T["T3"]["pass"]:
        return "OUTCOME_1_working_fix_to_a_documented_defect"
    if T["T1"]["pass"] and T["T4"]["pass"] and not T["T5"]["pass"]:
        if res["task_identity"]["is_task_filter"]:
            return "OUTCOME_4_task_filter_not_a_calibration_gate"
        return "OUTCOME_2_works_but_the_selector_is_unexplained"
    if not T["T1"]["pass"] and res.get("dev_T1_pass"):
        return "OUTCOME_3_threshold_was_fitted_to_seed_1"
    return "PROTOCOL_GAP_record_in_amendment"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--heldout", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = [score_batch(b) for b in load_batches(a.dev)]
    hel = [score_batch(b) for b in load_batches(a.heldout)]
    if not hel:
        print("no held-out batches found", file=sys.stderr)
        return 2

    res = {
        "protocol": PROTOCOL, "protocol_sha": PROTOCOL_SHA, "tau_sd": TAU_SD,
        "n_dev": len(dev), "n_heldout": len(hel),
        "T1": t1(hel), "T2": t2(hel), "T3": t3(hel),
        "T4": t4(hel), "T5": t5(hel), "T6": t6(hel),
        "task_identity": task_identity_overlap(hel),
        "dev_T1_pass": t1(dev)["pass"] if dev else None,
        "dev_summary": {
            "T1": t1(dev), "T4": t4(dev), "T5": t5(dev),
        } if dev else None,
        "heldout_means": {
            k: _pp(np.mean([r[k] - r["raw"] for r in hel]))
            for k in ("bc", "vgc", "oracle")
        },
    }
    res["outcome"] = decide(res)
    text = json.dumps(res, indent=2, sort_keys=True)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
