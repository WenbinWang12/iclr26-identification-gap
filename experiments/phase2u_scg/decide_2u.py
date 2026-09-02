"""Mechanical scorer for Phase-2U criteria U1-U7.

Committed BEFORE seeds 11-15 finished, so the outcome is a function of the frozen
criteria and not of the numbers.  Same shape as decide_2s.py / decide_2t.py.

Usage:
    python decide_2u.py --dev D1 D2 D3 --heldout H11 ... H15
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
from phase2u_scg.scg import (  # noqa: E402
    TAU_M, balanced_accuracy, bc_offset, margin, oracle_offset, scg_offset,
)

PROTOCOL = "notes/phase2u_scg_protocol.md"
PROTOCOL_SHA = "892ef1a8bb68bf1d044da031ae092fca73122c8c7ba77d2c8bd79486669e4426"

U1_WORST_FLOOR = -1.0        # pp, pooled AND per seed
U2_MEAN_MIN = 2.0            # pp, CI on the mean must exclude 0
U3_COST_FLOOR = -3.5         # pp vs plain BC
U4_APPLY_LO, U4_APPLY_HI = 0.10, 0.60
U5_RHO_MEDIAN_MIN = 0.25
U5_MIN_POSITIVE = 4
U5_MIN_BATCHES = 6


def parse_name(name: str) -> tuple[str, str, str]:
    m = re.fullmatch(r"seed(\d+)_stage(\d+)_(.+)\.npz", name)
    if not m:
        raise ValueError(f"unrecognised dump filename: {name}")
    return m.group(1), m.group(2), m.group(3)


def load_batches(dirs) -> list[dict]:
    out = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            Z = np.asarray(z["logits"], dtype=np.float64)
            if Z.ndim != 2 or Z.shape[1] != 2:
                continue
            y = np.asarray(z["labels"], dtype=int)
            if len(np.unique(y)) < 2:
                continue
            seed, stage, task = parse_name(f.name)
            out.append({"task": task, "seed": seed, "stage": stage,
                        "file": f.name, "Z": Z, "y": y})
    return out


def score_batch(b: dict) -> dict:
    Z, y = b["Z"], b["y"]
    raw = balanced_accuracy(Z, y)
    r = scg_offset(Z)
    s = margin(Z)
    return {"task": b["task"], "seed": b["seed"], "stage": b["stage"],
            "n": int(len(y)), "abs_m": r.abs_m, "sd": float(s.std(ddof=1)),
            "applied": bool(r.applied), "raw": raw,
            "bc": balanced_accuracy(Z, y, bc_offset(Z)),
            "scg": balanced_accuracy(Z, y, r.offset),
            "oracle": balanced_accuracy(Z, y, oracle_offset(Z, y))}


def _pp(x) -> float:
    return 100.0 * float(x)


def _group(rows, key="task") -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return out


def _ci(values_by_cluster, statistic=np.median):
    point, lo, hi = cluster_bootstrap_ci(values_by_cluster, draws=10000, seed=0,
                                        statistic=statistic)
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def _by_cluster(rows, key):
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(r["task"], []).append(key(r))
    return out


def u1(rows) -> dict:
    """The tail, pooled and per seed.  2T's threshold passed pooled then broke."""
    g = [_pp(r["scg"] - r["raw"]) for r in rows]
    gb = [_pp(r["bc"] - r["raw"]) for r in rows]
    per_seed = {s: min(_pp(r["scg"] - r["raw"]) for r in grp)
                for s, grp in _group(rows, "seed").items()}
    ok = min(g) >= U1_WORST_FLOOR and all(v >= U1_WORST_FLOOR
                                          for v in per_seed.values())
    return {"criterion": "U1", "worst_scg": min(g), "worst_bc": min(gb),
            "worst_by_seed": per_seed,
            "harmful_scg": sum(1 for v in g if v < -1.0),
            "harmful_bc": sum(1 for v in gb if v < -1.0),
            "n": len(rows), "threshold": U1_WORST_FLOOR, "pass": ok}


def u2(rows) -> dict:
    """Stated on the mean: §2 declares the median is 0 by construction."""
    ci = _ci(_by_cluster(rows, lambda r: _pp(r["scg"] - r["raw"])), np.mean)
    raw_mean = float(np.mean([_pp(r["scg"] - r["raw"]) for r in rows]))
    return {"criterion": "U2", **ci, "batch_mean": raw_mean,
            "threshold": U2_MEAN_MIN,
            "pass": ci["point"] >= U2_MEAN_MIN and ci["lo"] > 0.0}


def u3(rows) -> dict:
    ci = _ci(_by_cluster(rows, lambda r: _pp(r["scg"] - r["bc"])), np.mean)
    return {"criterion": "U3", **ci, "threshold": U3_COST_FLOOR,
            "pass": ci["point"] >= U3_COST_FLOOR}


def u4(rows) -> dict:
    rate = float(np.mean([r["applied"] for r in rows])) if rows else 0.0
    return {"criterion": "U4", "apply_rate": rate,
            "band": [U4_APPLY_LO, U4_APPLY_HI],
            "pass": U4_APPLY_LO <= rate <= U4_APPLY_HI}


def u5(rows) -> dict:
    """Within-task rho between |mean(s)| and BC's gain.  The mechanism criterion."""
    per_task = {}
    for task, grp in _group(rows).items():
        if len(grp) < U5_MIN_BATCHES:
            continue
        rho = _spearman([r["abs_m"] for r in grp],
                        [r["bc"] - r["raw"] for r in grp])
        if not np.isnan(rho):
            per_task[task] = float(rho)
    if not per_task:
        return {"criterion": "U5", "per_task": {}, "median": None, "pass": False,
                "note": "no task had enough held-out batches"}
    med = float(np.median(list(per_task.values())))
    npos = sum(1 for v in per_task.values() if v > 0)
    return {"criterion": "U5", "per_task": per_task, "median": med,
            "n_positive": npos, "n_tasks": len(per_task),
            "median_threshold": U5_RHO_MEDIAN_MIN,
            "min_positive": U5_MIN_POSITIVE,
            "pass": med > U5_RHO_MEDIAN_MIN and npos >= U5_MIN_POSITIVE}


def u6(rows, dev_rows) -> dict:
    """Is it the magnitude or the scale?  Reported, no pass/fail."""
    def sweep(rr, key, taus):
        best = None
        for t in taus:
            g = [_pp(r["scg_raw_gain"]) if r[key] >= t else 0.0 for r in rr]
            harmful = sum(1 for v in g if v < -1.0)
            rec = {"tau": t, "mean": float(np.mean(g)), "worst": min(g),
                   "harmful": harmful,
                   "apply_rate": float(np.mean([r[key] >= t for r in rr]))}
            if harmful == 0 and (best is None or rec["mean"] > best["mean"]):
                best = rec
        return best

    for r in rows + dev_rows:
        r["scg_raw_gain"] = r["bc"] - r["raw"]      # gain if the gate applies
        r["norm"] = r["abs_m"] / r["sd"] if r["sd"] > 0 else 0.0
    return {"criterion": "U6",
            "heldout_best_absm": sweep(rows, "abs_m",
                                       [round(0.2 * i, 1) for i in range(1, 16)]),
            "heldout_best_norm": sweep(rows, "norm",
                                       [round(0.1 * i, 1) for i in range(1, 31)]),
            "dev_best_absm": sweep(dev_rows, "abs_m",
                                   [round(0.2 * i, 1) for i in range(1, 16)]),
            "dev_best_norm": sweep(dev_rows, "norm",
                                   [round(0.1 * i, 1) for i in range(1, 31)])}


def u7(rows) -> dict:
    per = {t: float(np.mean([r["applied"] for r in g]))
           for t, g in _group(rows).items()}
    pure = sum(1 for v in per.values() if v in (0.0, 1.0))
    return {"criterion": "U7", "apply_rate_by_task": per,
            "n_tasks": len(per), "n_all_or_nothing": pure,
            "is_task_filter": pure == len(per)}


def _spearman(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, dtype=np.float64)
    r[order] = np.arange(1, x.size + 1, dtype=np.float64)
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for i in np.flatnonzero(counts > 1):
        m = inv == i
        r[m] = r[m].mean()
    return r


def decide(res: dict) -> str:
    """Mechanical branch over §4's five predeclared outcomes."""
    U = res
    if not U["U1"]["pass"]:
        return "OUTCOME_4_threshold_did_not_transfer_stop_pursuing_gated_bc"
    if not U["U2"]["pass"]:
        return "OUTCOME_3_gate_is_safe_and_pointless"
    # U1 and U2 both pass from here.
    nb = U["U6"].get("heldout_best_norm")
    ab = U["U6"].get("heldout_best_absm")
    norm_at_least_as_good = bool(nb and ab and nb["mean"] >= ab["mean"])
    if norm_at_least_as_good:
        return "OUTCOME_5_prefer_the_normalised_statistic"
    if U["U4"]["pass"] and U["U5"]["pass"] and U["U3"]["pass"]:
        return "OUTCOME_1_working_and_mechanistically_explained"
    if U["U4"]["pass"] and not U["U5"]["pass"]:
        return "OUTCOME_2_works_but_the_mechanism_is_not_established"
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

    res = {"protocol": PROTOCOL, "protocol_sha": PROTOCOL_SHA, "tau_m": TAU_M,
           "n_dev": len(dev), "n_heldout": len(hel),
           "heldout_seeds": sorted({r["seed"] for r in hel}),
           "U1": u1(hel), "U2": u2(hel), "U3": u3(hel), "U4": u4(hel),
           "U5": u5(hel), "U6": u6(hel, dev), "U7": u7(hel),
           "heldout_means": {k: float(np.mean([_pp(r[k] - r["raw"]) for r in hel]))
                             for k in ("bc", "scg", "oracle")},
           "dev_summary": {"U1": u1(dev), "U4": u4(dev), "U5": u5(dev)} if dev else None}
    res["outcome"] = decide(res)
    text = json.dumps(res, indent=2, sort_keys=True, default=float)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
