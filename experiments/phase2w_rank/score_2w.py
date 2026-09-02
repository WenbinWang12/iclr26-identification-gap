"""Phase-2W scorer: the exchange rate between LoRA rank and 18 floats.

Frozen before any rank other than 8 was trained. Reads run records only (never
logits, so this cannot be turned into a rule-fitting script) and delegates the
headline aggregation to the ALREADY-COMMITTED phase2k scorer, so the sweep is
comparable to the paper's +1.51 pp by construction.

Usage:
  python score_2w.py --runs r1=DIR r2=DIR ... [--ep4 r1=DIR ...] --out verdict_2w.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from experiments.phase2j_offset_conflict.decide import cluster_bootstrap_ci  # noqa: E402

# r=8 baseline recomputed by THIS scorer on the local records runs/phase2q_bpo
# (row median, task-clustered, 10000 draws, 279 rows, 12 tasks). NOT the paper's
# +1.51 pp: those records (runs/phase2k_qoc_converged) are absent from this repo and
# no estimator here reproduces that number. See protocol Amendment 1(b).
BASELINE_CI = (0.78, 3.65)
RANKS = (1, 2, 4, 8, 16, 32)
N_BOOT = 10000
SEED = 20260831


# run_qoc.py writes "qoc_seed{N}.json" with stages[].position and stages[].tasks{}.
# Globbing "record*.json" or reading stage["stage"] silently yields nothing / None,
# so both names are pinned by test_load_rows_matches_runner_filename.
RECORD_GLOB = "**/qoc_seed*.json"


def load_rows(run_dir: Path) -> list[dict]:
    """Every scorable (task, stage) row from one run, with the §4 fields only."""
    rows = []
    files = sorted(run_dir.glob(RECORD_GLOB)) or sorted(run_dir.glob("**/record*.json"))
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        for stage in rec.get("stages", []) if isinstance(rec, dict) else []:
            for task, e in (stage.get("tasks") or {}).items():
                if not e.get("scorable"):
                    continue
                rows.append(dict(
                    task=task,
                    stage=stage.get("position", stage.get("stage")),
                    scope_recoverable=float(e["scope_recoverable"]),
                    Delta_id_global=float(e["Delta_id_global"]),
                    R_raw=float(e["R_raw"]),
                    R_orc=float(e["R_orc"]),
                ))
    return rows


def by_task(rows: list[dict], field: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(r["task"], []).append(r[field])
    return out


def pp_ci(rows: list[dict], field: str = "scope_recoverable",
          draws: int = N_BOOT) -> tuple[float, float, float]:
    """Cluster bootstrap median in percentage points, via the committed phase2j helper."""
    point, lo, hi = cluster_bootstrap_ci(by_task(rows, field), draws=draws, seed=SEED)
    return 100 * point, 100 * lo, 100 * hi


def share(rows: list[dict]) -> float:
    """Identification share of recoverable loss: median over rows of Δ_id / (R_orc - R_raw)."""
    vals = [r["scope_recoverable"] / (r["R_orc"] - r["R_raw"])
            for r in rows if (r["R_orc"] - r["R_raw"]) > 1e-9]
    return float(np.median(vals)) if vals else float("nan")


def paired_delta(per_rank: dict[int, list[dict]], lo_r: int, hi_r: int,
                 draws: int = N_BOOT) -> dict:
    """Paired (task-clustered) difference Δ_id(hi) - Δ_id(lo), bootstrapped."""
    a, b = by_task(per_rank[lo_r], "scope_recoverable"), by_task(per_rank[hi_r], "scope_recoverable")
    common = sorted(set(a) & set(b))
    diffs = {t: [np.mean(b[t]) - np.mean(a[t])] for t in common}
    point, lo, hi = cluster_bootstrap_ci(diffs, draws=draws, seed=SEED)
    return {"point": 100 * point, "lo": 100 * lo, "hi": 100 * hi, "n_tasks": len(common)}


def _avg_ranks(v: list[float]) -> np.ndarray:
    """Ranks with ties averaged. Plain argsort-of-argsort assigns 0,1,2,... to tied
    values, which makes a perfectly flat curve look monotone (rho=1) instead of
    flat (rho=0) -- the sign error that would send a flat Δ_id curve to H_grow."""
    a = np.asarray(v, dtype=float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = _avg_ranks(x), _avg_ranks(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def decide(per_rank_seed: dict[int, dict[int, list[dict]]], ep4: dict[int, list[dict]] | None,
           draws: int = N_BOOT) -> dict:
    """per_rank_seed[rank][seed] -> rows.  Returns the frozen W1..W6 verdict."""
    ranks = [r for r in RANKS if r in per_rank_seed]
    pooled = {r: [row for rows in per_rank_seed[r].values() for row in rows] for r in ranks}

    curve = {}
    for r in ranks:
        pt, lo, hi = pp_ci(pooled[r], draws=draws)
        curve[r] = {"delta_id": pt, "lo": lo, "hi": hi,
                    "delta_id_global": pp_ci(pooled[r], "Delta_id_global", draws=draws)[0],
                    "R_raw": float(np.mean([x["R_raw"] for x in pooled[r]])),
                    "share": share(pooled[r]),
                    "n_rows": len(pooled[r])}

    w1 = {"criterion": "W1", "delta_id_at_r8": curve.get(8, {}).get("delta_id"),
          "baseline_ci": list(BASELINE_CI),
          "baseline_provenance": "runs/phase2q_bpo, this scorer, row median"}
    w1["pass"] = bool(w1["delta_id_at_r8"] is not None
                      and BASELINE_CI[0] <= w1["delta_id_at_r8"] <= BASELINE_CI[1])

    seeds = sorted({s for r in ranks for s in per_rank_seed[r]})
    rhos = []
    for s in seeds:
        xs = [r for r in ranks if s in per_rank_seed[r]]
        ys = [pp_ci(per_rank_seed[r][s], draws=draws)[0] for r in xs]
        if len(xs) >= 3:
            rhos.append(spearman([float(r) for r in xs], ys))
    med_rho = float(np.median(rhos)) if rhos else float("nan")

    w3 = paired_delta(pooled, ranks[0], ranks[-1], draws=draws) if len(ranks) >= 2 else {}
    w3.update({"criterion": "W3", "lo_rank": ranks[0], "hi_rank": ranks[-1]})

    # w3 always carries its criterion metadata, so bool(w3) is true even when
    # paired_delta was skipped for lack of two ranks. Guard on the CI itself.
    has_ci = "lo" in w3 and np.isfinite(w3.get("lo", np.nan)) and np.isfinite(w3.get("hi", np.nan))
    ci_excludes_zero = bool(has_ci and (w3["lo"] > 0 or w3["hi"] < 0))
    if not ci_excludes_zero or abs(med_rho) < 0.5:
        hyp = "H_flat"
    elif med_rho < 0:
        hyp = "H_shrink"
    else:
        hyp = "H_grow"
    w2 = {"criterion": "W2", "rho_per_seed": rhos, "median_rho": med_rho,
          "ci_excludes_zero": ci_excludes_zero, "hypothesis": hyp}

    shares = [curve[r]["share"] for r in ranks]
    w4 = {"criterion": "W4", "share_by_rank": {str(r): curve[r]["share"] for r in ranks},
          "share_falls": bool(len(shares) >= 2 and shares[-1] < shares[0]),
          "delta_id_falls": bool(len(ranks) >= 2 and curve[ranks[-1]]["delta_id"] < curve[ranks[0]]["delta_id"])}
    w4["shrink_claimable"] = bool(w4["share_falls"] and w4["delta_id_falls"])

    w5 = {"criterion": "W5", "note": "headline is the trend over all ranks; no rank selected post hoc",
          "ranks_reported": ranks}

    w6 = {"criterion": "W6", "ran": ep4 is not None}
    if ep4:
        er = [r for r in RANKS if r in ep4]
        ey = [pp_ci(ep4[r], draws=draws)[0] for r in er]
        w6.update({"ranks": er, "delta_id": {str(r): v for r, v in zip(er, ey)},
                   "rho": spearman([float(r) for r in er], ey)})
        w6["agrees"] = bool(not np.isnan(w6["rho"]) and not np.isnan(med_rho)
                            and (w6["rho"] * med_rho > 0 or abs(med_rho) < 0.5))

    if not w1["pass"]:
        outcome = "OUTCOME_4_r8_outside_published_CI_report_both_do_not_reconcile"
    elif hyp == "H_flat":
        outcome = "OUTCOME_1_rank_invariant_18_floats_never_substitutable_by_budget"
    elif hyp == "H_shrink":
        outcome = ("OUTCOME_2_rank_substitutes_report_exchange_rate"
                   if w4["shrink_claimable"] else "OUTCOME_1_level_fell_but_share_did_not_no_shrink_claim")
    else:
        outcome = "OUTCOME_3_more_capacity_more_identification_damage"

    return {"curve": {str(k): v for k, v in curve.items()}, "W1": w1, "W2": w2,
            "W3": w3, "W4": w4, "W5": w5, "W6": w6, "seeds": seeds, "outcome": outcome}


def parse_spec(specs: list[str]) -> dict[int, list[Path]]:
    out: dict[int, list[Path]] = {}
    for sp in specs or []:
        key, _, path = sp.partition("=")
        out.setdefault(int(key.lstrip("r")), []).append(Path(path))
    return out


def seed_of(run_dir: Path) -> int | None:
    """Seed parsed from the sweep's own directory name (w_rank[TAG]_r{R}_s{S}) or from
    the record filename (qoc_seed{S}.json).

    Enumerating the dirs instead would silently mislabel seeds whenever one run of a
    rank is missing: that rank's 'seed 0' would be a different actual seed than every
    other rank's, and W2 pairs the per-seed curves by that label.
    """
    m = re.search(r"_s(\d+)$", run_dir.name)
    if m:
        return int(m.group(1))
    for f in sorted(run_dir.glob(RECORD_GLOB)):
        m = re.search(r"qoc_seed(\d+)\.json$", f.name)
        if m:
            return int(m.group(1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="rN=DIR (repeatable per seed)")
    ap.add_argument("--ep4", nargs="*", default=None, help="rN=DIR for the epochs=4 arm")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    per_rank_seed: dict[int, dict[int, list[dict]]] = {}
    for rank, dirs in parse_spec(a.runs).items():
        for i, d in enumerate(dirs):
            rows = load_rows(d)
            if not rows:
                print("WARN no scorable rows in %s" % d, file=sys.stderr)
                continue
            sd = seed_of(d)
            if sd is None:
                sd = -(i + 1)
                print("WARN could not parse seed from %s; labelled %d" % (d, sd),
                      file=sys.stderr)
            per_rank_seed.setdefault(rank, {})[sd] = rows
    ep4 = {r: [x for d in dirs for x in load_rows(d)] for r, dirs in parse_spec(a.ep4).items()} if a.ep4 else None

    v = decide(per_rank_seed, ep4)
    Path(a.out).write_text(json.dumps(v, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(v, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
