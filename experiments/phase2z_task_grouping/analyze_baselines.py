"""Analyze RUNBOOK-3D results: the method vs published baselines at MATCHED budget.

Reads baselines_{method}_s{seed}.json (from run_baselines.py) and, when present,
the method's own combined 2x2 results_s{seed}.json (from run_combined.py) so the
proposed method's deployable and ceiling cells appear in the same table.

For each method, mean audit balanced accuracy over the SAME eligible tasks, per
seed and aggregate. The comparison is fair only because every run shares the
harness (stream, splits, seeds, budget, scorer); this is asserted by reading the
same task set from each JSON and refusing to compare across differing task sets.

The method columns pulled from results_s{seed}.json:
  method_gp_off = mean R_gp_off  (grouping + per-scope offset; ORACLE routing +
                  REFIT offset -- both upper bounds, labelled as such)
  method_sh     = mean R_sh      (shared no-offset; the naive-budget baseline the
                  paper's headline is measured against)
Baselines (all deployable, single rank-8 LoRA): seqft, olora, ewc.

No leaderboard, no test.json. A missing method is reported missing, never imputed.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINES = ["seqft", "olora", "ewc"]


def mean(xs):
    return sum(xs) / len(xs)


def load_baseline(method, seed):
    p = HERE / f"baselines_{method}_s{seed}.json"
    if not p.exists():
        return None
    payload = json.loads(p.read_text())
    return {r["task"]: r["R"] for r in payload["rows"]}


def load_method(seed):
    """Return per-task {task: (R_sh, R_gp_off)} from the combined 2x2, or None."""
    for name in (f"results_s{seed}.json",):
        p = HERE / name
        if p.exists():
            rows = json.loads(p.read_text())
            if rows and "R_gp_off" in rows[0]:
                return {r["task"]: (r["R_sh"], r["R_gp_off"]) for r in rows}
    return None


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3]
    # method -> seed -> mean R (over the common task set for that seed)
    table = defaultdict(dict)
    common_note = []

    for s in seeds:
        cols = {}
        for m in BASELINES:
            d = load_baseline(m, s)
            if d is not None:
                cols[m] = d
        meth = load_method(s)
        if meth is not None:
            cols["method_sh"] = {t: v[0] for t, v in meth.items()}
            cols["method_gp_off"] = {t: v[1] for t, v in meth.items()}
        if not cols:
            print(f"[skip] seed {s}: no results present")
            continue
        # restrict to tasks present in ALL loaded columns (fair comparison)
        common = set.intersection(*[set(d) for d in cols.values()])
        if not common:
            print(f"[warn] seed {s}: no common task set across methods; skipping")
            continue
        common_note.append((s, len(common)))
        print(f"\n=== seed {s}  (common tasks: {len(common)}) ===")
        for name, d in cols.items():
            table[name][s] = mean([d[t] for t in common]) * 100
            print(f"  {name:14s} mean R = {table[name][s]:.2f}")

    if not table:
        print("No results available.")
        return

    print("\n" + "=" * 74)
    print("AGGREGATE mean audit balanced accuracy (matched budget), seeds:",
          sorted({s for m in table.values() for s in m}))
    order = ["seqft", "olora", "ewc", "method_sh", "method_gp_off"]
    ref = None
    for name in order:
        if name not in table:
            continue
        vals = [table[name][s] for s in sorted(table[name])]
        line = f"  {name:14s} mean={mean(vals):.2f}  per-seed={['%.2f' % v for v in vals]}"
        print(line)
        if name == "method_gp_off":
            ref = mean(vals)

    if ref is not None:
        print("-" * 74)
        print("Method (grouping+offset, UPPER BOUNDS: oracle routing + refit "
              "offset) minus each baseline:")
        for name in ["seqft", "olora", "ewc"]:
            if name in table:
                b = mean([table[name][s] for s in sorted(table[name])])
                print(f"  method_gp_off - {name:8s} = {ref - b:+.2f}pp")
        print("NOTE: method_gp_off uses oracle routing + refit offset (upper "
              "bounds); the DEPLOYABLE method number is the router+stored cell "
              "from RUNBOOK 3A (analyze_router.py), which is the honest "
              "head-to-head vs these deployable baselines.")


if __name__ == "__main__":
    main()
