"""Analyze de-stale results: does grouping shrink the stored-offset staleness?

Reads destale_s{seed}.json (each has regimes 'shared' and 'grouped', each with
per-task R_none / R_stored / R_refit). Reports per seed and aggregate:
  staleness = mean(R_refit) - mean(R_stored)     (upper bound - deployable)
  deployable gain = mean(R_stored) - mean(R_none)
Headline: grouping shrinks staleness (shared_staleness - grouped_staleness),
and whether the sign is consistent across seeds. Nothing is fabricated; every
number traces to the frozen JSON.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def mean(xs):
    return sum(xs) / len(xs)


def agg(regime):
    none = mean([r["R_none"] for r in regime["rows"]]) * 100
    stored = mean([r["R_stored"] for r in regime["rows"]]) * 100
    refit = mean([r["R_refit"] for r in regime["rows"]]) * 100
    return none, stored, refit


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3]
    per_seed = {}
    for s in seeds:
        p = HERE / f"destale_s{s}.json"
        if not p.exists():
            print(f"[skip] seed {s}: {p.name} not present")
            continue
        data = json.loads(p.read_text())
        sh_n, sh_s, sh_r = agg(data["shared"])
        gp_n, gp_s, gp_r = agg(data["grouped"])
        st_sh = sh_r - sh_s          # shared staleness
        st_gp = gp_r - gp_s          # grouped staleness
        per_seed[s] = dict(
            sh_none=sh_n, sh_stored=sh_s, sh_refit=sh_r,
            gp_none=gp_n, gp_stored=gp_s, gp_refit=gp_r,
            st_sh=st_sh, st_gp=st_gp, shrink=st_sh - st_gp,
            dep_sh=sh_s - sh_n, dep_gp=gp_s - gp_n,
        )
        print(f"\n=== seed {s} ===")
        print(f"  SHARED : none={sh_n:.2f} stored={sh_s:.2f} refit={sh_r:.2f}"
              f"  staleness(refit-stored)={st_sh:+.2f}pp  dep(stored-none)={sh_s-sh_n:+.2f}pp")
        print(f"  GROUPED: none={gp_n:.2f} stored={gp_s:.2f} refit={gp_r:.2f}"
              f"  staleness(refit-stored)={st_gp:+.2f}pp  dep(stored-none)={gp_s-gp_n:+.2f}pp")
        print(f"  >> grouping shrinks staleness by {st_sh-st_gp:+.2f}pp")

    if not per_seed:
        print("No seeds available.")
        return

    print("\n" + "=" * 70)
    print("AGGREGATE over seeds:", sorted(per_seed))
    keys = ["st_sh", "st_gp", "shrink", "dep_sh", "dep_gp"]
    labels = {
        "st_sh": "shared staleness (refit-stored)",
        "st_gp": "grouped staleness (refit-stored)",
        "shrink": "grouping shrinks staleness by",
        "dep_sh": "shared deployable gain (stored-none)",
        "dep_gp": "grouped deployable gain (stored-none)",
    }
    for k in keys:
        vals = [per_seed[s][k] for s in sorted(per_seed)]
        m = mean(vals)
        signs = {(v > 0) for v in vals}
        consistent = "SAME SIGN" if len(signs) == 1 else "MIXED SIGN"
        print(f"  {labels[k]:40s} mean={m:+.2f}pp  per-seed={['%+.2f'%v for v in vals]}  [{consistent}]")


if __name__ == "__main__":
    main()
