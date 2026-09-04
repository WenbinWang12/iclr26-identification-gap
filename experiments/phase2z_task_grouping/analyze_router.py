"""Analyze RUNBOOK-3A results: how much does the deployable router cost, and does
the fully deployable cell (route + stored offset) still beat the baseline?

Reads router_s{seed}.json (a list of per-task rows written by run_router.py).
Reports per seed and aggregate, with sign-consistency across seeds:
  route accuracy to true family
  routing cost           = mean(R_grouped_orc) - mean(R_grouped_route)   [no offset]
  deployable e2e         = mean(R_grouped_route_stored) - mean(R_shared) [both deployable]
  oracle e2e (reference) = mean(R_grouped_orc_refit)   - mean(R_shared)  [both ceilings]

Verdict thresholds (mirror the frozen protocol note):
  SUCCESS  = deployable e2e > 0 on every seed;
  ROUTER   = routing cost recovers >= half the orc gain over shared.
Nothing is fabricated; every number traces to the frozen JSON. A mixed/negative
result is reported as such -- do NOT substitute the oracle number.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def mean(xs):
    return sum(xs) / len(xs)


def m(rows, k):
    return mean([r[k] for r in rows]) * 100


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3]
    per_seed = {}
    for s in seeds:
        p = HERE / f"router_s{s}.json"
        if not p.exists():
            print(f"[skip] seed {s}: {p.name} not present")
            continue
        rows = json.loads(p.read_text())
        d = dict(
            route_acc=mean([r["route_acc_to_true_family"] for r in rows]) * 100,
            R_shared=m(rows, "R_shared"),
            R_orc=m(rows, "R_grouped_orc"),
            R_route=m(rows, "R_grouped_route"),
            R_route_stored=m(rows, "R_grouped_route_stored"),
            R_orc_refit=m(rows, "R_grouped_orc_refit"),
        )
        d["routing_cost"] = d["R_orc"] - d["R_route"]           # no offset
        d["orc_gain"] = d["R_orc"] - d["R_shared"]              # oracle grouping gain
        d["route_gain"] = d["R_route"] - d["R_shared"]          # routed grouping gain
        d["dep_e2e"] = d["R_route_stored"] - d["R_shared"]      # fully deployable e2e
        d["orc_e2e"] = d["R_orc_refit"] - d["R_shared"]         # both-ceilings e2e
        per_seed[s] = d
        print(f"\n=== seed {s} ===")
        print(f"  route->true family : {d['route_acc']:.1f}%")
        print(f"  R_shared={d['R_shared']:.2f}  R_grouped_orc={d['R_orc']:.2f}  "
              f"R_grouped_route={d['R_route']:.2f}")
        print(f"  routing cost (orc-route)   : {d['routing_cost']:+.2f}pp")
        print(f"  routed grouping gain       : {d['route_gain']:+.2f}pp "
              f"(oracle {d['orc_gain']:+.2f}pp)")
        print(f"  deployable e2e (route+stored - shared) : {d['dep_e2e']:+.2f}pp")
        print(f"  reference oracle e2e (orc+refit-shared): {d['orc_e2e']:+.2f}pp")

    if not per_seed:
        print("No seeds available.")
        return

    print("\n" + "=" * 74)
    print("AGGREGATE over seeds:", sorted(per_seed))
    keys = ["route_acc", "routing_cost", "orc_gain", "route_gain",
            "dep_e2e", "orc_e2e"]
    labels = {
        "route_acc": "route->true family accuracy (%)",
        "routing_cost": "routing cost (orc - route)",
        "orc_gain": "oracle grouping gain (orc - shared)",
        "route_gain": "routed grouping gain (route - shared)",
        "dep_e2e": "DEPLOYABLE e2e (route+stored - shared)",
        "orc_e2e": "reference oracle e2e (orc+refit - shared)",
    }
    for k in keys:
        vals = [per_seed[s][k] for s in sorted(per_seed)]
        signs = {(v > 0) for v in vals}
        tag = "SAME SIGN" if len(signs) == 1 else "MIXED SIGN"
        unit = "" if k == "route_acc" else "pp"
        print(f"  {labels[k]:42s} mean={mean(vals):+.2f}{unit}  "
              f"per-seed={['%+.2f'%v for v in vals]}  [{tag}]")

    # Verdict
    dep = [per_seed[s]["dep_e2e"] for s in sorted(per_seed)]
    cost = mean([per_seed[s]["routing_cost"] for s in sorted(per_seed)])
    orc_gain = mean([per_seed[s]["orc_gain"] for s in sorted(per_seed)])
    recovered = (orc_gain - cost) / orc_gain if orc_gain else float("nan")
    print("-" * 74)
    print(f"SUCCESS (deployable e2e > 0 all seeds): {all(v > 0 for v in dep)}")
    print(f"Router recovers {recovered*100:.0f}% of the oracle grouping gain "
          f"(>=50% target).")


if __name__ == "__main__":
    main()
