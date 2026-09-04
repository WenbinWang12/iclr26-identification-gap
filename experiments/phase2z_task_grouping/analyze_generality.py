"""Analyze RUNBOOK-3B results: do the four 2x2 effect signs hold across task
orders and backbones?

Discovers every generality_{order}_{tag}_s{seed}.json under the results dir(s)
given on argv (default: this script's directory), groups by (order, tag) cell,
and for each cell reports the four effects aggregated over its seeds with
sign-consistency:
  e2e          = R_gp_off - R_sh      (both moves, both ceilings; paper headline)
  repr         = R_gp_off - R_sh_off  (grouping on top of offset)
  group_alone  = R_gp - R_sh          (grouping alone inside the 2x2)
  offset_alone = R_sh_off - R_sh      (offset alone)

Generality verdict per effect: SAME SIGN across ALL cells x seeds => the sign
is backbone/order-robust. Any cell flipping is reported, never hidden. Every
number traces to a frozen JSON; nothing is imputed for a missing cell.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

EFFECTS = {
    "e2e": ("R_gp_off", "R_sh"),
    "repr": ("R_gp_off", "R_sh_off"),
    "group_alone": ("R_gp", "R_sh"),
    "offset_alone": ("R_sh_off", "R_sh"),
}


def mean(xs):
    return sum(xs) / len(xs)


def cell_effects(rows):
    def m(k):
        return mean([r[k] for r in rows]) * 100
    vals = {k: m(k) for k in ("R_sh", "R_sh_off", "R_gp", "R_gp_off")}
    return {name: vals[a] - vals[b] for name, (a, b) in EFFECTS.items()}, vals


def main():
    roots = [Path(a) for a in sys.argv[1:]] or [Path(__file__).resolve().parent]
    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(sorted(root.rglob("generality_*_s*.json")))
    if not files:
        print("No generality_*_s*.json found under:", [str(r) for r in roots])
        return

    # cell (order, tag) -> seed -> payload
    cells = defaultdict(dict)
    for p in files:
        payload = json.loads(p.read_text())
        key = (payload["order"], payload["tag"])
        cells[key][payload["seed"]] = payload

    print(f"Discovered {len(files)} files across {len(cells)} cells.\n")
    # per-effect: collect (cell, seed) -> value to judge global sign robustness
    global_signs = defaultdict(list)

    for key in sorted(cells):
        order, tag = key
        seeds = sorted(cells[key])
        backbone = cells[key][seeds[0]]["backbone"]
        family = cells[key][seeds[0]]["family"]
        print("=" * 74)
        print(f"CELL  order={order}  backbone={backbone}  ({family}, tag={tag})"
              f"  seeds={seeds}")
        per_seed_eff = {}
        for s in seeds:
            eff, vals = cell_effects(cells[key][s]["rows"])
            per_seed_eff[s] = eff
            print(f"  seed {s}: R_sh={vals['R_sh']:.2f} R_sh_off={vals['R_sh_off']:.2f}"
                  f" R_gp={vals['R_gp']:.2f} R_gp_off={vals['R_gp_off']:.2f}")
        for name in EFFECTS:
            vs = [per_seed_eff[s][name] for s in seeds]
            signs = {(v > 0) for v in vs}
            tagn = "SAME" if len(signs) == 1 else "MIXED"
            print(f"    {name:13s} mean={mean(vs):+.2f}pp  "
                  f"per-seed={['%+.2f' % v for v in vs]}  [{tagn}]")
            for v in vs:
                global_signs[name].append(((order, tag, ), v))

    print("\n" + "=" * 74)
    print("GENERALITY VERDICT (sign robustness across ALL cells x seeds)")
    for name in EFFECTS:
        entries = global_signs[name]
        vals = [v for _, v in entries]
        pos = all(v > 0 for v in vals)
        neg = all(v < 0 for v in vals)
        robust = "SIGN-ROBUST(+)" if pos else "SIGN-ROBUST(-)" if neg else "FLIPS"
        flips = [c for c, v in entries if (v > 0) != (vals[0] > 0)]
        extra = f"  flipping cells: {sorted(set(flips))}" if flips else ""
        print(f"  {name:13s} {robust:15s} "
              f"n={len(vals)} mean={mean(vals):+.2f}pp{extra}")


if __name__ == "__main__":
    main()
