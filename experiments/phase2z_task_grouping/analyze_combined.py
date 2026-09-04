"""Honest 2x2 aggregation for the combined two-constraint method.

Reads combined_s{1,2,3}.json (each a list of per-task dicts with keys
R_sh, R_sh_off, R_gp, R_gp_off) and reports the matched-budget 2x2 plus the
three effect sizes that matter, per seed and aggregated. No fabrication: every
number is a mean over the frozen per-task audit accuracies in those files.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = [1, 2, 3]


def load():
    data = {}
    for s in SEEDS:
        p = HERE / f"combined_s{s}.json"
        if not p.exists():
            print(f"MISSING {p}")
            continue
        data[s] = json.load(open(p))
    return data


def main():
    data = load()
    if not data:
        print("No combined_s*.json found yet.")
        sys.exit(0)

    keys = ["R_sh", "R_sh_off", "R_gp", "R_gp_off"]
    print("=" * 82)
    print("Combined two-constraint method: 2x2 at matched fixed budget")
    print("  R_sh     shared adapter,  no offset")
    print("  R_sh_off shared adapter,  + per-scope offset")
    print("  R_gp     grouped adapters,no offset")
    print("  R_gp_off grouped adapters,+ per-scope offset   <- proposed method")
    print("=" * 82)

    seed_cell = {k: [] for k in keys}
    seed_eff = {"repr(gp_off-sh_off)": [], "e2e(gp_off-sh)": [],
                "group_alone(gp-sh)": [], "offset_alone(sh_off-sh)": []}

    for s in SEEDS:
        if s not in data:
            continue
        r = data[s]
        cell = {k: np.mean([x[k] for x in r]) * 100 for k in keys}
        for k in keys:
            seed_cell[k].append(cell[k])
        eff = {
            "repr(gp_off-sh_off)": cell["R_gp_off"] - cell["R_sh_off"],
            "e2e(gp_off-sh)": cell["R_gp_off"] - cell["R_sh"],
            "group_alone(gp-sh)": cell["R_gp"] - cell["R_sh"],
            "offset_alone(sh_off-sh)": cell["R_sh_off"] - cell["R_sh"],
        }
        for k, v in eff.items():
            seed_eff[k].append(v)
        print(f"\nseed {s}:")
        print(f"  R_sh={cell['R_sh']:.2f}  R_sh_off={cell['R_sh_off']:.2f}  "
              f"R_gp={cell['R_gp']:.2f}  R_gp_off={cell['R_gp_off']:.2f}")
        for k, v in eff.items():
            print(f"    {k:26s}: {v:+.2f}pp")

    print("\n" + "=" * 82)
    print("AGGREGATE (mean over seeds):")
    for k in keys:
        vals = seed_cell[k]
        print(f"  {k:10s}: {np.mean(vals):.2f}  (per-seed {[f'{v:.2f}' for v in vals]})")
    print("-" * 82)
    for k, vals in seed_eff.items():
        signs = "all+" if all(v > 0 for v in vals) else (
            "all-" if all(v < 0 for v in vals) else "MIXED")
        print(f"  {k:26s}: {np.mean(vals):+.2f}pp  "
              f"per-seed={[f'{v:+.2f}' for v in vals]}  sign={signs}")
    print("=" * 82)

    # Verdict
    e2e = seed_eff["e2e(gp_off-sh)"]
    repr_gain = seed_eff["repr(gp_off-sh_off)"]
    ok = all(v > 1.3 for v in e2e) and all(v > 0 for v in e2e)
    print(f"\nSUCCESS (grouped+offset beats shared-no-offset by >1.3pp all seeds): {ok}")
    print(f"Grouping still helps on top of offset (gp_off>sh_off all seeds): "
          f"{all(v > 0 for v in repr_gain)}")


if __name__ == "__main__":
    main()
