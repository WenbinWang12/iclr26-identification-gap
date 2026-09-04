"""Analyze RUNBOOK-3C results: does the method hold on a genuine K_S>=3
multi-task scope, and what is the per-scope q_m there?

Reads scopes_s{seed}.json (written by run_scopes.py). Reports, per seed and
aggregate with sign-consistency:
  * the 2x2 effects on the K_S>=3 MULTI-task subset (the open-theory regime),
  * the same effects on ALL tasks for reference,
  * per-scope q_m (m=1..) for every multi-task scope, both regimes.

Verdict:
  C1: e2e (gp_off - sh) > 0 on the K_S>=3 subset, every seed.
  C2: offset helps on the K_S>=3 subset (sh_off - sh > 0), every seed -- this is
      the output-layer constraint being relieved where Prop3 is open.
  C3: q_m REPORTED for the multi-task scope (not thresholded); q_1 > 0 confirms
      the scope is genuinely non-degenerate (a real d>=2 quantization instance).
Nothing is imputed; a subset that never forms (no CB) is reported as absent.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EFFECTS = {
    "e2e": ("R_gp_off", "R_sh"),
    "repr": ("R_gp_off", "R_sh_off"),
    "group_alone": ("R_gp", "R_sh"),
    "offset_alone": ("R_sh_off", "R_sh"),
}


def mean(xs):
    return sum(xs) / len(xs)


def subset_effects(rows):
    def m(k):
        return mean([r[k] for r in rows]) * 100
    vals = {k: m(k) for k in ("R_sh", "R_sh_off", "R_gp", "R_gp_off")}
    return {n: vals[a] - vals[b] for n, (a, b) in EFFECTS.items()}


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3]
    per_seed = {}
    for s in seeds:
        p = HERE / f"scopes_s{s}.json"
        if not p.exists():
            print(f"[skip] seed {s}: {p.name} not present")
            continue
        payload = json.loads(p.read_text())
        rows = payload["rows"]
        ksge3 = [r for r in rows if r.get("in_ksge3_multi")]
        per_seed[s] = {
            "all": subset_effects(rows),
            "ksge3": subset_effects(ksge3) if ksge3 else None,
            "n_ksge3": len(ksge3),
            "scopes": payload.get("ksge3_multi_scopes", {}),
            "qm_shared": payload.get("quantization_radius_shared", {}),
            "qm_grouped": payload.get("quantization_radius_grouped", {}),
        }
        print(f"\n=== seed {s} ===")
        print(f"  K_S>=3 multi scopes: {per_seed[s]['scopes']}")
        print(f"  n tasks in K_S>=3 multi subset: {per_seed[s]['n_ksge3']}")
        for label in ("all", "ksge3"):
            eff = per_seed[s][label]
            if eff is None:
                print(f"  [{label}] absent")
                continue
            print(f"  [{label}] " + "  ".join(
                f"{n}={eff[n]:+.2f}pp" for n in EFFECTS))
        print("  per-scope q_m (shared):")
        for sc, radii in per_seed[s]["qm_shared"].items():
            print("    " + sc + ": " + "  ".join(
                f"q_{m}={float(r):.4f}" for m, r in radii.items()))

    if not per_seed:
        print("No seeds available.")
        return

    print("\n" + "=" * 74)
    print("AGGREGATE (K_S>=3 multi-task subset) over seeds:", sorted(per_seed))
    have = [s for s in sorted(per_seed) if per_seed[s]["ksge3"] is not None]
    if not have:
        print("  No K_S>=3 multi-task subset formed in any seed (CB not admitted?).")
    else:
        for n in EFFECTS:
            vals = [per_seed[s]["ksge3"][n] for s in have]
            signs = {(v > 0) for v in vals}
            tag = "SAME SIGN" if len(signs) == 1 else "MIXED SIGN"
            print(f"  {n:13s} mean={mean(vals):+.2f}pp  "
                  f"per-seed={['%+.2f' % v for v in vals]}  [{tag}]")
        e2e = [per_seed[s]["ksge3"]["e2e"] for s in have]
        off = [per_seed[s]["ksge3"]["offset_alone"] for s in have]
        print("-" * 74)
        print(f"C1 (e2e>0 all seeds on K_S>=3): {all(v > 0 for v in e2e)}")
        print(f"C2 (offset helps, all seeds)  : {all(v > 0 for v in off)}")
    # C3: q_1 > 0 for the multi-task scope (genuine non-degenerate d>=2 instance)
    q1s = []
    for s in sorted(per_seed):
        for sc, radii in per_seed[s]["qm_shared"].items():
            if "1" in radii:
                q1s.append(float(radii["1"]))
    if q1s:
        print(f"C3 (q_1 reported; min={min(q1s):.4f} max={max(q1s):.4f}); "
              f"q_1>0 => genuine non-degenerate scope: {all(v > 0 for v in q1s)}")


if __name__ == "__main__":
    main()
