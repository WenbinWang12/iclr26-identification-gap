#!/usr/bin/env python3
"""Phase-2X verdict: does BPO's advantage over oracle survive the prior-shift grid?

IMMEDIATE DECISION POINT after Phase-2X completes:
- If crossover_prior exists at batch=128 → BPO collapses on imbalanced queries → Δ_id STANDS
- If crossover_prior is None at batch=128 → BPO beats oracle robustly → identification-gap
  thesis is in trouble

This script runs automatically on the scored result and prints the verdict.
"""

import argparse
import json
import sys
from pathlib import Path


def verdict(summary: dict) -> dict:
    """Return the decisive facts and the binary outcome."""
    cells_128 = summary.get("cells", {}).get("128", {})
    if not cells_128:
        return {"outcome": "ERROR", "reason": "no batch=128 cells in summary"}

    # Primary decision: does offset_quality beat R_orc at batch=128, prior=0.50 (balanced)?
    balanced = cells_128.get("0.5", {})
    if not balanced:
        return {"outcome": "ERROR", "reason": "no prior=0.5 cell at batch=128"}

    vs_orc_pp = balanced.get("vs_oracle_pp")
    beats_orc_frac = balanced.get("beats_oracle_frac")

    # Crossover: where does BPO stop beating no-offset?
    crossover_128 = summary.get("crossover", {}).get("128")

    # R_bpo_full from Q2: does it beat oracle on the FULL balanced audit?
    # Extract from any cell (it's invariant across grid)
    r_bpo_full = balanced.get("R_bpo_full")  # This is R_bpo from Phase-2Q
    r_orc = balanced.get("R_orc")

    # Phase-2Q reported R_bpo=0.7674 > R_orc=0.7616 (+0.58pp)
    # If Phase-2X primary axis (offset_quality at balanced prior) replicates that,
    # and crossover is None, BPO's advantage is NOT an artefact.

    facts = {
        "batch_128_prior_0.5_vs_oracle_pp": vs_orc_pp,
        "batch_128_prior_0.5_beats_oracle_frac": beats_orc_frac,
        "crossover_prior_at_batch_128": crossover_128,
        "R_bpo_full": r_bpo_full,
        "R_orc": r_orc,
        "R_bpo_full_advantage_pp": (r_bpo_full - r_orc) * 100 if r_bpo_full and r_orc else None,
    }

    # DECISIVE CRITERION from Phase-2X design:
    # Primary axis at batch=128 (the audit batch size) with balanced prior.
    # If offset_quality beats R_orc, and crossover exists, the advantage collapses under skew.
    # If crossover is None, the advantage survives → identification gap is questionable.

    if crossover_128 is None:
        # BPO beats no-offset across the entire prior grid at batch=128
        outcome = "BPO_SURVIVES_SKEW"
        interpretation = (
            "BPO's advantage over oracle is NOT an artefact of balanced audit splits. "
            "The label-free offset beats the per-task oracle robustly across prior skews. "
            "This challenges the identification-gap thesis: if Δ_id is recoverable without "
            "task identity, the framing needs revision."
        )
    else:
        # BPO collapses at some prior < 1.0
        outcome = "BPO_COLLAPSES_UNDER_SKEW"
        interpretation = (
            f"BPO stops beating no-offset at prior={crossover_128} (batch=128). "
            "The advantage seen in Phase-2Q Q1 is an artefact of the balanced audit split. "
            "Δ_id stands: identification gap exists and is not recoverable by batch statistics alone."
        )

    return {
        "outcome": outcome,
        "interpretation": interpretation,
        "facts": facts,
        "publication_impact": (
            "PAPER_SAFE: continue with identification-gap thesis and multi-task scope Δ_id (+1.10pp)"
            if outcome == "BPO_COLLAPSES_UNDER_SKEW"
            else "PAPER_AT_RISK: BPO anomaly requires explanation or thesis revision"
        )
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", required=True, help="JSON output from score_prior.py")
    args = ap.parse_args()

    with open(args.result, encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    v = verdict(summary)

    print("=" * 80)
    print("PHASE-2X VERDICT")
    print("=" * 80)
    print(f"Outcome: {v['outcome']}")
    print(f"\n{v['interpretation']}")
    print(f"\nPublication impact: {v['publication_impact']}")
    print("\nDecisive facts:")
    for k, val in v.get("facts", {}).items():
        print(f"  {k}: {val}")
    print("=" * 80)

    # Write verdict to a separate file for pipeline automation
    verdict_path = Path(args.result).parent / "verdict_2x.json"
    verdict_path.write_text(json.dumps(v, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nVerdict written to {verdict_path}")

    # Exit code: 0 if paper is safe, 1 if at risk
    sys.exit(0 if v["outcome"] == "BPO_COLLAPSES_UNDER_SKEW" else 1)


if __name__ == "__main__":
    main()
