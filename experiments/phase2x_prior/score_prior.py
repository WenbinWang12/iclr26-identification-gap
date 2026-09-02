"""Phase-2X: the prior-shift grid, computed offline from dumped audit logits.

WHY THIS PHASE EXISTS
---------------------
Phase-2Q measured `R_bpo = 0.7674 > R_orc = 0.7616`: a rule that reads **no
labels and no task identity** beat the per-task oracle.  If that survives, the
paper's identification framing is in trouble, because Δ_id is supposed to be the
part of shallow forgetting you *cannot* recover without task identity.

Q4 was the criterion built to kill it — refit BPO on deliberately imbalanced
mixes and see if the advantage is an artefact of our own 64/class balanced audit
split.  Q4 returned −0.67 pp (ratio 0.7) and −1.53 pp (ratio 0.9) and I recorded
it as a failure.  It is not a failure; it is **unreadable**.  `subsample_
imbalanced` draws *without replacement* and sizes the draw by the binding class,
so ratio 0.7 leaves 27 minority examples and ratio 0.9 leaves 7.  With 7
examples one flipped prediction moves balanced accuracy by 1/7 = 14.29 pp.  The
measurement quantum was larger than the effect for both cells.

THE FIX: SPLIT THE TWO QUESTIONS Q4 CONFLATED
---------------------------------------------
* **offset_quality** (primary).  The skewed draw is used *only* to infer the
  offset; that offset is then scored on the FULL balanced 128 examples.  Skew
  therefore perturbs offset inference and nothing else, and the accuracy quantum
  stays at 1/128 ≈ 0.78 pp.  This is the axis that answers "does a
  non-representative query batch corrupt what BPO infers?"
* **deployment_realism** (secondary, reported not decisive).  Fit and score on
  the same skewed draw — what Q4 did.  It is the honest deployment number but it
  inherits the small-minority quantum, so it is reported with its quantum
  alongside it and never used alone.

Sampling here is **with replacement**, which is what makes the whole grid
reachable: without replacement, p=0.9 at batch 128 would need 115 minority
examples out of 64 and is simply infeasible — the exact wall that capped Q4.
BPO reads only the logit distribution, so duplicated rows are legitimate inputs
to offset inference.  Duplicates never enter the primary accuracy number,
because the primary number is always scored on the untouched balanced split.

Reference levels (`R_raw`, `R_shr_global`-substitute, `R_orc`) are computed on
the full balanced split and are therefore **invariant across grid cells** — a
test-checkable property that catches a skewed draw leaking into the baselines.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from experiments.phase2j_offset_conflict.offsets import (balanced_accuracy,
                                                         fit_offset, gauge_fix)
from experiments.phase2q_bpo.bpo import fit_batch_prior_offset

DUMP_RE = re.compile(r"^seed(\d+)_stage(\d+)_(.+)\.npz$")

# Bounded by measurability, not by taste: at batch 16 a p=0.95 draw expects 0.8
# minority rows, so the cell cannot inform an offset at all.  Cells are kept only
# when the expected minority count is >= MIN_MINORITY.
PRIORS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
BATCHES = (16, 32, 64, 128, 256)
MIN_MINORITY = 8
DRAWS = 8


def load_dumps(dump_dir: Path) -> list[dict]:
    """Every per-stage npz written by run_qoc.py --dump-logits."""
    out = []
    for path in sorted(Path(dump_dir).glob("*.npz")):
        m = DUMP_RE.match(path.name)
        if not m:
            continue
        with np.load(path, allow_pickle=True) as z:
            out.append({
                "seed": int(m.group(1)),
                "stage": int(m.group(2)),
                "task": m.group(3),
                "logits": z["logits"].astype(np.float64),
                "labels": z["labels"].astype(np.int64),
                "verbalizer": [str(v) for v in z["verbalizer"]],
                "scope": str(z["scope"]),
                "trained_task": str(z["trained_task"]),
                "path": str(path),
            })
    return out


def skewed_indices(labels: np.ndarray, prior: float, size: int, *,
                   seed: int) -> np.ndarray:
    """`size` indices whose class-0 share is `prior`, sampled WITH replacement.

    With replacement is deliberate and is what makes the grid reachable at all;
    see the module docstring.  The realised share is exact up to integer
    rounding, independent of how many examples of each class the split holds.
    """
    labels = np.asarray(labels)
    K = int(labels.max()) + 1 if labels.size else 0
    if K < 2:
        raise ValueError("need at least 2 classes, got K=%d" % K)
    if not 0.0 < prior < 1.0:
        raise ValueError("prior must lie in (0, 1), got %r" % (prior,))
    per_class = [np.flatnonzero(labels == k) for k in range(K)]
    if any(len(idx) == 0 for idx in per_class):
        raise ValueError("every class must be present")

    rng = np.random.default_rng(seed)
    n0 = int(round(size * prior))
    n0 = min(max(n0, 1), size - 1)          # never degenerate to one class
    rest = size - n0
    take = [n0] + [rest // (K - 1)] * (K - 1)
    for i in range(rest - (rest // (K - 1)) * (K - 1)):
        take[1 + i] += 1
    picked = [rng.choice(per_class[k], size=n, replace=True)
              for k, n in enumerate(take) if n > 0]
    out = np.concatenate(picked)
    rng.shuffle(out)
    return out


def reference_levels(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Baselines on the FULL balanced split — invariant across grid cells."""
    per_task = gauge_fix(fit_offset(logits, labels))
    return {
        "R_raw": float(balanced_accuracy(logits, labels, None)),
        "R_orc": float(balanced_accuracy(logits, labels, per_task)),
        "R_bpo_full": float(balanced_accuracy(
            logits, labels, fit_batch_prior_offset(logits))),
        "n": int(logits.shape[0]),
        "K": int(logits.shape[1]),
    }


def score_entry(entry: dict, *, priors=PRIORS, batches=BATCHES,
                draws: int = DRAWS, seed: int = 0) -> list[dict]:
    """One row per (prior, batch) cell for one dumped stage/task."""
    logits, labels = entry["logits"], entry["labels"]
    K = logits.shape[1]
    if K < 2 or len(set(labels.tolist())) < K:
        return []                       # a cell-free entry, not a zero-effect one
    ref = reference_levels(logits, labels)

    rows = []
    for batch in batches:
        for prior in priors:
            # MIN_MINORITY gates the SECONDARY axis only.  It used to skip the
            # whole cell, which was wrong twice over: a 3-minority draw is a
            # perfectly legitimate *input* to offset inference (it is precisely
            # the pathological batch this phase exists to measure), and the
            # primary number is scored on the full balanced split so its quantum
            # is 1/n whatever the draw looks like.  Skipping those cells threw
            # away the most informative corner of the grid — the same mistake as
            # Q4, just relocated into the filter.
            measurable_on_draw = batch * (1.0 - prior) >= MIN_MINORITY
            oq, dr, drq = [], [], []
            for d in range(draws):
                idx = skewed_indices(labels, prior, batch,
                                     seed=seed + 1009 * d + 7 * batch)
                off = fit_batch_prior_offset(logits[idx])
                # PRIMARY: offset inferred from the skewed draw, scored on the
                # untouched balanced split.  Quantum 1/n, not 1/minority.
                oq.append(balanced_accuracy(logits, labels, off))
                # SECONDARY: fit and score on the same draw (what Q4 did).
                sub_lab = labels[idx]
                if measurable_on_draw and len(set(sub_lab.tolist())) == K:
                    dr.append(balanced_accuracy(logits[idx], sub_lab, off))
                    drq.append(1.0 / min(np.bincount(sub_lab, minlength=K)))
            row = {
                "task": entry["task"], "stage": entry["stage"],
                "seed_run": entry["seed"], "scope": entry["scope"],
                "prior": prior, "batch": batch, "draws": len(oq),
                "offset_quality": float(np.mean(oq)) if oq else None,
                "offset_quality_sd": float(np.std(oq)) if oq else None,
                "deployment_realism": float(np.mean(dr)) if dr else None,
                "deployment_quantum": float(np.mean(drq)) if drq else None,
                "expected_minority": float(batch * (1.0 - prior)),
                "realism_measurable": bool(measurable_on_draw),
                # Pinned by test_primary_axis_is_scored_on_the_full_balanced_split:
                # the number of rows the PRIMARY axis was scored on. It must equal
                # the full split, never the draw size, or the quantum silently
                # becomes 1/minority again and we are back in Q4's failure.
                "n_scored": int(logits.shape[0]),
            }
            row.update(ref)
            rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    """Grid means plus the crossover prior, per batch size.

    The crossover is where the label-free offset stops beating no-offset at all
    (`offset_quality - R_raw <= 0`).  Reported per batch because the whole point
    is that the effect is a function of how much minority signal the batch holds.
    """
    out = {"cells": {}, "crossover": {}, "n_rows": len(rows)}
    for batch in sorted({r["batch"] for r in rows}):
        per_prior = {}
        for prior in sorted({r["prior"] for r in rows if r["batch"] == batch}):
            sel = [r for r in rows
                   if r["batch"] == batch and r["prior"] == prior
                   and r["offset_quality"] is not None]
            if not sel:
                continue
            gain = [r["offset_quality"] - r["R_raw"] for r in sel]
            vs_orc = [r["offset_quality"] - r["R_orc"] for r in sel]
            per_prior[str(prior)] = {
                "n_entries": len(sel),
                "offset_quality": float(np.mean([r["offset_quality"] for r in sel])),
                "R_raw": float(np.mean([r["R_raw"] for r in sel])),
                "R_orc": float(np.mean([r["R_orc"] for r in sel])),
                "gain_over_raw_pp": float(np.mean(gain) * 100.0),
                "gain_over_raw_positive_frac": float(np.mean([g > 0 for g in gain])),
                "vs_oracle_pp": float(np.mean(vs_orc) * 100.0),
                "beats_oracle_frac": float(np.mean([v > 0 for v in vs_orc])),
                "deployment_realism": (
                    float(np.mean([r["deployment_realism"] for r in sel
                                   if r["deployment_realism"] is not None]))
                    if any(r["deployment_realism"] is not None for r in sel) else None),
            }
        out["cells"][str(batch)] = per_prior
        cross = None
        for prior in sorted(float(p) for p in per_prior):
            if per_prior[str(prior)]["gain_over_raw_pp"] <= 0.0:
                cross = prior
                break
        out["crossover"][str(batch)] = cross
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", required=True,
                    help="directory of seed*_stage*_*.npz from --dump-logits")
    ap.add_argument("--out", required=True, help="JSON summary path")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    entries = load_dumps(Path(args.dump_dir))
    if not entries:
        raise SystemExit("no npz dumps under %s" % args.dump_dir)
    print(f"Loaded {len(entries)} entries, processing...")
    rows, skipped = [], []
    for i, e in enumerate(entries):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(entries)} entries ({100*i//len(entries)}%)")
        got = score_entry(e, draws=args.draws, seed=args.seed)
        if not got:
            skipped.append({"task": e["task"], "stage": e["stage"],
                            "reason": "fewer than K classes present"})
        rows.extend(got)
    print(f"  Progress: {len(entries)}/{len(entries)} entries (100%)")

    summary = summarise(rows)
    # No silent caps: say out loud which entries and which cells were dropped.
    summary["skipped_entries"] = skipped
    summary["cell_rule"] = (
        "every (prior, batch) cell is scored on the PRIMARY axis; cells with "
        "expected minority < %d additionally report deployment_realism=None, "
        "because fit-and-score-on-the-same-draw is what Q4 did and it is "
        "unreadable there (quantum 1/minority)" % MIN_MINORITY)
    summary["n_entries"] = len(entries)
    summary["min_minority"] = MIN_MINORITY
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows},
                                         indent=2, sort_keys=True),
                              encoding="utf-8")
    print("entries=%d rows=%d skipped=%d" % (len(entries), len(rows), len(skipped)))
    for batch, cross in summary["crossover"].items():
        print("batch %-4s crossover_prior=%s" % (batch, cross))


if __name__ == "__main__":
    main()
