# Phase-2H D2 failure analysis — why the real-Transformer route is a NO-GO

Date: 2026-08-27. Development-only; no number here is a citable positive result.
This is the targeted root-cause analysis requested before deciding whether to
retitle (Route A). Both predeclared backbones (bert-tiny, bert-mini) were run
through the full three-seed D2 panel; both are NO-GO with F1+F2 failing and
F0+F3 passing.

## The two D2 verdicts

| gate | bert-tiny | bert-mini | threshold |
|---|---|---|---|
| F0 integrity | PASS | PASS | — |
| F1 probe balacc | 0.587 FAIL | 0.604 FAIL | ≥ 0.75 |
| F1 AMI / purity | 0.143 / 0.396 | 0.097 / 0.336 | 0.35 / 0.65 |
| F1 rare prec/rec | 0.35 / 0.22 | 0.26 / 0.15 | 0.50 / 0.50 |
| F2 worst-source gain | +1.55pp FAIL | +1.01pp FAIL | ≥ +3pp |
| F2 macro loss | +0.08pp OK | −0.35pp OK | ≤ 1pp |
| F3 offline worst BA | 0.787 PASS | 0.885 PASS | ≥ 0.65 |

The single predeclared representation fallback (bert-tiny → bert-mini) is now
used up. A stronger backbone raised the probe by only 1.7pp (0.587 → 0.604,
still near the ~0.17 six-class floor) and made F2 *worse* (+1.55 → +1.01pp),
while F3 improved (0.787 → 0.885). That pattern is the tell: the pipeline is
correct and LoRA genuinely learns; the *task/stream* gives the method no room.

## Root cause: the rare source is NOT the worst-group (assumption violated)

The method's premise is that a globally rare / under-sampled source is the group
that gets sacrificed under naive replay and therefore needs budget reallocated to
it. The per-source ER-FLOP balanced accuracies falsify that premise on this
stream. Source 5 (Cell_Phones_and_Accessories) is the globally rare source
(probability 0.01 in every phase vector), yet:

| run | worst source | worst BA | rare(5) BA | rare rank (of 6) | spread |
|---|---|---|---|---|---|
| tiny/1234 | 3 | 0.79 | 0.89 | 1st (best) | 0.10 |
| tiny/5678 | 3 | 0.80 | 0.88 | 1st | 0.08 |
| tiny/9012 | 2 | 0.81 | 0.84 | 4th | 0.05 |
| mini/1234 | 3 | 0.80 | 0.91 | 1st | 0.11 |
| mini/5678 | 2 | 0.83 | 0.85 | 4th | 0.04 |
| mini/9012 | 2 | 0.84 | 0.90 | 1st | 0.06 |

The rare source is the *best-learned* source in 3 of 6 runs and never the worst.
The worst source is a *high-frequency* source (2 or 3), and the whole per-source
spread is only 0.04–0.11.

### Why each gate failed, mechanistically

1. **F2 is a task-structure wall, backbone-independent.** Binary polarity
   (positive/negative review) is almost equally easy across all six product
   categories (BA 0.80–0.91). "Worst-group" here is set by *task difficulty*, not
   by sampling frequency, and the difficulty gap is tiny. There is no
   frequency-sacrificed group to recover, so even a true-label oracle cannot lift
   the worst source by 3pp — the headroom does not exist.

2. **F1 fails because polarity dominates category in the frozen embedding.** Six
   product categories share the same "good review / bad review" language; the
   frozen mean-pooled sentence signature separates polarity strongly and category
   weakly. This is a property of the task, not of tiny-vs-mini capacity — a
   stronger backbone moved the probe by <2pp.

3. **Why the linear proxy (phase2f PSR-LoRA) succeeds and this does not.** In the
   controlled linear stream the rare source is constructed to be orthogonal,
   under-sampled, and genuinely sacrificed — there "worst-group = rare group"
   holds and reallocation has something to recover. The real Amazon-polarity
   stream does not have that structure, so the mechanism has nothing to act on.

## Verdict

This is a **design–reality mismatch, not a fixable bug**:
- Not a pipeline defect (F0/F3 pass; LoRA learns; branch restore bit-identical).
- Not a backbone-capacity defect (bert-mini stronger on F3, no better on F1/F2).
- The stream violates the method's core precondition (rare ⇒ worst-group).

The only honest way to make SGCR viable on a real Transformer would be a *different*
stream where the rare/under-sampled source is genuinely the hardest (e.g. severe
per-source polarity imbalance, or cross-domain/cross-lingual sources with real
difficulty gaps), pre-registered fresh. That is new scope, not a rescue of this run.

Per the protocol Stage-D2 rule ("if the oracle fails, no pseudo-group method is
expected to rescue this exact stream/budget"), the D3 SGCR sweep is correctly not
run, and the fallback budget is exhausted. Recorded as a clean negative
feasibility boundary for the real-Transformer route on this stream/budget.
