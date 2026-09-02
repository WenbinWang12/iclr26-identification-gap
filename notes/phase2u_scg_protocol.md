# Phase-2U — Self-Confidence Gate (SCG): frozen protocol

Frozen 2026-08-31, while BABEL job 10273784 (seeds 11–15) was still running, so the
held-out seeds cannot have informed anything below. Seeds 1, 2, 3 are development —
I have looked at all three, and Phase-2T §A1.5 records that.

## 0. Why this rule, forced by 2T's failure

Phase-2T gated Batch Calibration on `sd(s)`, the spread of the batch's decision
statistic. On held-out seeds that failed T1 (worst −9.38 pp), T2 (median gain 0.00)
and T5 (within-task ρ median −0.18); the verdict was OUTCOME 5 and is recorded
unedited in 2T §A1.

2T §A1.3–A1.4 identified the error. I gated on a property of the **batch** when the
thing I needed to condition on was a property of the **correction**. Three facts from
210 development batches:

| fact | value | consequence |
| --- | --- | --- |
| every eval batch is exactly balanced | `\|π − 0.5\| = 0`, n = 128 | BC's marginal dependence cannot be the operative defect here |
| overshoot on harmful batches | `\|mean(s)\|/\|t_oracle\|` = **1.03** | BC's threshold is *right* where it loses; estimation error is not the problem |
| oracle headroom, harmful vs helpful | **2.34 pp** vs **9.38 pp** | BC loses where no offset helps, so its variance is all that remains |

So the question to ask is not "is this batch well separated" but "is the move BC
proposes large enough to be worth its own variance". The size of that move,
`|mean(s)|`, is label-free and is the best predictor of both the headroom (ρ = +0.62)
and BC's realised gain (ρ = +0.61) among every statistic tested — `sd` +0.10,
IQR +0.05, range +0.10, flip fraction +0.29 *with the wrong sign*.

## 1. The method

    s_i = Z[i,1] - Z[i,0]
    m   = mean(s)
    b_SCG = (0, -m)   if |m| >= TAU_M   else   0

BC's estimator, unchanged; BC's own correction magnitude as the gate. Stored state:
**0 floats**. Training is `--cl-method none`, byte-identical to baseline; this adds
no regulariser and no `sec:enumeration` row (query batch + current model, the same
row as BPO, SIO and VGC).

**Why a hard gate and not shrinkage.** Shrinkage keeps part of a move it has just
judged unreliable. Measured on the 210 development batches, every shrinkage variant
is dominated:

| rule | mean | worst | harmful | worst per dev seed |
| --- | --- | --- | --- | --- |
| plain BC | +5.62 | −14.84 | 35/210 | −14.84 / −14.84 / −9.38 |
| **hard, τ = 1.4** | **+3.25** | **+0.00** | **0/210** | **+0.00 / +0.00 / +0.00** |
| hard, τ = 1.2 | +3.76 | −7.81 | 1/210 | +0.00 / +0.00 / −7.81 |
| soft-threshold, τ = 0.8 | +2.08 | −3.12 | 8/210 | −3.12 / −0.78 / −2.34 |
| soft-threshold, τ = 1.0 | +1.28 | −1.56 | 2/210 | −0.78 / −0.78 / −1.56 |
| quadratic ramp, τ = 2.0 | +3.39 | −4.69 | 14/210 | −4.69 / −4.69 / −1.56 |

The hard gate at 1.4 is the only rule that is harmful-free on all three development
seeds, and it beats soft-0.8 on the mean as well. Shrinkage is therefore rejected on
measurement, not on taste.

## 2. The threshold and its provenance

`TAU_M = 1.4`, chosen on development seeds 1–3 pooled (210 batches):

| τ | kept | mean | median | worst | harmful | worst per dev seed |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 (BC) | 210/210 | +5.62 | +4.69 | −14.84 | 35/210 | −14.84 / −14.84 / −9.38 |
| 1.0 | 80/210 | +4.43 | +0.00 | −9.38 | 2/210 | −9.38 / +0.00 / −7.81 |
| 1.2 | 63/210 | +3.76 | +0.00 | −7.81 | 1/210 | +0.00 / +0.00 / −7.81 |
| **1.4** | **50/210** | **+3.25** | +0.00 | **+0.00** | **0/210** | **+0.00 / +0.00 / +0.00** |
| 1.6 | 30/210 | +2.42 | +0.00 | +0.00 | 0/210 | +0.00 / +0.00 / +0.00 |
| 2.0 | 16/210 | +1.70 | +0.00 | +0.00 | 0/210 | +0.00 / +0.00 / +0.00 |

1.4 is the smallest value on the grid that is harmful-free on **each** development
seed separately, not merely pooled. Larger values only cost mean gain.

Declared honestly: `TAU_M` is a development-set choice on three seeds, and the median
gain at 1.4 is **0.00** because the gate abstains on 160/210 batches. The claim is
about the mean and the tail, not the median, and U2 below is written accordingly —
requiring a positive median here would be requiring the method to be something it
is not.

**Scale dependence, stated as a limitation before scoring.** `TAU_M = 1.4` is in
logit units on this model (T5 verbalizer logits, LoRA, `n = 128`). Unlike `sd(s)`,
`|mean(s)|` has no natural normalisation, and dividing by `sd` was measurably worse
(ρ = +0.32 against +0.61). So this threshold is **not** claimed to transfer across
models or batch sizes, only across seeds and task streams of this setup. U6 records
the dimensionless variant's number so a reader can see the cost of that choice.

## 3. Criteria, thresholds predeclared

Scored on seeds **11–15** (job 10273784), held out. `cluster_bootstrap_ci`,
clusters = tasks, 10000 draws, seed 0, median. Arms `raw`, `BC`, `SCG`, `oracle`
recomputed from the same stored logits in one process.

- **U1 (the tail, the point of the method).** SCG's worst per-batch gain over raw
  is `≥ −1.0 pp` on the pooled held-out batches, **and** `≥ −1.0 pp` within every
  one of the five seeds taken separately. The per-seed clause exists because 2T's
  τ_sd passed pooled-on-seed-1 and then broke.
- **U2 (the gain survives).** Mean SCG − raw `≥ +2.0 pp`, and the cluster-bootstrap
  CI on the **mean** of SCG − raw excludes 0. Stated on the mean, not the median,
  because §2 declares the median is 0 by construction.
- **U3 (cost against plain BC).** Mean SCG − BC reported with CI; must be
  `≥ −3.5 pp`. SCG is expected to lose to BC on the mean — that is the price of the
  tail, and 2T's T3 was too loose at −2.0 to be informative once the median was 0.
- **U4 (non-vacuous).** Held-out apply rate in `[0.10, 0.60]`. The development rate
  is 50/210 = 0.238; a rate near 0 makes the rule a no-op and near 1 makes it BC.
- **U5 (the mechanism, within task).** Within each task with `≥ 6` held-out batches,
  Spearman ρ between `|mean(s)|` and BC's gain over raw. Median across tasks must
  exceed `+0.25`, and at least **4 of 5** qualifying tasks positive. 2T's T5 demanded
  *all* tasks positive with median > 0; that was simultaneously too strict on sign
  agreement and too weak on magnitude, and it is replaced here rather than reused.
- **U6 (is it really the magnitude, or the scale).** The same sweep with
  `|mean(s)|/sd(s)` in place of `|mean(s)|` is reported: its best harmful-free
  threshold, mean and apply rate. No pass/fail. This is the check that §2's scale
  choice was not arbitrary.
- **U7 (does the gate reduce to task identity).** Per-task apply rates; the rule is
  flagged if every task is 0.0 or 1.0. 2T's version of this did not fire but had
  almost no within-task variation to work with, so it is reported as a diagnostic
  and not as a pass/fail.

## 4. Predeclared outcomes

1. **U1, U2, U4, U5 pass and U3 within bound** → a working, mechanistically explained
   fix to a documented failure mode of a published method: BC is unsafe on
   low-correction batches and its own correction magnitude tells it so. Write up.
2. **U1, U2, U4 pass, U5 fails** → the tail fix is real but the explanation is not
   established. Report as a heuristic and say the mechanism claim failed.
3. **U1 passes, U2 fails** → the gate is safe and pointless. Report negative; BC's
   asymmetric risk profile survives as the finding.
4. **U1 fails** → `TAU_M` did not transfer, exactly as `TAU_SD` did not. Report as a
   second thresholding failure and stop pursuing gated BC; the paper's claim becomes
   that *no* label-free gate in this family fixed BC's tail across three attempts.
5. **U1 and U2 pass but U6 shows the dimensionless variant is at least as good** →
   report SCG with the normalised statistic named as the preferable form and the
   logit-scale threshold called out as a limitation of the frozen version.

Anything outside this list is recorded as a protocol gap in an amendment.

## 5. Anti-artefact checks

- Abstention returns exactly `0`, so predictions are bit-identical to raw.
- The gate reads `Z` only; label-freeness asserted at the signature level.
- `|mean(s)|` and the offset both read `Z[:,1] − Z[:,0]`, so the rule is gauge
  invariant up to float addition error.
- BC recomputed in-process from stored logits, never quoted.
- Development seeds 1–3 are reported separately and never pooled with held-out into
  a headline number.
- Seeds 11–15 were launched before `TAU_M` was selected; the launch time is in the
  hash ledger.
- No official `test.json` is read.

## 6. Cost

One GPU-hour per held-out seed for the runs already in flight, then zero — scoring
reads the `.npz` dumps.

---

# §A1 — Held-out verdict: OUTCOME 4 on U1, but U5 passes decisively

Appended 2026-08-31. Nothing above this line is edited. Scored by
`experiments/phase2u_scg/decide_2u.py` (SHA
`39dd665c5fea4f8f767a5c79c76768f46c34b4b24fe3ba88fe70a9cd2bd3c86a`, hashed before any
held-out logit was read) on **350 held-out batches from 5 seeds** (11–15, BABEL jobs
10273784 and 10273936), against 210 development batches from seeds 1–3.

## A1.1 The scored criteria

| criterion | held-out result | threshold | verdict |
| --- | --- | --- | --- |
| U1 worst SCG gain | **−2.34 pp**; per seed +0.00 / −0.78 / −2.34 / +0.00 / −1.56; harmful **2/350** vs BC's **39/350** | ≥ −1.0, pooled and per seed | **FAIL** |
| U2 mean SCG − raw | **+3.57 pp**, CI [+1.76, +5.92] | ≥ +2.0, CI > 0 | **pass** |
| U3 mean SCG − BC | −3.83 pp, CI [−5.98, −2.25] | ≥ −3.5 | **FAIL** |
| U4 apply rate | 0.271 | [0.10, 0.60] | **pass** |
| U5 within-task ρ(&#124;mean(s)&#124;, BC gain) | median **+0.719**, **8/8 tasks positive** | median > +0.25, ≥ 4/5 positive | **pass** |
| U6 normalised variant | `\|mean\|/sd` has **no** harmful-free threshold on held-out (`null`); `\|mean\|` best harmful-free τ = 2.2 at +1.38 mean | reported | — |
| U7 task identity | 0 of 8 tasks all-or-nothing; rates 0.09–0.58 | reported | not a task filter |

`decide()` returns `OUTCOME_4_threshold_did_not_transfer_stop_pursuing_gated_bc`,
because U1 gates that branch and U1 failed.

Mean gains over raw on held-out: BC **+7.40**, SCG **+3.57**, oracle **+10.66**.

## A1.2 The frozen verdict is OUTCOME 4 and stands. What it cost, precisely.

U1 asked for a worst case no worse than −1.0 pp and got **−2.34 pp**. The rule missed
by 1.34 pp on 2 batches out of 350. It is a real failure of a predeclared threshold
and it is reported as one; the gate is **not** harmful-free out of sample, which was
the entire promise.

U3 also failed: SCG gives up 3.83 pp of BC's mean, past the 3.5 pp I allowed. Note
that BC itself was *stronger* on these seeds (+7.40 vs +5.62 on development), which
mechanically widens the gap a gate must pay — I set U3 against development-era BC and
did not anticipate that.

## A1.3 What passed, and why it is the more interesting half

**U5 is the mechanism criterion and it passed on held-out data by a wide margin.**
Within-task Spearman ρ between `|mean(s)|` — the size of the correction BC proposes —
and BC's realised gain is positive in **8 of 8** tasks, median **+0.719**:

| task | ρ | | task | ρ |
| --- | --- | --- | --- | --- |
| BoolQA | +0.921 | | SST-2 | +0.690 |
| RTE | +0.868 | | WiC | +0.145 |
| MultiRC | +0.790 | | QQP | +0.068 |
| IMDB | +0.749 | | COPA | +0.000 |

This is the claim 2T's T5 tested and failed (median −0.18, 3/7 positive). Changing
the statistic from a property of the batch to a property of the correction moved it
from refuted to confirmed, on 5 seeds never used to choose anything. U7 rules out the
trivial explanation: apply rates run 0.09–0.58 and no task is all-or-nothing, so this
is within-task structure, not task identity.

So the honest summary is: **the diagnosis is confirmed and the remedy is not
calibrated.** BC's gain is predictable from its own correction magnitude; thresholding
on that magnitude at a value fitted on 3 seeds does not generalise tightly enough to
be safe.

## A1.4 The two failures have one signature, and I am not allowed to use it

Both held-out harmful batches have `|mean(s)|` above τ but a very small spread:

| seed | task | stage | `\|mean(s)\|` | `sd(s)` | gain |
| --- | --- | --- | --- | --- | --- |
| 13 | QQP | 10 | 1.52 | **0.23** | −2.34 |
| 15 | WiC | 15 | 2.02 | **0.33** | −1.56 |

The ep4 sensitivity arm's 3 harmful batches (reported in A1.5, never merged) have the
same signature: `sd` = 0.89, 0.38, 0.37.

The conjunction `|mean(s)| ≥ 1.4 AND sd(s) ≥ 0.5` is harmful-free on **all eight**
seeds run in this project (dev 1–3: 30 kept, +2.73, worst +0.00; held-out 11–15:
54 kept, **+3.04, worst +0.00, 0/350 harmful**), and would have passed U1 and U2.

**I am recording this and not claiming it.** The `sd ≥ 0.5` term was selected *after*
reading the held-out failures, so on those seeds it is a fitted parameter, and the
project now has zero unseen seeds for it. Presenting it as a held-out success would be
exactly the error 2T §A1.2 was written to document. It is a hypothesis with a stated
provenance, nothing more; testing it honestly costs five more seeds.

A related caution: 2T §A1.3 measured `|mean|/sd` as a *weaker* signal than `|mean|`
(ρ +0.32 vs +0.61, harmful/helpful medians 1.446 vs 1.580) and U6 confirms it has no
harmful-free threshold here at all. So the right reading of the conjunction is **not**
"normalise by sd" — the ratio fails — but "require both a large move and enough spread
to absorb it". Those are different conditions and only the conjunction survives.

## A1.5 The epochs=4 sensitivity arm, reported beside and never merged

Per §A3.3 of the 2S protocol, the epochs=4 arm is reported separately. On its 70
batches (seed 1 only): BC mean +4.88, worst −5.47, harmful 7/70; SCG at the frozen
τ = 1.4 mean +3.72, worst −5.47, harmful 3/70; oracle mean +8.24. Within-task
ρ median **+0.372** with 4/6 tasks positive — U5's mechanism reproduces on this arm
too. U1 would fail here as well, more badly than on the held-out seeds.

## A1.6 Standing

Three pre-registered rules have now failed in `sec:enumeration`'s query-batch row:
BPO (fails its own control), SIO (fails all six criteria), VGC (OUTCOME 5), and SCG
now fails U1/U3 while passing U2/U4/U5. §4 of the paper already states this as a
negative result about an admissible *source*; A1.3's confirmed mechanism is a
positive finding to state alongside it, at the strength the data supports:

> Batch calibration's benefit on a query batch is predicted, within task, by the
> magnitude of the correction it proposes (held-out ρ median +0.72, 8/8 tasks). A
> single global threshold on that magnitude is not sufficient to make it safe
> (held-out worst −2.34 pp against a −1.0 pp floor).

Seeds 1–3, 11–15 and the ep4 arm are all development from this point on. Any
successor — including A1.4's conjunction — needs its own frozen protocol and its own
fresh seeds.

