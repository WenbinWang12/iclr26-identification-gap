# Phase-2T — Variance-Gated Calibration (VGC): frozen protocol

Frozen 2026-09-01, **before seed 2 and seed 3 had finished**, so the two held-out
seeds cannot have informed any threshold below. Seed 1 is the development set and is
named as such throughout.

## 0. Why this method, forced by Phase-2S's failure

Phase-2S built SIO on synthetic logits and it failed every criterion on real Order-4
data (§A4). The reason was recorded: real `s = z_1 - z_0` is **unimodal**, class
separation is only 0.89 sd at the median, so the two-component mixture SIO fits has
nothing to find and its gate abstains on 78.6% of batches.

Dumping the real logits (`--dump-logits`, added because §A1.1 found the Alibaba runs
stored none) made three facts measurable on 70 real `K_S = 2` batches:

| fact | value | consequence |
| --- | --- | --- |
| a single offset's oracle gain | mean **+9.34 pp**, median +7.03, worst **+0.78** | the headroom is large and an offset never has to hurt |
| Batch Calibration's actual gain | mean **+6.07 pp**, median +5.47 | BC captures most of it — it is a strong method here |
| BC's worst case | **−14.84 pp**, harmful on **10/70** | BC's defect on real data is its **left tail**, not its mean |
| the harmful batches | **9 of 10 are QQP** | the damage is concentrated, not diffuse |
| `sd(s)` on harmful batches | 0.30–0.45 | |
| `sd(s)` on the best batches | 0.84–1.27 | a **label-free** signal separates them |

This reverses Phase-2S's framing, and the reversal is the point. Phase-2S tried to
replace BC's estimator on the theory that its marginal dependence was the defect.
On real logits BC's mean gain is +6.07 pp — replacing it is the wrong move. What BC
lacks is an abstention rule: when the verbalizer logits carry almost no signal
(`sd(s)` near zero), any offset moves a decision boundary through noise.

**Not a fourth regulariser.** Training is `--cl-method none`, byte-identical to
baseline, as in 2P/2Q/2S. 2N §5 outcome 4 stands. VGC occupies the same
admissibility row as BPO and SIO in `sec:enumeration` (query batch + current model);
it adds no row.

## 1. The method

For a batch of `n` queries with scope `S`, `K_S = 2`, verbalizer logits `Z`:

    s_i = Z[i,1] - Z[i,0]
    b_BC = (0, -mean(s))                     # Batch Calibration, unchanged
    b_VGC = b_BC   if sd(s) >= TAU_SD  else  0

That is the whole rule. It does not change BC's estimator, add a parameter to fit, or
store state. One threshold, one label-free statistic.

**Why `sd(s)` and not a confidence or entropy measure.** `sd(s)` is the spread of
the gauge-invariant decision statistic itself. If it is small, every example sits at
nearly the same distance from the boundary, so shifting the boundary flips a large
fraction of predictions at once and the direction of that flip is set by noise. This
is a statement about the decision geometry, not about calibration.

**Gauge invariance.** `sd(s)` and `mean(s)` both read only `Z[:,1] - Z[:,0]`, so
adding a constant to every logit changes nothing.

Stored state: **0 floats**, as with BPO and SIO.

## 2. The threshold, and its provenance

`TAU_SD = 0.50`, chosen on **seed 1 only** (the development set), from this sweep:

| `TAU_SD` | batches kept | mean | median | worst | harmful (< −1 pp) |
| --- | --- | --- | --- | --- | --- |
| 0.00 (plain BC) | 70/70 | +6.07 | +5.47 | **−14.84** | **10/70** |
| 0.40 | 48/70 | +6.02 | +3.52 | −9.38 | 2/70 |
| **0.50** | **36/70** | **+5.16** | +0.00 | **−1.56** | **1/70** |
| 0.60 | 32/70 | +4.77 | +0.00 | −1.56 | 1/70 |
| 1.00 | 16/70 | +2.31 | +0.00 | −1.56 | 1/70 |

0.50 is picked because it is where the worst case flattens: from 0.50 upward the
worst case is −1.56 pp and stays there, while the mean keeps falling. 0.40 keeps a
higher mean (+6.02) but leaves a −9.38 pp tail. The choice trades 0.9 pp of mean for
13.3 pp of tail.

**This threshold is a development-set choice and is declared as one.** It is frozen
here, before seeds 2 and 3 exist, so §4's criteria are a genuine held-out test. If
the held-out seeds need a different threshold, that is a failure of this protocol,
not an invitation to retune — see §5 outcome 3.

`sd(s)` medians by task on seed 1, recorded so the reader can see what the gate
actually selects: BoolQA 1.23, RTE 1.01, IMDB 0.85, SST-2 0.82, MultiRC 0.46,
WiC 0.46, COPA 0.35, QQP 0.33.

**The confound this creates, stated before measuring.** The gate keeps
BoolQA/RTE/IMDB/SST-2 and drops QQP/COPA/WiC. On seed 1 those are also the tasks
where BC helps and hurts respectively, so the gate could be fitting *task identity*
rather than a geometric property. §4 criterion **T5** exists solely to test that: if
`sd(s)` predicts harm *within* a task as well as across tasks, the signal is
geometric; if not, VGC is a task filter wearing a statistic's clothes, and §5
outcome 4 says so in those words.

## 3. Criteria, thresholds predeclared

Scored on **seeds 2 and 3** (held out; seed 1 reported separately as development).
Same statistics as every prior phase: `cluster_bootstrap_ci`, clusters = tasks,
10000 draws, seed 0, median. Arms: `raw`, `BC`, `VGC`, `oracle`, all from the same
stored logits in the same process.

- **T1 (does the gate remove the tail).** VGC's worst per-batch gain over raw is
  `≥ −3.0 pp`, versus BC's worst on the same batches. Mechanical, not statistical.
- **T2 (does it keep the mean).** Median of `VGC − raw ≥ +1.0 pp` with a
  cluster-bootstrap 95% CI strictly above 0.
- **T3 (what it costs against plain BC).** Median of `VGC − BC` reported with CI. VGC
  is expected to be **below** BC on the mean; the criterion is that the loss is
  `≥ −2.0 pp`, i.e. the tail removal is not bought with most of the gain.
- **T4 (is the gate non-vacuous).** The apply rate on held-out batches is
  `≥ 0.3` and `≤ 0.9`. Below 0.3 the rule is mostly abstention; above 0.9 it is
  plain BC with extra steps.
- **T5 (is the signal geometric or just task identity).** Within each task with
  `≥ 6` held-out batches, the Spearman correlation between `sd(s)` and BC's gain must
  be positive, and the median of those within-task correlations must exceed 0. This
  is the decisive criterion for the mechanism claim.
- **T6 (the harmful batches are the low-variance ones).** Of the batches where BC is
  harmful (`< −1 pp`) on held-out seeds, the fraction with `sd(s) < TAU_SD` is
  reported. No threshold; this quantifies how much of BC's tail the gate can see.

## 4. Predeclared outcomes

1. **T1, T2, T4, T5 pass and T3 within bound** → VGC is a working fix to a
   documented defect in a published method. Write it up; position against BC and the
   label-shift line; report the mean cost honestly as the price of the tail.
2. **T1, T2, T4 pass, T5 fails** → the tail removal is real but the mechanism is not
   geometric. Report as a working heuristic with an *unexplained* selector, and say
   the `sd(s)`-as-decision-geometry argument was not confirmed.
3. **T1 passes on seed 1 but not on held-out seeds** → the threshold was fitted to
   seed 1. Report as a development-set artefact, name `TAU_SD` as the overfitted
   quantity, and do not retune.
4. **T5 fails and the gate's kept/dropped set matches task identity** → VGC is a task
   filter, not a calibration gate. Report in those words; it is not admissible as a
   task-agnostic rule.
5. **T2 fails** → the gate removes the tail and the gain with it. Report as negative;
   BC's tail defect survives as the finding and VGC as a failed remedy.

Anything not on this list is recorded as a protocol gap in an amendment, as in 2R
and 2S, rather than patched retroactively.

## 5. Anti-artefact checks, declared before running

- Abstention must return **exactly** the raw prediction, bit-for-bit.
- `b_VGC` must be identical whether computed on `Z` or `Z + c·1`.
- BC is recomputed from the stored logits in the same process, never quoted from its
  paper or from Phase-2S's numbers.
- The gate reads `sd(s)`, which is label-free; asserted by computing it from `Z`
  alone with labels withheld from the function.
- Seed 1 numbers are reported as development throughout and never pooled with the
  held-out seeds into a single headline.
- No official `test.json` is read.

## 6. Cost

**Zero GPU.** The 70 seed-1 batches and the held-out seeds' batches are already
stored as `.npz` by the `--dump-logits` path. This is the first phase in this project
that costs nothing to run, and it is only possible because Phase-2S's §A1.1 failure
forced the dump path into existence.

---

# §A1 — Held-out verdict: OUTCOME 5. `sd(s)` is the wrong gate.

Appended 2026-08-31. Nothing above this line is edited. Scored by
`experiments/phase2t_vgc/decide_2t.py`
(SHA `36976c7046d09ea83067e2ab89df0e21d9b8985c945eb7543a233832315b0b8a`, hashed
before any seed-2/3 logit was read) on 140 held-out batches from seeds 2 and 3,
against 70 development batches from seed 1.

## A1.1 The scored criteria

| criterion | held-out result | threshold | verdict |
| --- | --- | --- | --- |
| T1 worst VGC gain | **−9.38 pp** (BC's worst −14.84; harmful 7/140 vs BC's 25/140) | ≥ −3.0 | **FAIL** |
| T2 median VGC − raw | **0.00 pp**, CI [0.00, +3.91] | ≥ +1.0 and CI > 0 | **FAIL** |
| T3 median VGC − BC | 0.00 pp, CI [−3.13, 0.00] | ≥ −2.0 | pass |
| T4 apply rate | 0.529 | [0.3, 0.9] | pass |
| T5 within-task ρ(sd, BC gain) | median **−0.18**, positive in only 3/7 tasks | median > 0, all positive | **FAIL** |
| T6 tail coverage | 18 of 25 harmful batches caught (0.72) | reported only | — |

Mean gains over raw on held-out: BC **+5.40**, VGC **+4.24**, oracle **+9.60**.

`decide()` returns `OUTCOME_5_gate_removes_the_gain_with_the_tail` — the branch
that fires when T2 fails, and it fires first by construction because a gate with no
gain makes its tail irrelevant.

## A1.2 Seed 1 passed the same criteria. That is the finding.

On the development seed, T1 passed (worst −1.56, harmful 1/70), T4 passed (0.514),
and even T5 was closer (median +0.19, positive in 4/6). The held-out seeds reverse
T1 by 7.8 pp and flip T5's sign. `TAU_SD = 0.50` was fitted to seed 1, exactly as
§2 warned it might be, and §5 outcome 3 was the branch I expected to need. Outcome 5
is worse: the gain went too.

Two of the three T5 tasks that were positive on seed 1 (BoolQA +0.11 → −0.42,
IMDB −0.72 → −0.18, SST-2 absent → −0.41) do not hold their sign. Only RTE is
consistently positive (+0.81 → +0.71). One task out of seven is not a mechanism.

The task-identity check did **not** fire (`is_task_filter` false, 7 of 8 tasks
all-or-nothing but WiC splits 0.38), so §5 outcome 4 does not apply. The gate is not
*literally* a task filter, but with only WiC varying inside a task it is close
enough that T5 had almost no within-task variation to detect a mechanism with. That
is a defect in T5's design against this data, not a rescue of the method.

## A1.3 What the held-out data says the defect actually is

Diagnostics run after the verdict, on all 210 batches (seeds 1–3, now all
development for any successor rule):

- **Every eval batch is exactly class-balanced** (`|π − 0.5| = 0`, 128 examples each).
  So on this benchmark BC's marginal dependence — the defect Phase-2S was built
  around — cannot be what makes BC lose. Phase-2S's confirmed slope finding stands as
  a statement about *skewed* batches; it is simply not the operative mechanism here.
- **Threshold accuracy is not the problem.** Median `|mean(s) − t_oracle| = 0.183`,
  and on harmful batches the overshoot ratio `|mean(s)|/|t_oracle|` is **1.03** — BC's
  threshold is essentially right where it loses. Replacing the mean with the median
  changes nothing (mean +5.28 vs +5.40, same worst case, better on only 45/140).
- **Harmful batches are the ones with nothing to win.** Oracle headroom is
  **2.34 pp** on harmful batches versus **9.38 pp** on the rest. BC loses where *no*
  offset helps, so its variance is all that is left.
- **`sd(s)` was the wrong reading of this.** It separates in the median (0.352 vs
  0.645) but overlaps in the tail, which is precisely why T1 failed.
- **Flip fraction is the wrong sign.** The share of predictions BC changes correlates
  *positively* with gain (ρ = +0.29/+0.27/+0.47 by seed) and does not separate
  harmful from helpful (0.391 vs 0.422). Discarded.
- **`|mean(s)|`, BC's own correction magnitude, is the best label-free predictor** of
  both the oracle headroom (ρ = **+0.62**) and BC's realised gain (ρ = **+0.61**),
  beating every spread measure (`sd` +0.10, IQR +0.05, range +0.10). Harmful median
  0.452 versus helpful 0.924.

24 of the 35 harmful batches are QQP, as on seed 1.

## A1.4 What I got wrong, stated plainly

I chose the gate statistic by looking at which quantity separated harmful from
helpful batches *in the median* on one seed, and did not check whether it separated
them *in the tail* — even though the tail was the entire stated target of the method.
`sd(s)` and `|mean(s)|` are correlated enough on seed 1 that the wrong one looked
adequate; on 210 batches `|mean(s)|` beats it by 6× in rank correlation with gain.

The deeper error is that I gated on a property of the *batch* when the criterion I
cared about was a property of the *correction*. `|mean(s)|` is not a batch statistic
in the same sense — it is the size of the step BC is proposing, so gating on it asks
BC how confident its own move is. That is the question T1 was really asking.

## A1.5 Standing

VGC-as-frozen is **negative** and joins BPO and SIO in `sec:enumeration`'s second
row. Three rules have now failed in that row; the row itself is the result.

Seeds 1, 2 and 3 are all development from this point on — I have looked at them.
Seeds 11–15 were launched (BABEL job 10273784) **before** any successor threshold was
selected, so a successor rule has a genuine 5-seed held-out set. Any successor needs
its own frozen protocol; it does not inherit T1–T6.


