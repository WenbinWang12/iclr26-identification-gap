# Phase-2N — Gauge-Fixed Logit Anchoring (GFA): frozen protocol

Frozen 2026-08-31, **before** any Phase-2N run.  Amendments are **appended**, never
edited in.  SHA-256 recorded in `notes/phase2n_sha.txt` at freeze time.

## 0. What measurement forces this method

Three results, all from this project's own runs, none from a paper:

1. **Phase-2K achievability ladder.**  no offset 0.6574 → one global offset 0.6803 →
   per-scope offset (free) 0.7097 → + m=4 codebook 0.7204, against a just-trained
   peak of 0.7381.  Offsets close 78 % of mean forgetting, but almost all of the
   reachable part is already in the *free* per-scope level.
2. **Phase-2L.**  An O-LoRA-form penalty `λ Σ‖A A_tᵀ‖_F²` fails criterion C twice
   (λ=0.5 and λ=1.0, 6 seeds).  Identified cause: it constrains `A` against a
   detached history while the optimizer keeps updating `B`, so the function that
   produces old tasks' verbalizer logits is never constrained.
3. **Phase-2M.**  A KL anchor on the softmax over `V_{t−1}` fails criterion 1 at
   −4.78 pp CI[−7.36, −2.25] — **and** its entire cost is removed by the free
   per-scope constant (ACC_scoped diff +0.28 pp CI[−2.97, +3.66]; damage fraction
   removed 1.059).  Criterion 4 *passed*: q_1 0.0761 → 0.0616, q_2 0.0284 → 0.0158.

Fact 3 is the whole argument for Phase-2N.  VLA regularised a quantity that is
**gauge-dependent**: the softmax over the union `V_{t−1}` moves when a constant is
added to one task's verbalizer logits, but no deployed decision does.  Every decision
is a restricted argmax over one task's own verbalizer subset, which is invariant to
that constant.  So VLA spent its entire budget on the one direction that is free to
repair post-hoc, and paid in the directions that are not.

`theory_output_layer_capacity_v1.md` §2 already names the fix: project onto the
zero-mean subspace `V_0` of each scope.  That projected contrast is **exactly** the
object Prop 2a is stated about.  Phase-2M's criterion 4 passing while criterion 1
failed is the direct evidence that radius-tightening in the wrong gauge is
insufficient; GFA tightens the same radii in the gauge the decision actually uses.

**This is a mechanism change, not a third regulariser tried on the same idea.**
Phase-2M §5 outcome 3 forbids the latter inside 2M, which is why this is a new
protocol with its own frozen criteria, written before its first run.

## 1. The method

At the boundary after task `t−1`, the live model **is** the snapshot `θ̄` — no copy,
no stored parameters, O(1) storage, data-free with respect to earlier tasks.

For each previous scope `S` (a set of tasks sharing a verbalizer), let `V_S` be that
scope's verbalizer first-piece column set, `|V_S| = K_S`.  For an input `x`:

* `z_S(x; θ) ∈ R^{K_S}` — logits restricted to `V_S`.
* **Gauge fixing:** `g_S(x; θ) = z_S(x; θ) − mean(z_S(x; θ)) · 1`, the zero-mean
  projection `P_0 z_S`.  Adding any constant to `z_S` leaves `g_S` unchanged, and
  leaves the restricted argmax unchanged.  `g_S` therefore carries exactly the
  decision-relevant part and nothing else.
* **Anchor:** `L_gfa = λ_g · (1/|𝒮|) Σ_{S ∈ 𝒮} mean_x ‖g_S(x; θ) − g_S(x; θ̄)‖²`,
  a squared error in the gauge-fixed contrast, averaged over previous scopes and over
  **current-task inputs only**.
* Singleton scopes (`K_S = 1`) are **excluded**: `g_S ≡ 0` identically, so they
  contribute no gradient.  Including them would only dilute the mean.  This is Prop 1
  restated — a singleton scope has no within-scope contrast to preserve.

Targets `g_S(x; θ̄)` are precomputed once per task, before training starts, from the
live pre-training model.  Squared error rather than KL: `g_S` is an unnormalised
contrast vector, not a distribution, and the whole point is to avoid a softmax whose
normalisation reintroduces the gauge.

Cost: one forward pass per task over `D_t`, same as Phase-2M.

**Predicted difference from VLA, stated now:** VLA's cost was entirely gauge; GFA
cannot pay that cost, because the projected target is invariant to it.  If GFA also
loses raw ACC, then the loss was never about gauge and §0's reading of fact 3 is
wrong.

## 2. Metrics — levels, as in Phase-2M §2

* **ACC** — mean over all 15 scorable tasks of restricted-argmax balanced accuracy on
  the `audit` split after the final task.  **Primary.**
* **ACC_scoped** — same with the free per-scope offset.  Primary for the composed
  system.
* **BWT** and **peak** — reported always, side by side, so a level change can always
  be attributed to retention or to plasticity.  Phase-2L was selected on a quantity
  that confounded these; that is not repeated.

Never `risk`, never official `test.json`.  Offsets fit on train-derived `risk`, scored
on disjoint `audit`.

## 3. Frozen criteria

Baseline `runs/phase2k_qoc_converged`, 3 seeds, same measurement path.  Cluster
bootstrap, **tasks as clusters**, 10 000 resamples, mean statistic.

* **Criterion 1 (primary).**  `ACC[GFA] − ACC[none]` CI excludes 0 on the positive
  side.
* **Criterion 2 (composed).**  `ACC_scoped[GFA] − ACC_scoped[none]` CI excludes 0.
* **Criterion 3 (plasticity guard).**  Mean peak not more than 2 pp below 0.7381.
* **Criterion 4 (gauge-invariance check — the new falsifiable one).**
  `ACC[GFA] − ACC[none]` and `ACC_scoped[GFA] − ACC_scoped[none]` must agree to
  within 2 pp.  GFA regularises only gauge-fixed contrasts, so the free per-scope
  constant should have **little left to repair**: the two diffs moving apart by more
  than 2 pp means the method is still leaking into the gauge, and §1's claim is
  wrong regardless of whether criterion 1 passes.
* **Criterion 5.**  Prop 1 exact: singleton scopes have `Δ_id_scoped ≡ 0`,
  0 violations.  A violation means mis-wiring and nothing may be quoted.
* **Criterion 6 (theory-coupled).**  q_1 and q_2 of shared conflict scopes decrease.
  Phase-2M passed this while failing criterion 1, so on its own it is **not**
  evidence the method works; it is reported to test whether the radii and the level
  move together *once the gauge is right*.
* **Training gate, unconditional.**  Median post-training `R_raw` > 0.60.

## 4. λ_g selection gate

Grid **{0.5, 2.0, 8.0}**, fixed, seed 1, first 6 tasks, `update` split only.  Scored
by `experiments/phase2m_vla/lambda_gate_vla.py` with `--method gfa` (register
`gfa: "gfa_lambda"` in `LAMBDA_KEY`), i.e. Phase-2L's `retention_score` unchanged:
each early task's final `update` accuracy minus its own peak, so plasticity cancels.
Ties take the smallest λ_g.

Endpoint rule as in Phase-2M §4: λ_g = 0 is a closed grid point because it must be
numerically inert; only the upper edge is open.  If the selection lands there still
improving outward, that is recorded as a limitation and the grid is **not** extended.

**Additionally frozen now, because Phase-2M §A1.2 exposed the hole:** the gate's
retention quantity and §2's level metric can disagree.  If they do again, the gate
still governs λ_g and §3 still governs whether the method works, and **both** tables
are reported.  No re-selection on levels after the fact.

## 5. Predeclared outcomes

1. **Criteria 1–4 pass.**  A working method: retention improves, and the improvement
   is in the decision-relevant subspace rather than the repairable gauge.  This is the
   paper.
2. **Criterion 1 passes, 2 fails.**  GFA and the free offset are redundant.  Still a
   working method; the two-level story collapses to one and must be written that way.
3. **Criterion 1 fails but criterion 4 passes.**  The gauge diagnosis was right and
   still insufficient: three mechanisms have now failed to raise the retained level,
   each for an identified reason.  **Then the paper is the negative result**, and it
   is written as the output-layer-capacity analysis with a measured achievability
   ladder plus three falsified interventions — not as a method paper with a weak
   method.  The ICLR contribution in that case is the ladder, Prop 1/2a/3, and the
   demonstration that q_m-tightening does not imply accuracy.
4. **Criterion 4 fails.**  §1's mechanism claim is wrong; report plainly and stop
   proposing anchors on this benchmark at this budget.

## 6. Anti-artifact checks, before any criterion is quoted

* λ_g = 0 with the GFA path live must equal `--cl-method none` numerically.
* The anchor must be exactly 0 on task 1 (`𝒮 = ∅`), asserted in code.
* **Gauge invariance, asserted in a test:** adding an arbitrary constant to all
  columns of a scope in the target must not change `L_gfa` by more than 1e−6.
* **Singleton exclusion, asserted in a test:** a scope with `K_S = 1` contributes
  exactly 0.
* Targets come from a frozen snapshot; no gradient reaches `θ̄`.
* Trainable parameter count identical to `none` (2 359 296), logged.
* No examples from tasks `< t` are read during task `t`'s training — by construction
  and by test.
* `order4_data.py` untouched; SHA-256 pin and Phase-2I frozen-dependency tests pass.
* `risk`/`audit` disjointness re-asserted per stage; official `test.json` unread.

## 7. Compute

Same shape as Phase-2M: ≈140 min/seed measured.  Gate 3 arms + reuse of Phase-2M's
`none` gate arm ≈ 30 min wall-clock on separate cards.  Confirmatory 3 seeds ≈ 2.5 h.
Strictly non-preemptive: per-card self-wait on `memory.free ≥ 6000`, never signal
another user's job.

## A1. λ_g selection gate — result (amendment, appended 2026-08-31)

Seed 1, first 6 tasks, order `MNLI, CB, WiC, COPA, QQP, BoolQA`, four scorable early
tasks.  23.4–23.7 min/arm, all rc=0.  `none` arm reused verbatim from Phase-2M
(`runs/phase2m_gate_none`, same seed, same settings — not re-run).  Scored by
`lambda_gate_vla.py --method gfa`, which imports Phase-2L's `retention_score`
unchanged; 73/73 Phase-2M+2N tests pass locally and on the box.

### A1.1 Retention (the frozen gate quantity: final − own peak)

| arm | λ_g | retention | COPA | MNLI | QQP | WiC |
|---|---|---|---|---|---|---|
| none | 0.0 | −0.0612 | −0.015 | −0.218 | +0.007 | −0.020 |
| g05 | 0.5 | −0.0566 | +0.003 | −0.277 | −0.025 | +0.072 |
| g20 | 2.0 | −0.0724 | +0.000 | −0.233 | −0.006 | −0.050 |
| g80 | 8.0 | **−0.0285** | +0.000 | **−0.099** | +0.000 | −0.015 |

**Selected λ_g = 8.0.  It beats `none`** (−0.0285 vs −0.0612), the first arm in this
project to do so on this quantity — Phase-2L's λ=0.5 (−0.0242 vs −0.0612 on its own
grid) and Phase-2M's λ_a=8.0 (−0.0630 vs −0.0612) did not.

MNLI is the informative cell.  It is task 1, it carries by far the largest drop, and
it is the task VLA degraded monotonically (−0.218 → −0.271 in λ_a).  Under GFA it
moves the other way: −0.218 → −0.099, less than half the drop.  QQP and COPA go to
exactly 0.000.  This is the pattern §1 predicted and Phase-2M did not produce.

### A1.2 Levels — and, for the first time, the two measures agree

| arm | λ_g | level (mean final) | Δ vs none | mean peak |
|---|---|---|---|---|
| none | 0.0 | 0.5865 | — | 0.6477 |
| g05 | 0.5 | 0.6220 | +3.55 pp | 0.6786 |
| g20 | 2.0 | 0.6319 | +4.54 pp | 0.7043 |
| g80 | 8.0 | **0.6507** | **+6.42 pp** | 0.6793 |

Phase-2M §A1.2 recorded a direct conflict: its gate quantity said VLA was worse
while the level said it was 4–5 pp better, and I refused to resolve it in the
method's favour.  Here there is nothing to resolve — λ_g = 8.0 is best on **both**,
and its peak (0.6793) is *below* g20's (0.7043), so the level gain is not a
plasticity trade.  That is a materially stronger signal than anything Phase-2L or
Phase-2M produced, and it is the reason to spend the confirmatory compute.

**Limitation, recorded, grid not extended.**  λ_g = 8.0 is the upper endpoint of
{0.5, 2.0, 8.0} and still improving outward (−0.0285 > g20's −0.0724).  §4 forbids
extending the grid after seeing this, for the reason in
`phase2l_additivity_protocol.md` §A1: adding points once the selection lands on an
edge is how a tuning-on-the-signal artifact is manufactured.  A larger λ_g might be
better; that question belongs to a future pre-registration, not to this one.

Non-monotonicity across the grid (0.5 better than 2.0 on retention, worse on level)
means the four-point curve is noisy at one seed.  The honest reading is "λ_g = 8.0 is
the best of three, on one seed, on both measures", not "retention increases in λ_g".

### A1.3 Wiring checks passed at the gate

* **λ_g = 0 equivalence** — `--cl-method gfa --gfa-lambda 0.0` vs `--cl-method none`:
  12/12 values identical.  Numerically inert with the GFA path live.
* **Anchor exactly 0 on task 1** — smoke run logs `gfa 0 -> 0 (S=0)`, then S grows
  1 → 2 → 3 as multi-task scopes accumulate; singleton scopes never enter.
* **Gauge invariance (§6's frozen 1e-6 check)** — passes.  The projection is computed
  in float64: in float32 the check missed its own threshold by 1.31e-6 through
  catastrophic cancellation, and the fix was to make the implementation meet the
  frozen threshold rather than to loosen the threshold.  A companion test records
  that the model-side invariance is bounded by float32 logit storage (`c · 2^-24`),
  not by the projection.
* **Scope identity** — scope keys are compared as column **sets**, not sequences:
  MNLI and CB share a verbalizer with different label ordering.  First version
  compared sequences and died at task 2 of the smoke run; both directions now pinned
  by tests (same set reordered accepted, genuinely different set refused).
* `order4_data.py` untouched.

### A1.4 Confirmatory arms

3 seeds at λ_g = 8.0 into `runs/phase2n_gfa_confirm`, no `--gate-update-eval`.
§3's criteria 1–6 are evaluated on this arm and nothing else.  No secondary λ arm this
time: Phase-2M's secondary existed because its selection margin was 0.25 pp; here the
margin over the next arm is 4.4 pp on retention and 1.9 pp on level, so a sensitivity
arm would buy less than the seeds it costs.

## A2. Confirmatory result — criteria 1–6 (amendment, appended 2026-08-31)

3 seeds at λ_g = 8.0 into `runs/phase2n_gfa_confirm`, 15/15 stages each, rc = 0,
135.6–140.1 min.  Baseline `runs/phase2k_qoc_converged` seeds 1–3, same measurement
path.  Scorer `experiments/phase2m_vla/decide_vla.py`; the phase2n + phase2m suites
(73 tests) pass locally and on the box, run before any number below was read.

### A2.1 Criteria

| # | criterion | result | value |
|---|---|---|---|
| 1 | ACC diff CI > 0 (primary) | **FAIL** | −6.29 pp, CI [−9.37, −3.16] |
| 2 | ACC_scoped diff CI excludes 0 | FAIL | +0.27 pp, CI [−5.24, +6.44] |
| 3 | peak not > 2 pp below 0.7381 | **FAIL** | 0.6839, i.e. 5.42 pp below |
| 4 | the two diffs agree within 2 pp | **FAIL** | \|−6.29 − 0.27\| = 6.55 pp |
| 5 | Prop 1 exact on singletons | PASS | 87 checked, 0 violations |
| 6 | q_1 and q_2 decrease | FAIL | q_1 **+0.0356**, q_2 **+0.0086** — both increased |
| gate | median post-training `R_raw` > 0.60 | PASS | 0.7402 |

ACC 0.6448 vs 0.7077.  ACC_scoped 0.7254 vs 0.7227.

### A2.2 §5's outcome 4 fired; §1's prediction is falsified

§1 predicted, verbatim: "VLA's cost was entirely gauge; GFA cannot pay that cost,
because the projected target is invariant to it.  If GFA also loses raw ACC, then the
loss was never about gauge and §0's reading of fact 3 is wrong."

GFA loses raw ACC by 6.29 pp — more than VLA's 4.78 pp.  The antecedent fired.
**The prediction is falsified, and §0's reading of fact 3 is wrong.**  Criterion 4,
written specifically to catch this, fails at 6.55 pp against its 2 pp tolerance, so
the falsification is on the protocol's own frozen terms, not by reinterpretation.

§5's **outcome 4** fired: "Criterion 4 fails.  §1's mechanism claim is wrong; report
plainly and stop proposing anchors on this benchmark at this budget."  Outcome 3
(criterion 1 fails, criterion 4 passes) did **not** fire — outcome 4 is the stricter
branch and supersedes it.  Accordingly no fourth anchor follows.

### A2.3 Damage decomposition — the same signature as Phase-2M §A2.3

| arm | ACC | ACC_scoped | free-offset gain |
|---|---|---|---|
| none | 0.7077 | 0.7227 | +1.51 pp |
| VLA λ_a = 8.0 (Phase-2M) | 0.6599 | 0.7255 | +6.56 pp |
| **GFA λ_g = 8.0** | **0.6448** | **0.7254** | **+8.06 pp** |

Raw damage −6.29 pp; after the free per-scope constant +0.27 pp; fraction of the
damage the offset removes **1.042** (VLA's was 1.059).  The three arms differ by
6.29 pp in ACC and by 0.27 pp in ACC_scoped: all three land on the same scoped level
(0.7227 / 0.7255 / 0.7254) and differ only in how much of it a per-scope constant has
to recover.  As in Phase-2M, "1.042" is a ratio of point estimates and is not
bootstrapped; read it as "essentially all", not as 4.2 % over-removal.

**This is the result that matters, and it is not what §1 predicted.**  The penalty is
gauge-invariant — §6's 1e-6 check passes at float64, and a test pins invariance to
target shifts of 5.0, −100.0, 42.0, 1e3.  Yet the damage it causes is still almost
exactly gauge-shaped.  The two facts are compatible for a reason the protocol did not
anticipate: making the *penalty* blind to the per-scope mean does not make the
*optimiser* leave that mean alone.  `P_0 z_S` is pinned; `mean(z_S)` is not penalised
at all, and the shared LoRA update is free to move it.  So the anchor spends capacity
holding a contrast fixed while the unconstrained direction absorbs the accommodation —
and unrestricted-argmax accuracy, which reads the absolute logits, pays for it.
Gauge-invariance of the objective does not imply gauge-invariance of the solution.

### A2.4 Where the damage falls — it is not the anchored scopes

Every scope on this benchmark has ≥ 2 label columns, so GFA anchors all 9 of them once
seen.  Splitting the 12 scorable tasks by whether their scope holds more than one task:

| class | task-seed obs | ACC GFA | ACC none | diff | scoped GFA | scoped none | diff |
|---|---|---|---|---|---|---|---|
| multi-task scope | 21 | 0.6333 | 0.6835 | −5.02 pp | 0.7247 | 0.6952 | **+2.95 pp** |
| single-task scope | 15 | 0.6608 | 0.7414 | **−8.06 pp** | 0.7263 | 0.7612 | −3.49 pp |

The scopes GFA was built for are the ones it helps once the gauge is quotiented out
(+2.95 pp scoped); the tasks that are alone in their scope are hurt most (−8.06 pp
raw, and −3.49 pp even after the offset, the only genuinely irreparable loss here).
Those tasks are never in the penalty's own sum, so their loss arrives through the
shared trunk.  A per-scope penalty with a shared adapter is not a per-scope
intervention.

### A2.5 The 6-task gate did not predict the 15-task result

§A1 selected λ_g = 8.0 because it was the first arm in this project to beat `none` on
the frozen retention quantity (−0.0285 vs −0.0612) *and* to lead on level.  At 15
tasks that reverses:

| quantity | none | GFA λ_g = 8.0 |
|---|---|---|
| retention (final − own peak) | −0.0713 | **−0.0803** |
| own just-trained peak | 0.7790 | **0.7251** |

So the gate's headline advantage was a 6-task artefact.  This is recorded as a
limitation of the selection procedure, not as grounds for re-selecting: re-running the
gate at 15 tasks and picking again is exactly the tuning-on-the-signal move §4
forbids.  What it does establish is that a 6-task gate is too short for this benchmark,
which is a finding about the protocol design and should be said out loud in the paper.

Per-task retention (GFA − none) is not uniform in sign: SST-2 +0.128, AGNews +0.066,
MNLI +0.037, MultiRC +0.023 improve; QQP −0.117, RTE −0.091, COPA −0.047 degrade.
§A1's MNLI observation (−0.218 → −0.099 at 6 tasks) survives in direction at 15 tasks
(−0.302 → −0.266) but is an order of magnitude smaller than it looked, and it is not
the dominant term — QQP alone loses more than MNLI gains.

Criterion 6 flipped sign relative to Phase-2M: VLA *tightened* the conflict radii
while losing accuracy (q_1 −0.0145, q_2 −0.0126); GFA *loosened* them (+0.0356,
+0.0086, both CIs touching 0) while losing more accuracy.  Both directions of the
radius move now coexist with a loss of level, which closes the remaining escape route:
q_m and ACC are not monotonically related in either direction.

### A2.6 What is sayable, and what is not

Sayable:

* Three interventions — orthogonal-subspace penalty (Phase-2L), gauge-dependent
  logit anchoring (Phase-2M), gauge-fixed logit anchoring (Phase-2N) — each with a
  pre-registered protocol and a tuned λ, all fail to raise retained restricted-argmax
  accuracy over `none` at this budget.  Two of the three *lower* it significantly.
* All three leave the scoped level statistically unchanged (0.7227 / 0.7255 / 0.7254),
  differing only in how much a free per-scope constant must repair.  That constant is
  a 4-parameter-per-scope fit; the interventions are 2 359 296-parameter ones.
* Tightening or loosening q_m does not predict the level, in either direction.
* Making the penalty gauge-invariant does not make the learned solution
  gauge-invariant. This is a concrete, measured failure mode for output-layer
  regularisers under a shared adapter, and it is the technical content Phase-2N adds.
* The achievability ladder (Phase-2K: 0.6574 → 0.6803 → 0.7097 → 0.7204, peak 0.7381)
  stands, and now has three failed attempts to climb it by regularisation.

Not sayable:

* That gauge was "the" mechanism behind VLA's damage. §A2.3 kills that reading.
* That GFA would work with a different λ_g, a longer gate, or per-scope adapters.
  Untested; saying so would be speculation dressed as a limitation.
* Any number from the 1.042 removal fraction as if it had an interval. It does not.
* Anything from `runs/phase2n_gate_*` as evidence about 15-task behaviour — §A2.5 is
  the record that it is not.

### A2.7 Retraction inside this amendment: §A2.4's +2.95 pp is not significant

§A2.4 reported that GFA gains +2.95 pp of `ACC_scoped` on multi-task scopes and called
those "the scopes GFA was built for".  Bootstrapped on the same 3 seeds, tasks as
clusters: **+2.95 pp, CI [−5.54, +11.94], n = 7 — not significant.**  It rests on two
tasks (MNLI +20.66 pp, QQP +19.53 pp) against two of the opposite sign (SST-2
−10.68 pp, MultiRC −9.64 pp).  The two *raw*-ACC losses in the same table are the ones
that survive: multi-task −5.02 pp CI [−9.97, −0.37], single-task −8.06 pp
CI [−11.01, −5.99].

So "GFA helps where the conflict is real" is **withdrawn**.  It was a point estimate
read from a table without an interval, which is the exact error §A2.3 flags about the
1.042 fraction, committed three paragraphs later.  Nothing downstream may cite it, and
in particular it is **not** grounds for a scope-gated GFA variant: there is no measured
effect to gate on.  §5's outcome 4 stands unmodified.
