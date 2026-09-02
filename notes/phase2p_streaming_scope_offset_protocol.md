# Phase-2P — Streaming Scope-Offset table (SSO): frozen protocol

Frozen 2026-08-31, **before any Phase-2P run exists**, before the method is
implemented, and after Phase-2N closed as a negative result (§A2, outcome 4).

## 0. Why this and not a fourth regulariser

Phase-2N §5 outcome 4 forbids further anchors: three parameter-space interventions
(2L orthogonal-subspace, 2M gauge-dependent anchor, 2N gauge-fixed anchor) all failed
to raise the retained level, two significantly.  This phase does **not** add a
regulariser.  Training is `--cl-method none`, byte-identical to the baseline.  What
changes is only what the output layer does at inference.

The measurement that licenses it, on `runs/phase2k_qoc_converged`, 3 seeds, tasks as
clusters, 10 000 draws, 12 scorable tasks:

| comparison | point | CI | verdict |
|---|---|---|---|
| per-scope (m=1) − global | +1.51 pp | [+0.43, +2.74] | **significant** |
| per-scope (m=1) − no offset | +2.11 pp | [+0.84, +3.50] | **significant** |
| per-task ORACLE − per-scope | +0.02 pp | [−1.00, +1.00] | zero |
| codebook m=4 − per-scope | +0.00 pp | [−1.02, +0.98] | zero |

Reading: indexing the output offset **by scope** recovers the identification gap that
indexing by task would, and the scope is readable from the prompt's own option list —
no task ID.  This is Prop 1 doing work, and it is the thesis's own prediction.

**The gap that makes this not yet a method.**  `R_shr_scoped` as currently measured
refits the table at every stage from the risk logits of **all seen tasks**
(`run_qoc.py:429–458`).  That is rehearsal: it re-reads old tasks' examples.  A
deployable version may store no examples.

## 1. The method

Prop 2 (geometric half, unconditional) already gives the target: the minimax shared
offset over a set of per-task optima is the **Chebyshev centre** of those optima.  That
object needs only the optima, not the data that produced them.

At the boundary of task `t`, after training and using **only task `t`'s own `risk`
split**:

1. `b_t* = gauge_fix(fit_offset(risk_logits[t]))` — a `(K_S − 1)`-dof vector.
2. `table[S(t)].append(b_t*)`, where `S(t)` is the scope key (the label tuple).
3. Store nothing else.  No examples, no logits, no per-task state for `t' < t`.

At inference, the scope is read from the prompt's option list, and:

* **SSO(m = 1)** applies the Chebyshev centre of `table[S]`.
* **SSO(m = 4)** applies the nearest codeword of an m = 4 codebook built over
  `table[S]`, routed by prototype — the Phase-2K router, which needs no task ID.

Stored state: `Σ_S |table[S]| · (K_S − 1)` floats.  On Order-4 that is 12 tasks × ≤ 2
dof ≈ 20 floats, against 2 359 296 trainable LoRA parameters.

**The risk this phase exists to measure — staleness.**  `b_t*` is fit against the model
as it stood at stage `t` and applied to the model as it stands at stage 15.  The
rehearsal version refits against the final model and does not pay this.  Whether a
stale table survives 14 further tasks of drift is not predictable from anything already
measured, which is why the criteria below are worth freezing.

## 2. Metrics

Scored on the `audit` split after the final task, restricted-argmax balanced accuracy,
mean over the 12 scorable tasks.  All offsets fit on train-derived `risk`.  Never
`risk` as a reported level, never official `test.json`.

* **`R_sso1`** — SSO(m = 1), stale table, Chebyshev centre.  **Primary.**
* **`R_sso4`** — SSO(m = 4), stale table, prototype router.
* **`R_shr_global`** — one global offset, refit at the final stage.  The deployable
  baseline this must beat.
* **`R_shr_scoped`** — the rehearsal per-scope table.  The **ceiling**, not a rival:
  it re-reads old data and SSO does not.
* **`R_raw`**, **`R_orc`** — reported always, as in 2J/2K/2L/2M/2N.

All six come from the **same run**, so no cross-run reproducibility assumption is
needed.  `run_qoc.py` has `torch.manual_seed` only and no deterministic-algorithms
flag, so a trajectory is *not* assumed reproducible across runs and no criterion below
compares across runs.

## 3. Frozen criteria

3 seeds, cluster bootstrap, tasks as clusters, 10 000 draws, mean statistic.

* **P1 (primary — deployability).**  `R_sso1 − R_shr_global` CI excludes 0 on the
  positive side.  This is the whole claim: a rehearsal-free, task-ID-free offset table
  beats the single global offset.
* **P2 (staleness cost).**  `R_shr_scoped − R_sso1` ≤ 1 pp at the point estimate.
  Passing means the stale table is as good as refitting on old data; failing quantifies
  what rehearsal buys.
* **P3 (no-harm).**  No individual task loses more than 2 pp against `R_raw` under
  SSO(m = 1).  A method that raises the mean while destroying one task is not a fix.
* **P4 (does the codebook buy anything).**  `R_sso4 − R_sso1` CI excludes 0.
* **P5 (Prop 1 exact, and the staleness probe).**  On singleton scopes the Chebyshev
  centre of a one-element table **is** that element, so SSO(m = 1) there equals the
  *stale* per-task offset.  `|R_sso1 − R_orc|` on singletons therefore measures pure
  staleness with no aggregation confound.  Reported, not thresholded.
* **P6 (cost accounting).**  Stored floats equal `Σ_S |table[S]| · (K_S − 1)` and no
  example, logit, or old-task tensor is retained.  Asserted by test, not by inspection.
* **Training gate, unconditional.**  Median post-training `R_raw` > 0.60, as in
  2K/2L/2M/2N.

## 4. Predeclared outcomes

1. **P1 and P2 pass.**  The positive result: a deployable output-layer fix that needs
   neither task ID nor stored examples, matching the rehearsal ceiling, at ~20 floats
   against 2.36M parameters.  Paired with 2L/2M/2N this is a complete paper — the
   constraint is real, three parameter-space interventions cannot touch it, and the
   cheap output-layer fix can.
2. **P1 passes, P2 fails.**  Still positive and still deployable; the staleness cost is
   then a measured quantity and the ceiling gap must be stated in the abstract, not
   buried.
3. **P1 fails and P5 shows large staleness.**  The table goes stale under drift.  That
   is a clean, publishable *why* for the negative branch and it is reported as such.
4. **P1 fails and P5 shows small staleness.**  Then the loss is in aggregation across
   tasks in a scope, not in drift, and the honest conclusion is that the +1.51 pp of
   §0 is a refitting artefact.  Say so.

## 5. What is already known, frozen here so it cannot be re-labelled later

* Rehearsal `R_shr_scoped − R_shr_global` = **+1.51 pp CI [+0.43, +2.74]** on the
  converged baseline.  Known before freezing.  P1 is about `R_sso1`, which is **not**
  known.
* Rehearsal m = 4 buys **+0.00 pp** over m = 1.  So **P4 is expected to fail** and is
  frozen anyway to keep the theory's quantization rung falsifiable rather than quietly
  dropped.
* On multi-task scopes alone, per-scope − global is +1.17 pp CI [−0.09, +2.91], **not
  significant**, and ORACLE − per-scope is +0.04 pp CI [−1.71, +1.71].  So most of the
  +1.51 pp comes from singleton scopes, where scope identifies the task.  **This is
  disclosed as the mechanism, not hidden**: the claim is "scope is the right index",
  and Prop 1 is exactly why that is not a triviality.
* Phase-2N §A2.7 withdrew a +2.95 pp point estimate for lacking an interval.  Every
  number in this phase's amendment carries a CI or is labelled a point estimate.

## 6. Anti-artifact checks, before any criterion is quoted

* **No old data in the table** — asserted by test: fitting task `t`'s entry may touch
  only `partitions[t].risk`.
* **Chebyshev centre matches Prop 2** — on a binary scope it must equal
  `(min + max) / 2` of the scalar optima; asserted by test against the closed form.
* Every stored and applied offset gauge-fixed to zero mean.
* SSO and the rehearsal ceiling scored on the **same** audit logits in the same stage.
* `order4_data.py` untouched, SHA-pinned; Phase-2I frozen-dependency tests pass.
* `risk`/`audit` disjointness re-asserted per stage; official `test.json` unread.
* Trainable parameter count identical to `none` (2 359 296), logged.

## 7. Compute

3 seeds × 15 tasks, `--cl-method none` plus offset recording, ~140 min/seed, cards
0/2/3, strictly non-preemptive self-wait on `memory.free ≥ 6000`.  No λ to select:
the method has no hyperparameter.  That is itself part of the claim.

---

## A1 — the confirmatory run landed on outcome 3, and the failure is larger than "stale" (2026-08-31)

`runs/phase2p_sso_s{1,2,3}`, 3 seeds × 15 tasks, 84–87 min/seed on cards 0/2/3,
`--cl-method none --record-sso`.  Judged by
`experiments/phase2p_sso/decide_sso.py` (18 tests), output
`runs/phase2p_decide.json`.  Reported before any follow-up run exists.

### A1.1 Criteria

| criterion | result | value |
| --- | --- | --- |
| training gate | **PASS** | median post-training `R_raw` 0.6914 > 0.60 |
| **P1** primary: `R_sso1 − R_shr_global` | **FAIL** | **−5.68 pp**, CI [−8.41, −3.16] |
| **P2** staleness cost: `R_shr_scoped − R_sso1` | **FAIL** | +7.18 pp, CI [+4.41, +10.07] |
| **P3** no-harm vs `R_raw` | **FAIL** | worst QQP **−14.84 pp**; 11 of 12 tasks negative |
| **P4** `R_sso4 − R_sso1` | **FAIL** (predeclared expected) | +0.52 pp, CI [−0.46, +1.74] |
| **P5** singleton staleness (reported, not thresholded) | — | mean **6.16 pp**, max 15.23 pp, 4/15 exactly 0 |
| **P6** stored floats | PASS | 35 floats, formula-asserted, no examples retained |

Levels, mean over the 12 scorable tasks: `R_raw` 0.7016, `R_shr_global` 0.7077,
**`R_sso1` 0.6509**, `R_sso4` 0.6561, `R_shr_scoped` 0.7227, `R_orc` 0.7229.

**§4's predeclared outcome 3 fires**: P1 fails and P5 shows large staleness.

### A1.2 The result is worse than the outcome-3 text anticipated

§4's outcome 3 says "the table goes stale under drift", which describes a method
that loses its advantage.  What happened is stronger: **SSO(m=1) is 5.08 pp below
doing nothing at all** (`R_sso1` 0.6509 vs `R_raw` 0.7016), and P3 shows 11 of 12
tasks are individually damaged — QQP −14.84 pp, SST-2 −7.81, AGNews −7.16, MNLI
−6.42, DBpedia −6.36.  Only BoolQA improves (+0.78 pp).  A stale offset is not a
weakened offset; it is an actively wrong one.  Applying it is worse than applying
none, which is a stronger negative than the protocol's own wording predicted, and
it is stated here in the direction that is against the method.

### A1.3 Where the damage comes from, and what rules it out

The §5 split, all three quantities from the same run:

| comparison | point | CI |
| --- | --- | --- |
| `R_sso1 − R_shr_global`, multi-task scopes | −6.75 pp | [−10.71, −2.88] |
| `R_sso1 − R_shr_global`, singleton scopes | −4.18 pp | [−6.61, −1.76] |
| `R_shr_scoped − R_sso1`, multi-task scopes | +7.91 pp | [+3.98, +12.12] |

**The singleton column is the diagnosis.**  On a singleton scope the Chebyshev
centre of a one-element table *is* that element (Prop 1, and P5 confirms it:
`sso1_radius` = 0 and 4 of 15 measurements are exactly 0), so there is **no
aggregation, no routing, and no conflict** — the only difference from the oracle
is *when* the offset was fit.  Singletons still lose 4.18 pp with a CI excluding
zero.  Therefore:

* The loss is **not** caused by aggregating disagreeing scope-mates.  It appears
  where there is nothing to aggregate.
* It is **not** caused by the router.  m=1 uses none, and P4 shows m=4 adds
  +0.52 pp, CI [−0.46, +1.74] — nothing.
* It is **not** caused by the gauge.  Every stored and applied offset is
  zero-mean by construction and by test.
* It **is** caused by fitting `b_t*` against `θ_t` and applying it at `θ_15`.
  P5's mean singleton gap of 6.16 pp (max 15.23 pp on AGNews) is that quantity
  measured with every other explanation removed.

### A1.4 What this retires

**The streaming route to a rehearsal-free per-scope table is closed at this
budget on this benchmark, and I am not proposing a variant of it.**  The reason is
not that P1 failed by a little; it is that the failure mechanism is
budget-independent in the wrong direction.  A stored optimum's value decays with
the number of subsequent tasks, and the whole point of a continual method is that
that number grows.  Any fix requires re-touching old tasks' data (rehearsal, which
the protocol forbids) or a drift model that would need its own frozen protocol,
its own gate, and its own confirmatory seeds — and §0's licensing measurement
(+1.51 pp, CI [+0.43, +2.74]) is the entire prize.  A mechanism whose ceiling is
+1.51 pp and whose measured cost is −5.68 pp does not justify that.

**What §0's +1.51 pp now means.**  It is *not* retracted — `R_shr_scoped`
0.7227 vs `R_shr_global` 0.7077 reproduces in this run to four decimals.  But
§4's outcome 4 warned that the honest reading might be "a refitting artefact",
and A1.3 shows precisely that: the +1.51 pp requires *fitting against the current
model*.  It is real, it is reachable with rehearsal, and it is **not** reachable
without it by storing offsets.  That distinction is now measured rather than
assumed, and it is what this phase contributes.

### A1.5 Sayable / not sayable

Sayable: on this stream and budget, a rehearsal-free streaming scope-offset table
is *harmful* — 5.08 pp below no offset at all, 5.68 pp below a single global
offset, damaging 11 of 12 tasks; and singleton scopes isolate staleness as the
cause (4.18 pp loss with no aggregation, routing, or gauge confound possible).

Sayable: the per-scope identification gain of +1.51 pp is real but requires
refitting against the current model, i.e. it is available to rehearsal and not to
offset storage.

Not sayable: that no rehearsal-free output-layer method can work.  One specific
construction (store per-task optima, apply the Chebyshev centre) failed, with a
measured mechanism.  Nothing here bounds a method that maintains its table using
only current-task data — that remains open and unattempted.

Not sayable: any claim from `R_sso4`.  P4 fails, and its point estimate rides on
the same broken table.

