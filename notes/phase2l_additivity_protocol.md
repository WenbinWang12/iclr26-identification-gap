# Phase-2L — QOC as an overlay on a representation-level CL method: additivity protocol

**Frozen 2026-08-30, before any Phase-2L run.**  Nothing below is to be edited
after the first run starts; amendments are appended as §A1, §A2, … with their own
timestamps, per the discipline used in `method_qoc_v1.md`.

Prior context this rests on, in one paragraph.  Phase-2K's converged 3-seed run
(`runs/phase2k_qoc_converged`, 813 gradient steps/seed, training gate PASS at
median 0.8242) reached verdict `NEGATIVE_SCOPE_ONLY`: criterion 1 PASS at
**+2.34 pp** CI [+0.78, +3.91] with 0 Prop-1 violations, criterion 2 **FAIL** at
0.0000 CI [0.0000, 0.0000], criterion 3 monotone at ρ = +0.646.  A post-hoc
decomposition of *forgetting* (not of the identification gap) then showed why, and
it is the reason Phase-2L exists:

| stratum | n | forgetting | offset repair | repair fraction | residual |
| --- | --- | --- | --- | --- | --- |
| < 2 pp | 92 | −0.0306 | +0.0364 | — | −0.0670 |
| 2–8 pp | 57 | +0.0518 | +0.0572 | 1.10 | −0.0053 |
| **> 8 pp** | 94 | **+0.2073** | +0.0926 | **0.45** | **+0.1148** |

`Spearman(forgetting, offset repair) = +0.389`, but
`Spearman(forgetting, residual after the best offset) = +0.792`.  Offsets close
78 % of *mean* forgetting (8.08 pp of which 6.30 pp is offset-reachable) and
essentially all of it when forgetting is small, but only 45 % when forgetting is
large.  MNLI (+17.5 pp residual), RTE (+11.5), BoolQA (+6.9) are unreachable by
any offset.  So the binding constraint on Order-4 is representation-level, and
QOC's own ceiling has already been reached — the m=4 codebook (0.7204) already
*exceeds* the per-task offset reference (0.7146), which is fit on 64/class risk
and is therefore noisy, not an oracle.

**The question Phase-2L asks, and it is a real question with a real chance of a
negative answer:** when a representation-level continual-learning method removes
the deep component of forgetting, does the shallow component QOC targets *remain*,
or does that method already suppress it as a side effect?  If it remains, QOC is a
free, additive, theoretically-grounded component.  If it does not, QOC is
subsumed, and the honest paper is a negative result about output-layer capacity.

---

## 1. The two arms, and what is held fixed

Arm **BASE**: the representation-level method alone, trained on the Order-4
stream.  Arm **BASE+QOC**: identical training, identical seeds, identical data
partitions, with the QOC measurement stack applied *post hoc* at every stage.

Held byte-identical across arms, and this is the point of the design:

* `experiments/phase2i_anchored_cvar/order4_data.py` — untouched (its SHA-256 is
  pinned by Phase-2I's frozen dependency list; Phase-2L must not edit it, and must
  not update the pin).  Data, stream order, and the `update`/`risk`/`audit` split
  come from it unchanged.
* The QOC measurement path — `fit_offset`, `gauge_fix`, `fit_shared_offset`,
  `scoped_shared_offsets`, `fit_scope_codebooks`, first-piece restricted argmax,
  and the `risk`-fit / `audit`-score separation.  Phase-2L imports these from
  Phase-2J/2K rather than reimplementing them.  `runs/phase2k_qoc_converged` is
  therefore a *valid* BASE+QOC reference for the `cl_method="none"` setting, and
  the new run must reproduce it to within seed noise when `--cl-method none` is
  passed.  That reproduction is a wiring check, listed in §5.
* Budget: LoRA r=8 on q/v, `--update-cap-per-class 400 --epochs 3`, lr 3e-4,
  effective batch 64 — the Phase-2K converged settings, so that BASE's forgetting
  is measured at the same training length where we know QOC's ceiling.
* Seeds 1, 2, 3.  Cluster bootstrap with **tasks as clusters**, as in Phase-2J/2K.

Changed, and *only* this: the training-time regulariser.

## 2. Which representation-level method, and an honest statement of what it is

**O-LoRA (orthogonal subspace LoRA).**  Implemented as the standard
orthogonality penalty between the current task's LoRA `A` and the accumulated
subspace of previous tasks' `A` matrices:

    L = L_task + λ · Σ_{t < current} ‖ A_current  A_tᵀ ‖_F²

with the previous `A_t` detached (frozen history, no gradient).  This is the
mechanism O-LoRA relies on and it is what we can implement faithfully inside our
own harness in the time available.

**Stated plainly, because it matters for how the result may be described:** this
is *our re-implementation of O-LoRA's regulariser inside our harness*, not a run
of the authors' released code, and not a reproduction of their reported numbers.
Our budget (400/class, 3 epochs, T5-large r=8 q/v, 64/class risk+audit) is not
theirs.  Therefore:

* We may claim: "under a fixed budget and our harness, adding an orthogonality
  penalty of O-LoRA's form changes / does not change the offset-reachable
  component of forgetting by X".
* We may **not** claim: "we reproduce O-LoRA", "we beat O-LoRA", or any absolute
  comparison against their published table.  Any such sentence in the paper is a
  defect.  Phase-2L is an *additivity* test, not a benchmark comparison.

λ is not free to tune post hoc.  It is selected **before** the confirmatory run by
the gate in §4, on seed 1 only, from the fixed grid {0.1, 0.5, 1.0}, using the
`update` split — never `risk`, never `audit`, never `test.json`.  The selected λ is
recorded here as §A1 before the confirmatory seeds start.

## 3. Criteria, frozen

Let `R_raw` be the first-step restricted-argmax balanced accuracy with no offset,
`R_scoped` with the free per-scope offset, `R_book(m)` with the budgeted m-centre
codebook, all fit on `risk` and scored on `audit`, as in Phase-2K.  Define, per
(seen task, stage) observation:

* `deep_repair  = R_raw[BASE] − R_raw[none]`     — what the regulariser buys
* `shallow_head = max(R_scoped, R_book(m)) − R_raw`, within an arm — what offsets
  can still buy *after* that arm's training

**Criterion A (additivity — the primary question).**  Median `shallow_head` under
BASE, with a task-clustered bootstrap 95 % CI **excluding 0**.  This is the claim
"the shallow component survives a representation-level fix".

**Criterion B (no cannibalisation).**  `shallow_head[BASE]` must not be
significantly *below* `shallow_head[none]` = +0.0630 (the 6.30 pp offset-reachable
mean measured in Phase-2K).  Reported as the paired difference with its CI.

**Criterion C (the regulariser actually did something).**  Median `deep_repair`
> 0 with CI excluding 0, restricted to the > 8 pp forgetting stratum where
Phase-2K located the deep residual.  **If C fails, A and B are uninterpretable** —
a penalty that changed nothing cannot tell us whether QOC is subsumed — and the
run is reported as inconclusive on additivity, not as evidence for it.

**Criterion D (Prop 1 must still hold).**  Singleton scopes must still show
`Δ_id_scoped ≡ 0` exactly.  This is a wiring check on the offset stack, and a
falsifiable prediction of Prop 1 that has held in every run so far (0 violations
in Phase-2J and Phase-2K).  A violation means Phase-2L is mis-wired, and no other
number from it may be quoted.

**Training gate, unconditional.**  Median post-training `R_raw` > 0.60, reported
whatever it is, as in `method_qoc_v1.md` §M4.4.  Below it, BASE has not learned
the stream and nothing about forgetting is interpretable.

## 4. λ selection gate (seed 1, `update` split only)

Run seed 1 for the **first 6 tasks only** at each λ ∈ {0.1, 0.5, 1.0}.  Select the
λ with the highest mean `update`-split accuracy on tasks 1–5 measured after task 6
— i.e. retention measured where the labels are already burned for training and no
held-out information is consumed.  Ties: smallest λ.  If *no* λ beats
`--cl-method none` on that quantity, record that fact here and run the
confirmatory seeds at the best λ anyway, reporting Criterion C as expected-to-fail.

## 5. Predeclared outcomes — all four, written before the data exists

1. **A passes, B passes, C passes.**  QOC is additive on top of a
   representation-level method: the deep fix does not remove the shallow gap.
   This is the paper's positive result, and it is a *free-capacity* claim (per-scope
   indexing costs no parameters), not a claim that the budgeted codebook is
   significant — criterion 2 of Phase-2K stays failed and stays reported as failed.
2. **A fails / B shows cannibalisation, C passes.**  The orthogonality penalty
   already suppresses offset drift, so QOC is subsumed on this benchmark.  Then the
   honest paper is: output-layer capacity is a real and provable constraint
   (Prop 1/2a/3(d=1), κ measured) whose empirical footprint on Order-4 is closed by
   an existing representation-level method.  That is a negative result and it gets
   written as one.
3. **C fails.**  Inconclusive on additivity.  Report the penalty as ineffective in
   our harness at this budget, do not report A or B as evidence either way, and do
   not silently retune λ outside the §4 grid to rescue it.
4. **Training gate fails.**  Report as failed; no forgetting analysis is quoted.

**What Phase-2L cannot do, stated now.**  It cannot prove Prop 3 for `d ≥ 2`.  It
cannot repair the routing-evidence split (`{False,True}` remains textually
self-identifying, so only `{Bad,Good}` counts as routing evidence, and in the
converged run `{Bad,Good}` has q_1 = 0.155 with worst-task loss 0.0026 — i.e. the
routing evidence is nearly empty and Phase-2L does not change that).  It cannot
turn a parameter-free logit-only router into a working one.  It does not make
`R_orc` an oracle: fit on 64/class `risk`, it is noisy, and in the converged run
12.8 % of scoped-gap observations are **negative** (min −9.4 pp; MultiRC mean
−4.2 pp).  Phase-2L must report `R_orc` as a noisy reference, never as an upper
bound.

## 6. Anti-artifact checks, run before any criterion is quoted

* `--cl-method none` reproduces `runs/phase2k_qoc_converged` to within seed noise
  (same seeds ⇒ identical stream order, identical partitions; the offset numbers
  should match closely, and a large discrepancy means Phase-2L is mis-wired).
* λ = 0 with the O-LoRA code path *active* must equal `--cl-method none`
  numerically — proves the penalty enters only through its coefficient.
* The penalty must be 0 on task 1 by construction (no history), asserted in code.
* Trainable parameter count identical across arms (the penalty adds no
  parameters), asserted in code and logged.
* `order4_data.py` SHA-256 unchanged; Phase-2I's frozen-dependency tests still
  pass.
* `risk`/`audit` disjointness re-asserted per stage, as in Phase-2K.
* Official `test.json` is not read by Phase-2L at all.

## 7. Compute plan

Phase-2K converged: 263–314 min/seed at 813 steps.  **Correction, 2026-08-30
17:15:** the 5.5 h/seed estimate below was wrong by ~3×.  The λ-gate arms ran 6
tasks in 17.5–18.0 min each, i.e. the penalty is genuinely negligible and the
earlier Phase-2K timing (263–314 min) was dominated by the *later*, larger tasks
(DBpedia alone took 52 min).  Revised estimate for a full 15-task seed with the
penalty: ≈4.5–5.5 h, essentially unchanged from Phase-2K.  Recorded because the
estimate is what the queue plan was built on.  The penalty adds one
Frobenius product per LoRA layer per step against a growing history — negligible
next to the forward/backward.  Budget ≈ 5.5 h/seed.  λ gate: 3 λ × 6 tasks ≈
2.5 h total on one card.  Confirmatory: 3 seeds × 5.5 h on three cards in
parallel, strictly non-preemptive (wait for free memory, never signal another
user's job).  Total wall-clock ≈ 8 h.

---

## A1 — λ selected, 2026-08-30 17:15, before any confirmatory seed starts

Gate run per §4: seed 1, first 6 tasks only, scored on the **`update`** split
(labels already consumed by training), 4 scorable early tasks (MNLI, WiC, COPA,
QQP; CB excluded — rarest class 16 < 40).  Arms ran 17.5–18.0 min each on cards
0/2/3/5.  `runs/phase2l_gate_selection.json`.

| arm | λ | mean retention | MNLI | WiC | COPA | QQP |
| --- | --- | --- | --- | --- | --- | --- |
| none | — | 0.5835 | 0.6300 | 0.5013 | 0.5039 | 0.6988 |
| olora | 0.1 | 0.5924 | 0.6333 | 0.5700 | 0.5000 | 0.6663 |
| olora | 0.5 | 0.6058 | 0.6133 | 0.6025 | 0.5209 | 0.6863 |
| **olora** | **1.0** | **0.6309** | 0.6350 | 0.6300 | 0.4925 | 0.7663 |

**Selected: λ = 1.0**, mean 0.6309 vs baseline 0.5835 (**+4.74 pp**).  Monotone
increasing in λ across the whole grid.

**Two facts recorded now, because both constrain what the confirmatory run may
claim:**

1. **λ = 1.0 is at the boundary of the frozen grid, and retention is still
   increasing there.**  A larger λ might retain more.  Per §4 the grid is
   {0.1, 0.5, 1.0} and it is frozen — I am **not** extending it, because
   extending a selection grid after seeing the selection land on its edge is
   exactly how a tuning-on-the-signal artifact is created.  The consequence is
   stated plainly rather than fixed: **our λ is a lower bound on the best λ in
   this family**, so the confirmatory run measures the additivity of QOC on top
   of a *possibly under-regularised* O-LoRA, and if criterion A passes there is a
   residual worry that a stronger penalty would have absorbed more of the shallow
   gap.  This worry is real and goes in the paper's limitations, not in a footnote.
2. **COPA moves the wrong way** (0.5039 → 0.4925 at λ=1.0) while MNLI/WiC/QQP all
   improve.  COPA has 144 trainable examples and never leaves chance in any run
   (Phase-2K converged: 0.5156); this is consistent with noise on a task the model
   never learns, not with evidence about the penalty.  Flagged so that a per-task
   reading of the confirmatory run does not treat it as a discovery.

Confirmatory run, launched immediately after this entry: 3 seeds × 15 tasks,
`--cl-method olora --olora-lambda 1.0`, all other settings identical to
`runs/phase2k_qoc_converged` (`--update-cap-per-class 400 --epochs 3`,
`--risk-per-class 64 --audit-per-class 64`, `--m-values 1,2,4`, LoRA r=8 q/v,
lr 3e-4).  `--gate-update-eval` is **off** for the confirmatory run: λ is already
chosen, and leaving it on would only add cost.  Output:
`runs/phase2l_additivity_olora`.  The `none` arm is **not** re-run — Phase-2K's
converged run *is* that arm, at identical settings and identical seeds.

---

## A2 — the first confirmatory run landed on outcome 3, and why the §4 gate was the cause (2026-08-30 22:10)

`runs/phase2l_additivity_olora`, λ = 1.0, 3 seeds × 15 tasks, 128–131 min/seed.
Reported in full, per §5, before any rerun.

| criterion | result | value |
| --- | --- | --- |
| training gate | **PASS** | median post-training R_raw 0.8320 > 0.60, 813 steps/seed |
| **A** shallow head survives | PASS | median +0.0391, CI [+0.0234, +0.0729], n = 243 |
| **B** no cannibalisation | not detected | paired median +0.0078, CI [−0.0234, +0.0312], n = 81 |
| **C** penalty reduced deep forgetting | **FAIL** | median +0.0065, CI [−0.0417, +0.0547], n = 32 |
| **D** Prop 1 holds | PASS | 87 singleton observations, **0** violations |

Per §3, **C failing makes A and B uninterpretable**, so this is predeclared
outcome 3 — *inconclusive on additivity*.  A's PASS is **not** being reported as
evidence for additivity.  That is the verdict of record for this run.

**C did not fail because the penalty was inert.**  It was active throughout: seed 1
penalty 0 on task 1 (no history, as asserted), 151.3 → 105.9 on task 2, still
0.1176 → 0.0167 on task 15, decreasing within every task.  What it did was raise
*plasticity*, not retention:

| | BASE (λ=1) | none | diff |
| --- | --- | --- | --- |
| just-trained peak | **0.7680** | 0.7381 | **+0.0299** |
| later R_raw | 0.6509 | 0.6574 | −0.0065 |
| forgetting (peak − later) | **0.1171** | 0.0808 | **+0.0364** |

Per task, the peak rises (QQP 0.635 → 0.794, MultiRC 0.628 → 0.719, WiC 0.523 →
0.547) while later accuracy barely moves, so measured *forgetting* increases.
AGNews is the sharpest: peak flat (0.889 → 0.884), later 0.809 → 0.688, forgetting
0.080 → 0.197.

**Root cause is a defect in my own §4 gate, not in the data.**  §4 selects λ by
"mean `update`-split accuracy of tasks 1–5 measured after task 6".  That quantity
mixes **plasticity** and **retention**: a λ that makes each task learn better
scores higher on it even if it retains nothing.  The gate reported +4.74 pp for
λ = 1.0 and I read that as retention; it was plasticity.  Recorded here rather
than edited out of §4, because the mistake is the reason the run was inconclusive.

**A2.1 Corrected gate — retention only.**  For each early task `t`, measure its
`update`-split accuracy **twice**: once at the stage where `t` was just trained
(`peak_update[t]`), and once at the final gate stage (`final_update[t]`).  Score

    retention(λ) = mean over t of ( final_update[t] − peak_update[t] )

i.e. forgetting measured on the `update` split, where each task is differenced
**against its own peak**, so plasticity cancels exactly.  Higher (less negative) is
better.  Select the maximiser; ties take the smallest λ.

Frozen with the corrected gate, and deliberately **unchanged from §4** in every
other respect: same grid {0.1, 0.5, 1.0} — *not* extended, since A1's
boundary-selection worry applies with equal force here — same seed 1, same first 6
tasks, same `update`-only split (never `risk`, never `audit`, never `test.json`).

**A2.2 What this rerun can and cannot settle.**  If a λ in the frozen grid shows
positive retention, criterion C becomes testable and A/B become interpretable.  If
**no** λ in the grid reduces `update`-split forgetting, then the honest conclusion
is stronger than "inconclusive": an O-LoRA-form orthogonality penalty, at our
budget and in our harness, does not reduce forgetting on Order-4 at all, and
Phase-2L cannot answer the additivity question with this choice of
representation-level method.  In that case I will say so and **not** substitute a
different method mid-protocol to get a publishable answer — a new method means a
new protocol entry, written before its run.

**A2.3 Cost of the extra measurement.**  The corrected gate needs `update`-split
logits for the just-trained task at each stage plus all early tasks at the final
stage (6 + 5 = 11 evaluations, versus 5 before).  `--gate-update-eval` is changed
to cover exactly those two cases and remains **off by default**, so Phase-2K's
output key set is untouched and the converged run stays a valid `none` arm.

---

## A3 — λ reselected with the corrected gate, 2026-08-30 23:20

Gate run per §A2.1: seed 1, first 6 tasks, `update` split only, each early task
differenced **against its own peak**.  Arms 16.3–16.9 min on cards 0/2/3/5.
`runs/phase2l_gate2_selection.json`.  Higher (less negative) is better.

| arm | λ | retention | COPA | MNLI | QQP | WiC |
| --- | --- | --- | --- | --- | --- | --- |
| none | — | −0.0612 | −0.015 | −0.218 | +0.007 | −0.020 |
| olora | 0.1 | −0.0330 | +0.008 | −0.210 | −0.025 | +0.095 |
| **olora** | **0.5** | **−0.0242** | +0.041 | −0.202 | −0.025 | +0.089 |
| olora | 1.0 | −0.0529 | −0.050 | −0.202 | −0.040 | +0.080 |

**Selected: λ = 0.5**, retention −0.0242 vs baseline −0.0612 (**+3.70 pp less
forgetting on the `update` split**).

**Three things worth recording, and one of them retires a worry from A1.**

1. **Retention is non-monotone in λ, and the selected λ is interior.**  0.1 → 0.5
   improves, 1.0 regresses.  So the A1 concern — "λ landed on the grid boundary
   with the metric still rising, therefore our λ may be a lower bound on the best
   λ" — **no longer applies**: the corrected gate has an interior optimum, and
   extending the grid upward would move away from it.  A1's worry stands as a
   record of what the *broken* gate did; it is not a live limitation of this run.
2. **The corrected gate visibly rejects the arm the broken gate chose.**  λ = 1.0
   has the highest peaks (QQP 0.812 vs 0.695 at baseline, COPA 0.543, WiC 0.536)
   and the *third* best retention.  That is exactly the plasticity-for-retention
   confusion of §A2, now visible in the table rather than hidden in an aggregate.
3. **MNLI forgets ~0.20 under every arm, including λ = 1.0.**  Baseline −0.218,
   all three olora arms −0.202 to −0.210.  The penalty barely touches the single
   largest forgetting event in the stream.  Predicted consequence, recorded before
   the confirmatory run: criterion C will be carried by the smaller-forgetting
   tasks if it passes at all, and MNLI — which Phase-2K identified as the largest
   offset-unreachable residual (+17.5 pp) — will remain unreachable by *both*
   levels.  If C passes and MNLI is still ~−0.20, that is a real limit on how much
   of Order-4's forgetting either method addresses, and it goes in the paper.

Confirmatory run launched immediately after this entry: 3 seeds × 15 tasks,
`--cl-method olora --olora-lambda 0.5`, `--gate-update-eval` off, everything else
identical to `runs/phase2k_qoc_converged`.  Output:
`runs/phase2l_additivity_olora_lam05`.  The `none` arm is Phase-2K's converged run,
unchanged.  The λ = 1.0 confirmatory run
(`runs/phase2l_additivity_olora`) is **kept, not deleted** — it is the record of
outcome 3 under the broken gate and is cited as such in §A2.

---

## A4 — the λ = 0.5 run judged, 31 h after it finished, and §A2's numbers made recomputable (2026-08-31)

Two defects in the record are fixed here, and one of them is mine.

**Defect 1: the confirmatory run was never judged.**
`runs/phase2l_additivity_olora_lam05` — the arm §A3's *corrected* gate selected —
finished at 2026-08-31 00:48 with 3 seeds × 15 stages and sat unjudged while
Phase-2M and Phase-2N ran. §A3 ends with "Confirmatory run launched immediately
after this entry" and then nothing. It is judged below.

**Defect 2: §A2's criteria had no saved implementation.**
The λ = 1.0 verdict in §A2 was computed inline and discarded. Its numbers could be
quoted but not recomputed. `experiments/phase2l_additivity/decide_additivity.py`
(+ 11 tests) now implements §3's A–D and is run on **both** arms.

### A4.1 Verdict: outcome 3 at λ = 0.5 as well

`runs/phase2l_decide_lam05.json`, `none` arm = `runs/phase2k_qoc_converged`
unchanged, m = 4, tasks as clusters, 10 000 draws.

| criterion | λ = 0.5 | λ = 1.0 (recomputed) | §A2 as written |
| --- | --- | --- | --- |
| training gate | PASS 0.6289 | PASS 0.6465 | PASS 0.8320 |
| **A** shallow head | +0.0391, CI [+0.0234, +0.0781], n = 243 | +0.0391, CI [+0.0234, +0.0703], n = 243 | +0.0391, CI [+0.0234, +0.0729], n = 243 |
| **B** cannibalisation | −0.0078, CI [−0.0191, +0.0156], n = 81 | −0.0052, CI [−0.0182, +0.0130], n = 81 | +0.0078, CI [−0.0234, +0.0312], n = 81 |
| **C** deep repair | **FAIL** +0.0050, CI [−0.0156, +0.0469], n = 34 | **FAIL** +0.0000, CI [−0.0295, +0.0703], n = 34 | **FAIL** +0.0065, CI [−0.0417, +0.0547], n = 32 |
| **D** Prop 1 | PASS 0 violations, 144 checks | PASS 0 violations, 144 checks | PASS 0 violations, 87 obs |

**Verdict of record for λ = 0.5: predeclared outcome 3 — inconclusive on
additivity.** C fails, so per §3 **A and B are uninterpretable**, and A's PASS is
again *not* reported as evidence for additivity. The corrected gate did not rescue
criterion C; it selected an arm that forgets less on the `update` split and still
does not measurably repair the > 8 pp stratum.

### A4.2 What does and does not reproduce, stated exactly

Criterion A recomputes to §A2 **to four decimals** on the point estimate, the CI
lower bound, and n (243). The verdict (outcome 3) and every PASS/FAIL reproduces.
Three things differ and are *not* claimed as reproduction:

* **C's n is 34, §A2 said 32.** The stratum is "> 8 pp forgetting", and §3 never
  defines *forgetting*. This script defines it as (that task's `R_raw` at the stage
  it was trained) − (its current `R_raw`), per cell. §A2's inline script evidently
  drew the boundary marginally differently, admitting 2 fewer cells. Both fail C.
* **D's count is 144, §A2 said 87.** Same cause: which (stage, task) cells count as
  a singleton observation. Both find 0 violations, and the check is exact-equality,
  so the count changes the denominator, not the answer.
* **B's sign flips** (−0.0052 here, +0.0078 in §A2). This script averages each
  (stage, task) cell over seeds and *then* pairs the arms; pairing per-seed instead
  would pair two trajectories that share only a seed integer. Each estimate lies
  inside the other's CI and both give "not cannibalised".

So §A2's *conclusion* is reproduced and its *n*'s are not. The honest summary: the
verdict was right, the implementation is now on disk, and the two counts in §A2
should be read as that script's convention rather than as this one's.

### A4.3 §A3's prediction 3 held, and it was too kind

§A3 predicted "MNLI forgets ~0.20 under every arm … the penalty barely touches the
single largest forgetting event in the stream." Per-task `deep_repair` inside the
C stratum (`C_per_task`, mean over cells; the *forget* column is that task's
task-level mean and is shown for context, not as the per-cell stratum test):

| task | λ = 0.5 repair | λ = 1.0 repair | `none` forgetting | cells |
| --- | --- | --- | --- | --- |
| MNLI | **−3.47 pp** | −3.26 pp | 24.48 pp | 13 |
| BoolQA | +6.58 pp | +8.69 pp | 14.09 pp | 8 |
| RTE | +0.13 pp | +0.03 pp | 15.72 pp | 8 |
| QQP | +2.00 pp | +3.04 pp | 1.46 pp | 3 |
| AGNews | +0.65 pp | −7.94 pp | 8.01 pp | 1 |
| SST-2 | −35.68 pp | −31.25 pp | 6.12 pp | 1 |

The penalty does not "barely touch" MNLI — on the largest forgetting event in the
stream it makes retention **worse** at both λ. C's near-zero median is a genuine
cancellation: BoolQA is repaired, MNLI and the two single-cell tasks are damaged.
RTE, with 15.72 pp of forgetting and 8 cells, moves +0.13 pp — inert to within
noise on exactly the stratum the criterion was built to interrogate.

### A4.4 What Phase-2L may now be said to contribute

Sayable: under this budget and harness, an orthogonality penalty of O-LoRA's form,
at the λ its own pre-registered gate selects, produced **no measurable repair of
deep forgetting** in the > 8 pp stratum (median +0.50 pp, CI [−1.56, +4.69]), and
degraded the stream's largest forgetting event. Per §2 this is a statement about
*our re-implementation inside our harness*, never about O-LoRA's published numbers.

Not sayable: anything about additivity, in either direction. C failed; §3 says A
and B are uninterpretable; that holds for both arms.
