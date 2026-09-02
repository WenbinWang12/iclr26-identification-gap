# Phase-2M — VLA: Verbalizer-Logit Anchoring for fixed-budget continual PEFT

**Frozen 2026-08-31, before any Phase-2M run.**  Amendments append as §A1, §A2, …
Nothing above the amendment line is edited after the first run starts.

## 0. Why this method, from measurements we already have

Three measured facts, all from `runs/phase2k_qoc_converged` and
`runs/phase2l_additivity_olora*` (6 seeds total, two λ values):

1. **Forgetting is mostly a systematic logit shift.**  Over 243 (seed, stage, task)
   observations, mean forgetting is 8.08 pp and a *per-task additive offset on the
   verbalizer logits* recovers 6.30 pp of it — **78 %**.  Offsets are `K−1`
   parameters per task.  So the dominant component of forgetting under a fixed
   LoRA budget is low-dimensional and lives in the output layer.
2. **What offsets cannot reach is concentrated in the largest events.**  Repair
   fraction is ≈1.0 when forgetting < 8 pp but **0.45** when > 8 pp;
   `Spearman(forgetting, post-offset residual) = +0.792`.  MNLI alone carries
   +17.5 pp of unreachable residual.
3. **A weight-space orthogonality penalty does not help, and we know why.**  Two
   independent λ selections (one with a defective gate, one corrected) and 6 seeds
   all give the same direction: peak accuracy rises (+2.45 to +2.99 pp) while
   retained accuracy falls (−0.65 to −0.87 pp), so measured forgetting *increases*
   by 3.3–3.6 pp.  Mechanistically: the penalty constrains `A` against a **detached**
   history, but the optimizer keeps updating `B` for every layer, so the function
   that produces old tasks' verbalizer logits is not constrained at all.  It buys
   capacity, not retention.  Recorded in `phase2l_additivity_protocol.md` §A2/§A3.

**VLA follows from (1) and (3): anchor the quantity that actually drifts.**  Not the
weights, not a subspace — the *verbalizer logit function itself*, which is both the
thing fact (1) says carries 78 % of forgetting and the thing the theory
(Prop 1/2a/3) is already stated about.  This is the first method in this project
whose regularised quantity and whose theory are the same object.

## 1. The method

Let `V_t = ⋃_{s ≤ t} Y_s` be the union of first-piece verbalizer token ids seen up
to task `t`.  Let `f_θ(x) ∈ R^{|V|}` be the model's first-decoding-step logits
restricted to `V`.

At the boundary after task `t−1`, snapshot the adapter as `θ̄` (the *previous*
model, which already encodes all earlier tasks — LwF-style, so storage is **O(1)**
in the number of tasks, not O(t)).

While training task `t` on its `update` split `D_t`:

    L = L_task(θ)  +  λ_a · (1/|B|) Σ_{x ∈ B}  KL( softmax(f_θ̄(x)[V_{t−1}] / τ)
                                                 ‖ softmax(f_θ(x)[V_{t−1}] / τ) )

**Data-free.** The anchor is evaluated on **current-task inputs only**; no examples
from earlier tasks are stored or replayed.  This matters for the fixed-budget
framing: VLA adds no parameters and no data store.

**Restricted to `V_{t−1}`**, the verbalizers of *previous* tasks — not the current
one.  Anchoring the current task's own verbalizer would fight `L_task` directly.

**Implementation: the anchor targets are precomputed once per task.**  Before
training task `t`, run one pass of `θ̄` over `D_t` and cache
`f_θ̄(x)[V_{t−1}]`.  Training then reads the cache — no second forward per step, so
the added cost is one evaluation pass per task (≈ a stage eval), not a 2× training
slowdown.  Cache size is `|D_t| × |V_{t−1}|` floats.

τ = 2.0, fixed, not tuned (standard distillation temperature).  λ_a is selected by
the gate in §4.

At inference, VLA composes with the **free** level of QOC — the scope-indexed
offset, which Phase-2K validated at +2.34 pp CI [+0.78, +3.91] with 0 Prop-1
violations.  The budgeted codebook level is *not* claimed; Phase-2K's criterion 2
failed and stays reported as failed.

## 2. Metrics — and a lesson from Phase-2L encoded as a rule

Phase-2L's criteria were stated on *forgetting*, a difference against each task's
own peak.  That is gameable: a method that raises peak scores worse on forgetting
while being no worse to deploy, and the mirror image — a method that lowers peak —
scores *better* on forgetting while being worse to deploy.  Both directions
mislead.  So Phase-2M's primary metric is a **level**, not a difference, and it is
the standard continual-learning one:

* **ACC** — mean over all 15 tasks of restricted-argmax balanced accuracy on the
  `audit` split, measured **after the final task**.  Primary.
* **ACC_scoped** — same, with the free scope-indexed offset applied.  Primary for
  the composed system.
* **BWT** — mean over tasks of (final accuracy − accuracy right after that task was
  trained).  Reported, secondary, *with* peak accuracies alongside so that a BWT
  change can always be attributed to the level or to the peak.
* **Peak (learning) accuracy** — mean post-training `R_raw`.  Reported always, so
  plasticity/retention trades are visible rather than aggregated away.

## 3. Criteria, frozen

Baseline is `runs/phase2k_qoc_converged` (`--cl-method none`, same seeds, same
settings, same measurement path).  Cluster bootstrap, **tasks as clusters**, 10 000
resamples, as in Phase-2J/2K/2L.

**Criterion 1 (primary — VLA improves the retained level).**  `ACC[VLA] − ACC[none]`
has a task-clustered 95 % CI **excluding 0** on the positive side.

**Criterion 2 (the composed system).**  `ACC_scoped[VLA] − ACC_scoped[none]` CI
excludes 0.  Together with criterion 1 this is the claim that the free offset level
and the anchor are not redundant.

**Criterion 3 (no plasticity collapse).**  Mean peak accuracy under VLA must not be
more than 2 pp below `none` (0.7381).  A method that only lowers peak would
trivially improve BWT and must not be reported as reducing forgetting.  This is the
guard that Phase-2L lacked in the opposite direction.

**Criterion 4 (theory-coupled, falsifiable).**  The anchor keeps per-task optimal
offsets closer together, so the Chebyshev radius `q_1` and the 2-centre radius
`q_2` of each conflict scope must **decrease** relative to `none`.  Measured with
the existing `scope_radii` output.  **This is a prediction that can fail while
criteria 1–3 pass**, and if it does, the mechanism story in §0 is wrong even if the
method works — that would have to be said plainly.

**Criterion 5 (Prop 1 must still hold).**  Singleton scopes have `Δ_id_scoped ≡ 0`
exactly.  0 violations in Phase-2J/2K/2L; a violation means mis-wiring and nothing
else from the run may be quoted.

**Training gate, unconditional.**  Median post-training `R_raw` > 0.60.

## 4. λ_a selection gate (seed 1, `update` split only)

Grid **{0.5, 2.0, 8.0}**, fixed.  Seed 1, first 6 tasks.  Score = mean over early
tasks of `update`-split accuracy at the final gate stage **minus** that task's own
`update`-split accuracy right after it was trained (the corrected retention
quantity of `phase2l_additivity_protocol.md` §A2.1 — plasticity cancels).  Ties
take the smallest λ_a.  Never `risk`, never `audit`, never `test.json`.

If the selected λ_a sits at a grid endpoint **with the score still improving in
that direction**, that fact is recorded in the amendment and treated as a
limitation — the grid is **not** extended after seeing the selection, for the
reason given in `phase2l_additivity_protocol.md` §A1.

## 5. Predeclared outcomes

1. **Criteria 1–3 pass.**  VLA improves the retained level without collapsing
   plasticity: a working method for fixed-budget continual PEFT, with the free
   offset level composing on top.  Criterion 4 then says whether the stated
   mechanism is also right.
2. **Criterion 1 passes, 2 fails.**  The anchor and the offset are redundant — the
   anchor already removes the systematic logit shift that offsets were repairing.
   That is still a working method, but the paper's two-level story collapses to one
   level and must be written that way.
3. **Criterion 1 fails.**  VLA does not work at this budget.  Report as a negative
   result alongside Phase-2L's, and do **not** try a third regulariser inside this
   protocol — a new mechanism needs a new protocol entry written before its run.
4. **Criterion 3 fails (plasticity collapse).**  Any BWT/forgetting improvement is
   discounted and reported as a plasticity trade, not as retention.

## 6. Anti-artifact checks, before any criterion is quoted

* λ_a = 0 with the VLA code path active must equal `--cl-method none` numerically.
* The anchor must be exactly 0 on task 1 (`V_0 = ∅`), asserted in code.
* Anchor targets come from a **frozen** snapshot: no gradient may reach `θ̄`.
* Trainable parameter count identical to `none` (2 359 296), logged.
* No examples from tasks `< t` are read during task `t`'s training — asserted by
  construction (the cache is built from `D_t` only) and tested.
* `order4_data.py` untouched; its SHA-256 pin and Phase-2I's frozen-dependency
  tests still pass.
* `risk`/`audit` disjointness re-asserted per stage; official `test.json` unread.

## 7. Compute

`none` and λ=0.5 arms of Phase-2L ran 86–131 min/seed.  VLA adds one evaluation
pass per task over `D_t` (≈ 400/class × 15 tasks) — estimate **+20–30 min/seed**,
so ≈110–120 min/seed.  Gate: 3 λ_a × 6 tasks ≈ 25 min each, 4 arms in parallel on
separate cards ≈ 30 min wall-clock.  Confirmatory: 3 seeds in parallel ≈ 2 h.
Strictly non-preemptive: wait for free memory, never signal another user's job.

## A1. λ_a selection gate — result (amendment, appended 2026-08-31)

Seed 1, first 6 tasks, order `MNLI, CB, WiC, COPA, QQP, BoolQA`.  Four scorable
early tasks (CB is not scorable on the `update` split at this cap).  Scored with
`experiments/phase2m_vla/lambda_gate_vla.py`, which **imports** Phase-2L's
`retention_score` rather than re-implementing it — a test pins
`gate_2m.retention_score is gate_2l.retention_score`, so both phases are provably
scored by one function.  Phase-2L's `select()` hard-validates `cl_method == "olora"`
and was therefore not reusable as-is; it was **not edited**, because that file is
the record of how λ = 0.5 was chosen in Phase-2L §A3.

### A1.1 The frozen gate quantity (retention: final − own peak)

| arm | λ_a | retention | COPA | MNLI | QQP | WiC |
|---|---|---|---|---|---|---|
| none | 0.0 | **−0.0612** | −0.015 | −0.218 | +0.007 | −0.020 |
| a05 | 0.5 | −0.0655 | +0.027 | −0.247 | −0.011 | −0.031 |
| a20 | 2.0 | −0.0828 | −0.040 | −0.268 | −0.031 | +0.009 |
| a80 | 8.0 | **−0.0630** | +0.001 | −0.271 | −0.010 | +0.027 |

**Selected λ_a = 8.0** (best retention among VLA arms).  It does **not** beat
`none` (−0.0630 vs −0.0612).  Per §4's standing instruction the confirmatory run
proceeds at this λ_a anyway and the retention-flavoured quantity is reported as
expected-to-fail.

**Limitation, recorded, grid not extended.**  λ_a = 8.0 is the **upper** endpoint of
{0.5, 2.0, 8.0} and its score is still improving outward (−0.0630 > a20's −0.0828).
§4 forbids extending the grid after seeing this, for the reason in
`phase2l_additivity_protocol.md` §A1.  The lower edge is *not* an open endpoint:
λ_a = 0 is a measured grid point, because it is numerically identical to `none`
(§A1.3), so the gate implementation treats the ladder as closed below.

**The selection is inside noise and I am saying so.**  a80 beats a05 by 0.25 pp on
four tasks from one seed.  Nothing in this table justifies a claim that 8.0 is
better than 0.5.  The gate's job here is only to remove the choice from my hands
after the fact, and that is all it did.

### A1.2 The level decomposition — the gate quantity disagrees with §2's primary

§2 froze **levels** as primary precisely because differences-against-peak are
gameable in both directions.  On the same four tasks:

| arm | λ_a | level (mean final) | Δ level vs none | mean peak |
|---|---|---|---|---|
| none | 0.0 | 0.5865 | — | 0.6477 |
| a05 | 0.5 | **0.6382** | **+5.17 pp** | 0.7037 |
| a20 | 2.0 | 0.6262 | +3.98 pp | 0.7090 |
| a80 | 8.0 | 0.6275 | +4.10 pp | 0.6905 |

So the gate's own numbers say VLA is *worse* while the protocol's primary metric
says it is 4–5 pp *better*.  Both are computed from the same measurements; they
differ only in whether each task's peak is subtracted.  This tension was already
present in the protocol as frozen — §2 chose levels, §4 chose a difference — and I
am recording it rather than resolving it in whichever direction flatters the
method.  **The frozen gate governs λ_a selection; §3's frozen criteria govern
whether the method works.**  Those are separate questions and get separate answers.

Per-task, the level gain is **plasticity-dominated**: WiC peak 0.521 → 0.661 and
QQP peak 0.695 → 0.833 (a05), with finals moving nearly in lockstep.  `none` sits at
chance on WiC (0.521, binary) — the anchor appears to stop the verbalizer collapsing
onto one output piece, which is a real effect but **not** the retention mechanism
§0 hypothesised.

And the anchored quantity moves the *wrong way where it should matter most*: MNLI is
task 1, has by far the largest drop, and degrades monotonically in λ_a
(−0.218 → −0.247 → −0.268 → −0.271).  If VLA worked as §0 claims, MNLI is the task
it should protect.  **Prediction for the confirmatory run, recorded before it
launches: criterion 4 (`q_1`, `q_2` must decrease) fails, and MNLI's final accuracy
under VLA is at or below `none`.**  If criteria 1–3 pass anyway, then per §3
criterion 4's wording the method works and the stated mechanism is wrong, and the
paper must say that in those words.

### A1.3 Wiring checks passed at the gate

* **λ_a = 0 equivalence** — `--cl-method vla --vla-lambda 0.0` vs `--cl-method none`:
  compared 24 values, all identical.  λ_a = 0 is numerically inert with the VLA code
  path live.
* **Anchor inert on task 1** — MNLI peak is `0.856` in *all four* arms, bit-identical,
  confirming `V_0 = ∅` and that nothing perturbs the first task.
* 30/30 Phase-2M tests pass locally and on the box.  Full local regression:
  171 passed, 1 pre-existing unrelated failure
  (`phase2_gpu_bridge/test_validation.py` run-binding schema; zero references to
  `phase2m`/`vla` in that package, and it fails identically without this phase's
  files present).

### A1.4 Confirmatory arms, pre-declared now

Cards are free (8 × ≥40 GB; the job needs ≈6 GB), so cost is not a reason to run one
arm.  Declared **before** any confirmatory seed starts:

* **Primary = λ_a 8.0**, the gate-selected value.  §3's criteria 1–5 are evaluated on
  this arm and on no other.  Whatever it says is the result.
* **Secondary = λ_a 0.5**, a sensitivity check, 3 matched seeds.  Reported in full
  next to the primary.  It exists because §A1.1's selection is inside noise, and a
  λ-sensitivity table is more honest than a single arm chosen by a 0.25 pp margin.
  It is **not** eligible to become the headline: no criterion is re-evaluated on it,
  and if it looks better than the primary that gets written down as "the gate
  selected the worse of two λ values", not as the method's score.

## A2. Confirmatory result (amendment, appended 2026-08-31)

3 seeds, 15/15 tasks, λ_a = 8.0, rc=0, 137.5–141.5 min/seed.  Baseline
`runs/phase2k_qoc_converged`, same 3 seeds, same measurement path.  Scored by
`experiments/phase2m_vla/decide_vla.py`, which imports Phase-2J's
`cluster_bootstrap_ci` (tasks as clusters, 10 000 resamples); a test pins the
function identity.  Phase-2K's `decide.py` was **not** reused: it scores 2K's
scoped *gaps*, whereas §3 scores *levels*.  44/44 Phase-2M tests pass.

### A2.1 Criteria table — primary arm λ_a = 8.0

| | criterion | verdict | value |
|---|---|---|---|
| 1 | ACC[VLA] − ACC[none] CI > 0 | **FAIL** | **−4.78 pp** CI[−7.36, −2.25] |
| 2 | ACC_scoped diff CI > 0 | FAIL | +0.28 pp CI[−2.97, +3.66] |
| 3 | peak not > 2 pp below 0.7381 | PASS | peak 0.7302, drop 0.79 pp |
| 4 | q_1 **and** q_2 decrease | **PASS** | q_1 −0.0145, q_2 −0.0125 |
| 5 | Prop 1 exact on singletons | PASS | 87 checked, 0 violations, worst 0.0 |
| — | training gate median R_raw > 0.60 | PASS | 0.7188 (n=36) |

ACC 0.6599 vs 0.7077.  ACC_scoped 0.7255 vs 0.7227.

**§5 outcome 3 fired: criterion 1 fails.  VLA does not work at this budget.**
Per §5 no third regulariser is tried inside this protocol; a new mechanism needs a
new protocol entry written before its run.

### A2.2 The two pre-registered predictions, judged

* **Prediction "criterion 4 fails" — FALSIFIED.**  I predicted in §A1.2 that the
  Chebyshev radii would not decrease.  They did: q_1 0.0761 → 0.0616 and
  q_2 0.0284 → 0.0158 over 8 shared scopes, both directionally down (neither CI
  excludes 0, so the effect is directional rather than significant).  The anchor did
  to the radii exactly what §0 said it would.  It just did not help ACC.
* **Prediction "MNLI ≤ none" — CONFIRMED.**  MNLI 0.5104 vs 0.5139, −0.35 pp.

So the mechanism story was right about its own quantity and wrong about the
consequence.  Tightening the conflict radii is **not sufficient** to raise the
retained level — which is a genuine, reportable constraint on the theory chain, not
a wiring bug.

### A2.3 Where the damage lives — the finding that supersedes the method

| arm | ACC (no offset) | ACC_scoped (free offset) | offset gain |
|---|---|---|---|
| none | 0.7077 | 0.7227 | +1.51 pp |
| VLA λ_a=8.0 | 0.6599 | 0.7255 | **+6.56 pp** |

VLA costs 4.78 pp of raw level; after the **free, zero-parameter** per-scope
constant, the gap to `none` is +0.28 pp with a CI containing 0.  The fraction of
VLA's damage removed by a per-scope constant is **1.059** — all of it, to within
noise.

**VLA's damage is a pure gauge error.**  It lives entirely inside the subspace a
per-scope constant spans.  This is diagnostic, not cosmetic: the KL anchor is taken
over the softmax across the *union* `V_{t−1}` of all previous verbalizer pieces,
while every deployed decision is an argmax over one task's *own* verbalizer subset.
Those differ by exactly a per-scope additive constant — the gauge of
`theory_output_layer_capacity_v1.md` §2.  So the anchor spends its budget holding
cross-scope offsets that no decision reads, and pays for it in the within-scope
contrasts that every decision reads.  The gauge-fixed contrast is the quantity Prop
2a is stated about, and it is *not* what VLA regularises.

Recorded as a limitation of this measurement: the 1.059 point estimate comes from
two 3-seed means and is not bootstrapped; the defensible claim is "removed to within
noise", not "removed 105.9 %".

### A2.4 Secondary arm λ_a = 0.5, reported in full

Seed 1 only (seeds 2 and 3 are still running, sharing cards with another user's
jobs; strictly non-preemptive).  Against the same-seed baseline: ACC 0.6975 vs
0.7540, **C1 −5.66 pp** CI[−9.81, −1.77] FAIL; ACC_scoped 0.7154 vs 0.7737,
**C2 −5.83 pp** CI[−9.16, −2.70] FAIL; C3 PASS (peak 0.7960, *above* the reference);
C4 PASS (q_1 −0.0365, q_2 −0.0232); C5 PASS 0/29; gate PASS 0.8477.

Per §A1.4 this arm is not eligible to be the headline and no criterion is
re-evaluated on it.  It is worse than the primary on both level criteria, so the
gate did not select the worse of the two λ values.  Its C2 failing while the
primary's C2 is ~0 is the one place the two arms disagree qualitatively, and with
one seed I am not going to interpret it.

### A2.5 What this licenses saying, and what it does not

Sayable: on Order-4 at this budget, with a converged trunk, **two** distinct
regularisers — an O-LoRA-form orthogonality penalty on `A` (Phase-2L) and a KL
anchor on previous verbalizer logits (Phase-2M) — fail to raise the retained level,
and each fails for an identified reason (2L constrains `A` while `B` stays free;
2M constrains the gauge rather than the gauge-fixed contrast).  Alongside Phase-2K's
achievability ladder this is a coherent negative-with-mechanism story about where the
remaining headroom on this benchmark is *not*.

Not sayable: that VLA "reduces forgetting" (criterion 1 fails), that tightening q_m
improves accuracy (criterion 4 passed while criterion 1 failed — that is the
counterexample), or that any of this reproduces O-LoRA (Phase-2L §2).

Protocol SHA-256 after this amendment is recorded in `notes/phase2m_sha.txt`;
the pre-amendment SHA was `7a0c74795260106da0104e3fa00a6d8a94f39849e85665105616fb5f68bd0ec1`
and before §A1 `b3225028f31e6e4ec98ed950f6874651d9e808c9e3e7ba8d6040678b2fbd0b5b`.

### A2.6 Secondary arm λ_a = 0.5 completed to 3 seeds — appended 2026-08-31

Seeds 2 and 3 finished. §A2.4 above is left exactly as written; this subsection
supersedes its numbers and **must be read with it**, because the two disagree and
the disagreement is the point.

`decide_vla --treat runs/phase2m_vla_lam05 --base runs/phase2k_qoc_converged
--seeds 1,2,3 --m 4`, prototype router, m=4:

| criterion | 1 seed (§A2.4) | 3 seeds | verdict |
| --- | --- | --- | --- |
| C1 ACC vs base | −5.66 pp, CI[−9.81, −1.77] | **+0.278 pp**, CI[−1.241, +1.903] | FAIL both, *for opposite reasons* |
| C2 ACC_scoped vs base | −5.83 pp, CI[−9.16, −2.70] | **+0.919 pp**, CI[−0.296, +2.276] | FAIL both |
| C3 plasticity | PASS, peak 0.7960 | PASS, peak 0.7803 vs ref 0.7381 | PASS |
| C4 q_m | PASS, q_1 −0.0365, q_2 −0.0232 | PASS, q_1 −0.0166 CI[−0.0906,+0.0409], q_2 −0.0195 CI[−0.0586, 0.0] | PASS, neither significant |
| C5 Prop-1 singletons | PASS 0/29 | PASS 0/87, worst gap 0.0 | PASS |
| training gate | PASS 0.8477 | PASS 0.8255, n=36 | PASS |

Levels, 3 seeds: ACC 0.7104 treat vs 0.7077 base; ACC_scoped 0.7319 vs 0.7227.

**What changed and what it costs us.** At one seed this arm looked like a −5.66 pp
disaster. At three it is statistically indistinguishable from the baseline with a
*positive* point estimate. C1 still fails, but the failure mode flipped from
"harmful" to "no detectable effect", and §A2.4's own refusal to interpret one seed
was the correct call.

Two consequences we are obliged to carry forward:

1. **The λ-dependence of VLA's harm is now on the record.** The primary arm
   (λ_a = 8.0, the value the §A1 gate selected) is −4.78 pp and remains the only
   headline-eligible reading per §A1.4. But "the anchor is harmful" is not a
   property of the anchor; it is a property of the anchor *at the λ the gate
   picked*. Any paper sentence asserting harm must attach the λ or say "at the
   gate-selected λ". This is a wording constraint on text already drafted.
2. **A one-seed level difference of ~6 pp survived none of the added seeds.** That
   is a calibration fact about this harness worth remembering the next time a
   single-seed number looks decisive.

No criterion is re-evaluated on this arm and it does not become the headline;
§A1.4 forbids both and this amendment does not revisit that.
