# Output-layer capacity under task-agnostic inference — formal statement v1

Draft 2026-08-29.  Nothing here is verified by experiment yet.  This note states
the theory track only; the empirical falsification test lives in
`notes/phase2j_offset_conflict_probe_protocol.md`.

## 0. Why this note exists

The decomposition of forgetting into a representation term and an output-layer
term is **established prior art** in vision continual learning: Davari et al.
(CVPR 2022) define representation forgetting via an optimal linear probe;
"Heads collapse, features stay" (arXiv 2512.07400) names the split shallow vs
deep and adds Neural-Collapse asymptotics; "Lost or Hidden?" (arXiv 2605.16374)
measures recoverability with a least-squares affine map.  We therefore do **not**
claim the decomposition.

All of that work shares two structural assumptions:

1. a per-task (or per-class) linear head exists and may be re-fitted, and
2. at evaluation time the task identity of a query is known.

Under (1)+(2) the output-layer term is *cheap*: re-fit the head.  This note
formalises what happens when both fail, which is the standard setting for
continual PEFT of a generative LLM with a single shared decoder and no
inference-time task router.  The claim is that the output-layer term stops being
a calibration nuisance and becomes a second, independent **capacity** constraint
— one that the existing rank-based forgetting bounds do not see.

## 1. Setting

A frozen backbone with adapter parameters θ produces, for input x, a hidden
state h_θ(x) ∈ R^d.  Decoding is by a frozen unembedding U ∈ R^{V×d} over the
shared vocabulary; for a classification-style task g with verbalizer set
Y_g ⊂ [V], the prediction is

    ŷ_g(x) = argmax_{y ∈ Y_g}  u_y^T h_θ(x)          (raw decoding)

Tasks g = 1..T arrive in a stream.  Let θ_g^post be the parameters right after
task g finished training, and θ_t the parameters at a later time t > g.

A **task-agnostic offset** is a single vector b ∈ R^V, applied identically to
every query with no knowledge of which task the query came from:

    ŷ_g^b(x) = argmax_{y ∈ Y_g}  ( u_y^T h_θ(x) + b_y )

The budget accounting is: b contributes |∪_g Y_g| scalars in total (for the
15-task Order-4 benchmark, at most 210), against 147,456 scalars for a rank-4
adapter bank on T5-small q/v — 0.14%.  So b is *parametrically* negligible.
The content of this note is that being parametrically negligible does not make
it *free*, because the binding constraint is not parameter count.

## 2. The two terms, restated for this setting

For task g at time t define
- raw retention        R_g^raw(t)  = acc_g(θ_t, no offset)
- oracle-offset        R_g^orc(t)  = max_{b_g ∈ R^{|Y_g|}} acc_g(θ_t, b_g)
                                     (a *separate* offset per task; requires
                                      task identity — this is the vision-CL
                                      "re-fit the head" quantity)
- shared-offset        R_g^shr(t)  = acc_g(θ_t, b*) where b* is one vector
                                     shared by all tasks, chosen once

Immediately R_g^raw ≤ R_g^shr ≤ R_g^orc.  Prior work studies the gap
(R_g^orc − R_g^raw): "shallow forgetting", declared cheap.  This note studies

    Δ_g^id(t) = R_g^orc(t) − R_g^shr(t) ≥ 0                    (identification gap)

the part of shallow forgetting that is **not** recoverable without task
identity.  Claim: Δ^id is the object that matters for task-agnostic continual
PEFT, it is invisible to the vision-CL framing (where it is zero by assumption),
and it is invisible to rank-based bounds (Section 4).

## 3. Conflict lower bound (target result)

Intuition: each task g wants its verbalizer logits shifted in a particular
direction; a single shared b must serve all of them; when the wanted directions
disagree, no shared b can satisfy all tasks at once, and someone must lose.

**Setup.** For each task g let b_g^* ∈ R^V be an optimal per-task offset
supported on Y_g.  Stack them as columns of B* = [b_1^*, ..., b_T^*] ∈ R^{V×T}.
Let P be the set of admissible shared offsets; if we additionally cap the
offset's own capacity at rank/dimension m (e.g. b constrained to an
m-dimensional subspace, or m distinct shared offsets selected by a task-free
gate), write P_m.

**Proposition 1 (disjoint-verbalizer case, exact).**  If the verbalizer sets Y_g
are pairwise disjoint, then a single shared b can realise every b_g^*
simultaneously, so Δ_g^id = 0 for all g.  Hence conflict requires
**verbalizer sharing** across tasks.

This is the honest boundary of the result and it must be stated first: the
whole phenomenon is driven by tasks that reuse the same output tokens.

**Verified on the pinned Order-4 task specs (2026-08-29), so the precondition is
not hypothetical.**  Exact shared-verbalizer groups:

| shared label set | K | tasks |
|---|---|---|
| `{false, true}` | 2 | WiC, QQP, BoolQA, MultiRC |
| `{contradiction, entailment, neutral}` | 3 | MNLI, CB |
| `{bad, good}` | 2 | IMDB, SST-2 |
| `{very negative, negative, neutral, positive, very positive}` | 5 | Yelp, Amazon |

**This table is string-level and therefore incomplete.**  Measured at the
tokenizer level (`runs/phase2j_verbalizer_audit.json`, 2026-08-29,
`google-t5/t5-large`) there are **14** shared first pieces against 12 shared
label strings, because `b` is indexed by vocabulary token, not by label text.
Four collisions are invisible above and must be in the accounting:
piece 71 (`COPA:A` / `DBpedia:Athlete`), 1769 (`AGNews:Business` /
`Yahoo:Business & Finance`), 2854 (`AGNews:Science or Technology` /
`Yahoo:Science & Mathematics`), and 182 (`▁very`, shared by *both* of
Yelp/Amazon's `very negative` and `very positive`).

That last one is not a detail: Yelp and Amazon's 5-way set has two labels with
the **same first piece**, so no first-step offset can separate them and both
tasks fall outside this section's setting.  See §5 and probe protocol
Amendment A1.  The last row of the table above is therefore *not* a usable
conflict group; the usable ones are rows 1–3 plus `Sports` (AGNews/Yahoo).

Per-token reuse is heavier still: `neutral` appears in 4 tasks (MNLI, CB, Yelp,
Amazon — i.e. across *different* groups), `entailment`/`contradiction` in 3
(MNLI, CB, RTE, where RTE drops `neutral` and so is a strict subset case), and
`sports` in 2 (AGNews, Yahoo).  The stream is 8 binary + 7 multiclass tasks.

Two consequences worth noting now.  First, the largest conflict group is exactly
the `true/false` quartet, which contains all three tasks Phase-2I already
struggled on (QQP, BoolQA, MultiRC) — the observed boundary drift sits precisely
where the theory predicts conflict.  Second, RTE ⊂ MNLI/CB label sets and
`neutral` bridging the NLI and sentiment groups mean the groups are not cleanly
separable, so the shared offset is coupled across groups; the accounting must be
per shared *token*, not per group.

**Proposition 2 (shared-verbalizer conflict).**  Suppose tasks g ∈ S all share
the same verbalizer set Y, and let b_g^* restricted to Y be the per-task optima.
Then for any shared b,

    max_{g ∈ S} [ R_g^orc − R_g^b ]  ≥  ω( { b_g^* } )

where ω is a spread functional of the per-task optima (a diameter-type term:
zero iff all b_g^* coincide).  The corresponding worst-task statement is a
minimax over a finite set of points, so the bound is attained by a Chebyshev
centre; the quantitative form of ω requires the margin-density assumption of
Section 5.

**Proposition 3 (budgeted offsets) — corrected form.**  My first draft of this
proposition asserted a spectral-tail bound

    max_g [ R_g^orc − R_g^{P_m} ]  ≥  Φ( σ_{m+1..r}(B*) )

by analogy with Eckart–Young, so that the Track-1 weight-subspace theorem would
be reused verbatim.  A numerical structural check (2026-08-29) shows this is
**the wrong shape** and the analogy does not survive:

- For a **binary** shared verbalizer, adding a constant to all of Y leaves the
  argmax unchanged, so the offset acts only through the single scalar
  s = b_{y1} − b_{y0}.  Per-task optima are then *scalars*, B* is 1×T, and there
  is no nontrivial singular-value tail at all.  Prop 3 degenerates to Prop 2.
  This case is not a corner case — it is `true/false`, shared by BoolQA,
  MultiRC, WiC and RTE in Order-4.
- The real constraint is not that b lies in a low-dimensional subspace.  It is
  that b **cannot be selected per task**.  The correct budgeted object is
  therefore "at most m task-free offsets, with a task-agnostic rule choosing
  among them", which is a **quantization** (worst-case clustering) problem over
  the set {b_g^*}, not a low-rank approximation of B*.

Corrected statement.  For a shared-verbalizer group with per-task optima
{b_g^*} ⊂ R^{K-1} (gauge-fixed by removing the additive constant), and a budget
of m task-free offsets,

    max_g [ R_g^orc − R_g^{P_m} ]  ≥  Ψ( q_m( {b_g^*} ) ),
    q_m(X) = min_{|C| = m}  max_{g}  dist( b_g^*, C )

the **worst-case m-quantization radius** of the per-task optima, with Ψ
nondecreasing and carrying the margin density κ of §5.  m = 1 recovers Prop 2
with q_1 = Chebyshev radius.

Numerical support (synthetic, K=6, T=8): q_m is monotone decreasing in m
(4.85 → 1.59 → 0.57 → 0.34 for m = 1..4), which is the right shape for a budget
bound; and in the binary conflict example of Prop 2 the minimax shared offset
lands at the Chebyshev centre (+0.025 against a midpoint of +0.033) with a
15.1 pp identification gap on the two conflicting tasks.

**Consequence for the paper's story.**  The output-layer budget is *not* a
second copy of the spectral-tail theorem, and we must stop saying it is.  It is
a covering/quantization constraint of a genuinely different type.  That is
arguably a better outcome — the two capacity constraints are then not redundant
— but it means Track-1 is *motivation*, not a donor proof, and the paper cannot
claim theorem reuse.

Status (updated 2026-08-29): Prop 1 immediate.  **Prop 2 is now proved** in
`notes/theory_prop2_proof.md`, but the proof forced a correction to the
statement above and that correction matters:

- As written here, Prop 2 bounds an *accuracy* difference below by a
  diameter-type `ω`.  That cannot hold unconditionally — accuracy differences
  are at most 1 while a diameter is unbounded.  The statement splits into
  **Prop 2a** (logit space: worst-case distance to the per-task optima is at
  least the Chebyshev radius `r_S ≥ diam/2`; exact, assumption-free, attained
  uniquely at the Chebyshev centre) and **Prop 2b** (accuracy space:
  `max_g [R_g^orc − R_g(b)] ≥ κ · min(r_S, t_0)` under the §5 margin-density
  assumption).
- So the placeholder `ω` is discharged as a **product** `κ · min(r_S, t_0)`, not
  as a pure geometric quantity.  The geometric half is unconditional; the
  accuracy half is not, and §5's κ must be measured, not assumed.
- The κ-assumption is **necessary**: `theory_prop2_proof.md` §4 gives a
  counterexample (two tasks with margins ±10) where the distance is large,
  then shrinks 1000×, while the accuracy loss stays exactly 1 — so no
  assumption-free proportionality between distance and loss exists.  This is
  verified numerically in `tests/test_prop2.py` (6/6 pass), along with
  `q_1 = ` Chebyshev radius, `r_S ≥ diam/2`, `r_S = 0` iff optima coincide, and
  monotonicity of `q_m` in `m`.

Prop 3 status, **updated 2026-08-30** (was: "`m ≥ 2` remains open" without
qualification).  The `d = 1` case is now closed in
`notes/theory_prop3_m2_proof.md`; the general case is not.

- **`d = 1` (binary shared verbalizer), any `m` — proved.**  Prop 3a gives the
  closed form `q_m(X) = min over (m−1) cut positions of the largest block
  half-range`, i.e. for `m = 2`,
  `q_2 = min_j max( (x_j − x_1)/2, (x_T − x_{j+1})/2 )` on sorted optima; Prop 3b
  gives a pigeonhole lower bound valid in every dimension and **tight in
  `d = 1`** (0/600 disagreements against the enumerator); Cor 3c transfers to
  accuracy under the κ of §5, **for per-task routing only**.  This is not a
  corner case: every *scorable* multi-task scope in Order-4 is `K = 2`, hence
  `d = 1` (`False|True`: WiC/QQP/BoolQA/MultiRC; `Bad|Good`: IMDB/SST-2).  So
  every `q_m` this project actually reports now sits under a proved statement.
- **`d ≥ 2`, `m ≥ 2` — still open.**  The minimiser is neither unique nor a
  single Chebyshev centre, and the convexity argument does not extend; Prop 3b
  holds but is strict, not tight (164/400 cases).  This blocks no reported
  number, because the `d ≥ 2` scopes are unscorable here (CB's rarest class has
  16 neutral examples; Yelp/Amazon fail the first-piece precondition).

Do not describe Prop 3 as proved in general — say "proved for binary shared
verbalizers".

## 4. Vacuity of rank-based bounds on this term (corrected)

The CoDyRA / DYRA family bounds forgetting through the size of the weight
update, schematically

    F(θ) ≤ ‖∇ℓ‖·‖ΔW‖ + (L/2)‖ΔW‖²,     ‖ΔW‖ ≤ √ρ · ‖B‖‖A‖.

**Correction to an earlier claim of mine.**  I previously asserted a strict
impossibility: that Δ^id can be made large with ‖ΔW‖ arbitrarily small.  That is
not right as stated.  Small ‖ΔW‖ implies small logit perturbation, which implies
small accuracy change *unless* many examples sit within that perturbation of the
decision boundary.  The correct statement is conditional:

**Proposition 4 (margin-conditional amplification).**  Let the per-example
verbalizer margin at time g have density bounded below by κ on [0, ε].  Then a
logit shift of size δ ≤ ε along the verbalizer-difference direction flips at
least κδ mass, so

    R_g^post − R_g^raw  ≥  κ · δ,      with δ ≤ c·‖ΔW‖·sup_x‖h(x)‖.

Consequently the rank-based bound is *loose by the factor κ*: a rank-1 update
aligned with a shared verbalizer-difference direction, of norm small enough to
keep the rank bound satisfied, can still move accuracy by κδ.  The rank bound
is therefore not vacuous in the absolute sense — it is **uninformative about
which of the two terms the damage lands in**, and it can be satisfied while
Δ^id is large.  That weaker claim is what the paper should make.

The consequence for allocators is the operative point: E²-LoRA (output-drift
energy), AdaLoRA (importance), CoDyRA (rank minimisation) all allocate rank on
signals that are *averages over the whole drift*.  None separates the component
that a zero-rank shared offset could absorb from the component that genuinely
needs subspace capacity.  Under a fixed lifetime budget, spending rank on the
former is waste — and, by Prop 2/3, spending offset capacity on conflicting
tasks cannot substitute for rank either.  Both budgets bind, for different
reasons.

## 5. What must be assumed, stated plainly

- **Margin density (κ).**  Needed for every accuracy-level statement.  It is an
  assumption about the data/model, and it is empirically checkable: κ is
  estimable from the histogram of verbalizer margins.  Report it.
- **Transfer from logits to accuracy.**  Prop 3 is naturally a statement about
  logit-space reconstruction error; converting to accuracy needs Prop 4's
  machinery and will carry κ.  Do not hide this.
- **Frozen unembedding.**  If U is also adapted, b is no longer the only
  output-layer degree of freedom and the accounting changes.  We keep U frozen,
  which is the standard LoRA-on-q/v setting.
- **One decision step, not one token per label.**  §1 writes the prediction as
  an argmax over verbalizer tokens y ∈ Y_g.  Measured on the box
  (2026-08-29, `runs/phase2j_smoke_small/smoke_report.json`), Order-4
  verbalizers are **not** all single-token under T5's sentencepiece: `True`
  is one piece (id 10998) but `False` is **three** (10747, 7, 15).  The
  formalism therefore must not be read as "one logit per label".  What it
  actually requires, and what holds, is that the labels of a task have
  **distinct first pieces**, so the first decode step already selects the
  label and b acts on that step.  Verified True for WiC and QQP; the probe
  checks it for every task it touches and must abort on any task where two
  labels share a first piece.  With multi-piece labels the accuracy-level
  statements are about first-step argmax, which coincides with greedy
  decoding's label choice but not necessarily with free-form generation.
- **Optimal offsets are estimated, not given.**  R^orc uses a *fitted* b_g^*.
  Our own BoolQA evidence (internal-fit threshold failing to generalise to the
  fresh confirmation set) shows this estimation error is real and not small at
  n=64.  The paper must report estimated-offset numbers, with the oracle offset
  as an upper reference only.

## 6. Relation to our existing artifacts

- QQP frozen-PCSM transfer: raw natural +15.78 pp but raw False recall
  −8.59 pp, while calibrated natural +10.45 pp, min-class +10.16 pp, and margin
  ROC-AUC +0.097.  Ranking improved, raw decoding boundary moved toward True.
  This is a direct observation of a large output-layer term in the generative
  setting.
- BoolQA sealed confirmation: raw +6.25 pp on both classes, but the
  frozen-threshold secondary endpoint gave False −9.375 pp with AUC +0.188.
  Threshold fitted internally did not transfer — evidence for Section 5's
  estimation caveat.

**Pre-existing quantitative evidence of conflict (extracted 2026-08-29).**  The
Phase-2I artifacts already fitted, per task, the optimal margin threshold on the
shared `true/false` verbalizer.  That threshold *is* the scalar contrast
s = b_true − b_false of §3, so these runs contain a direct measurement of the
per-task optima this theory is about:

| task | base s* | post-adaptation s* | shift |
|---|---:|---:|---:|
| QQP | −0.2520 | +0.3306 | **+0.5826** |
| BoolQA | −0.1122 | −0.1697 | **−0.0575** |

The two tasks moved in **opposite directions**, and the spread of the per-task
optima grew from 0.140 to 0.500 (3.6×) as a consequence of adaptation.  Under
Prop 2 the unavoidable worst-task loss of any single shared offset is controlled
by exactly this spread, so the mechanism is not hypothetical on our own data.

Two caveats, stated so this is not oversold: (i) these thresholds come from
T5-small at n≈128–256 per partition, so the spread carries real estimation
error, and BoolQA's own failure to transfer to fresh confirmation is the
cautionary case; (ii) with only two tasks a "spread" is a single number and
cannot establish the quantization form of Prop 3.  The probe exists to measure
this properly, on T5-large, across the full `{WiC, QQP, BoolQA, MultiRC}` group.
Artifact sources: `t5small_pcsm_lora_transfer_qqp_selection_seed43_20260829`
(`qualification_failure.json`) and
`t5small_pcsm_lora_boolqa_selection_seed43_20260829`
(`internal_qualification_lock.json`).
- Track-1 PSR-LoRA spectral-tail lower bound: originally intended as the donor
  theorem for Prop 3.  Per the correction in §3 this no longer holds — the
  output-layer budget is a quantization constraint, not a low-rank one.  Track-1
  remains the motivating precedent and the weight-subspace half of the
  two-capacity picture, but it is not reused as a proof.

These were previously logged as gate failures.  Under this framing they are the
motivating measurements, and they were obtained before this theory was written,
so they are *discovery* evidence — any confirmatory claim must come from
untouched partitions.

## 7. Immediate next steps

1. Prove Prop 2 properly (finite minimax; Chebyshev centre).
2. Attempt Prop 3; if the spectral form resists, fall back to a two-task
   corollary, which is already enough to motivate the method.
3. Estimate κ from real margin histograms on T5-large — this is cheap and also
   feeds the probe.
4. Do not write the method section until the probe in
   `phase2j_offset_conflict_probe_protocol.md` returns.
