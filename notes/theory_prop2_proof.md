# Proposition 2, proved

Written 2026-08-29 as a companion to `theory_output_layer_capacity_v1.md` §3,
where Prop 2 was stated with `ω` left as an unspecified "spread functional".
This note discharges that placeholder.  Prop 3 is *not* proved here.

## 1. What has to be proved, and what the honest statement is

The §3 draft asserts, for tasks `g ∈ S` sharing verbalizer set `Y`:

    max_{g ∈ S} [ R_g^orc − R_g^b ]  ≥  ω( { b_g^* } )        for every shared b

There is an immediate problem with reading this as an *accuracy* statement:
`R` is an accuracy, so the left side is bounded by 1, while any diameter-type
`ω` on offsets is unbounded.  So the inequality as literally written is false
for large spreads unless `ω` is capped.  The honest route is two separate
statements:

* **Prop 2a (logit space, exact, assumption-free).**  A shared offset cannot be
  simultaneously close to all per-task optima; the worst-case distance is
  exactly the Chebyshev radius.  This is where the "no free lunch" content is,
  and it needs no assumptions at all.
* **Prop 2b (accuracy space, needs margin density).**  Distance in logit space
  transfers to accuracy loss at a rate controlled by the margin density `κ`.
  This is where `ω` acquires a quantitative form, and it is *conditional*.

Conflating the two is what made the draft look stronger than it is.  Below,
2a is proved outright; 2b is proved under a stated assumption, with the
assumption's necessity demonstrated by counterexample.

## 2. Setting and gauge

Task `g` has verbalizer `Y = {y_0, …, y_{K−1}}` (shared across `g ∈ S`), hidden
state `h_g(x)`, frozen unembedding rows `u_y`.  Write the verbalizer-restricted
logit vector `ℓ_g(x) ∈ R^K`, `ℓ_g(x)_k = u_{y_k}^T h_g(x)`.  A shared offset
`b ∈ R^K` acts as `argmax_k (ℓ_g(x)_k + b_k)`.

**Gauge.**  For any `c ∈ R`, `b` and `b + c·1` induce the same argmax.  So the
offset acts only through its image in the quotient `R^K / span(1)`.  Fix the
gauge by projecting onto the zero-sum subspace `V₀ = {v : Σ_k v_k = 0}`,
`Π v = v − (mean v)·1`, and equip `V₀` with the Euclidean norm.  All offsets
below are elements of `V₀ ≅ R^{K−1}`.  For `K = 2` this is one-dimensional and
`b` is fully described by the scalar contrast `s = b_1 − b_0`, which is the
degeneration already noted in §3.

Let `R_g(b)` be task `g`'s accuracy under shared offset `b`, and
`b_g^* ∈ argmax_{v ∈ V₀} R_g(v)`, `R_g^orc = R_g(b_g^*)`.

## 3. Prop 2a — the geometric core (exact, no assumptions)

**Proposition 2a.**  For any finite `S` with per-task optima
`{b_g^* : g ∈ S} ⊂ V₀`, and for every shared `b ∈ V₀`,

    max_{g ∈ S} ‖ b − b_g^* ‖  ≥  r_S  :=  min_{v ∈ V₀} max_{g ∈ S} ‖ v − b_g^* ‖

with equality iff `b` is a Chebyshev centre of `{b_g^*}`.  Moreover
`r_S ≥ diam({b_g^*}) / 2`, and `r_S = 0` iff all `b_g^*` coincide.

*Proof.*  The definition of `r_S` is the infimum of
`F(v) = max_{g ∈ S} ‖v − b_g^*‖` over `V₀`, so the inequality is immediate.
`F` is a finite max of convex functions, hence convex; it is coercive
(`F(v) ≥ ‖v‖ − max_g ‖b_g^*‖`), so the infimum is attained on the closed convex
sublevel sets, and the minimiser set is exactly the Chebyshev centres.  Since
`V₀` is a Euclidean space the minimiser is unique, so equality holds iff `b` is
*the* Chebyshev centre.

For the lower bound, take any `g, g' ∈ S`.  For every `v`,
`‖b_g^* − b_{g'}^*‖ ≤ ‖v − b_g^*‖ + ‖v − b_{g'}^*‖ ≤ 2 F(v)`, so
`F(v) ≥ ‖b_g^* − b_{g'}^*‖ / 2`.  Maximising over the pair gives
`r_S ≥ diam / 2`.  If all optima coincide at `b^*` then `F(b^*) = 0`; conversely
`r_S = 0` forces `diam = 0`.  ∎

This is the precise sense in which a task-agnostic offset is *capacity-limited*:
the achievable "closeness to what each task wants" is floored by a geometric
quantity of the demands themselves, independent of the model, the data, or how
well we optimise. `r_S = q_1` in §3's notation, which is the `m = 1` case of the
quantization radius, so Prop 2a is also the base case of the corrected Prop 3.

**Prop 1 as a corollary.**  If the verbalizer sets are pairwise disjoint the
per-task offsets live on disjoint coordinate blocks, so
`b = Σ_g b_g^*` (extended by zero) realises every optimum simultaneously and
`r = 0`.  Sharing is what creates the floor.

## 4. Prop 2b — transferring to accuracy, and what it costs

Prop 2a bounds a *distance*, not an accuracy loss.  Going further needs an
assumption, because a distance can be large while costing nothing: if the
model's margins are all enormous, moving `b` by a bounded amount flips no
prediction and `R_g` is locally constant.

**Assumption (κ, margin density).**  Fix task `g` and a direction.  For `t ≥ 0`
let `N_g(t)` be the fraction of examples whose decision would flip when the
gauge-fixed offset moves distance `t` away from `b_g^*` along the worst
direction.  Assume `N_g(t) ≥ κ · min(t, t_max)` for `t ≤ t_0` — i.e. margins are
not degenerate near the decision boundary, at rate at least `κ`.

**Proposition 2b.**  Under the κ-assumption, for every shared `b`,

    max_{g ∈ S} [ R_g^orc − R_g(b) ]  ≥  κ · min( r_S , t_0 )

*Proof.*  Let `b` be given and pick `g` attaining `max_g ‖b − b_g^*‖ ≥ r_S`
(Prop 2a).  Write `t = ‖b − b_g^*‖`.  Every example that flips contributes a
loss relative to the oracle offset, so
`R_g^orc − R_g(b) ≥ N_g(min(t, t_0)) ≥ κ · min(t, t_0) ≥ κ · min(r_S, t_0)`,
using monotonicity of `N_g` and `t ≥ r_S`. ∎

So §3's `ω({b_g^*})` is discharged as `ω = κ · min(r_S, t_0)`: a *product* of a
purely geometric term (`r_S`, assumption-free) and a data-dependent rate (`κ`,
which must be measured).  The `min` with `t_0` is what fixes the boundedness
problem noted in §1 — accuracy loss saturates, distance does not.

**The assumption is necessary, not decorative.**  Counterexample: two tasks
sharing `{y_0, y_1}`, task 1 with all examples at logit gap `+10` for the true
class, task 2 all at `−10`.  Then `b_1^* = 0`, `b_2^*` must exceed `10`, so
`r_S ≥ 5` — yet `b = 0` already gives task 1 perfect accuracy and any
`b ∈ [0, 10)` gives task 2 zero, so the accuracy loss is `1`, not proportional
to `r_S`; and shrinking all gaps to `±0.01` leaves `r_S` tiny while the loss
stays `1`. Distance and loss are genuinely different quantities and no
assumption-free proportionality exists.  This is why κ must be reported from
real margin histograms, which the probe logs (`margin_mean`, `margin_p10`).

## 5. What this does and does not license

Proved: the geometric floor (2a), exactly, with the Chebyshev-centre
characterisation and the `diam/2` bound; Prop 1 as a corollary; and the
accuracy transfer (2b) under a stated, measurable assumption whose necessity is
demonstrated.

Not proved here: Prop 3 for `m ≥ 2`.  Prop 2a is its `m = 1` case, and the
`k`-centre structure means the `m ≥ 2` version is a covering problem whose
minimiser is *not* unique and *not* given by a single Chebyshev centre, so the
convexity argument in §3 does not extend.

**Superseded in part, 2026-08-30.**  `notes/theory_prop3_m2_proof.md` closes the
`d = 1` case for every `m` (Prop 3a closed form, Prop 3b pigeonhole bound tight
in `d = 1`, Cor 3c accuracy transfer under the same κ as §4 but restricted to
per-task routing).  Since `d = 1` is exactly the binary-shared-verbalizer case,
and every scorable multi-task scope in Order-4 is binary, the sentence above now
reads: open for `d ≥ 2`, closed for the case our data occupies.  The correct
phrasing of the remaining gap is "`m ≥ 2` in dimension `≥ 2`", not "`m ≥ 2`".

Also not claimed: that any of this predicts *which* tasks lose.  Prop 2a says
someone must be `r_S` away; it does not say who.  The empirical claim that the
worst-loser is predicted by the spread is criterion (3) of the probe protocol
and is a separate, falsifiable question.

---

## 6. κ, estimated from the real flip-fraction curves (2026-08-30)

Prop 2b's κ-assumption is no longer an unmeasured placeholder.  Estimated from
the Phase-2J probe records (`runs/phase2j_probe_full/probe_seed{2,3}.json`, 93
scorable task/stage entries per seed, T5-large LoRA r=8, Order-4), via
`experiments/phase2j_offset_conflict/estimate_kappa.py`; output in
`runs/phase2j_probe_full/kappa.json`.

**The number to quote is κ ≈ 0.018** (WiC, the worst task).  Median over tasks is
0.0625.  The worst task is what matters: κ must hold simultaneously for every task
sharing an offset, so a single small-κ task drags the bound to itself.  A median κ
would overstate the bound by ~3.4× here.

| task | κ (grid-safe) | κ (naive) | κ (p10 rough) |
| --- | --- | --- | --- |
| WiC | **0.0182** | 0.0312 | 0.0871 |
| SST-2 | 0.0195 | 0.0391 | 0.1012 |
| DBpedia | 0.0201 | 0.0502 | 0.0769 |
| COPA | 0.0391 | 0.0859 | 0.1777 |
| MNLI | 0.0625 | 0.1510 | 0.2656 |
| MultiRC | 0.0638 | 0.1133 | 0.1412 |
| RTE | 0.0781 | 0.1953 | 0.3253 |
| AGNews | 0.0859 | 0.1875 | 0.2768 |
| BoolQA | 0.0938 | 0.2344 | 0.2724 |
| Yahoo | 0.1794 | 0.2238 | 0.3502 |
| QQP | 0.2031 | 0.2500 | 0.6540 |
| IMDB | 0.2178 | 0.2500 | 0.6400 |

**Why three columns, and why the first is the honest one.**  `flip_fraction_lower_bound`
is recorded on a grid `t ∈ {0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4}`.  The naive estimate
`min_i N(t_i)/t_i` only verifies `N(t) ≥ κ·t` **at grid points**, but Prop 2b needs
it for every `t` in the interval.  Since `N` is non-decreasing, the worst case
inside `[t_i, t_{i+1}]` is `t → t_{i+1}⁻` with `N` still only `N(t_i)`, so the
genuinely safe slope is `min_i N(t_i)/t_{i+1}`.  Grid ratios reach 2.5×, and the
two columns do differ by up to 2.5× — this is not rounding, it decides whether the
reported κ is a lower bound at all.  The `p10` column is the older two-quantile
estimate, retained only to show it is optimistic by a further 2–3× and should not
be cited.

**Two limits I am not papering over.**  (i) For tasks with more than two classes
the flip fraction is itself a *lower* bound — only top-2 flips are counted — so κ
for MNLI/AGNews/DBpedia/Yahoo/Yelp/Amazon is conservative in a second, separate
way.  (ii) These κ come from the M3 run, where most tasks got 3–16 gradient steps;
margins from an under-trained model are compressed, which plausibly *depresses* κ.
The convergence rerun (method note §M4) will let this be re-estimated on a trained
model, and the two values should be reported side by side rather than the more
convenient one being chosen.

Consequence for the paper: Prop 2b is stated with a *measured* κ, and with the
honest reading that κ ≈ 0.018 makes the accuracy-half of Prop 2 a weak bound on
this benchmark.  The geometric half (Prop 2a, Chebyshev radius) needs no κ and
carries the load; the accuracy transfer is the part that degrades.
