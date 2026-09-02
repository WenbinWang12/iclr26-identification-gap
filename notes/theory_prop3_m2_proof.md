# Proposition 3 for m = 2 — proved for binary verbalizers, and the honest limits

Written 2026-08-30.  `theory_prop2_proof.md` §5 recorded Prop 3 for `m ≥ 2` as
**open**, with the reason: the `k`-centre minimiser is neither unique nor given by
a single Chebyshev centre, so Prop 2a's convexity argument does not extend.

This note closes the case that the paper's evidence actually rests on — **`m = 2`
for a binary shared verbalizer**, i.e. `dim = 1` — and states precisely what is
*not* closed.  It also records two things I found while proving it that change how
existing numbers must be reported.

## 0. Why `m = 2`, binary, is the case worth proving

Not because it is easy.  Because it is where every real conflict in Order-4 lives:

| scope | tasks | K | dim = K−1 |
| --- | --- | --- | --- |
| `False\|True` | WiC, QQP, BoolQA, MultiRC | 2 | **1** |
| `Bad\|Good` | IMDB, SST-2 | 2 | **1** |
| `contradiction\|entailment\|neutral` | MNLI, CB | 3 | 2 |
| `negative\|…\|very positive` | Yelp, Amazon | 5 | 4 |

The two multi-member scopes with `dim ≥ 2` are MNLI+CB (CB is not scorable, rarest
class 16 examples) and Yelp+Amazon (neither is scorable — they fail the
first-piece precondition, `very *` share piece 182).  So **every scorable
multi-task scope in this benchmark has `dim = 1`**, and the `m = 1 → 2` step that
§M3 reports as the informative comparison is entirely inside the case proved here.

## 1. Setting

As in `theory_prop2_proof.md` §2.  Binary verbalizer `Y = {y_0, y_1}`; the
gauge-fixed offset space is `V₀ ≅ R`, and an offset is fully described by the
scalar contrast `s = b_1 − b_0`.

**The `d = 1` claim is about the code's actual arrays, not just the abstract
quotient.**  Worth checking explicitly, because the codebook solver operates on
gauge-fixed vectors in `R^K`, not on scalars — so "d = 1" could have been a
statement about the quotient that the implementation silently violates.  It does
not: for `K = 2`, `gauge_fix` returns vectors of the form `(−t, t)`, the stack of
any number of them has rank 1 after removing a translation, and
`‖g_a − g_b‖₂ = |s_a − s_b| / √2`.  So the solver's Euclidean geometry on `R²` is
the scalar geometry up to the constant `1/√2`, and every `d = 1` result below
transfers verbatim (a constant factor cannot affect which codebook is optimal,
only the units of the radius).  Verified: on 300 random `K = 2` instances the
enumerator equals the pigeonhole quantity of §3 exactly, 0 disagreements.

This also explains the recorded `q_1 = 0.7071 = 1/√2` for `{False,True}`: the
four per-task contrasts span a range of exactly `2` in scalar units
(`2·q_1·√2 = 2`), and `0.7071` is that range halved and expressed in the solver's
vector norm.  The number is a unit convention, not a two-dimensional scope.  Tasks `g ∈ S`, `|S| = T`, have per-task optimal
contrasts `x_1 ≤ x_2 ≤ … ≤ x_T` (relabelled in sorted order; this ordering is the
whole reason the 1-dimensional case is tractable).

A budget of `m` task-free offsets is a **codebook** `C = {c_1, …, c_m} ⊂ R`
together with a task-agnostic rule assigning each task to a centre.  Define

    q_m(X) = min_{|C| ≤ m} max_{i} dist(x_i, C)

the worst-case `m`-quantization radius.

## 2. Prop 3a (m = 2, dim = 1) — exact closed form

**Proposition 3a.**  For `X = {x_1 ≤ … ≤ x_T} ⊂ R` and `m = 2`,

    q_2(X) = min_{1 ≤ j ≤ T−1}  max( (x_j − x_1)/2 , (x_T − x_{j+1})/2 )

attained by `C = { (x_1+x_j)/2 , (x_{j+1}+x_T)/2 }` for the minimising `j`.
Consequently `q_2(X) ≤ q_1(X) = (x_T − x_1)/2`, with equality iff `T ≤ 2`… *(see
the equality clause in §2.2 — it is `T = 1`, not `T ≤ 2`)*.

### 2.1 Proof

*Step 1: an optimal 2-codebook induces a contiguous split.*  Let `C = {c_1 ≤ c_2}`
be any codebook and assign each `x_i` to its nearest centre.  The nearest-centre
rule in `R` assigns `x_i` to `c_1` iff `x_i ≤ (c_1+c_2)/2`.  So the assignment is a
**threshold** rule: there is `j ∈ {0,…,T}` with `{x_1,…,x_j} → c_1` and
`{x_{j+1},…,x_T} → c_2`.  Hence only the `T+1` contiguous splits need be
considered — this is exactly the step that fails in `dim ≥ 2`, where Voronoi cells
of two centres are half-spaces but the *point set* need not be separable in a way
that enumerating orderings captures.

*Step 2: within a block, the best centre is the block's midrange.*  For a finite
block `B ⊂ R`, `min_c max_{x ∈ B} |x − c| = (max B − min B)/2`, attained uniquely
at `c = (max B + min B)/2`.  (One line: `max_x |x−c| ≥ ((max B − c) + (c − min B))/2`
by averaging the two endpoint distances, with equality iff `c` is the midpoint.)

*Step 3: combine.*  For split `j` the covering radius is
`max( (x_j − x_1)/2 , (x_T − x_{j+1})/2 )`, and minimising over `j` gives the
formula.  The empty-block splits `j = 0` and `j = T` reduce to `q_1`, which is
never better than the best proper split, so restricting to `1 ≤ j ≤ T−1` is
without loss.  ∎

### 2.2 The equality clause, stated correctly

`q_2 = q_1` iff `T = 1` (both are `0`).  For `T = 2`, `q_2 = 0 < q_1 = (x_2−x_1)/2`
whenever `x_1 ≠ x_2` — two centres cover two points exactly.  I initially wrote
`T ≤ 2` above and it is wrong; `T = 2` is precisely the `{Bad,Good}` case where the
data shows `q_2 = 0`, so getting this clause wrong would have contradicted our own
measurement.  Left visible rather than silently corrected.

## 3. Prop 3b (m = 2, dim = 1) — the pigeonhole lower bound

Prop 3a is a formula; a *bound* usable in the paper needs a quantity that does not
presuppose solving the problem.

**Proposition 3b.**  For any `m ≥ 1` and any `X ⊂ R^d` (any dimension),

    q_m(X)  ≥  (1/2) · max_{ T' ⊆ X, |T'| = m+1 }  min_{u ≠ v ∈ T'} ‖u − v‖

*Proof.*  Fix any `(m+1)`-subset `T'` and any codebook `C` with `|C| = m`.  By the
pigeonhole principle two distinct points `u, v ∈ T'` share a nearest centre `c`.
Then `‖u−v‖ ≤ ‖u−c‖ + ‖c−v‖ ≤ 2 max_i dist(x_i, C)`.  So the covering radius is at
least `min_{u≠v ∈ T'} ‖u−v‖ / 2`; maximise over `T'`.  ∎

**And in dim = 1 the bound is tight.**  For `d = 1` the inequality in Prop 3b holds
with **equality** for every `m` and every `X`.  Verified numerically against the
exact enumerator on 600 random instances (`n ∈ [2,7]`, `m ∈ [1,4]`, random scales):
**0 disagreements**.  So for binary verbalizers the pigeonhole quantity is not
merely a bound — it *is* `q_m`, which gives the budget bound a form that reads off
the geometry of the demands directly, with no optimisation.

For `d ≥ 2` the bound is strict in the majority of random instances (164 of 400
strictly greater), so this tightness claim must be confined to `d = 1`.  Prop 3b
itself remains valid in all dimensions as a lower bound.

## 4. Transfer to accuracy, and a limit that must be stated

Composing with the κ-machinery of `theory_prop2_proof.md` §4:

**Corollary 3c.**  Under the κ-assumption, for any budget of `m` task-free offsets
assigned **per task**,

    max_{g ∈ S} [ R_g^orc − R_g^{P_m} ]  ≥  κ · min( q_m , t_0 )

with `q_m` given exactly by Prop 3a for `m = 2, d = 1`.  Proof: identical to
Prop 2b with `r_S` replaced by `q_m`, using that some task is at distance ≥ `q_m`
from its assigned centre.

**The limit — this is important and I nearly stated the corollary too broadly.**
Corollary 3c requires the routing rule to be **per task** (all examples of a task
share a centre).  It is **false** for a general **per-example** router.
Counterexample, constructed and checked numerically: one binary task, 50 examples
with contrast `s = −3` and true label 1, 50 with `s = +3` and true label 0.  The
best single offset achieves accuracy `0.500`; two centres `{+4, −4}` with a router
that reads `sign(s)` achieves `1.000` — *above* the single-offset oracle `R^orc`,
so no bound of the form `R^orc − R^{P_m} ≥ (positive)` can hold.

Per-example routing is strictly more powerful than the object Prop 3 bounds.  Our
implementation contains both kinds: `route_prototype` and `route_batch_margin` are
per-task (one centre per task / per batch), while `route_confidence` is per-example
— and `route_confidence` is exactly the router that was measured to be degenerate
(oracle agreement 0.000–0.008).  So the theory and the reported numbers are
consistent, but **the corollary must be quoted with "per-task routing" attached**,
not as a statement about arbitrary task-free routers.

## 5. What is now proved, and what is still open

**Proved.** `q_2` in closed form for binary verbalizers (Prop 3a); the pigeonhole
lower bound for all `m` and all dimensions (Prop 3b); its **tightness in dim = 1**;
and the accuracy transfer under κ *for per-task routing* (Cor. 3c).  Together with
Prop 2a (`m = 1`) this covers every scorable multi-task scope in Order-4.

**Still open.** A closed form or matching bound for `m ≥ 2` in `d ≥ 2`.  Prop 3b
still holds there but is loose.  Since the only `d ≥ 2` multi-member scopes in this
benchmark are unscorable (CB too rare, Yelp/Amazon fail the precondition), this gap
does **not** affect any number we report — but the paper must say `d = 1` where it
means `d = 1`, and must not imply the general case is settled.

**Also still open.** Whether a per-example task-free router can be made to work at
all; §4 shows it has *more* headroom than the bound, and the measurement shows the
obvious construction fails. That asymmetry is an honest open question, not a defect.

## 6. Consequence for a claim already on record

`theory_output_layer_capacity_v1.md` §3 says the enumeration over "points plus
pairwise midpoints" computes `q_m` **exactly**, and `test_qoc.py`'s
`test_codebook_radius_equals_probe_quantization_radius` pins the method to that
quantity.  Prop 3a's Step 2 shows the claim is right in `d = 1` (midranges of
contiguous blocks are pairwise midpoints of the points themselves).  **It is wrong
in `d ≥ 2`**: for three points forming an equilateral triangle of side 1, the
enumeration returns `0.8660` while the true Chebyshev radius is
`1/√3 = 0.5774` — the circumcentre is not any point or pairwise midpoint, so the
enumeration **over-estimates by 1.5×**.

This does not invalidate any reported number, because the enumerator is only
consulted for `q_m` on scopes that are `d = 1` (or `n ≤ 2`, where the midpoint is
optimal by Step 2).  But the docstrings and the theory note claim exactness
unconditionally, and that claim must be narrowed to `d = 1 or n ≤ 2` rather than
left as-is.  Recorded here as a correction to my own earlier wording.
