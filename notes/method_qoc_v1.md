# QOC — Quantized Offset Codebook: method spec v1

Frozen 2026-08-29, **before** any method result exists.  This note is the
constructive counterpart of `theory_output_layer_capacity_v1.md` §3 Prop 3: the
theory lower-bounds what a budget of `m` task-free offsets must lose; this note
specifies a method that spends exactly that budget, so the two can be compared
on the same axis.

Everything here is measured with the **first-step restricted-argmax accuracy**
of probe protocol Amendment A4.5, never free-generation EM.  A4.7 established
that first-step accuracy is the stable quantity and EM is the noisy one.

## 1. What problem the method solves

Phase-2J measured, over the 15-task Order-4 stream, 3 seeds, T5-large + LoRA
r=8 on q/v:

    median Δ_id = R_orc − R_shr = 6.25 pp,  cluster-bootstrap 95% CI [3.9, 9.4] pp

`R_orc` needs task identity; `R_shr` is a single global offset `b ∈ R^V` and
needs none.  The 6.25 pp is the price of not knowing which task a query came
from.  QOC's job is to buy back as much of it as possible with a budget that
stays parametrically negligible, **without a replay buffer** and **without task
identity at inference**.

## 2. Two capacity sources, in order of cost

**Step 1 — verbalizer scoping (free).**  A single global `b` must assign one
value to each vocabulary token.  When two tasks share a token, that is a real
conflict (Prop 1: disjoint verbalizers ⇒ Δ_id = 0).  But the **active label set
is present in the input**: the O-LoRA prompt literally contains
`Option: neutral, entailment, contradiction`.  So replacing one global `b` by a
family indexed by the active label set,

    b^S  for each distinct verbalizer set S occurring in the stream,

is task-**agnostic** — it reads the query, not an oracle task ID.  This buys
real capacity for free: `neutral` can take one value under
`{contradiction, entailment, neutral}` and a different value under
`{very negative, negative, neutral, positive, very positive}`, which a global
`b` cannot do.

On the pinned Order-4 specs the distinct scopes and their memberships are:

| scope S | members |
|---|---|
| `{False, True}` | WiC, QQP, BoolQA, MultiRC |
| `{contradiction, entailment, neutral}` | MNLI, CB |
| `{Bad, Good}` | IMDB, SST-2 |
| `{negative, neutral, positive, very negative, very positive}` | Yelp, Amazon |
| `{contradiction, entailment}` | RTE |
| `{A, B}` | COPA |
| 3 more singletons | AGNews, DBpedia, Yahoo |

Scoping therefore reduces the conflict to **within-scope only**, and predicts
Δ_id = 0 for the four singleton scopes.  This prediction is falsifiable and is
reported as such.

**Step 2 — within-scope offset codebook (budgeted).**  Inside a scope the label
set is identical, so the input's option list cannot separate members: this is
exactly where Prop 3's `q_m` binds.  Per scope keep at most `m` gauge-fixed
offsets

    C^S = { c_1, ..., c_m } ⊂ R^{|S|},   c_j gauge-fixed (zero mean)

chosen as the **exact minimax (k-center) codebook** of the accumulated per-task
optima `{b_g^*}_{g ∈ S}`.  "Exact" matters: the codebook is built by enumerating
candidate centres (the points themselves plus all pairwise midpoints) so its
covering radius **equals** `q_m({b_g^*})` by construction, the same quantity the
theory bounds.  The method therefore does not merely resemble the bound; it
attains it in logit space.

## 3. Routing (the part that can fail)

At inference the scope is known, the member task is not.  A task-free router
picks `j ∈ [m]`.  Four routers are specified and **all four are reported**:

| router | signal | extra params | role |
|---|---|---|---|
| `single` | none, `m = 1` | 0 | lower reference = scoped shared offset |
| `confidence` | top1 − top2 margin of `logits + c_j`, argmax over `j` | **0** | parameter-free |
| `prototype` | nearest centroid of pooled encoder state, cosine | `m·d` per scope | the theory's "task-free gate" |
| `oracle` | true task → its cluster | — | **upper reference only, not a method** |

`prototype` centroids are computed from the **train-derived risk split only**.
The oracle router is never presented as a result; it exists to split the loss
into a quantization part and a routing part (§5).

Honest caveat, stated before seeing numbers: within the `{False, True}` scope the
four members have visibly different prompt formats (WiC has `word:`, BoolQA has
a passage, MultiRC has a question plus candidate answer).  A prototype router
will exploit that.  Prompt format is a legitimate input feature — it is not task
identity leakage — but it does mean the `prototype` number is optimistic for
settings where inputs are format-identical, and the paper must say so.  The
`confidence` router does not use format at all, which is why it is reported
alongside rather than as a fallback.

## 4. Budget accounting, exactly

Offsets: `Σ_S m·|S|` scalars.  For Order-4 with `m = 4`:
`4·(2+3+2+5+2+2+4+14+10) = 4·44 = 176` scalars.
Router prototypes (`prototype` only): `m·d` per non-singleton scope, `d = 1024`
for T5-large, 4 non-singleton scopes ⇒ `4·4·1024 = 16,384` scalars.
Stored per-task optima: `Σ_g |Y_g|` = 44 scalars.
**No stored examples.**  Total ≤ 16,604 scalars against ~6.3 M trainable LoRA
scalars for r=8 on q/v of T5-large — under 0.3 %.  The parameter-free
`confidence` router costs 176 scalars total, ~0.003 %.

## 5. What is measured, and the decomposition that is the actual claim

Per old task `g` at the end of the 15-task stream, all on the audit split, all
as first-step restricted-argmax balanced accuracy:

    R_raw            no offset
    R_shr_global     one global b (Phase-2J's baseline)
    R_shr_scoped     m = 1 per scope                        ← step 1
    R_qoc(m, router) m offsets per scope + task-free router  ← step 2
    R_orc            per-task offset, needs task identity    ← upper reference

The claim is the **decomposition of the identification gap**:

    R_orc − R_shr_global
      = (R_shr_scoped  − R_shr_global)   scope-recoverable, free
      + (R_qoc(m)      − R_shr_scoped)   codebook-recoverable at budget m
      + (R_orc         − R_qoc(m))       residual

and, separately, the split of the residual into

    quantization loss = R_orc − R_qoc(m, oracle router)
    routing loss      = R_qoc(m, oracle) − R_qoc(m, prototype)

Reporting both halves is the point: if the residual is quantization it is a
capacity statement and the theory's `q_m` predicts it; if it is routing it is an
inference-side statement and `q_m` does not.  Conflating them would be the easy
way to overclaim.

## 6. Predeclared success / failure conditions

Same discipline as the Phase-2J probe: stated now, not after.

QOC is a **positive result** if, over 3 seeds on the 15-task stream:

1. `R_shr_scoped ≥ R_shr_global` on median, and Δ_id is 0 for all four singleton
   scopes (this is Prop 1 as a falsifiable prediction, not a fit);
2. `R_qoc(4, prototype) − R_shr_scoped > 0` with a cluster-bootstrap 95 % CI
   (tasks as clusters) excluding 0 — i.e. the codebook buys something beyond
   scoping;
3. the measured worst-task loss per scope is **monotone non-increasing in
   `m ∈ {1,2,4}`** and ordered consistently with the measured `q_m` of that
   scope.

QOC is a **negative result** worth reporting if (1) holds but (2) fails: that
would say the identification gap is almost entirely cross-scope bookkeeping,
removable for free, with no genuine within-scope conflict left to spend budget
on — which would contradict Phase-2J's criterion-3 finding that ω predicts
retention loss, and would have to be reconciled explicitly.

It is a **failure of the theory's relevance**, not of the method, if the
`oracle` router is needed to get any gain: that would mean `q_m` is the wrong
object because the binding constraint is routing, not covering.

## 7. Deliberate non-goals

- QOC is **not** a rank-allocation method and does not compete with O-LoRA or
  E²-LoRA on their axis.  It is an output-layer mechanism that composes with any
  of them.  The comparison to run is therefore *base stream* vs *base stream +
  QOC*, on the same LoRA configuration, plus the same overlay on an O-LoRA
  stream if time permits.  Claiming to beat O-LoRA on rank allocation would be a
  category error.
- No constrained decoding (protocol A4.6): comparability with published O-LoRA
  EM numbers is kept by reporting free-generation EM unchanged and separately.
- No replay buffer, no growth in stored examples, no per-task adapters.

---

## Amendment M1 — the router was reading the task name out of the prompt

Appended 2026-08-30 after seed 1 of the first QOC run.  Written **before** any
re-run, and it invalidates part of that seed's routing numbers.

**M1.1 The defect.**  §3 said a prototype router "exploits prompt format" and
called that legitimate.  That understated it.  The official O-LoRA prompt begins

    Task:WiC\nDataset:WiC\nGiven a word and two sentences, ...
    Task:BoolQA\nDataset:BoolQA\nAccording to the following passage, ...

The **task identity is written into the input as literal text**.  A router that
reads the encoder state is therefore not inferring anything — it is reading the
task name.  Seed 1's prototype/oracle agreement (1.000 on IMDB, SST-2, MNLI,
MultiRC; 0.906–0.992 on BoolQA, QQP, WiC) is consistent with exactly that.  Any
"task-agnostic routing works" claim built on it would have been false.

**M1.2 Fix.**  The router's view of the input now strips the leading `Task:` and
`Dataset:` lines (`strip_task_header`).  Two routers are reported:

- `prototype` — stripped view.  **This is the honest number.**
- `prototype_official_leaky` — unstripped.  Reported only to quantify the leak;
  it is labelled `NOT_EVIDENCE` in the verdict JSON and must never be quoted as
  a method result.

Classification always uses the unmodified official prompt, so comparability with
published O-LoRA numbers is untouched; only the router's input view changes.

**M1.3 What remains in the stripped view, and why that is acceptable.**  The
instruction sentence still differs across tasks (`Given a word and two
sentences...` vs `According to the following passage...`).  That is not leakage:
any deployed system must tell the model what task to perform, so the instruction
is part of a legitimate input.  The dataset *name* is not — it is a benchmark
bookkeeping artifact.  The line is drawn there, and it is drawn explicitly rather
than left to the reader.

A stricter variant (identical instructions across a scope) would test something
stronger but would no longer be the official benchmark.  Not run; noted as a
limitation the paper must state when reporting routing.

**M1.4 A second, independent defect: `confidence` is degenerate.**  Measured
agreement with oracle on the `{False,True}` scope, seed 1: **0.000, 0.008, 0.008,
0.000** — it routes systematically *wrong*, not randomly.  The reason is clear in
hindsight and is a real flaw in the criterion, not a bug: per-sample top1−top2
margin is maximised by whichever centre most aggressively pushes every sample
toward one class, since that raises top1 and lowers top2 simultaneously.  So
"pick the centre with the largest margin" rewards the most extreme centre, not
the best-matched one.  This is now covered by a regression test
(`test_confidence_router_degeneracy_is_reproducible`) that constructs the failure
explicitly.

`route_confidence` is **kept and reported** rather than deleted: it is the first
thing one tries for a parameter-free router, and its failure is informative.

**M1.5 Added router: `batch_margin`.**  Same signal, aggregated over a batch
instead of per sample: pick the single centre with the largest *mean* margin over
the batch.  Averaging removes the degeneracy, because pushing everything toward
one class shrinks the margin of samples belonging to the other.  Cost, stated
plainly: it assumes queries **arrive batched by task**, which is a transductive
assumption and strictly weaker than per-sample routing.  It reads no task
identity and no labels, but it does use the fact that a batch is homogeneous.
Every number from it must carry that caveat.

**M1.6 Consequence for §6.**  The criteria are unchanged.  Criterion 2 is now
evaluated on the **stripped** `prototype` router, which is a strictly harder test
than what §6 was written against.  If it fails there but passes on the leaky
variant, the honest conclusion is that QOC's capacity gain is real (that is
criterion 3 and the oracle-router decomposition, which do not involve routing)
while **task-free routing on this benchmark is unsolved** — a partial result,
reported as such, not a positive one.

---

## Amendment M2 — routing results split into a trivial half and a real half

Appended 2026-08-30 after seed 1 of the corrected run.  M1's fix was necessary
but **not sufficient**, and the measurement says so plainly: stripping the
`Dataset:` header changed the prototype router's numbers by **nothing**.
`prototype` and `prototype_official_leaky` are digit-identical on all 7
multi-member-scope tasks (BoolQA 0.680, IMDB 0.906, MNLI 0.510, MultiRC 0.594,
QQP 0.469, SST-2 0.750, WiC 0.508).  So the router was never relying on the
dataset name; it relies on something else in the input.

**M2.1 What it relies on.**  Inspecting the stripped prompts by scope:

`{False, True}` — four members, four **different** instruction sentences:

| task | stripped prompt begins |
| --- | --- |
| WiC | `Given a word and two sentences, whether the word is used ...` |
| QQP | `Whether the "first sentence" and the "second sentence" have ...` |
| BoolQA | `According to the following passage, is the question true or false? ...` |
| MultiRC | `According to the following passage and question, is the candidate ...` |

`{Bad, Good}` — two members, **identical** instruction sentences:

| task | stripped prompt begins |
| --- | --- |
| IMDB | `What is the sentiment of the following paragraph? Choose one from the option` |
| SST-2 | `What is the sentiment of the following paragraph? Choose one from the option` |

These two scopes therefore test different things, and they must be reported
separately rather than pooled:

- On `{False, True}` the router is reading a per-task instruction string.  It is
  task-agnostic in the letter (no oracle ID) but not in the spirit: the tasks are
  textually self-identifying.  **This is not evidence that task-free routing
  works.**
- On `{Bad, Good}` the instructions are byte-identical, so the router's 1.00
  agreement and IMDB's **+0.156** gain must come from the input *body*
  distribution (full movie reviews vs short critic fragments).  Discriminating
  tasks by input distribution is exactly what a deployed task-free router would
  have to do, and here it works.  **This half is genuine evidence**, on one scope
  with two members.

**M2.2 Honest summary of the routing claim.**  QOC's codebook gain is real where
measured (IMDB +0.156, BoolQA +0.047, SST-2 +0.039 at m=4, seed 1) and the
`prototype` router recovers essentially all of the oracle-router gain.  But of
the two conflict scopes that produced gains, one is textually self-identifying.
The defensible claim is therefore narrow: *a nearest-prototype router over
encoder states recovers the codebook gain, and on the one scope where prompts are
byte-identical it still does so from input distribution alone*.  Anything broader
requires a benchmark with format-identical tasks, which Order-4 is not.

**M2.3 A specification error in §6's criterion 2, and how it is handled.**
Criterion 2 takes the median codebook gain over **all** scorable tasks.  But 8 of
the 12 scorable tasks sit in **singleton** scopes, where `q_1 = 0` already and a
codebook is mathematically incapable of helping.  The median is thus diluted by
structural zeros, and criterion 2 as frozen tests "are multi-member scopes a
majority of scorable tasks?" rather than "does the codebook buy anything?".
Measured: median +0.0000, CI [+0.0000, +0.0781] → **criterion 2 FAILS as
written**, and it is reported as failing.

A post-hoc restriction to multi-member scopes is computed and reported as
`post_hoc_conflict_scopes_only`, explicitly flagged `NOT_PREREGISTERED`.  It does
not convert the verdict.  Per §6 the outcome is therefore
**NEGATIVE_SCOPE_ONLY**: scoping is where the recoverable identification gap
mostly lives (median **+6.25 pp**, CI [+0.00, +7.81]), and the codebook adds a
further gain that is real on the three conflict-scope tasks but too concentrated
to move a 12-task median.

**M2.4 What this means for the paper, stated without spin.**  The strongest
honest result is not "our method wins".  It is the pair:

1. **Free capacity exists and is substantial.**  Indexing the offset by active
   label set — reading the prompt's option list, no task ID, zero extra
   parameters beyond the offsets themselves — recovers a median 6.25 pp of the
   identification gap.  MNLI +13.5 pp, DBpedia +8.5 pp, BoolQA/MultiRC +7.8 pp.
2. **Budgeted capacity behaves exactly as `q_m` predicts, where conflict
   actually exists.**  On `{False,True}`: q_m 0.7071 → 0.2033 → 0.0000 for
   m = 1,2,4 with worst-task loss 0.1094 → 0.0234 → 0.0000.  On `{Bad,Good}`:
   0.5701 → 0.0000 with loss 0.1484 → 0.0000.  Monotone, and ordered with `q_m`.

Caveat that must accompany (2): m = 4 saturates on `{False,True}` **because that
scope has exactly 4 members**, so m = 4 is the degenerate one-centre-per-task
case.  The informative comparison is m = 1 vs m = 2, where the codebook holds
half as many centres as tasks and still removes 79 % of the worst-task loss.

---

## M3 — three-seed result, final

Run completed 2026-08-30T04:58 on T5-large, LoRA r=8 q/v, 15-task Order-4,
seeds 1/2/3, 46–47 min each.  Verdict: **NEGATIVE_SCOPE_ONLY** per §6.
`runs/phase2k_qoc_full/{qoc_seed{1,2,3}.json, verdict_qoc.json, report.txt}`.

**Criterion 1 — PASS.**  Median scope-recoverable gain **+5.92 pp**,
cluster-bootstrap 95 % CI **[+1.56, +7.81] pp** (tasks as clusters, excludes 0).
Zero Prop-1 violations: all five singleton scopes have Δ_id_scoped ≡ 0 exactly,
which is the falsifiable prediction of Prop 1 and it holds.

Per task (3-seed mean, first-step restricted-argmax balanced accuracy):

| task | raw | global b | scoped | QOC(m=4) | oracle | scope+ | book+ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MNLI | 0.330 | 0.342 | 0.477 | 0.477 | 0.477 | **+0.135** | +0.000 |
| IMDB | 0.823 | 0.792 | 0.747 | 0.875 | 0.875 | −0.044 | **+0.128** |
| BoolQA | 0.565 | 0.576 | 0.677 | 0.690 | 0.695 | +0.102 | +0.013 |
| RTE | 0.547 | 0.529 | 0.615 | 0.615 | 0.615 | +0.086 | +0.000 |
| MultiRC | 0.516 | 0.536 | 0.609 | 0.615 | 0.615 | +0.073 | +0.005 |
| SST-2 | 0.542 | 0.677 | 0.716 | 0.789 | 0.792 | +0.039 | **+0.073** |
| DBpedia | 0.904 | 0.904 | 0.959 | 0.959 | 0.959 | +0.054 | +0.000 |
| COPA | 0.531 | 0.531 | 0.583 | 0.583 | 0.583 | +0.052 | +0.000 |
| AGNews | 0.727 | 0.727 | 0.777 | 0.777 | 0.777 | +0.051 | +0.000 |
| QQP | 0.497 | 0.487 | 0.500 | 0.503 | 0.500 | +0.013 | +0.003 |
| WiC | 0.497 | 0.497 | 0.495 | 0.510 | 0.510 | −0.003 | +0.016 |

IMDB's −0.044 on scoping is real and must not be hidden: sharing `{Bad,Good}`
with SST-2 costs IMDB, and it is the codebook (m ≥ 2) that recovers it (+0.128).
That is the cleanest single illustration of the paper's own thesis — free
capacity is not always enough, and the budgeted level is what fixes it.

**Criterion 2 — FAIL, as written and post-hoc.**  Median codebook gain
+0.0000, CI [+0.0000, +0.0156].  Restricted post-hoc to the 7 multi-member-scope
tasks: +0.0078, CI [+0.0000, +0.0391] — still includes 0.  So even after
correcting for the M2.3 dilution the codebook gain does not clear a
cluster-bootstrap CI on this benchmark.  Reported as failing.  The gains are real
but concentrated in 3 of 11 tasks (IMDB +12.8, SST-2 +7.3, BoolQA +1.3 pp), and 3
of 11 does not move a median.

**Criterion 3 — monotone, with the intended q_m ordering.**  Spearman between
q_m and worst-task loss over all (stage, scope, m) points: **ρ = +0.796**
(prototype).  Per conflict scope, 3-seed mean:

| scope | m | q_m | worst-task loss | centres |
| --- | --- | --- | --- | --- |
| `{False,True}` | 1 | 0.7484 | 0.0807 | 1 |
| `{False,True}` | 2 | 0.2394 | 0.0365 | 2 |
| `{False,True}` | 4 | 0.0000 | 0.0000 | 4 |
| `{Bad,Good}` | 1 | 0.5440 | 0.1510 | 1 |
| `{Bad,Good}` | 2 | 0.0000 | 0.0000 | 2 |

The m = 1 → 2 step is the informative one (m = 4 saturates trivially at
one-centre-per-task): **half as many centres as tasks removes 55 % of the
worst-task loss on `{False,True}` and 100 % on `{Bad,Good}`**.

**Routing — the honest read.**  `prototype` (stripped view) recovers the oracle
router's gain almost exactly: agreement 0.95–1.00, and its CI matches the oracle's
CI digit for digit ([+0.0000, +0.0156] for both).  So routing is *not* the binding
constraint here — quantization is (median routing loss +0.0000).  But per M2.1
this pools a trivial half (`{False,True}`, four different instruction sentences)
with a real half (`{Bad,Good}`, byte-identical instructions, where the +0.128 on
IMDB comes from input distribution alone).  Only the second half is evidence.

`confidence` and `batch_margin` both fail, as predicted in M1.4/M1.5:
`confidence` agreement 0.00–0.30, `batch_margin` 0.00 on every scope except MNLI's
degenerate single-centre case, with median gain −0.0000 CI [−0.1172, +0.0000] —
actively harmful.  **A parameter-free logit-only router does not work here.**

**M3.1 What the paper can claim, and what it cannot.**

*Can:* (i) the identification gap is real and mostly recoverable for free by
indexing the offset on the active label set (+5.92 pp median, CI excludes 0,
5/5 singleton scopes at exactly 0 as predicted); (ii) where verbalizer conflict
genuinely exists, worst-task loss follows `q_m` monotonically with ρ = +0.796,
and a codebook with m < |S| centres removes most of it; (iii) the mechanism is
quantization, not routing, on this benchmark.

*Cannot:* claim the codebook gain passes a preregistered significance test — it
does not, on either the frozen or the post-hoc form.  Claim task-free routing
works in general — one of two conflict scopes is textually self-identifying.
Claim a parameter-free router suffices — measured, it does not.

**M3.2 The one experiment that would change the verdict.**  Criterion 2 fails
because only 3 tasks sit in scopes with genuine conflict *and* enough training
signal.  Order-4's `{False,True}` quartet is the largest conflict group and its
members sit near 0.50 raw accuracy at 3 gradient steps per task, so there is
little for an offset to recover.  Training each task to convergence (rather than
the current 3–16 steps) is the change most likely to convert criterion 2, and it
does not require touching the method.  That is the next run, not a rewrite.

SHA-256 progression of this file: 7b0e820c… (frozen, pre-run) → 10ce4877… (M1)
→ 9badb858… (M2) → this revision adds M3 after the 3-seed run completed.
M3 was appended, never edited into §6; the frozen criteria are unchanged.

---

## M4 — convergence rerun, pre-registered before the run

**Frozen 2026-08-30, before any convergence run executes.**  M3's verdict stands
on its own; this amendment declares in advance what the rerun tests, so that its
outcome cannot be reinterpreted after the fact.

**M4.1 The defect being fixed.**  `--epochs` was a **dead parameter**:
`run_qoc.py` declared it and never read it, and `train_one_task` made exactly one
pass over the update split.  With `cap_per_class=200` and effective batch 64 this
gave **74 gradient steps for the entire 15-task stream** — 3 steps for most tasks,
1 for CB.  WiC/COPA/QQP therefore sat at 0.497–0.531 raw balanced accuracy, i.e.
chance.  A model that never learned a task cannot meaningfully forget it, so both
the "forgetting" framing and criterion 2's premise were weak by construction.
This is an implementation defect on my side, not a property of the benchmark.

**M4.2 What changes, and what deliberately does not.**  Two changes only:
`epochs` now takes effect (fresh shuffle per epoch, seeded by `(seed, epoch)`),
and a new `--update-cap-per-class` enlarges **only** the update split.
Everything else — model, LoRA config, lr, effective batch, scope construction,
codebook solver, routers, scoring, criteria — is untouched.

The risk/audit splits stay **byte-identical** to the M3 run.  This is the point of
adding `--update-cap-per-class` instead of simply raising `--cap-per-class`:
risk/audit are drawn by rank *within the capped group*, so raising the cap would
silently swap which examples are evaluated and make the two runs incomparable.
`grow_update_split` draws extra update examples from the full train pool minus the
reserved risk/audit ids; `test_update_split.py` pins the invariance.  Consequently
**M3 and M4 are comparable task-by-task on identical evaluation sets**, and the
only varying quantity is the amount of training.

The function lives in `run_qoc.py`, not `order4_data.py`, because the latter's
SHA-256 is checked by Phase-2I's frozen dependency list
(`run_pcsm_lora_transfer.py`); editing it would break the audit chain of already
sealed results.  I attempted the edit there first and the guard caught it, which
is the guard working as designed.

**M4.3 Configuration.**  `--update-cap-per-class 400 --epochs 3`, seeds 1/2/3,
all else at M3 defaults.  Predicted ≈1045 gradient steps (14× M3).  Chosen as the
smallest setting that gives every binary task ≳35 steps while keeping a 3-seed
sweep inside one overnight window on a shared box.

**M4.4 Predeclared outcomes.**  Criterion 1/2/3 as frozen in §6, unchanged, plus
one gate that must be reported regardless of how the criteria land:

* **Training gate.**  Median raw balanced accuracy across scorable tasks, measured
  immediately after each task is trained (`R_post_raw`), must exceed **0.60**.  If
  it does not, the run has *not* achieved convergence and criterion 2's outcome
  carries no more weight than M3's — I will report the gate as failed rather than
  quote the criteria as if the premise held.
* **Criterion 2 converts** only if the median codebook gain over scorable tasks
  has a cluster-bootstrap 95 % CI excluding 0, on the frozen (not post-hoc) form.
* **Criterion 2 does not convert.**  Then the honest conclusion is that the
  codebook gain is real but concentrated (M3 found it in 3 of 11 tasks) and does
  not clear a median test on this benchmark at any training length we can afford.
  That is a reportable negative and the paper keeps the M3.1 split of claims.
* **Criterion 1 must not regress.**  If longer training *reduced* the scope gain
  below its M3 CI, the free-capacity claim itself is training-length dependent and
  must be reported as such.  I predict it will *grow*, because a model that has
  actually learned each task has a larger per-task optimal offset to recover; that
  prediction is on record here and can falsify me.

**M4.5 What this run cannot fix.**  It does not prove Prop 3 for `m ≥ 2`; it does
not change the routing-evidence split of M2.1 (`{False,True}` remains textually
self-identifying, so only `{Bad,Good}` counts); and it does not make a
parameter-free logit-only router work.  Those are separate open items.

**M4.6 Smoke run, and one anomaly recorded before the full run reports.**
3-task smoke (MNLI/CB/WiC, `--update-cap-per-class 400 --epochs 3`) finished
exit=0 in 26 min on a shared card.  Post-training raw balanced accuracy, smoke vs
M3, same evaluation examples:

| task | steps M3 → M4 | R_raw M3 → M4 |
| --- | --- | --- |
| MNLI | 4 → 57 | 0.5156 → **0.8281** |
| CB (not scorable) | 4 → 10 | 0.5417 → 0.6667 |
| WiC | 3 → 38 | 0.4922 → **0.5000** |

MNLI confirms the diagnosis: the dead `epochs` parameter, not the benchmark, was
holding accuracy at chance.  **WiC did not move**, despite its training loss
falling 4.2881 → 0.2816.  Fitting the training objective while staying at chance
on held-out audit data is overfitting on 800 examples, not a scoring bug — WiC is
word-sense-in-context, the hardest task in the stream and the one where T5-large
with LoRA r=8 on q/v has least to work with.  I am recording this **before** the
15-task run reports so it cannot be reinterpreted afterwards: if WiC stays at
0.50 in the full run, the training gate must be reported as failing *for WiC*
even if the median clears 0.60, and WiC's contribution to criterion 1/2 medians
should be read with that in mind.  It also means κ for WiC (0.0182, the worst
task, `notes/theory_prop2_proof.md` §6) will not improve on a trained model.

**M4.7 Card assignment (operational, no bearing on results).**  Three seeds run
in parallel on cards 1/4/6, pinned by `CUDA_VISIBLE_DEVICES`.  The queue's
`pick_card` selects the emptiest card by free memory, so three drivers polling
simultaneously all claimed card 1 — observed, then corrected by pinning and
setting `gpu_gib=0` so the queue no longer chooses.  Each stage still waits for
≥6 GiB free on *its own* card before starting, so the non-preemptive rule is
unchanged: wait, never displace another user's job.
