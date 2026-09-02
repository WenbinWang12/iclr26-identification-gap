# Phase-2J — offset-conflict probe protocol (GO/KILL)

Written 2026-08-29, **before** any T5-large stream run.  Freeze this file and
record its SHA-256 before the first measurement.

## 0. The one question

Does the **identification gap** — the part of output-layer forgetting that a
task-agnostic shared offset cannot remove — actually exist and actually predict
worst-task retention, on a real generative LLM under the official benchmark?

If yes, the two-capacity framing of
`notes/theory_output_layer_capacity_v1.md` has an empirical anchor and the
method (rank budget + task-free offset capacity, CVaR over offset-corrected
regret) is motivated by measurement rather than by analogy.

If no, we do not have a paper on output-layer capacity, and we revert to the
pure capacity-allocation line (Track-1 spectral tail + oracle headroom) and say
so.  **Both outcomes are publishable inputs; only one of them is this paper.**

## 1. Why the previous probe design was wrong

An earlier plan made the main quantity
ρ = (EM_corrected − EM_raw) / (EM_post − EM_raw), the recoverable fraction of
forgetting.  That is exactly Davari et al.'s representation-forgetting gap and
its answer is already known from vision CL ("large").  Measuring it again in a
new setting is a replication, not a contribution.  ρ is retained here as a
**diagnostic**, not as the decision criterion.

## 2. Primary quantities

Per old task g, at stream time t after g has been learned:

| symbol | definition | requires task ID? |
|---|---|---|
| `R_raw`   | official free-generation normalized EM, no offset | no |
| `R_orc`   | EM with the best offset fitted **for task g alone** | yes (upper reference) |
| `R_shr`   | EM with a single offset shared by all seen tasks | no |
| `Δ_id`    | `R_orc − R_shr` — the identification gap | — |
| `ρ`       | `(R_orc − R_raw)/(R_post − R_raw)` — diagnostic only | — |

Offsets are additive on verbalizer-token logits only, fitted by balanced
logistic calibration on a **train-derived** risk split, never on test.  Report
both the fitted-offset value and, separately, the oracle-offset upper reference;
per `theory_output_layer_capacity_v1.md` §5 the fitted one is the honest number.

**Conflict statistic.**  Stack fitted per-task offsets as columns of `B̂`
(restricted to shared verbalizer tokens).  Report
- `ω̂` = spread of the per-task optima (max pairwise distance), and
- the singular-value tail `σ_{m+1..r}(B̂)` for m = 1, 2, 4.

## 3. Predeclared GO/KILL criterion

Run over the Order-4 **prefix of the first 8 tasks** (not all 15 — the prefix is
where verbalizer sharing already occurs and it halves the cost), 3 seeds,
T5-large, LoRA r=8 on q/v, official prompts, 200/class cap.

GO requires **all three**:

1. **The gap is real:** median over old tasks of `Δ_id` ≥ 3 pp, with a paired
   cluster-bootstrap 95% CI excluding 0.
2. **It is not just estimation noise:** `Δ_id` computed with oracle offsets
   (upper reference) exceeds `Δ_id` computed with fitted offsets by less than
   half of `Δ_id` itself — i.e. the gap is driven by genuine conflict, not by
   our inability to fit an offset.
3. **It predicts what we claim:** across the 3 seeds × old tasks, the conflict
   statistic `ω̂` correlates with worst-task retention loss, Spearman ρ ≤ −0.4.

KILL if (1) fails: no identification gap ⇒ no output-layer capacity story.
If (1) and (2) hold but (3) fails, the theory survives but the *allocation*
claim does not; fall back to a diagnostic-plus-theory paper and say so.

## 4. What is measured, and what is sealed

- Fit all offsets on the train-derived risk partition.  `test.json` is **not**
  read by this probe at all; the probe reports EM on a held-out train-derived
  audit partition.  Rationale: official test is already burned as development
  (recorded in `phase2i_acquisition_rescue_v2_runlog.md`), so it cannot serve as
  a confirmation set, and this probe is explicitly exploratory.
- Log per-example verbalizer margins so κ (margin density,
  `theory_output_layer_capacity_v1.md` §5) is estimated from the same run.
- No oracle, no five-arm panel, no rank exchange.  This probe touches none of
  the sealed Phase-2I partitions.

## 5. Cost

Measured on the shared box 2026-08-29 (T5-large, LoRA q/v, 512 src, ~6GB free):
229 ms/step at batch 4, generation 31 ex/s at batch 32.  Official global batch
64 is reached by gradient accumulation ×16 (batch 8 OOMs at this memory
ceiling); this changes no mathematics but must be declared.

8 tasks × 200/class × 1 epoch, plus per-stage re-evaluation of all earlier
tasks, ≈ 3–5 GPU-hours per seed ⇒ **~12 GPU-hours for 3 seeds**.  Fits inside
one card at the current shared-memory ceiling.

## 6. Reporting discipline

- The probe is **exploratory and post-hoc relative to QQP/BoolQA**, whose
  results motivated the theory.  Say so.
- Any headline claim must be confirmed later on partitions untouched at the time
  the criterion above was frozen.
- Report failures in full, including the case where fitted offsets transfer
  badly — that is itself the §5 caveat and belongs in the paper.

## 7. Amendment A1 — 2026-08-29, before any Δ_id was measured

§1–§6 above were frozen at 2026-08-29T07:14:42Z with SHA-256
`fc74fe4614576d3ebe84a2c284a66214362a0f5ef29eb5f91ac5d13e2f3dd5d9`.  This
amendment is appended, not edited in, so the original text and its hash remain
checkable.  It is triggered by a **measurement about the tokenizer**, made
before any identification gap was computed on any task
(`runs/phase2j_verbalizer_audit.json`, model `google-t5/t5-large`):

**A1.1 Two Order-4 tasks are outside the formalism.**  Yelp (position 9) and
Amazon (position 10) both use the 5-way set
`{very negative, negative, neutral, positive, very positive}`.  Under T5's
sentencepiece, `very negative` and `very positive` share their first piece
(id 182, `▁very`).  A task-agnostic offset on the first decode step therefore
**cannot** separate those two labels, for any b.  These tasks are not evidence
against the theory; they are outside its stated precondition.  The probe must
detect this and either skip the pair or score them on a two-step criterion,
declared in advance.  **Decision: skip Yelp and Amazon for Δ_id, and report
them explicitly as excluded-by-precondition rather than omitting them.**

**A1.2 Sharing is at the piece level, and it is heavier than the string-level
table suggested.**  14 first pieces are shared across tasks, versus 12 shared
label strings.  The four groups the theory relies on all survive at piece level:
`10998/10747` (`True`/`False`) across **WiC, QQP, BoolQA, MultiRC**;
`3`/`27252`/`7163` (`entailment`/`contradiction`/`neutral`) across
**MNLI, CB, RTE**; `1804`/`3862` (`Good`/`Bad`) across **IMDB, SST-2**;
`5716` (`Sports`) across **AGNews, Yahoo**.

**A1.3 Four collisions are invisible to string-level analysis** and couple b
without any shared label string.  These must be in the accounting:

| first piece | colliding labels |
| --- | --- |
| 71 | `COPA:A`, `DBpedia:Athlete` |
| 182 | `Yelp:very negative/very positive`, `Amazon:very negative/very positive` |
| 1769 | `AGNews:Business`, `Yahoo:Business & Finance` |
| 2854 | `AGNews:Science or Technology`, `Yahoo:Science & Mathematics` |

`offsets.py::fit_shared_offset` already indexes by label token rather than
position, so it handles A1.2/A1.3 correctly **provided the caller keys tasks by
first piece id, not by label string**.  That is now a requirement on the runner.

**A1.4 Consequence for scope.**  The `{True,False}` quartet is complete only at
position 14 (MultiRC).  A probe over the first 8 tasks reaches 3 of 4
(WiC, QQP, BoolQA) and captures the NLI triple in full; it does not test the
quantization form on a 4-point set.

## 8. Amendment A2 — 2026-08-29, §5 cost estimate was wrong; scope widened

Also written before any Δ_id was measured.

**A2.1 The §5 figure does not follow from §5's own numbers.**  Measured on the
box at 2026-08-29T18:41 (`runs/phase2j_smoke_large_accum/smoke_report.json`,
T5-large, LoRA q/v r=8, bf16 backbone with fp32 LoRA factors, 512 source
tokens, batch 4 × grad-accum 16 = official global batch 64, card shared with
another user's 77 GB job): **2.777 s per global step ⇒ 23.05 train examples/s,
52.79 eval examples/s, peak 2.10 GiB.**

Under §5's own stated budget — 8 tasks, 200 per class, 1 epoch, plus re-
evaluation of every earlier task on 128-example audit sets — that is
3200 training examples and 4608 evaluation examples, i.e. **≈0.07 GPU-hours per
seed, ≈0.2 GPU-hours for 3 seeds**.  The §5 claim of "3–5 GPU-hours per seed
⇒ ~12 GPU-hours" is **not reproducible from those numbers** and is withdrawn;
it overstated the cost by roughly two orders of magnitude.  The earlier
229 ms/step measurement it cited was a *micro-batch* step (batch 4, no
accumulation), so it was never the right unit for a global-batch budget.

**A2.2 Scope widened to the full 15-task stream.**  The 8-task scope existed
only to control that cost.  With cost no longer binding, the probe runs the
whole pinned Order-4 stream, which is also the actual benchmark rather than a
prefix of it.  Δ_id is scored on the **13 tasks that satisfy the A1
precondition**; Yelp and Amazon are still trained on, because they are part of
the stream and their interference is real, but they are not scored for Δ_id and
are reported as excluded-by-precondition.  This closes the `{True,False}`
quartet (MultiRC, position 14) and adds the `Sports` collision (Yahoo,
position 15).

Widening before any result is visible cannot launder a negative into a
positive, and it makes the test harder rather than easier: more tasks means more
opportunities for the shared offset to be adequate, which is the null.

**A2.3 The GO/KILL criterion in §3 is unchanged.**  All three conditions,
their thresholds, and the cluster-bootstrap procedure stand exactly as frozen.
Only the task count and the cost estimate move.

## 9. Amendment A3 — 2026-08-29, §3's own sampling budget is infeasible on CB

Also written before any Δ_id was measured.  This one corrects an error in the
frozen §3, not in an estimate.

**A3.1 §3 asks for a budget one task cannot supply.**  §3 declares "200/class
cap" and §4 fits offsets on a train-derived risk split with a disjoint audit
split; the working figure was 64 per class for each.  Measured class counts in
the pinned `train.json` files:

| task | n | K | rarest class |
| --- | --- | --- | --- |
| CB | 250 | 3 | **16** (`neutral`) |
| COPA | 400 | 2 | 195 |
| all others | 2000–14000 | 2–14 | ≥ 759 |

**CB's `neutral` class has 16 training examples in total.**  risk=64 plus
audit=64 is impossible there, and even a feasible split (7/8) would leave
single-digit per-class samples on both sides.  The frozen §3 budget was never
checked against the data; that is my error and it is recorded rather than
quietly adjusted.

**A3.2 Resolution.**  Per-task risk/audit allowances are derived from the
rarest class as `risk = min(64, ⌊(rarest−1)/2⌋)`,
`audit = min(64, (rarest−1) − risk)`, and the realised allowance for every task
is written into the output JSON.  All 14 non-CB tasks receive the full 64/64, so
for them nothing changes from §3.  Tasks whose rarest class holds fewer than
**40** examples are **excluded from Δ_id scoring** — with a handful of samples
per class the fitted offset and the score it is measured against are both
dominated by sampling noise, so any Δ_id computed there measures estimation
error, which is precisely what criterion (2) exists to rule out.  On the pinned
data this gate excludes **CB only**.

**A3.3 Cost of A3.2, stated plainly.**  CB is still trained on and still
contributes its conflicting demands to the shared offset, but it is not scored.
So the NLI group contributes **MNLI and RTE** to Δ_id rather than all three, and
RTE's label set is a subset of MNLI's.  The `{True,False}` quartet
(WiC/QQP/BoolQA/MultiRC) is therefore the primary evidence for the
quantization form, with `Good`/`Bad` (IMDB/SST-2) and `Sports` (AGNews/Yahoo)
as secondary.  If the headline result ends up resting on the NLI group, the
CB exclusion must be reported as a limitation in the same breath.

**A3.4 The GO/KILL thresholds remain untouched.**  A3 changes which tasks are
scorable and how many samples back each score.  It does not change the 3 pp
threshold, the half-of-Δ_id condition, the ρ ≤ −0.4 threshold, or the
cluster-bootstrap procedure.

---

## Amendment A4 — the "first step decides the label" assumption, measured

Appended 2026-08-29 **after** the three-seed GO verdict, because the verdict
itself surfaced the inconsistency.  A4 does not change any threshold or any
number in the verdict; it identifies which of §2's two definitions of `R_raw`
the probe actually measures, and bounds the discrepancy.

**A4.1 The observation.**  §2 defines `R_raw` as official free-generation
normalized EM, while the offset acts only on the first decoding step.  The
probe records both.  On 11 of 15 tasks they agree **digit for digit**
(AGNews 0.7461 / 74.61, DBpedia 0.8638 / 86.38, WiC 0.4922 / 49.22, ...).
On four they do not.  Worst case, seed 1: MNLI first-step balanced accuracy
0.3385 versus free-generation EM 21.35.  Seed 2 gave MNLI 4.17.

**A4.2 The mechanism is tokenization, not misclassification.**  Measured on the
trained model over the audit split (seed 1):

| task | first-step argmax | free-gen EM | argmax=gen | gen 1st piece = argmax column | off-label gen |
| --- | --- | --- | --- | --- | --- |
| MNLI | 0.3385 | 21.35 | 0.6823 | **1.0000** | 0.3177 |
| CB | 0.3750 | 37.50 | 0.8750 | **1.0000** | 0.1250 |
| RTE | 0.5469 | 53.12 | 0.9609 | 0.9844 | 0.0312 |
| COPA | 0.5000 | 49.22 | 0.9922 | **1.0000** | 0.0078 |
| 11 others | — | — | ≥ 0.9844 | ≥ 0.9906 | ≤ 0.0156 |

`gen 1st piece = argmax column` is **1.0000** on MNLI: for every single audit
example, the token greedy decoding emits first is exactly the column the
restricted first-step argmax selected.  The first-step decision is not being
overridden.  The divergence is entirely in the **continuation** — the decoded
string leaves the label set after the first piece.  Off-label generation, not
a different first-step choice, is the whole of the gap.

**A4.3 A hypothesis I formed and then falsified.**  Recorded rather than
deleted, because the discarded explanation is the one a reader will reach for.

*Hypothesis.*  T5's sentencepiece vocabulary has no `▁entailment`; the string
splits as `['▁', 'en', 'tail', 'ment']`, so its **first piece is id 3, the bare
`▁`** — a degenerate piece that prefixes any out-of-vocabulary whole word and
carries no information about `entailment`.  The three tasks with a degenerate
first piece are exactly the three NLI tasks:

| task | labels | first pieces |
| --- | --- | --- |
| MNLI | neutral, entailment, contradiction | 7163, **3**, 27252 |
| CB | entailment, contradiction, neutral | **3**, 27252, 7163 |
| RTE | contradiction, entailment | 27252, **3** |

The story would be: column 3 wins, and greedy decoding then wanders into some
other `▁`-prefixed word.

*Falsified.*  Breaking the off-label rate down by **which label the first step
actually chose** (seed 1, audit split) kills it:

| task | argmax choice | n | pieces | off-label |
| --- | --- | --- | --- | --- |
| MNLI | contradiction | 191 | 1 | 0.3194 |
| MNLI | neutral | 1 | 1 | 0.0 |
| MNLI | **entailment** | **0** | 4 | — |
| CB | contradiction | 18 | 1 | 0.0 |
| CB | neutral | 6 | 1 | 0.5 |
| CB | **entailment** | **0** | 4 | — |
| RTE | contradiction | 110 | 1 | 0.0091 |
| RTE | entailment | 18 | 4 | 0.1667 |
| COPA | A | 128 | 1 | 0.0078 |

On MNLI and CB the degenerate column is chosen **zero times**, so it cannot be
responsible for any off-label generation there.  MNLI's entire 0.3177 off-label
rate sits on `contradiction`, a **single-piece** label.  The multi-piece
`entailment` does carry the worst per-choice rate where it is chosen at all
(RTE, 0.1667 vs 0.0091), so piece count is *a* contributing factor — but it is
not the mechanism, and the headline MNLI number is not explained by it.

*What the evidence actually says.*  The decoded strings are
`'contradiction'`, `'contradiction'`, `'contradiction sentence 2'`, … — the
chosen label followed by extra tokens.  That is a **termination** failure, not a
label-identity failure: the model commits to the right first piece and then
fails to emit EOS, so official EM scores the whole string as wrong.  The
distinguishing measurement is whether the chosen label remains a *prefix* of the
decoded string; that is `offlabel_with_label_prefix`, added to the probe and
reported in A4.7.  COPA's residual 0.0078 is one example.

Note also that the degenerate `▁` collides across MNLI, CB and RTE, and `▁A`
(id 71) is claimed by both COPA's `A` and DBpedia's `Athlete`.  Both are real
first-piece collisions and belong in A4.4 regardless of this task's outcome.

**A4.4 Full cross-task first-piece collision table** (this supersedes the
string-level table in `theory_output_layer_capacity_v1.md` §3, which was
annotated there as incomplete):

| first piece | id | claimants |
| --- | --- | --- |
| `▁` (degenerate) | 3 | MNLI/entailment, CB/entailment, RTE/entailment |
| `▁True` | 10998 | WiC, QQP, BoolQA, MultiRC |
| `▁Fal` | 10747 | WiC, QQP, BoolQA, MultiRC (`False` = 3 pieces) |
| `▁neutral` | 7163 | MNLI, CB, Yelp, Amazon |
| `▁contradiction` | 27252 | MNLI, CB, RTE |
| `▁very` | 182 | Yelp/Amazon `very negative` **and** `very positive` |
| `▁Good` / `▁Bad` | 1804 / 3862 | IMDB, SST-2 |
| `▁negative` / `▁positive` | 2841 / 1465 | Yelp, Amazon |
| `▁Sports` | 5716 | AGNews, Yahoo |
| `▁Business` | 1769 | AGNews/`Business`, Yahoo/`Business & Finance` |
| `▁Science` | 2854 | AGNews/`Science or Technology`, Yahoo/`Science & Mathematics` |
| `▁A` | 71 | COPA/`A`, DBpedia/`Athlete` |

Two entries here were not in the earlier string-level table: `▁A` and the
degenerate `▁`.  Both are cross-task collisions between labels whose *strings*
differ, which is precisely the case a string-level audit cannot see.

**A4.5 What this does and does not cost us.**

*Does not:* the identification gap is defined and measured on the first-step
quantity throughout — `R_orc`, `R_shr` and `Δ_id` all come from
`collect_logits`, never from `generate`.  A4.2 shows the first-step decision is
faithfully executed by decoding on every task including MNLI.  So the GO verdict
(median Δ_id 6.25 pp, cluster-bootstrap 95% CI [3.9, 9.4] pp, ρ = −0.556 over
42 points, 11 scorable tasks, 3 seeds) is unaffected.

*Does:* §2's claim that `R_raw` is "official free-generation EM" is **false as
written** for the NLI group and mildly false for COPA.  What the probe reports
as `R_raw_balanced` is first-step restricted-argmax accuracy.  These are not
interchangeable, and the paper must say which one every number is.  Concretely:

1. §2's `R_raw` row is amended to read **"first-step restricted-argmax accuracy
   on the task's verbalizer columns."**  The free-generation EM is retained as a
   separately named quantity, `free_gen_em`, reported alongside.
2. Any claim of the form "an offset recovers X points of official EM" is **not
   licensed** by this probe on MNLI/CB/RTE.  An offset provably fixes the
   first-step choice; where decoding fails to terminate, fixing that choice does
   not make the decoded string equal the label.  Paper-level claims about EM must
   either restrict to the 11 clean tasks or add constrained decoding.
3. The theory's §5 assumption "one decision step, not one token per label" is
   **necessary, not cosmetic**: A4.2/A4.3 exhibit four benchmark tasks where the
   decision-level statement holds exactly (first piece emitted = argmax column,
   1.0000 on MNLI) while the string-level statement fails.  The assumption is
   what separates the two, so it has to be stated wherever EM is quoted.

**A4.6 The clean fix, deferred deliberately.**  Constraining decoding to the
verbalizer set (score each label string's full sequence, take the argmax) would
make free-generation EM equal first-step accuracy by construction on all 15
tasks.  It is also a departure from the official O-LoRA evaluation, so adopting
it silently would break comparability with published numbers.  Decision: keep
the official free-generation number as the comparability anchor, report
first-step accuracy as the quantity the theory is about, and report both.
Do not present either as the other.

**A4.7 The mechanism, measured.**  Third probe run, seed 1, all 15 tasks,
audit split.  `prefix` = of the off-label generations, the fraction where the
first-step argmax label is still a **prefix** of the decoded string.  `noEOS` =
fraction that never emitted EOS within `max_new_tokens=8`.

| task | argmax | free-gen EM | off-label | prefix | noEOS |
| --- | --- | --- | --- | --- | --- |
| MNLI | 0.3385 | 21.35 | 0.3177 | **1.00** | 0.0000 |
| CB | 0.3750 | 37.50 | 0.1250 | **1.00** | 0.0000 |
| RTE | 0.5469 | 53.12 | 0.0312 | 0.75 | 0.0000 |
| COPA | 0.5000 | 49.22 | 0.0078 | **1.00** | 0.0000 |
| IMDB | 0.8984 | 88.28 | 0.0156 | **1.00** | 0.0000 |
| AGNews, Amazon, BoolQA, DBpedia, MultiRC, QQP, SST-2, WiC, Yahoo, Yelp | — | = argmax×100 | **0.0000** | — | 0.0000 |

**Every off-label generation on 14 of 15 tasks retains the argmax label as a
prefix.**  The single exception is one RTE example (1 of 110 `contradiction`
choices).  `noEOS` is 0.0000 everywhere, so this is not a truncation artifact:
the model *does* terminate, but only after emitting extra tokens
(`'contradiction sentence 2'` rather than `'contradiction'`).

So the inconsistency is a **decoder termination** failure, fully downstream of
the label decision:

1. The first-step decision is executed faithfully — `gen 1st piece = argmax
   column` is 1.0000 on MNLI (A4.2).
2. The decision is *preserved* through decoding — the label survives as a prefix
   in 100% of off-label cases (this table).
3. Official EM is exact-match, so a correct label plus trailing tokens scores 0.

This makes the gap a property of how little each task was trained (3 gradient
steps for MNLI at this scale, so the "stop after the label" behaviour is not yet
learned), not a property of the label geometry the theory is about.  It also
explains the seed spread flagged earlier — MNLI free-gen EM 21.35 (seed 1),
4.17 (seed 2), 31.25 (seed 3) against an essentially flat first-step accuracy of
0.3385 / 0.3333 / 0.3177.  **First-step accuracy is the stable quantity; the EM
is the noisy one**, which is the opposite of what one would assume, and it is
why A4.5's renaming matters.

*Consequence for A4.3's hypothesis:* definitively closed.  Multi-piece labels
are not the cause.  MNLI's whole off-label mass sits on `contradiction`, a
single-piece label, and the decoded prefix is always correct.  The degenerate
`▁` first piece for `entailment` remains a genuine defect of the verbalizer
design — it is a first-piece collision across MNLI/CB/RTE (A4.4) and it means
those three tasks cannot be separated by a first-step offset in the `entailment`
direction — but it is not what produced the EM/argmax discrepancy.

*Consequence for the paper:* the discrepancy does **not** threaten the
formalism.  It threatens any sentence that quotes EM as if it measured the
label decision.  A4.5's two-name rule is sufficient, plus one added line: when
reporting free-generation EM on the NLI group, state that the residual is
trailing-token termination and give the prefix-accuracy number alongside, since
that is the quantity a constrained decoder would recover.
