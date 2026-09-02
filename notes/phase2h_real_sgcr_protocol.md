# Phase-2H: Real-Transformer SGCR bridge protocol

**Status: DEVELOPMENT / UNRUN.**

**Date drafted:** 2026-08-27 (Asia/Shanghai).

This document is a development protocol, not a preregistration and not a
result record.  No dependency has been installed, no model or dataset has been
downloaded for this phase, and no run described here has been executed.  Any
number below is a proposed configuration or an acceptance threshold, not an
observed outcome.

The purpose of Phase-2H is to test whether the failure-driven SGCR mechanism
survives the smallest credible transition from controlled linear reduced-rank
regression to:

1. real text;
2. a pretrained Transformer;
3. an actual factorized LoRA update trained by AdamW; and
4. the model's nonlinear cross-entropy loss.

A positive development run would justify writing a separate hash-locked fresh
confirmation protocol.  It would not, by itself, establish an LLM-scale,
multi-benchmark, or ICLR-level empirical result.

## Current environment and execution boundary

At the time this protocol was written, the local environment contained:

- PyTorch `2.9.1+cpu`, with no CUDA device;
- `datasets`, but no `transformers` or `peft` package;
- no cached Hugging Face model suitable for this experiment.

Phase-2H therefore begins with a dependency/model/data preflight.  This
document does **not** authorize a dependency installation or a long run.  The
first implementation should use a manually audited LoRA wrapper and require
only `transformers` in addition to the packages already present; `peft` is not
required.

## Claim boundary

The intended positive claim, if a later fresh confirmation passes, is narrow:

> On a hidden-domain, mixed, heavy-tailed stream of real sentiment reviews, a
> single fixed-rank Transformer LoRA using source-free stable-activation cells,
> minimum-plus-shared replay, and a small robust replay correction improves
> worst-source retention over byte- and compute-matched replay while preserving
> mean performance.

The phase must not be described as evidence that:

- all forgetting is a capacity problem;
- activation cells consistently recover true tasks or domains;
- `K_max=2R` estimates the number of latent sources;
- the reused audit coreset gives a population-level safety theorem;
- the method works on LLMs, generation, arbitrary task sequences, or
  unobservable/contradictory tasks; or
- the current controlled SGCR-v2 numbers are Transformer evidence.

## Why this task is identifiable

The benchmark uses one label semantics in every hidden domain:

```text
stars in {1,2} -> negative
stars in {4,5} -> positive
stars == 3     -> excluded
```

Thus the six sources do not define six incompatible target maps.  The learner
is asked to learn one sentiment function under covariate and frequency shift.
This differs from the Phase-2B negative construction, in which statistically
identical inputs could require incompatible source-specific outputs without an
observable context.

The source itself is not assumed to be perfectly recoverable.  Product-domain
vocabulary and writing style may make it statistically visible in frozen
Transformer activations, but that is an empirical precondition.  The
observability gates below must pass before SGCR is trained.  If the frozen
representation cannot expose the source structure, the experiment is killed
instead of routing by the true category.

To remove literal conflicts and leakage:

- normalize whitespace and Unicode before hashing text;
- remove exact normalized-text duplicates across splits;
- remove and report every cross-category duplicate or duplicate with
  inconsistent polarity;
- never concatenate `product_category` to the model input; and
- keep source metadata in a report-only evaluator table keyed by immutable
  example ID, outside every learner-facing record.

At inference, prediction is always `single BERT + single LoRA + fixed head`.
There is no source classifier, task ID, cell ID, router, or adapter selection.
Cells affect historical storage and replay weights only.

## Data and split construction

Use the English configuration of the Multilingual Amazon Reviews Corpus
(MARC), pinning the exact repository revision and raw-file hashes in the data
manifest.  The canonical fields are `review_id`, `product_id`, `review_title`,
`review_body`, `stars`, and `product_category`.

The source list is selected in a count-only preflight before any model forward
pass:

1. apply the polarity filter above;
2. count positive and negative examples per category in each official split;
3. retain categories with at least 512 examples of each polarity in train and
   at least 32 of each polarity in both validation and test;
4. rank eligible categories by their minimum train-polarity count, descending,
   with the category string as the deterministic tie break; and
5. take the first six and write the ordered list to the immutable data
   manifest.

If fewer than six categories qualify, stop and revise the protocol before any
model outcome is observed.  Categories may not be replaced because their
frozen activations or method results look unfavorable.

Use `review_title + " [SEP] " + review_body`, maximum sequence length 64.
Store canonical replay records as fixed tensors rather than Python strings:

```text
input_ids[64]       int32
attention_mask[64]  uint8
label               int8
priority            float64
example_id          uint64
cell_id              uint8  # SGCR/cell arms only
```

Official validation is development-only.  Official test is never used for
hyperparameter choice, early stopping, the audit gate, source selection beyond
the count-only eligibility rule, or representation selection.  A later fresh
confirmation must evaluate official test only after its code/configuration is
locked.

## Stream construction and chronology

Each paired arm receives the same examples, order, common initialization, and
learner random seed.

- 60 windows, 96 examples per window;
- six warmup windows followed by 54 drift windows;
- labels are balanced within each realized source/window block, up to the
  deterministic one-example remainder for an odd count;
- train/audit role is a permanent 50/50 hash split of immutable example ID and
  a fixed protocol salt, shared across arms;
- audit-role examples are never used in gradients, replay training, center
  estimation, or arrival-frequency estimation.

Every warmup window contains 16 examples from each source, eight per polarity.
Warmup is a common initialization stage, not an oracle input to the learner:
the data generator uses source metadata to construct the benchmark, but the
learner receives only tokens and polarity.

The 54 drift windows form three consecutive 18-window phases.  Before selecting
the 2--4 sources present in a window, use the corresponding source-probability
vector:

```text
phase A: [0.55, 0.22, 0.10, 0.07, 0.05, 0.01]
phase B: [0.10, 0.55, 0.22, 0.07, 0.05, 0.01]
phase C: [0.22, 0.10, 0.55, 0.07, 0.05, 0.01]
```

The ordered category manifest defines indices 0--5.  A distinct seeded stream
draw chooses `2 + (window_index mod 3)` sources without replacement according
to the phase probabilities, then distributes 96 records among selected sources
by a conditioned multinomial with at least two examples per selected source.
The exact algorithm and RNG consumption order must be unit-tested and hashed.
Report realized source counts; do not repair a seed because its rare-source
count is inconvenient.

For window `t`:

1. historical replay may contain only records from windows `<t`;
2. split the current records into permanent train/audit roles;
3. perform the permitted learner updates on current-train plus historical-train
   replay;
4. compare candidates on audit-role records only;
5. deploy one candidate, including its optimizer state; and
6. only then offer the current records to their historical stores.

Current-audit records may be used in the candidate gate at their arrival but
never become gradient examples later.  No official validation/test example
enters the online state.

## Model and genuine LoRA path

Initial CPU model: `prajjwal1/bert-tiny` (`L=2`, hidden size 128).  Pin model,
tokenizer, `transformers`, PyTorch, and tokenizer-library revisions in every
run manifest.

Wrap the `query` and `value` linear projection in each of the two attention
layers with

```text
W(x) = W0(x) + (alpha / rank) * B @ A @ x
rank = 4
alpha = 8
LoRA dropout = 0
```

Freeze all pretrained weights.  The four LoRA modules contain 4,096 trainable
parameters in total.  Add a two-class linear head over attention-mask mean
pooled final hidden states.

During the six common warmup windows, train the head and LoRA identically for
all arms.  At the warmup boundary:

- freeze the classification head permanently;
- clone one common model/optimizer/buffer initialization into every arm;
- initialize the activation codebook; and
- start all reported continual comparisons from this common checkpoint.

Post-warmup, LoRA factors are the only trainable model parameters.  This makes
the nonlinear bridge stronger than merely attaching a continually trained
linear head.  The implementation must log trainable names, gradient norms,
factor norms, and the numerical rank of every `B @ A`; numerical rank above 4,
a gradient on a frozen base/head parameter, or a zero/non-finite update is an
integrity failure.

Initial development optimizer proposal:

```text
AdamW
LoRA learning rate: 5e-4
warmup-head learning rate: 1e-3
betas: (0.9, 0.999)
eps: 1e-8
weight decay: 0
gradient norm clip: 1.0
batch size: 16
mixed precision: disabled
```

Optimizer choice may be calibrated only on the declared development seeds and
official validation.  It must be frozen for every arm before any fresh
confirmation seed is run.

## Stable activation cells

The controlled `vec(xx^T)` signature is not carried over.  Transformer hidden
coordinates have a fixed sign and the quadratic signature would waste memory.

For each example, run the same frozen BERT with LoRA contributions disabled,
in evaluation mode and under `no_grad`.  Attention-mask mean-pool the final
hidden states, subtract the train-role warmup mean, and L2 normalize.  This
128-dimensional value is the source-free stable signature.  It is recomputed
from stored tokens when necessary and is not stored per replay example.

- `K_max=8`, inherited as a resolution cap, not as an estimator of six sources;
- deterministic non-empty spherical k-means at the warmup boundary;
- refresh every ten drift windows from train-store signatures only;
- align refreshed centers to persistent IDs using an exact assignment for
  eight cells;
- assign audit examples using train-derived centers without allowing them to
  change a center;
- mark a cell robust-eligible only if it has train and audit support and its
  train-only coherence is at least `0.60 * median_cell_coherence`.

Instantaneous per-example gradients are deliberately excluded from the cell
identity.  Controlled diagnostics found that residual-gradient signatures
were noisy and contaminated the rare group.  Real per-cell cross-entropy is
used for robust weighting, not for clustering.

## Minimum-plus-shared historical stores

Maintain permanently disjoint train and audit stores.  Each uses:

- a minimum of six priority-reservoir records per cell; and
- one global shared priority overflow for every remaining slot.

Unused cell minimums return to the shared overflow rather than wasting fixed
capacity.  Refresh may relabel retained records from train-derived centers and
rebuild bases/overflow, but it may not change train/audit role or priority.

Train arrival counts alone define the empirical frequency vector `q0`.
Source/category metadata must not be present in either store.

## Actual nonlinear robust replay and audit gate

After warmup, each window begins from one deployed LoRA/optimizer state:

1. take eight common AdamW steps using current-train and historical replay
   sampled according to `q0`;
2. snapshot that common state;
3. branch A takes four additional `q0` steps;
4. restore the common state and branch B takes four small robust steps; and
5. use disjoint audit records to choose A or B, restoring both model and
   optimizer state of the selected branch.

Every size-16 step uses eight current-train and eight historical-train examples
when history is available; the first window uses 16 current examples.  Sampling
draws and current minibatches should be coupled across branches where possible.

For branch B, use the eight lowest-priority train records in each eligible cell
as a deterministic loss probe.  Before each of its four updates, compute the
actual per-example Transformer cross-entropy and aggregate by cell.  Starting
from `q=q0`, update

```text
q_g <- Normalize(q_g * exp(0.1 * L_g / mean_active_loss))
```

over eligible cells, retain the original total eligible mass, and cap
`q_g / q0_g` at 4 before renormalizing.  Replay is then sampled from cells
according to `q` and uniformly within a selected cell.  This is a small-step
nonlinear robust correction, not an exact Group DRO solver or a nonlinear
analogue of the closed-form RRR optimum.

For the audit gate, evaluate both branches on:

- the eight lowest-priority audit records per supported cell; and
- current-window audit records.

Use train-arrival `q0` for the frequency-weighted historical audit mean.  Select
branch B only if all conditions hold:

```text
historical mean CE(B) <= historical mean CE(A) + 0.01
current audit CE(B)   <= current audit CE(A)   + 0.01
worst eligible-cell CE(B) < worst eligible-cell CE(A)
```

Otherwise select branch A.  Missing audit support, fewer than two eligible
cells, a non-finite loss, or an invalid rank forces branch A.  Audit records are
not refit after selection.  Because the audit coreset is reused across windows,
this is empirical buffer safety only.

## Fixed historical-byte budget

Define the primary logical historical-storage envelope as the exact bytes of
256 canonical non-cell replay records:

```text
M_history = 256 * bytes(
    int32 input_ids[64]
  + uint8 attention_mask[64]
  + int8 label
  + float64 priority
  + uint64 example_id
)
```

Under these declared dtypes, the raw envelope is 86,272 bytes.  SGCR must deduct
from this envelope:

- one `uint8` cell label per retained record;
- eight `float32[128]` centers;
- one `float32[128]` warmup mean;
- train arrival counts, occupancy masks, and all persistent counters (the
  per-record priority and example ID are already charged in the record schema);
- any other persistent numeric controller state actually introduced by the
  implementation.

The remaining capacity is split 50/50 between train and audit stores.  The
implementation derives capacity from actual tensor `element_size * numel`, not
from the approximate count in this prose, and asserts that six records per cell
fit in both roles.  If they do not, stop rather than silently increase the
budget.

Global ER may use metadata bytes it does not need to retain more examples under
the same envelope.  Persistent-loss baselines must charge their loss/age
metadata and therefore may retain fewer.  This is stricter and more informative
than equal-example-count matching.

The common frozen base, deployed LoRA, classifier head, and optimizer are
reported separately from historical storage.  Candidate model/optimizer
snapshots are transient controller workspace and must be included in peak
method memory.  For a strict peak-memory comparison, every baseline receives an
equal preallocated scratch tensor.  Report both logical persistent bytes and
measured process peak/RSS; neither may be described as the other.

## Baselines and resource matching

All trainable arms use the same common warmup checkpoint, LoRA class, optimizer,
current minibatches, history chronology, precision, and paired seeds.

Required arms:

1. **Sequential LoRA:** no historical examples; diagnostic forgetting arm.
2. **Global ER-step:** global uniform priority reservoir under `M_history`, 12
   deployed optimizer steps per window.
3. **Global ER-FLOP:** the main conservative ER comparator, 16 optimizer steps
   per window to match SGCR's total branch-update count.
4. **Stable-cell q0-step / q0-FLOP:** identical SGCR cells/store, with all
   deployed steps using `q0`.  The step arm takes 12 updates and the primary
   mechanism comparator takes 16, isolating minimum-plus-shared coverage and
   ensuring that SGCR cannot win merely because it computed both branches.
5. **Persistent-loss CVaR/JTT:** source-free global buffer plus one charged
   `float32` loss EMA per record.  Initialize the EMA from the first train loss,
   update it as `ema <- 0.9 * ema + 0.1 * current_loss` whenever that record is
   evaluated, and draw replay from a 50/50 mixture of the full reservoir and
   the highest-EMA 20% of retained records.  It receives the same 16 updates
   and total probe/audit forward budget as SGCR.
6. **True-category oracle replay:** development-only source-labelled minimum
   storage and group weighting, with the same bytes/compute; establishes whether
   perfect grouping could help.
7. **Offline-joint rank-4 LoRA:** report-only unbounded-data capacity and
   optimization upper bound.

The source-free deployable arms never receive `product_category`.  The oracle
and source metrics run through a separate evaluator interface whose tensors are
rejected by learner-facing type/schema tests.

In addition to update counts, ledger current/replay tokens, forward calls,
backward calls, probe forwards, audit forwards, checkpoint copies, wall time,
and peak memory.  To compute-match extra SGCR probes/audits, ER-FLOP and the
other primary baselines execute the same number and shapes of report-only
forward calls on their own buffers; these values may not affect their updates.

## Pre-method feasibility gates

These gates use only count-only data checks, the common warmup checkpoint,
frozen signatures, development seeds, and development validation.  SGCR may not
advance to a fresh confirmation unless all pass.

### F0: integrity and non-contradiction

- one polarity mapping is used in all categories;
- normalized text is disjoint across train/validation/test after deduplication;
- all cross-category duplicates and conflicts are removed and counted;
- learner traces, replay records, batches, signatures, and model calls contain
  no category/source field;
- replay chronology and permanent train/audit disjointness tests pass.

### F1: source-free observability

On development validation only, using the frozen signatures:

- a post-hoc source linear probe reaches balanced accuracy at least 0.75;
- the actual learner spherical-k-means reaches adjusted mutual information at
  least 0.35 and Hungarian/many-to-one purity at least 0.65; and
- for the globally rare source, its best matching cell has both precision and
  recall at least 0.50.

The source probe and matching are evaluator-only diagnostics.  Their weights or
labels are never available to SGCR.  High probe performance with low k-means
performance diagnoses the signature/clustering implementation; low probe
performance diagnoses an unobservable frozen representation.

### F2: oracle recoverability

Across three paired development stream seeds, true-category oracle replay must:

- improve final worst-source balanced accuracy over ER-FLOP by at least three
  percentage points on average; and
- lose no more than one percentage point of macro balanced accuracy on average.

If the oracle fails, no pseudo-group method is expected to rescue this exact
stream/budget, so SGCR tuning stops.

### F3: nonlinear LoRA viability

- the common warmup checkpoint and offline-joint rank-4 LoRA both exceed random
  balanced accuracy;
- offline-joint worst-source balanced accuracy is at least 0.65;
- disabling LoRA after post-warmup adaptation measurably changes logits and
  reduces held-out performance, rather than all behavior being attributable to
  a trainable head;
- every LoRA module receives finite nonzero gradients and has numerical rank at
  most four; and
- no frozen base or post-warmup head parameter changes.

## Development stages and permitted decisions

### Stage D0: static preflight

No training.  Resolve dataset/model revisions, build the count-only category
manifest, validate duplicates/splits/tokenization, and measure a 50-step CPU
microbenchmark.  If a batch step exceeds 0.8 seconds, use a shortened smoke run
or move the full panel to a GPU; do not compensate by reducing confirmation
seeds after seeing results.

### Stage D1: one-seed pipeline smoke

Use only 20 windows and one explicitly marked development seed.  Test finite
losses, chronology, role separation, rank, byte accounting, branching restore,
and source-schema rejection.  No outcome from this stage is citable.

### Stage D2: three-seed feasibility panel

Run frozen signatures, source probe/k-means diagnostics, offline joint, ER-FLOP,
and true-category oracle.  Evaluate F0--F3.  Do not implement a long SGCR sweep
if these gates fail.

One predeclared representation fallback is permitted after an F1 failure:
revise this protocol to use `prajjwal1/bert-mini`, repeat D0--D2 from scratch,
and mark all BERT-tiny outcomes burned.  Source labels may not be used to choose
a layer, concatenate category tokens, train a router, or hand-design clusters.

### Stage D3: SGCR and ablations

Run SGCR, both stable-cell q0 arms, ER-step, ER-FLOP, persistent-loss CVaR/JTT,
and the oracle on the same three development seeds.  Hyperparameters may change only
inside this development stage, and every tried setting is recorded.  Once a
candidate is chosen, freeze code/config/data and derive new seeds mechanically
before writing a separate confirmation protocol.

Approximate CPU planning targets, to be replaced by the D0 benchmark, are:

- setup/download: 10--25 minutes;
- preprocessing: 5--20 minutes;
- one-seed smoke: 15--45 minutes;
- three-seed development: 1--3 hours; and
- a later 5--10-seed main comparison: approximately 4--12 CPU hours.

## Development success and kill criteria

Phase-2H is a **development GO** only if all feasibility gates pass and, on the
three paired development seeds:

- SGCR's final worst-source balanced accuracy exceeds ER-FLOP on average;
- SGCR's macro balanced accuracy is no more than one point below ER-FLOP;
- stable-cell q0-FLOP improves tail over global ER-FLOP, indicating a coverage
  effect at matched total updates;
- SGCR improves tail over stable-cell q0-FLOP, indicating value from the robust
  correction despite the q0 arm's stronger deployed-step budget; and
- the robust branch is selected at least once and rejected at least once, so
  both correction and fallback mechanisms are exercised.

Any of the following is a **KILL/REVISE**, not permission to keep opening seeds:

- F0, F1, F2, or F3 fails;
- category/source enters learner state or text;
- only the category oracle is positive while pseudo-cells are not;
- the gain disappears against ER-FLOP and exists only against ER-step;
- all gain is explained by a continually trained classifier head;
- robust replay fails to beat stable-cell q0;
- the mean cost exceeds one percentage point;
- the byte or chronology ledger is unequal/invalid;
- non-finite loss, rank violation, stale audit training, or branch-restore error
  occurs; or
- a positive effect appears only after changing sources, imbalance, budget, or
  seeds in response to observed outcomes.

If stable-cell q0 wins but robust weighting does not, the robust component must
be removed from the proposed method and the paper story revised.  It may not be
kept merely because it helped the controlled solver.

## Requirements for a later confirmation

No confirmation is part of this document.  After development, create a new
`LOCKED_UNRUN` protocol and hash manifest before executing any fresh seed.

The recommended minimum is five fresh stream schedules crossed with two learner
seeds; a paper-grade panel should use at least ten schedules crossed with three
learner seeds, averaging learner replicates within schedule.  The primary
intersection-union gates should require:

1. lower 95% paired confidence bound for
   `SGCR macro BA - ER-FLOP macro BA > -0.01`;
2. lower 95% paired confidence bound for
   `SGCR worst-source BA - ER-FLOP worst-source BA > 0`;
3. a point tail gain of at least two percentage points;
4. lower 95% paired confidence bound for
   `SGCR worst-source BA - stable-cell-q0-FLOP worst-source BA > 0`; and
5. positive tail comparison to persistent-loss CVaR/JTT as a secondary gate.

Report per-source accuracy/CE, macro and frequency-weighted accuracy, minimum
source accuracy, backward transfer, current acquisition, buffer source survival,
cell occupancy/coherence, source-cell AMI/purity, branch-selection frequency,
all seed-level rows, and exact resource ledgers.  Wins are descriptive and do
not replace the paired interval.

## Intended implementation layout

```text
experiments/phase2h_real_sgcr/
  prepare_marc.py
  lora_bert.py
  buffers.py
  sgcr.py
  baselines.py
  feasibility.py
  run_development.py
  run_confirmatory.py       # create only with the later locked protocol
  tests/
    test_no_source_leak.py
    test_chronology.py
    test_rank.py
    test_byte_budget.py
    test_branch_restore.py
```

The first coding milestone is D0/D1 plus unit tests, not an end-to-end seed
sweep.  Until this protocol has actual run records, the paper and README must
continue to state that no real-Transformer SGCR result exists.

---

## Execution log (append-only; do not rewrite prior entries)

### 2026-08-27 — bert-tiny D1/D2 run, D2 = NO-GO, then predeclared fallback

The protocol body above was drafted UNRUN.  These are the actual run records; the
body is left unedited (honesty: do not rewrite the drafted plan after running).

- **D0** (bert-tiny, L=2, hidden=128): microbench 40.5 ms / 16-example CPU step; OK.
- **D1** (bert-tiny, 20-window smoke, seed 1234, cap 8000): PASS but NON-CITABLE.
  Required a structural fix to the data budget: `build_stream`/`build_from_ids`
  gained an `n_windows` parameter so the smoke builds only the 20-window prefix
  (byte-identical to the full-60-window prefix because the stream RNG is consumed
  strictly in window order); negatives are sparse in category-file order so the
  materialize cap was raised (8000 smoke / 12000 panel).  Branch-restore deviation
  0.0, head frozen, max_rank 4, finite losses.
- **D2** (bert-tiny, 3 dev seeds [1234,5678,9012], cap 12000): **NO-GO.**
  F0 PASS; F3 PASS (offline-joint worst BA 0.787, rank<=4, frozen unchanged, LoRA
  learns).  **F1 FAIL** (source-free probe BA 0.587 << 0.75, AMI 0.143, purity
  0.396, rare prec/rec 0.35/0.22).  **F2 FAIL** (true-label oracle lifts worst
  source only +1.55pp over ER-FLOP, need +3pp; macro loss 0.08pp OK).  Per the
  Stage-D2 rule "if the oracle fails, no pseudo-group method is expected to rescue
  this exact stream/budget", the D3 SGCR sweep was NOT run.  Full record:
  `experiments/phase2h_real_sgcr/D2_RESULT.md`, `d2_panel.log`.

- **Fallback taken (this section's decision):** exercising the ONE predeclared
  representation fallback permitted after an F1 failure (Stage-D2 above).  Backbone
  switched `prajjwal1/bert-tiny` (L=2, hidden=128) -> `prajjwal1/bert-mini`
  (L=4, hidden=256).  D0--D2 repeat from scratch.  **All bert-tiny D1/D2 outcomes
  are BURNED** (development-only, non-citable; retained only as an honest record).
  Byte envelope unchanged (86,272 B); signature dim 128 -> 256 charged in
  `buffers._SIGNATURE_DIM` (cell capacity 120 -> 113 records/store, still >= 6/cell).
  LoRA trainable params 4,096 -> 16,384 (4 layers x q,v).  No source label is used
  to choose a layer, add tokens, train a router, or design clusters.

  **HONEST RISK, recorded before the rerun:** bert-mini addresses F1
  (representation observability) but NOT necessarily F2 (oracle headroom).  If the
  worst source is already near-saturated under a stronger backbone, F2 can fail
  again — and this is the LAST permitted fallback.  A second NO-GO is the final
  Track-2 negative result for this stream/budget, to be reported as such.
