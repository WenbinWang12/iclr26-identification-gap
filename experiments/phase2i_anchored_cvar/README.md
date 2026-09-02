# Phase-2I anchored-CVaR implementation scaffold

Status: **auditable five-arm runner implemented; full T5-small result is a
separate run artifact, not checked into this directory**.

This directory starts the locked Phase-2I protocol in
`../../notes/phase2i_anchored_cvar_global_rank_protocol.md`.  It currently
implements:

- per-example anchored retention regret and acquisition eligibility;
- the exact capped-weight empirical upper-tail CVaR objective;
- task-level normalized retention for evaluation.
- the locked five-arm GO/KILL decision, so a weak result cannot be promoted by
  changing thresholds after the run.
- pinned O-LoRA Order-4 train/test acquisition and prompt construction;
- a fixed 144-atom T5-small q/v LoRA bank with exact payload exports;
- disjoint risk-train/audit reservoirs and metadata-free group-free views;
- the five-arm sequential runner in `run_order4_panel.py`, including stage
  checkpoints, resource ledgers, rank logs, and completion sentinels.

Run the dependency-light checks from the repository root:

```powershell
python -m pytest experiments/phase2i_anchored_cvar/tests -q
```

Full-order **wiring/acquisition smoke** (use a fresh output directory):

```powershell
python -m experiments.phase2i_anchored_cvar.run_order4_panel `
  --data-root D:\phase2i_order4_data `
  --model-path D:\hf_phase2i\manual\t5-small `
  --output-root D:\phase2i_order4_seed42 `
  --tasks all --arms all --seed 42 `
  --per-class-cap 4 --risk-per-class 1 --audit-per-class 1 `
  --eval-per-task 4 --batch-size 2 --replay-batch-size 2 `
  --max-source-length 64 --max-target-length 8 --smoke
```

`--smoke` preserves the selected Order-4 chronology; it only reduces caps and
sequence lengths.  `--tiny-random-model` is an integration-test switch and is
never a T5-small result.  The current adaptive-rank mechanism is explicitly
logged as a deterministic gradient/norm proxy exchange, not as the final
measured shortlist/probe controller.

The command above is not decision-grade: four sealed examples per task make EM
move in 25-point increments, the training cap is only four examples per class,
and source length is 64 rather than the locked 512.  It validates execution,
chronology, acquisition eligibility, controller separation, and physical
payload equality; it cannot establish positive headroom or rank-controller
superiority.

A decision-grade single-seed headroom run removes `--smoke` and uses the locked
probe caps/lengths:

```powershell
python -m experiments.phase2i_anchored_cvar.run_order4_panel `
  --data-root D:\phase2i_order4_data `
  --model-path D:\hf_phase2i\manual\t5-small `
  --output-root D:\phase2i_order4_headroom_seed42 `
  --tasks all --arms all --seed 42 `
  --per-class-cap 200 --risk-per-class 10 --audit-per-class 10 `
  --eval-per-task 200 --batch-size 8 --replay-batch-size 8 `
  --max-source-length 512 --max-target-length 50
```

That remains a T5-small feasibility panel, not a direct comparison to the
paper's T5-large numbers.  Multi-seed confirmation and a measured
shortlist/probe rank exchange are required before treating a positive wiring
signal as the paper's method contribution.

## Train-only acquisition rescue

`run_acquisition_rescue.py` searches update LR, epoch checkpoint, nested
per-class cap, and loss objective on QQP, BoolQA, and MultiRC without loading
benchmark evaluation or development data.  Strict runs use semantic-group
disjoint train-derived update/tune-audit/confirm-audit partitions.  Version 2
derives the natural label prior from every verified pinned train row, hashes
that derivation, and reports natural-prior EM, balanced EM, per-class recall,
worst-class recall, and valid-label rate.  Its gate requires natural-prior gain
strictly above 5 points, valid-label rate at least .99, and no class recall drop
below 5 points.  The legacy `normalized_em` key remains a balanced-EM alias.

Keep confirmation sealed during pilot/successive-halving runs:

```powershell
python -m experiments.phase2i_anchored_cvar.run_acquisition_rescue `
  --data-root D:\phase2i_order4_data `
  --model-path D:\hf_phase2i\manual\t5-small `
  --output-root D:\phase2i_acquisition_selection_seed42 `
  --tasks QQP BoolQA MultiRC --seed 42 `
  --lrs 0.0003 0.001 0.003 --epochs 1 3 5 10 `
  --caps-per-class 16 32 64 `
  --loss-objectives sequence_mean_balanced hf_token_ce `
  --tune-audit-per-class 32 --confirm-audit-per-class 64 `
  --batch-size 8 --max-source-length 512 --max-target-length 50 `
  --truncation-mode official_right --selection-only
```

Then confirm exactly one durable task selection in a fresh output directory:

```powershell
python -m experiments.phase2i_anchored_cvar.run_acquisition_rescue `
  --data-root D:\phase2i_order4_data `
  --model-path D:\hf_phase2i\manual\t5-small `
  --output-root D:\phase2i_acquisition_confirm_qqp_seed42 `
  --confirm-selection `
    D:\phase2i_acquisition_selection_seed42\QQP\selection.json
```

Within each LR/cap/objective tuple, requested epochs are evaluated from one
shared trajectory rather than retrained from scratch.  In selection-only mode,
if a sequence-level objective is requested, `hf_token_ce` is excluded from the
selection grid.  After a durable selection lock is written, exactly one
token-CE control is trained from the same initialization at the selected
LR/epoch/cap/sampler; it is marked selection-ineligible and never receives
tune evaluation or confirm access during selection.  A later locked
confirmation evaluates both selected and matched-control checkpoints on the
same confirm audit.  Token-CE-only and search-then-confirm invocations retain
their historical behavior.  Token CE uses Hugging Face T5's
`output.loss` (all-target-token CE), matching Seq2SeqTrainer reduction.
`field_aware` is an explicit truncation ablation: for BoolQA it protects the
question and `Answer:` suffix; for MultiRC it protects the question, candidate
answer, and `Answer:` suffix while allocating remaining tokens to the passage.
The default `official_right` reproduces the upstream right-truncation behavior.
The `sequence_mean_balanced` objective averages per-example target-token-mean
NLL; `sequence_mean_prior` applies the unnormalized importance weight
`pi(label)/q(label)`, where both the full-train prior and balanced-sampler `q`
are derived and recorded.  Every selection candidate and post-selection
control has a content-addressed checkpoint/metrics registry entry and a
fail-closed resource-ledger validation.  The confirmation bundle records the
matched token-CE checkpoint and the pre-control selection-lock hash.

## Acquisition-conditioned oracle headroom

`run_acquisition_conditioned_headroom.py` is the corrected two-arm runner.  It
uses field-aware length 512, a fresh optimizer for every task/phase, current-
only acquisition before any replay, and then an exposure-matched
current+historical
consolidation phase.  Current anchors enter history only after consolidation.
The comparison isolates risk allocation at fixed rank 4:

- `uniform_er_rank4`: deployable group-free uniform replay;
- `task_id_oracle`: privileged worst-task DRO replay.

Its primary acquisition gate is natural-prior free-generation EM on a new,
exactly class-balanced train-derived audit: per-class recalls are reweighted by
the label prior from the complete pinned train split.  QQP normalized-question
connected components, BoolQA normalized passages, and MultiRC normalized
paragraph+question pairs are indivisible across update/risk/audit.  Audit does
not prioritize large groups: it targets each label's complete-train multi-row
representation, uses seeded hash-random group priority with deterministic
sparse subset DP, then fills the exact label quota with hash-random singletons.
The manifest records audit and complete-corpus group counts, size histograms,
and multi-row row rates.  Phase-A
uses per-example target-sequence mean NLL weighted by `pi(y)/q(y)`.  Phase-B
keeps that current-data objective; sampled replay rows use the same unweighted
per-example sequence NLL in both arms.  The primary gate requires gain **strictly greater
than** 5 points, post-generation validity at least 0.99, and no class-recall
drop below -0.05.  Balanced accuracy and absolute worst-class recall are
auxiliary only.
The official test loader is unreachable until both arms have durable
`TRAINING_COMPLETED` sentinels.  With the default fail-closed setting, any task
failing any guard stops the run while official test remains sealed.  Even
`--allow-failed-acquisition` only permits a train-only diagnostic continuation;
it never opens official test.  Official admission additionally requires the
exact 15-task Order-4 sequence and a real T5-small run: every task prefix,
`--smoke`, and `--tiny-random-model` is intrinsically test-sealed.  The offline
evaluator independently rechecks stage completeness and cross-arm resource
parity before its sole sealed-data load point.

The frozen development schedule is one acquisition epoch for all tasks except
MultiRC (two), LR 1e-3, and at most 96 update examples per class.  QQP/BoolQA=1
and MultiRC=2 reproduce the earlier row-held-out selections, but do not imply
that a strict natural-prior/group-disjoint rescue has passed.  Run the strict
train-only prerequisite first; only after every task passes should the full
headroom command be launched:

```powershell
python -m experiments.phase2i_anchored_cvar.run_acquisition_conditioned_headroom `
  --data-root D:\phase2i_order4_data `
  --model-path D:\hf_phase2i\manual\t5-small `
  --output-root D:\phase2i_acq_conditioned_headroom_seed42 `
  --tasks all --arms all --seed 42 `
  --update-cap-per-class 96 --risk-per-class 16 `
  --acquisition-audit-examples 64 `
  --default-acquisition-epochs 1 `
  --acquisition-epoch-overrides MultiRC=2 `
  --consolidation-epochs 1 --lr 0.001 `
  --batch-size 4 --replay-batch-size 4 `
  --max-source-length 512 --max-target-length 50
```

This remains a development oracle-headroom diagnostic.  It is not a formal
benchmark confirmation and it does not implement adaptive rank allocation.
Each arm performs the same full-buffer controller-loss scoring pass, even
though the uniform arm ignores task groups, and the artifact asserts matched
optimizer steps plus current/replay example exposures.  Aggregate non-pad token
counts are reported separately; the runner makes no FLOP-matched claim.
The initial base, every task's incoming state, every immediate
pre-consolidation Phase-A adapter, every post-consolidation deployed adapter,
and final adapters are provenance-linked by path and SHA.  For task `t`, the
primary offline metric uses `pre_t` = initial state for task 1 and the previous
stage's deployed state thereafter:
`R_t=(final_t-pre_t)/max(phaseA_post_t-pre_t,0.05)`, with
`BWT_t=final_t-phaseA_post_t`.  A common-initial-base version is retained only
as an explicitly auxiliary diagnostic.  The raw final accuracy gap is reported
separately and is never called retention.  `--stop-after-task` creates an
auditable prefix with test sealed; automatic crash resume is not implemented.
