# Phase-2I acquisition-conditioned oracle headroom

Status (2026-08-29): runner and smoke test implemented; full T5-small run is
**not authorized by the current acquisition evidence**.

## Why the previous trigger is invalid

The independent rescue artifacts froze QQP and BoolQA at LR 1e-3, 96 examples
per class, one epoch, and MultiRC at the same LR/cap with two epochs.  They do
not jointly establish the strict prerequisite:

- the audits contain equal counts per label, so their EM is balanced accuracy,
  not natural-prior accuracy;
- reweighting the BoolQA confirmation recalls by its training prior changes the
  apparent +10.35-point balanced gain to about -1.46 points;
- MultiRC was row/example-ID-disjoint, but 38/512 confirmation rows shared a
  paragraph+question semantic group with update or tune data.

Therefore these results diagnose learnability and select a development
schedule; they do not yet trigger official-test oracle evaluation.

## Locked corrected probe

Implementation:
`experiments/phase2i_anchored_cvar/run_acquisition_conditioned_headroom.py`.

For task *t* and each arm:

1. Measure pre-acquisition free-generation EM and per-example loss on a
   train-derived audit/risk split.
2. Reset AdamW and run current-only acquisition.  The frozen schedule is LR
   1e-3, cap 96/class, one epoch except MultiRC (two).
3. Measure natural-prior audit EM on an exactly class-balanced audit by
   reweighting per-class recall with the label prior from the complete pinned
   train split.  The primary gate is post minus pre **strictly greater than** 5
   points.  Post-generation valid-label rate must be at least 0.99 and every
   class recall delta must be at least -0.05.  Balanced accuracy and absolute
   worst-class recall are auxiliary.
4. At the Phase-A model, refresh risk on historical anchors only.
5. Reset AdamW and run one exposure-matched consolidation epoch containing the
   current batch plus historical replay.
6. Admit current risk/audit anchors after consolidation.

Semantic split units are QQP normalized-question connected components, BoolQA
normalized passages, and MultiRC normalized paragraph+question pairs; other
tasks use atomic examples.  The acquisition audit never uses descending group
size priority.  It targets complete-train label-conditional multi-row
representation, applies seeded SHA-256 random priority within strata, selects
whole multi-row groups with deterministic sparse DP, and uses hash-random
singleton fillers for the exact per-label quota.  The manifest reports group
counts, group-size histograms, row counts by size, and audit-versus-corpus
multi-row rates.  Phase-A optimizes per-example target-sequence mean
NLL with exact `pi(y)/q(y)` weights, where `pi` is the complete-train prior and
`q` is the balanced update sampler.  Phase-B retains that current-data
objective, while replay rows use the same unweighted per-example sequence NLL
in both arms after sampling.  The two arms are
`uniform_er_rank4` and `task_id_oracle`; both keep a uniform rank-four q/v LoRA
mask and identical acquisition/consolidation optimizer-step counts.  The only
intended intervention is uniform historical sampling versus task-ID
worst-group DRO sampling.

The default is fail-closed.  If any acquisition guard fails, the process stops
before official test.  `--allow-failed-acquisition` can continue train-only
diagnostics but still cannot write `TRAINING_COMPLETED` or access test.  Official
test admission requires the exact 15-task Order-4 sequence, a real T5-small
checkpoint, every acquisition guard, independently revalidated cross-arm
resource parity, and both arm sentinels.  Selected prefixes, `--smoke`, and
`--tiny-random-model` are intrinsically test-sealed even when every selected
gate passes.
The offline pass validates and loads the initial base, every task's incoming
state, its immediate pre-consolidation Phase-A adapter, every deployed
post-consolidation state, and the final adapter.  Incoming `pre_t` is the
initial state for task 1 and the previous stage's deployed state for every
later task.  The primary definitions are
`acquisition_t=phaseA_post_t-pre_t`,
`BWT_t=final_t-phaseA_post_t`, and
`R_t=(final_t-pre_t)/max(acquisition_t,0.05)`.  Natural-prior-reweighted
lower-tail retention uses these task-specific incoming states.  The
common-initial-pretrained-base version is explicitly auxiliary only.  Raw
balanced-subsample final accuracy and raw final arm gaps are separate
diagnostics, not retention.

Both arms score the complete replay buffer; the uniform controller ignores its
scores, while the oracle converts task-group risks into sampling weights.  The
runner asserts equal per-stage optimizer steps and current/replay example
exposures, records controller-forward counts, and reports aggregate non-pad
token differences.  It does not claim FLOP matching without padded-shape,
attention, and controller-arithmetic instrumentation.  `--stop-after-task`
writes an auditable prefix and keeps test sealed; automatic crash resume is not
implemented.

## Verification performed

- focused protocol tests cover task-specific schedules, all three semantic
  grouping rules, hash-random/no-size-first exact audit selection, balanced
  splits and full-train priors, strict gate
  guards, objective parity after sampling, controller/resource ledgers,
  stop/failure sealing, and initial/incoming/immediate/deployed/final offline
  evaluation;
- a one-task tiny-random smoke deliberately failed acquisition and verified
  that both arm and root official-test sentinels remained sealed;
- tiny-random smoke artifacts are wiring evidence only and have no scientific
  interpretation.  No full T5-small training has been launched.

Pinned seed-42 split-only verification (no model training) gives multi-row
audit rows versus complete-train rows: QQP `2/64` versus `49/2000`, BoolQA
`7/64` versus `214/2000`, and MultiRC `20/64` versus `629/2000`.  This replaces
the invalid size-first enrichments (`39/64` and `64/64` for QQP/BoolQA).
