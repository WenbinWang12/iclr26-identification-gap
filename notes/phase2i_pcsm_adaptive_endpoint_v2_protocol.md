# PCSM adaptive-endpoint v2 protocol lock

Lock time: `2026-08-29T11:32:31+08:00` (Asia/Shanghai; filesystem creation
time of this lock).

Status: run-local method-development lock.  This is not an external
preregistration.  It was written after observing BoolQA selection/confirmation
and the QQP v1 internal audit, but before the primary agent or the independent
design reviewer inspected any MultiRC v1 internal metric or result artifact.

## Motivation and evidence roles

The frozen PCSM v1 recipe learned a strong QQP ranking but failed its raw
class-safety guard: raw natural-prior EM improved by 15.782 pp and AUC by
0.0970, while False recall fell by 8.594 pp.  Independently fitted thresholds
on the same internal partition yielded a 10.454 pp natural-prior gain and a
10.156 pp minimum class gain.  This QQP internal result is explicitly
**discovery evidence** and cannot validate v2.

BoolQA's existing raw result is retrospectively compatible with v2, but is
not a prospective v2 test.  QQP development and fresh confirmation remain the
first clean validation partitions for the calibrated fallback.  MultiRC data
retain their role according to the access ledger produced by the already
running frozen-v1 trajectory; no v2 claim may label a partition blind if that
trajectory accessed it before v2 evaluation.

## Frozen training recipe

Training is unchanged from PCSM v1 transfer:

- T5-small LoRA; sequence-mean gold-label NLL weighted by the pinned natural
  train prior divided by the paired sampler prior 0.5;
- 384 fit rows per class, two complete cycles, deterministic 2+2 paired
  batches, 384 optimizer steps, AdamW, learning rate 3e-4, no weight decay;
- eval-mode gradients with dropout disabled; exactly one eligible checkpoint
  at step 384;
- semantic-group-disjoint fit, internal, development, and fresh confirmation
  partitions; no test/dev corpus is used for fitting;
- no target-specific learning-rate, epoch, rank, prompt, or checkpoint search.

## Internal endpoint-selection rule

For every task, compute raw free-generation metrics and constrained-label
margins on the train-derived internal partition.  Fit base and post decision
thresholds independently with the same exhaustive natural-prior threshold
fitter on that internal partition.  The thresholds are then immutable.

Define the common representation checks:

1. post **raw free-generation** valid-label rate is at least 0.99;
2. post-minus-base margin ROC-AUC is strictly positive.

Endpoint selection is ordered and deterministic:

1. Select `raw` if the common checks pass and raw natural-prior EM gain is
   strictly greater than 0.05 and the minimum raw per-class recall delta is at
   least -0.05.
2. Otherwise select `internal_frozen_calibrated` only if all of the following
   hold: the common checks pass; raw natural-prior EM gain is strictly greater
   than 0.05; the raw failure is a class-safety failure (minimum raw class
   delta below -0.05); calibrated natural-prior EM gain is strictly greater
   than 0.05; and the minimum calibrated per-class recall delta is at least
   -0.05.
3. Otherwise the task fails internal qualification.  No endpoint is selected
   and development/confirmation remain inaccessible.

The fallback therefore repairs a measured decision-boundary shift; it cannot
turn absent raw acquisition, invalid free generation, or non-improving ranking
into a pass.  The valid-label check always comes from raw free generation,
because constrained binary decoding would make that check vacuous.

## Development and confirmation gates

The selected endpoint is frozen before opening development.  Development may
not switch endpoints or refit either threshold.  It passes only if, for that
selected endpoint:

- natural-prior EM gain is strictly greater than 0.05;
- minimum per-class recall delta is at least -0.05;
- raw post valid-label rate is at least 0.99; and
- margin ROC-AUC delta is strictly positive.

Only a development pass permits one atomic claim on a fresh confirmation
partition.  Confirmation uses the same selected endpoint, frozen thresholds,
checkpoint, split, and gates exactly once.  Calibrated fallback results must
be reported as calibrated/constrained-label accuracy, not silently relabeled
as raw generation accuracy.  Raw and alternative-endpoint results remain
diagnostics and cannot rescue a failed selected endpoint.

Uncertainty is frozen as a paired semantic-group cluster bootstrap with seed
`43002`, 10,000 valid replicates, and a 95% percentile interval.  Sample
semantic groups with replacement, carry every row and its paired base/post
prediction whenever a group is drawn, and compute class recalls followed by
the pinned-prior estimand.  A replicate lacking either label is discarded and
redrawn until 10,000 valid replicates are obtained.  No row-level interval may
replace this cluster interval for MultiRC.  A point-estimate gate pass is not a
statistical-significance claim.

## Stopping rule and oracle boundary

No official continual oracle-headroom run is authorized unless BoolQA, QQP,
and MultiRC each pass a fresh one-shot confirmation by more than 5 pp under
their internally selected endpoint, while satisfying validity, class-safety,
and AUC checks.  Failure of any task stops the oracle transition.

Implementation must be a new v2 runner/format with its own source hash.  The
frozen PCSM v1 runners and artifacts must not be edited or overwritten.
