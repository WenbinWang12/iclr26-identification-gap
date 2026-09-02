# SGCR-v4 independent implementation audit

Date: 2026-08-27 (Asia/Shanghai)

Status: **DEVELOPMENT AUDIT**. This note does not upgrade the burned 8x8 panel
to prospective evidence.

## Bottom line

No high-severity implementation defect was found that would directly fabricate
the reported SGCR-v4 result.  An independent rerun exactly reproduced the
worst world:

```text
t39817664 / s37032991 / learner31770922
v4  mean/tail = 0.950502 / 0.499460
v2  mean/tail = 0.933801 / 0.783754
PSR mean/tail = 0.959343 / 0.660865
```

The audit found no source/test-label leakage, future-window leakage,
train/audit row overlap, rank violation, or RRR/SVD orientation error.  The
minimal self-test and a hash check of all ten SGCR-v2 locked files passed.

The v4 result is therefore credible only as a descriptive result for the
specified burned panel and one learner seed.  It is not confirmation of method
effectiveness or safety.

## Claim-critical limitations

1. The 64 worlds had already been opened and used to develop v3/v4.  The
   crossed interval does not account for this adaptive method-selection bias or
   for learner-seed variability.
2. Four worlds have `v4 tail - PSR < -0.10` (worst `-0.161405`), so v4 is not
   Pareto-safe or reliable per stream.
3. The spectral path was selected in only 3/64 worlds.  The positive average
   cannot be attributed to the latest spectral idea, and v4-v2 tail has crossed
   95% interval `[-0.01061,+0.07017]`.

## Implementation and accounting qualifications

- The arrival-frequency state is exact only after the first codebook fit.  The
  first 20 windows are initialized from the retained train view: in the worst
  world this used 672 retained rows rather than all 960 train-role arrivals and
  had initial frequency L1 error 0.155; the final error was about 0.0105.
- Audit `Y` is selection-only, but full-window `X` affects the input-diverse
  coreset before the row split.  The row-level, uncorrected UCB is consequently
  only a conditional coreset stability heuristic, not a population confidence
  bound.
- `69,408 / 73,728` is a logical persistent numeric-payload ledger, not actual
  or peak resident memory.  It excludes the common model, Python containers,
  RNG state, and transient workspace.
- The comparison is conservative in one respect but not a formally symmetric
  byte ledger: the legacy PSR arm holds 16 raw windows and also stores 12 input
  directions, whereas the runner applies the metadata-inclusive assertion only
  to v4.
- The runner omits v4's same-pool frequency-q0 arm, so the effects of the shared
  coreset, activation weighting, audit selection, and spectral candidates are
  not fully isolated.

## Authorized wording

> On an already-burned controlled linear panel, SGCR-v4 showed a strong average
> tail signal relative to PSR with a small mean cost.  Independent audit found
> no direct leakage or solver error, but four hidden-source catastrophes, the
> absence of a fresh panel, and a single learner seed preclude an effectiveness
> or safety claim.  The newly added spectral path was not the observed driver.

