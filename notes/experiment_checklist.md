# Experiment Checklist

## Phase 0: Audit

- [ ] Verify requested LoRA rank, factor shapes, and numerical rank for every old checkpoint.
- [ ] Record exact train tokens, optimizer steps, replay bytes, and optimizer-state bytes.
- [ ] Reproduce sequential LoRA with at least three seeds before adding FCRA.

## Phase 1: Synthetic Falsification

- [x] Generate streams with known innovation rank and curvature spectrum.
- [x] Include aligned, orthogonal, cancelling, and recurrent update sequences.
- [x] Verify the Taylor expansion and remainder trace against step radius.
- [ ] Export a publication figure for the Taylor remainder against step radius.
- [x] Verify spectral-tail capacity exactly in the quadratic setting.
- [x] Verify safe-nullspace contraction with fixed additive scaling.
- [x] Demonstrate the cross-block counterexample for naive top-k allocation.

## Phase 2: Controlled Mixture Drift

- [ ] Normalize FPB, FiQA-SA, and financial-news sentiment labels and prompts.
- [ ] Treat financial sentiment as a low-conflict control, not assumed capacity pressure.
- [ ] Build a mixed-skill stream with one SFT loss and source IDs hidden from learners.
- [ ] Pre-register and run an innovation-rank pilot before comparing methods.
- [ ] Create stationary, smooth, abrupt-recurrent, and expanding-support schedules.
- [ ] Hide source labels from all learners.
- [ ] Report separate capacity-, active-compute-, and history-memory-matched comparisons.
- [ ] Sweep `R={8,16,32,64}` and `k={2,4,8}` where feasible.
- [ ] Log predicted historical cost and actual old-loss change every window.

## Phase 3: Chronological Streams

- [ ] Require a documented pretraining cutoff before the stream or build a post-cutoff slice.
- [ ] Run base-model contamination probes and report suspected memorized items separately.
- [ ] Build TemporalWiki and StreamingQA windows using timestamps only for ordering.
- [ ] Label stable, new, updated, and obsolete facts for evaluation.
- [ ] Audit a stratified label sample and retain both historical and current truth.
- [ ] Report stale-answer persistence separately from stable-fact forgetting.
- [ ] Measure general-capability drift on an unchanged held-out suite.

## Baselines

- [ ] Sequential LoRA, dense rank-R, random-k, and hard partition/freeze.
- [ ] Replay-LoRA with byte-matched memory.
- [ ] O-LoRA and SLICE.
- [ ] E2-LoRA; ProCL/NSR/LiteLoRA with all bank/router memory accounted for.
- [ ] Euclidean SVD, curvature-weighted SVD, DELLA, and TIES for consolidation.
- [ ] Offline joint, growing rank, per-window oracle adapters, and full FT as oracles only.

## Statistics and Reporting

- [ ] At least 3 seeds; target 5 for main tables.
- [ ] Paired stream realizations and paired bootstrap confidence intervals.
- [ ] Mean, standard deviation, per-stream win rate, and failure cases.
- [ ] No best-seed reporting.
- [ ] Publish total/active parameters, memory, FLOPs, tokens, time, and latency.
- [ ] Pre-register primary metrics and numerical-rank thresholds before the final sweep.
- [ ] Keep full historical evaluation reporting-only; all decisions use memory counted in budget.
