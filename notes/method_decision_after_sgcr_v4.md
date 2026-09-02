# Method decision after SGCR-v4

Date: 2026-08-27 (Asia/Shanghai)

## Decision

SGCR is no longer an "all methods have no effect" direction.  Two controlled
results support a real average mechanism:

- the prospectively locked SGCR-v2 run strictly improved tail retention over
  same-pool q0 and excess-CVaR while remaining mean-non-inferior to PSR;
- the later, fully burned SGCR-v4 development panel improved tail over PSR by
  `+0.08941`, with crossed 95% interval `[+0.03933,+0.13948]`, and paid
  `-0.00472` mean, within the one-point non-inferiority margin.

Neither result closes the method claim.  SGCR-v2 missed its primary PSR-tail
gate (`[-0.0018,+0.1310]`), and SGCR-v4 was developed on those opened worlds,
used one learner seed, and produced four catastrophic worlds below PSR by more
than ten points.  SGCR-v4 therefore remains **safety FAIL** and must not be run
on a new confirmation panel in its current form.

## What the evidence actually supports

The observed driver is the combination of:

1. a shared uniform plus input-direction-diverse bounded coreset;
2. source-free activation cells rather than noisy instantaneous gradients;
3. train-only cumulative arrival frequencies for the mean constraint;
4. train-only group-reweighted rank allocation; and
5. an empirical audit selector with an unconditional reference fallback.

The added fixed-teacher spectral path was selected in only 3/64 development
worlds.  It is not the empirical explanation for the average gain.

## Why the remaining failures cannot be thresholded away

All four catastrophes selected the activation path, but many matched positive
worlds selected the same path.  Candidate round, paired mean-UCB, worst-UCB
gain, coherence, support, arrival/retained mismatch, spectral gap, oracle
purity, and oracle NMI overlap strongly between failures and successes.  q0
fallback repairs only two catastrophes and substantially hurts the other two;
the same-pool spectral fallback also does not solve them.

The closest root cause is an identifiability mismatch: a pseudo-cell can have
good global purity while failing to isolate the hidden worst source.  Existing
learner-observable statistics do not reliably reveal that event.  Continuing
to tune a scalar threshold on these 64 worlds would be seed-specific repair.

## Recommended next method

The next deployable design should be **reference-anchored multi-view SGCR**, not
another UCB threshold on the current selector:

1. Keep one byte-matched direction-covered raw pool and exact post-warmup
   arrival counts.
2. Generate the activation-group candidate entirely from train folds.
3. Generate a PSR-style reference operator from the same raw pool and budget.
4. Evaluate risk over an uncertainty set containing both activation cells and
   whole-window/direction groups, using multiple fixed whole-window audit folds.
5. Deploy the SGCR challenger only when its mean constraint and tail advantage
   agree across views and folds; otherwise deploy the reference operator.

This changes the selection object rather than pretending the current
pseudo-cell worst is the true worst source.  It still cannot create information
when the latent source is unobservable, so representation observability remains
a precondition rather than a theorem.

No further threshold or candidate-grid tuning should use the present 64-world
panel.  A new controlled development panel is required before this design can
earn a fresh confirmation.

## Immediate empirical priority

Move first to the locked-down feasibility sequence in
`notes/phase2h_real_sgcr_protocol.md`:

1. verify that hidden MARC product domains are visible in frozen BERT
   activations;
2. verify that source-free cells recover the rare source sufficiently;
3. verify with a true-category oracle that the fixed-budget trade-off is
   actually repairable;
4. only then implement the real rank-4 Transformer-LoRA controller and compare
   with byte- and FLOP-matched ER, q0, and persistent-loss baselines.

If observability or oracle feasibility fails, kill the benchmark/method pair.
If both pass, this is a cleaner and more valuable next result than further
optimization of the controlled linear panel.

