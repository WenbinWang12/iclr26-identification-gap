# Phase 2G GCDR locked exploratory test

Status: written before the first execution of the seeds below.  This is a local
prospective lock, not an externally registered preregistration.

## Development choice

The phase-2f development panel (teachers `20260701, 20260710..13`, streams
`6060,6061`) was used to compare pseudo-group counts and mean margins.  We freeze
the simplest stable setting before opening the test panel:

- method: `explore_gcdr.run_gcdr`;
- phase-2f 4-uniform + 12-input-direction coverage working set;
- duplicate windows removed rather than double weighted;
- 6 gradient-signature pseudo-groups;
- 24 multiplicative-weights rounds, eta `0.7`;
- actual weighted empirical RRR re-estimated every round;
- empirical working-set mean-NMSE margin `0.01` relative to uniform empirical RRR;
- rank `R=4`, ridge `1e-6`;
- learner seeds `7,17,29`, averaged within each teacher x stream world.

No phase-2g configuration change is permitted after the test output is viewed.

## Previously unused test panel

- teacher seeds: `31415901..31415905`;
- stream seeds: `16180331..16180334`;
- 5 x 4 crossed worlds, three learner-buffer seeds per world;
- regime: phase-2f `capacity_limited` benchmark.

Before writing this file, exact-string searches found no occurrence of these
teacher or stream seeds in the workspace or the latest Claude main-session log.

## Decision criteria

All criteria use the 20 per-world values after averaging the three learner seeds.
The reported crossed interval is the existing two-way additive random-effects
interval with conservative `df=min(5,4)-1=3`; it is not called a generic
cluster-robust sandwich interval.

1. Mean non-inferiority to frozen phase-2f PSR: the lower 95% crossed CI of
   `GCDR mean - PSR mean` must exceed `-0.01`.
2. Tail superiority to frozen phase-2f PSR: the lower 95% crossed CI of
   `GCDR worst-source retention - PSR worst-source retention` must exceed zero.
3. Tail superiority to excess-CVaR: the corresponding lower 95% crossed CI must
   exceed zero.

Passing (2), rather than merely beating an older replay baseline, is the primary
evidence that the targeted repair adds value over the current method.  A failure
on any criterion is reported as a failed targeted remedy.

