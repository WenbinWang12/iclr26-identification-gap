# Phase-0 Diagnostic Pilot (Frozen Before the Confirmatory Run)

## Purpose

This pilot tests whether a fixed-rank adapter can have a measurable online
tracking failure before any FCRA implementation is treated as evidence. It
separates four cases:

1. compatible observations with order interference (a memory/online negative
   control);
2. orthogonal compatible observations (a no-interference control);
3. changing full-matrix targets (a conflict and acquisition-matching control);
4. an actually rank-inadequate stream (a capacity control).

The compatible stream is deliberately not an allocation claim. Its common
rank-R solution lets us test whether a sequential update can lose historical
observations even when rank is not the bottleneck. A positive result there is
evidence for online interference or bounded history, not for FCRA.

## Exact models

### Compatible matrix-sensing stream

Draw a fixed matrix `X_star` with numerical rank `R`. Window `t` supplies one
normalized input direction `v_t` and the noiseless target
`y_t = X_star v_t`. Its loss is

`L_t(X) = 0.5 * ||X v_t - y_t||_2^2`.

The current-only minimum-change update is

`X_t = X_(t-1) + (y_t - X_(t-1) v_t) v_t^T`.

It fits the current direction exactly. With zero initialization, the iterates
remain in the column space of `X_star` and therefore have rank at most `R`.
This rank-preservation condition is tested, not assumed for arbitrary
initialization. The offline rank-R optimum has zero prefix loss because
`X_star` is feasible. Non-orthogonal directions can nevertheless reintroduce
old residuals; repeated directions eventually remove this transient.

The exact replay oracle solves the prefix least-squares problem from all
stored directions. It is an online feasibility upper bound, not a proposed
method. Its storage and solve cost are logged separately. On this compatible
construction, the rank-R offline prefix solution is already exact, so
`replay_loss == offline_rank_loss` by construction. The resulting
`addressable_gap` is retained as a raw comparison but is **not an independent
addressability test**; the manifest marks this explicitly.

### Full-observation target stream

For rank-R matrices `M_t`, use `L_t(X) = 0.5 ||X-M_t||_F^2`. The high-rank
prefix optimum is the mean `M_bar`; the rank-R prefix optimum is its truncated
SVD. A current-only sequential baseline uses `X_t=M_t`. A cancellation stream
can make the raw sequential prefix gap large while the current acquisition is
perfect. The acquisition-matched oracle for this full-observation control is
the current target itself; thus its `acquisition_matched_gap` is zero. The
ordinary replay gap is reported separately and is not used to claim that
conflict is addressable. This prevents conflict from being misreported as
allocation.

## Primary quantities

For each prefix `t`, let `H_t(X)` be the average prefix loss, `O_inf,t` the
offline high-rank loss, `O_R,t` the offline rank-R loss, and `B_t=H_t(X_0)`
the no-adaptation baseline. The eligible normalized quantities are

`capacity_ratio = (O_R,t-O_inf,t)/(B_t-O_inf,t)`

`total_online_gap = (H_t(X_seq)-O_R,t)/(B_t-O_inf,t)`

`addressable_gap = (H_t(X_seq)-H_t(X_replay))/(B_t-O_inf,t)`.

For full-observation conflict controls, report the separate quantity
`acquisition_matched_gap = (H_t(X_seq)-H_t(X_acq_match))/(B_t-O_inf,t)`.
Here `X_acq_match` is the current target. It is a defined zero control for
rank-feasible stationary/canceling targets; in the rank-inadequate control it
is the nonzero rank-specific acquisition residual and is not used as a
conflict gate.
All gap fields also retain an unclamped `*_raw` value; machine-scale noise is
clamped only in the normalized convenience field.

The denominator is only valid when it exceeds `1e-12 * max(B_t, 1)`. Raw
losses are always reported. Current acquisition is reported separately as
`L_t(X_seq)-min_rankR L_t`, and past retention uses the prefix excluding the
current window. No normalized value is imputed for an ineligible prefix.

## Frozen decision rules

These are diagnostic gates, not claims that a method is good.

- Compatible-stream capacity adequacy: median `capacity_ratio <= 0.10` and
  every eligible prefix `<= 0.20`.
- Compatible-stream online-interference signal: the 20-seed **paired-by-seed
  contrast** (correlated minus orthogonal) lower 95% bootstrap bound for the
  normalized prefix mean of `total_online_gap` is at least `0.10`; current
  acquisition ratio is at most `0.05` on at least 95% of eligible prefixes.
- Addressability check: this gate is **not applicable** to the compatible
  stream because replay equals the exact rank-R offline oracle. It remains a
  field for future incompatible streams. The cancellation control instead
  requires positive raw gap and zero acquisition-matched gap.
- Orthogonal control: AUC is at most `0.02`.
- Allocation is **not** advanced from this pilot unless a same-budget
  rank-specific gap remains after comparing rank-R and an unconstrained
  sequential baseline, with matched acquisition and history. If the two are
  indistinguishable, the result is classified as memory/online interference.
  On the compatible construction this null is structural: zero initialization
  keeps even the unconstrained trajectory at rank at most `R`, so its
  rank-specific gap is a diagnostic sanity check rather than a positive test.
- The full-observation cancellation control must fail the acquisition-matched
  addressable-gap gate even if its raw `total_online_gap` is positive.

The confirmatory run uses 20 fixed seeds for stochastic rotations and the exact
deterministic controls. A separate 100-seed sensitivity run may characterize
the observed right tail, but it does not replace the frozen 20-seed gate. A
positive result is reported with raw per-seed traces, median, mean, and
seed-wise uncertainty, not a best seed. Thresholds are frozen before the
remote confirmatory run.
Each output manifest carries `run_role`: `primary_frozen_20_seed` for the
confirmatory artifact or `sensitivity_only_100_seed` for the exploratory scan.

## Resource ledger

Every run records Python and NumPy versions, host, cwd, seed, dimensions,
requested and numerical rank, a hash of generated directions/targets, realized
direction-dot statistics, fixed `pinv` and rank tolerances, protocol/source/
test hashes, update count, replay state bytes, optimizer state bytes (zero for
the exact probes), wall time, and output file hashes. GPU resources are
intentionally unused in this exact phase; moving to PyTorch or an LLM is a
separate stage gated on this audit.
Output and generated-input hashes are provenance evidence, not a cross-host
equality gate: QR/BLAS or NumPy differences may change low-order floating-point
bytes even with the same seed. The remote result must be judged by the frozen
gate outcomes, raw per-seed traces, direction statistics, and exact matching
source/protocol/test hashes, with its environment recorded. A cross-host
`input_sha256` or output-hash difference must be reported, but does not fail
the confirmation when the structural traces and numerical metrics agree within
the recorded tolerance.

## Interpretation

Passing the compatible diagnostic establishes only that order and bounded
history can matter despite a sufficient rank. It does not validate FCRA. A
failure to find a rank-specific residual is a reason to remove or weaken the
capacity-allocation claim before spending GPU budget.
