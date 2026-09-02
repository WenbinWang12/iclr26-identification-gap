# Phase-2c Minimal FCRA vs. Capacity Protocol

Date frozen: 2026-08-24 (revised after methodological audit)

Status: frozen candidate for the allocation-vs-capacity claim that motivates the
paper. Deterministic NumPy regression/falsification check on a controlled linear
mixture. Not neural-network, real-LoRA, or benchmark evidence. No remote
confirmation or cross-provider audit has been run yet.

Source SHA256 (`experiments/phase2c_fcra_minimal/phase2c_fcra_minimal.py`):
`b11e1ad9df3a56bbb76de22b06164ef0a06f992dfee2c1156bf10a10000f9547`

This protocol is frozen BEFORE the citable primary run. The primary run reads
this file, records its SHA into `manifest.protocol_sha256`, and is produced
after this file exists, so the manifest timestamp and SHA are genuine.

## Motivation and claim boundary

Huang et al. (arXiv:2605.29548) show rare-task retention arises from *reduced
interference*: once enough capacity (width) is devoted to common tasks, their
gradients weaken and stop overwriting rare-task features. Their remedy is scale.
Continual PEFT cannot buy scale (frozen backbone, one deployed adapter, fixed
total rank). This suite tests the resulting question on a controlled stream:
**when capacity cannot grow, can a fixed budget recover the reduced-interference
benefit purely through allocation over time?**

Authorized claim: on this controlled linear-mixture continual stream, a fixed
occupied-subspace budget with curvature-aware allocation (FCRA) recovers
retention of the empirically-rarest task. Under the pairwise-ordering metric,
FCRA matches-or-beats matched-budget reservoir replay on every seed and strictly
beats dense sequential updating on every seed; the curvature-whitened Grassmann
distance `d_t` is *consistent with* (not an independent predictor of) which arm
retains the rare task. NOT authorized: OLMo, real LLMs, real LoRA training,
benchmark performance, FCRA optimality, nonlinear transfer, or any claim that
reservoir replay *always* loses the rare task.

## Frozen configuration

`d=16`, `K=8` orthonormal task directions, fixed budget `R=4` (`R<K` forces
competition), at most `p_max=3` protected slots (>=1 active slot kept for
plasticity), frequencies `pi_k ~ k^{-1.5}`, stream length `T=600`, Grassmann
step `eta=0.5`, curvature damping `lambda=1e-3`, reservoir buffer `8`, residual
novelty threshold `0.30`. Fixed seeds: directions `20260529`, primary stream
`12345`. Robustness sweep: stream seeds `2000..2039` (task geometry held fixed;
only the arrival stream is reseeded). Retention threshold for "retained":
`s_rare >= 0.5`.

## Task-identity-free rare metric and task-identity-free allocation

The rare task is the **empirically least-frequent** direction in the realized
finite stream (`argmin` of task counts), not a nominal label. In a heavy-tailed
draw the nominal tail index need not be the rarest realized task; the method
sees no labels and must protect whatever is actually starved. Retention is
`s_k = ||U^T w_k||^2` for the occupied basis `U` (protected + active columns).

FCRA is task-identity-free: both the allocation score and the eviction rule use
only a direction's own curvature `v^T (A + lambda I)^{-1} v`, computed from the
stored unit vector. No task id, task count, or label is ever consulted. (An
earlier version scored eviction with `1/(count[task_id] + lambda)`; this was
mathematically equivalent for orthonormal tasks but leaked the task label and
has been removed.)

## Three arms (matched OCCUPIED RANK; auxiliary state NOT matched)

1. `sequential_dense` -- all `R` directions active; every window rotates them
   toward the arriving task. Frequent tasks dominate; the rare direction is
   overwritten.
2. `reservoir_replay` -- dense updates plus reservoir replay from a fixed buffer
   of `8` task ids. The rare task is under-represented in the buffer by
   frequency, so replay usually (not always) fails to rescue it.
3. `fcra` -- score a candidate direction by current gain per unit historical
   curvature, `z^T (A + lambda I)^{-1} z`. With `A = sum_k n_k w_k w_k^T` and
   orthonormal tasks the score is exactly `1/(n_k + lambda)`, highest for the
   least-frequent direction, so the rarest-seen direction wins a protected slot
   and later common updates cannot rotate it out. Protection is a consequence of
   the score alone, not hand-tuned to the rare task.

All arms are held at the same **occupied rank** `R=4`. Auxiliary state is NOT
matched and we do not claim it is: FCRA additionally maintains a `d x d`
curvature matrix `A`, up to `p_max` protected `d`-vectors, and solves one
`d x d` system per window. The manifest `resource_ledger` reports each arm's
auxiliary bytes explicitly (`auxiliary_state_matched = false`).

## Curvature-whitened expressivity distance `d_t`

`d_t` is the curvature-WHITENED principal-angle (chordal) distance from the
occupied subspace to the rare direction, `d_t = (sum_i sin^2 theta_i)^{1/2}`,
with angles measured after the theory's whitening embedding
`phi_t(.) = (A_amb + lambda I)^{1/2} .` (theory.tex). It is reported as a
geometry consistency check, not as independent evidence: `d_t` is computed from
the same occupied subspace as the retention, so we phrase the tie-in as "`d_t`
is consistent with retention", never "`d_t` predicts retention".

## Gates (all must pass)

- `D2C_budget_matched_all_arms`: every arm occupies at most `R` directions.
- `D2C_sequential_loses_rare`: sequential rare retention `< 0.2`.
- `D2C_fcra_recovers_rare`: fcra rare retention `>= sequential + 0.4`.
- `D2C_fcra_ge_reservoir_rare`: fcra rare retention `>= reservoir` (frozen seed).
- `D2C_fcra_keeps_plasticity`: fcra common-task retention `>= 0.2`.
- `D2C_dG_consistent_with_retention`: fcra whitened `d_t <= sequential d_t`.
- `D2C_sweep_fcra_ge_reservoir_all`: fcra `>=` reservoir on all `40` seeds.
- `D2C_sweep_fcra_gt_sequential_all`: fcra `>` sequential on all `40` seeds.

## Robustness (reported honestly)

Across the 40-seed sweep (`2000..2039`): FCRA matches-or-beats reservoir on
`40/40` seeds and strictly beats sequential on `40/40`; FCRA retains the rare
task on `40/40`, sequential on `0/40`, and **reservoir alone retains it on
`4/40`** (seeds `2001, 2003, 2033, 2038`). We do NOT claim reservoir always
loses the rare task; the claim is the pairwise ordering plus FCRA's per-seed
retention. Headline (frozen primary seed `12345`): fcra `s_rare=1.000`,
sequential and reservoir `s_rare=0.000`, all arms at occupied rank 4; fcra keeps
common-task retention `0.286`; whitened `d_t` is `~0` for fcra and `1.0` for
both losers.

## Resource ledger

`gpu_used = false`, `optimizer_state_bytes = 0`, `external_model_bytes = 0`,
`occupied_rank_matched = true`, `auxiliary_state_matched = false`. The manifest
records `per_arm_auxiliary_bytes` in float64 bytes: sequential stores an
`R`-column basis only; reservoir adds an `8`-entry int64 id buffer; FCRA adds a
`d x d` curvature matrix, `p_max` protected `d`-vectors, and a per-window
`(A + lambda I)^{-1}` solve. CPU-only, deterministic.
