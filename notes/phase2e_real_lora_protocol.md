# Phase-2E — Real factored LoRA trained by gradient descent (PRE-FREEZE)

**Status:** pre-registered before the citable primary run.
**Source under test:** `experiments/phase2e_real_lora/phase2e_real_lora.py`
**Source SHA-256 (frozen):** `08f752c2388b559a539619bdc5b7daba10e2590ee9dc8d7cc2c431d10f85c2eb`

Written and SHA-recorded BEFORE the primary run so the manifest `protocol_sha256`
and ordered timestamps are real. The citable run is `outputs/primary/`.

## Purpose

Answer the review point that D2-C/D2-D used a Grassmann-subspace abstraction
(retention = projection of a task direction onto the occupied subspace) rather
than a real low-rank adapter trained by gradient descent. Here the adapter is a
genuine factored LoRA `Delta W = B A` (`B in R^{d x R}`, `A in R^{R x d}`, `R<K`)
trained by real deterministic gradient descent on a real mean-squared regression
loss, and retention is the adapter's ACTUAL held-out loss per task, not a
geometric surrogate. This is also the setting on which the full metric family and
the resource-matched ablations are reported.

## Controlled teacher (synthetic; boundary stated)

`K` rank-one teacher maps `T_k = u_k v_k^T` with orthonormal input directions
`v_k` and output directions `u_k`. Task-`k` inputs concentrate along `v_k` plus
small isotropic noise, so the accumulated input Gram `sum_k n_k v_k v_k^T` is
exactly the historical input curvature the theory uses. Retention of task `k` is
`1 - ||M v_k - u_k||^2` for the deployed adapter `M = B A` (1 = reproduced, 0 =
adapter contributes nothing along `v_k`; can go negative if the adapter actively
harms that direction). The stream is clocked and task-identity-free: one window =
one task's batch, tasks arrive at heavy-tailed frequency `pi_k ~ k^{-1.5}`, and no
task label is ever exposed. FCRA's score and eviction use only a direction's own
curvature `v^T (A_in + lambda I)^{-1} v`.

## Arms (all train the SAME factored-LoRA class by the SAME GD rule)

- `sequential_dense` — all `R` ranks active every window (standard continual LoRA).
- `reservoir_replay` — plus a fixed reservoir buffer of past batches.
- `fair_reservoir` — reservoir sized to MATCH FCRA's auxiliary bytes. NOTE: FCRA's
  auxiliary state (`d*d` curvature + `p_max*d` protected basis = 8,960 bytes) is
  in fact SMALLER than a single replay window, so the byte-match rounds down and
  `fair_reservoir` equals `reservoir_replay` here. This is reported honestly and
  strengthens the result: the memory confound runs in FCRA's favor — a reservoir
  with ~29x more auxiliary memory (262,144 bytes) still fails to retain the rare
  task.
- `fcra` — the full allocator on rank-one atoms (score / protect / evict /
  consolidate); protected atoms are frozen; protection maintains the
  top-`p_max`-by-score set with demotion, so a late high-score (rare) atom can
  displace a lower-score protected atom.
- Ablations isolating the mechanism: `fcra_random_score` (curvature score replaced
  by a stable per-atom random key everywhere it is used — candidate, eviction
  target, and protection selection), `fcra_no_protect`, `fcra_no_consolidate`.

## Metric family (reported per arm; no cherry-pick)

rare-task retention, common-task mean retention, frequency-weighted average
retention, worst-group (worst SEEN task) retention, and backward transfer (BWT).
"Rare" and "worst" range only over tasks that actually appeared (count >= 1): a
count-0 task was never in the stream, so scoring it is a definitional artifact,
not forgetting. The expected and reported story: FCRA wins rare and worst-group
but LOSES frequency-weighted average — reallocation of a fixed budget, not a
uniform reduction in forgetting.

## Gates (all must pass)

Mechanism firing: FCRA allocates, protects, and evicts at least once.
Real-loss result: sequential loses rare (`< 0.3`); FCRA recovers rare
(`>= seq + 0.4`); FCRA beats sequential worst-group (`>= seq + 0.15`).
Honest cost: FCRA freq-weighted `<=` reservoir (reallocation, not free lunch).
Confound controls: `fair_reservoir` still loses rare (`<= fcra - 0.4`);
`fcra_random_score` loses rare (`<= fcra - 0.3`) — the curvature score is the cause.
Sweep (40 seeds, 3000-3039): FCRA retains rare on 40/40 and `>` sequential on 40/40.

## Run discipline

1. Protocol finalized; SHA-256 recorded above.
2. `python experiments/phase2e_real_lora/phase2e_real_lora.py --output-dir
   experiments/phase2e_real_lora/outputs/primary --run-role primary_frozen`.
3. Manifest records source/protocol SHA, numpy/platform, output hashes, resource
   ledger. `outputs/primary/` is the only citable run; probes are deleted.

CPU-only; no GPU; no network. Determinism via fixed `numpy.random.default_rng`
seeds; no wall-clock in the computation. Boundary:
`controlled_regression_real_lora_only` — NOT LLM / benchmark / real-NLP evidence,
no optimality, no nonlinear transfer (see D2-B, negative).
