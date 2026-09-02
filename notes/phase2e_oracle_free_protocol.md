# Phase-2E oracle-free rebuild (v2 + v3) — record

**Status:** post-audit rebuild of the RETRACTED `phase2e_real_lora.py`. This note
records the corrected design, the frozen source SHAs, and the pre-registered
stop-rule together with its (negative) outcome. Written after the runs because it
documents a correction, not a fresh pre-registration; the stop-rule threshold,
however, was fixed BEFORE the soft-protection retest (see below).

## Sources (frozen)

- `experiments/phase2e_real_lora/phase2e_v2_oracle_free.py`
  SHA-256 `0d67351b368ed0e2420abedeaf6b8b55b5f7cb4c8e83932ca6ccc2cd4c225951`
- `experiments/phase2e_real_lora/phase2e_v3_pareto.py`
  SHA-256 `4ea20a92362efc9a6d95c9168539fddef57e846f5ec9a3e5b168b1287aef6d38`

## What was wrong (2026-08-26 audit of the retracted run)

1. Direction ORACLE: FCRA read `v=V[:,k]` for residual/score/atom-init.
2. Analytic "held-out" retention `1-||Mv-u||^2`, not an independent batch.
3. Replay buffer (262 KB) held by FCRA but omitted from the ledger; the
   "confound runs in FCRA's favor" claim was false.

## Corrected design

- **Oracle-free by construction.** Every trainer (`run_dense`, `run_cvar_replay`,
  `run_fcra`) takes only `(cfg, batches, seed)`; `batches` is a list of `(X,Y)`.
  No `V`, `U`, or `k` is in scope. FCRA discovers its candidate direction as the
  top singular vector of the residual LoRA gradient `dM - (dM Q)Q^T` (occupied
  span projected out). A label-blindness gate hashes the trajectory and confirms it
  is a function of `(X,Y)` only.
- **Honest metric.** Retention is an independent held-out batch (separate RNG),
  scale-normalized `1 - ||MX^T-Y||^2/||Y||^2`, evaluated by the driver.
- **Full ledger.** Replay buffer counted for reservoir, cvar_replay AND fcra; fcra
  additionally reports its curvature + protected-basis bytes ON TOP (so fcra holds
  MORE memory, not less).

## Two streams

- **v2 orthogonal R<K** — retained only as a ZERO-SUM negative control (protecting
  a rare direction forces evicting a more frequent one; no method can raise both
  rare and freq-weighted). Result: the retracted `+1.00` rare headline was an
  ORACLE ARTIFACT — oracle-free rare retention collapses to `-0.04`, sweep 40/40 to
  2/40.
- **v3 nonorthogonal / redundant / mixed** — the honest arena. Nonorthogonal
  teacher with a near-redundant component pair; each window mixes 2–4 latent
  components. Offline rank-R mean–tail Pareto frontier computed by scalarizing
  `(1-rho)*freq-mean + rho*smooth-worst` over held-out component losses and sweeping
  `rho`. Each online arm's `(freq-weighted, worst-group)` point and its distance to
  the frontier are reported.

## Pre-registered stop-rule (threshold fixed before the soft-protect retest)

The decision arm (redesigned soft-protection FCRA) must beat CVaR-replay-only on
worst-group retention (margin +0.05) on **>= 70% (28/40)** of sweep seeds to claim
the allocator adds value; otherwise collapse to risk-aware replay and make NO
positive method claim. `gamma_prot=5.0` fixed; not tuned after seeing results.

## Outcome (NEGATIVE, reported honestly)

- Hard protection FREEZES protected atoms → on mixed windows it starves the
  trainable rank and diverges (worst-group `-2.03`, sweep 9/40).
- Soft protection (protected atoms stay trainable, anchored by an L2 drift penalty)
  FIXES that: worst-group `-0.40`, and it is the CLOSEST arm to the offline frontier
  (`0.42` vs sequential `0.48`). A clean result on protection *design*.
- But soft-protect beats CVaR-replay on the tail on only **11/40** seeds — short of
  the 28/40 bar. **Verdict: `allocator_no_value`.** No positive method claim; the
  paper reports diagnosis + Pareto frontier + this honest negative. Method
  development stopped here (tuning to rescue the number would re-introduce the
  rigging the audit removed).

CPU-only, deterministic (`numpy.random.default_rng`), no network, no GPU. Boundary:
controlled factored low-rank LINEAR proxy — NOT LLM / benchmark / real-NLP /
nonlinear (see D2-B, negative).
