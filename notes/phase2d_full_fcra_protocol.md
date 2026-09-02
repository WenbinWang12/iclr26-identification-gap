# Phase-2D — Full FCRA (Algorithm 1) confirmatory protocol (PRE-FREEZE)

**Status:** pre-registered before the citable primary run.
**Source under test:** `experiments/phase2d_full_fcra/phase2d_full_fcra.py`
**Source SHA-256 (frozen):** `5ebe60adb04c52cee00df5c6ebafe608e7f5290a3219da5706d316dbabdc5ce4`

This protocol is written and its SHA recorded BEFORE the primary run so that the
manifest's `protocol_sha256` and the ordered timestamps are real evidence, not
back-filled. The primary run is `outputs/primary/` and is the only citable run.

## Purpose

Close the review finding that the paper's method section (Algorithm 1: historical
trust region, four atom states, top-k allocation score, curvature-weighted SVD
consolidation, validation/backtracking) was strictly broader than the minimal
D2-C allocator that produced the positive rare-task-retention result. Here the
FULL algorithm is implemented and each mechanism is verified to actually fire, so
the method described in Section 4 and the method tested are the same object.

## What is (and is not) claimed

**Claimed (controlled linear mixture only):**
- The full FCRA algorithm runs end to end with every Algorithm-1 mechanism
  exercised on a stream that genuinely offers that mechanism work.
- On the orthogonal heavy-tail stream (Stream A, = the D2-C setting) it reproduces
  the rare-task-retention ordering at matched occupied rank R=4: full FCRA retains
  the rare task on all 40 seeds, strictly beats dense sequential on all 40, and is
  >= reservoir replay on >= 39/40 (NOT claimed uniform vs reservoir — the full
  algorithm differs slightly from the minimal D2-C allocator and reservoir is
  marginally higher on at most one seed).
- The rare-task recovery comes at a MEASURED COST to frequency-weighted average
  retention (reported per arm): this is worst-group / rare-task retention, not an
  average improvement.

**NOT claimed:** real LoRA / LLM / benchmark evidence; optimality of the rule;
nonlinear transfer (see D2-B, a negative result); matched auxiliary memory/compute
(only occupied rank is matched — full FCRA additionally stores a d×d curvature
matrix, a replay buffer, and solves damped systems per window; reported in the
resource ledger with `auxiliary_state_matched: false`).

## Design decision recorded up front (protection criterion)

Section 4's prose "protect atoms of high historical curvature energy" is the WRONG
criterion for retaining a rare task: a rare direction has LOW accumulated curvature
`w^T A w`. We instead protect the direction of highest curvature-AWARE score
`q = z^T (A + λI)^{-1} z`, i.e. the direction the stream will not re-teach and
whose loss is unrecoverable. This is the SAME score that drives acquisition and
eviction, so all three share one derivation. `sections/04_method.tex` is updated
to state this; the experiment implements the corrected criterion.

## Two streams (one cannot honestly exercise every mechanism)

- **Stream A — allocation (orthogonal K=8, π_k ∝ k^{-1.5}, d=16, R=4, T=600).**
  The headline. Exercises free-slot allocation, reusable-atom coefficient update,
  protection (p_max=3), and label-free eviction, and gives the rare-task ordering.
  It is conflict-free by construction, so the historical trust region is a
  deliberate NO-OP here (eps_hist large): forcing it to bind on a conflict-free
  stream would be an artifact. Consolidation likewise does not fire (nothing
  redundant). 40-seed sweep on seeds 2000–2039.

- **Stream B — mechanism stress (redundant + abrupt-recurrent, R=4, p_max=1,
  T=400).** Exists ONLY to exercise consolidation and the trust region on genuine
  redundancy/recurrence; NO retention claim is made on it. Two near-collinear task
  pairs (angle 0.5 rad) create genuine redundancy; an abrupt-recurrent schedule
  makes directions leave and return so accumulated curvature resists re-learning
  and the trust region genuinely binds. Consolidation merges a redundant active
  block by weighted truncated SVD, keeping the smallest rank that preserves ≥ 90%
  curvature energy (`consol_redundancy_tol = 0.10`); a non-redundant block is NOT
  force-merged (that would destroy real capacity) — the algorithm falls through to
  eviction. Consolidation, a capacity-RELEASING structural op, is granted a larger
  validation slack (`consol_slack = 0.20`) than incremental allocation
  (`val_budget = 0.05`), analogous to eviction; both slacks are reported.

## Gates (all must pass; `all_required_pass`)

Stream A: budget matched (≤R all arms); state_free / state_reusable / state_protected
/ eviction each fired; sequential loses rare (<0.2); full FCRA recovers rare
(≥ seq + 0.4); full FCRA keeps plasticity (common ≥ 0.15).
Stream B: budget matched; consolidation fired (≥1); state_recyclable fired (≥1);
trust region active (binding+blocked+backtracks ≥ 1).
Sweep (40 seeds): full FCRA retains rare on 40/40; > sequential on 40/40;
≥ reservoir on ≥ 39/40 (n-1, explicitly not claimed uniform).

## Run discipline

1. This protocol is finalized and its SHA-256 is recorded above.
2. `python experiments/phase2d_full_fcra/phase2d_full_fcra.py --output-dir
   experiments/phase2d_full_fcra/outputs/primary --run-role primary_frozen`.
3. The manifest records `source_sha256`, `protocol_sha256` (this file), numpy/
   platform, output hashes, and the resource ledger. `outputs/primary/` is the
   only citable run; probe runs are deleted.

CPU-only; no GPU; no network. Determinism via fixed `numpy.random.default_rng`
seeds; no `Date.now`/wall-clock in the computation.
