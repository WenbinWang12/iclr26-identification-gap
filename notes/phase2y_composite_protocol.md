# Phase-2Y — Composite Method (PSR + Adaptive Rank + SCG): Protocol

**Status:** DESIGN READY, awaiting Phase-2X verdict to proceed

Frozen 2026-09-02, **before any Phase-2Y run exists**, contingent on Phase-2X verdict judging the paper SAFE (BPO collapses under prior skew).

## 0. Why a composite method after three standalone failures

Phase-2K QOC, Phase-2U SCG, and Phase-2W rank all failed their primary criteria as standalone methods, but **each showed partial mechanism success**:

| Phase | Method | Primary Failure | Mechanism Success |
| --- | --- | --- | --- |
| 2K | QOC (offset codebook) | Criterion 2: m-monotonicity 0.0000 CI [0, 0] | Criterion 1: +2.34pp CI [+0.78, +3.91]; Criterion 3: ρ=+0.646; offsets repair 78% of mean forgetting |
| 2U | SCG (self-confidence gate) | U1: worst −2.34pp (threshold −1.0pp) | U5 mechanism: ρ=+0.72 between \|mean(s)\| and BC gain (8/8 tasks); mean +3.57pp on held-out |
| 2W | Rank scaling | 32× flat: ±1.30pp = noise floor | — (uniform scaling failed) |

**Post-hoc decomposition** (Phase-2K converged → Phase-2L §0):

Offset repairs **only 45%** of large forgetting (>8pp stratum):

| Stratum | n | Forgetting | Offset Repair | Repair Fraction | Residual |
| --- | --- | --- | --- | --- | --- |
| < 2 pp | 92 | −0.0306 | +0.0364 | — | −0.0670 |
| 2–8 pp | 57 | +0.0518 | +0.0572 | 1.10 | −0.0053 |
| **> 8 pp** | 94 | **+0.2073** | +0.0926 | **0.45** | **+0.1148** |

**Key tasks with high residual (>8pp):**
- MNLI: +17.5pp residual (offset cannot reach)
- RTE: +11.5pp residual
- BoolQA: +6.9pp residual

**QOC's ceiling:** m=4 codebook (0.7204) already **exceeds** per-task oracle (0.7146), fitted on the same 64/class risk split. The binding constraint is **representation-level forgetting**, not shallow forgetting.

**Central hypothesis:** A three-stage composite can break QOC's ceiling by addressing:
1. **Stage 1 (PSR):** Rehearsal source selection targeting rare-class coverage
2. **Stage 2 (Adaptive Rank):** Representation-level capacity allocation conditional on measured deep forgetting
3. **Stage 3 (SCG):** Query-time offset correction with self-confidence gating

Unlike standalone methods, this composite **matches mechanism to failure mode**:
- PSR provides training signal for tasks with deep forgetting
- Adaptive rank addresses representation-level constraints (the 55% offsets miss)
- SCG handles distributional drift (shallow forgetting) at query time

## 1. Stage 1: PSR Source Selection (Rare-Class Coverage)

From Phase-2F honest rerun + remedy (memory: `phase2f-honest-rerun-and-remedy.md`):

**Root cause of Phase-2F v1 failure:** rare-source under-coverage led to cluster-level collapse on development seeds.

**Remedy that survived held-out validation:** Coverage buffer guarantees minimum representation of rare classes:
- `cover_first = 4`: minimum examples from first rare class
- `cover_rest = 12`: minimum examples from remaining rare classes
- All 3 Phase-2F criteria (a, b, c) **PASSED** on new held-out seeds

**Application in Phase-2Y:**
- At each task boundary, select rehearsal sources using PSR with coverage buffer (4+12)
- Stored: ~16 examples/task × 15 tasks = **240 examples total**
- Budget: 240 examples ≈ 0.01% of 2.36M trainable parameters (negligible)
- Use stored examples for:
  1. Measuring residual forgetting (input to Stage 2)
  2. Validation set for task-adaptive SCG thresholds (Stage 3)

**Wiring:** Reuse `experiments/phase2f_psr_lora/psr_lora.py` with coverage buffer enabled.

## 2. Stage 2: Adaptive Rank Allocation (Forgetting-Conditional)

**Novel hypothesis** (never tested in Phase-2W):

Phase-2W showed **uniform** rank scaling (1× to 32×) is flat (±1.30pp = noise floor), but forgetting is **task-heterogeneous** (MNLI +17.5pp residual vs Yelp 0.0pp). **Allocate rank budget to tasks with measured deep forgetting**, not uniformly.

**Method:**
- Total budget: 120 rank units (same as baseline: 15 tasks × r=8)
- At task `t`, measure residual forgetting on all previous tasks using stored PSR examples:
  ```
  residual_t = forgetting_t - offset_repair_t
  ```
- Allocate rank proportional to residual forgetting:
  ```python
  r_t = base_rank + sensitivity * (residual_t - median_residual) / scale
  ```
- Constraint: `Σr_t = 120` (redistribute existing budget, never grow it)
- Base rank: `r_min = 4` per task (minimum capacity floor)
- Sensitivity: `λ ∈ [0, 1]` (hyperparameter; 0=uniform, 1=fully proportional)

**Key difference from Phase-2W:** Rank is **conditioned on signal** (measured forgetting), not a blind uniform scaling.

**Implementation challenge:**
- PEFT LoRA `r` is per-module (q, v), not per-task
- Options:
  1. **Task-conditional masking:** Train full r=16 modules, mask parameters per task
  2. **Progressive rank growth:** Start r=4, grow to allocated r_t at task boundary
  3. **Mixture of ranks:** Low-rank base (r=4 shared) + task-specific high-rank adapter

**For Phase-2Y v1:** Use task-conditional masking (simplest to implement with existing PEFT).

**Sensitivity sweep:** Development seeds test λ ∈ {0.3, 0.5, 0.8}; select on Y1 mechanism criterion.

## 3. Stage 3: SCG Offset Correction (Query-Time)

From Phase-2U (memory: `phase2u-scg-and-heldout-discipline.md`):

**Mechanism that worked:** U5 passed with ρ=+0.72 between `|mean(s)|` and BC gain (8/8 tasks). The **size of BC's proposed correction** predicts both oracle headroom and BC's realized gain.

**What failed:** U1 (threshold transfer). `τ_m = 1.4` chosen on dev seeds 1–3 did not transfer to held-out seeds (worst −2.34pp).

**Phase-2Y fix:** Task-adaptive thresholds instead of global τ_m.

**Method:**
- At task `t`, fit `τ_m^t` on PSR examples (validation set) by grid search:
  - Sweep τ ∈ [0.8, 1.0, 1.2, 1.4, 1.6, 2.0]
  - Select τ that maximizes mean gain while keeping worst ≥ −1.0pp on PSR examples
- At inference on task `t`:
  ```python
  s = Z[:, 1] - Z[:, 0]  # decision statistic (binary tasks)
  m = mean(s)
  if |m| >= τ_m^t:
      apply offset b = (0, -m)
  else:
      return Stage 2 output (no offset)
  ```

**Stored state:** 15 scalars (one τ_m per task) + 0 floats for BC itself = **15 floats**.

**Why task-adaptive helps:** Phase-2U's global τ_m failed because different tasks have different signal-to-noise ratios. Task-specific thresholds fitted on PSR examples adapt to that heterogeneity.

## 4. Full Pipeline

**Training (per task `t`):**
1. **PSR Stage 1:** Select ~16 rehearsal examples with coverage buffer (4+12)
2. **Adaptive Rank Stage 2:**
   - Measure residual forgetting on tasks 1...(t-1) using PSR examples
   - Allocate rank r_t based on forgetting profile
   - Train LoRA with allocated rank on task t's training data + PSR rehearsal
3. **SCG Stage 3:**
   - Fit τ_m^t on PSR examples (validation split)
   - Store threshold

**Inference (query batch on task `t`):**
1. Forward pass through Stage 2 (LoRA with allocated rank) → logits Z
2. Compute decision statistic s and |mean(s)|
3. If |mean(s)| >= τ_m^t: apply BC offset (Stage 3)
4. Else: return Stage 2 output directly

**Stored state total:**
- PSR examples: 240 examples ≈ 0.01% of parameters
- SCG thresholds: 15 floats
- Rank allocation policy: 15 integers (metadata, not parameters)

## 5. Metrics and baselines

Scored on `audit` split after task 15, restricted-argmax balanced accuracy, mean over 12 scorable tasks (singleton scopes excluded per singleton-defect correction).

**Primary arms:**
- `R_composite`: Full 3-stage pipeline (PSR + Adaptive Rank + SCG)
- `R_baseline`: LoRA r=8, no CL method, no offset
- `R_orc`: Per-task oracle offset (ceiling, needs task ID)
- `R_qoc_ceiling`: m=4 codebook from Phase-2K converged (0.7204)

**Ablations (mechanism decomposition):**
- `R_stage1`: PSR only (no adaptive rank, no SCG)
- `R_stage1+2`: PSR + Adaptive Rank (no SCG)
- `R_stage2`: Adaptive Rank only (no PSR, no SCG)
- `R_stage2+3`: Adaptive Rank + SCG (no PSR)

All from the **same run** where possible; ablations that change training (Stage 1 on/off, Stage 2 rank) require separate runs.

## 6. Frozen criteria

3 development seeds (1, 2, 3), 3 held-out seeds (11, 12, 13).

Cluster bootstrap CI: tasks as clusters, 10,000 draws, mean statistic, seed 0.

**Development phase (seeds 1–3):**
- Select sensitivity λ ∈ {0.3, 0.5, 0.8} on Y1 mechanism
- Fit task-adaptive τ_m per task
- Record all ablations

**Held-out phase (seeds 11–13):**

- **Y1 (mechanism: rank allocation tracks forgetting).** Spearman ρ between allocated rank `r_t` and measured residual forgetting across tasks. Median across 3 held-out seeds must exceed **+0.50**. If this fails, adaptive rank is noise and the method reduces to PSR+SCG.

- **Y2 (ceiling break: composite exceeds QOC).** `R_composite − R_qoc_ceiling` cluster-bootstrap CI on the mean must exclude 0 on the positive side. QOC's ceiling (0.7204) is the target to beat. If this fails, representation-level capacity is saturated.

- **Y3 (no-harm: composite beats baseline on most tasks).** `R_composite ≥ R_baseline` on at least **10 of 12** scorable tasks, point estimate per task. If this fails, the composite is harmful and should not be deployed.

- **Y4 (non-vacuous: composite applies adaptively).** Fraction of tasks with allocated rank r_t differing from median by >2 must be ≥ 0.30. If this fails, adaptive allocation collapsed to near-uniform and Stage 2 is a no-op.

- **Y5 (SCG mechanism survives in composite).** Within each task with ≥6 held-out batches, Spearman ρ between `|mean(s)|` and Stage 3's gain over Stage 2 output. Median across qualifying tasks must exceed **+0.25**. If this fails, Stage 3 lost its mechanism in the composite context.

**Training gate (unconditional):** Median post-training `R_baseline` > 0.60 (sanity check).

## 7. Predeclared outcomes

1. **Y1, Y2, Y3, Y4, Y5 all pass** → All three stages contribute mechanistically; composite breaks QOC's ceiling by addressing representation-level forgetting. **Write up as positive result.** This is the target outcome for publication.

2. **Y1, Y2, Y3 pass; Y4 or Y5 fails** → Composite works but one stage's mechanism is unclear. Report as a working method with partial explanation. **Still publishable** as a positive result if Y2 (ceiling break) is robust.

3. **Y2 fails (ceiling not broken)** → QOC's ceiling (0.7204) is model capacity limit, not method limit. Representation-level forgetting on MNLI/RTE/BoolQA is **irreparable** under the LoRA r=8 budget. Report as **negative result**: the binding constraint is model capacity, and adaptive rank does not overcome it. Pivot to "measurement paper" or Amendment 2 (constant-scale control).

4. **Y1 fails (rank allocation is noise)** → Adaptive rank collapsed to noise, same as Phase-2W. Composite reduces to PSR+SCG. Re-score with uniform rank and report as a two-stage method if it still works.

5. **Y3 fails (composite is harmful)** → Three-stage pipeline introduces new failure modes. Report failure and stop pursuing composite. Fall back to best standalone method (if any).

Anything outside this list is recorded as a protocol gap in an amendment.

## 8. Anti-artefact checks

- **Rank budget invariant:** `Σr_t = 120` verified at each task boundary
- **PSR coverage buffer enforced:** Minimum 4+12 rare-class examples per task
- **SCG label-freeness:** Stage 3 reads only logits Z, never labels
- **Singleton scope exclusion:** Multi-task scopes only (7 of 12), per singleton-defect correction
- **Development vs held-out separation:** Sensitivity λ and τ_m fitted on seeds 1–3; seeds 11–13 held out until criteria are frozen
- **No test.json contamination:** Official test split never read
- **Ablation consistency:** Stage 1+2 must match Stage 1+2+3 when Stage 3 abstains (verify on PSR examples)

## 9. Implementation notes

**Adaptive rank LoRA wiring:**
- Implement task-conditional parameter masking in `peft` LoraConfig
- Train with max rank r_max=16 (to accommodate MNLI's r=15 at λ=0.8)
- At task t, mask parameters beyond allocated r_t
- Verify mask gradients are zeroed

**PSR integration:**
- Reuse `experiments/phase2f_psr_lora/psr_lora.py`
- Enable coverage buffer: `cover_first=4, cover_rest=12`
- Store examples in `/runs/phase2y_composite/psr_store_s{seed}.pkl`

**SCG integration:**
- Reuse `experiments/phase2u_scg/` offset logic
- Fit τ_m per task on PSR validation split (80/20 split of 16 examples)
- Store thresholds in `/runs/phase2y_composite/scg_thresholds_s{seed}.json`

**Development run structure:**
```
for seed in 1, 2, 3:
    for sensitivity in 0.3, 0.5, 0.8:
        run_phase2y(seed, sensitivity, out=f"runs/phase2y_dev_s{seed}_lam{sensitivity}")
```

**Held-out run structure:**
```
# Use best sensitivity from development
best_lambda = select_on_Y1_mechanism(dev_results)
for seed in 11, 12, 13:
    run_phase2y(seed, best_lambda, out=f"runs/phase2y_heldout_s{seed}")
```

## 10. GPU budget and timeline

**Development (9 runs: 3 seeds × 3 sensitivities):**
- Estimated time per run: ~2.0h (longer than baseline due to rank switching overhead)
- Total: 9 × 2.0h = **18h**
- Parallelizable: 3 concurrent jobs × 6h = 6h wall-clock on Babel

**Held-out (3 runs: 3 seeds × 1 best sensitivity):**
- Total: 3 × 2.0h = **6h**
- Parallelizable: 3 concurrent jobs × 2h = 2h wall-clock

**Grand total: 24h GPU, 8h wall-clock** (with 3-way parallelism on Babel)

## 11. Contingencies

**If Phase-2X judges paper AT RISK (BPO survives skew):**
- Phase-2Y is **deferred**
- Priority shifts to revising identification-gap thesis or explaining BPO anomaly
- Composite method may still be relevant under revised framing

**If Phase-2Y outcomes 3, 4, or 5 occur (ceiling not broken or mechanism fails):**
- Amendment 2: Constant-scale control arm (memory context mentions this)
- Alternative framing: "measurement paper" documenting Δ_id without a working method
- Pivot to different problem: multi-task learning, not continual learning

**If adaptive rank LoRA wiring proves infeasible:**
- Fall back to two-stage method: PSR + SCG (no adaptive rank)
- Or explore mixture-of-ranks architecture (requires more design)

---

**Approval gate:** This protocol is ready to execute **immediately upon Phase-2X verdict = PAPER SAFE**. User approval required to launch development runs.

Next step: Await Phase-2X completion (~1h), analyze verdict, then launch Phase-2Y if paper is safe.
