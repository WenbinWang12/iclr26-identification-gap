# Phase-2Y — Composite Method: PSR + SCG + Adaptive Rank

**Status:** DESIGN DRAFT, contingent on Phase-2X verdict

## 0. Motivation: why composite after three failures

Phase-2K QOC, Phase-2U SCG, Phase-2W rank all failed as standalone methods, but each showed **partial mechanism success**:

| method | what failed | what worked |
| --- | --- | --- |
| QOC (2K) | criterion 2 (m-monotonicity) 0.0000 | criterion 1 (+2.34pp) and 3 (ρ=+0.646); offset repairs 78% of mean forgetting |
| SCG (2U) | U1 (worst −2.34pp, threshold −1.0pp) | U5 mechanism (ρ=+0.72, 8/8 tasks); mean +3.57pp on held-out |
| rank (2W) | 32× flat (±1.30pp = noise floor) | — |

**Post-hoc analysis** (Phase-2K converged, 2L §0):
- Offsets repair **45%** of large forgetting (>8pp stratum)
- **55%** is representation-level: MNLI +17.5pp residual, RTE +11.5, BoolQA +6.9
- QOC天花板 m=4 (0.7204) already exceeds per-task oracle (0.7146)

**The binding constraint is representation-level.** Offsets alone cannot reach it. Rank scaling alone is too noisy. But:
- PSR-LoRA (Phase-2F) showed rare-source coverage can target deep forgetting
- SCG's mechanism (|mean(s)| correlates with BC gain) works within-task
- Adaptive rank allocation (conditioned on forgetting signal) was never tested

**Hypothesis:** A three-stage pipeline can break QOC's ceiling by addressing representation-level forgetting that offsets cannot reach.

## 1. The composite method

### Stage 1: PSR source selection (rare-class coverage)
From Phase-2F honest rerun + remedy (memory: phase2f-honest-rerun-and-remedy.md):
- **Root cause of 2F v1 failure:** rare-source under-coverage
- **Remedy that survived held-out:** coverage buffer (cover 4+12 minimum per rare class)
- All 3 criteria PASSED on new held-out seeds

**Apply:** At each task boundary, select rehearsal sources using PSR with coverage-buffer remedy.
- Stored: ~16 examples/task × 15 tasks = 240 examples total
- Budget: 0.01% of 2.36M trainable parameters (negligible)

### Stage 2: Adaptive rank allocation (forgetting-conditional)
**New hypothesis** (never tested):
- Phase-2W showed fixed-rank scaling is flat
- But 2K/2L showed forgetting is **task-heterogeneous**: MNLI/RTE/BoolQA have >11pp residual, others <2pp
- **Allocate rank budget to tasks with measured deep forgetting**

**Method:**
- Budget: r=8 × 15 tasks = 120 total rank units (same as baseline)
- At task `t`, measure forgetting on previous tasks using stored PSR examples
- Allocate rank proportional to residual forgetting (after best offset):
  ```
  r_t = r_base + λ * (forgetting_t - median_forgetting) / scale
  ```
- Constraint: Σr_t = 120 (redistribute, don't grow)

**Key difference from 2W:** rank is **conditioned on signal**, not uniform scaling.

### Stage 3: SCG offset correction (query-time)
From Phase-2U (memory: phase2u-scg-and-heldout-discipline.md):
- U5 mechanism: ρ=+0.72 between |mean(s)| and BC gain
- U1 failed on threshold transfer, not mechanism

**Apply:** At inference, apply SCG with **task-adaptive threshold**:
- τ_m per task, selected on PSR examples (validation set)
- If |mean(s)| >= τ_m: apply BC offset
- Else: use representation output (Stage 2)

**Why this helps:** SCG handles distributional drift (shallow forgetting); Stage 2 handles deep forgetting.

## 2. Why this might break the ceiling

QOC's ceiling (m=4 > oracle) is because:
1. Offsets repair only 45% of large forgetting
2. The residual is representation-level

Phase-2Y addresses both:
1. **Stage 2** (adaptive rank) directly targets representation-level forgetting
2. **Stage 1** (PSR) provides training signal for tasks with deep forgetting
3. **Stage 3** (SCG) handles shallow forgetting that Stage 2 misses

**Failure modes to test:**
- Rank allocation noise > signal (same as 2W)
- PSR coverage insufficient for rank training
- Three-stage complexity introduces new failure modes

## 3. Experimental design (draft)

**Training:**
- Base: T5-large, LoRA r=8 baseline
- Order-4 stream, 15 tasks
- Stage 1: PSR coverage buffer (4+12)
- Stage 2: Adaptive rank, budget-neutral (Σr=120)
- Stage 3: SCG with task-adaptive τ_m

**Metrics:**
- Primary: R_composite vs R_orc (multi-task scope only, 7 scopes)
- Mechanism: 
  - Y1: rank allocation correlates with residual forgetting (ρ > 0.5)
  - Y2: composite beats QOC ceiling (R > 0.7204)
  - Y3: Stage 2 alone vs Stage 1+2 vs full composite (ablation)

**Seeds:** 3 development, 3 held-out (following Phase-2U discipline)

**Criteria (draft, to be tightened):**
- Y1 (mechanism): ρ(allocated_rank, residual_forgetting) > +0.5, median across seeds
- Y2 (ceiling break): R_composite − R_qoc_ceiling CI excludes 0 on positive side
- Y3 (no-harm): R_composite ≥ R_baseline on 12/12 scorable tasks

**Predeclared outcomes:**
1. All pass → composite method works, write up
2. Y1 fails → rank allocation is noise, stop pursuing adaptive rank
3. Y2 fails → ceiling is model capacity, not method; negative result
4. Y3 fails → composite is harmful, report failure

## 4. Dependencies and timeline

**Depends on:**
- Phase-2X verdict: if BPO survives, identification-gap thesis needs revision first
- PSR-LoRA coverage buffer code (already exists, Phase-2F)
- Adaptive rank LoRA implementation (needs new code)

**Estimated GPU time:**
- 3 dev seeds: ~3.6h (1.2h each, converged settings)
- 3 held-out: ~3.6h
- Total: ~7.2h on Babel

**Alternative if Phase-2Y fails:**
- Revert to "measurement paper": Δ_id exists (+1.10pp), no method works, ceiling is model capacity
- Or explore different problem: constant-scale control (Amendment 2, memory context)

## 5. Open questions before implementation

1. **Adaptive rank mechanics:** How to actually implement budget-neutral reallocation in PEFT?
   - LoRA r is per-layer, not per-task
   - May need task-conditional routing or parameter masking
   
2. **PSR examples as validation set:** 240 examples enough to select 15 task-specific τ_m?
   - Might need held-out split of PSR examples
   
3. **Three-stage pipeline complexity:** Are we overfitting to development failures?
   - Ablation (Stage 2 alone, Stage 1+2, full composite) is critical

**Decision point:** Implement Phase-2Y only if:
- Phase-2X judges paper SAFE (BPO collapses)
- User confirms pursuing composite method
- Adaptive rank implementation is feasible

---

**Next step:** Await Phase-2X verdict, then decide:
- If BPO collapses → implement Phase-2Y
- If BPO survives → revise identification-gap thesis or pivot to different problem
