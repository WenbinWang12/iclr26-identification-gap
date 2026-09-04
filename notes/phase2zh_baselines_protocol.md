# Phase-2Z-H (RUNBOOK 3D) — matched-budget published baselines: frozen protocol

Frozen 2026-09-04, **before any Phase-2Z-H run exists**. §07 lists "no
head-to-head against published continual-PEFT methods at matched budget" as a
limitation. This phase runs established baselines on the IDENTICAL harness so the
method has an external comparison. Implemented in `run_baselines.py`, tabled by
`analyze_baselines.py`.

## 1. What "matched budget / matched harness" means here

Every baseline shares, byte-for-byte with the method runs: the Order-4 stream and
its tuple order, the eligibility filter (`eligible_tasks`, `min_rarest=40`), the
`prepare_task_partitions` risk/audit splits, the seeds (1–3), the restricted-argmax
balanced-accuracy scorer on the `audit` split, and the trainable-parameter budget
— ONE rank-8 LoRA on q/v (the same 2.36M scalars as the paper's shared adapter and
as its 4×rank-2 grouped adapter). `test.json` is never opened. Only the TRAINING
rule differs, so an accuracy difference is a pure algorithm effect at fixed budget.

## 2. Baselines and their faithful mapping

* **seqft** — sequential fine-tuning of the single rank-8 LoRA, no anti-forgetting.
  The naive continual lower bound. Faithful and complete.
* **olora** — O-LoRA (Wang et al., EMNLP 2023). Mechanism reproduced: a penalty
  `λ·Σ_l ‖A_l^cur (A_l^prev)ᵀ‖_F²` pushing each task's LoRA-A rows orthogonal to
  the **frozen accumulated** A-rows of all previous tasks, added to the LM loss
  during training; previous A-rows are concatenated (not overwritten) after each
  task. Same single rank-8 adapter, so budget is matched. Deviations from the
  paper (documented, not hidden): (i) single shared adapter with row-accumulation
  rather than O-LoRA's per-task adapter concatenation — required to hold budget
  fixed; (ii) λ fixed at 0.5, not tuned per task. These make it an O-LoRA
  *mechanism* baseline at matched budget, not a reproduction of the paper's
  absolute numbers, and it is reported as such.
* **ewc** — EWC-style regularization CL: diagonal-Fisher-weighted quadratic anchor
  `0.5·λ·Σ F_i (θ_i − θ_i^prev)²` toward post-previous-task LoRA weights, Fisher
  estimated on ≤8 micro-batches of the task's own update split. Representative
  regularization-CL baseline at the same budget. λ fixed at 1.0.

## 3. NOT implemented here — scoped out, never stubbed as if run

* **E²-LoRA** (drift-energy rank pool, [[e2lora-average-only-and-ca-ablation]]):
  needs an output-drift-energy signal and a global rank pool reallocated across
  tasks — components outside this harness. Marked for the collaborator in RUNBOOK
  §3D. `run_baselines.py` will **refuse** `--method e2lora` (not a choice), so no
  empty/placeholder JSON can be mistaken for a run.
* **NSR / replay**: needs a replay buffer of stored examples, which the
  audit/risk-split discipline of this harness deliberately excludes. Same refusal.

If a reader sees no `baselines_e2lora_*.json` / `baselines_nsr_*.json`, that is
correct: they were not run, and `analyze_baselines.py` reports absent methods as
absent, never imputed.

## 4. Metrics and the honest head-to-head

`analyze_baselines.py` tables mean audit balanced accuracy per method over the
**common task set** (asserted identical across methods per seed; it refuses to
compare across differing task sets). It pulls the method's own cells from
`results_s{seed}.json`:
* `method_sh` = mean `R_sh` (shared, no offset) — the naive-budget baseline;
* `method_gp_off` = mean `R_gp_off` — grouping + per-scope offset, **but with
  oracle routing and refit offset, i.e. two UPPER BOUNDS**.

The fair, deployable head-to-head against these deployable baselines is the
**router+stored** cell from RUNBOOK 3A (`analyze_router.py`), NOT `method_gp_off`.
The analyzer prints this caveat in its own output so the ceiling number is never
quoted as the head-to-head result. This is the load-bearing honesty rule of 3D.

## 5. Frozen criteria (3 seeds; sign consistency)

* **B1 (sanity ordering).** `olora ≥ seqft` and `ewc ≥ seqft` in mean, every seed
  (anti-forgetting should not hurt vs naive on a matched budget). If a baseline
  underperforms naive seqft, its λ is mis-set — reported as such, not as evidence
  about the method.
* **B2 (headline placement, reported).** `method_gp_off − {seqft,olora,ewc}` and,
  once 3A exists, `method_route_stored − {seqft,olora,ewc}` reported per seed with
  sign. The deployable comparison (3A) is the one quoted in the paper; the ceiling
  comparison is labelled ceiling.
* **Training gate.** Median `seqft` R > 0.60 (the single-LoRA harness reaches a
  usable operating point); else the baseline harness is void.

## 6. Predeclared outcomes

1. **Deployable method beats all three baselines (via 3A number).** Report the
   matched-budget table with router+stored as the method column; strongest
   external-validity result.
2. **Deployable method beats seqft/ewc but not olora.** Report honestly: O-LoRA's
   orthogonality is competitive at this budget; the method's advantage is
   {offset / grouping} specifically, quantified per component. No cherry-picking.
3. **Deployable method does not beat the best baseline.** Then the contribution is
   the two-constraint *analysis* and the offset (a zero-rank, deploy-cheap add-on
   composable with ANY of these baselines), not a new SOTA allocator. Reframe the
   claim to that; do NOT quote the oracle `method_gp_off` to manufacture a win.
4. **A baseline underperforms seqft (B1 fails).** Its regularizer strength is
   mis-set; rerun that baseline with a corrected λ (a training-side fix, allowed —
   it is not tuning the *method* to a target), or report it as λ-sensitive with the
   value used. Never drop it silently.

## 7. Anti-artifact checks

* Single rank-8 LoRA for every baseline — budget re-logged; identical to method's
  shared adapter (`build_single_lora` vs `build_adapters`'s `shared` cfg).
* O-LoRA previous basis is **detached/frozen** (`snapshot_A` clones with
  `.detach()`); the penalty cannot leak gradients into past-task directions.
* EWC anchor + Fisher captured AFTER each task from that task's own update split;
  never audit/test.
* Same eligible task set as the method (same `eligible_tasks`); analyzer refuses
  mismatched task sets.
* `--method e2lora`/`nsr` are not valid choices → impossible to emit a fake run.

## 8. Compute

Per method: 3 seeds, one sequential pass over the stream (~10–12 h wall each on
T5-large, 80 GB). BABEL `-p preempt --qos=preempt_qos -A bapoczos --gres=gpu:1`,
Blackwell refused (exit 75). Method selected at submit via `--export=ALL,METHOD=…`.
λ values fixed here (olora 0.5, ewc 1.0), declared before running.
