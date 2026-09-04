# Phase-2Z-G (RUNBOOK 3C) — K_S≥3 multi-task scope: frozen protocol

Frozen 2026-09-04, **before any Phase-2Z-G run exists**. The per-scope offset
(Prop 3) is only *proved* closed for binary scopes (d = K−1 = 1, d=1 branch);
for d ≥ 2 the worst-case m-quantization radius q_m is the open object. §07 lists
"non-binary scope offset unproven" as an open item. This phase tests the method
in exactly that regime. Implemented in `run_scopes.py`, judged by
`analyze_scopes.py`.

## 1. Why the paper never exercises d ≥ 2

Scopes are keyed by verbalizer. Among eligible Order-4 tasks the K ≥ 3 tasks
(MNLI 3-way, AGNews 4-way, DBpedia 14-way, Yahoo 10-way) each have a **unique**
label set, so each is a **singleton** scope: its per-scope offset trivially
equals its own optimum (Chebyshev centre of a 1-point set) and q_m ≡ 0. The
d ≥ 2 *multi-task* quantization regime — the one Prop 3 leaves open — is thus
never actually run in the paper's 2×2. This is a real coverage gap, stated
honestly, not a claimed result.

## 2. What changes (two edits, both logged)

1. **Scope key = label SET, not ordered tuple.** MNLI (`neutral, entailment,
   contradiction`) and CB (`entailment, contradiction, neutral`) share the SAME
   three labels in a different order. Keying by `frozenset`/sorted-tuple couples
   them into one 3-way NLI scope. `fit_shared_offset` is already token-indexed,
   so coupling by set is exactly what its worst-task-loss objective expects.
2. **Relaxed floor** `--scope-min-rarest 12` (paper uses 40) admits CB (rare
   class ≈ 16). CB is the ONLY task the relaxed floor adds; every other task
   already clears 40. CB's small rare class ⇒ small risk/audit for CB only,
   flagged per task and never averaged into a headline without the flag.

Nothing else changes: budget (shared rank-8 == 4×rank-2), grouping map, splits,
gauge-fixing, restricted-argmax balanced accuracy on `audit`, `test.json` unread.

## 3. Metrics

Per task, the 2×2 corners (`R_sh, R_sh_off, R_gp, R_gp_off`) plus its scope key,
K_S (labels in the scope), scope size (tasks in the scope), and an
`in_ksge3_multi` flag (K_S ≥ 3 AND ≥ 2 member tasks). Aggregated over BOTH the
K_S≥3-multi subset (the open regime) and ALL tasks (reference). Per multi-task
scope, q_m for m = 1.. via `offsets.conflict_statistics` (reused verbatim), under
both the shared and grouped regimes (q_m is a property of the trained model's
per-task optima, so it is regime-specific).

## 4. Frozen criteria (3 seeds; sign consistency)

* **C1 (primary).** `e2e = R_gp_off − R_sh` > 0 on the K_S≥3-multi subset, EVERY
  seed. The method's end-to-end win survives in the non-binary multi-task regime.
* **C2 (output constraint present).** `offset_alone = R_sh_off − R_sh` > 0 on the
  K_S≥3-multi subset, every seed. Confirms the output-layer gap the offset relieves
  is real for d ≥ 2, not only for binary scopes.
* **C3 (genuine d ≥ 2 instance, reported).** q_1 for the multi-task scope is
  reported and > 0 (a singleton or coincident-optima scope would give q_1 = 0).
  Reported, not thresholded for a pass/fail on the method — it certifies the scope
  is a real non-degenerate quantization instance so C1/C2 are meaningful.
* **Training gate, unconditional.** Median `R_sh` (no offset, all tasks) > 0.60.

## 5. Predeclared outcomes

1. **C1 and C2 pass, q_1 > 0.** The method generalizes to the open d ≥ 2 regime
   empirically; §07's "non-binary scope unproven" becomes "unproven but measured
   positive on a genuine 3-way NLI scope (MNLI+CB), q_1 = …". The theory stays
   open; the empirical coverage gap is closed.
2. **C1 passes, C2 fails on the subset.** Grouping carries the subset but the
   shared offset does not help there (worst-task-loss objective conceding CB or
   MNLI). Report the offset as scope-degenerate on this scope and give the
   grouping-only number as the subset win. Do not average the failure away.
3. **C1 fails on the subset.** The non-binary multi-task scope is where the method
   does NOT hold. Report it plainly, with the per-scope q_m and the two tasks'
   individual deltas, and scope the headline claim to binary/singleton scopes.
   This is a real boundary of the method, reported, not hidden.
4. **CB not admitted / no multi-scope forms.** The relaxed floor still failed a
   precondition (first-piece collision) or CB's classes are too small for any
   risk/audit split. Report that the d ≥ 2 multi-task regime could not be built
   from Order-4 and that a purpose-built stream (3D-style external data) is the
   next step — not a result about the method.

## 6. Anti-artifact checks, before any criterion is quoted

* **CB is the only added task** — `eligible_tasks(min_rarest=12)` logged against
  the paper's `min_rarest=40` list; the diff must be exactly {CB} (else the floor
  admitted something unintended and the run is void).
* **Scope coupling is by label set** — MNLI and CB must land in ONE scope
  (`scope_size == 2`, `K_S == 3`); asserted in the JSON (`ksge3_multi_scopes`).
* **q_m from per-task optima on the risk split only** — never audit, never test;
  `conflict_statistics` reuses the paper's exact `_worst_case_quantization_radius`.
* **CB rare-class flagged** — `n_audit`/`K` per task in the JSON expose CB's small
  eval set; it is reported per task, never silently averaged into the headline.
* Offsets gauge-fixed; deployable and reference numbers on the SAME audit logits.
* Matched 2.36M budget re-logged.

## 7. Compute

3 seeds, trains both regimes then scores + q_m, ~10–12 h wall, 80 GB. BABEL
`-p preempt --qos=preempt_qos -A bapoczos --gres=gpu:1`, Blackwell refused
(exit 75). One new hyperparameter (`--scope-min-rarest`) fixed at 12 in the sweep,
declared here before running.
