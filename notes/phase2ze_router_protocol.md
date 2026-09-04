# Phase-2Z-E (RUNBOOK 3A) — deployable router + stored-offset 2×2: frozen protocol

Frozen 2026-09-04, **before any Phase-2Z-E run exists**. Written after the
combined 2×2 (Phase-2Z-C) and the de-stale study (Phase-2Z-D) both landed
positive but at two upper bounds. This phase replaces both ceilings with
deployable rules and measures the cost. Implemented in
`experiments/phase2z_task_grouping/run_router.py`, judged by `analyze_router.py`.

## 0. Why this phase

The paper's headline (+11.76 pp end-to-end) is scored at two upper bounds
(§07 limitation, README caveat #1):
* **routing** uses the task's TRUE family adapter at eval (oracle routing);
* the **offset** column is fit against the FINAL model (refit), not stored.

The paper itself calls closing the grouped-oracle→grouped-routed gap "the first
measurement we owe" (§01, §07). This phase does exactly that, and simultaneously
swaps the refit offset for the deployable stored offset validated in Phase-2Z-D
(grouped staleness 8.22→1.95 pp; grouped stored-offset standalone gain MIXED,
mean −0.51 pp — a caveat carried in verbatim, not softened).

## 1. The method under test

Same trained model as Phase-2Z-D (shared rank-8 vs 4×rank-2 grouped, matched
2.36M budget, epochs 7, Order-4, T5-large). Two deployable substitutions:

* **Task-free NCM router.** After training, store per eligible task the mean of
  the **base (LoRA-disabled) encoder's** masked-mean-pooled last-hidden-state
  over its own `risk` split (a class mean in encoder space). At eval each audit
  query is routed — **with no task index** — to the family of the nearest stored
  prototype (L2), and scored under that family's rank-2 adapter. The router reads
  the base encoder, which is adapter-independent, so the feature a query is routed
  on cannot depend on which adapter the routing selects.
* **Deployable stored offset.** The per-scope Chebyshev-centre offset fit at each
  task's boundary and frozen (Phase-2Z-D's SSO `stored` source), applied by scope
  read from the prompt's option list. The refit offset is still computed but only
  reported as a reference ceiling.

## 2. Metrics

Scored on the `audit` split after the final task, restricted-argmax balanced
accuracy, mean over eligible tasks (same eligibility as 2Z-C/2Z-D). All offsets
fit on train-derived `risk`; official `test.json` unread. Every quantity from
the **same run** (no cross-run reproducibility assumed). Per task we record, under
three routing rules {shared, grouped-oracle, grouped-route} × three offsets
{none, stored, refit}, plus the router's per-task accuracy to the true family.

Headline aggregates:
* **routing cost** = `R_grouped_orc − R_grouped_route` (no offset);
* **deployable e2e** = `R_grouped_route_stored − R_shared` (both moves deployable);
* **reference oracle e2e** = `R_grouped_orc_refit − R_shared` (both ceilings — must
  reproduce the +11.76 pp regime within seed noise as an internal consistency check).

## 3. Frozen criteria (3 seeds; sign consistency across seeds)

* **E1 (primary — deployability).** `deployable e2e` > 0 on **every** seed. This is
  the whole claim: the fully deployable cell still beats the shared/no-offset
  baseline.
* **E2 (router quality).** The routed grouping gain recovers ≥ 50% of the oracle
  grouping gain: `(R_grouped_route − R_shared) ≥ 0.5 · (R_grouped_orc − R_shared)`,
  mean over seeds. Failing quantifies how much the oracle bought.
* **E3 (routing accuracy, reported).** Mean router→true-family accuracy, per task
  and pooled. Reported, not thresholded — it explains E1/E2.
* **E4 (no silent ceiling substitution).** `analyze_router.py` reports the
  deployable numbers as primary; the oracle/refit numbers appear only under
  "reference". Asserted by the analyzer's output layout, not by prose.
* **Training gate, unconditional.** Median post-training `R_shared` (no offset)
  > 0.60, as in every prior phase; else the run is void.

## 4. Predeclared outcomes

1. **E1 and E2 pass.** The upper bounds were not load-bearing: the method is
   deployable end-to-end. README caveat #1 is discharged; the headline is restated
   with the deployable number as primary and the oracle number as the ceiling.
2. **E1 passes, E2 fails.** Still deployable and still a net win, but the router
   leaves measurable gain on the table vs oracle routing. Report the routed number
   as the method's operating point and the oracle as the ceiling, with the gap
   stated in the abstract.
3. **E1 fails, E3 shows poor routing.** The NCM router is the bottleneck. Clean,
   publishable: the representation gain is real (oracle) but this router cannot
   harvest it; a better task-free router is the open problem. Do **not** revert to
   the oracle number as if deployable.
4. **E1 fails, E3 shows good routing.** Then the loss is the stored offset (the
   MIXED-sign standalone gain from 2Z-D biting end-to-end), not the router. Report
   grouping-alone (routed, no offset) as the deployable win and state that the
   deployable offset does not compose here.

## 5. What is already known, frozen so it cannot be re-labelled

* Oracle-routing + refit-offset 2×2 (Phase-2Z-C): `R_gp_off − R_sh = +11.76 pp`,
  all seeds +. The reference oracle e2e here must reproduce this regime; a large
  disagreement means a harness bug, not a finding.
* Grouped stored-offset standalone gain (Phase-2Z-D) is **MIXED sign**, mean
  −0.51 pp. So outcome 4 is a live possibility and is not treated as a surprise.
* DBpedia (14-way) loses under grouping at rank 2 (2Z-C); it is reported per task,
  not averaged away, and its routing behavior is expected to be the noisiest.

## 6. Anti-artifact checks, before any criterion is quoted

* **Router reads the base encoder only** — asserted in code via
  `model.disable_adapter()`; the routed feature is adapter-independent.
* **No task index at routing** — the router sees only pooled encoder states and
  stored prototypes; the true family is used solely to *score* routing accuracy
  (E3), never to route.
* **Prototypes fit on `risk` only** — never `audit`, never `test`.
* Every stored and applied offset gauge-fixed to zero mean (inherited from 2Z-D).
* Deployable and ceiling numbers scored on the **same** audit logits per task.
* `order4_data.py`, `offsets.py`, `run_grouping.py` grouping map untouched;
  matched 2.36M budget re-logged.
* `risk`/`audit` disjointness inherited from `prepare_task_partitions`; same seed
  ⇒ identical splits to 2Z-C/2Z-D.

## 7. Compute

3 seeds, trains both regimes then routes (destale-sized: ~10 h wall, 80 GB),
BABEL `-p preempt --qos=preempt_qos -A bapoczos --gres=gpu:1`, Blackwell nodes
refused (exit 75). No hyperparameter to select: NCM has none.
