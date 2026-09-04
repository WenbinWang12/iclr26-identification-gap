# Phase-2Z-F (RUNBOOK 3B) — generality across orders × backbones: frozen protocol

Frozen 2026-09-04, **before any Phase-2Z-F run exists**. The paper's 2×2 is
measured on exactly one stream order (canonical Order-4) and one backbone
(T5-large). §07 lists "single backbone, single stream order" as an external
validity limitation. This phase reruns the identical 2×2 under other orders and
backbones and asks: **do the four effect signs hold?** Implemented in
`experiments/phase2z_task_grouping/run_generality.py` (reuses `orders.py` +
`backbones.py`), judged by `analyze_generality.py`.

## 1. What varies, what is held fixed

Only two things change per cell: the **training order** (`--order`) and the
**backbone** (`--model`). Everything else is byte-identical to `run_combined.py`:
eligibility filter, `prepare_task_partitions` splits, per-scope offset fit
(`fit_scope_offsets`), gauge-fixing, restricted-argmax balanced accuracy on the
`audit` split, `GROUP_OF`/`GROUP_NAMES` family map, matched fixed budget
(shared rank-8 == 4×rank-2). Any 2×2 difference across cells is therefore a pure
order/backbone effect, not a harness change.

* **Orders** (`orders.py`, explicit name permutations, no RNG): `canonical`,
  `reverse`, `shuffleA`, `shuffleB`. Each asserted at import to be a permutation
  of the 15 tasks.
* **Backbones** (`backbones.py`): `seq2seq` (T5 family — path delegates to
  `run_probe.train_one_task`, byte-identical to the paper for T5) and `causal`
  (decoder-only LM — distinct scoring at the **last prompt position** and
  next-token LM loss with prompt positions masked to −100). The verbalizer
  coordinate is the label's **first sub-word piece id** in both paths.

## 2. Cells to run

Priority order (each cell = seeds 1–3):
1. `canonical / t5-large` — **internal consistency check**: must reproduce the
   Phase-2Z-C regime (e2e ≈ +11.76 pp, all four effects +). A large disagreement
   is a harness bug, not a finding.
2. `reverse / t5-large`, `shuffleA / t5-large`, `shuffleB / t5-large` — order
   sensitivity on the paper backbone.
3. `canonical / t5-3b` — scale on the same family.
4. `canonical / <decoder-only LM>` — cross-architecture (causal path).

Cells 2–4 are independent; run as compute allows. A missing cell is reported as
missing by the analyzer, never imputed.

## 3. Frozen criteria (per effect, across ALL cells × seeds)

The four effects (mean over eligible tasks, ×100 pp):
* `e2e = R_gp_off − R_sh` (both moves; paper headline),
* `repr = R_gp_off − R_sh_off` (grouping on top of offset),
* `group_alone = R_gp − R_sh`,
* `offset_alone = R_sh_off − R_sh`.

* **G1 (primary — sign robustness).** `e2e` > 0 on **every** cell × seed. This is
  the generality claim: the end-to-end win is not an artifact of one order or one
  backbone.
* **G2 (component robustness).** `repr` and `offset_alone` each > 0 on every
  cell × seed. If one flips, the composed win may survive while a component does
  not — reported per cell, not averaged away.
* **G3 (consistency anchor).** The `canonical / t5-large` cell reproduces the
  Phase-2Z-C aggregate within seed noise (e2e within ±2 pp of +11.76). Else the
  generality harness diverged from the paper harness and no other cell is quoted.
* **Training gate, unconditional.** Median post-training `R_sh` (no offset) > 0.60
  per cell; else that cell is void (esp. relevant for the causal backbone, whose
  scoring/decoding path is new).

## 4. Predeclared outcomes

1. **G1 holds on all cells.** External validity limitation discharged for the
   tested orders/backbones; §07 updated with the cell table, e2e per cell.
2. **G1 holds on orders but a backbone flips a component (G2).** Report the sign
   table honestly: the method's end-to-end win generalizes across orders; on
   backbone X, component Y flips — state which and why (likely the causal scoring
   path or a family with degenerate verbalizer pieces). No averaging over the flip.
3. **G1 flips on some order.** Order sensitivity is real; report the flipping
   order(s) explicitly and the mean e2e, and scope the headline claim to the
   orders where it holds. Do **not** report only the favorable orders.
4. **Causal cell fails the training gate.** The decoder-only scoring/decode path
   did not reach a usable operating point; report the seq2seq (T5-3b) generality
   result and mark the causal backbone as not-yet-supported, not as a negative
   finding about the method.

## 5. Anti-artifact checks, before any criterion is quoted

* **Order is a pure permutation** — `orders.py` asserts each order is a
  permutation of the canonical 15 at import; a non-permutation raises, never runs.
* **T5 path unchanged** — the seq2seq branch of `backbones.train_one_task`
  delegates to `run_probe.train_one_task`; `canonical / t5-large` must match 2Z-C
  (G3), which is exactly this check.
* **Causal scoring reads the last real token** — left padding is set at load so
  `logits[:, -1, :]` is the final prompt token for every row; verified by the
  training gate (a mis-indexed score would fail the >0.60 gate).
* **Verbalizer coordinate identical across backbones** — both paths score the
  label's first sub-word piece id (`first_piece_ids`); the offset machinery is
  unchanged.
* **Offsets fit on `risk` only**, scored on `audit`; official `test.json` unread.
* Matched 2.36M-scalar budget re-logged per cell (`--shared-rank 8 --group-rank 2`);
  a `shared_rank != K·group_rank` mis-set is warned in the runner.

## 6. Compute

Per cell: 3 seeds, trains both regimes (shared then grouped) then scores, ~10–12 h
wall each on T5-large; T5-3b larger memory. BABEL `-p preempt --qos=preempt_qos
-A bapoczos --gres=gpu:1 --mem=80G`, Blackwell nodes refused (exit 75). Cell
selected at submit time via `--export=ALL,ORDER=...,MODEL=...`.
