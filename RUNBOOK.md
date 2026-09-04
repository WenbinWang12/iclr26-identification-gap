# RUNBOOK — reproduce and extend the experiments

This file is for a collaborator running the larger-scale experiments (and for an
AI coding tool reading the repo). It tells you **what to run, in what order, and
how to report results** so every number stays traceable. Read §0 first — the
honesty protocol is not optional; it is the point of this project.

The current method and headline numbers are in [`README.md`](README.md). The full
frozen protocol (stream, splits, metrics, training gate) is in
`appendix/experimental_details.tex`. Per-phase protocols are in `notes/*_protocol.md`.

---

## 0. Ground rules (honesty protocol) — read before running anything

1. **Every reported number traces to a frozen run.** No number goes into a table,
   the paper, or a README that was not produced by a committed script over
   committed data and written to a committed `*.json`. If you cannot point at the
   file that produced it, it does not exist.
2. **Freeze the protocol before the run.** For any new experiment, write a
   `notes/<phase>_protocol.md` *first*, stating the method, the criteria with
   **numeric thresholds**, and a set of **predeclared outcomes** (what each
   possible result would mean). Record its SHA and never edit a criterion after
   the result is known — append an amendment with its own date/hash instead. Use
   any existing `notes/*_protocol.md` as the template.
3. **No oracle leakage at evaluation.** Never read the task index, the task label
   set, or the data generator direction to produce a *deployable* number. Where a
   quantity is an upper bound (oracle routing, offset refit on the final model),
   label it as an upper bound in the output and in the writeup. (This project has
   already caught and retracted one oracle-init artifact; do not add another.)
4. **Report negatives honestly. Do not tune to a target.** If a result is mixed,
   weak, or negative, report it in weak/negative wording and keep it. Improving
   the *method* is welcome; massaging the *number* is not. Preserve every caveat
   already in `README.md` / the paper when you touch adjacent text.
5. **Matched budget, matched splits, matched seeds.** All comparisons are internal:
   arms differ in exactly one component at an identical trainable-parameter budget,
   identical `risk`/`audit` splits, identical seeds. Official test files are never
   opened.

---

## 1. Environment

- **Backbone / libs**: T5-large (`google-t5/t5-large`), `torch`, `transformers`,
  `peft`, `numpy`. bf16. Single 40–80 GB GPU is enough for the current runs.
- **HF mirror + cache**: the `run_*.py` scripts already set `HF_HOME` and
  `HF_ENDPOINT` defaults; override via env if your cluster differs.
- **Data**: pass `--data-root <dir>` = the Order-4 stream directory (per-task
  `train` files). The task set, labels, and verbalizers come from
  `experiments/phase2i_anchored_cvar/order4_data.py` (`ORDER4_TASKS`). Tasks are
  filtered at runtime by `eligible_tasks(...)`: first-piece precondition +
  rarest-class ≥ 40 (so CB, Yelp, Amazon are excluded from scored means).
- **Smoke test first**: `experiments/phase2z_task_grouping/sanity.sh` runs
  `sanity_adapters.py` on `t5-small` and asserts the fixed-budget adapter wiring
  (`K × R/K == R`). Run it before any GPU job.
- **SLURM**: the `sweep_*.sh` headers (`-A bapoczos`, `/home/pengq/...`,
  venv auto-discovery under `/data/user_data/pengq/*venv*`) are **babel-specific
  examples**. Adapt the account, paths, and venv discovery to your cluster. The
  Python entry points (`run_*.py`) are cluster-agnostic; only the wrappers are not.

---

## 2. Reproduce the current result (do this BEFORE extending anything)

Run all three scripts on seeds 1, 2, 3 with the frozen args, then aggregate.

```bash
cd <repo-root>
export PYTHONPATH=<repo-root>
DATA=/path/to/order4          # your --data-root
COMMON="--data-root $DATA --epochs 7 --cap-per-class 400 \
        --batch-size 4 --grad-accum 16 --lr 3e-4 --dtype bfloat16 \
        --shared-rank 8 --group-rank 2 --lora-alpha 32 --lora-dropout 0.05 \
        --risk-per-class 32 --audit-per-class 32 --eval-batch-size 4"

for S in 1 2 3; do
  python experiments/phase2z_task_grouping/run_grouping.py $COMMON \
    --seed $S --out experiments/phase2z_task_grouping   # writes results_s$S.json
  python experiments/phase2z_task_grouping/run_combined.py $COMMON \
    --seed $S --out experiments/phase2z_task_grouping   # writes combined_s$S.json
  python experiments/phase2z_task_grouping/run_destale.py $COMMON \
    --seed $S --out experiments/phase2z_task_grouping   # writes destale_s$S.json
done

python experiments/phase2z_task_grouping/analyze_combined.py
python experiments/phase2z_task_grouping/analyze_destale.py
```

**Acceptance check** — the aggregate must reproduce (within seed noise) the frozen
values already committed:

| quantity | expected |
|---|---|
| 2×2 corners `R_sh / R_sh_off / R_gp / R_gp_off` | 68.68 / 75.25 / 78.07 / 80.43 |
| end-to-end `R_gp_off − R_sh` | +11.76 pp (all seeds +) |
| grouping on top of offset `R_gp_off − R_sh_off` | +5.18 pp (all seeds +) |
| staleness shared → grouped | 8.22 → 1.95 pp (all seeds +) |

If your fresh run disagrees materially, **stop and reconcile before extending** —
do not proceed on top of an unreproduced baseline.

**Note on the current upper bounds** (these are what §3A closes):
`run_grouping.py` / `run_combined.py` currently evaluate the grouped arm with
**oracle scope routing** (the correct family adapter is activated using the task's
true group), and the offset column uses the **refit** offset (fit against the
final model). `run_destale.py` additionally scores the **stored** offset
(`R_stored`, deployable) vs the refit offset (`R_refit`, upper bound) under each
regime.

---

## 3. Larger-scale experiments

Four axes, in priority order. Each names the paper limitation (§07) it closes,
the concrete code change, and the success/kill criterion. **Freeze a protocol note
per axis before running** (§0.2).

### 3A. Deployable router + stored-offset 2×2  *(highest priority)*

**Why.** §07 "Our proposed method is scored at two upper bounds" and "Grouping
de-stales the stored offset…": the paper's own stated *first measurement we owe*.

**What to build.**
- A **task-free family router**: activate a family adapter from the input alone,
  no task index. Start with the candidate the paper names — nearest **stored
  class-mean in encoder state** — computed per family from `risk`-split encoder
  states. `run_grouping.py`'s docstring already reserves `R_grouped_route`; the
  eval loop does **not** implement it yet, so add it alongside `R_grouped_orc`.
- Re-run the **full 2×2 with the stored offset** (deployable), not the refit
  offset, so the proposed cell is fully deployable end-to-end.

**Report.** The grouped-oracle → grouped-routed **gap** (per task + mean), and the
stored-offset 2×2 corners next to the current refit-offset corners.

**Success / kill.** Success = routed grouped arm keeps a positive, all-seed
end-to-end gain over shared/no-offset with the stored offset. Kill/honest-negative
= router recovers < (state a threshold in the protocol, e.g. half) of the
oracle→routed gap, or the stored-offset end-to-end gain is mixed in sign — report
it as such; do not switch back to the oracle number silently.

### 3B. Generality — orders × backbones

**Why.** §07 "One stream, one order, one model." Order effects in continual
learning are large and unmeasured here.

**What to build.**
- Parameterize the **task order** (add a `--order` arg or an order table; permute
  the sequence `train_stream` iterates). Run ≥ 3 orders besides Order-4.
- Second/third **backbone**: `--model` already exists. Add `google-t5/t5-3b` and a
  **decoder-only** LM (the restricted-argmax verbalizer decode must be reimplemented
  for a causal LM — the T5 seq2seq path in `collect_logits` will not transfer as-is;
  scope this in the protocol).

**Report.** The 2×2 effects per (order, backbone); state whether all four effect
signs hold across every cell.

**Success / kill.** Success = end-to-end and grouping-on-top-of-offset stay
positive across orders and on ≥ 1 larger/decoder backbone. Report any cell where a
sign flips — that is a finding, not a failure to hide.

### 3C. Theory coverage — scorable K_S ≥ 3 shared scopes

**Why.** §07 "proved in a special case…": Proposition `qm` is proved only for
binary shared verbalizers (`d = 1`); `m ≥ 2, d ≥ 2` is **OPEN**. Every scorable
multi-task scope on Order-4 is binary because CB / Yelp / Amazon are excluded.

**What to build.**
- Assemble a stream with **scorable non-binary shared scopes** (e.g. a 3-way NLI
  scope with enough rarest-class examples to survive the ≥ 40 filter; or add tasks
  that share a ≥ 3-label verbalizer). Adjust `eligible_tasks` thresholds only in a
  documented, pre-frozen way.
- Compute per-scope `q_m` (the m-centre quantization radius) on these scopes and
  run the 2×2 restricted to the K_S ≥ 3 subset.

**Report.** Per-scope `q_1`, `q_2`; whether the offset still recovers gap on ≥ 3-way
scopes; the 2×2 on the K_S ≥ 3 subset. This is where new theory would be needed —
flag any empirical behavior the `d = 1` proof does not cover.

### 3D. Published baselines at matched budget

**Why.** §07 "No comparison against published continual-PEFT methods." Enables an
*external* comparison; today all comparisons are internal.

**What to build.**
- Run **O-LoRA** (a proper implementation, not our re-implemented penalty),
  **E²-LoRA**, **NSR** at the **identical 2.36M-scalar budget**, identical
  `risk`/`audit` splits, identical seeds. No leaderboard / test-file use.

**Report.** A matched-budget table: each baseline vs shared/no-offset vs the
proposed grouped+offset cell, on the same audit metric. State clearly that these
are re-implementations at our budget/splits, not reproductions of published tables.

**Success / kill.** No "we beat SOTA" claim is required or wanted; the deliverable
is an honest matched-budget comparison. If a baseline beats the proposed method,
report it.

---

## 4. Output contract (keep results traceable)

- Each run writes `experiments/phase2z_task_grouping/<name>_s{seed}.json`, a list
  (or `{regime: {rows: [...]}}` for de-stale) of **per-task audit accuracies** with
  the task name, group, K, n_audit, and each arm's balanced accuracy. Match the
  existing schema in `combined_s*.json` / `destale_s*.json`.
- Every new axis ships a matching `analyze_<axis>.py` that aggregates from the
  frozen JSONs and prints per-seed + aggregate effect sizes with **sign
  consistency** across seeds (copy the pattern in `analyze_combined.py` /
  `analyze_destale.py`). Aggregation must read only committed JSONs — never
  recompute from a live model at report time.
- Use `balanced_accuracy`, `fit_offset`, `fit_shared_offset`, `gauge_fix` from
  `experiments/phase2j_offset_conflict/offsets.py`; `collect_logits`,
  `train_one_task`, `precondition_ok` from `.../run_probe.py`; and the grouping
  `GROUP_OF` / `build_adapters` / `train_stream` from `run_grouping.py`. Do not
  reimplement these.

---

## 5. Reporting back

1. Commit the frozen `*_s{seed}.json`, the new `analyze_*.py`, the run scripts, and
   the pre-frozen `notes/<phase>_protocol.md` (with its recorded SHA).
2. Open a PR summarizing: which axis, the aggregate numbers with per-seed sign
   consistency, and which §07 limitation it addresses. Quote the honesty caveats
   that still apply.
3. Keep negative and mixed results in the PR. A clean negative that closes an open
   question is a valid, wanted contribution here.
