# Two Capacity Constraints, One Fixed Budget

Grouping and offsets in task-agnostic continual PEFT. ICLR 2026 paper draft +
reference implementation, harness, and frozen decision artifacts.

> **Extending the experiments?** Read [`RUNBOOK.md`](RUNBOOK.md). It is written
> for a collaborator (and their AI coding tool) and states exactly what to run,
> in what order, and how to report results back so they stay traceable.

## What this is

Continual parameter-efficient fine-tuning is budgeted almost entirely in
*trainable parameters*. This paper identifies a **second** constraint a parameter
budget does not buy, and proposes a method that relieves both at a bit-identical
budget.

When a generative model serves a stream of tasks **without task identity**, tasks
that reuse output tokens (e.g. `False`/`True`, `Bad`/`Good`) must share one
decision rule over those tokens. We separate two things that both look like
"forgetting":

- **Output layer — the identification gap `Δ^id`.** The part of recoverable
  forgetting that needs to know *which* task a query came from. It has **zero
  rank**: a per-label additive offset corrects it, costing a handful of floats.
  It is exactly zero when verbalizer sets are pairwise disjoint
  (Proposition, falsifiable boundary), and it is *not* a parameter-count problem.
- **Representation layer — deep forgetting.** A single shared adapter cannot hold
  fifteen tasks' worth of representation change; late tasks overwrite early ones.
  Three parameter-space penalty/anchoring interventions failed to touch this.

## The method (two matched-budget moves)

- **Output move — per-scope offset.** Read the prompt's option list, look up its
  *scope* (the set of tasks sharing that verbalizer), add the offset that best
  serves that scope. Zero rank, ~**33 floats** on this stream, reads **no task
  index**.
- **Representation move — task-family grouping.** Split the tasks into **K = 4**
  semantic families and give each a **rank-2** adapter. The four hold exactly the
  parameters of one shared **rank-8** adapter (`K × R/K = R`, checked at runtime),
  so this is a *reallocation*, not extra capacity. Each family adapter is
  perturbed only by its own family, containing cross-family interference.

The clean test is the **2×2** at one budget: `{shared, grouped} × {no offset, +offset}`.

## Headline result (Order-4, T5-large LoRA, 3 seeds)

2×2 balanced-accuracy corners (mean over seeds), all traceable to
`experiments/phase2z_task_grouping/combined_s{1,2,3}.json` via
`analyze_combined.py`:

| | no offset | + per-scope offset |
|---|---|---|
| **shared** (rank 8) | 68.68 | 75.25 |
| **grouped** (4×rank 2) | 78.07 | 80.43 ← proposed |

Effect sizes (positive on **every** seed):

| effect | value |
|---|---|
| end-to-end (grouped+offset − shared/no-offset) | **+11.76 pp** |
| grouping on top of the offset (`R_gp_off − R_sh_off`) | +5.18 pp |
| grouping alone in the 2×2 (`R_gp − R_sh`) | +9.39 pp |
| offset alone (`R_sh_off − R_sh`) | +6.58 pp |

The two moves are **independent and additive**. Grouping also **de-stales** the
deployable stored offset: staleness drops from **8.22 pp → 1.95 pp** (all three
seeds; `analyze_destale.py` over `destale_s{1,2,3}.json`).

### Honesty caveats (do not drop these when quoting the numbers)

- The `+11.76 pp` is scored at **two upper bounds**: oracle scope routing (family
  adapter set from the scope at eval) and the offset **fit against the final
  model** on each task's `risk` split. Both are ceilings, not deployable rules.
  Closing the grouped-oracle→grouped-routed gap with a task-free router is the
  first measurement we owe — see `RUNBOOK.md` §3A.
- The stored (deployable) offset's **standalone** net gain over no offset under
  grouping is **mixed in sign** (mean −0.51 pp). Grouping removes the *harm*
  staleness caused on a shared adapter (−5.08 pp), but on this stream the stored
  offset is not a reliable standalone improvement; its value is that it **composes
  with grouping** near the upper bound.
- Every number is an **internal** comparison at matched budget on `audit` splits
  carved from training files; official test files are never opened, so absolute
  accuracies are **not** leaderboard-comparable.

## Repository map

| Path | Purpose |
|---|---|
| `main.tex` | ICLR entry point |
| `sections/` | Main paper (01 intro … 08 conclusion) |
| `appendix/proofs.tex` | Proofs, counterexamples, claim-status ledger (two OPEN cases flagged) |
| `appendix/experimental_details.tex` | Full reproducible protocol (stream, splits, metrics, gate) |
| `references.bib` | Bibliography (41 verified references) |
| `experiments/phase2z_task_grouping/` | **Current method**: grouping + combined 2×2 + de-stale, with frozen result JSONs and analyzers |
| `notes/*_protocol.md` | Per-phase protocols, frozen with criteria+thresholds before each run |
| `notes/claim_assumption_ledger.md` | Claim-to-assumption audit, incl. self-retractions |
| `RUNBOOK.md` | How to reproduce and extend the experiments (collaborator-facing) |

Key entry points in `experiments/phase2z_task_grouping/`:
`run_grouping.py` (shared vs grouped, oracle + learned-router scaffold),
`run_combined.py` (the 2×2), `run_destale.py` (stored vs refit offset under each
regime), `analyze_combined.py` / `analyze_destale.py` (canonical aggregation),
`sweep_*.sh` (SLURM templates), `sanity.sh` (t5-small smoke test).

## Build

From this directory:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Clean rebuild:

```powershell
latexmk -C
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Keep `\iclrfinalcopy` commented during anonymous review.

## Superseded work (provenance only)

Earlier iterations of this project pursued different methods that are **not** part
of the current paper. They are kept for provenance and integrity, not claimed:

- **PSR-LoRA** (Pareto-Safe Rank Reallocation, `experiments/phase2f_psr_lora/`)
  and **FCRA** (Fixed-pool Curvature-aware Rank Allocation,
  `experiments/phase2c_fcra_minimal/`, `.../phase2d_full_fcra/`,
  `.../phase2e_real_lora/`) were controlled linear/factored-proxy studies. The
  Phase-2e self-audit that caught and retracted an oracle-init artifact (a `+1.00`
  headline that collapsed to `−0.04` oracle-free) is preserved in the ledger.
- Intermediate diagnostic and negative-result phases (`phase2k` QOC, `phase2n/2m`
  anchors, `phase2p` SSO, `phase2q` BPO, `phase2s` SIO, `phase2t` VGC, `phase2u`
  SCG, `phase2w` rank, `phase2y` enhanced) back the paper's *reported* negative
  and boundary results; their frozen outputs and protocols remain in place.

The current paper's claims rest only on the two-constraint method above.
