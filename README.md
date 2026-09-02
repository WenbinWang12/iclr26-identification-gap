# Continual PEFT as Online Capacity Allocation

ICLR-format research draft for fixed-budget continual PEFT under
task-identity-free, clocked temporal mixture drift.

## Status

- The project uses the official ICLR 2026 style and bibliography files. As of
  2026-07-16, an official ICLR 2027 author kit is not yet available.
- The theory is written with explicit assumptions and full appendix proofs.
- The exact NumPy Phase-0 diagnostic has now been run locally and on the
  `alibaba-10` CPU server. Its narrow order/history-interference result is
  recorded in `notes/phase0_remote_run.md`; the PEFT/LLM experimental section
  remains a protocol only.
- The audited Phase-1 synthetic theory suite has been reproduced locally and
  on `alibaba-10` (`7/7` tests and `163/163` gates on both hosts). The
  conditional result and four-provider post-result audit are recorded in
  `notes/phase1_remote_run.md`; this is not a neural or benchmark result.
- The theory-consistency checks are frozen and passing: expressivity predicts
  forgetting (D1) and rare-task retention (D2-A), while a minimal nonlinear model
  (D2-B, a two-layer ReLU MLP — **not** a Transformer) is an **honest negative** —
  the curvature-tail prediction does not transfer out of the linear/
  Kronecker-quadratic regime, so no real-LoRA or LLM claim is warranted yet. These
  are controlled NumPy artifacts, not neural or benchmark results.
- **PSR-LoRA (Pareto-Safe Rank Reallocation) is the paper's main method and
  passes an internally pre-specified, hash-recorded held-out kill test (Phase-2f /
  D2-C).** PSR-LoRA keeps a single fixed-rank-`R` adapter and, each window,
  re-solves a risk-reweighted reduced-rank regression from a bounded buffer of
  total budget `B=16`. The budget is split for **coverage**: a small uniform
  (Vitter) reservoir (`B_u=4`) for an unbiased view of history, plus an
  input-direction-diverse buffer (`B_g=12`) that keeps rare input directions
  (hence rare sources) represented — a single uniform reservoir under-samples the
  rare/worst source, the diagnosed cause of the earlier tail-coverage failure. It
  estimates a pooled ridge-LS teacher from the buffer union only, then runs a
  minimax multiplicative-weights loop that re-solves the closed-form weighted
  rank-`R` problem each round, keeping the iterate with the smallest buffered
  worst-window loss (round 0 = uniform solve, so the update is never worse than
  uniform by construction — this replaces rollback). It is oracle-free (a function
  of `(X,Y)` only; the diversity score uses the top right singular direction of
  the window inputs, never labels). The distinction the evidence turns on is
  **implicit vs. explicit allocation**: every allocator that spends the budget
  implicitly (dense sequential, reservoir/GSS/MIR replay, A-GEM gradient
  projection, and the uniform-weight solve) clusters at the spectral-tail capacity
  floor (worst-group tail `≈0.58–0.61`); coverage **and** reweighting together
  raise the worst-served source above it. On twenty held-out worlds (five new
  teacher geometries × four new stream seeds, capacity-limited regime, `R=4<K`,
  oracle-free, independent held-out retention) PSR-LoRA reaches worst-group tail
  `0.792±0.062` (best arm, `94%` of the offline frontier `0.839`) at
  frequency-weighted mean `0.970±0.005`, and all three criteria pass on the
  cluster-robust two-way crossed CI (df=3, the decision object; IID CI shown only
  for transparency): (a) non-inferior mean vs. excess-CVaR `+0.0027`
  (`[−0.0026,+0.0079]`); (b) strictly higher tail vs. excess-CVaR `+0.1782`
  (`[+0.0696,+0.2869]`); (c) reweighting given coverage vs. the uniform-weight
  solve `+0.0945` (`[+0.0477,+0.1413]`). The **coverage** contribution on its own
  (`+0.0918` vs. the single-reservoir ablation) is real in the mean but its
  cluster CI `[−0.0085,+0.1922]` **straddles 0** — reported as the mechanism that
  flips (b), not an independently CI-significant increment. This is a
  **paired-average** superiority, not uniform dominance: PSR-LoRA beats excess-CVaR
  on `19/20` worlds and the uniform solve on `20/20`. Off-floor it does no harm
  (compressible regime, ties the uniform solve) and degrades gracefully without a
  false rescue (conflicting regime). Frozen citable run:
  `experiments/phase2f_psr_lora/` with the held-out kill test in
  `preregister_heldout.py` and the frozen `outputs_heldout.txt` (current SHA-256:
  `psr_lora.py 24ad55c3…`, `preregister_heldout.py 9fd78bd7…`,
  `outputs_heldout.txt b11c3528…`; see `notes/phase2f_psr_lora_protocol.md`). The
  older `compare.py` / `outputs_compare_v2.txt` is the **superseded** pre-coverage
  uniform-16 record (tail 0.678, (b)/(c) fail), kept for provenance.
  Controlled linear factored proxy only — not a real-LoRA / LLM / benchmark
  result; combined with D2-B the honest boundary is that reallocation
  **redistributes** the fixed budget to raise worst-group retention at matched
  mean, only in the linear regime where the floor is exactly characterized — it
  does not and cannot lower the aggregate floor.
- **Iteration-era method FCRA (Phases 2c–2e) is superseded by PSR-LoRA and is
  not part of the paper.** FCRA (Fixed-pool Curvature-aware Rank Allocation) was
  the earlier fixed-pool-of-atoms design (historical trust region, four atom
  states, top-`k` score, curvature-weighted SVD consolidation,
  validation/backtracking). Its frozen runs remain for provenance only
  (`experiments/phase2c_fcra_minimal/`, `experiments/phase2d_full_fcra/`,
  `experiments/phase2e_real_lora/`, with the pre-frozen `notes/phase2*_protocol.md`),
  and the Phase-2e self-audit that **caught and retracted an oracle-init
  artifact** (the FCRA arm had read the generator direction `v=V[:,k]` and
  reported an analytic, not held-out, retention; the retracted `+1.00` headline
  collapses to `−0.04` oracle-free) is preserved as an integrity record in the
  ledger (`C15` retracted, `C16` honest negative). None of these FCRA phases is
  claimed in the paper; `C10/C13/C14` are marked superseded in the ledger.

## Main claim

The draft does **not** claim that all forgetting is a capacity problem. Under a
local second-order model, the historical retention gap is decomposed into:

1. irreducible conflict among distributions;
2. approximation error of the fixed-rank update class; and
3. online estimation, allocation, and optimization error.

PSR-LoRA maintains one fixed-rank-`R` adapter without task-ID routing and, each
clocked window, re-solves a risk-reweighted reduced-rank regression from a
bounded buffer — explicit reallocation of the fixed budget rather than growing,
freezing, or per-window slot activation.

## Files

| Path | Purpose |
|---|---|
| `main.tex` | ICLR entry point |
| `paper_macros.tex` | Paper-specific notation and theorem environments |
| `sections/` | Main paper sections |
| `appendix/proofs.tex` | Proofs, counterexamples, and claim-status ledger |
| `appendix/experimental_details.tex` | Reproducible evaluation protocol |
| `references.bib` | Bibliography, including 2026 nearest work |
| `notes/research_plan_cn.md` | Chinese response to advisor comments |
| `notes/claim_assumption_ledger.md` | Claim-to-assumption audit |
| `notes/experiment_checklist.md` | Implementation and experiment checklist |
| `notes/phase0_remote_run.md` | Frozen local/remote Phase-0 confirmation record |
| `notes/phase1_remote_run.md` | Frozen local/remote Phase-1 theory confirmation record |
| `experiments/phase2f_psr_lora/` | **PSR-LoRA main method + internally pre-specified, hash-recorded held-out kill test (D2-C): `preregister_heldout.py` + frozen `outputs_heldout.txt`; `notes/phase2f_psr_lora_protocol.md` transcribes the code-frozen protocol. Superseded pre-coverage record: `compare.py` + `outputs_compare_v2.txt`** |
| `experiments/phase2c_fcra_minimal/` | Superseded FCRA iteration (provenance only); frozen outputs + `notes/phase2c_fcra_protocol.md` |
| `experiments/phase2d_full_fcra/` | Superseded full-FCRA iteration (provenance only); frozen outputs + `notes/phase2d_full_fcra_protocol.md` |
| `experiments/phase2e_real_lora/` | Superseded FCRA factored proxy + oracle-init retraction record (provenance/integrity); `notes/phase2e_oracle_free_protocol.md` |
| `figures/README.md` | Required evidence and figure specifications |

## Build

From this directory:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

For a clean rebuild:

```powershell
latexmk -C
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Keep `\iclrfinalcopy` commented during anonymous review. Replace the 2026
`.sty` and `.bst` together when the next official author kit is released.
