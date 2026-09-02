# Phase-2H Stage-D2 feasibility panel — RESULT: NO-GO

> **BURNED 2026-08-27.** All BERT-tiny D1/D2 outcomes below are development-only
> and are now marked BURNED: the single predeclared representation fallback was
> taken (backbone → `prajjwal1/bert-mini`, hidden 256), and D0–D2 are being rerun
> from scratch. This file is retained unedited as an honest record of the bert-tiny
> NO-GO. The bert-mini rerun result will be written to a separate file.

Run 2026-08-27, `run_development.py --stage D2 --cap 12000`, dev seeds
[1234, 5678, 9012], BERT-tiny + rank-4 LoRA, McAuley Amazon-Reviews-2023 6-source
drift stream, 86,272-byte historical envelope. Log: `d2_panel.log`.

**Development-only. No outcome here is a citable result.** D2 decides only whether
SGCR (D3) may run. It may not.

## Verdict

| gate | result | key numbers | threshold |
|---|---|---|---|
| F0 integrity | PASS | no source leak / split-pure / chronology / role-disjoint | — |
| **F1 source-free observability** | **FAIL** | probe balacc 0.587; AMI 0.143; purity 0.396; rare prec/rec 0.35/0.22 | 0.75 / 0.35 / 0.65 / 0.50 |
| **F2 oracle recoverability** | **FAIL** | worst-source gain +1.55pp (macro loss 0.08pp OK) | ≥ +3pp |
| F3 nonlinear LoRA viability | PASS | offline-joint worst 0.787; rank≤4; frozen unchanged; grads finite nonzero | worst ≥ 0.65 |

Per-seed arm results (worst-source / macro balanced accuracy on validation):

| seed | ER-FLOP worst | ER-FLOP macro | oracle worst | oracle macro |
|---|---|---|---|---|
| 1234 | 0.787 | 0.846 | 0.813 | 0.843 |
| 5678 | 0.795 | 0.844 | 0.811 | 0.831 |
| 9012 | 0.810 | 0.838 | 0.815 | 0.851 |

## Honest interpretation (two failures corroborate, not noise)

1. **F1**: the frozen BERT-tiny mask-mean-pool signature barely separates the six
   sources in a source-free way (probe 0.587, just above the 1/6 chance floor,
   far below 0.75). Pseudo-cells (SGCR's clustering) therefore cannot align to
   real sources — the precondition for the whole SGCR mechanism. Diagnosis emitted
   by the gate: `unobservable frozen representation (probe weak)`.
2. **F2 (more fundamental)**: even given TRUE source labels, oracle group weighting
   lifts worst-source by only +1.55pp over ER-FLOP (needs +3pp). ER-FLOP is already
   well-balanced (per-source 0.79–0.89, worst ≈ 0.79) — this stream/budget leaves
   almost no recoverable worst-group headroom.

Protocol ("F2 oracle recoverability"): *"If the oracle fails, no pseudo-group
method is expected to rescue this exact stream/budget, so SGCR tuning stops."*
Accordingly **D3 SGCR sweep was NOT run** — deliberately, per protocol, not
because of any error. F0/F3 passing proves the pipeline is correct and LoRA is
genuinely learning; the task simply gives this method no room.

## Predeclared fallback available (protocol "Stage D2")

*"One predeclared representation fallback is permitted after an F1 failure: revise
this protocol to use `prajjwal1/bert-mini`, repeat D0–D2 from scratch, and mark all
BERT-tiny outcomes burned. Source labels may not be used to choose a layer,
concatenate category tokens, train a router, or hand-design clusters."*

The bert-mini fallback addresses F1 (representation observability). It does NOT
obviously address F2 (oracle headroom): if worst-group is already near-saturated
under a stronger backbone too, F2 may still fail. This is a genuine negative-result
risk that must be reported honestly rather than engineered around.

## BERT-tiny outcomes status

If the bert-mini fallback is taken, all BERT-tiny D1/D2 outcomes above are marked
BURNED (development-only, non-citable) and are retained only as an honest record.
