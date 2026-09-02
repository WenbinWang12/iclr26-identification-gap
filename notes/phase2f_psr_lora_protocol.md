# Phase-2f PSR-LoRA — held-out kill-test protocol (post-hoc transcription)

**Status: POST-HOC TRANSCRIPTION.** This note was written on **2026-08-27**,
*after* the held-out run, to give the paper, `README.md`, and the ledger (C17) a
single human-readable pointer to the pre-specification that actually lives in
committed code. It does **not** predate the run and must not be read as a
timestamped pre-registration. The real, machine-checkable pre-specification is
the frozen source of `preregister_heldout.py` (the full protocol is stated
verbatim in its module docstring) plus the frozen `DEFAULT` config in
`psr_lora.py`. This note transcribes those; where it and the code disagree, the
**code is authoritative**.

## Honesty note on "hash-recorded" (read this before citing the protocol)

The discipline claim is that the method was **fixed before the numbers**. What is
durably verifiable for that claim:

1. `preregister_heldout.py` is committed source whose docstring fixes the arena,
   the 20 held-out worlds, the three kill criteria, the statistic, and the
   "run once, report whatever comes out" rule.
2. `psr_lora.py`'s `DEFAULT` freezes the exact method config
   (`buf_u=4, buf_g=12, coverage=True, reweight=True, R=4`) that
   `preregister_heldout._cfg()` pins.
3. The held-out seeds (teachers `20280001..5` × streams `8001..4`) are disjoint
   from the development family and from the earlier burned test set.

What is **NOT** durably verifiable: there is **no surviving timestamped log of
the source hashes taken before the run**. The only `sha256` in the code is
`harness.py`'s runtime *label-blindness* hash (which proves oracle-freeness, not
pre-run timing). The SHA-256 table below was therefore computed **at transcription
time (2026-08-27)**, not before the run; it anchors the artifacts *going forward*
but does not by itself prove the method predated the numbers. That claim rests on
the committed code above, not on a hash log. (The paper's Reproducibility
Statement must be worded accordingly — "the method, criterion, and seeds are
fixed in committed source," not "hashes recorded before running proved to be
verifiable.")

## Frozen artifacts — SHA-256 (computed 2026-08-27, transcription time)

| File | SHA-256 |
|---|---|
| `psr_lora.py` (method core; DEFAULT freezes 4/12 coverage) | `24ad55c3c2a6c155fd2009c3d6fe4d7e648dfa7ef1db5871fab612034d567f58` |
| `preregister_heldout.py` (held-out kill test) | `9fd78bd721053f14cf3822c1b6ef822b1ee5808941fe569f6551af74e8e2e6c3` |
| `outputs_heldout.txt` (frozen result record) | `b11c352875235ab0745422bcd6e168845f0d4dc502300314da4a6be67b547da0` |
| `offfloor_heldout.py` (off-floor no-harm/graceful check) | `4bd31e8246cf17ceffc63b720826edf66b06c2d3a6bac7810bfb1c568135aed0` |
| `outputs_offfloor_heldout.txt` (frozen off-floor record) | `beebba5b7891e00944b2536027061d954b84f3ce61b8f7167769c05e4e770a49` |
| `baselines.py` (real-CVaR / replay / A-GEM / EWC suite) | `1beb9edcd02916f6d3cae0b84b8b403aa121627386b5093a79dcde47fa7bf408` |
| `benchmark_v4.py` (teacher / stream / offline frontier) | `04bcb94dc17cde7a68b1749d678fb2c8b512e6116767292602800ae878ad1a2c` |
| `harness.py` (metrics + label-blindness hash) | `de1bbd731816630b7bfc6463f5c67225e1586134afbb60969c38aab2d4f05ce8` |

Re-check any time with, from `experiments/phase2f_psr_lora/`:
`sha256sum psr_lora.py preregister_heldout.py outputs_heldout.txt ...`

## Method (frozen)

Coverage-aware PSR-LoRA: one fixed-rank-`R=4` adapter `M=BA`. A single bounded
buffer of total budget `B=16` is split for **coverage**: `buf_u=4` uniform
(Vitter) reservoir + `buf_g=12` input-direction-diverse buffer (`_div_offer`
keeps windows whose top right singular direction of the centered inputs is most
mutually orthogonal, dropping the most redundant). The union feeds a minimax
multiplicative-weights loop that re-solves the closed-form weighted rank-`R`
reduced-rank regression each round and keeps the iterate with the smallest
**buffered** worst-window loss (round 0 = uniform solve ⇒ never worse than
uniform on the buffered worst group by construction). Oracle-free: a function of
`(X,Y)` only; the diversity score reads inputs, never labels. `coverage=False`
recovers the pre-remedy single uniform reservoir (the coverage ablation);
`reweight=False` recovers the uniform-weight buffer-RRR (the reweighting
ablation).

## Arena and worlds (frozen)

- Regime: `capacity_limited` — one global linear teacher `T*`, correlated source
  directions, `R=4<K`. The only regime with a genuine rank-`R` floor.
- Held-out worlds: teachers `[20280001, 20280002, 20280003, 20280004, 20280005]`
  × streams `[8001, 8002, 8003, 8004]` = 20 worlds, crossed 5×4.
- Per-arm buffer-draw RNG seed `RUN_SEED = 7`, fixed for all arms.

## Kill criterion (frozen; all three required to CONFIRM)

Decision object = the **cluster-robust two-way crossed 95% CI**
(`compare._paired_ci_crossed`), conservative `df = min(n_t, n_s) − 1 = 3`. The IID
CI is printed for transparency only and is never the decision object.

- **(a) non-inferior mean** vs the strongest-mean baseline (excess-CVaR here),
  margin −0.01: pass if paired mean ≥ −0.01.
- **(b) strictly higher tail** vs excess-CVaR: pass if the cluster CI on the
  per-world tail difference excludes 0.
- **(c) reweighting given coverage** vs `buffer_rrr_uniform` (same coverage
  buffer, uniform weights): pass if the cluster CI excludes 0.

Additionally reported, NOT a gate: the coverage-alone contribution vs
`psr_no_coverage`; and off-floor no-harm (compressible) / graceful (conflicting).

## Result (frozen — `outputs_heldout.txt`, verdict CONFIRM)

- psr_lora: mean **0.970±0.005**, tail **0.792±0.062** (tail/offline **0.94**).
  Offline oracle tail 0.839.
- (a) +0.0027, cluster 95% CI [−0.0026, +0.0079] — PASS (within −0.01 margin).
- (b) +0.1782, wins 19/20, cluster 95% CI [+0.0696, +0.2869] — PASS.
- (c) +0.0945, wins 20/20, cluster 95% CI [+0.0477, +0.1413] — PASS.
- [info] coverage alone +0.0918, wins 18/20, cluster 95% CI
  **[−0.0085, +0.1922] straddles 0** — reported as a super-additive component,
  not an independently significant effect.
- Off-floor (`outputs_offfloor_heldout.txt`, same frozen 4/12 config):
  compressible 0.992/0.988 (ties coverage-only 0.992/0.988, no harm); conflicting
  psr tail 0.237 vs coverage-only 0.235 vs excess-CVaR 0.189 (graceful; the small
  edge over excess-CVaR is not claimed as a rescue).

## Seed ledger (discipline — do not violate)

- **DEV family** (design/selection ONLY): teachers `[20260701, 20260710, 20260711,
  20260712, 20260713]` × streams `[6060, 6061]`.
- **BURNED test set #1** (v1/v2 kill tests — spent; also used to observe the (b)
  failure and read source labels for the H1 diagnosis, so any re-run on these is
  POST-HOC, never independent): teachers `[20270001..20270005]` × streams
  `[7001..7004]`.
- **BURNED test set #2** (this held-out confirmation — now spent): teachers
  `[20280001..20280005]` × streams `[8001..8004]`.
- **AVAILABLE for any future single frozen test**: seeds disjoint from ALL of the
  above (e.g. teachers `20290001+`, streams `9001+`). Pick once, freeze, run once.

## Caveats to carry into every citation (from the adversarial audit)

1. The 2027-burned-seed cross-check that also passes (b) is **POST-HOC**, not
   independent confirmation — those seeds motivated the coverage remedy. Cite it
   only as a post-hoc robustness check, never as headline.
2. Criterion (c) is **draw-sensitive**: passes on this held-out draw, fails on the
   burned draw (whose teachers are ~2× more homogeneous). "All three pass" is
   partly a property of this draw.
3. The **coverage buffer** — the only genuinely new component — is not CI-significant
   in its own isolation (item [info] above). Report as super-additive with
   reweighting, modest and somewhat draw-sensitive.
4. A window can sit in both U and G (overlap 0–2/world); double-counted in the
   pooled teacher, but it affects psr and `buffer_rrr_uniform` identically, so it
   does not bias (c). Minor, disclosed.

## Superseded predecessor (provenance only)

`compare.py` / `outputs_compare_v2.txt` is the **pre-coverage uniform-16** record
(tail 0.678; (b)/(c) FAIL on the cluster CI). It is kept for provenance and is
NOT the method's result. `compare._cfg()` is pinned `coverage=False` so it
reproduces that superseded record byte-identically; see its SUPERSEDED header.
