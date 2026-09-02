# SGCR-v2 Fresh Confirmatory Result

**Run date:** 2026-08-27  
**Protocol:** `notes/sgcr_v2_confirmatory_protocol.md`  
**Code/manifest:** `experiments/phase2g_targeted_remedy/locked_sha256_v1.txt`  
**Status:** completed exactly once; **primary intersection-union verdict: FAIL**

The fresh panel contained 8 teacher seeds x 8 stream seeds x 3 learner seeds.
Learner seeds were averaged within each of the 64 teacher-stream worlds before
the preregistered inference.  Confidence intervals below are the locked two-way
additive random-effects crossed intervals with `df = 7`; IID paired intervals
are reported only as diagnostics.

## Aggregate performance

| Method | Mean NMSE score | Worst-source tail score | Raw learner tail SD |
|---|---:|---:|---:|
| SGCR-v2 | 0.9631 +/- 0.0077 | 0.7785 +/- 0.0891 | 0.1088 |
| q0 | 0.9671 +/- 0.0066 | 0.5340 +/- 0.1368 | 0.1503 |
| PSR | 0.9689 +/- 0.0064 | 0.7139 +/- 0.1289 | 0.1336 |
| CVaR | 0.9641 +/- 0.0083 | 0.5489 +/- 0.1440 | 0.1733 |

## Locked gates

| Gate | Paired effect | Wins | IID 95% CI | Crossed 95% CI | Decision |
|---|---:|---:|---:|---:|---|
| (a) Mean, SGCR-v2 - PSR; NI margin -0.01 | -0.0059 | 6/64 | [-0.0071, -0.0047] | [-0.0083, -0.0034] | PASS |
| (b) Tail, SGCR-v2 - PSR | +0.0646 | 50/64 | [+0.0390, +0.0902] | [-0.0018, +0.1310] | **FAIL** |
| (c) Tail, SGCR-v2 - CVaR | +0.2297 | 63/64 | [+0.1995, +0.2599] | [+0.1614, +0.2980] | PASS |
| Secondary: tail, SGCR-v2 - q0 | +0.2446 | 63/64 | [+0.2114, +0.2778] | [+0.1670, +0.3222] | PASS |

Gate (b) crossed-interval variance components were 0.0051448 (teacher),
0.0004089 (stream), and 0.0060169 (residual), with standard error 0.0281.
Although its point estimate, win count, and IID interval are positive, its
locked crossed lower bound is below zero.  Because all three primary gates had
to pass, the only valid preregistered primary conclusion is **FAIL**.

## Resource and integrity checks

- Actual replay occupancy: 1354 / 1354 records.
- Persistent numeric payload: 73695 / 73728 float-equivalents.
- Final missing-audit fallbacks: 0 / 192 learner runs.
- Final zero-round fallbacks: 0 / 192 learner runs.
- No optional stopping or outlier deletion was used.

## Claim supported by this run

Within the controlled linear reduced-rank-regression benchmark, SGCR-v2 gives a
large, crossed-CI-positive worst-source improvement over q0 and CVaR while
remaining mean-non-inferior to PSR.  It does **not** yet establish
worst-source superiority over PSR under the preregistered crossed inference,
and it is not evidence on a real nonlinear Transformer/LoRA model.

