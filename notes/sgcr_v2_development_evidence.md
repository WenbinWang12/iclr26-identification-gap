# SGCR-v2 development evidence (not confirmatory)

Date: 2026-08-27 (Asia/Shanghai)

Status: all panels in this note have been opened and are permanently marked as
development/burned. None of the numbers below may be presented as a prospective
held-out result.

## Why v1 was rejected

The first final-buffer gradient-clustering GCDR failed its locally locked panel:

- GCDR mean/tail: `0.9680 / 0.6707`;
- PSR mean/tail: `0.9722 / 0.7148`;
- tail delta to PSR: `-0.0441`, wins `4/20`, crossed 95% CI
  `[-0.1422,+0.0541]`.

The next streaming prototype was also rejected. On the same burned panel it had
mean/tail `0.9695 / 0.6762`, versus PSR `0.9722 / 0.7148`; tail delta was
`-0.0386`, with only `3/20` wins. Independent code review found unequal
training pools, stale warmup memory, fixed `G=8` matching planted `K`, and a
final-only fit.

## Failure-driven SGCR-v2 changes

`experiments/phase2g_targeted_remedy/explore_sgcr_v2.py` implements:

1. normalized frozen projective-activation signatures `vec(xx^T)`, after the
   gradient-only signatures were shown to have low behavior purity;
2. a structural resolution cap `K_max=2R` rather than reading `cfg["K"]`;
3. permanent, disjoint train/audit roles;
4. per-cell minimum priority reservoirs plus a shared overflow;
5. a same-training-pool frequency baseline and robust candidates;
6. train-only cell coherence filtering for diffuse catch-all cells;
7. a small-step (`eta=0.1`) 24-step Hedge/Pareto path, motivated by an observed
   discontinuous rank switch at `eta=0.7`;
8. a one-per-cent empirical mean-NMSE gate and no final train+audit refit;
9. model refresh/deployment every 20 windows;
10. explicit persistent numeric-state accounting, including centers, raw
    examples, priority keys, ids, labels, and counters.

## Original development panel

Teachers `20260701, 20260710..20260713`, streams `6060,6061`, learner seed `7`:

| Method | Mean | Tail | Tail delta vs PSR | Wins |
|---|---:|---:|---:|---:|
| SGCR-v2 | 0.9711 | 0.8807 | +0.0539 | 10/10 |
| PSR | 0.9748 | 0.8268 | -- | -- |

This panel was used to choose the small Pareto step and is development only.

## Burned 5x4x3 stress panel

Teachers `31415901..31415905`, streams `16180331..16180334`, learner seeds
`7,17,29`. Learner seeds are averaged inside each of the 20 crossed worlds.

| Method | Mean | Tail | raw learner-seed tail SD |
|---|---:|---:|---:|
| SGCR-v2 | 0.9668 | 0.7932 | 0.1350 |
| same-pool frequency q0 | 0.9715 | 0.5754 | 0.1948 |
| PSR | 0.9722 | 0.7148 | 0.1706 |
| excess-CVaR | 0.9688 | 0.6271 | 0.1789 |

Descriptive prospective-style tests (still post-hoc):

- mean non-inferiority SGCR-v2 minus PSR: `-0.0054`; crossed 95% CI
  `[-0.0091,-0.0018]`, passing the `-0.01` margin;
- tail superiority SGCR-v2 minus PSR: `+0.0784`, wins `15/20`; crossed 95% CI
  `[-0.0130,+0.1699]`, **not passing** strict superiority;
- tail superiority SGCR-v2 minus excess-CVaR: `+0.1661`, wins `20/20`;
  crossed 95% CI `[+0.0652,+0.2670]`, passing strict superiority;
- tail mechanism SGCR-v2 minus the same-pool frequency q0: `+0.2178`, wins
  `20/20`; crossed 95% CI `[+0.0841,+0.3516]`, passing strict superiority;
- actual persistent raw occupancy: mean/min/max `1354/1354/1354` entries.

Interpretation: SGCR-v2 is the first implementation in this project with a
large, stable average positive tail effect, mean non-inferiority, and a positive
crossed interval against excess-CVaR. It has **not yet established superiority
to PSR**, because that crossed interval still includes zero and every design
choice was made after this panel became development data.

## Rejected follow-up variants

- `eta=0.7`: discontinuous low-rank switches, large teacher-dependent failures;
- final refit on train+audit: reduced the original-dev result to `8/10` wins and
  enlarged the mean cost;
- 75/25 train/audit: mean improved but tail fell to `+0.0336`, `8/10` wins;
- 2/3--1/3 train/audit: one `-0.199` tail failure, `9/10` wins.

The frozen candidate therefore retains the 50/50 split and does not refit on
audit. A genuinely unused seed panel is required before any confirmatory claim.
