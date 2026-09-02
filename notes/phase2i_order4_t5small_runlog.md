# T5-small official Long-CL Order-4 five-arm run log

Status: **20/class development panel complete; headroom signal is negative**  
Lock date: 2026-08-28 (Asia/Shanghai)

## Scope

This is the Phase-2I feasibility probe, not a reproduction of the published
T5-large O-LoRA numbers and not yet an LLM result.  It uses the official
O-LoRA data, task order, verbalizers, instructions, and normalized exact-match
evaluator, while replacing the backbone with `google-t5/t5-small`.

Pinned sources:

- O-LoRA commit: `07117e1fc4a5f5ad9308a815a42cee8f46502dc8`
- T5-small revision: `df1b051c49625cf57a3d0d8d3863ed4d13564fe4`

Order-4 is locked as:

`MNLI -> CB -> WiC -> COPA -> QQP -> BoolQA -> RTE -> IMDB -> Yelp -> Amazon -> SST-2 -> DBpedia -> AG News -> MultiRC -> Yahoo`.

The released O-LoRA `scripts/long.sh` is not this order.  The runner therefore
constructs Order-4 and every seen-task evaluation prefix from the locked
manifest rather than invoking that script.

## Data-integrity decision

For every one of the 15 datasets in the pinned O-LoRA repository, `dev.json`
and `test.json` resolve to the same Git blob.  Consequently:

- repository `dev` is never used for model selection, weighting, rank
  allocation, early stopping, or the empirical audit gate;
- update/risk/audit roles are assigned deterministically inside `train.json`;
- audit examples never enter a gradient batch;
- the official `test.json` is opened only by the evaluator.

The prompt intentionally includes the official public `Task:` and `Dataset:`
fields.  The group-free controller is forbidden from reading task/dataset IDs,
but this benchmark is not described as task-identity-free.

## Locked five arms

1. `fixed_er`: fixed uniform rank, single cumulative LoRA, uniform replay.
2. `mean_rank_er`: mean-oriented global rank exchange, uniform replay.
3. `oracle_dro_rank`: training-task-ID anchored group DRO and risk-aware rank.
4. `cvar_uniform`: group-free anchored CVaR and uniform layer rank.
5. `cvar_rank`: group-free anchored CVaR and risk-aware global rank.

All arms adapt the 36 T5 query/value projections.  The deployed budget is
exactly 144 rank-one atoms (rank-4 equivalent), or 147,456 trainable payload
scalars.  The training bank has at most eight slots per projection.  Scaling is
fixed relative to the reference rank and must not change when atoms move.

The first implementation uses a deterministic combined-gradient/norm proxy for
one-atom rank exchanges.  It must be reported only as a Phase-2I proxy, not as
the proposed shortlist--probe algorithm and not silently equated with AdaLoRA,
SLAO, or another paper's released method.  In particular, `fixed_er` is only a
fixed-rank capacity reference; it is not the published SLAO algorithm.

## Execution ladder forced by local hardware

The host has an i5-10210U, 16 GiB RAM, and CPU-only PyTorch.  Measured
T5-small throughput at source length 128 / target length 16 is about 2.26
training examples/s and 3.57 generated examples/s.  The following ladder is
therefore fixed before looking at results:

1. full-order wiring smoke: tiny per-class cap and tiny sealed-test cap;
2. full-order development panel: 20/class, one seed, 50 test examples/task,
   diagonal plus final evaluation;
3. decision-grade acquisition/headroom run: 200/class, first one seed and then
   three seeds only if the one-seed run shows usable acquisition.

Stages 1 and 2 are pipeline evidence only and cannot pass the preregistered
GO/KILL gate.  A full 200/class five-arm seed is estimated at 18--30 CPU hours;
three seeds at 54--90 hours.  Moving stage 3 to a GPU changes hardware only,
not the locked manifests or method configuration.

## Result ledger

### Full-order wiring panel (completed)

Artifact root:
`D:\phase2i_runs\t5small_official_order4_full15_fivearm_wiring_seed42_20260828`

Configuration: seed 42, all 15 Order-4 tasks, five arms, cap 4/class,
risk 1/class, audit 1/class, sealed evaluation 4/task, batch 2, source length
64, target/generation length 8, one epoch/task.  This is explicitly a wiring
smoke and **cannot** be used for GO/KILL or a positive method claim: a single
test example is 25 accuracy points.

All five arms wrote `COMPLETED`, 15 stage checkpoints, predictions, anchor and
buffer ledgers, and exact payload audits.  Every canonical T5-small stage used
144 active atoms = 147,456 active scalars = 589,824 FP32 bytes.  The two
uniform-rank arms stayed at rank 4 in all 36 projections; adaptive proxy arms
ended with ranks in `[1, 8]` while preserving sum-rank 144.

| Arm | Final mean EM | Mean BWT | Lower-tail normalized retention | Eligible risk anchors | Accepted exchanges |
|---|---:|---:|---:|---:|---:|
| `fixed_er` | 0.3833 | 0.0833 | 0.0000 | 44/60 | 0 |
| `mean_rank_er` | 0.3667 | 0.0833 | 0.0000 | 45/60 | 15 |
| `oracle_dro_rank` | 0.3833 | 0.1500 | -1.6667 | 45/60 | 15 |
| `cvar_uniform` | 0.3667 | 0.0667 | 0.0000 | 43/60 | 0 |
| `cvar_rank` | 0.4500 | 0.1167 | 0.3333 | 40/60 | 15 |

The apparent `cvar_rank` advantage is not evidence: the evaluation grid is
too coarse, and the task-aware oracle is worse on the same noisy tail metric.
Its only valid interpretation is that all five intended code paths diverge
while satisfying the same payload.

MNLI's pinned `test.json` contains 131 rows with the placeholder target `-`.
The sealed evaluator excludes only those rows, never maps them to a class, and
records exclusion SHA-256
`39c29947f8891a3a3661433d2477dada3ab91aa84883c47cf385eeab82458a2e`.
No other task has an excluded test row.  The train loader remains strict.

Independent audit reproduced the full fixed arm exactly: initial fingerprint,
base scores, all diagonal/final scores, and 15/15 stage `adapter.pt` SHA-256
hashes matched.  The test suite contains 40 passing checks.

### Full-order 20/class development panel (completed)

Artifact root:
`D:\phase2i_runs\t5small_official_order4_full15_fivearm_dev20_seed42_20260828`

Configuration: seed 42, all 15 Order-4 tasks, all five arms, cap 20/class,
risk 2/class, audit 2/class, deterministic class-balanced sealed evaluation
50/task, batch/replay batch 4, source length 128, target/generation length 16,
learning rate 1e-3, one epoch/task.  All arms processed 1,076 current and 1,028
replay examples.  This remains a development probe rather than a
decision-grade or paper result.

| Arm | Final mean EM | Mean BWT | Min retention | Worst-2 retention | Lower-tail-20 retention | Worst-3 raw BWT | Wall min | Accepted exchanges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_er` | 0.4640 | +0.0080 | -2.0000 | -1.6000 | -1.0667 | -0.0800 | 29.7 | 0 |
| `mean_rank_er` | **0.4813** | **+0.0133** | -2.0000 | -1.8000 | -1.3333 | **-0.0667** | 30.8 | 15 |
| `oracle_dro_rank` | 0.4560 | -0.0053 | -2.0000 | **-1.2000** | **-0.9333** | -0.0800 | 22.7 | 15 |
| `cvar_uniform` | 0.4707 | +0.0027 | -2.4000 | -2.2000 | -1.6000 | -0.1667 | 22.1 | 0 |
| `cvar_rank` | 0.4493 | -0.0187 | -2.4000 | -2.2000 | -1.4667 | -0.1867 | 22.1 | 15 |

Integrity checks found 75/75 stage payloads exact: 144 active atoms,
147,456 active trainable scalars, and 589,824 FP32 active-weight bytes.  The
uniform arms remained rank 4 in every projection; adaptive arms ended in
`[1, 8]` with sum-rank 144.  All five arm sentinels and the panel sentinel are
present.  The post-run test suite is `40 passed`.  The panel manifest SHA-256
is `0b350e7953470a8123a8ee0ce01b1f9d1c640ac5784f09ddae4cd8e6b00a85d4`;
the summary SHA-256 is
`34e3468dd3d6c060204fbc13548364639ff4883a25581fc25342659c5037ed48`.

#### Acquisition audit and decision

The pre-specified acquisition prerequisite fails before any tail comparison
can be interpreted.  Every arm has at least two primary tasks below +5 points:

- `fixed_er`: QQP -10 points, BoolQA -6 points;
- `mean_rank_er`: QQP -10, BoolQA -6, MultiRC 0;
- `oracle_dro_rank`: QQP -8, BoolQA 0, MultiRC -2;
- `cvar_uniform`: QQP -6, BoolQA -10, MultiRC -2;
- `cvar_rank`: QQP -10, BoolQA -10, MultiRC -2.

QQP and BoolQA have a 0.60 frozen-base EM on the balanced 50-example panels.
Most trained arms collapse toward an almost constant `True` prediction and
score approximately 0.50.  Their negative acquisition is then divided by the
fixed 0.05 denominator floor, producing -2.0/-2.4 retention values.  These
never-acquired tasks dominate the reported lower tail.  Consequently, the
oracle's apparent +0.40 normalized-tail advantage over the strongest-mean
proxy is not usable headroom: it loses 2.53 raw mean points, strict minimum is
unchanged, and its worst-three raw BWT (-0.08) does not beat fixed ER (-0.08)
or the mean-rank proxy (-0.0667).

The group-free result is negative.  Relative to fixed ER, `cvar_uniform` and
`cvar_rank` lower the normalized tail by 0.5333 and 0.4000 respectively.  Rank
exchange raises the three-task normalized tail by 0.1333 over uniform CVaR,
but leaves strict minimum and worst-two unchanged and loses 2.13 mean points.
The difference is entirely driven by MultiRC, another task that failed the
acquisition prerequisite.  It is therefore not evidence for dynamic rank.

As a diagnostic only, recomputing the lower 20% after excluding tasks below
the locked +5-point acquisition threshold gives 0.5013 (`fixed_er`), 0.8412
(`mean_rank_er`), 0.8395 (`oracle_dro_rank`), 0.6427 (`cvar_uniform`), and
0.6560 (`cvar_rank`).  On acquired tasks, the oracle is therefore -0.0017
versus the mean-rank proxy, while rank exchange adds only +0.0133 over uniform
CVaR, below the locked +0.02 contribution threshold.  This post-hoc diagnostic
does not replace the pre-specified all-task metric; it identifies why that
metric failed.

The locked GO/KILL gate is not formally evaluated because this is only one
20/class seed and lacks update-step-matched, token-FLOP-matched, SLAO+ER, and
released mean-adaptive controls.  It is nevertheless already ineligible to
pass criterion 1, and the observed oracle/group-free patterns do not justify
spending local CPU time on the predeclared 200/class five-arm run unchanged.
All arms used 271 optimizer steps, but their logged token totals differ
(228,451; 228,451; 215,477; 228,938; 226,216 in table order), so no
FLOP-matched claim is made.

### Interpretation limits of the current runner

- `fixed_er` is fixed-rank single-LoRA + ER, not a reproduction of SLAO.
- `mean_rank_er` is the local gradient/norm proxy, not AdaLoRA or OA-Adapter.
- Rank exchange is a deterministic proxy, not the proposed measured
  shortlist--probe mechanism.
- The current CVaR objective uses a fixed penalty mixture; mean/current dual
  constraints, EMA/two-hit persistence, and the audit gate are not yet
  implemented.
- Training tokens and wall time are logged, but the extra full-buffer risk
  forward is not yet FLOP-matched.
- The runner has stage artifacts but no crash-resume implementation.

The next experiment should be an acquisition-only rescue probe on QQP,
BoolQA, and MultiRC: no replay or risk controller, several epochs/learning
rates, and full or substantially larger sealed evaluation.  Only a setting
that learns every task by at least five points should be frozen and rerun as a
five-arm headroom panel.  If T5-small cannot clear that prerequisite under a
reasonable per-task budget, kill T5-small for method selection rather than
interpreting its normalized-retention tail.
