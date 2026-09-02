# SGCR-v4 burned 8x8 development result

Date: 2026-08-27 (Asia/Shanghai)

Status: **DEVELOPMENT / BURNED / NOT CONFIRMATORY**.  These 64 teacher-stream
worlds were opened by the earlier SGCR-v2 locked run.  This evaluation used only
learner seed `31770922`.  No result below may be represented as prospective
evidence.

Code:

- `experiments/phase2g_targeted_remedy/explore_sgcr_v4.py`
- `experiments/phase2g_targeted_remedy/selftest_sgcr_v4.py`
- `experiments/phase2g_targeted_remedy/burned_panel_sgcr_v4.py`

The minimal self-test and `py_compile` passed before the panel was run.

## Frozen development method used in this run

- one shared 14-window pool: 3 uniform and 11 input-direction-diverse slots;
- deterministic row-disjoint train/audit views within retained windows;
- persistent aligned activation centers and train-only cumulative arrival
  counts; arrival counts, not retained group sizes, define q0;
- train-only activation candidates (six frequency-to-uniform candidates and 24
  Hedge candidates); no candidate-generation read of audit X/Y;
- a fixed-teacher window-spectral path containing round 0 through round 12 and
  `rho in {0.25,0.50,0.75,1.00}` interpolants from q0 to each MW weight;
- q0 is an unconditional fallback;
- a challenger must satisfy paired example-level mean-delta UCB `<= 0.01`,
  then the feasible model with the lowest worst-cell UCB is selected;
- logical persistent state: `69,408 / 73,728` float-equivalents.

The UCB is a conditional coreset stability regularizer, not a population
certificate: the audit rows come from X-selected retained windows and share
window context with train rows.

## Aggregate results

| Method | Mean | Tail | Mean SD | Tail SD |
|---|---:|---:|---:|---:|
| SGCR-v4 | 0.964297 | 0.801809 | 0.006989 | 0.099622 |
| SGCR-v2 | 0.963179 | 0.772032 | 0.008893 | 0.110153 |
| PSR | 0.969018 | 0.712402 | 0.006496 | 0.125604 |

Paired comparisons across 64 worlds:

| Contrast | Mean delta | Wins | IID 95% CI | Crossed 95% CI |
|---|---:|---:|---:|---:|
| v4 mean - PSR | -0.00472 | 11/64 | [-0.00604,-0.00340] | [-0.00693,-0.00251] |
| v4 tail - PSR | +0.08941 | 57/64 | [+0.06226,+0.11656] | [+0.03933,+0.13948] |
| v4 mean - v2 | +0.00112 | 37/64 | [-0.00058,+0.00282] | [-0.00110,+0.00334] |
| v4 tail - v2 | +0.02978 | 45/64 | [+0.00618,+0.05338] | [-0.01061,+0.07017] |

The full spectral path was selected in only 3/64 worlds; a train-only activation
candidate was selected in 61/64.  Therefore the average improvement cannot be
attributed primarily to the new spectral path.

## Teacher effects

| Teacher | Tail v4-PSR | Tail v4-v2 | Tail wins vs PSR |
|---|---:|---:|---:|
| 39817664 | +0.185806 | +0.008325 | 6/8 |
| 15328524 | +0.031984 | +0.011054 | 8/8 |
| 84645970 | +0.094312 | +0.018344 | 8/8 |
| 74522946 | +0.150011 | +0.096821 | 8/8 |
| 71897575 | +0.086377 | +0.037477 | 8/8 |
| 51205904 | +0.037348 | -0.043672 | 5/8 |
| 77534061 | +0.025239 | +0.009198 | 7/8 |
| 25975093 | +0.104179 | +0.100667 | 7/8 |

## Failure cases and fixed stop verdict

Worst world:

```text
t39817664/s37032991
v4  mean/tail = 0.950502 / 0.499460
v2  mean/tail = 0.933801 / 0.783754
PSR mean/tail = 0.959343 / 0.660865
v4 tail - PSR = -0.161405
selected = activation_hedge_24
```

Four worlds had `v4 tail - PSR < -0.10`: `39817664/37032991`,
`39817664/33758660`, `51205904/78983795`, and `25975093/26352979`.
All used activation-path candidates.  Thus the pseudo-cell audit can prefer a
model that is substantially worse on the hidden worst source even after the
paired mean-UCB constraint.

Fixed stop rules:

- mean v4-PSR `>= -0.01`: **PASS** (`-0.00472`);
- tail v4 `>=` tail v2: **PASS** (`+0.02978`);
- no world with tail delta `< -0.10`: **FAIL** (worst `-0.16141`);
- aggregate tail v4-PSR `> 0`: **PASS** (`+0.08941`).

Overall stop verdict: **FAIL safety / do not advance this implementation to a
fresh confirmatory panel**.  It is a positive average strong-baseline result on
burned data, but not a reliable Pareto-safe method.  The predeclared condition
that would kill the line for nonpositive average tail was not triggered;
nevertheless, the catastrophic-world condition was triggered, so no additional
linear-benchmark hyperparameter tuning should be justified from this panel.

