# Phase-2Y Track A Enhanced Hyperparameters

## Changes from Development Run

| Parameter | Development | Enhanced | Rationale |
|-----------|-------------|----------|-----------|
| `--epochs` | 3 | **7** | Insufficient training (74 steps for 15 tasks) |
| `--psr-budget` | 16 | **32** | Double rehearsal coverage for offset stability |
| `--scg-tau-grid` | 6 values (0.8-2.0) | **10 values (0.4-3.0)** | Wider search space for threshold tuning |
| `--time-limit` | 4h | **8h** | Account for 2.3× longer training |

## Expected Impact

**Development results (epochs=3, budget=16):**
- SCG vs raw: +0.92pp ± 0.27pp
- Status: PARTIAL (below +2pp threshold)
- Zero-gain tasks: 5/12 (COPA, QQP, BoolQA, IMDB, MultiRC)

**Expected enhanced results:**
- More training epochs → better LoRA adaptation
- Double PSR budget → stabler offset estimation
- Wider tau grid → better per-task threshold fitting
- Target: SCG vs raw > +2pp

## Execution

```bash
# On Babel cluster
cd /home/pengq/iclr26
sbatch experiments/phase2y_composite/sweep_two_stage_enhanced.sh
```

Seeds 1-3, estimated 6-8 hours per seed.
