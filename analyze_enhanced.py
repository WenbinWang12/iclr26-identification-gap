import json
import numpy as np
from pathlib import Path

seeds = [1, 2, 3]
all_seeds = []
for s in seeds:
    p = Path(f"C:/Users/lenovo/Desktop/ICLR_Fixed_Budget_Continual_PEFT/enhanced_results_s{s}.json")
    with open(p) as f:
        data = json.load(f)
    # data may be a list of task dicts or a dict with a key
    if isinstance(data, dict):
        # find the list
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    all_seeds.append(data)

print("=" * 80)
print("Phase-2Y Track A ENHANCED Results (epochs=7, PSR budget=32, wide tau grid)")
print("=" * 80)
print(f"Seeds: {len(all_seeds)}")
print(f"Tasks: {len(all_seeds[0])}")
print()

for seed_idx, results in zip(seeds, all_seeds):
    r_raw = np.mean([r["R_raw"] for r in results])
    r_orc = np.mean([r["R_orc"] for r in results])
    r_scg = np.mean([r["R_scg"] for r in results])
    print(f"Seed {seed_idx}:")
    print(f"  R_raw:  {r_raw:.4f}")
    print(f"  R_orc:  {r_orc:.4f}")
    print(f"  R_scg:  {r_scg:.4f}")
    print(f"  SCG vs raw:    {(r_scg - r_raw)*100:+.2f}pp")
    print(f"  SCG vs oracle: {(r_scg - r_orc)*100:+.2f}pp")
    print()

r_raw_all = np.array([np.mean([r["R_raw"] for r in seed]) for seed in all_seeds])
r_orc_all = np.array([np.mean([r["R_orc"] for r in seed]) for seed in all_seeds])
r_scg_all = np.array([np.mean([r["R_scg"] for r in seed]) for seed in all_seeds])

print("=" * 80)
print("Aggregate (mean +/- std across seeds)")
print("=" * 80)
print(f"R_raw:  {r_raw_all.mean():.4f} +/- {r_raw_all.std():.4f}")
print(f"R_orc:  {r_orc_all.mean():.4f} +/- {r_orc_all.std():.4f}")
print(f"R_scg:  {r_scg_all.mean():.4f} +/- {r_scg_all.std():.4f}")
print()

scg_vs_raw = r_scg_all - r_raw_all
scg_vs_orc = r_scg_all - r_orc_all
print(f"SCG vs raw:    {scg_vs_raw.mean()*100:+.2f}pp +/- {scg_vs_raw.std()*100:.2f}pp")
print(f"SCG vs oracle: {scg_vs_orc.mean()*100:+.2f}pp +/- {scg_vs_orc.std()*100:.2f}pp")
print()

print("=" * 80)
print("Success Criteria: SCG > raw + 2pp")
print("=" * 80)
print(f"Result: {scg_vs_raw.mean()*100:+.2f}pp")
if scg_vs_raw.mean() > 0.02:
    print("[SUCCESS]")
elif scg_vs_raw.mean() > 0:
    print("[PARTIAL] positive but < 2pp")
else:
    print("[FAIL]")
print()

# comparison to dev
print("=" * 80)
print("Dev vs Enhanced comparison")
print("=" * 80)
print(f"Dev:      SCG vs raw = +0.92pp")
print(f"Enhanced: SCG vs raw = {scg_vs_raw.mean()*100:+.2f}pp")
print(f"Delta:    {(scg_vs_raw.mean()*100 - 0.92):+.2f}pp")
print()

# per-task
print("=" * 80)
print("Per-task analysis")
print("=" * 80)
n_tasks = len(all_seeds[0])
task_gains = {}
for ti in range(n_tasks):
    name = all_seeds[0][ti]["task"]
    gains = [seed[ti]["R_scg"] - seed[ti]["R_raw"] for seed in all_seeds]
    task_gains[name] = (np.mean(gains), np.min(gains), np.max(gains))

for task, (m, lo, hi) in sorted(task_gains.items(), key=lambda x: x[1][0]):
    flag = "  <- zero" if abs(m) < 1e-6 else ""
    print(f"{task:12s}: mean={m*100:+.2f}pp  [{lo*100:+.2f}, {hi*100:+.2f}]{flag}")
