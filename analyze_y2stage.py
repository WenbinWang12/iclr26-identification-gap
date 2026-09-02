import json
import numpy as np

# Manually paste the results
results_s1 = [
  {"task": "MNLI", "R_raw": 0.6145833333333334, "R_orc": 0.7916666666666666, "R_scg": 0.6041666666666666},
  {"task": "WiC", "R_raw": 0.515625, "R_orc": 0.609375, "R_scg": 0.546875},
  {"task": "COPA", "R_raw": 0.5, "R_orc": 0.5625, "R_scg": 0.5},
  {"task": "QQP", "R_raw": 0.703125, "R_orc": 0.875, "R_scg": 0.703125},
  {"task": "BoolQA", "R_raw": 0.78125, "R_orc": 0.8125, "R_scg": 0.78125},
  {"task": "RTE", "R_raw": 0.8125, "R_orc": 0.875, "R_scg": 0.84375},
  {"task": "IMDB", "R_raw": 1.0, "R_orc": 1.0, "R_scg": 1.0},
  {"task": "SST-2", "R_raw": 0.96875, "R_orc": 0.96875, "R_scg": 0.96875},
  {"task": "DBpedia", "R_raw": 0.8839285714285714, "R_orc": 0.9866071428571429, "R_scg": 0.8861607142857143},
  {"task": "AGNews", "R_raw": 0.8515625, "R_orc": 0.921875, "R_scg": 0.859375},
  {"task": "MultiRC", "R_raw": 0.640625, "R_orc": 0.75, "R_scg": 0.640625},
  {"task": "Yahoo", "R_raw": 0.75625, "R_orc": 0.784375, "R_scg": 0.759375}
]

results_s2 = [
  {"task": "MNLI", "R_raw": 0.5416666666666666, "R_orc": 0.7083333333333334, "R_scg": 0.5729166666666666},
  {"task": "WiC", "R_raw": 0.5, "R_orc": 0.640625, "R_scg": 0.53125},
  {"task": "COPA", "R_raw": 0.5, "R_orc": 0.625, "R_scg": 0.5},
  {"task": "QQP", "R_raw": 0.71875, "R_orc": 0.84375, "R_scg": 0.71875},
  {"task": "BoolQA", "R_raw": 0.828125, "R_orc": 0.859375, "R_scg": 0.828125},
  {"task": "RTE", "R_raw": 0.609375, "R_orc": 0.875, "R_scg": 0.671875},
  {"task": "IMDB", "R_raw": 0.90625, "R_orc": 0.9375, "R_scg": 0.90625},
  {"task": "SST-2", "R_raw": 0.90625, "R_orc": 0.96875, "R_scg": 0.90625},
  {"task": "DBpedia", "R_raw": 0.9285714285714286, "R_orc": 0.9910714285714286, "R_scg": 0.9330357142857143},
  {"task": "AGNews", "R_raw": 0.8046875, "R_orc": 0.9375, "R_scg": 0.8125},
  {"task": "MultiRC", "R_raw": 0.75, "R_orc": 0.75, "R_scg": 0.75},
  {"task": "Yahoo", "R_raw": 0.709375, "R_orc": 0.746875, "R_scg": 0.709375}
]

results_s3 = [
  {"task": "MNLI", "R_raw": 0.6458333333333334, "R_orc": 0.7916666666666666, "R_scg": 0.6770833333333334},
  {"task": "WiC", "R_raw": 0.5, "R_orc": 0.671875, "R_scg": 0.5},
  {"task": "COPA", "R_raw": 0.5, "R_orc": 0.6875, "R_scg": 0.5},
  {"task": "QQP", "R_raw": 0.796875, "R_orc": 0.828125, "R_scg": 0.796875},
  {"task": "BoolQA", "R_raw": 0.671875, "R_orc": 0.734375, "R_scg": 0.671875},
  {"task": "RTE", "R_raw": 0.71875, "R_orc": 0.84375, "R_scg": 0.765625},
  {"task": "IMDB", "R_raw": 0.875, "R_orc": 0.890625, "R_scg": 0.875},
  {"task": "SST-2", "R_raw": 0.90625, "R_orc": 0.984375, "R_scg": 0.9375},
  {"task": "DBpedia", "R_raw": 0.9598214285714286, "R_orc": 0.9910714285714286, "R_scg": 0.9642857142857143},
  {"task": "AGNews", "R_raw": 0.84375, "R_orc": 0.9296875, "R_scg": 0.859375},
  {"task": "MultiRC", "R_raw": 0.609375, "R_orc": 0.703125, "R_scg": 0.609375},
  {"task": "Yahoo", "R_raw": 0.721875, "R_orc": 0.7625, "R_scg": 0.721875}
]

all_seeds = [results_s1, results_s2, results_s3]

print("=" * 80)
print("Phase-2Y Track A Results (PSR + SCG)")
print("=" * 80)
print(f"Seeds: {len(all_seeds)}")
print(f"Tasks: {len(all_seeds[0])}")
print()

# Per-seed statistics
for seed_idx, results in enumerate(all_seeds, 1):
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

# Aggregate statistics
r_raw_all = [np.mean([r["R_raw"] for r in seed]) for seed in all_seeds]
r_orc_all = [np.mean([r["R_orc"] for r in seed]) for seed in all_seeds]
r_scg_all = [np.mean([r["R_scg"] for r in seed]) for seed in all_seeds]

print("=" * 80)
print("Aggregate (mean ± std across seeds)")
print("=" * 80)
print(f"R_raw:  {np.mean(r_raw_all):.4f} ± {np.std(r_raw_all):.4f}")
print(f"R_orc:  {np.mean(r_orc_all):.4f} ± {np.std(r_orc_all):.4f}")
print(f"R_scg:  {np.mean(r_scg_all):.4f} ± {np.std(r_scg_all):.4f}")
print()

scg_vs_raw_mean = np.mean(np.array(r_scg_all) - np.array(r_raw_all))
scg_vs_orc_mean = np.mean(np.array(r_scg_all) - np.array(r_orc_all))

print(f"SCG vs raw:    {scg_vs_raw_mean*100:+.2f}pp ± {np.std(np.array(r_scg_all) - np.array(r_raw_all))*100:.2f}pp")
print(f"SCG vs oracle: {scg_vs_orc_mean*100:+.2f}pp ± {np.std(np.array(r_scg_all) - np.array(r_orc_all))*100:.2f}pp")
print()

# Success criteria
print("=" * 80)
print("Success Criteria Check")
print("=" * 80)
print(f"Criterion: SCG > raw + 2pp")
print(f"Result: {scg_vs_raw_mean*100:+.2f}pp")
if scg_vs_raw_mean > 0.02:
    print("[SUCCESS]")
elif scg_vs_raw_mean > 0:
    print("[PARTIAL] positive but < 2pp threshold")
else:
    print("[FAIL] negative or zero")
print()

# Per-task breakdown for worst cases
print("=" * 80)
print("Per-task analysis (worst cases)")
print("=" * 80)

task_gains = {}
for task_idx in range(len(all_seeds[0])):
    task_name = all_seeds[0][task_idx]["task"]
    gains = []
    for seed in all_seeds:
        gain = seed[task_idx]["R_scg"] - seed[task_idx]["R_raw"]
        gains.append(gain)
    task_gains[task_name] = (np.mean(gains), np.min(gains))

# Sort by mean gain
sorted_tasks = sorted(task_gains.items(), key=lambda x: x[1][0])

for task, (mean_gain, min_gain) in sorted_tasks[:5]:
    print(f"{task:12s}: mean={mean_gain*100:+.2f}pp, worst={min_gain*100:+.2f}pp")
