# Phase-0 Diagnostic Harness

This directory contains exact, small probes for the fixed-budget continual
PEFT question. It is intentionally independent of LoRA, a language model, and
GPU libraries. The compatible matrix-sensing stream tests online interference
with a common rank-R solution; the full-observation streams test conflict,
cancellation, and rank insufficiency.

Run from this directory after installing NumPy:

```powershell
python -m unittest discover -v
python phase0_diagnostic.py --output-dir outputs\local_20 --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 --run-role primary_frozen_20_seed
python phase0_diagnostic.py --output-dir outputs\local_100 --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99 --run-role sensitivity_only_100_seed
```

The output contains `per_prefix.csv`, `summary.json`, and `manifest.json` with
source/protocol/test hashes and a generated-input fingerprint, numerical
settings, environment, per-seed
traces, independent and paired seed bootstrap summaries, and explicit gate
statuses. The compatible replay field is deliberately marked
non-independent because it equals the exact rank-R offline oracle. The
protocol and frozen gates are in `notes/phase0_pilot_protocol.md`.
