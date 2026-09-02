# Phase-1 Synthetic Theory Checks

This directory contains deterministic NumPy falsification and regression
checks for the paper's conditional mathematical claims. It is not an FCRA,
LoRA, language-model, replay, or independent-addressability experiment.

Run from this directory after installing NumPy:

```powershell
python -m unittest discover -v
python phase1_theory_checks.py --output-dir outputs\local_primary --run-role primary_frozen
```

The frozen constructions, gates, tolerances, and claim boundary are in
`notes/phase1_theory_protocol.md`. A run writes `per_case.csv`, `traces.json`,
`summary.json`, and `manifest.json`. Source, protocol, and test hashes are
cross-host identity gates; generated output hashes are provenance only.
