# Phase-1b: Curvature-Weighted Expressivity Geometry

Deterministic NumPy checks for the curvature-weighted Grassmann measure of
`appendix/proofs.tex` (Appendix, *Expressive Capacity on a Curvature-Weighted
Manifold*). These are proposition regression/sanity checks, **not** neural or
benchmark evidence.

## Status (pre-D1)

- `check_expressivity_geometry`: Prop E1/E2/E3 sanity on controlled subspaces
  and known spectra. Metric zero on identical subspaces, positive on
  equal-rank-distinct subspaces, scale-invariant; effective capacity reduces
  to the spectral tail at the rank-R point; whitened span nests under PSD sum.
- The forgetting-prediction check `check_expressivity_predicts_forgetting`
  (the D1 result that gates any citation) is **not yet implemented**.
- No protocol file is frozen yet; `manifest.protocol_sha256` reads
  `PROTOCOL_NOT_FROZEN_YET` until a `notes/phase1b_expressivity_protocol.md`
  is frozen.

## Claim boundary

Authorized: that the geometric objects behave as Propositions E1/E2/E3 state on
controlled subspaces. **Not** authorized: LLMs, FCRA, real data, nonlinear
LoRA optimality, or that the measure predicts real forgetting.

## Run

```powershell
python phase1b_expressivity.py --output-dir outputs/local_geometry_v1 --run-role primary_frozen
```
