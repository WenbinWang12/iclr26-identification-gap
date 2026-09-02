# Phase-1b Expressivity Geometry and Forgetting-Prediction Protocol

Date frozen: 2026-08-19

Status: frozen candidate for the curvature-weighted expressivity measure of
Appendix~\ref{app:expressivity} (Propositions E1, E2, E3). This is a
deterministic NumPy regression and falsification check, not neural-network or
benchmark evidence. No remote confirmation or four-provider cross-check has
been run yet; the first citable local candidate must be generated after this
protocol is frozen and must match the source/protocol SHA256 recorded in its
manifest.

## Scope and claim boundary

This suite verifies that the geometric objects of Propositions E1/E2/E3 behave
as stated on controlled subspaces and known spectra (Case G:
`expressivity_geometry`), and that on a controlled linear-LoRA stream the
curvature-weighted distance `d_G` and effective capacity `kappa_G` predict
signed forgetting and that LoRA and full fine-tuning sit on one
`kappa_G`-forgetting axis (Case D: `expressivity_predicts_forgetting`).

It is **not** evidence that the assumptions hold in neural networks. It does
not validate FCRA, real LoRA training, language models, task-free learning,
independent addressability, nonlinear model merging, or any
benchmark-performance claim. It does not establish that `d_G = 0` implies
output-function equivalence (Remark, Prop E1).

## Frozen numerical policy

- All computations use NumPy `float64` on CPU. `gpu_used = false`.
- Algebraic scalar identities use absolute tolerance `2e-11`.
- Projector and subspace-containment checks use tolerance `1e-9`; damped
  whitening self-distances use tolerance `1e-5` to absorb the `lambda>0`
  numerical floor (Prop E1 scale-invariance is a `lambda->0` limiting
  statement; the strict form holds under joint rescaling).
- Numerical rank uses a relative cutoff of `1e-10` times the largest singular
  or absolute eigenvalue, with a scale floor of one.
- Inequality gates include their tolerance explicitly; no tolerance is changed
  after observing a result.
- Source, protocol, and test SHA256 values must match byte-for-byte across
  hosts. Generated output hashes are provenance only. Floating-point outputs
  are compared by the frozen numerical tolerances, not byte equality.
- Deterministic constructions (fixed seed `12345` for the planted stream,
  fixed orthonormal bases via QR) replace Monte Carlo sampling.

## Case G: expressivity geometry (Prop E1/E2/E3)

- `p = 12`, a fixed PSD curvature, damping `lambda = 1e-3`.
- E1(a): identical occupied subspace => `d_G = 0`.
- E1(b): two distinct `r = 3` subspaces => `d_G > 0` (equal rank, not
  equivalent).
- E1(c): symmetry `d_G(U,V) = d_G(V,U)`.
- E1(d): joint rescale `(K, lambda) -> c(K, lambda)` leaves `d_G` unchanged
  (strict form; fixed-`lambda` rescale is a `lambda->0` limiting statement).
- E2: `kappa_G` equals the curvature-whitened spectral tail and reduces to
  Theorem spectral-capacity at the rank-R point; `kappa_G` is non-increasing in
  occupied rank.
- E3: `dim ker(A + K) <= dim ker(A)` for PSD `K`; an independent sensitive
  mode strictly reduces the zero-interference dimension (Grassmann restatement
  of Theorem safe-subspace).

## Case D: forgetting prediction (the gating result)

- Planted linear stream `d_out = d_in = 8`, `T = 6` windows, two disjoint mode
  blocks of size `3` within shared orthonormal bases so early/late occupied
  subspaces are genuinely distinct. Shared Kronecker curvature, linear risk so
  spectral tails are exactly computable.
- `R = 2` rank budget for LoRA-style arms.
- D-line (d_G, equivalence): self-distance zero; same-rank disjoint-window
  adapters have `d_G > 0`; early-emphasis adapter retains early windows better
  than late-emphasis and vice versa (subspace alignment predicts retention
  direction).
- D-line (kappa_G, capacity): forgetting decreases with occupied rank;
  `kappa_G` decreases with rank; Spearman correlation `corr(kappa_G, forgetting)`
  across ranks `>= 0.5` and `kappa_G` dominates the effective-rank and
  update-norm baselines.
- D-line (unification): full fine-tuning (effective rank `r_eff`) is not worse
  than the rank-`r_eff` LoRA adapter -- LoRA and full FT sit on one
  `kappa_G`-forgetting axis, not the `R->infty` collapse.

## Resource ledger

`gpu_used = false`, `optimizer_state_bytes = 0`, `replay_bytes = 0`,
`external_model_bytes = 0`. CPU-only by design.
