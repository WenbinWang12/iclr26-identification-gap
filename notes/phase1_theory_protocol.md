# Phase-1 Synthetic Theory Falsification Protocol

Date frozen: 2026-07-23

Status: audited amendment frozen before the first remote Phase-1 confirmation.
Local directories `local_primary_v1`, `local_primary_v2`, and
`local_primary_v3` are retained as non-citable pre-audit artifacts; their
manifests do not identify this final amended source/protocol pair. The first
citable local candidate must be generated after this amendment and must match
all three current identity hashes.

## Scope and claim boundary

This deterministic NumPy suite is a regression and falsification check for the
conditional mathematical statements in C1, C2, C3, C4, C5, C6, C7, C8, and
the aligned local consolidation statement C9. It is not evidence that the
assumptions hold in neural networks. It does not validate FCRA (C10), LoRA,
language models, task-free learning, independent addressability, replay, or
any benchmark-performance claim.

The design was reviewed before implementation by a read-only four-provider
cross-check archived at
`_artifacts/crosscheck/20260723-142012-7774cca6`. Its frozen prompt SHA256 is
`7d5a3eb123c43ffa3b6d303e0dc9c386e7591227004916bb680ba2d100d3631f`.
The verified workers were GPT `gpt-5.5` at `xhigh`, Claude
`claude-opus-4-8` at `max`, GLM `glm-5.2` at `ultra`, and DeepSeek
`deepseek-v4-pro` at `xhigh`. The implementation incorporates their required
non-circular controls, threshold definitions, tie handling, and assumption
guards.

## Frozen numerical policy

- All computations use NumPy `float64` on CPU.
- Algebraic scalar identities and frozen scalar values use absolute tolerance
  `2e-11`.
- SVD-derived rank-R competitor inequalities use the same `2e-11` slack; the
  recorded gate relation is explicit (`eq`, `le`, or `ge`).
- Projector and subspace-containment checks use tolerance `1e-9`.
- Numerical rank uses a relative cutoff of `1e-10` times the largest singular
  or absolute eigenvalue, with a scale floor of one.
- Inequality gates include their tolerance explicitly; no tolerance is changed
  after observing a result.
- Source, protocol, and test SHA256 values must match byte-for-byte across
  hosts. Generated output hashes are provenance only. Floating-point outputs
  are compared by the frozen numerical tolerances, not byte equality.
- Fixed matrices and exact sign enumeration replace Monte Carlo sampling.

## Case A: local Taylor expansion and projected stationarity

Use `p=4`, `x_s=0`, `H=diag(1,2,3,4)`, `rho=0.7`, and the unit vector
`u=(1,2,-2,0)/3`. Define

`L(x)=g^T x + 0.5 x^T H x + (rho/6)(u^T x)^3`.

The Hessian is exactly `rho`-Lipschitz. Evaluate parallel, orthogonal, and
mixed directions at radii `2^{-j}`, `j=0,...,6`. Required gates check the
black-box loss difference against `(rho/6)(u^T d)^3`, the cubic bound, parallel
tightness, and cubic log-log slope. For `U=span(e1,e2)`, test one gradient
orthogonal to all of U and one with a nonzero U component. A single accidental
orthogonal displacement is not accepted as a projected-stationarity test.

## Case B: sequential accumulation, cancellation, and innovations

Use the directly evaluated old loss `L_old(x)=0.5 x^T H x` with
`H=diag(1,2,3)`. Required paths are aligned, H-orthogonal, cancelling
`(+e1,-e1)`, and recurrent (returning to earlier parameter points). Check the
path identity and every increment's cross plus quadratic terms. Cancellation
must be non-monotone and return to zero.

For independent innovations, enumerate all `2^4` Rademacher sign vectors for
the fixed means and directions embedded in the source. The enumerated expected
forgetting must equal

`0.5 ||sum_t mu_t||_H^2 + 0.5 sum_t v_t^T H v_t = 0.125`.

A quartic-perturbed old loss is an out-of-assumption guard: the exact quadratic
increment identity must visibly fail rather than being silently generalized.

## Case C: decomposition and curvature-whitened rank capacity

First verify the conflict-capacity-online accounting identity on two explicit
matrix risks with nonzero values in all three terms. The frozen comparator
values are total gap `3`, conflict `1`, capacity `0.5`, and online gap
`1.5`; the full comparator must be stationary and both rank-constrained points
must have numerical rank at most one.

For spectral capacity, use fixed positive-definite left and right curvature
factors and a whitened `5 x 4` target with singular values
`(4,2,1,0.25)`. For every `R=0,...,4`, directly measure the weighted loss of
the whitened truncated-SVD solution and compare it with the frozen tails
`(10.53125,2.53125,0.53125,0.03125,0)`. Check numerical rank, the finite-budget
lower bound, deterministic rank-R perturbation competitors, and the Euclidean
SVD comparator. These controls do not prove global optimality numerically, but
they prevent the implementation from merely printing the analytic tail.
Deterministic competitor factors use the frozen prescribed left and right
modes, not SVD-returned vector signs, so legal LAPACK sign choices cannot alter
their cross-host traces.

For three layers with singular values `(4,1)`, `(3,0.5)`, and `(2,1.5)`, a
global budget of three retains `(4,3,2)` and has residual `1.75`; exhaustive
integer allocation must agree. A persistent rank-one innovation stream checks
known rank growth and the fixed-rank spectral tail. Each prefix is measured
independently with an SVD at the frozen rank R=2; the measured prefix tail must
equal the sum of the omitted prescribed modal energies, not merely the expected
rank count.

## Case D: exact and approximate safe spaces and useful-update price

In ambient dimension six, accumulate rank-one PSD sensitivities along `e1`,
`e2`, `(e3+e4)/sqrt(2)`, repeated `e2`, a zero matrix, and
`(e3-e4)/sqrt(2)`. Expected nullities including time zero are
`(6,5,4,3,3,3,2)`. Compare the cumulative-matrix null projector with an
independently computed stacked-row null projector, test containment, and
require the repeated and zero steps to be no-ops.

For the fixed dictionary in the source, first gate numerical full column rank
`4`. Expected coefficient nullities are `(4,3,2,1,1,1,0)`. The coefficient null projector must equal the
preimage projector obtained from stacked sensitivity rows times the dictionary.
No comparison across a changing dictionary is made.

The approximate safe space is the intersection of complements of each
window's normalized sensitive span, not a re-thresholded cumulative matrix.
Use `tau=0.15`, eigenvalues separated from the threshold, and rescalings
`1e-6` and `1e6`. Sensitive-span projectors must be invariant and repeated
spans must not consume a new dimension. A signed non-PSD accumulation is an
out-of-assumption guard that must exhibit possible kernel expansion.

For C7, check the exact constrained-gain solution in both cases from the paper:
zero price when the gain direction has a historical-null component, and
`gamma^2/(2 z^T A^dagger z)` after the gain lies in the range of a full-rank A.

## Case E: block allocation and cross-block failure

Enumerate every support of size at most two. First check the top-k rule on an
actually block-diagonal positive-definite surrogate. Then set damping to zero
and use the already positive-definite true screening matrix

`A(rho)=[[1,rho,0],[rho,1,0],[0,0,1]]`, `D=I`,
`z=(1,1,0.9)`, `gamma=1`, `k=2`.

At `rho=0.9`, diagonal scores select `{1,2}`. Its true utility is `20/19`
and cost is `19/40`; `{1,3}` and `{2,3}` tie with utility `1.81` and cost
`50/181`. The test computes the SPD eigenvalues and `delta` rather than
hard-coding the spectral sandwich. It must state that a strictly better support exists, not that
the optimum is unique. The spectral-equivalence parameter is `delta=0.9`, the
bound factor is `19`, and the observed cost ratio must lie below it. The same
factor is also checked at `rho=0.2`, while `rho=0` is the exact block-diagonal
control. Fixed-support utility uses the principal matrix `A_SS`, never a slice
of the global inverse. A one-block support is not used as a cross-block failure
case because cross terms cannot affect it.

## Artifacts and acceptance

Each run writes `per_case.csv`, `traces.json`, `summary.json`, and
`manifest.json`. Every gate records its relation (`eq`, `le`, or `ge`) and a
one-sided violation, so a large positive value for a lower-bound gate is not
misreported as an equality error. Acceptance requires:

1. all frozen required gates pass locally;
2. all unit tests pass locally and on the Linux NumPy environment;
3. source, protocol, and test hashes match across hosts;
4. all remote gates pass and numeric outputs agree within the frozen
   tolerances;
5. the resource ledger records `gpu_used=false` and
   `optimizer_state_bytes=0`;
6. a post-result four-provider read-only cross-check reviews the complete
   artifacts before the result is cited.

Failure of any required gate is retained and reported; the protocol is not
weakened or rerun under altered thresholds.
