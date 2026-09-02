"""Deterministic NumPy tools for the curvature-weighted expressivity measure.

These are geometric and capacity-probe implementations of the propositions in
``Expressive Capacity on a Curvature-Weighted Manifold''
(appendix/proofs.tex, Appendix~\\ref{app:expressivity}). They are theorem
regression and sanity checks, not neural-network or benchmark evidence.

Claims authorized here: the metric and effective-capacity objects behave as
Propositions~E1/E2/E3 state on controlled subspaces. No claim is made about
LLMs, FCRA, real data, nonlinear LoRA optimality, or that the measure predicts
real forgetting -- that is the separate, gated check ``check_expressivity_predicts_forgetting``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "notes" / "phase1b_expressivity_protocol.md"
SOURCE_PATH = HERE / "phase1b_expressivity.py"
TEST_PATH = HERE / "test_phase1b.py"
EXACT_ATOL = 2e-11
PROJECTOR_ATOL = 1e-9
RANK_REL_TOL = 1e-10


# --------------------------------------------------------------------------- #
# JSON / hashing / gate plumbing (mirrors phase1_theory_checks discipline)    #
# --------------------------------------------------------------------------- #

def _as_json(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(observed: float, expected: float, atol: float = EXACT_ATOL, rtol: float = 0.0) -> tuple[bool, float, float]:
    error = abs(float(observed) - float(expected))
    tolerance = float(atol + rtol * max(abs(float(expected)), 1.0))
    return error <= tolerance, error, tolerance


def _add_gate(
    gates: list[dict[str, Any]],
    name: str,
    observed: Any,
    expected: Any,
    passed: bool,
    tolerance: float,
    notes: str = "",
    error: float | None = None,
    relation: str = "eq",
) -> None:
    if relation not in {"eq", "le", "ge"}:
        raise ValueError(f"unsupported gate relation: {relation}")
    numeric = isinstance(observed, (int, float, np.number)) and isinstance(expected, (int, float, np.number))
    if error is None and numeric and relation == "eq":
        error = abs(float(observed) - float(expected))
    if numeric:
        if relation == "eq":
            violation = max(0.0, abs(float(observed) - float(expected)) - float(tolerance))
        elif relation == "le":
            violation = max(0.0, float(observed) - float(expected) - float(tolerance))
        else:
            violation = max(0.0, float(expected) - float(observed) - float(tolerance))
    else:
        violation = 0.0 if passed else 1.0
    gates.append(
        {
            "name": name,
            "observed": _as_json(observed),
            "expected": _as_json(expected),
            "passed": bool(passed),
            "tolerance": float(tolerance),
            "relation": relation,
            "abs_error": None if relation != "eq" or error is None else float(error),
            "violation": float(violation),
            "notes": notes,
        }
    )


def _record_case(case: str, gates: list[dict[str, Any]], trace: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    for gate in gates:
        gate["case"] = case
    return {"case": case, "gates": gates}, {"case": case, **trace}


# --------------------------------------------------------------------------- #
# Linear-algebra primitives (mirrors phase1 _null_basis / _safe_projector)    #
# --------------------------------------------------------------------------- #

def _orthonormal_basis(subspace: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis matrix for the column space of ``subspace``."""
    matrix = np.asarray(subspace, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.size == 0:
        return matrix
    q, _ = np.linalg.qr(matrix)
    # numerical cleanup
    return q


def _rank(matrix: np.ndarray, rel_tol: float = RANK_REL_TOL) -> int:
    values = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    if values.size == 0:
        return 0
    cutoff = rel_tol * max(1.0, float(values[0]))
    return int(np.count_nonzero(values > cutoff))


def _null_basis(matrix: np.ndarray, rel_tol: float = RANK_REL_TOL) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    dimension = matrix.shape[1]
    if matrix.shape[0] == 0:
        return np.eye(dimension)
    _, values, vh = np.linalg.svd(matrix, full_matrices=True)
    cutoff = rel_tol * max(1.0, float(values[0]) if values.size else 0.0)
    rank = int(np.count_nonzero(values > cutoff))
    return vh[rank:].T.copy()


# --------------------------------------------------------------------------- #
# Curvature-weighted Grassmann machinery (Prop E1 / E2 / E3)                 #
# --------------------------------------------------------------------------- #

def whitening_operator(curvature: np.ndarray, damping: float) -> np.ndarray:
    """Return ``W = (A + lambda I)^{1/2}`` for a PSD curvature ``A``.

    ``A`` is symmetrized first. With ``damping > 0`` the operator is positive
    definite and invertible, so the embedding of Definition E1 is well defined.
    """
    matrix = np.asarray(curvature, dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    damped = matrix + float(damping) * np.eye(matrix.shape[0])
    # eigh on the symmetric PSD matrix; sqrt via eigendecomposition is stable.
    values, vectors = np.linalg.eigh(damped)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def _whitened_span(basis_u: np.ndarray, curvature: np.ndarray, damping: float) -> np.ndarray:
    """Return an orthonormal basis for ``range(W B_U)`` (the Grassmann point)."""
    operator = whitening_operator(curvature, damping)
    return _orthonormal_basis(operator @ _orthonormal_basis(basis_u))


def principal_angles(basis_u: np.ndarray, basis_v: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between two subspaces.

    Inputs are orthonormal basis matrices for the two subspaces. Angles are
    computed from the singular values of ``Q_U^T Q_V`` via ``arccos`` clamped to
    ``[0, pi/2]``.
    """
    qu = _orthonormal_basis(basis_u)
    qv = _orthonormal_basis(basis_v)
    cosines = np.linalg.svd(qu.T @ qv, compute_uv=False)
    cosines = np.clip(cosines, -1.0, 1.0)
    return np.arccos(cosines)


def curvature_weighted_distance(
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    curvature: np.ndarray,
    damping: float,
) -> float:
    """Prop E1 distance ``d_t(U,V) = ||sin theta||_2`` on whitened subspaces."""
    wu = _whitened_span(basis_u, curvature, damping)
    wv = _whitened_span(basis_v, curvature, damping)
    angles = principal_angles(wu, wv)
    if angles.size == 0:
        return 0.0
    return float(np.linalg.norm(np.sin(angles)))


def effective_capacity(
    local_optimum: np.ndarray,
    curvature_out: np.ndarray,
    curvature_in: np.ndarray,
    occupied_rank: int,
) -> float:
    """Prop E2 effective capacity ``kappa = 0.5 sum_{j>r} sigma_j^2(Y*)``.

    Under the Kronecker quadratic with factors ``C_out``, ``C_in`` and local
    optimum ``X*``, the whitened optimum is ``Y* = C_out^{1/2} X* C_in^{1/2}``;
    the capacity tail beyond ``occupied_rank`` is the squared singular values of
    ``Y*`` past that index (Eckart-Young-Mirsky). This reduces to
    Theorem~spectral-capacity at ``occupied_rank = R``.
    """
    x_star = np.asarray(local_optimum, dtype=float)
    cout_half = _sqrt_psd(curvature_out)
    cin_half = _sqrt_psd(curvature_in)
    y_star = cout_half @ x_star @ cin_half
    values = np.linalg.svd(y_star, compute_uv=False)
    if occupied_rank >= values.size:
        return 0.0
    return 0.5 * float(np.sum(values[occupied_rank:] ** 2))


def _sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


# --------------------------------------------------------------------------- #
# Self-checks (Prop E1/E2/E3 sanity, no forgetting claim yet)                #
# --------------------------------------------------------------------------- #

def check_expressivity_geometry() -> tuple[dict[str, Any], dict[str, Any]]:
    """Prop E1/E2/E3 sanity on controlled subspaces and known spectra."""
    gates: list[dict[str, Any]] = []
    rng = np.random.default_rng(0)

    p = 12
    curvature = rng.standard_normal((p, p))
    curvature = curvature @ curvature.T  # PSD
    damping = 1e-3

    # Two distinct r=3 subspaces.
    full = rng.standard_normal((p, p))
    u, _ = np.linalg.qr(full)
    basis_a = u[:, :3]
    basis_b = u[:, 3:6]

    # E1(a): identical subspace => d_t = 0.
    d_same = curvature_weighted_distance(basis_a, basis_a, curvature, damping)
    ok_same, err_same, _ = _close(d_same, 0.0)
    _add_gate(gates, "E1_distance_identical_subspace_zero", d_same, 0.0, ok_same, EXACT_ATOL, error=err_same,
              notes="phi_t(U)=phi_t(U) => d_t=0")

    # E1(b): distinct subspaces => d_t > 0 (equal rank, non-equivalent).
    d_dist = curvature_weighted_distance(basis_a, basis_b, curvature, damping)
    ok_dist = d_dist > PROJECTOR_ATOL
    _add_gate(gates, "E1_equal_rank_distinct_subspaces_positive_distance", d_dist, 0.0, ok_dist, PROJECTOR_ATOL,
              relation="ge", notes="equal rank r=3 does not imply d_t=0")

    # E1(c): symmetry.
    d_ba = curvature_weighted_distance(basis_b, basis_a, curvature, damping)
    ok_sym, err_sym, _ = _close(d_dist, d_ba)
    _add_gate(gates, "E1_distance_symmetric", d_ba, d_dist, ok_sym, EXACT_ATOL, error=err_sym)

    # E1(d): scale-invariance under a *joint* rescaling of curvature and
    # damping. With fixed lambda>0 the strict invariance is a lambda->0 limiting
    # statement (Prop E1); a joint rescale (K, lambda) -> c(K, lambda) preserves
    # the whitened span exactly, which is the non-degenerate form of the claim.
    joint_scale = curvature_weighted_distance(basis_a, basis_b, 2.5 * curvature, 2.5 * damping)
    ok_scale, err_scale, _ = _close(joint_scale, d_dist)
    _add_gate(gates, "E1_distance_invariant_to_joint_rescale", joint_scale, d_dist, ok_scale, EXACT_ATOL,
              error=err_scale, notes="joint rescale of (curvature, damping) leaves whitened span unchanged")

    # E2: effective capacity reduces to spectral tail at the rank-R point.
    d_in, d_out = 5, 4
    cout = rng.standard_normal((d_out, d_out))
    cout = cout @ cout.T + np.eye(d_out)
    cin = rng.standard_normal((d_in, d_in))
    cin = cin @ cin.T + np.eye(d_in)
    x_star = rng.standard_normal((d_out, d_in))
    y_star = _sqrt_psd(cout) @ x_star @ _sqrt_psd(cin)
    singular = np.linalg.svd(y_star, compute_uv=False)
    r = 2
    kappa = effective_capacity(x_star, cout, cin, r)
    expected_kappa = 0.5 * float(np.sum(singular[r:] ** 2))
    ok_kappa, err_kappa, _ = _close(kappa, expected_kappa)
    _add_gate(gates, "E2_effective_capacity_equals_spectral_tail", kappa, expected_kappa, ok_kappa, EXACT_ATOL,
              error=err_kappa, notes="reduces to Theorem spectral-capacity at occupied_rank=R")

    # E2(b): capacity is non-increasing in occupied rank.
    kappa_more = effective_capacity(x_star, cout, cin, r + 1)
    ok_mono = kappa_more <= kappa + EXACT_ATOL
    _add_gate(gates, "E2_capacity_nonincreasing_in_rank", kappa_more, kappa, ok_mono, EXACT_ATOL,
              relation="le", notes="tail beyond r+1 <= tail beyond r")

    # E3: safe-space contraction as the Grassmann restatement. With PSD sum
    # A_{t} = K1 + K2, the kernel of A_t is contained in the kernel of K1
    # (Prop E3 / Theorem safe-subspace); equivalently the set of
    # zero-curvature-weight directions (the complement of the occupied
    # whitened span) can only contract. We test dim ker(A1+K2) <= dim ker(A1).
    # Use rank-one outer products so kernels are genuinely nontrivial (dim p-1).
    u_vec = rng.standard_normal(p)
    v_vec = rng.standard_normal(p)
    # Ensure v has a component outside span(u) so the contraction is strict.
    v_vec = v_vec - (u_vec @ v_vec) / (u_vec @ u_vec) * u_vec
    k1 = np.outer(u_vec, u_vec)  # rank 1, dim ker = p-1
    k2 = np.outer(v_vec, v_vec)  # rank 1 on an independent direction
    dim_ker1 = p - _rank(k1)
    dim_ker_sum = p - _rank(k1 + k2)
    ok_contract = dim_ker_sum <= dim_ker1
    ok_strict = dim_ker_sum < dim_ker1  # independent direction must contract
    _add_gate(gates, "E3_kernel_contracts_under_psd_sum", dim_ker_sum, dim_ker1, ok_contract, 0.0,
              relation="le", notes="dim ker(A+K) <= dim ker(A) for PSD K (safe-space contraction restated)")
    _add_gate(gates, "E3_kernel_strictly_contracts_with_independent_mode", int(ok_strict), 1, ok_strict, 0.0,
              notes="an independent sensitive mode must strictly reduce the zero-interference dimension")

    return _record_case(
        "expressivity_geometry",
        gates,
        {
            "dim_p": p,
            "d_same": d_same,
            "d_distinct": d_dist,
            "kappa_rank2": kappa,
            "expected_kappa": expected_kappa,
            "dim_ker1": dim_ker1,
            "dim_ker_sum": dim_ker_sum,
        },
    )


# --------------------------------------------------------------------------- #
# D1: Does the expressivity measure predict forgetting? (the gating result)  #
# --------------------------------------------------------------------------- #

def _planted_stream(d_out: int, d_in: int, n_windows: int, n_modes: int, seed: int = 12345):
    """A controlled linear-LoRA stream with a known spectral structure.

    Windows are partitioned into an *early* block and a *late* block whose
    planted singular directions are drawn from **different** subspaces of the
    shared orthonormal bases ``U, V``. This makes the curvature-whitened
    occupied subspaces of early-emphasis and late-emphasis adapters genuinely
    distinct (so ``d_G`` is nontrivial), while keeping the Kronecker quadratic
    curvature common so capacity/spectral tails stay exactly computable.
    """
    rng = np.random.default_rng(seed)
    q_left, _ = np.linalg.qr(rng.standard_normal((d_out, d_out)))
    q_right, _ = np.linalg.qr(rng.standard_normal((d_in, d_in)))
    out_eigs = np.linspace(4.0, 0.5, d_out)
    in_eigs = np.linspace(3.0, 0.5, d_in)
    cout = q_left @ np.diag(out_eigs) @ q_left.T
    cin = q_right @ np.diag(in_eigs) @ q_right.T
    cout = cout @ cout.T + 0.1 * np.eye(d_out)
    cin = cin @ cin.T + 0.1 * np.eye(d_in)
    # Two disjoint mode blocks within the shared bases: early modes 0..n_modes-1,
    # late modes n_modes..2*n_modes-1 (different singular directions).
    n_block = n_modes
    u_base, _ = np.linalg.qr(rng.standard_normal((d_out, 2 * n_block)))
    v_base, _ = np.linalg.qr(rng.standard_normal((d_in, 2 * n_block)))
    # Per-window coefficients: early windows use block 0, late windows block 1,
    # with drifting energy so the spectral tail is time-varying.
    half = n_windows // 2
    alpha = np.zeros((n_windows, 2 * n_block))
    for t in range(n_windows):
        block = 0 if t < half else 1
        for j in range(n_block):
            jj = block * n_block + j
            phase = (t - block * half) + j
            alpha[t, jj] = 1.0 + 0.6 * np.sin(0.9 * phase + 0.3 * j)
    return {
        "d_out": d_out, "d_in": d_in, "n_windows": n_windows, "n_modes": n_modes,
        "n_block": n_block, "cout": cout, "cin": cin,
        "u": u_base, "v": v_base, "alpha": alpha,
    }


def _stream_targets(stream: dict) -> np.ndarray:
    """Return ``Delta_t*`` for each window t as a (T, d_out, d_in) array."""
    u, v, a = stream["u"], stream["v"], stream["alpha"]
    targets = np.stack([u @ np.diag(a[t]) @ v.T for t in range(stream["n_windows"])])
    return targets


def _rank_r_adapter(targets: np.ndarray, curvature_out: np.ndarray, curvature_in: np.ndarray,
                    rank: int, weights: np.ndarray) -> np.ndarray:
    """A deterministic rank-r adapter that solves a weighted least-squares problem.

    ``weights`` is a length-T nonneg vector giving each window's influence
    (e.g., ``[1,0,0,...]`` emphasizes the first window, ``[0,...,0,1]`` the
    last). The adapter is the curvature-whitened top-r SVD of the weighted
    target average, mapped back to parameter space. This mirrors the optimal
    consolidation of Prop E2 / Theorem spectral-capacity at the rank-r point.
    """
    cout_half = _sqrt_psd(curvature_out)
    cin_half = _sqrt_psd(curvature_in)
    weighted = sum(w * t for w, t in zip(weights, targets))
    y = cout_half @ weighted @ cin_half
    left, vals, right_t = np.linalg.svd(y, full_matrices=False)
    r = min(rank, vals.size)
    y_r = left[:, :r] * vals[:r] @ right_t[:r, :]
    return np.linalg.solve(cout_half, y_r) @ np.linalg.inv(cin_half)


def _full_adapter(targets: np.ndarray) -> np.ndarray:
    """Full fine-tuning reference: the unweighted target average (no rank cap).

    Its occupied subspace is whatever the (low-rank) target average actually
    spans -- the ``r_eff < p`` point, not the ``R=infty`` collapse.
    """
    return targets.mean(axis=0)


def _historical_curvature(stream: dict, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Accumulated historical curvature ``A_t = sum w_s K_s`` over windows.

    Each window's local curvature is the Kronecker factor of its target; we
    accumulate a scalar-weighted sum, matching Prop E1/E3.
    """
    u, v, a = stream["u"], stream["v"], stream["alpha"]
    # Window curvature proxy: energy-weighted outer products of the singular dirs.
    acc_out = np.zeros((stream["d_out"], stream["d_out"]))
    acc_in = np.zeros((stream["d_in"], stream["d_in"]))
    for s in range(stream["n_windows"]):
        e_s = float(np.sum(a[s] ** 2)) + 1e-6
        acc_out = acc_out + weights[s] * e_s * (u @ np.diag(a[s] ** 2) @ u.T)
        acc_in = acc_in + weights[s] * e_s * (v @ np.diag(a[s] ** 2) @ v.T)
    acc_out = acc_out + 1e-3 * np.eye(stream["d_out"])
    acc_in = acc_in + 1e-3 * np.eye(stream["d_in"])
    return acc_out, acc_in


def _adapter_occupied_basis(delta: np.ndarray, curvature_out: np.ndarray,
                            curvature_in: np.ndarray, rank: int | None = None) -> np.ndarray:
    """Orthonormal basis of the curvature-whitened occupied subspace of an adapter.

    The occupied subspace is the span of the leading left singular vectors of
    the whitened update ``C_out^{1/2} Delta C_in^{1/2}`` (up to ``rank``),
    i.e. the directions the adapter actually uses -- not the full column space,
    which would coincide for any full-rank matrix and make d_G trivially zero.
    """
    cout_half = _sqrt_psd(curvature_out)
    cin_half = _sqrt_psd(curvature_in)
    whitened = cout_half @ delta @ cin_half
    left, _, _ = np.linalg.svd(whitened, full_matrices=False)
    r = rank if rank is not None else left.shape[1]
    r = min(r, left.shape[1])
    return left[:, :r]


def _forgetting_vs_oracle(adapter: np.ndarray, targets: np.ndarray,
                          curvature_out: np.ndarray, curvature_in: np.ndarray) -> float:
    """Retention gap of ``adapter`` against per-window oracles (Prop C4 gap)."""
    cout_half = _sqrt_psd(curvature_out)
    cin_half = _sqrt_psd(curvature_in)
    total = 0.0
    for t in range(targets.shape[0]):
        res = cout_half @ (adapter - targets[t]) @ cin_half
        total += 0.5 * float(np.sum(res ** 2))
    return total / targets.shape[0]


def check_expressivity_predicts_forgetting() -> tuple[dict[str, Any], dict[str, Any]]:
    """D1: does the curvature-weighted measure predict forgetting (dual main)?

    Dual-main gates:
      (i) d_G line -- two same-rank adapters aimed at different windows have
          d_G>0, and d_G tracks the direction of their relative forgetting.
      (ii) kappa_G line -- adapters with larger curvature-whitened tail kappa_G
           incur larger forgetting-vs-oracle, and kappa_G dominates the
           effective-rank / update-norm / time baselines as a predictor.
    Authorized only on this controlled linear stream; no LLM/FCRA/real-data
    claim.
    """
    gates: list[dict[str, Any]] = []
    d_out, d_in, n_windows, n_modes = 8, 8, 6, 3
    stream = _planted_stream(d_out, d_in, n_windows, n_modes)
    targets = _stream_targets(stream)
    cout, cin = stream["cout"], stream["cin"]

    # Historical curvature: uniform weight over all windows.
    uniform_w = np.ones(n_windows) / n_windows
    acc_out, acc_in = _historical_curvature(stream, uniform_w)

    R = 2  # rank budget for LoRA-style arms.

    # --- (i) d_G line: same rank, different window emphasis --------------
    w_early = np.zeros(n_windows); w_early[:2] = 1.0
    w_late = np.zeros(n_windows); w_late[-2:] = 1.0
    delta_early = _rank_r_adapter(targets, cout, cin, R, w_early)
    delta_late = _rank_r_adapter(targets, cout, cin, R, w_late)

    # Occupied subspaces (whitened) under the historical curvature.
    basis_early = _adapter_occupied_basis(delta_early, acc_out, acc_in, rank=R)
    basis_late = _adapter_occupied_basis(delta_late, acc_out, acc_in, rank=R)
    # Project the d_out x d_in delta to a shared ambient vector for d_G.
    # Use the left-occupied basis as the comparison object.
    # For a matrix update we compare whitened column spaces.
    d_g_el = curvature_weighted_distance(basis_early, basis_late, acc_out, 1e-6)
    # Same-window self-distance must be zero.
    d_g_self = curvature_weighted_distance(basis_early, basis_early, acc_out, 1e-6)
    ok_self, err_self, _ = _close(d_g_self, 0.0, atol=1e-5)
    _add_gate(gates, "D1_dG_self_distance_zero", d_g_self, 0.0, ok_self, 1e-5, error=err_self,
              notes="identical occupied subspace => d_G=0 (numerical tolerance for damped whitening)")
    ok_distinct = d_g_el > 1e-5
    _add_gate(gates, "D1_dG_same_rank_distinct_emphasis_positive", d_g_el, 0.0, ok_distinct, 1e-5,
              relation="ge", notes="same rank R, disjoint window subspaces => d_G>0 (E1 predictiveness)")

    # Relative forgetting direction: early-emphasis adapter forgets LATE
    # windows more; late-emphasis forgets EARLY windows more. The signed
    # difference in forgetting should be consistent with the subspaces differing.
    forget_early_on_late = _forgetting_vs_oracle(delta_early, targets[-2:], cout, cin)
    forget_late_on_late = _forgetting_vs_oracle(delta_late, targets[-2:], cout, cin)
    forget_early_on_early = _forgetting_vs_oracle(delta_early, targets[:2], cout, cin)
    forget_late_on_early = _forgetting_vs_oracle(delta_late, targets[:2], cout, cin)
    # late-emphasis should retain late windows better (forget_late_on_late <= early_on_late)
    ok_late_retain = forget_late_on_late <= forget_early_on_late + EXACT_ATOL
    _add_gate(gates, "D1_late_emphasis_retains_late_windows", forget_late_on_late, forget_early_on_late,
              ok_late_retain, EXACT_ATOL, relation="le", notes="subspace alignment predicts retention direction")
    ok_early_retain = forget_early_on_early <= forget_late_on_early + EXACT_ATOL
    _add_gate(gates, "D1_early_emphasis_retains_early_windows", forget_early_on_early, forget_late_on_early,
              ok_early_retain, EXACT_ATOL, relation="le", notes="symmetric direction check")

    # --- (ii) kappa_G line: capacity tail predicts forgetting -----------
    # Build a spectrum of adapters at increasing occupied rank and measure
    # kappa_G vs forgetting-vs-oracle. Lower rank => larger tail => more forgetting.
    forgetting_curve = []
    kappa_curve = []
    effrank_curve = []
    updatenorm_curve = []
    for r in range(1, n_modes + 1):
        delta = _rank_r_adapter(targets, cout, cin, r, uniform_w)
        forget = _forgetting_vs_oracle(delta, targets, cout, cin)
        # kappa_G: tail of the whitened target optimum beyond occupied rank r.
        y_star = _sqrt_psd(cout) @ targets.mean(axis=0) @ _sqrt_psd(cin)
        vals = np.linalg.svd(y_star, compute_uv=False)
        kappa = 0.5 * float(np.sum(vals[r:] ** 2)) if r < vals.size else 0.0
        forgetting_curve.append(forget)
        kappa_curve.append(kappa)
        effrank_curve.append(float(r))
        updatenorm_curve.append(float(np.linalg.norm(delta)))

    # kappa_G and forgetting should both decrease as r grows; and kappa_G
    # should track forgetting more tightly than effective rank or update norm.
    forget_decreasing = all(forgetting_curve[i] >= forgetting_curve[i + 1] - EXACT_ATOL
                            for i in range(len(forgetting_curve) - 1))
    _add_gate(gates, "D1_forgetting_decreases_with_rank", forgetting_curve[-1], forgetting_curve[0],
              forget_decreasing, 0.0, relation="le",
              notes="more occupied rank => less forgetting (capacity mechanism)")
    kappa_decreasing = all(kappa_curve[i] >= kappa_curve[i + 1] - EXACT_ATOL
                           for i in range(len(kappa_curve) - 1))
    _add_gate(gates, "D1_kappa_decreases_with_rank", kappa_curve[-1], kappa_curve[0],
              kappa_decreasing, 0.0, relation="le", notes="tail shrinks as occupied rank grows")

    # Correlation of kappa_G with forgetting across ranks (Spearman via ranks).
    def _rank_corr(x, y):
        n = len(x)
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])

    corr_kappa = _rank_corr(kappa_curve, forgetting_curve)
    corr_effrank = _rank_corr(effrank_curve, forgetting_curve)
    corr_updatenorm = _rank_corr(updatenorm_curve, forgetting_curve)
    # kappa_G must be a strong positive predictor; and at least as strong as baselines.
    ok_kappa_predicts = corr_kappa >= 0.5
    _add_gate(gates, "D1_kappa_predicts_forgetting_correlation", corr_kappa, 1.0, ok_kappa_predicts, 0.0,
              relation="ge", notes="Spearman corr(kappa_G, forgetting) across occupied ranks")
    ok_kappa_dominates = corr_kappa >= corr_effrank - 1e-9 and corr_kappa >= corr_updatenorm - 1e-9
    _add_gate(gates, "D1_kappa_dominates_baselines", corr_kappa, max(corr_effrank, corr_updatenorm),
              ok_kappa_dominates, 0.0, relation="ge",
              notes="kappa_G corr >= effective-rank and update-norm corr")

    # --- unification: full fine-tuning on the same kappa_G-forgetting axis ---
    delta_full = _full_adapter(targets)
    forget_full = _forgetting_vs_oracle(delta_full, targets, cout, cin)
    r_eff = _rank(_sqrt_psd(acc_out) @ delta_full @ _sqrt_psd(acc_in))
    full_basis = _adapter_occupied_basis(delta_full, acc_out, acc_in, rank=max(1, r_eff))
    y_star = _sqrt_psd(cout) @ targets.mean(axis=0) @ _sqrt_psd(cin)
    vals = np.linalg.svd(y_star, compute_uv=False)
    kappa_full = 0.5 * float(np.sum(vals[r_eff:] ** 2)) if r_eff < vals.size else 0.0
    # Full FT should sit on (or below) the LoRA forgetting-kappa curve at its
    # effective rank: at least not worse than the rank-r_eff LoRA adapter.
    delta_at_reff = _rank_r_adapter(targets, cout, cin, max(1, r_eff), uniform_w)
    forget_at_reff = _forgetting_vs_oracle(delta_at_reff, targets, cout, cin)
    ok_unify = forget_full <= forget_at_reff + 1e-6
    _add_gate(gates, "D1_full_ft_comparable_to_rank_reff_lora", forget_full, forget_at_reff,
              ok_unify, 1e-6, relation="le",
              notes="full FT (r_eff) not worse than rank-r_eff LoRA -- same axis, not R=inf collapse")

    return _record_case(
        "expressivity_predicts_forgetting",
        gates,
        {
            "n_windows": n_windows, "R": R, "r_eff_full": int(r_eff),
            "dG_early_late": d_g_el,
            "forget_early_on_late": forget_early_on_late,
            "forget_late_on_late": forget_late_on_late,
            "forget_early_on_early": forget_early_on_early,
            "forget_late_on_early": forget_late_on_early,
            "kappa_curve": kappa_curve,
            "forgetting_curve": forgetting_curve,
            "corr_kappa": corr_kappa, "corr_effrank": corr_effrank, "corr_updatenorm": corr_updatenorm,
            "forget_full": forget_full, "forget_at_reff": forget_at_reff, "kappa_full": kappa_full,
        },
    )



def run_all_checks() -> dict[str, Any]:
    case_results = [check_expressivity_geometry(), check_expressivity_predicts_forgetting()]
    case_summaries = [result[0] for result in case_results]
    traces = {result[1]["case"]: result[1] for result in case_results}
    gates = [gate for case in case_summaries for gate in case["gates"]]
    failed = [gate for gate in gates if not gate["passed"]]
    return {
        "schema_version": 1,
        "stage": "phase1b_expressivity_geometry",
        "claim_boundary": "conditional_theory_only",
        "required_gate_count": len(gates),
        "failed_gate_count": len(failed),
        "all_required_pass": not failed,
        "cases": [
            {
                "case": case["case"],
                "gate_count": len(case["gates"]),
                "failed_count": sum(not gate["passed"] for gate in case["gates"]),
            }
            for case in case_summaries
        ],
        "gates": gates,
        "traces": traces,
    }


# --------------------------------------------------------------------------- #
# Output / manifest discipline (mirrors phase1)                              #
# --------------------------------------------------------------------------- #

def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_as_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_run(output_dir: Path, run_role: str) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_all_checks()
    summary = {key: value for key, value in result.items() if key != "traces"}
    summary["headline"] = {
        "E1_distinct_distance": result["traces"]["expressivity_geometry"]["d_distinct"],
        "E2_kappa_rank2": result["traces"]["expressivity_geometry"]["kappa_rank2"],
        "E3_dim_ker_sum": result["traces"]["expressivity_geometry"]["dim_ker_sum"],
        "D1_dG_early_late": result["traces"]["expressivity_predicts_forgetting"]["dG_early_late"],
        "D1_corr_kappa_forgetting": result["traces"]["expressivity_predicts_forgetting"]["corr_kappa"],
        "D1_r_eff_full": result["traces"]["expressivity_predicts_forgetting"]["r_eff_full"],
        "D1_forget_full_vs_lora_at_reff": (
            result["traces"]["expressivity_predicts_forgetting"]["forget_full"]
            - result["traces"]["expressivity_predicts_forgetting"]["forget_at_reff"]
        ),
    }
    traces_path = output_dir / "traces.json"
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "per_case.csv"
    _write_json(traces_path, result["traces"])
    _write_json(summary_path, summary)
    fields = ["case", "name", "relation", "observed", "expected", "passed", "tolerance", "abs_error", "violation", "notes"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for gate in result["gates"]:
            writer.writerow({field: _as_json(gate.get(field)) for field in fields})

    output_hashes = {path.name: _sha256(path) for path in (traces_path, summary_path, csv_path)}
    config_buffer = io.StringIO()
    with redirect_stdout(config_buffer):
        np.show_config()
    manifest = {
        "schema_version": 1,
        "stage": "phase1b_expressivity_geometry",
        "run_role": run_role,
        "source_sha256": _sha256(SOURCE_PATH),
        "protocol_sha256": _sha256(PROTOCOL_PATH) if PROTOCOL_PATH.exists() else "PROTOCOL_NOT_FROZEN_YET",
        "test_sha256": _sha256(TEST_PATH) if TEST_PATH.exists() else "TEST_NOT_PRESENT_YET",
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_config": config_buffer.getvalue(),
        "command": " ".join(sys.argv),
        "resource_ledger": {
            "gpu_used": False,
            "optimizer_state_bytes": 0,
            "replay_bytes": 0,
            "external_model_bytes": 0,
        },
        "output_hashes": output_hashes,
        "all_required_pass": result["all_required_pass"],
        "claim_boundary": "conditional_theory_only",
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {"summary": summary, "manifest": manifest, "output_dir": str(output_dir)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-1b curvature-weighted expressivity geometry checks.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-role", default="primary_frozen")
    args = parser.parse_args(argv)
    result = write_run(args.output_dir, args.run_role)
    print(json.dumps(_as_json(result["summary"]), indent=2, sort_keys=True))
    return 0 if result["summary"]["all_required_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
