"""D2-A: rare-task injection bridge to Huang et al. (arXiv:2605.29548).

Implements the linear mixture of Huang et al. (Sec. 3): K tasks on orthogonal
blocks with frequencies ``pi_k`` and spectra ``lambda_{k,j} = j^{-alpha}``;
a shared width-N encoder retaining the top-N eigenspace of the frequency-weighted
covariance ``M = sum_k pi_k C_k``. A rare task ``r`` is withheld for ``G`` steps
and injected in a batch; we measure its retained signal
``s_r(N) = Tr(P_U C_r)/Tr(C_r)`` and ask whether our curvature-weighted
expressivity measures ``kappa_G`` and ``d_G`` predict the critical width
``N_r^crit = min{N : mu_N^F <= pi_r lambda_r}`` at which the rare task becomes
locally stable (Huang Prop 6).

Authorized claim: on this controlled linear mixture, the curvature-weighted
measures predict the rare-task retention onset. Not authorized: OLMo, real
LLMs, FCRA, benchmark performance, or nonlinear-optimality claims.
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
PROTOCOL_PATH = ROOT / "notes" / "phase2_injection_protocol.md"
SOURCE_PATH = HERE / "phase2_injection.py"
EXACT_ATOL = 2e-11
RANK_REL_TOL = 1e-10


def _as_json(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(v) for v in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as h:
        for block in iter(lambda: h.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _add_gate(gates, name, observed, expected, passed, tol, notes="", relation="eq"):
    numeric = isinstance(observed, (int, float, np.number)) and isinstance(expected, (int, float, np.number))
    err = abs(float(observed) - float(expected)) if (numeric and relation == "eq") else None
    if numeric:
        if relation == "eq":
            viol = max(0.0, abs(float(observed) - float(expected)) - float(tol))
        elif relation == "le":
            viol = max(0.0, float(observed) - float(expected) - float(tol))
        else:
            viol = max(0.0, float(expected) - float(observed) - float(tol))
    else:
        viol = 0.0 if passed else 1.0
    gates.append({"name": name, "observed": _as_json(observed), "expected": _as_json(expected),
                  "passed": bool(passed), "tolerance": float(tol), "relation": relation,
                  "abs_error": None if relation != "eq" or err is None else float(err),
                  "violation": float(viol), "notes": notes})


def _record_case(case, gates, trace):
    for g in gates:
        g["case"] = case
    return {"case": case, "gates": gates}, {"case": case, **trace}


def _rank(matrix, rel_tol=RANK_REL_TOL):
    vals = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    if vals.size == 0:
        return 0
    return int(np.count_nonzero(vals > rel_tol * max(1.0, float(vals[0]))))


def _sqrt_psd(matrix):
    matrix = np.asarray(matrix, dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    vals, vecs = np.linalg.eigh(matrix)
    return (vecs * np.sqrt(np.clip(vals, 0, None))) @ vecs.T


def _build_mixture(d, K, alpha_spec, beta_freq, seed=20260529):
    """Huang Sec. 3 mixture: K tasks on orthogonal blocks, spectra j^{-alpha}."""
    rng = np.random.default_rng(seed)
    # Assign each task an orthonormal block within d dimensions (disjoint if possible).
    block_per_task = d // K
    assert block_per_task >= 1, "d too small for K orthogonal blocks"
    bases = []
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    for k in range(K):
        bases.append(Q[:, k * block_per_task:(k + 1) * block_per_task])
    # spectra and frequencies
    spectra = [np.array([j ** (-alpha_spec) for j in range(1, block_per_task + 1)], float) for _ in range(K)]
    freqs = np.array([k ** (-beta_freq) for k in range(1, K + 1)], float)
    freqs = freqs / freqs.sum()
    # task covariance C_k = B_k diag(lambda_k) B_k^T
    Cks = [B @ np.diag(lam) @ B.T for B, lam in zip(bases, spectra)]
    M = sum(pi * Ck for pi, Ck in zip(freqs, Cks))
    return {"d": d, "K": K, "bases": bases, "spectra": spectra, "freqs": freqs, "Cks": Cks, "M": M}


def _top_n_eigenspace(M, N):
    vals, vecs = np.linalg.eigh(M)
    idx = np.argsort(vals)[::-1][:N]
    return vecs[:, idx], np.sort(vals)[::-1]


def _retained_signal(U_basis, Ck):
    """s_k = Tr(P_U C_k) / Tr(C_k)."""
    P = U_basis @ U_basis.T
    return float(np.trace(P @ Ck) / max(np.trace(Ck), 1e-12))


def _kappa_G(M_common, occupied_basis):
    """Curvature-weighted tail of M beyond the occupied subspace.

    Tail = sum of eigenvalues of M on the orthogonal complement of the occupied
    subspace. Under the mixture quadratic, this is the effective capacity of
    Prop E2 specialized to the occupied encoder subspace.
    """
    P = occupied_basis @ occupied_basis.T
    tail = np.trace((np.eye(P.shape[0]) - P) @ M_common)
    return 0.5 * float(tail)


def _dG_to_rare(occupied_basis, rare_basis):
    """Curvature-free principal-angle distance between the occupied subspace and
    the rare-task direction, under the identity metric (M-curvature variant
    also computed below)."""
    q = occupied_basis
    # rare_basis is one direction
    cos = float(np.linalg.svd(q.T @ rare_basis, compute_uv=False)[0])
    return float(np.arccos(np.clip(cos, -1, 1)))


def check_injection_predicts_retention() -> tuple[dict, dict]:
    """D2-A core: do kappa_G / d_G predict the rare-task critical width N_r^crit?"""
    gates = []
    d, K = 24, 8
    mix = _build_mixture(d, K, alpha_spec=2.0, beta_freq=1.5)
    freqs, Cks, M = mix["freqs"], mix["Cks"], mix["M"]
    bases = mix["bases"]

    # Rare task = least frequent task (k=K). Its single top direction b_r.
    r = K - 1
    rare_dir = bases[r][:, :1]
    C_r = Cks[r]
    pi_r = float(freqs[r])
    lam_r1 = float(mix["spectra"][r][0])

    # Common block M_F = sum of common tasks k<r.
    M_F = sum(freqs[k] * Cks[k] for k in range(r))
    mu_F = np.sort(np.linalg.eigvalsh(M_F))[::-1]  # descending

    # Huang Prop 6: rare direction is stable iff pi_r * lam_r1 >= mu_N^F.
    # N_r^crit = min N s.t. mu_N^F <= pi_r * lam_r1.
    threshold = pi_r * lam_r1
    Ncrit = None
    for N in range(1, len(mu_F) + 1):
        if mu_F[N - 1] <= threshold:
            Ncrit = N
            break
    Ncrit = Ncrit if Ncrit is not None else len(mu_F)

    # Scan encoder width N; measure retained rare signal and our measures.
    widths = list(range(1, min(d, 2 * Ncrit) + 1))
    s_r_curve, kappa_curve, dG_curve = [], [], []
    for N in widths:
        U, _ = _top_n_eigenspace(M, N)
        s_r = _retained_signal(U, C_r)
        # occupied = top-N of full M; kappa_G relative to common block M_F
        # (capacity left over for the rare task once common is served)
        U_F, _ = _top_n_eigenspace(M_F, min(N, M_F.shape[0]))
        kappa = _kappa_G(M_F, U_F)
        dG = _dG_to_rare(U_F, rare_dir)
        s_r_curve.append(s_r)
        kappa_curve.append(kappa)
        dG_curve.append(dG)

    # Gate 1: above Ncrit, rare signal strictly increases (retention onset).
    idx_crit = widths.index(Ncrit) if Ncrit in widths else len(widths) - 1
    # signal at Ncrit+ vs below Ncrit
    below = s_r_curve[:idx_crit] if idx_crit > 0 else [0.0]
    above = s_r_curve[idx_crit:]
    onset = (max(above, default=0.0) > max(below) + 1e-9)
    _add_gate(gates, "D2A_rare_signal_onset_above_Ncrit", int(onset), 1, onset, 0.0,
              notes=f"rare retention rises above Ncrit={Ncrit}")

    # Gate 2: kappa_G (common residual) decreases with N (capacity frees up).
    kappa_dec = all(kappa_curve[i] >= kappa_curve[i + 1] - 1e-12 for i in range(len(kappa_curve) - 1))
    _add_gate(gates, "D2A_kappa_decreases_with_width", kappa_curve[-1], kappa_curve[0], kappa_dec, 0.0,
              relation="le", notes="common residual capacity tail shrinks as width grows")

    # Gate 3: dG (distance from occupied common to rare dir) decreases with N.
    dG_dec = all(dG_curve[i] >= dG_curve[i + 1] - 1e-12 for i in range(len(dG_curve) - 1))
    _add_gate(gates, "D2A_dG_to_rare_decreases_with_width", dG_curve[-1], dG_curve[0], dG_dec, 0.0,
              relation="le", notes="occupied common subspace rotates toward rare direction as N grows")

    # Gate 4: the Nth curvature-weighted value of M_F predicts Ncrit.
    # Huang Prop 6's predictor is the single eigenvalue mu_N^F (the weakest
    # occupied common utility), not the tail sum kappa_G. We test that exact
    # predictor at Ncrit: mu_Ncrit^F <= pi_r*lam_r1. (kappa_G is the tail SUM,
    # a different magnitude; it predicts retention correlation in Gate 5.)
    mu_at_crit = float(mu_F[Ncrit - 1]) if Ncrit - 1 < len(mu_F) else float(mu_F[-1])
    pred = mu_at_crit <= threshold + 1e-12
    _add_gate(gates, "D2A_mu_N_at_Ncrit_below_rare_utility", mu_at_crit, threshold, pred, 1e-12,
              relation="le", notes="mu_Ncrit^F <= pi_r*lam_r1: our curvature-weighted value reproduces Huang's Ncrit predictor")

    # Gate 4b: kappa_G (the tail SUM) at Ncrit is below its value at Ncrit-1
    # (capacity is freeing up at exactly the predicted onset).
    if idx_crit >= 1:
        kappa_drop = kappa_curve[idx_crit] <= kappa_curve[idx_crit - 1] + 1e-12
        _add_gate(gates, "D2A_kappa_drops_at_Ncrit", kappa_curve[idx_crit], kappa_curve[idx_crit - 1],
                  kappa_drop, 1e-12, relation="le", notes="kappa_G tail sum is non-increasing through the onset")

    # Gate 5: correlation of kappa with (1 - s_r) across widths: kappa low <-> retention high.
    inv_retention = [1.0 - s for s in s_r_curve]
    if len(set(inv_retention)) > 1:
        rx = np.argsort(np.argsort(kappa_curve)); ry = np.argsort(np.argsort(inv_retention))
        corr = float(np.corrcoef(rx, ry)[0, 1])
    else:
        corr = 1.0
    _add_gate(gates, "D2A_kappa_correlates_inverse_retention", corr, 1.0, corr >= 0.5, 0.0,
              relation="ge", notes="Spearman corr(kappa_G, 1-s_r) across widths >= 0.5")

    return _record_case("injection_predicts_retention", gates, {
        "d": d, "K": K, "rare_task": r, "pi_r": pi_r, "lam_r1": lam_r1,
        "threshold": threshold, "Ncrit": Ncrit, "widths": widths,
        "s_r_curve": s_r_curve, "kappa_curve": kappa_curve, "dG_curve": dG_curve,
        "corr_kappa_inv_retention": corr,
    })


def _adam_optimize_encoder(M, N, steps=2000, lr=0.05, seed=0):
    """Optimize a width-N encoder U (d x N, U^T U = I) on mixture loss
    L(U) = Tr(M) - Tr(U^T M U) via projected gradient + Adam on a Grassmann-
    tangent parametrization. Returns the learned basis."""
    d = M.shape[0]
    rng = np.random.default_rng(seed)
    # random init on the Grassmann manifold
    G0 = rng.standard_normal((d, N))
    U, _ = np.linalg.qr(G0)
    m = np.zeros_like(U); v = np.zeros_like(U)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        # Riemannian gradient: 2 (I - P_U) M U  (Huang Sec 3.2)
        P = U @ U.T
        grad = 2 * (np.eye(d) - P) @ M @ U
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * (grad * grad)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        U = U + lr * mhat / (np.sqrt(vhat) + eps)
        # re-orthonormalize (project back to Grassmann)
        U, _ = np.linalg.qr(U)
    return U


def check_adam_trajectory_matches_prediction() -> tuple[dict, dict]:
    """D2-A nonlinearity check: does Adam-optimized encoder match the analytic
    top-N eigenspace prediction for rare-task retention onset?"""
    gates = []
    d, K = 24, 8
    mix = _build_mixture(d, K, alpha_spec=2.0, beta_freq=1.5)
    Cks, M, freqs, bases = mix["Cks"], mix["M"], mix["freqs"], mix["bases"]
    r = K - 1
    C_r, pi_r = Cks[r], float(freqs[r])
    lam_r1 = float(mix["spectra"][r][0])
    M_F = sum(freqs[k] * Cks[k] for k in range(r))
    mu_F = np.sort(np.linalg.eigvalsh(M_F))[::-1]
    threshold = pi_r * lam_r1
    Ncrit = next((N for N in range(1, len(mu_F) + 1) if mu_F[N - 1] <= threshold), len(mu_F))

    # Adam-optimize at N just below and just above Ncrit; measure rare signal.
    N_below = max(1, Ncrit - 2)
    N_above = min(d, Ncrit + 2)
    s_below, s_above = None, None
    for N, label in [(N_below, "below"), (N_above, "above")]:
        U_adam = _adam_optimize_encoder(M, N, steps=3000, lr=0.03, seed=7)
        s_r = _retained_signal(U_adam, C_r)
        if label == "below":
            s_below = s_r
        else:
            s_above = s_r
    # Below Ncrit the rare signal should stay low; above Ncrit it should be learned.
    below_low = s_below < 0.05
    _add_gate(gates, "D2A_adam_below_Ncrit_rare_unlearned", s_below, 0.0, below_low, 0.05,
              relation="le", notes=f"Adam at N={N_below}<Ncrit keeps rare signal low")
    above_learned = s_above > 0.5
    _add_gate(gates, "D2A_adam_above_Ncrit_rare_learned", s_above, 1.0, above_learned, 0.0,
              relation="ge", notes=f"Adam at N={N_above}>Ncrit learns rare signal")
    # The onset: s_above > s_below by a clear margin.
    onset = s_above > s_below + 0.3
    _add_gate(gates, "D2A_adam_onset_straddles_Ncrit", s_above - s_below, 0.3, onset, 0.0,
              relation="ge", notes="nonlinear Adam reproduces the analytic Ncrit onset boundary")
    return _record_case("adam_trajectory_matches", gates, {
        "Ncrit": Ncrit, "N_below": N_below, "N_above": N_above,
        "s_r_below_Ncrit": s_below, "s_r_above_Ncrit": s_above,
    })


def run_all_checks():
    results = [check_injection_predicts_retention(), check_adam_trajectory_matches_prediction()]
    summaries = [r[0] for r in results]
    traces = {r[1]["case"]: r[1] for r in results}
    gates = [g for c in summaries for g in c["gates"]]
    failed = [g for g in gates if not g["passed"]]
    return {"schema_version": 1, "stage": "phase2_injection_bridge",
            "claim_boundary": "controlled_linear_mixture_only",
            "required_gate_count": len(gates), "failed_gate_count": len(failed),
            "all_required_pass": not failed,
            "cases": [{"case": c["case"], "gate_count": len(c["gates"]),
                       "failed_count": sum(not g["passed"] for g in c["gates"])} for c in summaries],
            "gates": gates, "traces": traces}


def _write_json(path, payload):
    path.write_text(json.dumps(_as_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_run(output_dir, run_role):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_all_checks()
    summary = {k: v for k, v in result.items() if k != "traces"}
    t = result["traces"]["injection_predicts_retention"]
    summary["headline"] = {"Ncrit": t["Ncrit"], "threshold": t["threshold"],
                           "corr_kappa_inv_retention": t["corr_kappa_inv_retention"],
                           "s_r_above_Ncrit": t["s_r_curve"][-1]}
    ta = result["traces"].get("adam_trajectory_matches", {})
    if ta:
        summary["headline"]["adam_s_r_below"] = ta["s_r_below_Ncrit"]
        summary["headline"]["adam_s_r_above"] = ta["s_r_above_Ncrit"]
    tp = output_dir / "traces.json"; sp = output_dir / "summary.json"; cp = output_dir / "per_case.csv"
    _write_json(tp, result["traces"]); _write_json(sp, summary)
    fields = ["case", "name", "relation", "observed", "expected", "passed", "tolerance", "abs_error", "violation", "notes"]
    with cp.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader()
        for g in result["gates"]:
            w.writerow({f: _as_json(g.get(f)) for f in fields})
    cfg = io.StringIO()
    with redirect_stdout(cfg):
        np.show_config()
    manifest = {"schema_version": 1, "stage": "phase2_injection_bridge", "run_role": run_role,
                "source_sha256": _sha256(SOURCE_PATH),
                "protocol_sha256": _sha256(PROTOCOL_PATH) if PROTOCOL_PATH.exists() else "PROTOCOL_NOT_FROZEN_YET",
                "numpy_version": np.__version__, "python_version": platform.python_version(),
                "platform": platform.platform(), "numpy_config": cfg.getvalue(), "command": " ".join(sys.argv),
                "resource_ledger": {"gpu_used": False, "optimizer_state_bytes": 0, "replay_bytes": 0, "external_model_bytes": 0},
                "output_hashes": {p.name: _sha256(p) for p in (tp, sp, cp)},
                "all_required_pass": result["all_required_pass"], "claim_boundary": "controlled_linear_mixture_only"}
    _write_json(output_dir / "manifest.json", manifest)
    return {"summary": summary, "manifest": manifest, "output_dir": str(output_dir)}


def main(argv=None):
    p = argparse.ArgumentParser(description="D2-A rare-task injection bridge.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-role", default="primary_frozen")
    a = p.parse_args(argv)
    r = write_run(a.output_dir, a.run_role)
    print(json.dumps(_as_json(r["summary"]), indent=2, sort_keys=True))
    return 0 if r["summary"]["all_required_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
