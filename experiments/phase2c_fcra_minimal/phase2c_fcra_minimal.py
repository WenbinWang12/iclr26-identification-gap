"""D2-C: minimal FCRA vs. capacity, the allocation-recovers-interference bridge.

Motivation (allocation vs. capacity). Huang et al. (arXiv:2605.29548) show that
larger models retain rare tasks through *reduced interference*: once enough
capacity (width) is devoted to common tasks, common-task gradients weaken and
stop overwriting slowly-accumulated rare-task features. Their remedy is scale.
Continual PEFT cannot buy scale: the backbone is frozen, one adapter is
deployed, and the total rank budget is fixed and does not grow with the stream.

This controlled experiment asks the resulting question directly: *when capacity
cannot grow, can a fixed budget recover the reduced-interference benefit purely
by how it is allocated over time?* We run a clocked, task-identity-free stream
of Huang-style orthogonal tasks arriving at their natural (heavy-tailed)
frequency, with a fixed occupied-subspace budget R < K, and compare three arms
under matched budget and matched replay memory:

  1. sequential_dense   -- naive continual updates rotate all R directions toward
                           whatever arrived recently (the "small model" / naive
                           continual baseline). Frequent tasks dominate; the rare
                           direction is overwritten.
  2. reservoir_replay    -- matched-capacity global reservoir replay. The rare
                           task is under-represented in the buffer by frequency,
                           so replay does not rescue it.
  3. fcra                -- curvature-aware allocation. It scores a candidate
                           direction by current gain per unit *historical*
                           curvature, z^T (A + lambda I)^{-1} z; a rarely-hit
                           direction has small historical curvature and therefore
                           a HIGH score, so it wins a protected slot, and later
                           common updates cannot rotate it out. This makes
                           Huang's reduced-interference mechanism *active* rather
                           than a passive by-product of scale.

The protection rule is not hand-tuned to the rare task: because the accumulated
curvature is A = sum_k n_k w_k w_k^T with orthonormal w_k, the score of task k is
exactly 1/(n_k + lambda), which is highest for the least-frequent task. FCRA
protects the rarest-seen directions as a consequence of the score alone.

Authorized claim: on this controlled linear-mixture continual stream, a fixed
budget with curvature-aware allocation recovers rare-task retention that dense
sequential updating and matched reservoir replay lose, and the curvature-weighted
measures (kappa_G, d_G) predict which arm retains the rare task. NOT authorized:
OLMo, real LLMs, real LoRA training, benchmark performance, or any claim that
FCRA is optimal or that the mechanism transfers to nonlinear networks.
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
PROTOCOL_PATH = ROOT / "notes" / "phase2c_fcra_protocol.md"
SOURCE_PATH = HERE / "phase2c_fcra_minimal.py"

# Frozen numerical policy (mirrors phase2_injection).
EXACT_ATOL = 2e-11
RANK_REL_TOL = 1e-10

# Frozen stream / model configuration.
CFG = {
    "d": 16,          # ambient dimension
    "K": 8,           # number of orthogonal tasks
    "R": 4,           # fixed occupied-subspace budget (R < K forces competition)
    "p_max": 3,       # max protected slots (>=1 active slot kept for plasticity)
    "beta_freq": 1.5, # frequency exponent pi_k ~ k^{-beta}
    "T": 600,         # stream length (clocked windows)
    "eta": 0.5,       # Grassmann step size
    "lam": 1e-3,      # curvature damping in the allocation score
    "res_buffer": 8,  # reservoir buffer capacity (fixed replay memory)
    "res_thresh": 0.30,  # residual novelty needed to consider allocating/protecting
    "seed_dirs": 20260529,
    "seed_stream": 12345,
    # Robustness sweep: 40 independent stream draws (task geometry fixed by
    # seed_dirs; only the arrival stream is reseeded). Reported honestly -- we do
    # NOT claim reservoir always loses the rare task, only that FCRA matches or
    # beats reservoir on every seed and strictly beats sequential on every seed.
    "sweep_seeds": list(range(2000, 2040)),
    "retain_thresh": 0.5,  # s_rare >= this counts as "rare task retained"
}


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


# ---------------------------------------------------------------------------
# Stream and geometry helpers (self-contained; mirror phase2_injection).
# ---------------------------------------------------------------------------

def _build_directions(d, K, seed):
    """K orthonormal task directions w_k in R^d (columns of a QR factor)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q[:, :K]  # d x K, orthonormal columns


def _frequencies(K, beta):
    f = np.array([k ** (-beta) for k in range(1, K + 1)], float)
    return f / f.sum()


def _sample_stream(K, freqs, T, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(K, size=T, p=freqs).astype(int)


def _orthonormalize(M):
    """Column-orthonormalize via QR, dropping numerically-zero columns."""
    if M.shape[1] == 0:
        return M
    Q, R = np.linalg.qr(M)
    keep = np.abs(np.diag(R)) > RANK_REL_TOL * max(1.0, float(np.abs(np.diag(R)).max()))
    return Q[:, keep]


def _project_out(vecs, basis):
    """Remove the component of columns of `vecs` lying in span(basis)."""
    if basis.shape[1] == 0:
        return vecs
    return vecs - basis @ (basis.T @ vecs)


def _grassmann_step(active, w, eta):
    """One Riemannian ascent step of Tr(U^T (w w^T) U) on the active columns,
    then re-orthonormalize. Rotates active directions toward w."""
    if active.shape[1] == 0:
        return active
    P = active @ active.T
    grad = 2.0 * (np.eye(active.shape[0]) - P) @ np.outer(w, w) @ active
    stepped = active + eta * grad
    return _orthonormalize(stepped)


def _retained_signal(U, w):
    """s = ||U^T w||^2 for unit w (= Tr(P_U w w^T)/Tr(w w^T))."""
    if U.shape[1] == 0:
        return 0.0
    return float(np.sum((U.T @ w) ** 2))


def _kappa_G(M_common, U):
    """Curvature-weighted residual tail of the common block beyond span(U)."""
    d = M_common.shape[0]
    P = U @ U.T if U.shape[1] else np.zeros((d, d))
    return 0.5 * float(np.trace((np.eye(d) - P) @ M_common))


def _whitening_operator(A_amb, lam):
    """Symmetric damped-curvature whitening operator W = (A_amb + lam I)^{1/2},
    the embedding phi_t(U) = W U required by the theory (theory.tex, d_t
    definition). Returned as a symmetric PSD matrix."""
    d = A_amb.shape[0]
    vals, vecs = np.linalg.eigh(A_amb + lam * np.eye(d))
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def _dG_to_rare(U, rare_dir, whiten):
    """Curvature-WHITENED principal-angle distance from span(U) to the rare
    direction, d_t = (sum_i sin^2 theta_i)^{1/2}, with angles measured after the
    theory's whitening embedding phi_t(.) = W . (theory.tex). This is NOT a
    monotone transform of the raw retention ||U^T rare_dir||^2: the whitening
    operator W = (A_amb + lam I)^{1/2} stretches directions by accumulated
    historical curvature, so d_t reflects how well the occupied subspace covers
    the rare direction *in the curvature metric the capacity theorem uses*."""
    if U.shape[1] == 0:
        return 1.0  # sin(pi/2) = 1: fully expressed as chordal distance
    WU = _orthonormalize(whiten @ U)
    wr = whiten @ rare_dir
    wr = wr / (np.linalg.norm(wr) + 1e-12)
    if WU.shape[1] == 0:
        return 1.0
    cos = float(np.linalg.svd(WU.T @ wr.reshape(-1, 1), compute_uv=False)[0])
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.sqrt(max(0.0, 1.0 - cos * cos)))  # sin theta = chordal d_t


# ---------------------------------------------------------------------------
# Continual arms.
# ---------------------------------------------------------------------------

def _full_basis(protected, active):
    cols = []
    if protected.shape[1]:
        cols.append(protected)
    if active.shape[1]:
        cols.append(active)
    if not cols:
        return np.zeros((protected.shape[0], 0))
    return np.concatenate(cols, axis=1)


def run_sequential_dense(W, freqs, stream, cfg):
    """All R directions are active; every window rotates them toward w_{k_t}."""
    d, R, eta = cfg["d"], cfg["R"], cfg["eta"]
    rng = np.random.default_rng(cfg["seed_dirs"] + 1)
    active = _orthonormalize(rng.standard_normal((d, R)))
    for k in stream:
        active = _grassmann_step(active, W[:, k], eta)
    return active, np.zeros((d, 0))


def run_reservoir_replay(W, freqs, stream, cfg):
    """Dense updates plus matched-capacity reservoir replay. The buffer stores
    arriving task ids by reservoir sampling (frequency-weighted), so the rare
    task is rarely in the buffer and replay does not rescue it."""
    d, R, eta, B = cfg["d"], cfg["R"], cfg["eta"], cfg["res_buffer"]
    rng = np.random.default_rng(cfg["seed_dirs"] + 2)
    active = _orthonormalize(rng.standard_normal((d, R)))
    buffer = []  # reservoir of task ids
    seen = 0
    for k in stream:
        active = _grassmann_step(active, W[:, k], eta)
        # reservoir sampling into a fixed buffer of capacity B
        seen += 1
        if len(buffer) < B:
            buffer.append(int(k))
        else:
            j = int(rng.integers(0, seen))
            if j < B:
                buffer[j] = int(k)
        # one replay step toward a uniformly sampled buffered direction
        if buffer:
            kb = int(buffer[int(rng.integers(0, len(buffer)))])
            active = _grassmann_step(active, W[:, kb], eta)
    return active, np.zeros((d, 0))


def run_fcra(W, freqs, stream, cfg):
    """Curvature-aware allocation with protection and recycling.

    A = sum of past w w^T (historical curvature). Score of a candidate direction
    w is w^T (A + lambda I)^{-1} w = 1/(n_k + lambda) for orthonormal tasks: the
    rarest-seen direction scores highest. A novel, high-residual, high-score
    direction wins a protected slot; when protected slots are full it evicts the
    most-common (lowest-score) protected direction. Active (unprotected) slots do
    the usual dense rotation toward the current task."""
    d, R, eta = cfg["d"], cfg["R"], cfg["eta"]
    lam, p_max, res_thresh = cfg["lam"], cfg["p_max"], cfg["res_thresh"]
    rng = np.random.default_rng(cfg["seed_dirs"] + 3)

    A = np.zeros((d, d))
    protected_dirs = []   # list of unit vectors (NO task id -- task-identity-free)
    n_active = R - 0      # active columns; shrinks as protected grows
    active = _orthonormalize(rng.standard_normal((d, R)))

    def _prot_matrix():
        if not protected_dirs:
            return np.zeros((d, 0))
        return np.stack(protected_dirs, axis=1)

    def _curv_score(v, A_mat):
        # Curvature-aware score of a *direction*: v^T (A + lam I)^{-1} v.
        # High => this direction has accumulated little historical curvature, i.e.
        # it has been rarely served. Computed from the stored vector alone; no
        # task id or count is consulted, so eviction is task-identity-free.
        return float(v @ np.linalg.solve(A_mat + lam * np.eye(d), v))

    for k in stream:
        w = W[:, k]
        A += np.outer(w, w)
        prot = _prot_matrix()

        # residual novelty of w against everything currently occupied
        occupied = _full_basis(prot, active)
        resid = w - occupied @ (occupied.T @ w) if occupied.shape[1] else w
        resid_norm = float(np.linalg.norm(resid))

        # curvature-aware score of the candidate direction: high => rarely served
        wn = w / (np.linalg.norm(w) + 1e-12)
        score = _curv_score(wn, A)

        allocate = resid_norm >= res_thresh
        if allocate:
            if len(protected_dirs) < p_max:
                protected_dirs.append(wn)
            else:
                # eviction: recompute each protected direction's OWN curvature
                # score from its stored vector (label-free). Evict the slot with
                # the lowest score (the most-served direction) if the candidate
                # is rarer-served than it.
                prot_scores = [_curv_score(v, A) for v in protected_dirs]
                j_min = int(np.argmin(prot_scores))
                if score > prot_scores[j_min] + 1e-12:
                    protected_dirs[j_min] = wn
            # rebuild active slots orthogonal to the (possibly new) protected set,
            # holding the TOTAL occupied rank at R: n_active = R - #protected.
            prot = _prot_matrix()
            n_active = max(0, R - prot.shape[1])
            active = _orthonormalize(_project_out(active, prot))
            if active.shape[1] < n_active:
                pad = _project_out(rng.standard_normal((d, n_active - active.shape[1])), prot)
                pad = _orthonormalize(_project_out(pad, active)) if active.shape[1] else _orthonormalize(pad)
                active = np.concatenate([active, pad], axis=1) if active.shape[1] else pad
            active = active[:, :n_active]  # truncate: protected + active <= R
        else:
            # ordinary plastic update on active slots, kept orthogonal to protected
            prot = _prot_matrix()
            n_active = max(0, R - prot.shape[1])
            active = _grassmann_step(active, w, eta)
            active = _orthonormalize(_project_out(active, prot))
            active = active[:, :n_active]  # protected + active <= R

    return active, _prot_matrix()


ARMS = {
    "sequential_dense": run_sequential_dense,
    "reservoir_replay": run_reservoir_replay,
    "fcra": run_fcra,
}


def _run_one_stream(W, cfg, stream):
    """Run all arms on one arrival stream; return per-arm metrics dict."""
    d, K, R, lam = cfg["d"], cfg["K"], cfg["R"], cfg["lam"]
    freqs = _frequencies(K, cfg["beta_freq"])
    # The rare task is defined task-identity-free: the EMPIRICALLY least-frequent
    # direction in this finite stream, not a nominal label. In a heavy-tailed
    # draw the nominal tail index need not be the rarest realized task, and the
    # method sees no labels -- it must protect whatever is actually starved.
    counts = np.bincount(stream, minlength=K)
    rare = int(np.argmin(counts))
    rare_dir = W[:, rare]
    common_ids = [k for k in range(K) if k != rare]
    # common block curvature (empirical), used for kappa_G / theory tie-in
    M_common = sum((counts[k] / len(stream)) * np.outer(W[:, k], W[:, k]) for k in common_ids)
    # Accumulated ambient historical curvature A_amb = sum_k n_k w_k w_k^T, used
    # to build the whitening operator W = (A_amb + lam I)^{1/2} for d_t.
    A_amb = sum(float(counts[k]) * np.outer(W[:, k], W[:, k]) for k in range(K))
    whiten = _whitening_operator(A_amb, lam)

    arm_out = {}
    for name, fn in ARMS.items():
        active, protected = fn(W, freqs, stream, cfg)
        U = _full_basis(protected, active)
        s_rare = _retained_signal(U, rare_dir)
        s_common = [_retained_signal(U, W[:, k]) for k in common_ids]
        arm_out[name] = {
            "occupied_rank": int(U.shape[1]),
            "protected_rank": int(protected.shape[1]),
            "s_rare": s_rare,
            "s_common_mean": float(np.mean(s_common)),
            "kappa_G": _kappa_G(M_common, U),
            "dG_to_rare": _dG_to_rare(U, rare_dir, whiten),
        }
    arm_out["_meta"] = {"rare_task": rare, "rare_count": int(counts[rare])}
    return arm_out


def _sweep(W, cfg):
    """Run every arm across the frozen 40-seed sweep; aggregate honestly.

    We report per-seed retention and three ordering statistics:
      - fcra_ge_reservoir: fraction of seeds with s_rare(fcra) >= s_rare(reservoir)
      - fcra_gt_sequential: fraction of seeds with s_rare(fcra) > s_rare(sequential)
      - reservoir_retains: fraction of seeds where reservoir alone retains rare
    The last is reported precisely because it is NOT zero: reservoir sometimes
    rescues the rare task by chance; the claim is the pairwise ordering, not that
    reservoir always fails.
    """
    freqs = _frequencies(cfg["K"], cfg["beta_freq"])
    rt = cfg["retain_thresh"]
    per_seed = []
    for s in cfg["sweep_seeds"]:
        stream = _sample_stream(cfg["K"], freqs, cfg["T"], s)
        arm = _run_one_stream(W, cfg, stream)
        per_seed.append({
            "seed": int(s),
            "rare_task": arm["_meta"]["rare_task"],
            "s_rare_sequential": arm["sequential_dense"]["s_rare"],
            "s_rare_reservoir": arm["reservoir_replay"]["s_rare"],
            "s_rare_fcra": arm["fcra"]["s_rare"],
            "dG_sequential": arm["sequential_dense"]["dG_to_rare"],
            "dG_fcra": arm["fcra"]["dG_to_rare"],
        })
    n = len(per_seed)
    fcra_ge_res = sum(p["s_rare_fcra"] >= p["s_rare_reservoir"] - 1e-9 for p in per_seed)
    fcra_gt_seq = sum(p["s_rare_fcra"] > p["s_rare_sequential"] + 1e-9 for p in per_seed)
    fcra_retains = sum(p["s_rare_fcra"] >= rt for p in per_seed)
    res_retains = sum(p["s_rare_reservoir"] >= rt for p in per_seed)
    seq_retains = sum(p["s_rare_sequential"] >= rt for p in per_seed)
    res_retain_seeds = [p["seed"] for p in per_seed if p["s_rare_reservoir"] >= rt]
    # d_t ordering agrees with retention ordering (theory tie-in), reported as a
    # fraction -- NOT as independent evidence, since d_t is computed from the same
    # occupied subspace. It is a geometry check that the curvature-whitened
    # distance and the retention move together, not a separate predictor.
    dG_agrees = sum(
        (p["dG_fcra"] <= p["dG_sequential"] + 1e-9) == (p["s_rare_fcra"] >= p["s_rare_sequential"] - 1e-9)
        for p in per_seed
    )
    return {
        "n_seeds": n,
        "seeds": list(cfg["sweep_seeds"]),
        "fcra_ge_reservoir_count": int(fcra_ge_res),
        "fcra_gt_sequential_count": int(fcra_gt_seq),
        "fcra_retains_rare_count": int(fcra_retains),
        "reservoir_retains_rare_count": int(res_retains),
        "sequential_retains_rare_count": int(seq_retains),
        "reservoir_retains_rare_seeds": res_retain_seeds,
        "dG_ordering_agrees_count": int(dG_agrees),
        "retain_thresh": rt,
        "per_seed": per_seed,
    }


def check_allocation_recovers_retention() -> tuple[dict, dict]:
    cfg = CFG
    d, K, R, lam = cfg["d"], cfg["K"], cfg["R"], cfg["lam"]
    W = _build_directions(d, K, cfg["seed_dirs"])
    freqs = _frequencies(K, cfg["beta_freq"])
    stream = _sample_stream(K, freqs, cfg["T"], cfg["seed_stream"])
    counts = np.bincount(stream, minlength=K)
    rare = int(np.argmin(counts))
    rare_dir = W[:, rare]
    common_ids = [k for k in range(K) if k != rare]
    arm_out = _run_one_stream(W, cfg, stream)
    arm_out.pop("_meta", None)

    # Robustness sweep across 40 frozen stream seeds (reported honestly).
    sweep = _sweep(W, cfg)

    seq = arm_out["sequential_dense"]
    res = arm_out["reservoir_replay"]
    fcra = arm_out["fcra"]

    gates = []
    # Budget matched: every arm occupies at most R directions.
    budget_ok = all(v["occupied_rank"] <= R for v in arm_out.values())
    _add_gate(gates, "D2C_budget_matched_all_arms", max(v["occupied_rank"] for v in arm_out.values()), R,
              budget_ok, 0, relation="le", notes="every arm occupies at most R directions (fixed budget)")

    # G_A: sequential loses the rare task under the fixed budget (the problem).
    _add_gate(gates, "D2C_sequential_loses_rare", seq["s_rare"], 0.2, seq["s_rare"] < 0.2, 0.0,
              relation="le", notes=f"sequential rare retention {seq['s_rare']:.3f} < 0.2")

    # G_B: FCRA recovers rare retention well above sequential (core positive result).
    _add_gate(gates, "D2C_fcra_recovers_rare", fcra["s_rare"], seq["s_rare"] + 0.4,
              fcra["s_rare"] >= seq["s_rare"] + 0.4, 0.0, relation="ge",
              notes=f"fcra rare {fcra['s_rare']:.3f} >= sequential {seq['s_rare']:.3f} + 0.4")

    # G_C: FCRA is at least as good as matched-capacity reservoir replay on rare.
    _add_gate(gates, "D2C_fcra_ge_reservoir_rare", fcra["s_rare"], res["s_rare"] - 1e-9,
              fcra["s_rare"] >= res["s_rare"] - 1e-9, 0.0, relation="ge",
              notes=f"fcra rare {fcra['s_rare']:.3f} >= reservoir {res['s_rare']:.3f} at matched budget")

    # G_D: FCRA keeps meaningful plasticity (common retention not destroyed).
    _add_gate(gates, "D2C_fcra_keeps_plasticity", fcra["s_common_mean"], 0.2,
              fcra["s_common_mean"] >= 0.2, 0.0, relation="ge",
              notes=f"fcra common-task retention {fcra['s_common_mean']:.3f} >= 0.2")

    # G_E: theory tie-in -- the curvature-WHITENED Grassmann distance d_t of the
    # arm that retains the rare task is smaller (theory.tex whitening embedding).
    # This is a geometry consistency check, NOT independent evidence: d_t is
    # computed from the same occupied subspace as the retention. The whitening
    # (A_amb+lam I)^{1/2} makes d_t distinct from the raw retention, but the two
    # are still coupled, so we phrase this as "d_t is consistent with", not
    # "d_t predicts", retention.
    dg_consistent = fcra["dG_to_rare"] <= seq["dG_to_rare"] + 1e-9
    _add_gate(gates, "D2C_dG_consistent_with_retention", fcra["dG_to_rare"], seq["dG_to_rare"],
              dg_consistent, 0.0, relation="le",
              notes=f"fcra whitened d_t {fcra['dG_to_rare']:.3f} <= sequential {seq['dG_to_rare']:.3f}")

    # G_F: robustness across the 40-seed sweep -- FCRA matches-or-beats reservoir
    # on EVERY seed (pairwise ordering, not "reservoir always fails").
    _add_gate(gates, "D2C_sweep_fcra_ge_reservoir_all", sweep["fcra_ge_reservoir_count"], sweep["n_seeds"],
              sweep["fcra_ge_reservoir_count"] == sweep["n_seeds"], 0, relation="ge",
              notes=f"fcra >= reservoir on {sweep['fcra_ge_reservoir_count']}/{sweep['n_seeds']} seeds")

    # G_G: FCRA strictly beats sequential on EVERY seed.
    _add_gate(gates, "D2C_sweep_fcra_gt_sequential_all", sweep["fcra_gt_sequential_count"], sweep["n_seeds"],
              sweep["fcra_gt_sequential_count"] == sweep["n_seeds"], 0, relation="ge",
              notes=f"fcra > sequential on {sweep['fcra_gt_sequential_count']}/{sweep['n_seeds']} seeds; "
                    f"reservoir alone retains rare on only {sweep['reservoir_retains_rare_count']}/{sweep['n_seeds']} "
                    f"(seeds {sweep['reservoir_retains_rare_seeds']})")

    return _record_case("allocation_recovers_retention", gates, {
        "config": cfg, "rare_task": rare, "freqs": freqs.tolist(),
        "stream_len": int(len(stream)), "arms": arm_out, "sweep": sweep,
    })


def run_all_checks():
    results = [check_allocation_recovers_retention()]
    summaries = [r[0] for r in results]
    traces = {r[1]["case"]: r[1] for r in results}
    gates = [g for c in summaries for g in c["gates"]]
    failed = [g for g in gates if not g["passed"]]
    return {"schema_version": 1, "stage": "phase2c_fcra_minimal",
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
    trace = result["traces"]["allocation_recovers_retention"]
    arms = trace["arms"]
    summary["headline"] = {name: {"s_rare": v["s_rare"], "s_common_mean": v["s_common_mean"],
                                   "kappa_G": v["kappa_G"], "dG_to_rare": v["dG_to_rare"],
                                   "occupied_rank": v["occupied_rank"], "protected_rank": v["protected_rank"]}
                           for name, v in arms.items()}
    sw = trace["sweep"]
    summary["sweep_headline"] = {
        "n_seeds": sw["n_seeds"],
        "fcra_ge_reservoir_count": sw["fcra_ge_reservoir_count"],
        "fcra_gt_sequential_count": sw["fcra_gt_sequential_count"],
        "fcra_retains_rare_count": sw["fcra_retains_rare_count"],
        "reservoir_retains_rare_count": sw["reservoir_retains_rare_count"],
        "sequential_retains_rare_count": sw["sequential_retains_rare_count"],
        "reservoir_retains_rare_seeds": sw["reservoir_retains_rare_seeds"],
    }
    tp = output_dir / "traces.json"; sp = output_dir / "summary.json"; cp = output_dir / "per_case.csv"
    _write_json(tp, result["traces"]); _write_json(sp, summary)
    fields = ["case", "name", "relation", "observed", "expected", "passed", "tolerance", "abs_error", "violation", "notes"]
    with cp.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader()
        for g in result["gates"]:
            w.writerow({f: _as_json(g.get(f)) for f in fields})
    cfg_str = io.StringIO()
    with redirect_stdout(cfg_str):
        np.show_config()
    # Honest per-arm auxiliary-state accounting in float64 BYTES. These are NOT
    # matched across arms and we do not claim they are: reservoir stores B task
    # ids; FCRA stores a d x d curvature matrix A, p_max protected d-vectors, and
    # solves a d x d system each window. We report each arm's peak auxiliary state
    # so the reader can see FCRA's memory/compute overhead explicitly rather than
    # hiding it behind a "matched replay" label.
    d = CFG["d"]
    F8 = 8  # bytes per float64
    aux_bytes = {
        "sequential_dense": {"active_basis_bytes": d * CFG["R"] * F8, "replay_state_bytes": 0,
                             "curvature_matrix_bytes": 0, "note": "R active directions only"},
        "reservoir_replay": {"active_basis_bytes": d * CFG["R"] * F8,
                             "replay_state_bytes": CFG["res_buffer"] * 8,  # B int64 task ids
                             "curvature_matrix_bytes": 0,
                             "note": f"buffer of {CFG['res_buffer']} int64 task ids"},
        "fcra": {"active_basis_bytes": d * CFG["R"] * F8, "replay_state_bytes": 0,
                 "curvature_matrix_bytes": d * d * F8,  # A matrix
                 "protected_basis_bytes": CFG["p_max"] * d * F8,
                 "per_window_solve_flops_dxd": True,
                 "note": "d x d curvature A + p_max protected vecs + per-window (A+lam I)^{-1} solve; "
                         "NOT matched to replay -- FCRA uses more auxiliary state/compute"},
    }
    manifest = {"schema_version": 1, "stage": "phase2c_fcra_minimal", "run_role": run_role,
                "source_sha256": _sha256(SOURCE_PATH),
                "protocol_sha256": _sha256(PROTOCOL_PATH) if PROTOCOL_PATH.exists() else "PROTOCOL_NOT_FROZEN_YET",
                "numpy_version": np.__version__, "python_version": platform.python_version(),
                "platform": platform.platform(), "numpy_config": cfg_str.getvalue(), "command": " ".join(sys.argv),
                "resource_ledger": {"gpu_used": False, "optimizer_state_bytes": 0,
                                     "external_model_bytes": 0,
                                     "occupied_rank_matched": True,
                                     "auxiliary_state_matched": False,
                                     "per_arm_auxiliary_bytes": aux_bytes},
                "output_hashes": {p.name: _sha256(p) for p in (tp, sp, cp)},
                "all_required_pass": result["all_required_pass"], "claim_boundary": "controlled_linear_mixture_only"}
    _write_json(output_dir / "manifest.json", manifest)
    return {"summary": summary, "manifest": manifest, "output_dir": str(output_dir)}


def main(argv=None):
    p = argparse.ArgumentParser(description="D2-C minimal FCRA vs. capacity.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-role", default="primary_frozen")
    a = p.parse_args(argv)
    r = write_run(a.output_dir, a.run_role)
    print(json.dumps(_as_json(r["summary"]), indent=2, sort_keys=True))
    return 0 if r["summary"]["all_required_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
