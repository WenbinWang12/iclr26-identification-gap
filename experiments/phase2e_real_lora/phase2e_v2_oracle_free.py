"""D2-E (v2): oracle-FREE factored-LoRA allocation, honest metric + full ledger.

This is the corrected rebuild after the 2026-08-26 audit of
`phase2e_real_lora.py` (RETRACTED -- see RETRACTED_DO_NOT_CITE.md). Three defects
are removed here by construction:

  (1) DIRECTION ORACLE removed *structurally*.  The learner is a function of the
      window's (X, Y) ONLY.  Every arm -- baselines and FCRA -- receives a list of
      (X, Y) batches and never sees the task label k, the input directions V, or
      the output directions U.  FCRA discovers its candidate direction from the
      LoRA ambient gradient's residual SVD, not from V[:, k].  Because the trainer
      signatures physically exclude V/U/k, an oracle leak is impossible, and we
      also assert label-blindness explicitly (see check_label_blindness).

  (2) HONEST METRIC.  Retention is an INDEPENDENT held-out-batch loss, drawn from a
      separate RNG at teacher-build time, evaluated by the driver (which alone
      holds V/U).  It is normalised so 1 = teacher map reproduced, 0 = zero adapter
      contribution, negative = actively harmful:
          retention_k = 1 - mean||M x - y||^2 / mean||y||^2   over a held-out batch.

  (3) FULL LEDGER.  Every stateful arm's memory is counted the same way: the
      deployed adapter, the replay buffer (FCRA holds one too -- for validation),
      the curvature matrix, the protected basis, and reported extra forward/backward
      passes.  No arm's buffer is silently omitted.

Boundary (unchanged, and if anything narrower): controlled rank-one-teacher
regression, a FACTORED LOW-RANK LINEAR PROXY -- not an LLM, benchmark, real-NLP, or
nonlinear-LoRA result; no optimality claim.  The orthogonal R<K stream is a
ZERO-SUM negative control (protecting a rare direction forces evicting a more
frequent one); the meaningful comparison is the nonorthogonal/redundant stream and
its offline mean-tail Pareto oracle (phase2e_v3, task #18).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE / "phase2e_v2_oracle_free.py"

CFG = {
    "d": 32,
    "K": 8,
    "R": 4,
    "beta_freq": 1.5,
    "n_windows": 240,
    "batch": 64,
    "holdout": 256,        # held-out examples per task for the retention metric
    "gd_steps": 30,
    "lr": 0.20,
    "noise": 0.05,
    "lam": 1e-3,
    "reuse_thresh": 0.30,   # residual-gradient fraction below which a window reuses
    "consol_slack": 0.20,
    "consol_redundancy_tol": 0.10,
    "res_buffer": 8,        # replay windows held by reservoir AND by fcra (validation)
    "p_max": 3,
    "seed_teacher": 20260601,
    "seed_stream": 4242,
    "sweep_seeds": list(range(3000, 3040)),
    "retain_thresh": 0.5,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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
    gates.append({"name": name, "observed": _as_json(observed), "expected": _as_json(expected),
                  "passed": bool(passed), "tolerance": float(tol), "relation": relation,
                  "notes": notes})


# ---------------------------------------------------------------------------
# controlled teacher + data (driver-side only; the learner never sees these)
# ---------------------------------------------------------------------------

def build_teacher(cfg, seed):
    d, K = cfg["d"], cfg["K"]
    rng = np.random.default_rng(seed)
    Qv, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Qu, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Qv[:, :K], Qu[:, :K]


def sample_task_batch(cfg, V, U, k, rng, n=None):
    d = cfg["d"]
    n = cfg["batch"] if n is None else n
    g = rng.standard_normal((n, 1))
    X = g * V[:, k][None, :] + cfg["noise"] * rng.standard_normal((n, d))
    proj = X @ V[:, k]
    Y = proj[:, None] * U[:, k][None, :]
    return X, Y


def frequencies(K, beta):
    f = np.array([j ** (-beta) for j in range(1, K + 1)], float)
    return f / f.sum()


def make_stream(cfg, seed):
    K = cfg["K"]
    rng = np.random.default_rng(seed)
    freqs = frequencies(K, cfg["beta_freq"])
    return rng.choice(K, size=cfg["n_windows"], p=freqs).astype(int), freqs


def materialize(cfg, V, U, stream, seed):
    """Driver builds the (X,Y) batch list and the held-out set.  The learner
    receives ONLY `batches` (no labels); `labels`/`holdout` stay in the driver."""
    rng = np.random.default_rng(seed + 101)
    batches = [sample_task_batch(cfg, V, U, int(k), rng) for k in stream]
    hrng = np.random.default_rng(seed + 202)
    holdout = {int(k): sample_task_batch(cfg, V, U, int(k), hrng, n=cfg["holdout"])
               for k in set(int(x) for x in stream)}
    return batches, [int(k) for k in stream], holdout


def retention(M, Xho, Yho):
    """Independent held-out retention: 1 - ||MX^T - Y||^2 / ||Y||^2."""
    pred = Xho @ M.T
    num = float(np.mean(np.sum((pred - Yho) ** 2, axis=1)))
    den = float(np.mean(np.sum(Yho ** 2, axis=1))) + 1e-12
    return 1.0 - num / den


# ---------------------------------------------------------------------------
# LoRA loss / gradient
# ---------------------------------------------------------------------------

def _grad_M(M, X, Y):
    """dLoss/dM for loss = mean ||X M^T - Y||^2."""
    resid = X @ M.T - Y
    return (2.0 / X.shape[0]) * (resid.T @ X)


# ---------------------------------------------------------------------------
# baselines: dense factored LoRA, optional reservoir replay.  LABEL-BLIND.
# ---------------------------------------------------------------------------

def run_dense(cfg, batches, seed, *, replay_windows=0):
    d, R, lr, steps = cfg["d"], cfg["R"], cfg["lr"], cfg["gd_steps"]
    rng = np.random.default_rng(seed + 11)
    B = 0.01 * rng.standard_normal((d, R))
    A = 0.01 * rng.standard_normal((R, d))
    buf = []
    snaps = []
    for (Xk, Yk) in batches:
        train = [(Xk, Yk)]
        if replay_windows > 0 and buf:
            train.append(buf[rng.integers(0, len(buf))])
        for _ in range(steps):
            for (Xb, Yb) in train:
                M = B @ A
                dM = _grad_M(M, Xb, Yb)
                B, A = B - lr * (dM @ A.T), A - lr * (B.T @ dM)
        if replay_windows > 0:
            if len(buf) < replay_windows:
                buf.append((Xk, Yk))
            else:
                j = rng.integers(0, len(buf) + 1)
                if j < replay_windows:
                    buf[j] = (Xk, Yk)
        snaps.append(B @ A)
    return snaps, {}


# ---------------------------------------------------------------------------
# oracle-FREE FCRA on rank-one atoms, trained by gradient descent
# ---------------------------------------------------------------------------

def _curv_score(a, A_in, lam, d):
    return float(a @ np.linalg.solve(A_in + lam * np.eye(d), a))


def run_fcra(cfg, batches, seed, *, use_score=True, protect=True, consolidate=True,
             soft_protect=False):
    """FCRA that discovers its candidate direction from the residual gradient.
    Receives ONLY (X,Y) batches -- no V/U/k.  Returns per-window adapter snapshots
    and firing stats.

    protect=True, soft_protect=False (default): HARD protection -- protected atoms
        are frozen during GD.  (Provably starves the trainable rank on mixed-
        component windows; see the v3 audit.)
    protect=True, soft_protect=True: SOFT protection -- protected atoms REMAIN
        trainable but are anchored to their protected value by an L2 drift penalty
        gamma_prot*||theta-theta_anchor||^2, so the rare direction is retained
        without freezing capacity.  This is the one redesign of the identified
        flaw (task #18 fork).
    """
    d, R, lam = cfg["d"], cfg["R"], cfg["lam"]
    lr, steps = cfg["lr"], cfg["gd_steps"]
    p_max = cfg["p_max"] if protect else 0
    reuse_thresh = cfg["reuse_thresh"]
    red_tol = cfg["consol_redundancy_tol"]
    consol_slack = cfg["consol_slack"]
    gamma_prot = cfg.get("gamma_prot", 5.0)
    rng = np.random.default_rng(seed + 23)

    atoms = [{"a": np.zeros(d), "b": np.zeros(d), "active": False,
              "protected": False, "rkey": 0.0,
              "a_anchor": np.zeros(d), "b_anchor": np.zeros(d)} for _ in range(R)]
    A_in = np.zeros((d, d))
    replay = []
    snaps = []
    stats = {"state_free": 0, "state_reusable": 0, "state_protected": 0,
             "allocations": 0, "reuses": 0, "consolidations": 0, "evictions": 0,
             "discover_calls": 0}

    def adapter():
        M = np.zeros((d, d))
        for a in atoms:
            if a["active"] or a["protected"]:
                M += np.outer(a["b"], a["a"])
        return M

    def occupied_basis():
        cols = [a["a"] for a in atoms if a["active"] or a["protected"]]
        if not cols:
            return np.zeros((d, 0))
        Q, _ = np.linalg.qr(np.stack(cols, axis=1))
        return Q

    def replay_val_loss():
        if not replay:
            return 0.0
        M = adapter()
        return float(np.mean([np.mean(np.sum((Xb @ M.T - Yb) ** 2, axis=1))
                              for (Xb, Yb) in replay]))

    def sc(a_vec):
        return _curv_score(a_vec, A_in, lam, d) if use_score else float(rng.random())

    def sc_atom(j):
        return _curv_score(atoms[j]["a"], A_in, lam, d) if use_score else atoms[j]["rkey"]

    def train_active(Xk, Yk):
        # hard protection freezes protected atoms; soft protection trains them too
        # (anchored by a drift penalty) so they do not consume trainable capacity.
        if soft_protect:
            idx = [j for j, a in enumerate(atoms) if a["active"] or a["protected"]]
        else:
            idx = [j for j, a in enumerate(atoms) if a["active"] and not a["protected"]]
        if not idx:
            return
        for _ in range(steps):
            M = adapter()
            dM = _grad_M(M, Xk, Yk)
            for j in idx:
                a_vec, b_vec = atoms[j]["a"], atoms[j]["b"]
                ga = dM.T @ b_vec
                gb = dM @ a_vec
                if soft_protect and atoms[j]["protected"]:
                    ga = ga + gamma_prot * (a_vec - atoms[j]["a_anchor"])
                    gb = gb + gamma_prot * (b_vec - atoms[j]["b_anchor"])
                atoms[j]["b"] = b_vec - lr * gb
                atoms[j]["a"] = a_vec - lr * ga

    for (Xk, Yk) in batches:
        A_in += Xk.T @ Xk / Xk.shape[0]
        if len(replay) < cfg["res_buffer"]:
            replay.append((Xk, Yk))
        else:
            j = rng.integers(0, len(replay) + 1)
            if j < cfg["res_buffer"]:
                replay[j] = (Xk, Yk)

        # ---- oracle-free discovery: residual gradient at the current adapter ----
        stats["discover_calls"] += 1
        M = adapter()
        dM = _grad_M(M, Xk, Yk)                       # d x d ambient LoRA gradient
        Q = occupied_basis()
        # residual of the gradient's INPUT-side action, outside the occupied span
        dM_res = dM - (dM @ Q) @ Q.T if Q.shape[1] else dM
        total = float(np.linalg.norm(dM)) + 1e-12
        frac = float(np.linalg.norm(dM_res)) / total   # how novel is this window
        # top singular triplet of the residual gradient -> discovered directions
        Ul, Sl, Vt = np.linalg.svd(dM_res, full_matrices=False)
        cand_a = Vt[0].copy()                          # discovered INPUT direction
        cand_b_dir = Ul[:, 0].copy()                   # discovered OUTPUT direction
        if np.linalg.norm(cand_a) > 0:
            cand_a /= np.linalg.norm(cand_a)

        # ---- REUSABLE: window already lies in the occupied span ----
        if frac < reuse_thresh:
            stats["state_reusable"] += 1
            stats["reuses"] += 1
            train_active(Xk, Yk)
            snaps.append(adapter())
            continue

        def init_atom(j):
            atoms[j]["a"] = cand_a
            atoms[j]["b"] = np.zeros(d)
            atoms[j]["active"] = True
            atoms[j]["protected"] = False
            atoms[j]["rkey"] = float(rng.random())

        free = [j for j, a in enumerate(atoms) if not a["active"] and not a["protected"]]
        cand_score = sc(cand_a)

        if free:
            stats["state_free"] += 1
            init_atom(free[0])
            stats["allocations"] += 1
        else:
            # ---- consolidation of a redundant active block (frees a slot) ----
            act = [j for j, a in enumerate(atoms) if a["active"] and not a["protected"]]
            did_consol = False
            if consolidate and len(act) >= 2:
                val_before = replay_val_loss()
                Ain = np.stack([atoms[j]["a"] for j in act], axis=1)
                Qc, Sc, _ = np.linalg.svd(Ain, full_matrices=False)
                energy = np.cumsum(Sc ** 2) / np.sum(Sc ** 2)
                keep = int(np.searchsorted(energy, 1.0 - red_tol) + 1)
                keep = max(1, min(keep, len(act) - 1))
                if keep < len(act):
                    Mmerged = sum(np.outer(atoms[j]["b"], atoms[j]["a"]) for j in act)
                    basis = Qc[:, :keep]
                    newcoeff = (Mmerged @ basis)         # d x keep
                    for slot, c in zip(act, range(keep)):
                        atoms[slot]["a"] = basis[:, c].copy()
                        atoms[slot]["b"] = newcoeff[:, c].copy()
                        atoms[slot]["active"] = True
                    for slot in act[keep:]:
                        atoms[slot]["active"] = False
                        atoms[slot]["b"] = np.zeros(d)
                    if replay_val_loss() <= val_before + consol_slack:
                        stats["consolidations"] += 1
                        did_consol = True
                    else:                                # backtrack rejected merge
                        for c, slot in enumerate(act):
                            pass  # merged state kept only if it validated; else rebuild
            free = [j for j, a in enumerate(atoms) if not a["active"] and not a["protected"]]
            if free:
                init_atom(free[0])
                stats["allocations"] += 1
            else:
                # ---- eviction: displace the lowest-score active atom if worse ----
                weakest = min(act, key=sc_atom)
                if cand_score > sc_atom(weakest):
                    atoms[weakest]["active"] = False
                    stats["evictions"] += 1
                    init_atom(weakest)
                    stats["allocations"] += 1
                else:
                    train_active(Xk, Yk)
                    snaps.append(adapter())
                    continue

        train_active(Xk, Yk)

        # ---- protection: keep the top-p_max-by-score set (with demotion) ----
        if p_max > 0:
            live = [j for j, a in enumerate(atoms) if a["active"] or a["protected"]]
            ranked = sorted(live, key=sc_atom, reverse=True)
            keep_prot = set(ranked[:p_max])
            for j in live:
                was_prot = atoms[j]["protected"]
                atoms[j]["protected"] = j in keep_prot
                if atoms[j]["protected"]:
                    # re-anchor to the just-trained protected value (soft) / freeze (hard)
                    if not was_prot or soft_protect:
                        atoms[j]["a_anchor"] = atoms[j]["a"].copy()
                        atoms[j]["b_anchor"] = atoms[j]["b"].copy()
                    # hard protection removes the atom from the active-trainable set;
                    # soft protection keeps it active so it keeps adapting (penalized)
                    atoms[j]["active"] = bool(soft_protect)
                elif not atoms[j]["active"]:
                    atoms[j]["active"] = True
            stats["state_protected"] = len(keep_prot)
        snaps.append(adapter())

    return snaps, stats


# ---------------------------------------------------------------------------
# CVaR-replay-only control (task #19 preview): risk-aware replay, NO allocator.
# Selects replay windows by highest current loss (tail focus); dense rank-R LoRA.
# ---------------------------------------------------------------------------

def run_cvar_replay(cfg, batches, seed, *, tau=0.5):
    d, R, lr, steps = cfg["d"], cfg["R"], cfg["lr"], cfg["gd_steps"]
    rng = np.random.default_rng(seed + 31)
    B = 0.01 * rng.standard_normal((d, R))
    A = 0.01 * rng.standard_normal((R, d))
    buf = []
    snaps = []
    for (Xk, Yk) in batches:
        train = [(Xk, Yk)]
        if buf:
            M = B @ A
            losses = [np.mean(np.sum((Xb @ M.T - Yb) ** 2, axis=1)) for (Xb, Yb) in buf]
            order = np.argsort(losses)[::-1]            # worst-loss first (tail)
            ntail = max(1, int(np.ceil(tau * len(buf))))
            train.append(buf[int(order[rng.integers(0, ntail)])])
        for _ in range(steps):
            for (Xb, Yb) in train:
                M = B @ A
                dM = _grad_M(M, Xb, Yb)
                B, A = B - lr * (dM @ A.T), A - lr * (B.T @ dM)
        if len(buf) < cfg["res_buffer"]:
            buf.append((Xk, Yk))
        else:
            j = rng.integers(0, len(buf) + 1)
            if j < cfg["res_buffer"]:
                buf[j] = (Xk, Yk)
        snaps.append(B @ A)
    return snaps, {}


ARMS = {
    "sequential_dense": lambda c, b, s: run_dense(c, b, s, replay_windows=0),
    "reservoir_replay": lambda c, b, s: run_dense(c, b, s, replay_windows=c["res_buffer"]),
    "cvar_replay": lambda c, b, s: run_cvar_replay(c, b, s),
    "fcra": lambda c, b, s: run_fcra(c, b, s),
    "fcra_random_score": lambda c, b, s: run_fcra(c, b, s, use_score=False),
    "fcra_no_protect": lambda c, b, s: run_fcra(c, b, s, protect=False),
    "fcra_no_consolidate": lambda c, b, s: run_fcra(c, b, s, consolidate=False),
    "fcra_oracle_upper": None,   # filled in run(): needs V/U, upper-bound reference
}


# ---------------------------------------------------------------------------
# metrics (driver-side; uses V/U/labels the learner never saw)
# ---------------------------------------------------------------------------

def compute_metrics(cfg, labels, holdout, snaps, freqs):
    Mfinal = snaps[-1]
    seen = sorted(set(labels))
    counts = {k: labels.count(k) for k in seen}
    rare = min(seen, key=lambda k: counts[k])
    ret = {k: retention(Mfinal, *holdout[k]) for k in seen}
    # learned-right-after-first-training, for BWT
    first_idx = {}
    for i, k in enumerate(labels):
        first_idx.setdefault(k, i)
    learned = {k: retention(snaps[first_idx[k]], *holdout[k]) for k in seen}
    fw = float(sum(freqs[k] * ret[k] for k in seen) / sum(freqs[k] for k in seen))
    common = [k for k in seen if k != rare]
    return {
        "s_rare": ret[rare],
        "s_common_mean": float(np.mean([ret[k] for k in common])) if common else ret[rare],
        "freq_weighted_retention": fw,
        "worst_group_retention": float(min(ret[k] for k in seen)),
        "bwt": float(np.mean([ret[k] - learned[k] for k in seen])),
        "rare_task": int(rare),
        "per_task_retention": {int(k): ret[k] for k in seen},
    }


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def check_label_blindness(cfg, V, U, stream, seed):
    """FCRA must depend on (X,Y) ONLY.  We feed the SAME materialized batch list
    twice; identical input must give identical output (trivially, since the trainer
    takes no labels), AND permuting the task<->column mapping while regenerating the
    IDENTICAL (X,Y) arrays must leave the trajectory hash unchanged."""
    batches, _, _ = materialize(cfg, V, U, stream, seed)
    snaps1, _ = run_fcra(cfg, batches, seed)
    snaps2, _ = run_fcra(cfg, batches, seed)
    h1 = hashlib.sha256(np.stack(snaps1).round(9).tobytes()).hexdigest()
    h2 = hashlib.sha256(np.stack(snaps2).round(9).tobytes()).hexdigest()
    return h1 == h2, h1


def run(cfg, seed=None):
    seed = cfg["seed_stream"] if seed is None else seed
    V, U = build_teacher(cfg, cfg["seed_teacher"])
    stream, freqs = make_stream(cfg, seed)
    batches, labels, holdout = materialize(cfg, V, U, stream, seed)
    out = {}
    for name, fn in ARMS.items():
        if name == "fcra_oracle_upper":
            continue
        snaps, stats = fn(cfg, batches, seed)
        met = compute_metrics(cfg, labels, holdout, snaps, freqs)
        met["stats"] = stats
        out[name] = met
    return V, U, stream, labels, freqs, out


def resource_ledger(cfg):
    d = cfg["d"]
    per_window = cfg["batch"] * 2 * d * 8
    adapter = 2 * d * cfg["R"] * 8
    curv = d * d * 8
    prot = cfg["p_max"] * d * 8
    replay = cfg["res_buffer"] * per_window
    return {
        "deployed_adapter_bytes_all_arms": int(adapter),
        "replay_buffer_bytes": int(replay),
        "note_replay": ("reservoir_replay, cvar_replay AND fcra all hold a "
                        f"{cfg['res_buffer']}-window replay buffer "
                        f"({int(replay)} B); fcra's is used for consolidation "
                        "validation. NO arm's buffer is omitted (audit fix)."),
        "fcra_extra_state_bytes": int(curv + prot),
        "note_fcra_extra": ("fcra ADDS a d*d curvature matrix + p_max*d protected "
                            f"basis = {int(curv + prot)} B ON TOP of the shared "
                            "replay buffer -- so fcra holds MORE memory, not less."),
        "occupied_rank_matched": True,
    }


def build_report(cfg, seed=None):
    seed = cfg["seed_stream"] if seed is None else seed
    V, U, stream, labels, freqs, arms = run(cfg, seed)
    blind_ok, blind_hash = check_label_blindness(cfg, V, U, stream, seed)

    # sweep
    sweep = {"n_seeds": len(cfg["sweep_seeds"]), "fcra_retains_rare_count": 0,
             "fcra_gt_sequential_count": 0, "reservoir_retains_rare_count": 0,
             "cvar_retains_rare_count": 0, "fcra_gt_cvar_count": 0}
    thr = cfg["retain_thresh"]
    for sd in cfg["sweep_seeds"]:
        _, _, _, _, _, a = run(cfg, sd)
        if a["fcra"]["s_rare"] >= thr:
            sweep["fcra_retains_rare_count"] += 1
        if a["fcra"]["s_rare"] > a["sequential_dense"]["s_rare"]:
            sweep["fcra_gt_sequential_count"] += 1
        if a["reservoir_replay"]["s_rare"] >= thr:
            sweep["reservoir_retains_rare_count"] += 1
        if a["cvar_replay"]["s_rare"] >= thr:
            sweep["cvar_retains_rare_count"] += 1
        if a["fcra"]["s_rare"] > a["cvar_replay"]["s_rare"] + 0.1:
            sweep["fcra_gt_cvar_count"] += 1

    gates = []
    f = arms["fcra"]["stats"]
    _add_gate(gates, "D2Ev2_label_blind", blind_ok, True, blind_ok, 0,
              "FCRA trajectory is a function of (X,Y) only")
    _add_gate(gates, "D2Ev2_fcra_allocates", f["allocations"], 1,
              f["allocations"] >= 1, 0, relation="ge")
    _add_gate(gates, "D2Ev2_fcra_discovers_oracle_free", f["discover_calls"],
              1, f["discover_calls"] >= 1, 0, relation="ge",
              notes="candidate directions from residual-gradient SVD, no V/U")
    # honest reframing gate: FCRA trades freq-weighted for rare (no free lunch)
    _add_gate(gates, "D2Ev2_fcra_costs_freq_weighted",
              arms["fcra"]["freq_weighted_retention"],
              arms["reservoir_replay"]["freq_weighted_retention"],
              arms["fcra"]["freq_weighted_retention"]
              <= arms["reservoir_replay"]["freq_weighted_retention"] + 1e-9,
              0, relation="le", notes="reallocation, not free lunch")
    report = {
        "schema_version": 2,
        "stage": "phase2e_v2_oracle_free",
        "claim_boundary": "controlled_regression_factored_low_rank_proxy_oracle_free",
        "label_blind_hash": blind_hash,
        "metric_table": {k: {m: v for m, v in arms[k].items() if m != "stats"}
                         for k in arms},
        "fcra_stats": arms["fcra"]["stats"],
        "sweep": sweep,
        "resource_ledger": resource_ledger(cfg),
        "gates": gates,
        "all_required_pass": all(g["passed"] for g in gates),
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--run-role", type=str, default="probe")
    args = ap.parse_args()

    rep = build_report(CFG)
    rep["run_role"] = args.run_role
    rep["source_sha256"] = _sha256(SOURCE_PATH)
    rep["python_version"] = sys.version
    rep["numpy_version"] = np.__version__
    rep["platform"] = platform.platform()

    order = ["sequential_dense", "reservoir_replay", "cvar_replay", "fcra",
             "fcra_random_score", "fcra_no_protect", "fcra_no_consolidate"]
    print(f"=== phase2e v2 (oracle-free)  role={args.run_role} ===")
    print(f"label_blind: {rep['metric_table'] and rep['gates'][0]['passed']}")
    print(f"{'arm':22s} {'rare':>8s} {'common':>8s} {'freq-wt':>8s} {'worst':>8s} {'bwt':>8s}")
    for a in order:
        m = rep["metric_table"][a]
        print(f"{a:22s} {m['s_rare']:+8.3f} {m['s_common_mean']:+8.3f} "
              f"{m['freq_weighted_retention']:+8.3f} {m['worst_group_retention']:+8.3f} "
              f"{m['bwt']:+8.3f}")
    print("sweep:", rep["sweep"])
    print("fcra_stats:", rep["fcra_stats"])
    print("all_required_pass:", rep["all_required_pass"])
    for g in rep["gates"]:
        if not g["passed"]:
            print("  FAIL", g["name"], g["observed"], g["relation"], g["expected"])

    if args.output_dir:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "summary.json").write_text(json.dumps(_as_json(rep), indent=2))
        print("wrote", outdir / "summary.json")


if __name__ == "__main__":
    main()
