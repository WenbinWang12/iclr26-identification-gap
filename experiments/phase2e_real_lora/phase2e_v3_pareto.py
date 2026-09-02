"""D2-E (v3): the REAL decision arena -- nonorthogonal / redundant / mixed stream
with an offline rank-R mean-tail Pareto oracle.

Codex's audit is explicit that the orthogonal R<K stream (phase2e_v2) is a ZERO-SUM
negative control: protecting a rare direction there NECESSARILY evicts a more
frequent one, so no method can raise both rare and frequency-weighted retention and
"beats everything" is impossible by construction.  The honest arena is a stream
with SHARED / REDUNDANT structure, where a good allocator can retain the rare
latent mode WITHOUT paying full price on the frequent ones, and the honest object
is the mean-tail Pareto frontier -- not across-the-board superiority.

This module:
  * builds a nonorthogonal teacher: K latent components with controlled pairwise
    correlation, including a near-redundant pair (so consolidation has real work)
    and one genuinely-distinct RARE component;
  * clocks a stream where each window MIXES 2-4 latent components (removing the
    implicit one-window-one-task boundary of v2);
  * reuses the v2 LABEL-BLIND arms verbatim (they take (cfg, batches, seed) only);
  * computes the offline rank-R Pareto frontier by scalarizing (1-rho)*freq-mean +
    rho*smooth-worst over held-out task losses and sweeping rho, optimizing a real
    rank-R factored adapter by gradient descent from multiple inits;
  * reports each online arm's (freq-weighted retention, worst-group retention) and
    its distance to the offline frontier;
  * evaluates the PRE-REGISTERED stop-rule: does oracle-free FCRA beat
    CVaR-replay-only on the tail here?  If not, the allocator adds nothing and the
    method should collapse to risk-aware replay (no DR-GSA).

Boundary: controlled factored low-rank linear proxy, oracle-free; NOT LLM /
benchmark / real-NLP / nonlinear.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE / "phase2e_v3_pareto.py"

# reuse the audited, label-blind machinery from v2
from phase2e_v2_oracle_free import (  # noqa: E402
    _as_json, _sha256, _add_gate, _grad_M, retention,
    run_dense, run_fcra, run_cvar_replay,
)

CFG = {
    "d": 16,
    "K": 8,
    "R": 4,
    "beta_freq": 1.5,
    "n_windows": 240,
    "batch": 64,
    "holdout": 256,
    "comps_per_window": (2, 4),     # each window mixes this many latent components
    "corr": 0.6,                    # base off-diagonal correlation of latent dirs
    "redundant_pair_angle": 0.15,   # near-collinear pair (radians) -> real redundancy
    "gd_steps": 30,
    "lr": 0.15,
    "noise": 0.05,
    "lam": 1e-3,
    "reuse_thresh": 0.30,
    "consol_slack": 0.20,
    "consol_redundancy_tol": 0.10,
    "res_buffer": 8,
    "p_max": 3,
    "seed_teacher": 20260602,
    "seed_stream": 5252,
    "sweep_seeds": list(range(3100, 3140)),
    "retain_thresh": 0.5,
    "pareto_rhos": [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0],
    "pareto_inits": 4,
    "pareto_steps": 400,
    "pareto_lr": 0.10,
    "smooth_worst_beta": 8.0,
    "gamma_prot": 5.0,          # soft-protection drift-penalty strength
}


# ---------------------------------------------------------------------------
# nonorthogonal / redundant teacher + mixed-component stream
# ---------------------------------------------------------------------------

def build_teacher(cfg, seed):
    """K nonorthogonal input directions with a near-redundant pair; distinct
    output directions.  Returns unit-column V (d x K), U (d x K)."""
    d, K = cfg["d"], cfg["K"]
    rng = np.random.default_rng(seed)
    # start from a random Gaussian basis, then correlate toward a shared anchor
    base = rng.standard_normal((d, K))
    anchor = rng.standard_normal((d, 1))
    anchor /= np.linalg.norm(anchor)
    V = (1.0 - cfg["corr"]) * base + cfg["corr"] * anchor
    # make components 0 and 1 a near-redundant pair (real consolidation work)
    ang = cfg["redundant_pair_angle"]
    perp = base[:, 1] - (base[:, 1] @ V[:, 0]) * V[:, 0] / (np.linalg.norm(V[:, 0]) ** 2 + 1e-12)
    perp /= (np.linalg.norm(perp) + 1e-12)
    V[:, 1] = np.cos(ang) * (V[:, 0] / np.linalg.norm(V[:, 0])) + np.sin(ang) * perp
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    Qu, _ = np.linalg.qr(rng.standard_normal((d, d)))
    U = Qu[:, :K]
    return V, U


def frequencies(K, beta):
    f = np.array([j ** (-beta) for j in range(1, K + 1)], float)
    return f / f.sum()


def make_stream(cfg, seed):
    """Each window is a SET of 2-4 latent components (mixed), drawn so that
    component appearance frequency is heavy-tailed."""
    K = cfg["K"]
    rng = np.random.default_rng(seed)
    freqs = frequencies(K, cfg["beta_freq"])
    lo, hi = cfg["comps_per_window"]
    windows = []
    for _ in range(cfg["n_windows"]):
        m = rng.integers(lo, hi + 1)
        comps = rng.choice(K, size=m, replace=False, p=freqs)
        windows.append(sorted(int(c) for c in comps))
    return windows, freqs


def sample_mixed_batch(cfg, V, U, comps, rng, n=None):
    """Batch whose inputs are a mixture over the window's components; target is the
    sum of each component's rank-one teacher applied to the input.  Inputs are
    normalized to unit RMS row-norm (applied identically to every arm); because the
    teacher is linear and the retention metric is scale-normalized by ||Y||^2, this
    is a well-posedness rescaling only, not a change of problem."""
    d = cfg["d"]
    n = cfg["batch"] if n is None else n
    X = cfg["noise"] * rng.standard_normal((n, d))
    for c in comps:
        g = rng.standard_normal((n, 1))
        X = X + g * V[:, c][None, :]
    rms = np.sqrt(np.mean(np.sum(X ** 2, axis=1))) + 1e-12
    X = X / rms
    Y = np.zeros((n, d))
    for c in comps:
        proj = X @ V[:, c]
        Y = Y + proj[:, None] * U[:, c][None, :]
    return X, Y


def materialize(cfg, V, U, windows, seed):
    rng = np.random.default_rng(seed + 101)
    batches = [sample_mixed_batch(cfg, V, U, comps, rng) for comps in windows]
    # per-COMPONENT held-out set: isolate component c to measure its retention
    hrng = np.random.default_rng(seed + 202)
    holdout = {}
    for c in set(int(x) for w in windows for x in w):
        holdout[c] = sample_mixed_batch(cfg, V, U, [c], hrng, n=cfg["holdout"])
    return batches, windows, holdout


# ---------------------------------------------------------------------------
# offline rank-R mean-tail Pareto oracle
# ---------------------------------------------------------------------------

def _factor_opt(cfg, holdout, comps, weights_fn, seed):
    """Optimize a rank-R factored M to minimize weights_fn(losses)-scalarized
    objective, by GD from several inits.  Returns best M."""
    d, R = cfg["d"], cfg["R"]
    best_M, best_J = None, np.inf
    for it in range(cfg["pareto_inits"]):
        rng = np.random.default_rng(seed + 700 + it)
        B = 0.05 * rng.standard_normal((d, R))
        A = 0.05 * rng.standard_normal((R, d))
        for _ in range(cfg["pareto_steps"]):
            M = B @ A
            losses = np.array([np.mean(np.sum((holdout[c][0] @ M.T - holdout[c][1]) ** 2, axis=1))
                               for c in comps])
            w = weights_fn(losses)
            dM = np.zeros((d, d))
            for wi, c in zip(w, comps):
                dM = dM + wi * _grad_M(M, holdout[c][0], holdout[c][1])
            B, A = B - cfg["pareto_lr"] * (dM @ A.T), A - cfg["pareto_lr"] * (B.T @ dM)
        M = B @ A
        losses = np.array([np.mean(np.sum((holdout[c][0] @ M.T - holdout[c][1]) ** 2, axis=1))
                           for c in comps])
        J = float(weights_fn(losses) @ losses)
        if J < best_J:
            best_J, best_M = J, M
    return best_M


def pareto_frontier(cfg, V, U, holdout, freqs, seed):
    """Trace (freq-weighted retention, worst-group retention) achievable by the best
    rank-R adapter as we sweep the mean<->tail weighting rho."""
    comps = sorted(holdout.keys())
    fw = np.array([freqs[c] for c in comps]); fw = fw / fw.sum()
    beta = cfg["smooth_worst_beta"]
    pts = []
    for rho in cfg["pareto_rhos"]:
        def wfn(losses, rho=rho):
            s = np.exp(beta * (losses - losses.max()))
            s = s / s.sum()                      # smooth-argmax (tail emphasis)
            return (1.0 - rho) * fw + rho * s
        M = _factor_opt(cfg, holdout, comps, wfn, seed)
        ret = {c: retention(M, *holdout[c]) for c in comps}
        fwret = float(sum(freqs[c] * ret[c] for c in comps) / sum(freqs[c] for c in comps))
        worst = float(min(ret.values()))
        pts.append({"rho": rho, "freq_weighted": fwret, "worst_group": worst})
    return pts


def _point_of(cfg, holdout, freqs, M):
    comps = sorted(holdout.keys())
    ret = {c: retention(M, *holdout[c]) for c in comps}
    fwret = float(sum(freqs[c] * ret[c] for c in comps) / sum(freqs[c] for c in comps))
    return fwret, float(min(ret.values())), ret


def dist_to_frontier(pt, frontier):
    """Signed shortfall of a (freq-wtd, worst) point from the frontier: 0 = on/above,
    positive = below.  Distance to nearest frontier point that dominates it."""
    fx, wy = pt
    best = np.inf
    for f in frontier:
        # how far this point is from being dominated by frontier point f
        gap = max(0.0, f["freq_weighted"] - fx) ** 2 + max(0.0, f["worst_group"] - wy) ** 2
        best = min(best, gap)
    return float(np.sqrt(best))


# ---------------------------------------------------------------------------
# arms (reuse v2 label-blind trainers) + metrics on component holdouts
# ---------------------------------------------------------------------------

ARMS = {
    "sequential_dense": lambda c, b, s: run_dense(c, b, s, replay_windows=0),
    "reservoir_replay": lambda c, b, s: run_dense(c, b, s, replay_windows=c["res_buffer"]),
    "cvar_replay": lambda c, b, s: run_cvar_replay(c, b, s),
    "fcra": lambda c, b, s: run_fcra(c, b, s),
    "fcra_soft_protect": lambda c, b, s: run_fcra(c, b, s, soft_protect=True),
    "fcra_no_protect": lambda c, b, s: run_fcra(c, b, s, protect=False),
    "fcra_no_consolidate": lambda c, b, s: run_fcra(c, b, s, consolidate=False),
}


def metrics_for(cfg, windows, holdout, snaps, freqs):
    Mfinal = snaps[-1]
    comps = sorted(holdout.keys())
    counts = {c: sum(c in w for w in windows) for c in comps}
    rare = min(comps, key=lambda c: counts[c])
    ret = {c: retention(Mfinal, *holdout[c]) for c in comps}
    first_idx = {}
    for i, w in enumerate(windows):
        for c in w:
            first_idx.setdefault(c, i)
    learned = {c: retention(snaps[first_idx[c]], *holdout[c]) for c in comps}
    fwret = float(sum(freqs[c] * ret[c] for c in comps) / sum(freqs[c] for c in comps))
    common = [c for c in comps if c != rare]
    return {
        "s_rare": ret[rare],
        "s_common_mean": float(np.mean([ret[c] for c in common])) if common else ret[rare],
        "freq_weighted_retention": fwret,
        "worst_group_retention": float(min(ret.values())),
        "bwt": float(np.mean([ret[c] - learned[c] for c in comps])),
        "rare_comp": int(rare),
        "point": [fwret, float(min(ret.values()))],
    }


def run(cfg, seed):
    V, U = build_teacher(cfg, cfg["seed_teacher"])
    windows, freqs = make_stream(cfg, seed)
    batches, windows, holdout = materialize(cfg, V, U, windows, seed)
    out = {}
    for name, fn in ARMS.items():
        snaps, stats = fn(cfg, batches, seed)
        m = metrics_for(cfg, windows, holdout, snaps, freqs)
        m["stats"] = stats
        out[name] = m
    return V, U, windows, freqs, holdout, out


def build_report(cfg):
    seed = cfg["seed_stream"]
    V, U, windows, freqs, holdout, arms = run(cfg, seed)
    frontier = pareto_frontier(cfg, V, U, holdout, freqs, seed)
    for name in arms:
        pt = arms[name]["point"]
        arms[name]["dist_to_frontier"] = dist_to_frontier(pt, frontier)

    # pre-registered stop-rule: the DECISION arm is the redesigned soft-protect FCRA
    # (task #18 fork), evaluated against CVaR-replay-only on the TAIL (worst-group).
    # Hard-protect fcra is kept only as the pre-redesign reference.
    decision_arm = "fcra_soft_protect"
    fcra_tail = arms[decision_arm]["worst_group_retention"]
    cvar_tail = arms["cvar_replay"]["worst_group_retention"]
    fcra_beats_cvar_tail = fcra_tail > cvar_tail + 0.05

    # sweep
    sweep = {"n_seeds": len(cfg["sweep_seeds"]), "fcra_gt_cvar_tail": 0,
             "fcra_gt_reservoir_tail": 0, "fcra_closer_to_frontier_than_cvar": 0,
             "hard_fcra_gt_cvar_tail": 0}
    for sd in cfg["sweep_seeds"]:
        Vs, Us, ws, fs, ho, a = run(cfg, sd)
        fr = pareto_frontier(cfg, Vs, Us, ho, fs, sd)
        if a[decision_arm]["worst_group_retention"] > a["cvar_replay"]["worst_group_retention"] + 0.05:
            sweep["fcra_gt_cvar_tail"] += 1
        if a[decision_arm]["worst_group_retention"] > a["reservoir_replay"]["worst_group_retention"] + 0.05:
            sweep["fcra_gt_reservoir_tail"] += 1
        if a["fcra"]["worst_group_retention"] > a["cvar_replay"]["worst_group_retention"] + 0.05:
            sweep["hard_fcra_gt_cvar_tail"] += 1
        d_fcra = dist_to_frontier(a[decision_arm]["point"], fr)
        d_cvar = dist_to_frontier(a["cvar_replay"]["point"], fr)
        if d_fcra < d_cvar:
            sweep["fcra_closer_to_frontier_than_cvar"] += 1

    gates = []
    _add_gate(gates, "D2Ev3_frontier_monotone",
              all(frontier[i]["worst_group"] <= frontier[i + 1]["worst_group"] + 0.05
                  for i in range(len(frontier) - 1)),
              True, True, 0, "offline frontier trades mean for worst as rho rises")
    _add_gate(gates, "D2Ev3_fcra_label_blind_ran", arms["fcra"]["stats"]["discover_calls"],
              1, arms["fcra"]["stats"]["discover_calls"] >= 1, 0, relation="ge")
    # NOTE: this is a DECISION report, not a pass/fail-to-publish gate.  The
    # stop-rule result is recorded whether positive or negative.
    decision = {
        "decision_arm": decision_arm,
        "decision_arm_worst_group": fcra_tail,
        "hard_fcra_worst_group": arms["fcra"]["worst_group_retention"],
        "cvar_worst_group": cvar_tail,
        "beats_cvar_on_tail_primary": bool(fcra_beats_cvar_tail),
        "sweep_decision_arm_gt_cvar_tail": sweep["fcra_gt_cvar_tail"],
        "sweep_hard_fcra_gt_cvar_tail": sweep["hard_fcra_gt_cvar_tail"],
        "threshold_needed": int(np.ceil(0.7 * sweep["n_seeds"])),
        "verdict": ("allocator_adds_value" if sweep["fcra_gt_cvar_tail"] >= 0.7 * sweep["n_seeds"]
                    else "allocator_no_value_collapse_to_risk_aware_replay"),
    }
    return {
        "schema_version": 3,
        "stage": "phase2e_v3_pareto",
        "claim_boundary": "controlled_nonorthogonal_redundant_factored_low_rank_proxy_oracle_free",
        "metric_table": {k: {m: v for m, v in arms[k].items() if m != "stats"} for k in arms},
        "fcra_stats": arms["fcra"]["stats"],
        "offline_pareto_frontier": frontier,
        "sweep": sweep,
        "decision": decision,
        "gates": gates,
        "all_required_pass": all(g["passed"] for g in gates),
    }


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
             "fcra_soft_protect", "fcra_no_protect", "fcra_no_consolidate"]
    print(f"=== phase2e v3 (nonorthogonal + Pareto)  role={args.run_role} ===")
    print(f"{'arm':22s} {'rare':>8s} {'freq-wt':>8s} {'worst':>8s} {'dist2front':>11s}")
    for a in order:
        m = rep["metric_table"][a]
        print(f"{a:22s} {m['s_rare']:+8.3f} {m['freq_weighted_retention']:+8.3f} "
              f"{m['worst_group_retention']:+8.3f} {m['dist_to_frontier']:11.4f}")
    print("\noffline frontier (rho, freq-wt, worst):")
    for f in rep["offline_pareto_frontier"]:
        print(f"  rho={f['rho']:.2f}  fw={f['freq_weighted']:+.3f}  worst={f['worst_group']:+.3f}")
    print("\nfcra_stats:", rep["fcra_stats"])
    print("sweep:", rep["sweep"])
    print("DECISION:", rep["decision"])

    if args.output_dir:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "summary.json").write_text(json.dumps(_as_json(rep), indent=2))
        print("wrote", outdir / "summary.json")


if __name__ == "__main__":
    main()
