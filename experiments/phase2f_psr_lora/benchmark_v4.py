"""v4 benchmark: a well-posed fixed-rank continual-regression arena.

The retracted v3 stream was ill-posed in three ways the audit identified:
  * its "redundant" pair had parallel INPUTS but orthogonal OUTPUTS -> the two
    rank-one maps were nearly orthogonal, i.e. a CONFLICT, not mergeable knowledge;
  * the target was a per-window ad-hoc sum of components, so no single stationary
    operator existed for a routerless adapter to aim at;
  * the "offline Pareto oracle" was 400 steps of nonconvex GD sharing the eval data.

v4 fixes all three by construction:

  ONE global teacher.  There is a single target operator T* with y = T* x + noise
  for EVERY source.  Latent sources differ only in their INPUT distribution
  x ~ D_c (a direction v_c in input space).  A single routerless rank-R adapter
  therefore has a well-defined target (T*), and per-source retention is just how
  well the adapter reproduces T* on that source's inputs.  Because
      L_c(M) = E_{x~D_c} || (M - T*) x ||^2 = tr((M-T*) Sigma_c (M-T*)^T),
  every per-source loss is an exact convex quadratic in M, and for any nonnegative
  weighting w the best rank-R adapter has a CLOSED FORM (weighted reduced-rank
  regression, Eckart-Young after whitening) -- a genuine near-oracle, not GD.

  Rare source  = a rarely-clocked input direction v_c.  Fixed rank R < K forces the
  adapter to keep only R of the K singular directions of T*, so "which sources are
  retained" is literally the spectral-tail capacity floor of the theory.

Three regimes:
  1. compressible      rank(T*) <= R : no capacity floor; only online error is at
                                       stake; mean-optimal reference ~ perfect.
  2. capacity_limited  rank(T*)  = K > R : genuine spectral floor; real mean-tail
                                       tradeoff; includes a truly redundant source
                                       pair (same v AND same u) so consolidation has
                                       real work.
  3. conflicting       per-source teachers T_c disagree on shared inputs : no single
                                       operator satisfies all even at full rank ->
                                       impossibility negative control.

Offline reference: for a sweep of source weightings, the closed-form weighted-RRR
rank-R optimum, FIT on an independent oracle-train sample and EVALUATED on a
separate test sample; the Pareto-dominant set of the realized (mean, worst) test
points is an inner bound on the best achievable rank-R frontier.

Boundary: controlled linear factored proxy, oracle-free for the learner; NOT an
LLM / benchmark / nonlinear result.
"""

from __future__ import annotations

import numpy as np

import harness as H


# ---------------------------------------------------------------------------
# teacher + sources
# ---------------------------------------------------------------------------

def _orthonormal(d, n, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q[:, :n]


def _correlated_dirs(d, K, corr, rng):
    """K unit input directions sharing a common anchor with strength `corr`, so
    sources OVERLAP (serving a common direction partially serves a rare one).  This
    shared structure is what makes lifting the tail cost less than a full slot; with
    orthonormal directions the tail is un-liftable (pure zero-sum)."""
    base = rng.standard_normal((d, K))
    anchor = rng.standard_normal((d, 1)); anchor /= np.linalg.norm(anchor)
    V = (1.0 - corr) * base + corr * anchor
    return V / np.linalg.norm(V, axis=0, keepdims=True)


def build_teacher(regime, cfg, seed):
    """Return a dict with the global teacher and per-source geometry.

    Fields:
      Tstar        (d x d) global teacher operator (regime 1/2)
      Tc           list of K per-source teachers (regime 3 only; else all == Tstar)
      Vin          (d x K) unit input directions v_c defining each source
      U, S         output dirs and singular values (for interpretability)
      noise_in     isotropic input-noise std (keeps Sigma_w PD)
    """
    d, K, R = cfg["d"], cfg["K"], cfg["R"]
    corr = cfg.get("corr", 0.5)
    rng = np.random.default_rng(seed)
    # correlated input source directions (shared structure -> liftable tail)
    Vin = _correlated_dirs(d, K, corr, rng)
    Uout = _orthonormal(d, K, rng)             # output directions

    if regime == "compressible":
        # No capacity floor: ALL source directions live inside one r0<=R dimensional
        # subspace, and T* is well-conditioned on it.  Then every source has strong
        # signal (high SNR ceiling) AND a rank-r0<=R adapter reproduces T* on all of
        # them, so the tail ceiling is the noise floor, not a capacity floor.
        r0 = min(R - 1, K)
        sub = _orthonormal(d, r0, rng)                     # the shared r0-dim subspace
        coeff = _correlated_dirs(r0, K, corr, rng)         # K dirs within the subspace
        Vin = sub @ coeff                                  # (d x K), all in span(sub)
        Vin /= np.linalg.norm(Vin, axis=0, keepdims=True)
        Uout_sub = _orthonormal(d, r0, rng)
        s0 = np.linspace(2.0, 1.2, r0)                     # well-conditioned on the subspace
        Tstar = (Uout_sub * s0[None, :]) @ sub.T
        S = np.concatenate([s0, np.zeros(K - r0)])
        Tc = [Tstar] * K

    elif regime == "capacity_limited":
        # T* full rank over K directions with a decaying spectrum: a rank-R adapter
        # must drop capacity -> genuine spectral-tail floor.  Correlated inputs make
        # the tradeoff real (tail liftable).  Sources 0,1 are GENUINELY redundant
        # (identical input AND output) so consolidation has real work.
        S = np.linspace(2.0, 0.6, K)
        Vin[:, 1] = Vin[:, 0]
        Uout[:, 1] = Uout[:, 0]
        S[1] = S[0]
        Tstar = (Uout * S[None, :]) @ Vin.T
        Tc = [Tstar] * K

    elif regime == "conflicting":
        # per-source teachers DISAGREE on a SHARED input direction: every source puts
        # mass on the same v0 but demands a source-specific, mutually orthogonal
        # output u_c.  M v0 can be only one vector, so at most one source is served
        # there -> impossibility, even at full rank.
        S = np.linspace(2.0, 1.0, K)
        Tstar = (Uout * S[None, :]) @ Vin.T
        v0 = Vin[:, 0].copy()
        Tc = []
        for c in range(K):
            # source c: input concentrated on the shared v0 (set below via Vin), maps
            # v0 -> u_c with unit gain; all other directions map by Tstar.
            base = Tstar - S[0] * np.outer(Uout[:, 0], v0)   # remove v0's Tstar action
            Tc.append(base + 1.0 * np.outer(Uout[:, c], v0))  # ... replace with u_c
        # all sources share the same input direction v0 (identical input distribution)
        Vin = np.repeat(v0[:, None], K, axis=1)
    else:
        raise ValueError(regime)

    return {"regime": regime, "Tstar": Tstar, "Tc": Tc, "Vin": Vin,
            "U": Uout, "S": S, "noise_in": cfg["noise_in"]}


def frequencies(K, beta):
    f = np.array([j ** (-beta) for j in range(1, K + 1)], float)
    return f / f.sum()


def make_stream(cfg, seed):
    """Clock windows; each window mixes 2-4 sources drawn heavy-tailed. Labels are
    used only by the driver to build data; the learner never receives them."""
    K = cfg["K"]
    rng = np.random.default_rng(seed)
    freqs = frequencies(K, cfg["beta_freq"])
    lo, hi = cfg["comps_per_window"]
    windows = []
    for _ in range(cfg["n_windows"]):
        m = int(rng.integers(lo, hi + 1))
        comps = rng.choice(K, size=m, replace=False, p=freqs)
        windows.append(sorted(int(c) for c in comps))
    return windows, freqs


def _sample_source(teacher, cfg, c, n, rng):
    """Draw n examples whose inputs come from source c's distribution: concentrated
    on direction v_c with isotropic input noise; label by that source's teacher."""
    d = cfg["d"]
    v = teacher["Vin"][:, c]
    coeff = rng.standard_normal((n, 1))
    X = coeff * v[None, :] + teacher["noise_in"] * rng.standard_normal((n, d))
    Tc = teacher["Tc"][c]
    Y = X @ Tc.T + cfg["noise_out"] * rng.standard_normal((n, d))
    return X, Y


def _sample_window(teacher, cfg, comps, n, rng):
    """A window's batch mixes its sources' inputs (labels stripped)."""
    per = max(1, n // len(comps))
    Xs, Ys = [], []
    for c in comps:
        X, Y = _sample_source(teacher, cfg, c, per, rng)
        Xs.append(X); Ys.append(Y)
    X = np.concatenate(Xs, 0); Y = np.concatenate(Ys, 0)
    # scale-normalize inputs to unit RMS (applied to every arm; well-posedness only)
    rms = np.sqrt(np.mean(np.sum(X ** 2, axis=1))) + 1e-12
    return X / rms, Y / rms


def materialize(teacher, cfg, windows, seed):
    """Learner batches (X,Y only) + per-source oracle-train and test holdouts
    (driver-side, used for the reference and for retention eval)."""
    rng = np.random.default_rng(seed + 11)
    batches = [_sample_window(teacher, cfg, w, cfg["batch"], rng) for w in windows]
    srcs = sorted({c for w in windows for c in w})
    otr = np.random.default_rng(seed + 22)
    tst = np.random.default_rng(seed + 33)
    oracle_train = {c: _sample_source(teacher, cfg, c, cfg["holdout"], otr) for c in srcs}
    test = {c: _sample_source(teacher, cfg, c, cfg["holdout"], tst) for c in srcs}
    return batches, srcs, oracle_train, test


# ---------------------------------------------------------------------------
# retention metric (independent test data)
# ---------------------------------------------------------------------------

def source_retention(M, Xte, Yte):
    """1 - ||MX^T - Y||^2 / ||Y||^2 on an independent test batch (scale-free)."""
    num = np.sum((Xte @ M.T - Yte) ** 2)
    den = np.sum(Yte ** 2) + 1e-12
    return float(1.0 - num / den)


# ---------------------------------------------------------------------------
# closed-form offline reference: weighted reduced-rank regression
# ---------------------------------------------------------------------------

# The weighted reduced-rank regression solver and second-moment estimator are shared
# with the PSR-LoRA method; they live in harness.py so there is exactly ONE
# implementation.  Thin aliases keep this module's internal names.
_second_moment = H.second_moment
weighted_rrr = H.weighted_rrr


def _est_teacher_from_oracle(oracle_train, srcs, d):
    """Least-squares estimate of the (shared) teacher from pooled oracle-train data.
    In regimes 1/2 this recovers T*; in regime 3 it returns the best single-operator
    compromise (which is the point -- no operator fits all)."""
    Xs = np.concatenate([oracle_train[c][0] for c in srcs], 0)
    Ys = np.concatenate([oracle_train[c][1] for c in srcs], 0)
    # ridge for stability
    G = Xs.T @ Xs + 1e-6 * np.eye(d)
    return (np.linalg.solve(G, Xs.T @ Ys)).T


def offline_frontier(teacher, cfg, oracle_train, test, srcs, freqs):
    """Pareto-dominant set of (freq-weighted retention, worst-group retention) test
    points, over a sweep of source weightings, each solved in closed form."""
    d, R = cfg["d"], cfg["R"]
    Test_M = _est_teacher_from_oracle(oracle_train, srcs, d)
    Sig = {c: _second_moment(oracle_train[c][0]) for c in srcs}
    fw = np.array([freqs[c] for c in srcs]); fw = fw / fw.sum()

    weightings = []
    # (a) pure frequency-weighted mean (the mean-optimal end of the frontier)
    weightings.append(fw.copy())
    # (b) inverse-frequency powers: progressively tilt toward rare sources (the tail
    #     end).  gamma=0 -> freq-mean; large gamma -> concentrate on the rarest.
    for gamma in [0.5, 1.0, 1.5, 2.0, 3.0]:
        w = fw ** (1.0 - gamma)                 # w_c ∝ f_c^{1-gamma}
        weightings.append(w / w.sum())
    # (c) uniform and blends between freq-mean and uniform
    weightings.append(np.ones(len(srcs)) / len(srcs))
    for rho in [0.25, 0.5, 0.75]:
        w = (1 - rho) * fw + rho * (np.ones(len(srcs)) / len(srcs))
        weightings.append(w / w.sum())
    # (d) upweight each single source in turn (corners of the tail)
    for i in range(len(srcs)):
        w = fw.copy(); w[i] += 1.0; weightings.append(w / w.sum())

    pts = []
    for w in weightings:
        M = weighted_rrr(Test_M, [Sig[c] for c in srcs], list(w), R)
        ret = {c: source_retention(M, *test[c]) for c in srcs}
        fwret = float(sum(freqs[c] * ret[c] for c in srcs) / sum(freqs[c] for c in srcs))
        worst = float(min(ret.values()))
        pts.append({"freq_weighted": fwret, "worst_group": worst})
    # keep Pareto-dominant points (maximize both)
    front = []
    for p in pts:
        if not any(q["freq_weighted"] >= p["freq_weighted"] - 1e-9 and
                   q["worst_group"] >= p["worst_group"] - 1e-9 and
                   (q["freq_weighted"] > p["freq_weighted"] + 1e-9 or
                    q["worst_group"] > p["worst_group"] + 1e-9) for q in pts):
            front.append(p)
    front.sort(key=lambda p: p["freq_weighted"])
    return front, pts


def dist_to_frontier(point, frontier):
    fx, wy = point
    best = np.inf
    for f in frontier:
        gap = max(0.0, f["freq_weighted"] - fx) ** 2 + max(0.0, f["worst_group"] - wy) ** 2
        best = min(best, gap)
    return float(np.sqrt(best))


DEFAULT_CFG = {
    "d": 24, "K": 8, "R": 4, "beta_freq": 1.5, "n_windows": 240, "batch": 96,
    "holdout": 512, "comps_per_window": (2, 4), "noise_in": 0.1, "noise_out": 0.03,
    "corr": 0.5, "seed_teacher": 20260701, "seed_stream": 6060,
}


# ---------------------------------------------------------------------------
# self-tests: the benchmark must have the properties we claim
# ---------------------------------------------------------------------------

def _mean_worst(M, test, srcs, freqs):
    ret = {c: source_retention(M, *test[c]) for c in srcs}
    fwret = float(sum(freqs[c] * ret[c] for c in srcs) / sum(freqs[c] for c in srcs))
    return fwret, float(min(ret.values())), ret


def _gd_rank_r_reference(teacher, cfg, oracle_train, test, srcs, freqs, steps=1500):
    """A GD rank-R fit to freq-weighted mean, as a sanity check that the CLOSED-FORM
    reference is at least as good (the v3 'oracle' was only this)."""
    d, R = cfg["d"], cfg["R"]
    rng = np.random.default_rng(0)
    B = 0.05 * rng.standard_normal((d, R)); A = 0.05 * rng.standard_normal((R, d))
    fw = {c: freqs[c] for c in srcs}
    tot = sum(fw.values())
    for _ in range(steps):
        M = B @ A
        dM = np.zeros((d, d))
        for c in srcs:
            X, Y = oracle_train[c]
            g = (2.0 / X.shape[0]) * ((X @ M.T - Y).T @ X)
            dM += (fw[c] / tot) * g
        B, A = B - 0.05 * (dM @ A.T), A - 0.05 * (B.T @ dM)
    return _mean_worst(B @ A, test, srcs, freqs)


def run_tests():
    results = []
    cfg = dict(DEFAULT_CFG)

    print("regime: compressible (rank(T*) <= R -> no capacity floor)")
    teacher = build_teacher("compressible", cfg, cfg["seed_teacher"])
    H._t(np.linalg.matrix_rank(teacher["Tstar"], tol=1e-6) <= cfg["R"],
         f"rank(T*)={np.linalg.matrix_rank(teacher['Tstar'], tol=1e-6)} <= R={cfg['R']}", results)
    windows, freqs = make_stream(cfg, cfg["seed_stream"])
    batches, srcs, otr, test = materialize(teacher, cfg, windows, cfg["seed_stream"])
    front, pts = offline_frontier(teacher, cfg, otr, test, srcs, freqs)
    best_mean = max(p["freq_weighted"] for p in pts)
    best_worst = max(p["worst_group"] for p in pts)
    H._t(best_mean > 0.93, f"reference near-perfect mean on compressible ({best_mean:.3f})", results)
    H._t(best_worst > 0.90, f"reference can also protect the tail on compressible ({best_worst:.3f})", results)

    print("regime: capacity_limited (rank(T*)=K>R -> real spectral floor)")
    tcap = build_teacher("capacity_limited", cfg, cfg["seed_teacher"])
    H._t(np.linalg.matrix_rank(tcap["Tstar"], tol=1e-6) > cfg["R"],
         f"rank(T*)={np.linalg.matrix_rank(tcap['Tstar'], tol=1e-6)} > R", results)
    wc, fc = make_stream(cfg, cfg["seed_stream"])
    bc, sc, otrc, testc = materialize(tcap, cfg, wc, cfg["seed_stream"])
    frontc, ptsc = offline_frontier(tcap, cfg, otrc, testc, sc, fc)
    best_mean_c = max(p["freq_weighted"] for p in ptsc)
    best_worst_c = max(p["worst_group"] for p in ptsc)
    worst_at_best_mean = [p["worst_group"] for p in ptsc if p["freq_weighted"] == best_mean_c][0]
    # the capacity floor shows in the TAIL: no single rank-R adapter serves every
    # source (best worst-group is bounded away from 1), even though frequent sources
    # can be near-perfect (mean high).
    H._t(best_worst_c < 0.95, f"capacity floor present in the tail (best worst-group {best_worst_c:.3f} < 1)", results)
    # a real tradeoff: emphasizing the tail LIFTS worst-group above what the
    # mean-optimal weighting achieves -- i.e. the tail is a choosable axis.
    H._t(best_worst_c > worst_at_best_mean + 0.02,
         f"mean-tail tradeoff real (worst-group {worst_at_best_mean:.3f}->{best_worst_c:.3f} by tilting to the tail)", results)
    H._t(len(frontc) >= 2, f"frontier non-degenerate ({len(frontc)} Pareto points)", results)

    print("closed-form reference vs GD (must be >= GD)")
    cf_mean, cf_worst, _ = _mean_worst(
        weighted_rrr(_est_teacher_from_oracle(otrc, sc, cfg["d"]),
                     [_second_moment(otrc[c][0]) for c in sc],
                     [fc[c] for c in sc], cfg["R"]), testc, sc, fc)
    gd_mean, gd_worst, _ = _gd_rank_r_reference(tcap, cfg, otrc, testc, sc, fc)
    H._t(cf_mean >= gd_mean - 0.01, f"closed-form mean {cf_mean:.3f} >= GD {gd_mean:.3f} (near-optimal, not GD)", results)

    print("regime: conflicting (no single operator fits -> impossibility control)")
    tcf = build_teacher("conflicting", cfg, cfg["seed_teacher"])
    wcf, fcf = make_stream(cfg, cfg["seed_stream"])
    bcf, scf, otrcf, testcf = materialize(tcf, cfg, wcf, cfg["seed_stream"])
    # even a FULL-RANK (R=d) reference cannot make every source happy
    full_cfg = dict(cfg); full_cfg["R"] = cfg["d"]
    Mfull = weighted_rrr(_est_teacher_from_oracle(otrcf, scf, cfg["d"]),
                         [_second_moment(otrcf[c][0]) for c in scf],
                         [1.0] * len(scf), full_cfg["R"])
    _, worst_full, retf = _mean_worst(Mfull, testcf, scf, fcf)
    H._t(worst_full < 0.6, f"conflict: even full-rank leaves a source unsatisfied (worst ret {worst_full:.3f})", results)

    n_fail = sum(1 for ok, _ in results if not ok)
    print(f"\n{len(results)-n_fail}/{len(results)} passed")
    return n_fail


if __name__ == "__main__":
    import sys
    sys.exit(1 if run_tests() else 0)
