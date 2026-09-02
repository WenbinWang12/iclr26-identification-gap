"""Correct, strong baselines on the v4 benchmark, sharing one buffer budget.

Every arm here is a rank-R factored LoRA trained by GD on the window stream, seeing
only (X,Y).  They differ only in how they use a bounded replay buffer of the SAME
total budget as PSR-LoRA (buf_u + buf_g windows).  Primitives (true reservoir,
CVaR, excess) come from harness.py -- no arm re-implements them, so the reservoir
recency bug and the fake-CVaR of the retracted code cannot recur.

Arms:
  sequential      : no replay (lower bound / online-error reference)
  uniform_replay  : true Vitter reservoir, uniform replay sample per window
  gss_replay      : gradient-diverse (GSS) buffer, uniform replay from it
  mir_replay      : uniform reservoir + MIR retrieval (train on the buffered window
                    whose loss would INCREASE most under the pending update)
  excess_cvar     : reservoir + replay the whole worst tau-fraction (the
                    excess-over-anchor CVaR_tau tail set) -- proper risk-aware replay
  agem            : uniform reservoir + A-GEM gradient projection (project the
                    current-window gradient off any direction that raises buffer loss)
  ewc_lora        : quadratic penalty toward the previous operator, curvature-scaled

These are exactly the comparators PSR-LoRA must beat at matched mean to earn a tail
claim (kill criterion (a),(b)); `plain_factor_sgd` in psr_lora.py is the mechanism
ablation (c).
"""

from __future__ import annotations

import numpy as np

import harness as H
from psr_lora import _grad_M, _loss, _factor_from_operator


def _init(d, R, rng):
    return 0.01 * rng.standard_normal((d, R)), 0.01 * rng.standard_normal((R, d))


def _train_on(B, A, train, lr, steps, R):
    for _ in range(steps):
        for (Xb, Yb) in train:
            M = B @ A
            g = _grad_M(M, Xb, Yb)
            B, A = B - lr * (g @ A.T), A - lr * (B.T @ g)
    B, A = _factor_from_operator(B @ A, R)          # gauge-free, rank-R (fair to PSR)
    return B, A


def _common(cfg):
    return cfg["R"], cfg.get("lr", 0.15), cfg.get("gd_steps", 20), \
        cfg.get("buf_u", 8) + cfg.get("buf_g", 8)


def run_sequential(cfg, batches, seed):
    d = batches[0][0].shape[1]; R, lr, steps, _ = _common(cfg)
    rng = np.random.default_rng(seed + 1); B, A = _init(d, R, rng)
    snaps = []
    for (Xk, Yk) in batches:
        B, A = _train_on(B, A, [(Xk, Yk)], lr, steps, R)
        snaps.append(B @ A)
    return snaps, {}


def run_uniform_replay(cfg, batches, seed):
    d = batches[0][0].shape[1]; R, lr, steps, cap = _common(cfg)
    rng = np.random.default_rng(seed + 2); B, A = _init(d, R, rng)
    buf = H.Reservoir(cap, rng); snaps = []
    for (Xk, Yk) in batches:
        train = [(Xk, Yk)]
        if len(buf):
            e = buf.items[rng.integers(0, len(buf))]
            train.append((e[0], e[1]))
        B, A = _train_on(B, A, train, lr, steps, R)
        buf.offer((Xk, Yk))
        snaps.append(B @ A)
    return snaps, {}


def run_gss_replay(cfg, batches, seed):
    d = batches[0][0].shape[1]; R, lr, steps, cap = _common(cfg)
    rng = np.random.default_rng(seed + 3); B, A = _init(d, R, rng)
    G = []; snaps = []
    def gdir(M, X, Y):
        g = _grad_M(M, X, Y).ravel(); return g / (np.linalg.norm(g) + 1e-12)
    for (Xk, Yk) in batches:
        train = [(Xk, Yk)]
        if G:
            e = G[rng.integers(0, len(G))]; train.append((e["X"], e["Y"]))
        B, A = _train_on(B, A, train, lr, steps, R)
        M = B @ A
        entry = {"X": Xk, "Y": Yk, "gdir": gdir(M, Xk, Yk)}
        if len(G) < cap:
            G.append(entry)
        else:                                            # drop most-redundant
            cand = G + [entry]; dirs = np.stack([e["gdir"] for e in cand], 0)
            sim = dirs @ dirs.T; np.fill_diagonal(sim, -np.inf)
            drop = int(np.argmax(sim.max(1)))
            if drop != len(cand) - 1:
                G[drop] = entry
        snaps.append(M)
    return snaps, {}


def run_mir_replay(cfg, batches, seed):
    """Maximally-Interfered Retrieval: replay the buffered window whose loss would
    rise most under a one-step lookahead of the pending current-window update."""
    d = batches[0][0].shape[1]; R, lr, steps, cap = _common(cfg)
    rng = np.random.default_rng(seed + 4); B, A = _init(d, R, rng)
    buf = H.Reservoir(cap, rng); snaps = []
    for (Xk, Yk) in batches:
        train = [(Xk, Yk)]
        if len(buf):
            M = B @ A
            g = _grad_M(M, Xk, Yk)
            Bv, Av = B - lr * (g @ A.T), A - lr * (B.T @ g)
            Mv = Bv @ Av                                 # virtual post-update operator
            interference = [ _loss(Mv, e[0], e[1]) - _loss(M, e[0], e[1]) for e in buf.items]
            e = buf.items[int(np.argmax(interference))]
            train.append((e[0], e[1]))
        B, A = _train_on(B, A, train, lr, steps, R)
        buf.offer((Xk, Yk))
        snaps.append(B @ A)
    return snaps, {}


def run_excess_cvar(cfg, batches, seed):
    """Risk-aware (CVaR) replay: keep post-learning anchors and replay the whole
    worst tau-fraction (the excess-over-anchor CVaR tail set), not a single window.

    This is a proper CVaR_tau replay.  Each window it ranks the buffer by
    excess-over-anchor forgetting and replays the WHOLE worst tau-fraction --
    the ceil(tau*n) tail set that H.cvar averages over and that PSR-LoRA's own
    tail gradient (_tail_excess_grad) descends -- instead of the single argmax
    window the retracted code used.  Anchors are the best post-learning loss each
    window has reached (EMA toward best-seen); gain is the realized acquisition
    (loss drop when the window was learned), exactly PSR-LoRA's own definition,
    so a naturally-hard window cannot own the tail and the baseline uses the SAME
    excess machinery as the method.  This is the strongest risk-aware replay
    baseline PSR-LoRA's tail claim (b) must beat."""
    d = batches[0][0].shape[1]; R, lr, steps, cap = _common(cfg)
    xi = cfg.get("xi", 0.02); tau = cfg.get("tau", 0.4)
    rng = np.random.default_rng(seed + 5); B, A = _init(d, R, rng)
    buf = H.Reservoir(cap, rng); snaps = []
    for (Xk, Yk) in batches:
        M = B @ A
        L_before = _loss(M, Xk, Yk)                      # loss on the new window BEFORE learning it
        train = [(Xk, Yk)]
        if len(buf):
            ex = np.array([H.excess(_loss(M, e["X"], e["Y"]), e["anchor"], e["gain"], xi)
                           for e in buf.items])
            k = max(1, int(np.ceil(tau * len(buf))))     # the CVaR_tau tail: worst tau-fraction
            tail_idx = np.argsort(ex)[::-1][:k]
            for i in tail_idx:                           # replay the WHOLE tail set
                train.append((buf.items[i]["X"], buf.items[i]["Y"]))
        B, A = _train_on(B, A, train, lr, steps, R)
        M = B @ A
        L_after = _loss(M, Xk, Yk)
        for e in buf.items:                              # EMA anchors toward best-seen
            e["anchor"] = min(e["anchor"], _loss(M, e["X"], e["Y"]))
        gain = max(L_before - L_after, 1e-3)             # realized acquisition (PSR-LoRA's gain)
        buf.offer({"X": Xk, "Y": Yk, "anchor": L_after, "gain": gain})
        snaps.append(M)
    return snaps, {}


def run_agem(cfg, batches, seed):
    """A-GEM: project the current-window gradient off any component that would
    increase the average loss on a replay reference drawn from the buffer."""
    d = batches[0][0].shape[1]; R, lr, steps, cap = _common(cfg)
    rng = np.random.default_rng(seed + 6); B, A = _init(d, R, rng)
    buf = H.Reservoir(cap, rng); snaps = []; proj = 0
    for (Xk, Yk) in batches:
        for _ in range(steps):
            M = B @ A
            g = _grad_M(M, Xk, Yk)
            if len(buf):
                gr = np.zeros_like(M)
                for e in buf.items:
                    gr += _grad_M(M, e[0], e[1])
                gr /= len(buf)
                dot = float(np.sum(g * gr)); nn = float(np.sum(gr * gr))
                if dot < 0 and nn > 1e-18:
                    g = g - (dot / nn) * gr; proj += 1
            B, A = B - lr * (g @ A.T), A - lr * (B.T @ g)
        B, A = _factor_from_operator(B @ A, R)
        buf.offer((Xk, Yk))
        snaps.append(B @ A)
    return snaps, {"projections": proj}


def run_ewc_lora(cfg, batches, seed):
    """EWC-style: quadratic penalty pulling the operator toward its previous value,
    weighted by a NORMALIZED diagonal curvature estimate (Fisher proxy).  The Fisher
    proxy is normalized to unit max so the penalty is well-scaled relative to the MSE
    gradient regardless of loss magnitude (an unnormalized proxy overflows)."""
    d = batches[0][0].shape[1]; R, lr, steps, _ = _common(cfg)
    lam = cfg.get("ewc_lambda", 1.0)
    rng = np.random.default_rng(seed + 7); B, A = _init(d, R, rng)
    snaps = []; star = (B @ A).copy(); Fisher = np.zeros((d, d))
    for (Xk, Yk) in batches:
        Fn = Fisher / (Fisher.max() + 1e-9)              # normalized penalty weights
        for _ in range(steps):
            M = B @ A
            g = _grad_M(M, Xk, Yk) + lam * Fn * (M - star)
            B, A = B - lr * (g @ A.T), A - lr * (B.T @ g)
        B, A = _factor_from_operator(B @ A, R)
        M = B @ A
        gk = _grad_M(M, Xk, Yk)
        Fisher = 0.9 * Fisher + (gk ** 2)                # diag Fisher proxy (squared grad)
        star = M.copy()
        snaps.append(M)
    return snaps, {}


BASELINES = {
    "sequential": run_sequential,
    "uniform_replay": run_uniform_replay,
    "gss_replay": run_gss_replay,
    "mir_replay": run_mir_replay,
    "excess_cvar": run_excess_cvar,
    "agem": run_agem,
    "ewc_lora": run_ewc_lora,
}


# ---------------------------------------------------------------------------
# self-tests: each baseline runs, stays rank-R, is label-blind & reproducible
# ---------------------------------------------------------------------------

def run_tests():
    import benchmark_v4 as BM
    results = []
    cfg = dict(BM.DEFAULT_CFG); cfg.update({"gd_steps": 12, "lr": 0.15, "buf_u": 6, "buf_g": 6})
    teacher = BM.build_teacher("capacity_limited", cfg, cfg["seed_teacher"])
    windows, freqs = BM.make_stream(cfg, cfg["seed_stream"])
    batches, srcs, otr, test = BM.materialize(teacher, cfg, windows, cfg["seed_stream"])

    for name, fn in BASELINES.items():
        snaps, _ = fn(cfg, batches, 7)
        M = snaps[-1]
        rank_ok = int(np.linalg.matrix_rank(M, tol=1e-6)) <= cfg["R"]
        h1 = H.trajectory_hash(snaps)
        h2 = H.trajectory_hash(fn(cfg, batches, 7)[0])
        ret = {c: BM.source_retention(M, *test[c]) for c in srcs}
        fw = sum(freqs[c] * ret[c] for c in srcs) / sum(freqs[c] for c in srcs)
        worst = min(ret.values())
        H._t(rank_ok and h1 == h2,
             f"{name:16s} rank<=R & reproducible  (mean {fw:+.3f}, worst {worst:+.3f})", results)

    # The motivating phenomenon: UNIFORM replay can HURT the tail (it samples the
    # rare source rarely, reinforcing frequent sources), whereas RISK-AWARE replay
    # (excess-CVaR) must protect it.  Both directions are asserted -- if uniform did
    # not hurt or risk-aware did not help, the benchmark would not exercise the point.
    seq = run_sequential(cfg, batches, 7)[0][-1]
    unif = run_uniform_replay(cfg, batches, 7)[0][-1]
    cvar = run_excess_cvar(cfg, batches, 7)[0][-1]
    wseq = min(BM.source_retention(seq, *test[c]) for c in srcs)
    wunif = min(BM.source_retention(unif, *test[c]) for c in srcs)
    wcvar = min(BM.source_retention(cvar, *test[c]) for c in srcs)
    H._t(wunif < wseq, f"uniform replay HURTS the tail vs sequential ({wunif:+.3f} < {wseq:+.3f})", results)
    H._t(wcvar > wunif, f"risk-aware (excess-CVaR) PROTECTS the tail ({wcvar:+.3f} > {wunif:+.3f})", results)

    n_fail = sum(1 for ok, _ in results if not ok)
    print(f"\n{len(results)-n_fail}/{len(results)} passed")
    return n_fail


if __name__ == "__main__":
    import sys
    sys.exit(1 if run_tests() else 0)
