"""PSR-LoRA: Pareto-Safe Rank Reallocation for continual LoRA (single-buffer
minimax reduced-rank re-solve).

Design (deliberately minimal -- one mechanism, testable in isolation):

  * ONE fixed-rank-R adapter M = B A.  There is NO four-state atom machine, NO
    inverse-frequency score, NO discrete eviction, NO permanent freeze, and (unlike
    the earlier defensive prototype preserved at the bottom of this file) NO
    gradient projection or accept/rollback loop on the main path.

  * ONE bounded buffer of capacity buf_u + buf_g = the SAME total budget a replay
    baseline holds, split for COVERAGE: a small true (Vitter) uniform reservoir
    (buf_u) for an unbiased view of history, PLUS an input-direction-diverse buffer
    (buf_g) that guarantees rare input directions survive.  A single uniform
    reservoir under-samples the rare/worst SOURCE -- the diagnosed cause of the
    tail-coverage failure on some teacher geometries (corr(worst-source buffer count,
    tail gain) = +0.44); the diversity buffer restores tail coverage at NO extra
    memory.  Both parts are oracle-free (the diversity score uses only the window
    inputs' top singular direction, never labels or teacher internals) and
    deterministic given the offer order, so the adapter each window is a pure
    function of the buffer contents plus the fixed seed -- reproducible by hash.
    coverage=False recovers the plain single uniform reservoir (the pre-remedy
    method), kept as the coverage ablation.  Nothing else is retained between windows.

  * Each window (`_minimax_rrr`): from the buffered (X,Y) ALONE, (i) estimate a
    pooled ridge-LS teacher T-hat (`_pooled_teacher`, ridge rho threaded from cfg),
    then (ii) run `rounds` of multiplicative-weights ascent that grows the weight of
    the currently worst-fit buffered windows and RE-SOLVES the closed-form weighted
    rank-R reduced-rank regression (`H.weighted_rrr`) each round, KEEPING the iterate
    with the smallest realized buffered worst-window loss.  Round 0 uses uniform
    weights, so the kept operator is never worse than plain uniform-weight buffer-RRR
    on the buffered worst group -- a safety guarantee by construction that replaces
    the old rollback machinery (no divergence possible; the uniform solve is always
    a feasible fallback).

The distinctive content vs a replay baseline is EXPLICIT reallocation: the R-th
direction is re-solved toward the worst-served buffered window by the minimax
reweighting, instead of being captured implicitly by the frequent sources' gradient
mass.  `run_buffer_rrr_uniform` (reweight=False) is the ablation that shares the SAME
buffer and SAME closed-form solver but drops the reweighting -- if PSR-LoRA does not
beat it on the tail, the reallocation mechanism adds nothing (kill criterion (c)).

The A-GEM-style gradient-projection + accept/rollback prototype is retained below as
`run_psr_grad_defensive`, an HONEST NEGATIVE CONTROL: on the capacity_limited stream
it sits at the uniform-RRR floor because it allocates rank implicitly through
gradient dynamics.  The excess-over-anchor metric it uses (relu(loss - anchor)/gain)
and the two auxiliary buffers it maintains belong to THAT control, not the method.

Boundary: controlled linear factored proxy, learner sees (X,Y) only; oracle-free.
"""

from __future__ import annotations

import numpy as np

import harness as H


def _grad_M(M, X, Y):
    """Ambient gradient of mean squared error wrt M."""
    resid = X @ M.T - Y
    return (2.0 / X.shape[0]) * (resid.T @ X)


def _loss(M, X, Y):
    return float(np.mean(np.sum((X @ M.T - Y) ** 2, axis=1)))


def _retract_rankR(M, R):
    """Project an ambient operator onto rank <= R (Eckart-Young)."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    return (U[:, :R] * S[:R][None, :]) @ Vt[:R, :]


def _factor_from_operator(M, R):
    """Balanced rank-R factors (B,A) with B A = P_R(M): B=U S^{1/2}, A=S^{1/2} V^T.
    This is the gauge-free factorization -- factor norms are the singular values."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    U, S, Vt = U[:, :R], S[:R], Vt[:R, :]
    root = np.sqrt(S)
    return U * root[None, :], root[:, None] * Vt


# ---------------------------------------------------------------------------
# excess metric on buffered windows
# ---------------------------------------------------------------------------

def _excess_list(M, buf, xi, e_max):
    """Per-entry excess forgetting over stored anchors."""
    out = []
    for e in buf:
        L = _loss(M, e["X"], e["Y"])
        out.append(float(H.excess(L, e["anchor"], e["gain"], xi, e_max)))
    return np.array(out) if out else np.zeros(0)


def _avg_excess_grad(M, buf, xi):
    """Ascent gradient of mean excess over a buffer (indicator-gated, gain-scaled)."""
    if not buf:
        return np.zeros_like(M), 0.0
    g = np.zeros_like(M)
    vals = []
    for e in buf:
        L = _loss(M, e["X"], e["Y"])
        ex = H.excess(L, e["anchor"], e["gain"], xi, np.inf)
        vals.append(ex)
        if L > e["anchor"] + xi:                       # active (relu on)
            g += (1.0 / (e["gain"] + 1e-12)) * _grad_M(M, e["X"], e["Y"])
    g /= len(buf)
    return g, float(np.mean(vals))


def _tail_excess_grad(M, buf, xi, tau):
    """Ascent gradient of the excess-CVaR (worst tau-fraction) over a buffer."""
    if not buf:
        return np.zeros_like(M), 0.0
    ex = _excess_list(M, buf, xi, np.inf)
    n = len(buf)
    k = max(1, int(np.ceil(tau * n)))
    tail_idx = np.argsort(ex)[::-1][:k]
    g = np.zeros_like(M)
    for i in tail_idx:
        e = buf[i]
        if _loss(M, e["X"], e["Y"]) > e["anchor"] + xi:
            g += (1.0 / (e["gain"] + 1e-12)) * _grad_M(M, e["X"], e["Y"])
    g /= k
    return g, float(np.mean(np.sort(ex)[::-1][:k]))


# ---------------------------------------------------------------------------
# gradient-diverse (GSS-style) buffer maintenance
# ---------------------------------------------------------------------------

def _window_grad_dir(M, X, Y):
    g = _grad_M(M, X, Y).ravel()
    return g / (np.linalg.norm(g) + 1e-12)


def _gss_offer(G, entry, cap):
    """Greedily keep the most gradient-diverse set: if full, drop the entry whose
    gradient direction is most redundant (highest max cosine to the others)."""
    if len(G) < cap:
        G.append(entry); return
    cand = G + [entry]
    dirs = np.stack([e["gdir"] for e in cand], 0)
    sim = dirs @ dirs.T
    np.fill_diagonal(sim, -np.inf)
    max_sim = sim.max(axis=1)                            # redundancy of each entry
    drop = int(np.argmax(max_sim))
    if drop != len(cand) - 1:                            # replace a redundant old one
        G[drop] = entry


# ---------------------------------------------------------------------------
# input-direction-diverse buffer maintenance (coverage remedy)
# ---------------------------------------------------------------------------

def _indir(X):
    """Dominant input direction of a window = top right singular vector of the
    centered inputs.  Oracle-free (uses X only; no Y, no teacher).  This is the
    coordinate along which the window most excites the operator, so keeping a set of
    mutually-orthogonal such directions guarantees rare SOURCES stay represented."""
    _, _, vt = np.linalg.svd(X - X.mean(0, keepdims=True), full_matrices=False)
    return vt[0]


def _div_offer(G, entry, cap):
    """Keep the most input-direction-diverse set of windows: if full, drop the entry
    whose direction is most redundant (highest max |cosine| to the others).  Sign-
    invariant (|cosine|), so +v and -v count as the same direction.  Deterministic
    given the offer order.  Mirrors `_gss_offer` but on input directions, which are
    stable per source (a rare source's direction cannot be crowded out by frequent
    sources the way its gradient mass can)."""
    if len(G) < cap:
        G.append(entry); return
    cand = G + [entry]
    D = np.stack([e["idir"] for e in cand], 0)
    sim = np.abs(D @ D.T); np.fill_diagonal(sim, -np.inf)
    drop = int(np.argmax(sim.max(1)))                    # most redundant entry
    if drop != len(cand) - 1:                            # replace a redundant old one
        G[drop] = entry


# ---------------------------------------------------------------------------
# PSR-LoRA
# ---------------------------------------------------------------------------

DEFAULT = {
    "R": 4, "buf_u": 4, "buf_g": 12, "ridge": 1e-6, "coverage": True,
    # --- minimax reweighting (the reallocation mechanism) ---
    "rounds": 12,        # multiplicative-weights rounds; we keep the best iterate, so
                         #   more rounds can only help -- not a risk knob.
    "mw_eta": 1.0,       # MW step (textbook default 1.0); the only real knob.
    "reweight": True,    # False -> uniform-weight buffer-RRR (the mechanism ablation)
    # --- params used only by the grad-defensive negative control below ---
    "gd_steps": 20, "lr": 0.15, "tau": 0.4, "xi": 0.02,
    "eps_avg": 0.15, "eps_tail": 0.15, "gamma": 0.02, "e_max": 5.0,
    "ls_shrink": 0.5, "ls_tries": 3, "anchor_ema": 0.5,
    "lambda_hist": 0.0, "accept": "current",
}


def _pooled_teacher(items, d, ridge):
    """Oracle-free ridge least-squares teacher estimate from the buffered (X,Y)."""
    Xs = np.concatenate([e["X"] for e in items], 0)
    Ys = np.concatenate([e["Y"] for e in items], 0)
    G = Xs.T @ Xs + ridge * np.eye(d)
    return (np.linalg.solve(G, Xs.T @ Ys)).T


def _minimax_rrr(items, d, R, rounds, eta, reweight, ridge=1e-6):
    """Choose weights over buffered windows and return the rank-R operator that
    MINIMIZES the buffered worst-window loss.

    We run `rounds` of multiplicative-weights ascent that grows the weight of the
    currently-worst windows and re-solves the closed-form weighted rank-R RRR each
    round, KEEPING the iterate with the smallest realized worst-window loss.  Round
    0 uses uniform weights, so the returned operator is never worse than plain
    uniform-weight RRR on the buffered worst group -- a safety guarantee by
    construction (no divergence, no rollback machinery needed).

    `ridge` is the pooled-teacher ridge rho; it is threaded from the config so the
    appendix rho sweep actually reaches the estimator (default 1e-6 == the frozen
    main-run value, so the default is numerically identical to the previous code).

    reweight=False stops after the uniform round -> the ablation that isolates the
    reallocation: same buffer, same closed-form solver, no risk reweighting."""
    n = len(items)
    Th = _pooled_teacher(items, d, ridge)
    Sg = [H.second_moment(e["X"]) for e in items]
    w = np.ones(n) / n
    best_M, best_worst, best_round = None, np.inf, 0
    n_rounds = 0 if not reweight else rounds
    for r in range(n_rounds + 1):
        M = H.weighted_rrr(Th, Sg, list(w), R)
        losses = np.array([_loss(M, e["X"], e["Y"]) for e in items])
        worst = float(np.max(losses))
        if worst < best_worst - 1e-12:
            best_worst, best_M, best_round = worst, M, r
        # multiplicative-weights ascent toward the worst windows
        w = w * np.exp(eta * losses / (losses.mean() + 1e-12))
        w /= w.sum()
    return best_M, best_round


def run_psr_lora(cfg, batches, seed, *, reweight=None, coverage=None, **over):
    """PSR-LoRA: coverage-aware bounded-buffer minimax risk-reweighted rank reallocation.

    Returns (snaps, stats).  The learner sees only (X,Y) windows (oracle-free).

    Buffer (total budget buf_u + buf_g = a replay baseline's).  With coverage=True
    (the method) the budget is split for TAIL COVERAGE: a buf_u-slot uniform (Vitter)
    reservoir U for an unbiased view of history, plus a buf_g-slot input-direction-
    diverse buffer G (`_div_offer` on `_indir`) that guarantees rare input directions
    -- hence rare SOURCES -- stay represented.  The union U u G is the working set.
    coverage=False uses a single uniform reservoir of the same total size (the
    pre-remedy method; the coverage ablation).

    From the working set alone we (i) estimate a pooled teacher by ridge LS, (ii) run
    minimax multiplicative-weights reweighting that re-solves the closed-form weighted
    rank-R RRR and keeps the operator with the smallest buffered worst-window loss.
    That operator IS the adapter (rank-R by construction).  This is genuine capacity
    REALLOCATION: the R-th direction is re-solved toward the worst-served source
    instead of being captured implicitly by the frequent sources' gradient mass, and
    the coverage buffer makes sure that worst source is actually IN the working set.

    reweight=False -> uniform-weight RRR on the SAME working set: the ablation
    isolating that the minimax reweighting (not merely buffer + closed-form solve)
    lifts the tail (criterion (c))."""
    p = dict(DEFAULT); p.update({k: cfg[k] for k in cfg if k in DEFAULT}); p.update(over)
    if reweight is not None:
        p["reweight"] = reweight
    if coverage is not None:
        p["coverage"] = coverage
    d = batches[0][0].shape[1]
    R = p["R"]
    cap = p["buf_u"] + p["buf_g"]
    rng = np.random.default_rng(seed + 41)
    if p["coverage"]:
        U = H.Reservoir(p["buf_u"], rng)                 # unbiased history
        G = []                                           # input-direction-diverse coverage
    else:
        U = H.Reservoir(cap, rng)                        # single uniform reservoir (ablation)
        G = None
    snaps, rounds_used = [], []
    M = np.zeros((d, d))
    for (Xk, Yk) in batches:
        U.offer({"X": Xk, "Y": Yk})
        if p["coverage"]:
            _div_offer(G, {"X": Xk, "Y": Yk, "idir": _indir(Xk)}, p["buf_g"])
            items = U.items + [{"X": e["X"], "Y": e["Y"]} for e in G]
        else:
            items = U.items
        M, r_used = _minimax_rrr(items, d, R, p["rounds"], p["mw_eta"],
                                 p["reweight"], p["ridge"])
        rounds_used.append(r_used)
        snaps.append(M.copy())
    stats = {"cap": cap, "mean_picked_round": float(np.mean(rounds_used)),
             "reweight": p["reweight"], "coverage": p["coverage"]}
    return snaps, stats


def run_buffer_rrr_uniform(cfg, batches, seed, **over):
    """Ablation (c) for the reallocation claim: SAME coverage working set and SAME
    closed-form rank-R RRR solver, but UNIFORM weights (no minimax reweighting).
    If PSR-LoRA does not beat this on the tail, the reweighting adds nothing."""
    return run_psr_lora(cfg, batches, seed, reweight=False, **over)


def run_psr_no_coverage(cfg, batches, seed, **over):
    """Coverage ablation: the pre-remedy method -- minimax reweighted RRR on a SINGLE
    uniform reservoir of the same total size (buf_u + buf_g), no diversity buffer.
    If PSR-LoRA (coverage=True) does not beat this on the tail, the coverage buffer
    adds nothing over a plain reservoir of the same budget."""
    return run_psr_lora(cfg, batches, seed, coverage=False, **over)


# ---------------------------------------------------------------------------
# grad-dynamics negative control (the original defensive design)
# ---------------------------------------------------------------------------
# Kept verbatim as an HONEST NEGATIVE CONTROL: A-GEM-style projection + accept/
# rollback on a factored SGD adapter.  On the v4 capacity_limited stream this design
# is stuck at the uniform-RRR capacity floor (worst-group ~0.57) because it allocates
# the rank budget IMPLICITLY through gradient dynamics, which the frequent sources
# dominate; it can refuse to harm the tail but cannot re-solve for it.  That contrast
# is exactly what motivates the closed-form reallocation above, so this stays runnable
# and its correctness properties (real rollback, trust-region actually binds) remain
# under test -- they are the two properties the retracted D2-D code FAILED.


def run_psr_grad_defensive(cfg, batches, seed, *, constrained=True, use_tail=True, **over):
    """Original defensive grad-projection design (negative control)."""
    p = dict(DEFAULT); p.update({k: cfg[k] for k in cfg if k in DEFAULT}); p.update(over)
    d = batches[0][0].shape[1]
    R = p["R"]
    rng = np.random.default_rng(seed + 41)
    B = 0.01 * rng.standard_normal((d, R))
    A = 0.01 * rng.standard_normal((R, d))
    U = H.Reservoir(p["buf_u"], rng)                     # true uniform reservoir
    G = []                                               # gradient-diverse buffer
    snaps, stats = [], {"accepts": 0, "rejects": 0, "projections": 0, "linesearch": 0}

    def excess_budgets(M):
        _, a = _avg_excess_grad(M, U.items, p["xi"])
        _, q = _tail_excess_grad(M, G, p["xi"], p["tau"])
        return a, q

    for (Xk, Yk) in batches:
        M0 = B @ A
        L_before = _loss(M0, Xk, Yk)
        a0, q0 = excess_budgets(M0)

        # ---- constrained descent of the current window ----
        eta = p["lr"]
        accepted = False
        for attempt in range(p["ls_tries"] + 1):
            Bt, At = B.copy(), A.copy()
            for _ in range(p["gd_steps"]):
                M = Bt @ At
                g_cur = _grad_M(M, Xk, Yk)
                if constrained:
                    g_avg, _ = _avg_excess_grad(M, U.items, p["xi"])
                    g_tail, _ = (_tail_excess_grad(M, G, p["xi"], p["tau"]) if use_tail
                                 else (np.zeros_like(M), 0.0))
                    g_hist = g_avg + g_tail
                    # A-GEM projection: if descending g_cur would raise hist risk
                    # (<g_cur,g_hist> < 0), remove the offending component.
                    dot = float(np.sum(g_cur * g_hist))
                    hn = float(np.sum(g_hist * g_hist))
                    if dot < 0 and hn > 1e-18:
                        g_cur = g_cur - (dot / hn) * g_hist
                        stats["projections"] += 1
                    # ACTIVE recovery (defensive design's attempt): also descend the
                    # historical excess.  g_hist is excess-normalized (1/gain), so it
                    # is badly scaled as a raw step direction; treat lambda_hist as a
                    # pure DIRECTION-mixing ratio and renorm the blend to g_cur's
                    # magnitude (no overflow).  On v4 this still cannot lift the tail.
                    if p["lambda_hist"] > 0.0:
                        g0 = float(np.linalg.norm(g_cur))
                        hnorm = float(np.linalg.norm(g_hist))
                        if hnorm > 1e-18 and g0 > 1e-18:
                            g_blend = g_cur + p["lambda_hist"] * (g0 / hnorm) * g_hist
                            bn = float(np.linalg.norm(g_blend))
                            if bn > 1e-18:
                                g_cur = g_blend * (g0 / bn)
                # factored gradient step (keeps rank <= R by construction)
                Bt, At = Bt - eta * (g_cur @ At.T), At - eta * (Bt.T @ g_cur)
            # retract to rank R, then rebalance to the gauge-free balanced factors
            Bt, At = _factor_from_operator(Bt @ At, R)
            Mnew = Bt @ At
            if not constrained:
                B, A = Bt, At; accepted = True; break
            # ---- acceptance test on the historical budgets ----
            L_after = _loss(Mnew, Xk, Yk)
            a1, q1 = excess_budgets(Mnew)
            margin = p["gamma"] * max(L_before, 1e-6)
            cur_better = L_after <= L_before - margin
            safe_avg = a1 <= a0 + p["eps_avg"]
            safe_tail = (q1 <= q0 + p["eps_tail"]) if use_tail else True
            if p["accept"] == "pareto":
                avg_better = a1 <= a0 - p["gamma"] * max(a0, 1e-6)
                tail_better = use_tail and (q1 <= q0 - p["gamma"] * max(q0, 1e-6))
                progressed = cur_better or avg_better or tail_better
                cur_safe = L_after <= L_before + p["eps_avg"]        # don't wreck current
                if progressed and cur_safe and safe_avg and safe_tail:
                    B, A = Bt, At; accepted = True; break
            else:
                if cur_better and safe_avg and safe_tail:
                    B, A = Bt, At; accepted = True; break
            eta *= p["ls_shrink"]
            if attempt >= 1:
                stats["linesearch"] += 1
        if accepted:
            stats["accepts"] += 1
        else:
            stats["rejects"] += 1                        # keep old B, A (full rollback)

        # ---- record anchors/gains for this window, update buffers ----
        M = B @ A
        L_final = _loss(M, Xk, Yk)
        gain = max(L_before - L_final, 1e-3)
        entry = {"X": Xk, "Y": Yk, "anchor": L_final, "gain": gain,
                 "gdir": _window_grad_dir(M, Xk, Yk)}
        # refresh anchors of existing buffered windows toward their current loss (EMA)
        for e in U.items:
            e["anchor"] = (1 - p["anchor_ema"]) * e["anchor"] + p["anchor_ema"] * min(
                e["anchor"], _loss(M, e["X"], e["Y"]))
        U.offer(dict(entry))
        _gss_offer(G, dict(entry), p["buf_g"])
        snaps.append(M.copy())
    return snaps, stats


def run_plain_factor_sgd(cfg, batches, seed, **over):
    """SGD reference: SAME buffers, SAME loss, no constraint/accept/tail -- ordinary
    factor SGD on current window + a uniform-replay sample (drops through the
    grad-defensive control with the constraint off)."""
    return run_psr_grad_defensive(cfg, batches, seed, constrained=False, use_tail=False, **over)


# ---------------------------------------------------------------------------
# self-tests: the mechanisms must do what we claim
# ---------------------------------------------------------------------------

def _toy_stream(seed=0, n=30, d=12, batch=48):
    rng = np.random.default_rng(seed)
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    S = np.linspace(2.0, 0.7, d)
    T = (U * S[None, :]) @ V.T
    batches = []
    for _ in range(n):
        c = int(rng.integers(0, 6))
        X = rng.standard_normal((batch, 1)) * V[:, c][None, :] + 0.1 * rng.standard_normal((batch, d))
        rms = np.sqrt(np.mean(np.sum(X ** 2, 1))) + 1e-12
        X = X / rms
        Y = X @ T.T + 0.02 * rng.standard_normal((batch, d))
        batches.append((X, Y))
    return batches


def _mw_worst(items, M):
    return float(max(_loss(M, e["X"], e["Y"]) for e in items))


def run_tests():
    results = []
    cfg = {"R": 4, "gd_steps": 15, "lr": 0.15}
    batches = _toy_stream(seed=1)

    # === PSR-LoRA core (minimax reweighted RRR) ===
    print("core: adapter stays rank <= R:")
    snaps, stats = run_psr_lora(cfg, batches, seed=1)
    M = snaps[-1]
    S_ = np.linalg.svd(M, compute_uv=False)
    H._t(int(np.sum(S_ > 1e-6)) <= cfg["R"], f"adapter stays rank <= R ({int(np.sum(S_>1e-6))})", results)

    print("core: reproducible from (X,Y)+seed (oracle-free, deterministic):")
    h1 = H.trajectory_hash(run_psr_lora(cfg, batches, seed=3)[0])
    h2 = H.trajectory_hash(run_psr_lora(cfg, batches, seed=3)[0])
    H._t(h1 == h2, "deterministic & reproducible", results)

    print("core: safety-by-construction (never worse than uniform on buffered worst):")
    # Reconstruct the final buffer for both arms (same reservoir seed -> same buffer)
    # and check the returned operator's buffered worst-window loss is <= the uniform
    # (round-0) solution's.  This is the guarantee that replaces rollback.
    d = batches[0][0].shape[1]
    rngb = np.random.default_rng(1 + 41)
    buf = H.Reservoir(cfg["R"] * 0 + DEFAULT["buf_u"] + DEFAULT["buf_g"], rngb)
    for (Xk, Yk) in batches:
        buf.offer({"X": Xk, "Y": Yk})
    M_mm, _ = _minimax_rrr(buf.items, d, cfg["R"], DEFAULT["rounds"], DEFAULT["mw_eta"], True)
    M_un, _ = _minimax_rrr(buf.items, d, cfg["R"], DEFAULT["rounds"], DEFAULT["mw_eta"], False)
    H._t(_mw_worst(buf.items, M_mm) <= _mw_worst(buf.items, M_un) + 1e-9,
         f"minimax worst {_mw_worst(buf.items, M_mm):.4f} <= uniform worst {_mw_worst(buf.items, M_un):.4f}", results)

    print("core: reweighting is not a no-op (differs from uniform buffer-RRR):")
    s_rw, _ = run_psr_lora(cfg, batches, seed=1, reweight=True)
    s_un, _ = run_buffer_rrr_uniform(cfg, batches, seed=1)
    H._t(not np.allclose(s_rw[-1], s_un[-1]),
         "minimax-reweighted RRR differs from uniform-weight buffer-RRR", results)

    print("core: minimax lowers the buffered worst-window loss vs uniform (meta):")
    # On a stream with a genuine capacity floor the reweighting must reduce the worst
    # buffered loss for at least most windows -- if it never did, the mechanism would
    # be inert and this test SHOULD fail.
    improved = 0
    rng_a = np.random.default_rng(1 + 41); buf_a = H.Reservoir(DEFAULT["buf_u"] + DEFAULT["buf_g"], rng_a)
    for (Xk, Yk) in batches:
        buf_a.offer({"X": Xk, "Y": Yk})
        mm, _ = _minimax_rrr(buf_a.items, d, cfg["R"], DEFAULT["rounds"], DEFAULT["mw_eta"], True)
        un, _ = _minimax_rrr(buf_a.items, d, cfg["R"], DEFAULT["rounds"], DEFAULT["mw_eta"], False)
        if _mw_worst(buf_a.items, mm) < _mw_worst(buf_a.items, un) - 1e-9:
            improved += 1
    H._t(improved >= len(batches) // 2,
         f"minimax beats uniform on the buffered worst in {improved}/{len(batches)} windows", results)

    # === coverage buffer (the tail-coverage remedy) ===
    print("coverage: total buffer budget stays matched (|U|+|G| <= buf_u+buf_g):")
    # replay the exact offer sequence run_psr_lora uses and check neither part overflows
    rng_c = np.random.default_rng(1 + 41)
    Uc = H.Reservoir(DEFAULT["buf_u"], rng_c); Gc = []
    for (Xk, Yk) in batches:
        Uc.offer({"X": Xk, "Y": Yk})
        _div_offer(Gc, {"X": Xk, "Y": Yk, "idir": _indir(Xk)}, DEFAULT["buf_g"])
    H._t(len(Uc.items) <= DEFAULT["buf_u"] and len(Gc) <= DEFAULT["buf_g"]
         and len(Uc.items) + len(Gc) <= DEFAULT["buf_u"] + DEFAULT["buf_g"],
         f"working set {len(Uc.items)}+{len(Gc)} <= {DEFAULT['buf_u']}+{DEFAULT['buf_g']} budget", results)

    print("coverage: diverse buffer retains rare directions under a redundant flood:")
    # offer cap_g mutually-orthogonal directions once each, then flood with duplicates of
    # direction 0.  A diversity buffer MUST keep all cap_g distinct directions (max off-
    # diagonal |cos| ~ 0); a uniform reservoir would let the flood crowd the rare ones out.
    dd, cap_g = 12, 6
    Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((dd, dd)))
    Gd = []
    for j in range(cap_g):
        _div_offer(Gd, {"idir": Q[:, j].copy()}, cap_g)
    for _ in range(50):
        _div_offer(Gd, {"idir": Q[:, 0].copy()}, cap_g)
    Dm = np.stack([e["idir"] for e in Gd], 0)
    sim = np.abs(Dm @ Dm.T); np.fill_diagonal(sim, 0.0)
    H._t(float(sim.max()) < 0.1,
         f"diverse buffer stays orthogonal under duplicate flood (max|cos|={sim.max():.3f})", results)

    print("coverage: coverage buffer is not a no-op (differs from single reservoir):")
    s_cov, _ = run_psr_lora(cfg, batches, seed=1, coverage=True)
    s_nocov, _ = run_psr_no_coverage(cfg, batches, seed=1)
    H._t(not np.allclose(s_cov[-1], s_nocov[-1]),
         "coverage-buffer solution differs from same-budget single-reservoir solution", results)

    print("coverage: reproducible by hash (oracle-free, deterministic):")
    hc1 = H.trajectory_hash(run_psr_lora(cfg, batches, seed=5, coverage=True)[0])
    hc2 = H.trajectory_hash(run_psr_lora(cfg, batches, seed=5, coverage=True)[0])
    H._t(hc1 == hc2, "coverage path deterministic & reproducible", results)

    # === grad-dynamics negative control: the two properties D2-D FAILED ===
    print("control: rejected-update full recovery (real rollback):")
    snaps, stats = run_psr_grad_defensive(cfg, batches, seed=1, eps_avg=-1.0, eps_tail=-1.0, gamma=10.0)
    H._t(stats["accepts"] == 0 and stats["rejects"] == len(batches),
         f"impossible budget -> all {stats['rejects']} windows rejected", results)
    first = snaps[0]
    H._t(all(np.array_equal(first, s) for s in snaps),
         "rejected updates leave the adapter byte-identical across all windows", results)

    print("control: trust-region radius changes the actual update path length:")
    def path_len(snaps):
        return float(sum(np.linalg.norm(snaps[i] - snaps[i - 1]) for i in range(1, len(snaps))))
    s_strict, st_strict = run_psr_grad_defensive(cfg, batches, seed=1, eps_avg=0.02, eps_tail=0.02, gamma=0.05)
    s_loose, st_loose = run_psr_grad_defensive(cfg, batches, seed=1, eps_avg=1.0, eps_tail=1.0, gamma=0.0)
    H._t(path_len(s_loose) > path_len(s_strict) + 1e-6,
         f"looser budget permits more movement (path {path_len(s_loose):.3f} > {path_len(s_strict):.3f}; "
         f"accepts {st_loose['accepts']} vs {st_strict['accepts']})", results)

    n_fail = sum(1 for ok, _ in results if not ok)
    print(f"\n{len(results)-n_fail}/{len(results)} passed")
    return n_fail


if __name__ == "__main__":
    import sys
    sys.exit(1 if run_tests() else 0)
