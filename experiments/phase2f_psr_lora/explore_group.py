"""EXPLORATION (not a claim, not the kill test): does a GROUP-DRO minimax -- over
oracle-free recovered source groups -- fix the worst-window/worst-source mismatch
the failure diagnosis found?

Runs ONLY on DEV + non-held-out explore teachers (seeds 2026xxxx), never on the
frozen held-out worlds (20270001..5 / 7001..4).  If group-minimax clearly beats
window-minimax here and approaches the oracle source-weighted upper bound, THEN we
freeze it into psr_lora.py and pre-register a single held-out kill test.

Three arms compared on tail (worst-SOURCE held-out retention):
  window-minimax : current PSR-LoRA (minimax over buffered WORST WINDOW).
  group-minimax  : minimax over GROUPS from spherical-k-means of window input
                   directions (oracle-free: uses X only).  rare source starts at
                   weight 1/G, not 1/n, and the group loss targets a source proxy.
  oracle-source  : upweight the TRUE worst source 30x (uses Vin -> UPPER BOUND only).

Also reports cluster PURITY vs the true source labels (Vin used for EVALUATION only).

Run:  python explore_group.py
"""

from __future__ import annotations

import numpy as np

import benchmark_v4 as BM
import psr_lora as P
import harness as H
from compare import REGIME, RUN_SEED, _cfg

DEV_TEACHERS = [20260701, 20260710, 20260711, 20260712, 20260713]   # NOT held-out
DEV_STREAMS = [6060, 6061]                                          # NOT 7001..4


# ---------------------------------------------------------------------------
# oracle-free source-group recovery
# ---------------------------------------------------------------------------

def _window_indir(X):
    """Dominant input direction of a window (top eigenvector of X^T X). Oracle-free."""
    Sig = X.T @ X
    w, V = np.linalg.eigh(Sig)
    return V[:, -1]


def _spherical_kmeans(dirs, G, rng, iters=30):
    """Cluster unit vectors by |cosine| (sign-invariant). Returns integer labels."""
    n = dirs.shape[0]
    G = min(G, n)
    C = dirs[rng.choice(n, G, replace=False)].copy()
    labels = -np.ones(n, int)
    for _ in range(iters):
        new = np.argmax(np.abs(dirs @ C.T), axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for g in range(G):
            mem = dirs[labels == g]
            if len(mem):
                _, _, vt = np.linalg.svd(mem, full_matrices=False)
                C[g] = vt[0]
    return labels


def _minimax_group_rrr(items, d, R, rounds, eta, ridge, G, seed):
    """Minimax over recovered source GROUPS (not windows). Keeps the iterate with the
    smallest buffered worst-GROUP loss. Round 0 = group-uniform weights."""
    n = len(items)
    Th = P._pooled_teacher(items, d, ridge)
    Sg = [H.second_moment(e["X"]) for e in items]
    dirs = np.stack([_window_indir(e["X"]) for e in items], 0)
    dirs = dirs * np.sign(dirs[:, [0]] + 1e-12)          # sign-canonicalize
    rng = np.random.default_rng(seed)
    labels = _spherical_kmeans(dirs, G, rng)
    groups = [np.where(labels == g)[0] for g in range(labels.max() + 1)]
    groups = [gg for gg in groups if len(gg)]
    Gn = len(groups)
    wg = np.ones(Gn) / Gn
    best_M, best_worst = None, np.inf
    for r in range(rounds + 1):
        w = np.zeros(n)
        for gi, gg in enumerate(groups):
            w[gg] = wg[gi] / len(gg)                      # spread group weight over members
        M = H.weighted_rrr(Th, Sg, list(w), R)
        losses = np.array([P._loss(M, e["X"], e["Y"]) for e in items])
        gloss = np.array([losses[gg].mean() for gg in groups])
        worst = float(gloss.max())
        if worst < best_worst - 1e-12:
            best_worst, best_M = worst, M
        wg = wg * np.exp(eta * gloss / (gloss.mean() + 1e-12))
        wg /= wg.sum()
    return best_M, labels


def _run_group(cfg, batches, seed, G):
    d = batches[0][0].shape[1]
    cap = cfg["buf_u"] + cfg["buf_g"]
    rng = np.random.default_rng(seed + 41)
    buf = H.Reservoir(cap, rng)
    M = np.zeros((d, d)); last_labels = None
    for (Xk, Yk) in batches:
        buf.offer({"X": Xk, "Y": Yk})
        M, last_labels = _minimax_group_rrr(buf.items, d, cfg["R"], cfg.get("rounds", 12),
                                            cfg.get("mw_eta", 1.0), cfg.get("ridge", 1e-6),
                                            G, seed + 41)
    return M, buf, last_labels


def _worst_source_tail(M, srcs, test):
    r = {c: BM.source_retention(M, *test[c]) for c in srcs}
    return min(r.values()), min(r, key=r.get)


def _true_win_source(X, Vin):
    energy = np.abs(X @ Vin).mean(0)
    return int(np.argmax(energy))                         # dominant true source of a window


def _purity(buf, labels, Vin):
    truth = np.array([_true_win_source(e["X"], Vin) for e in buf.items])
    tot = 0
    for g in range(labels.max() + 1):
        mem = truth[labels == g]
        if len(mem):
            vals, cnts = np.unique(mem, return_counts=True)
            tot += cnts.max()
    return tot / len(truth)


def main():
    cfg = _cfg()
    print(f"{'world':14s} {'winMM':>6s} {'grpG6':>6s} {'grpG8':>6s} "
          f"{'oracle':>6s} {'purG8':>6s}  {'best'}")
    agg = {"win": [], "g6": [], "g8": [], "oracle": []}
    for ts in DEV_TEACHERS:
        teacher = BM.build_teacher(REGIME, cfg, ts)
        Vin = teacher["Vin"]
        for ss in DEV_STREAMS:
            windows, freqs = BM.make_stream(cfg, ss)
            batches, srcs, otr, test = BM.materialize(teacher, cfg, windows, ss)

            M_win = P.run_psr_lora(cfg, batches, RUN_SEED)[0][-1]
            t_win, _ = _worst_source_tail(M_win, srcs, test)

            M_g6, _, _ = _run_group(cfg, batches, RUN_SEED, 6)
            t_g6, _ = _worst_source_tail(M_g6, srcs, test)
            M_g8, buf8, lab8 = _run_group(cfg, batches, RUN_SEED, 8)
            t_g8, _ = _worst_source_tail(M_g8, srcs, test)
            pur8 = _purity(buf8, lab8, Vin)

            # oracle upper bound: upweight the TRUE worst source
            _, wsrc = _worst_source_tail(M_win, srcs, test)
            d = cfg["d"]
            Th = P._pooled_teacher(buf8.items, d, cfg.get("ridge", 1e-6))
            Sig = [BM._second_moment(test[c][0]) for c in srcs]
            w = np.ones(len(srcs)); w[srcs.index(wsrc)] = 30.0
            M_or = H.weighted_rrr(Th, Sig, list(w / w.sum()), cfg["R"])
            t_or, _ = _worst_source_tail(M_or, srcs, test)

            agg["win"].append(t_win); agg["g6"].append(t_g6)
            agg["g8"].append(t_g8); agg["oracle"].append(t_or)
            best = max([("win", t_win), ("g6", t_g6), ("g8", t_g8)], key=lambda x: x[1])[0]
            print(f"t{ts%100000:05d}/s{ss:04d} {t_win:6.3f} {t_g6:6.3f} {t_g8:6.3f} "
                  f"{t_or:6.3f} {pur8:6.2f}  {best}")

    w = np.array(agg["win"]); g6 = np.array(agg["g6"])
    g8 = np.array(agg["g8"]); orc = np.array(agg["oracle"])
    print("\n--- summary over %d explore worlds (tail = worst-source retention) ---" % len(w))
    print(f"  window-minimax : {w.mean():.3f}")
    print(f"  group-minimax6 : {g6.mean():.3f}   (vs window {g6.mean()-w.mean():+.3f}, "
          f"wins {int(np.sum(g6>w))}/{len(w)})")
    print(f"  group-minimax8 : {g8.mean():.3f}   (vs window {g8.mean()-w.mean():+.3f}, "
          f"wins {int(np.sum(g8>w))}/{len(w)})")
    print(f"  oracle-source  : {orc.mean():.3f}   (upper bound)")
    bestg = g6 if g6.mean() >= g8.mean() else g8
    print(f"  group closes {100*(bestg.mean()-w.mean())/(orc.mean()-w.mean()+1e-9):.0f}% "
          f"of the window->oracle gap")


if __name__ == "__main__":
    main()
