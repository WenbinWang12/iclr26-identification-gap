"""EXPLORATION (not a claim, not the kill test): does a COVERAGE-AWARE buffer fix the
rare-source under-coverage the failure diagnosis pinned as the robust cause (H1)?

Diagnosis found: on worlds where PSR fails to beat excess-CVaR on the tail, the true
worst SOURCE appears in the 16-window uniform reservoir only ~1.2 times (vs ~2.4 on
winning worlds), corr(count, gap)=+0.44.  A pure uniform reservoir under-samples the
rare input direction, so the pooled teacher and the window-minimax never 'see' it.

Fix under test (budget-preserving, oracle-free): split the SAME B=16 budget into
  U : buf_u=8 uniform (Vitter) reservoir      -> unbiased average history
  G : buf_g=8 input-direction-diverse buffer  -> guarantees rare directions survive
and feed U u G to the SAME window-minimax RRR that already works.  Total buffer bytes
identical to every baseline; the only change is WHICH windows are kept.

Runs ONLY on DEV + non-held-out explore teachers -- never the frozen held-out worlds.

Arms (tail = worst-SOURCE held-out retention):
  uniform16 : current PSR-LoRA (single uniform reservoir, cap 16).
  cover8+8  : coverage buffer (8 uniform + 8 direction-diverse), same 16 budget.
  cover4+12 : more diversity slots.

Run:  python explore_coverage.py
"""

from __future__ import annotations

import numpy as np

import benchmark_v4 as BM
import psr_lora as P
import harness as H
from compare import REGIME, RUN_SEED, _cfg

DEV_TEACHERS = [20260701, 20260710, 20260711, 20260712, 20260713]   # NOT held-out
DEV_STREAMS = [6060, 6061]                                          # NOT 7001..4


def _indir(X):
    """Dominant input direction of a window (top right singular vector). Oracle-free."""
    _, _, vt = np.linalg.svd(X - X.mean(0, keepdims=True), full_matrices=False)
    return vt[0]


def _div_offer(G, entry, cap):
    """Keep the most input-direction-diverse set: if full, drop the entry whose
    direction is most redundant (highest max |cosine| to the others)."""
    if len(G) < cap:
        G.append(entry); return
    cand = G + [entry]
    D = np.stack([e["idir"] for e in cand], 0)
    sim = np.abs(D @ D.T); np.fill_diagonal(sim, -np.inf)
    drop = int(np.argmax(sim.max(1)))
    if drop != len(cand) - 1:
        G[drop] = entry


def run_coverage(cfg, batches, seed, buf_u, buf_g):
    """PSR-LoRA with a coverage buffer: buf_u uniform + buf_g direction-diverse,
    same window-minimax RRR on the union."""
    d = batches[0][0].shape[1]
    rng = np.random.default_rng(seed + 41)
    U = H.Reservoir(buf_u, rng)
    G = []
    snaps = []
    M = np.zeros((d, d))
    for (Xk, Yk) in batches:
        entry = {"X": Xk, "Y": Yk, "idir": _indir(Xk)}
        U.offer({"X": Xk, "Y": Yk})
        _div_offer(G, entry, buf_g)
        items = U.items + [{"X": e["X"], "Y": e["Y"]} for e in G]
        M, _ = P._minimax_rrr(items, d, cfg["R"], cfg.get("rounds", 12),
                              cfg.get("mw_eta", 1.0), cfg["reweight"] if "reweight" in cfg else True,
                              cfg.get("ridge", 1e-6))
        snaps.append(M.copy())
    return snaps


def _tail(M, srcs, test):
    r = {c: BM.source_retention(M, *test[c]) for c in srcs}
    return min(r.values())


def _worst_src_bufcount(batches, seed, buf_u, buf_g, wsrc, Vin):
    """How many buffered windows are dominated by the worst source (coverage check)."""
    rng = np.random.default_rng(seed + 41)
    U = H.Reservoir(buf_u, rng); G = []
    for (Xk, Yk) in batches:
        entry = {"X": Xk, "Y": Yk, "idir": _indir(Xk)}
        U.offer({"X": Xk, "Y": Yk}); _div_offer(G, entry, buf_g)
    items = U.items + G
    cnt = 0
    for e in items:
        if int(np.argmax(np.abs(e["X"] @ Vin).mean(0))) == wsrc:
            cnt += 1
    return cnt


def _meantail(M, srcs, freqs, test):
    r = {c: BM.source_retention(M, *test[c]) for c in srcs}
    mean = sum(freqs[c] * r[c] for c in srcs) / sum(freqs[c] for c in srcs)
    return float(mean), float(min(r.values()))


def main():
    cfg = dict(_cfg()); cfg["reweight"] = True
    print(f"{'world':14s} {'uni16':>6s} {'c4+12':>6s} {'c2+14':>6s}  "
          f"{'mUni':>6s} {'mC412':>6s}  {'best'}")
    agg = {"u16": [], "c412": [], "c214": [], "mu": [], "mc412": [], "mc214": []}
    for ts in DEV_TEACHERS:
        teacher = BM.build_teacher(REGIME, cfg, ts); Vin = teacher["Vin"]
        for ss in DEV_STREAMS:
            windows, freqs = BM.make_stream(cfg, ss)
            batches, srcs, otr, test = BM.materialize(teacher, cfg, windows, ss)

            M_u = P.run_psr_lora(cfg, batches, RUN_SEED)[0][-1]
            mu, t_u = _meantail(M_u, srcs, freqs, test)

            M_c412 = run_coverage(cfg, batches, RUN_SEED, 4, 12)[-1]
            mc412, t_c412 = _meantail(M_c412, srcs, freqs, test)
            M_c214 = run_coverage(cfg, batches, RUN_SEED, 2, 14)[-1]
            mc214, t_c214 = _meantail(M_c214, srcs, freqs, test)

            agg["u16"].append(t_u); agg["c412"].append(t_c412); agg["c214"].append(t_c214)
            agg["mu"].append(mu); agg["mc412"].append(mc412); agg["mc214"].append(mc214)
            best = max([("u16", t_u), ("c4+12", t_c412), ("c2+14", t_c214)],
                       key=lambda x: x[1])[0]
            print(f"t{ts%100000:05d}/s{ss:04d} {t_u:6.3f} {t_c412:6.3f} {t_c214:6.3f}  "
                  f"{mu:6.3f} {mc412:6.3f}  {best}")

    u = np.array(agg["u16"]); c412 = np.array(agg["c412"]); c214 = np.array(agg["c214"])
    mu = np.array(agg["mu"]); mc412 = np.array(agg["mc412"]); mc214 = np.array(agg["mc214"])
    print("\n--- summary over %d explore worlds (tail = worst-source retention) ---" % len(u))
    print(f"  uniform16  : tail {u.mean():.3f}   mean {mu.mean():.4f}")
    print(f"  cover4+12  : tail {c412.mean():.3f}   (vs uni {c412.mean()-u.mean():+.3f}, "
          f"wins {int(np.sum(c412>u))}/{len(u)})   mean {mc412.mean():.4f} ({mc412.mean()-mu.mean():+.4f})")
    print(f"  cover2+14  : tail {c214.mean():.3f}   (vs uni {c214.mean()-u.mean():+.3f}, "
          f"wins {int(np.sum(c214>u))}/{len(u)})   mean {mc214.mean():.4f} ({mc214.mean()-mu.mean():+.4f})")


if __name__ == "__main__":
    main()
