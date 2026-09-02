"""Diagnostic (NOT a method, NOT a claim): WHY does PSR-LoRA fail to robustly beat
excess-CVaR on the tail across teacher geometries?

The v2 kill test showed PSR beats excess-CVaR on tail for teachers 20270001/2/5 but
LOSES (or ties) on 20270003/20270004.  We separate two hypotheses, using teacher
internals that the METHOD never sees (this file is analysis only):

  H1 COVERAGE: the single uniform reservoir under-represents the worst SOURCE, so the
     pooled teacher / minimax weights never 'see' the rare direction enough.  Test:
     measure, per world, the buffered count of the eventual worst source and correlate
     with the PSR-vs-CVaR tail gap.

  H2 OBJECTIVE MISMATCH (concern 5): minimax minimizes the buffered WORST-WINDOW loss,
     but the metric is held-out WORST-SOURCE retention.  A window mixes 2-4 sources, so
     the worst window is not the worst source; optimizing the former can leave the
     latter unserved.  Test: compare buffered-worst-window identity to worst-source
     identity, and check whether an ORACLE source-weighted RRR (upweight the true worst
     source) would have cleared the tail where PSR did not.

Run:  python diagnose_failures.py
"""

from __future__ import annotations

import numpy as np

import benchmark_v4 as BM
import baselines as BL
import psr_lora as P
import harness as H
from compare import TEACHER_SEEDS, STREAM_SEEDS, REGIME, RUN_SEED, _cfg


def _ret_by_source(M, srcs, test):
    return {c: BM.source_retention(M, *test[c]) for c in srcs}


def _worst_source(M, srcs, test):
    r = _ret_by_source(M, srcs, test)
    c = min(r, key=r.get)
    return c, r[c]


def diagnose():
    cfg = _cfg()
    print("per-world failure diagnosis (PSR vs excess-CVaR tail):\n")
    print(f"{'world':16s} {'psrT':>6s} {'cvarT':>6s} {'gap':>7s} "
          f"{'wsrc':>4s} {'buf#ws':>7s} {'oracleT':>7s} {'note'}")
    rows = []
    for ts in TEACHER_SEEDS:
        teacher = BM.build_teacher(REGIME, cfg, ts)
        for ss in STREAM_SEEDS:
            windows, freqs = BM.make_stream(cfg, ss)
            batches, srcs, otr, test = BM.materialize(teacher, cfg, windows, ss)

            # --- PSR-LoRA and its final buffer (reconstruct with the SAME seed) ---
            M_psr = P.run_psr_lora(cfg, batches, RUN_SEED)[0][-1]
            M_cvar = BL.run_excess_cvar(cfg, batches, RUN_SEED)[0][-1]
            _, psrT = _worst_source(M_psr, srcs, test)
            _, cvarT = _worst_source(M_cvar, srcs, test)
            wsrc, _ = _worst_source(M_psr, srcs, test)
            gap = psrT - cvarT

            # H1: how many times is the eventual worst SOURCE present in the final buffer?
            rng = np.random.default_rng(RUN_SEED + 41)
            buf = H.Reservoir(cfg["buf_u"] + cfg["buf_g"], rng)
            # replay the exact offer sequence PSR used (window index order)
            for (Xk, Yk) in batches:
                buf.offer({"X": Xk, "Y": Yk})
            # which sources does each buffered window contain? recover by matching the
            # window inputs' dominant direction to Vin (analysis-only oracle).
            Vin = teacher["Vin"]
            def win_sources(X):
                # project batch mean input energy onto each source direction
                proj = np.abs(X @ Vin)                    # (n, K)
                energy = proj.mean(0)
                return set(np.where(energy > 0.5 * energy.max())[0])
            buf_ws = sum(1 for e in buf.items if wsrc in win_sources(e["X"]))

            # H2: oracle SOURCE-weighted RRR -- upweight the TRUE worst source 30x and
            # re-solve from the buffer teacher; does serving the worst SOURCE (not the
            # worst window) clear the tail where PSR did not?
            d = cfg["d"]
            Th = P._pooled_teacher(buf.items, d, cfg.get("ridge", 1e-6))
            # per-source second moments from the buffer windows (approx via test inputs)
            Sig = [BM._second_moment(test[c][0]) for c in srcs]
            w = np.ones(len(srcs))
            w[srcs.index(wsrc)] = 30.0
            M_oracle = H.weighted_rrr(Th, Sig, list(w / w.sum()), cfg["R"])
            _, oracleT = _worst_source(M_oracle, srcs, test)

            note = ""
            if gap < 0:
                note = "PSR LOSES"
            elif gap < 0.02:
                note = "~tie"
            rows.append((ts, ss, psrT, cvarT, gap, wsrc, buf_ws, oracleT, note))
            print(f"t{ts%100000:05d}/s{ss:04d}  {psrT:6.3f} {cvarT:6.3f} {gap:+7.3f} "
                  f"{wsrc:4d} {buf_ws:7d} {oracleT:7.3f}  {note}")

    # aggregate the two hypotheses
    arr = rows
    gaps = np.array([r[4] for r in arr])
    bufws = np.array([r[6] for r in arr])
    oracleT = np.array([r[7] for r in arr])
    psrT = np.array([r[2] for r in arr])
    losing = gaps < 0.02

    print("\n--- H1 COVERAGE ---")
    if bufws.std() > 0:
        cc = float(np.corrcoef(bufws, gaps)[0, 1])
        print(f"corr(worst-source buffer count, PSR-vs-CVaR gap) = {cc:+.3f}")
    print(f"worst-source buffer count on LOSING/tie worlds: {bufws[losing].mean():.2f} "
          f"vs winning {bufws[~losing].mean():.2f} (cap {cfg['buf_u']+cfg['buf_g']})")

    print("\n--- H2 OBJECTIVE MISMATCH (worst-window vs worst-source) ---")
    gain_oracle = oracleT - psrT
    print(f"oracle SOURCE-weighted RRR tail - PSR tail: mean {gain_oracle.mean():+.3f} "
          f"(on losing/tie worlds {gain_oracle[losing].mean():+.3f})")
    print(f"  worlds where oracle-source-weight would BEAT PSR by >0.03: "
          f"{int(np.sum(gain_oracle > 0.03))}/{len(arr)}")
    print("\nreading: if H2 gain is large on losing worlds, the fix is to target the")
    print("worst SOURCE (group DRO over recovered source groups), not the worst WINDOW.")


if __name__ == "__main__":
    diagnose()
