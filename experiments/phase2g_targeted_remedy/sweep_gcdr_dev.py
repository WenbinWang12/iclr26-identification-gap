"""Development-only sweep for the exploratory GCDR remedy.

All rows use the phase-2f development seeds.  Do not treat this file's output as
held-out evidence; its sole purpose is to choose a frozen structural setting for
a later, disjoint test.
"""

from __future__ import annotations

import numpy as np

import explore_gcdr as E


SETTINGS = [
    {"groups": g, "mean_margin": margin, "rounds": 24, "eta": 0.7}
    for g in (3, 4, 5, 6, 8, 10, 12)
    for margin in (0.01, 0.03)
]


def main():
    cfg = E._cfg()
    worlds = []
    for ts in E.DEV_TEACHERS:
        teacher = E.BM.build_teacher(E.REGIME, cfg, ts)
        for ss in E.DEV_STREAMS:
            windows, freqs = E.BM.make_stream(cfg, ss)
            batches, srcs, _, test = E.BM.materialize(teacher, cfg, windows, ss)
            psr = E.P.run_psr_lora(cfg, batches, E.RUN_SEED)[0][-1]
            cvar = E.BL.run_excess_cvar(cfg, batches, E.RUN_SEED)[0][-1]
            worlds.append((batches, srcs, freqs, test,
                           E._mean_tail(psr, srcs, freqs, test),
                           E._mean_tail(cvar, srcs, freqs, test)))

    psr_mean = np.mean([w[4][0] for w in worlds])
    psr_tail = np.mean([w[4][1] for w in worlds])
    cvar_mean = np.mean([w[5][0] for w in worlds])
    cvar_tail = np.mean([w[5][1] for w in worlds])
    print(f"PSR  mean={psr_mean:.4f} tail={psr_tail:.4f}")
    print(f"CVaR mean={cvar_mean:.4f} tail={cvar_tail:.4f}\n")
    print(f"{'G':>3s} {'margin':>7s} {'mean':>8s} {'tail':>8s} "
          f"{'dPSR':>8s} {'winP':>5s} {'winC':>5s}")
    for setting in SETTINGS:
        mean, tail = [], []
        for batches, srcs, freqs, test, psr_mt, cvar_mt in worlds:
            M, _ = E.run_gcdr(cfg, batches, E.RUN_SEED, **setting)
            m, t = E._mean_tail(M, srcs, freqs, test)
            mean.append(m)
            tail.append(t)
        mean = np.asarray(mean)
        tail = np.asarray(tail)
        psr_world = np.asarray([w[4][1] for w in worlds])
        cvar_world = np.asarray([w[5][1] for w in worlds])
        print(f"{setting['groups']:3d} {setting['mean_margin']:7.3f} "
              f"{mean.mean():8.4f} {tail.mean():8.4f} "
              f"{(tail-psr_world).mean():+8.4f} "
              f"{np.sum(tail>psr_world):5d} {np.sum(tail>cvar_world):5d}")


if __name__ == "__main__":
    main()
