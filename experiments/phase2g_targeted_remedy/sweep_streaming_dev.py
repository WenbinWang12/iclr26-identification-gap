"""Small structural sweep for streaming GCDR on the development panel only."""

from __future__ import annotations

import numpy as np

import explore_gcdr as E
import explore_streaming_gcdr as S


SETTINGS = [
    # Clean test of the failure-driven hypothesis: stable frozen-activation
    # geometry identifies the latent behaviour, while noisy instantaneous
    # residual gradients are not allowed to dominate cluster assignment.
    {"groups": 7, "activation_weight": 1.0, "gradient_weight": 0.0,
     "refresh_every": 20},
    {"groups": 8, "activation_weight": 1.0, "gradient_weight": 0.0,
     "refresh_every": 20},
    # One two-view control.  This checks whether adding the gradient view helps
    # once it is put on the same scale as the activation view.
    {"groups": 8, "activation_weight": 1.0, "gradient_weight": 1.0,
     "refresh_every": 20},
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
            worlds.append((batches, srcs, freqs, test,
                           E._mean_tail(psr, srcs, freqs, test)))

    psr_mean = np.mean([w[4][0] for w in worlds])
    psr_tail = np.mean([w[4][1] for w in worlds])
    print(f"PSR mean/tail {psr_mean:.4f}/{psr_tail:.4f}\n")
    print(f"{'G':>2s} {'gw':>5s} {'refresh':>7s} {'mean':>8s} "
          f"{'tail':>8s} {'dPSR':>8s} {'wins':>5s}")
    for setting in SETTINGS:
        values = []
        for batches, srcs, freqs, test, _ in worlds:
            M, _ = S.run_streaming_gcdr(cfg, batches, E.RUN_SEED, **setting)
            values.append(E._mean_tail(M, srcs, freqs, test))
        values = np.asarray(values)
        psr_world = np.asarray([w[4][1] for w in worlds])
        print(f"{setting['groups']:2d} {setting['gradient_weight']:5.1f} "
              f"{setting['refresh_every']:7d} "
              f"{values[:,0].mean():8.4f} {values[:,1].mean():8.4f} "
              f"{(values[:,1]-psr_world).mean():+8.4f} "
              f"{np.sum(values[:,1]>psr_world):5d}")


if __name__ == "__main__":
    main()
