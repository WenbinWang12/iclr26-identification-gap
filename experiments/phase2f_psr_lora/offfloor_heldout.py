"""Off-floor robustness at the FROZEN coverage config (buf_u=4, buf_g=12).

The paper's off-floor numbers (compressible/conflicting) were previously taken
from compare.py, which runs the 8/8 hybrid -- a DIFFERENT method than the frozen
4/12 held-out run in preregister_heldout.py.  To keep every reported number
traceable to ONE method, this regenerates the off-floor no-harm / graceful checks
using the SAME frozen config as the held-out kill test.

It also re-verifies two held-out worlds against outputs_heldout.txt so we know the
environment reproduces the frozen numbers before trusting the fresh off-floor ones.

Run:  python offfloor_heldout.py > outputs_offfloor_heldout.txt
"""
from __future__ import annotations

import numpy as np

import benchmark_v4 as BM
import baselines as BL
import psr_lora as P
from compare import _mean_tail
from preregister_heldout import _cfg, TEACHER_SEEDS, STREAM_SEEDS, RUN_SEED


ARMS = {
    "psr_lora": lambda cfg, b, s: P.run_psr_lora(cfg, b, s)[0][-1],
    "excess_cvar": lambda cfg, b, s: BL.run_excess_cvar(cfg, b, s)[0][-1],
    "buffer_rrr_uniform": lambda cfg, b, s: P.run_buffer_rrr_uniform(cfg, b, s)[0][-1],
}


def verify_two_worlds():
    """Reproduce the first two held-out worlds (capacity_limited) and check the psr
    tail against the frozen outputs_heldout.txt values (0.805, 0.751)."""
    cfg = _cfg()
    expect = {(20280001, 8001): 0.805, (20280001, 8002): 0.751}
    print("== reproducibility spot-check vs outputs_heldout.txt (frozen 4/12) ==")
    ok = True
    ts = TEACHER_SEEDS[0]
    teacher = BM.build_teacher("capacity_limited", cfg, ts)
    for ss in STREAM_SEEDS[:2]:
        windows, freqs = BM.make_stream(cfg, ss)
        batches, srcs, otr, test = BM.materialize(teacher, cfg, windows, ss)
        M = ARMS["psr_lora"](cfg, batches, RUN_SEED)
        _, t = _mean_tail(M, srcs, freqs, test)
        exp = expect[(ts, ss)]
        match = abs(t - exp) < 5e-4
        ok = ok and match
        print(f"  world t{ts}/s{ss}: psr tail {t:.3f}  (expected {exp:.3f})  "
              f"{'OK' if match else 'MISMATCH'}")
    print(f"  reproducibility: {'PASS' if ok else 'FAIL'}\n")
    return ok


def offfloor():
    """Same off-floor protocol as compare.robustness_offfloor, but at the frozen
    4/12 config: 3 teachers x 2 streams on the compressible (no floor) and
    conflicting (impossibility) regimes."""
    cfg = _cfg()
    out = {}
    for regime in ("compressible", "conflicting"):
        agg = {a: [[], []] for a in ARMS}
        for ts in TEACHER_SEEDS[:3]:
            teacher = BM.build_teacher(regime, cfg, ts)
            for ss in STREAM_SEEDS[:2]:
                windows, freqs = BM.make_stream(cfg, ss)
                batches, srcs, otr, test = BM.materialize(teacher, cfg, windows, ss)
                for a in agg:
                    M = ARMS[a](cfg, batches, RUN_SEED)
                    m, t = _mean_tail(M, srcs, freqs, test)
                    agg[a][0].append(m); agg[a][1].append(t)
        out[regime] = {a: (float(np.mean(v[0])), float(np.mean(v[1]))) for a, v in agg.items()}
        print(f"[off-floor @ frozen 4/12] {regime}:")
        for a, (m, t) in out[regime].items():
            print(f"    {a:20s} mean {m:.3f} tail {t:.3f}")
    comp = out["compressible"]; conf = out["conflicting"]
    no_harm = comp["psr_lora"][0] >= comp["buffer_rrr_uniform"][0] - 0.005
    graceful = conf["psr_lora"][0] >= conf["buffer_rrr_uniform"][0] - 0.005
    print(f"[off-floor] compressible mean not hurt: {'PASS' if no_harm else 'FAIL'}; "
          f"conflicting graceful (>= uniform solve): {'PASS' if graceful else 'FAIL'}")
    return out


if __name__ == "__main__":
    verify_two_worlds()
    offfloor()
