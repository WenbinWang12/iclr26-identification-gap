"""CROSS-CHECK -- a POST-HOC ROBUSTNESS CHECK ONLY (NOT the strongest evidence, NOT an
independent confirmation): run the FROZEN 4/12 coverage method on the OLD BURNED seeds
(2027xxxx) -- the very worlds on which the PRE-coverage method FAILED criterion (b) in
the honest v2 rerun (only 3/5 teacher geometries positive).

HONESTY CORRECTION (2026-08-27, after codex review): an earlier version of this docstring
called this "the strongest single piece of evidence."  That was an OVERCLAIM and is
RETRACTED.  The 2027 seeds are NOT independent of the hypothesis: they were used to
OBSERVE the (b) failure, to READ source labels for the H1 root-cause diagnosis, and to
MOTIVATE the coverage remedy.  Re-running the frozen method on the worlds that generated
the hypothesis is confirmatory re-analysis, so it controls for the seed draw only weakly
and CANNOT be cited as draw-controlled evidence or as stronger than the held-out test.
The ONLY independent confirmation is the held-out (2028xxxx) PASS in preregister_heldout.py.

This is a POST-HOC re-analysis, not a new kill test.  It reports, it does not decide, and
it must always be reported WITH its post-hoc limitation stated in the same breath.

Run:  python crosscheck_burned.py > outputs_crosscheck_burned.txt
"""

from __future__ import annotations

import numpy as np

import benchmark_v4 as BM
import baselines as BL
import psr_lora as P
from compare import _paired_ci_crossed, _mean_tail

BURNED_T = [20270001, 20270002, 20270003, 20270004, 20270005]   # v1/v2 burned test teachers
BURNED_S = [7001, 7002, 7003, 7004]
REGIME = "capacity_limited"
RUN_SEED = 7


def _cfg():
    """FROZEN coverage config -- identical to psr_lora.DEFAULT and preregister_heldout."""
    cfg = dict(BM.DEFAULT_CFG)
    cfg.update({"gd_steps": 20, "lr": 0.15, "buf_u": 4, "buf_g": 12,
                "tau": 0.4, "xi": 0.02, "coverage": True, "reweight": True})
    return cfg


def main():
    cfg = _cfg()
    psr, nocov, cvar, unif = [], [], [], []
    for ts in BURNED_T:
        teacher = BM.build_teacher(REGIME, cfg, ts)
        for ss in BURNED_S:
            w, fr = BM.make_stream(cfg, ss)
            b, srcs, otr, test = BM.materialize(teacher, cfg, w, ss)
            psr.append(_mean_tail(P.run_psr_lora(cfg, b, RUN_SEED)[0][-1], srcs, fr, test)[1])
            nocov.append(_mean_tail(P.run_psr_no_coverage(cfg, b, RUN_SEED)[0][-1], srcs, fr, test)[1])
            cvar.append(_mean_tail(BL.run_excess_cvar(cfg, b, RUN_SEED)[0][-1], srcs, fr, test)[1])
            unif.append(_mean_tail(P.run_buffer_rrr_uniform(cfg, b, RUN_SEED)[0][-1], srcs, fr, test)[1])
    psr = np.array(psr); nocov = np.array(nocov); cvar = np.array(cvar); unif = np.array(unif)

    print("FROZEN coverage (4/12) PSR-LoRA on the OLD BURNED seeds 2027xxxx")
    print("(the worlds where the PRE-coverage method FAILED (b): v2 was 3/5 teachers positive)\n")
    print(f"  psr_lora tail            {psr.mean():.3f}")
    print(f"  psr_no_coverage tail     {nocov.mean():.3f}   (coverage lift {psr.mean()-nocov.mean():+.3f})")
    print(f"  excess_cvar tail         {cvar.mean():.3f}")
    print(f"  buffer_rrr_uniform tail  {unif.mean():.3f}\n")

    for tag, D in [("(b) psr - excess_cvar", psr - cvar), ("(c) psr - buffer_rrr_uniform", psr - unif)]:
        m, lo, hi, se, comps, df = _paired_ci_crossed(D, len(BURNED_T), len(BURNED_S))
        per_t = D.reshape(len(BURNED_T), len(BURNED_S)).mean(1)
        print(f"  {tag}: mean {m:+.4f}  wins {int((D>0).sum())}/20  "
              f"crossed95 [{lo:+.4f},{hi:+.4f}] (df={df}) -> {'PASS' if lo>0 else 'FAIL'}")
        print(f"      per-teacher means {per_t.round(3)}  (all5 positive: {bool((per_t>0).all())})")

    print("\nREADING (POST-HOC robustness check, NOT independent confirmation): (b) flips")
    print("FAIL->PASS on the SAME burned seeds once coverage is added (v2 pre-coverage: 3/5")
    print("teachers positive, CI straddled 0).  These seeds MOTIVATED the remedy, so this is")
    print("confirmatory re-analysis, not draw-controlled evidence -- do NOT cite as headline.")
    print("(c) remains draw-sensitive: FAILS here but PASSES on the 2028xxxx held-out set.")
    print("The independent confirmation is the held-out PASS; this file only corroborates it.")


if __name__ == "__main__":
    main()
