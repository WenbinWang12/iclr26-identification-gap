"""Locked phase-2g test; protocol is notes/phase2g_gcdr_locked_test.md."""

from __future__ import annotations

import numpy as np

import explore_gcdr as E
from compare import _paired_ci, _paired_ci_crossed


TEACHER_SEEDS = [31415901, 31415902, 31415903, 31415904, 31415905]
STREAM_SEEDS = [16180331, 16180332, 16180333, 16180334]
LEARNER_SEEDS = [7, 17, 29]
METHOD = {"groups": 6, "rounds": 24, "eta": 0.7, "mean_margin": 0.01}
MEAN_MARGIN = 0.01


def _crossed(tag, diff, strict=True):
    iid = _paired_ci(diff)
    crossed = _paired_ci_crossed(diff, len(TEACHER_SEEDS), len(STREAM_SEEDS))
    mean, lo, hi, se, comps, df = crossed
    passed = lo > (0.0 if strict else -MEAN_MARGIN)
    print(f"{tag}: {'PASS' if passed else 'FAIL'}")
    print(f"  paired mean {mean:+.4f}; world wins {int(np.sum(diff > 0))}/{len(diff)}")
    print(f"  IID 95% CI [{iid[1]:+.4f},{iid[2]:+.4f}] (descriptive)")
    print(f"  crossed random-effects 95% CI [{lo:+.4f},{hi:+.4f}], "
          f"SE={se:.4f}, df={df}, var={tuple(float(x) for x in comps)}")
    return passed


def main():
    cfg = E._cfg()
    out = {name: {"mean": [], "tail": [], "seed_mean": [], "seed_tail": []}
           for name in ("gcdr", "psr", "cvar")}
    print("locked method:", METHOD)
    print("learner seeds:", LEARNER_SEEDS)
    for ts in TEACHER_SEEDS:
        teacher = E.BM.build_teacher(E.REGIME, cfg, ts)
        for ss in STREAM_SEEDS:
            windows, freqs = E.BM.make_stream(cfg, ss)
            batches, srcs, _, test = E.BM.materialize(teacher, cfg, windows, ss)
            per = {name: [] for name in out}
            for learner_seed in LEARNER_SEEDS:
                M_g, _ = E.run_gcdr(cfg, batches, learner_seed, **METHOD)
                M_p = E.P.run_psr_lora(cfg, batches, learner_seed)[0][-1]
                M_c = E.BL.run_excess_cvar(cfg, batches, learner_seed)[0][-1]
                per["gcdr"].append(E._mean_tail(M_g, srcs, freqs, test))
                per["psr"].append(E._mean_tail(M_p, srcs, freqs, test))
                per["cvar"].append(E._mean_tail(M_c, srcs, freqs, test))
            for name in out:
                vals = np.asarray(per[name])
                out[name]["seed_mean"].extend(vals[:, 0])
                out[name]["seed_tail"].extend(vals[:, 1])
                out[name]["mean"].append(float(vals[:, 0].mean()))
                out[name]["tail"].append(float(vals[:, 1].mean()))
            print(f"t{ts}/s{ss}: "
                  f"G {out['gcdr']['mean'][-1]:.3f}/{out['gcdr']['tail'][-1]:.3f}  "
                  f"P {out['psr']['mean'][-1]:.3f}/{out['psr']['tail'][-1]:.3f}  "
                  f"C {out['cvar']['mean'][-1]:.3f}/{out['cvar']['tail'][-1]:.3f}")

    print("\n=== summaries: mean over 20 worlds after within-world seed averaging ===")
    for name in out:
        m = np.asarray(out[name]["mean"])
        t = np.asarray(out[name]["tail"])
        raw_t = np.asarray(out[name]["seed_tail"])
        print(f"{name:5s}: mean {m.mean():.4f} +/- {m.std():.4f}; "
              f"tail {t.mean():.4f} +/- {t.std():.4f}; "
              f"raw learner-seed tail SD {raw_t.std():.4f}")

    gm = np.asarray(out["gcdr"]["mean"])
    gt = np.asarray(out["gcdr"]["tail"])
    pm = np.asarray(out["psr"]["mean"])
    pt = np.asarray(out["psr"]["tail"])
    ct = np.asarray(out["cvar"]["tail"])
    print("\n=== locked criteria ===")
    a = _crossed("(a) mean non-inferiority GCDR - PSR", gm - pm, strict=False)
    b = _crossed("(b) tail superiority GCDR - PSR", gt - pt, strict=True)
    c = _crossed("(c) tail superiority GCDR - excess-CVaR", gt - ct, strict=True)
    print("\nVERDICT:", "PASS targeted remedy" if a and b and c else "FAIL targeted remedy")
    return 0 if a and b and c else 1


if __name__ == "__main__":
    raise SystemExit(main())
