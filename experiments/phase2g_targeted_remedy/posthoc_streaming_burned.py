"""Post-hoc stress test of streaming stable-group replay on a burned panel.

The seeds in this file were already opened by ``heldout_gcdr.py``.  Results from
this script are therefore development evidence only and can never be reported as
confirmatory.  Its purpose is to decide whether the repaired method is stable
enough to justify one final, genuinely fresh test.
"""

from __future__ import annotations

import numpy as np

import explore_gcdr as E
import explore_streaming_gcdr as S
from compare import _paired_ci, _paired_ci_crossed


TEACHER_SEEDS = [31415901, 31415902, 31415903, 31415904, 31415905]
STREAM_SEEDS = [16180331, 16180332, 16180333, 16180334]
LEARNER_SEEDS = [7, 17, 29]
METHOD = {
    "groups": 8,
    "activation_weight": 1.0,
    "gradient_weight": 0.0,
    "refresh_every": 20,
    "rounds": 24,
    "eta": 0.7,
    "mean_margin": 0.01,
}


def _report(tag, diff, margin=0.0):
    iid = _paired_ci(diff)
    crossed = _paired_ci_crossed(
        diff, len(TEACHER_SEEDS), len(STREAM_SEEDS)
    )
    mean, lo, hi, se, comps, df = crossed
    print(f"{tag}: mean {mean:+.4f}; wins {int(np.sum(diff > 0))}/{len(diff)}")
    print(f"  IID 95% CI [{iid[1]:+.4f},{iid[2]:+.4f}]")
    print(f"  crossed 95% CI [{lo:+.4f},{hi:+.4f}], SE={se:.4f}, "
          f"df={df}, threshold={margin:+.4f}, pass={lo > margin}")
    print(f"  crossed variance components={tuple(float(x) for x in comps)}")


def main():
    cfg = E._cfg()
    out = {name: {"mean": [], "tail": [], "raw_tail": []}
           for name in ("stable", "psr", "cvar")}
    selected_rounds = []
    print("POST-HOC / BURNED PANEL -- NOT CONFIRMATORY")
    print("method:", METHOD)
    for ts in TEACHER_SEEDS:
        teacher = E.BM.build_teacher(E.REGIME, cfg, ts)
        for ss in STREAM_SEEDS:
            windows, freqs = E.BM.make_stream(cfg, ss)
            batches, srcs, _, test = E.BM.materialize(teacher, cfg, windows, ss)
            per = {name: [] for name in out}
            for learner_seed in LEARNER_SEEDS:
                M_s, stats = S.run_streaming_gcdr(
                    cfg, batches, learner_seed, **METHOD
                )
                M_p = E.P.run_psr_lora(cfg, batches, learner_seed)[0][-1]
                M_c = E.BL.run_excess_cvar(cfg, batches, learner_seed)[0][-1]
                per["stable"].append(E._mean_tail(M_s, srcs, freqs, test))
                per["psr"].append(E._mean_tail(M_p, srcs, freqs, test))
                per["cvar"].append(E._mean_tail(M_c, srcs, freqs, test))
                selected_rounds.append(stats["selected_round"])
            for name in out:
                values = np.asarray(per[name])
                out[name]["mean"].append(float(values[:, 0].mean()))
                out[name]["tail"].append(float(values[:, 1].mean()))
                out[name]["raw_tail"].extend(values[:, 1].tolist())
            print(f"t{ts}/s{ss}: "
                  f"S {out['stable']['mean'][-1]:.3f}/{out['stable']['tail'][-1]:.3f}  "
                  f"P {out['psr']['mean'][-1]:.3f}/{out['psr']['tail'][-1]:.3f}  "
                  f"C {out['cvar']['mean'][-1]:.3f}/{out['cvar']['tail'][-1]:.3f}")

    print("\n=== aggregate after within-world learner-seed averaging ===")
    arrays = {}
    for name in out:
        mean = np.asarray(out[name]["mean"])
        tail = np.asarray(out[name]["tail"])
        arrays[name] = (mean, tail)
        print(f"{name:6s}: mean {mean.mean():.4f} +/- {mean.std():.4f}; "
              f"tail {tail.mean():.4f} +/- {tail.std():.4f}; "
              f"raw seed tail SD {np.std(out[name]['raw_tail']):.4f}")
    sm, st = arrays["stable"]
    pm, pt = arrays["psr"]
    _, ct = arrays["cvar"]
    print("\n=== descriptive prospective-style criteria (still post-hoc) ===")
    _report("mean non-inferiority stable - PSR", sm - pm, margin=-0.01)
    _report("tail superiority stable - PSR", st - pt)
    _report("tail superiority stable - CVaR", st - ct)
    print("selected robust round zero rate: "
          f"{np.mean(np.asarray(selected_rounds) == 0):.3f}")


if __name__ == "__main__":
    main()
