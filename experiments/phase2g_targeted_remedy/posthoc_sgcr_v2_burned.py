"""Post-hoc SGCR-v2 stress test on the already-opened phase-2g panel."""

from __future__ import annotations

import numpy as np

import explore_gcdr as E
import explore_sgcr_v2 as V
from compare import _paired_ci, _paired_ci_crossed


TEACHER_SEEDS = [31415901, 31415902, 31415903, 31415904, 31415905]
STREAM_SEEDS = [16180331, 16180332, 16180333, 16180334]
LEARNER_SEEDS = [7, 17, 29]
METHOD = {
    "max_groups": None,       # structural rule K_max = 2R
    "warmup_windows": 16,
    "refresh_every": 20,
    "minimum_per_cell": 32,
    "rounds": 24,
    "eta": 0.1,
    "mean_margin": 0.01,
    "min_audit_gain": 0.0,
    "audit_fraction": 0.50,
    "refit_on_all": False,
}


def _report(tag, difference, threshold):
    iid = _paired_ci(difference)
    crossed = _paired_ci_crossed(
        difference, len(TEACHER_SEEDS), len(STREAM_SEEDS)
    )
    mean, lo, hi, se, components, df = crossed
    print(f"{tag}: mean {mean:+.4f}; wins "
          f"{int(np.sum(difference > 0))}/{len(difference)}")
    print(f"  IID 95% CI [{iid[1]:+.4f},{iid[2]:+.4f}]")
    print(f"  crossed 95% CI [{lo:+.4f},{hi:+.4f}], SE={se:.4f}, "
          f"df={df}, threshold={threshold:+.4f}, pass={lo > threshold}")
    print(f"  variance components={tuple(float(x) for x in components)}")


def main():
    cfg = E._cfg()
    result = {name: {"mean": [], "tail": [], "raw_tail": []}
              for name in ("sgcr2", "q0", "psr", "cvar")}
    occupancy = []
    selected = []
    print("POST-HOC / BURNED PANEL -- NOT CONFIRMATORY")
    print("method:", METHOD)
    for teacher_seed in TEACHER_SEEDS:
        teacher = E.BM.build_teacher(E.REGIME, cfg, teacher_seed)
        for stream_seed in STREAM_SEEDS:
            windows, frequencies = E.BM.make_stream(cfg, stream_seed)
            batches, sources, _, test = E.BM.materialize(
                teacher, cfg, windows, stream_seed
            )
            per = {name: [] for name in result}
            for learner_seed in LEARNER_SEEDS:
                model, stats = V.run_sgcr_v2(
                    cfg, batches, learner_seed, return_frequency_baseline=True,
                    **METHOD
                )
                q0 = stats["evaluation_frequency_baseline_model"]
                psr = E.P.run_psr_lora(cfg, batches, learner_seed)[0][-1]
                cvar = E.BL.run_excess_cvar(cfg, batches, learner_seed)[0][-1]
                per["sgcr2"].append(
                    E._mean_tail(model, sources, frequencies, test)
                )
                per["q0"].append(
                    E._mean_tail(q0, sources, frequencies, test)
                )
                per["psr"].append(
                    E._mean_tail(psr, sources, frequencies, test)
                )
                per["cvar"].append(
                    E._mean_tail(cvar, sources, frequencies, test)
                )
                occupancy.append(stats["actual_train_entries"]
                                 + stats["actual_audit_entries"])
                selected.append(stats["selected_round"])
            for name in result:
                values = np.asarray(per[name])
                result[name]["mean"].append(float(values[:, 0].mean()))
                result[name]["tail"].append(float(values[:, 1].mean()))
                result[name]["raw_tail"].extend(values[:, 1].tolist())
            print(f"t{teacher_seed}/s{stream_seed}: "
                  f"S {result['sgcr2']['mean'][-1]:.3f}/{result['sgcr2']['tail'][-1]:.3f}  "
                  f"Q {result['q0']['mean'][-1]:.3f}/{result['q0']['tail'][-1]:.3f}  "
                  f"P {result['psr']['mean'][-1]:.3f}/{result['psr']['tail'][-1]:.3f}  "
                  f"C {result['cvar']['mean'][-1]:.3f}/{result['cvar']['tail'][-1]:.3f}")

    arrays = {}
    print("\n=== aggregate after within-world learner-seed averaging ===")
    for name in result:
        mean = np.asarray(result[name]["mean"])
        tail = np.asarray(result[name]["tail"])
        arrays[name] = mean, tail
        print(f"{name:6s}: mean {mean.mean():.4f} +/- {mean.std():.4f}; "
              f"tail {tail.mean():.4f} +/- {tail.std():.4f}; "
              f"raw learner tail SD {np.std(result[name]['raw_tail']):.4f}")
    sm, st = arrays["sgcr2"]
    qm, qt = arrays["q0"]
    pm, pt = arrays["psr"]
    _, ct = arrays["cvar"]
    print("\n=== descriptive prospective-style criteria (still post-hoc) ===")
    _report("mean non-inferiority SGCR-v2 - PSR", sm - pm, -0.01)
    _report("tail superiority SGCR-v2 - PSR", st - pt, 0.0)
    _report("tail superiority SGCR-v2 - CVaR", st - ct, 0.0)
    _report("tail mechanism SGCR-v2 - same-pool q0", st - qt, 0.0)
    print(f"occupancy mean/min/max {np.mean(occupancy):.1f}/"
          f"{np.min(occupancy)}/{np.max(occupancy)}")
    print(f"final selected-round-zero rate "
          f"{np.mean(np.asarray(selected) == 0):.3f}")


if __name__ == "__main__":
    main()
