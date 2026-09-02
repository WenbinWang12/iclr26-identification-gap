"""Fixed DEVELOPMENT evaluation on the already-burned SGCR-v2 8x8 panel.

This is not confirmatory: every seed below was opened by the SGCR-v2 locked run.
Only learner seed 31770922 is used, as fixed before this script was executed.
"""

from __future__ import annotations

import time

import numpy as np

import explore_gcdr as E
import explore_sgcr_v2 as V2
import explore_sgcr_v4 as V4
from compare import _paired_ci, _paired_ci_crossed


TEACHERS = [
    39817664, 15328524, 84645970, 74522946,
    71897575, 51205904, 77534061, 25975093,
]
STREAMS = [
    37032991, 78983795, 10988186, 84417538,
    20724470, 26352979, 33758660, 55439889,
]
LEARNER = 31770922


def _rank(model, tolerance=1e-8):
    singular = np.linalg.svd(model, compute_uv=False)
    if singular[0] == 0.0:
        return 0
    return int(np.sum(singular > tolerance * singular[0]))


def _ci(label, difference):
    iid = _paired_ci(difference)
    crossed = _paired_ci_crossed(difference, len(TEACHERS), len(STREAMS))
    print(
        f"{label}: mean={difference.mean():+.5f}; wins="
        f"{int(np.sum(difference > 0))}/{len(difference)}; "
        f"IID95=[{iid[1]:+.5f},{iid[2]:+.5f}]; "
        f"crossed95=[{crossed[1]:+.5f},{crossed[2]:+.5f}]"
    )


def main():
    cfg = E._cfg()
    result = {
        name: {"mean": [], "tail": []}
        for name in ("v4", "v2", "psr")
    }
    worlds = []
    selected = []
    start = time.perf_counter()
    print("DEVELOPMENT / BURNED 8x8 PANEL -- ONE LEARNER")
    print("teachers", TEACHERS)
    print("streams", STREAMS)
    print("learner", LEARNER)

    for teacher_seed in TEACHERS:
        teacher = E.BM.build_teacher(E.REGIME, cfg, teacher_seed)
        for stream_seed in STREAMS:
            windows, frequencies = E.BM.make_stream(cfg, stream_seed)
            batches, sources, _, test = E.BM.materialize(
                teacher, cfg, windows, stream_seed
            )
            v4, stats = V4.run_sgcr_v4(cfg, batches, LEARNER)
            v2 = V2.run_sgcr_v2(cfg, batches, LEARNER)[0]
            psr = E.P.run_psr_lora(cfg, batches, LEARNER)[0][-1]
            for name, model in (("v4", v4), ("v2", v2), ("psr", psr)):
                if _rank(model) > cfg["R"]:
                    raise RuntimeError(f"rank violation: {name}")
                mean, tail = E._mean_tail(
                    model, sources, frequencies, test
                )
                result[name]["mean"].append(float(mean))
                result[name]["tail"].append(float(tail))
            if stats["persistent_numeric_floats"] > stats["numeric_budget_floats"]:
                raise RuntimeError("v4 persistent budget violation")
            selected.append(stats["selected_candidate"])
            worlds.append((teacher_seed, stream_seed))
            index = len(worlds) - 1
            print(
                f"{index + 1:02d}/64 t{teacher_seed}/s{stream_seed}: "
                f"V4 {result['v4']['mean'][-1]:.4f}/"
                f"{result['v4']['tail'][-1]:.4f}  "
                f"V2 {result['v2']['mean'][-1]:.4f}/"
                f"{result['v2']['tail'][-1]:.4f}  "
                f"P {result['psr']['mean'][-1]:.4f}/"
                f"{result['psr']['tail'][-1]:.4f}  "
                f"sel={selected[-1]} elapsed={time.perf_counter()-start:.0f}s",
                flush=True,
            )

    arrays = {
        name: (
            np.asarray(values["mean"]), np.asarray(values["tail"])
        )
        for name, values in result.items()
    }
    print("\n=== aggregate ===")
    for name, (mean, tail) in arrays.items():
        print(
            f"{name}: mean={mean.mean():.6f} sd={mean.std():.6f}; "
            f"tail={tail.mean():.6f} sd={tail.std():.6f}"
        )

    v4_mean, v4_tail = arrays["v4"]
    v2_mean, v2_tail = arrays["v2"]
    psr_mean, psr_tail = arrays["psr"]
    print("\n=== paired comparisons ===")
    _ci("V4 mean - PSR", v4_mean - psr_mean)
    _ci("V4 tail - PSR", v4_tail - psr_tail)
    _ci("V4 mean - V2", v4_mean - v2_mean)
    _ci("V4 tail - V2", v4_tail - v2_tail)

    delta_psr = (v4_tail - psr_tail).reshape(len(TEACHERS), len(STREAMS))
    delta_v2 = (v4_tail - v2_tail).reshape(len(TEACHERS), len(STREAMS))
    print("\n=== teacher tail effects ===")
    for row, teacher_seed in enumerate(TEACHERS):
        print(
            f"t{teacher_seed}: V4-PSR={delta_psr[row].mean():+.6f}; "
            f"V4-V2={delta_v2[row].mean():+.6f}; "
            f"winsPSR={int(np.sum(delta_psr[row] > 0))}/8"
        )

    flat_delta = v4_tail - psr_tail
    worst = int(np.argmin(flat_delta))
    print("\n=== worst case ===")
    print(
        f"t{worlds[worst][0]}/s{worlds[worst][1]}: "
        f"delta={flat_delta[worst]:+.6f}; "
        f"V4={v4_mean[worst]:.6f}/{v4_tail[worst]:.6f}; "
        f"V2={v2_mean[worst]:.6f}/{v2_tail[worst]:.6f}; "
        f"PSR={psr_mean[worst]:.6f}/{psr_tail[worst]:.6f}; "
        f"selected={selected[worst]}"
    )

    stop_mean = bool((v4_mean - psr_mean).mean() >= -0.01)
    stop_tail_v2 = bool(v4_tail.mean() >= v2_tail.mean())
    stop_no_catastrophe = bool(np.min(flat_delta) >= -0.10)
    strong_baseline_positive = bool(flat_delta.mean() > 0.0)
    print("\n=== fixed stop verdict ===")
    print("mean-PSR >= -0.01:", "PASS" if stop_mean else "FAIL")
    print("tail V4 >= tail V2:", "PASS" if stop_tail_v2 else "FAIL")
    print("no tail delta < -0.10:", "PASS" if stop_no_catastrophe else "FAIL")
    print(
        "tail V4 - PSR > 0:",
        "PASS" if strong_baseline_positive else
        "FAIL -- kill further linear-benchmark tuning",
    )
    print(f"wall seconds={time.perf_counter()-start:.1f}")


if __name__ == "__main__":
    main()

