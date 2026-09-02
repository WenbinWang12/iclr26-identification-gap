"""Inspect the largest SGCR-v2 failure on the burned development panel."""

from __future__ import annotations

import numpy as np

import explore_gcdr as E
import explore_sgcr_v2 as V


TEACHER_SEED = 31415903
STREAM_SEED = 16180334
LEARNER_SEED = 7


def _retentions(model, sources, test):
    return {c: E.BM.source_retention(model, *test[c]) for c in sources}


def _behavior_proxy(X, teacher):
    x = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    source = np.argmax((x @ teacher["Vin"]) ** 2, axis=1)
    source[source == 1] = 0  # planted c0/c1 are observationally identical
    return source


def _contingency(labels, behavior, groups):
    values = sorted(set(int(x) for x in behavior))
    table = np.zeros((groups, len(values)), dtype=int)
    for label, source in zip(labels, behavior):
        table[int(label), values.index(int(source))] += 1
    purity = table.max(axis=1).sum() / table.sum()
    return values, table, purity


def main():
    cfg = E._cfg()
    teacher = E.BM.build_teacher(E.REGIME, cfg, TEACHER_SEED)
    windows, frequencies = E.BM.make_stream(cfg, STREAM_SEED)
    batches, sources, _, test = E.BM.materialize(
        teacher, cfg, windows, STREAM_SEED
    )
    model, stats = V.run_sgcr_v2(
        cfg, batches, LEARNER_SEED, return_debug=True
    )
    baseline, baseline_stats = V.run_sgcr_v2(
        cfg, batches, LEARNER_SEED, rounds=0, return_debug=True
    )
    psr = E.P.run_psr_lora(cfg, batches, LEARNER_SEED)[0][-1]
    cvar = E.BL.run_excess_cvar(cfg, batches, LEARNER_SEED)[0][-1]

    for name, candidate in (("SGCR-v2", model), ("q0", baseline),
                            ("PSR", psr), ("CVaR", cvar)):
        mean, tail = E._mean_tail(candidate, sources, frequencies, test)
        retention = _retentions(candidate, sources, test)
        worst = min(retention, key=retention.get)
        print(f"{name:7s}: mean={mean:.4f}, tail={tail:.4f}, "
              f"worst=c{worst}, per-source={retention}")

    print("\nfinal gate:", {
        key: stats[key] for key in (
            "selected_round", "baseline_audit_mean_nmse",
            "baseline_audit_worst_nmse", "selected_audit_mean_nmse",
            "selected_audit_worst_nmse", "feasible_candidates",
            "arrival_frequencies", "train_group_sizes", "audit_group_sizes",
            "selected_round_zero_rate"
        )
    })
    print("q0 final gate:", {
        key: baseline_stats[key] for key in (
            "baseline_audit_mean_nmse", "baseline_audit_worst_nmse"
        )
    })
    for partition in ("train", "audit"):
        X = stats[f"debug_{partition}_x"]
        labels = stats[f"debug_{partition}_labels"]
        behavior = _behavior_proxy(X, teacher)
        values, table, purity = _contingency(labels, behavior, 2 * cfg["R"])
        print(f"\n{partition} behavior columns {values}; purity={purity:.3f}")
        print(table)


if __name__ == "__main__":
    main()
