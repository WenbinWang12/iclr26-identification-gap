"""DEVELOPMENT-ONLY diagnosis of SGCR-v4's burned catastrophic worlds.

This script opens no fresh seed.  It reruns the three teacher rows containing
the four already reported catastrophes, captures the final learner-visible
selection state, and uses source labels only after fitting for oracle diagnosis.
It does not modify SGCR-v4 or any locked artifact.
"""

from __future__ import annotations

import math
import re

import numpy as np

import explore_gcdr as E
import explore_sgcr_v2 as V
import explore_sgcr_v4 as V4


TEACHERS = [39817664, 51205904, 25975093]
STREAMS = [
    37032991, 78983795, 10988186, 84417538,
    20724470, 26352979, 33758660, 55439889,
]
LEARNER = 31770922
CATASTROPHES = {
    (39817664, 37032991),
    (39817664, 33758660),
    (51205904, 78983795),
    (25975093, 26352979),
}


def _mean_tail(model, sources, frequencies, test):
    return E._mean_tail(model, sources, frequencies, test)


def _source_for_entry(entry, windows, batch):
    window, row = entry["id"]
    components = windows[int(window)]
    per_source = int(batch) // len(components)
    return int(components[min(int(row) // per_source, len(components) - 1)])


def _contingency(grouped, windows, batch, groups, sources):
    source_to_column = {source: j for j, source in enumerate(sources)}
    table = np.zeros((groups, len(sources)), dtype=int)
    for group, entries in enumerate(grouped):
        for entry in entries:
            source = _source_for_entry(entry, windows, batch)
            table[group, source_to_column[source]] += 1
    total = int(table.sum())
    cell_purity = float(table.max(axis=1).sum() / total)
    source_concentration = float(table.max(axis=0).sum() / total)
    return table, cell_purity, source_concentration


def _normalized_mi(table):
    total = float(table.sum())
    joint = table / total
    pg = joint.sum(axis=1)
    ps = joint.sum(axis=0)
    mi = 0.0
    for g in range(table.shape[0]):
        for s in range(table.shape[1]):
            if joint[g, s] > 0.0:
                mi += joint[g, s] * math.log(
                    joint[g, s] / (pg[g] * ps[s])
                )
    hg = -float(np.sum(pg[pg > 0.0] * np.log(pg[pg > 0.0])))
    hs = -float(np.sum(ps[ps > 0.0] * np.log(ps[ps > 0.0])))
    return float(mi / math.sqrt(hg * hs)) if hg > 0.0 and hs > 0.0 else 0.0


def _weighted_teacher_spectrum(grouped, q0, ridge):
    xs, ys, ws = [], [], []
    for weight, entries in zip(q0, grouped):
        X, Y = V._xy(entries)
        xs.append(X)
        ys.append(Y)
        ws.append(np.full(len(entries), float(weight) / len(entries)))
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    weight = np.concatenate(ws)
    weight /= weight.sum()
    cxx = X.T @ (weight[:, None] * X) + ridge * np.eye(X.shape[1])
    cyx = Y.T @ (weight[:, None] * X)
    teacher = np.linalg.solve(cxx, cyx.T).T
    root, _ = E._sqrt_and_inv_sqrt(cxx)
    return np.linalg.svd(teacher @ root, compute_uv=False)


def _selected_round(name):
    match = re.search(r"(?:hedge_|_r)(\d+)", name)
    return int(match.group(1)) if match else 0


def _candidate_metrics(stats):
    rows = stats["candidate_audit"]
    feasible = [row for row in rows if row["feasible"]]
    selected = next(
        row for row in rows if row["name"] == stats["selected_candidate"]
    )
    baseline = next(row for row in rows if row["name"] == "frequency_baseline")
    competitors = sorted(row["worst_ucb"] for row in feasible)
    selection_margin = (
        competitors[1] - competitors[0] if len(competitors) > 1 else np.nan
    )
    spectral = [
        row for row in feasible if row["name"].startswith("window_spectral_")
    ]
    best_spectral = min(
        (row["worst_ucb"] for row in spectral), default=np.nan
    )
    best_spectral_name = (
        min(spectral, key=lambda row: row["worst_ucb"])["name"]
        if spectral else None
    )
    return {
        "paired_mean_ucb": float(selected["mean_delta_ucb"]),
        "mean_gate_slack": float(0.01 - selected["mean_delta_ucb"]),
        "worst_ucb_gain": float(baseline["worst_ucb"] - selected["worst_ucb"]),
        "selection_margin": float(selection_margin),
        "best_spectral_ucb": float(best_spectral),
        "best_spectral_name": best_spectral_name,
        "activation_over_spectral": float(
            best_spectral - selected["worst_ucb"]
        ),
        "spectral_feasible": int(len(spectral)),
        "baseline_worst_ucb": float(baseline["worst_ucb"]),
        "selected_worst_ucb": float(selected["worst_ucb"]),
    }


def _diagnose(teacher_seed, stream_seed):
    cfg = E._cfg()
    teacher = E.BM.build_teacher(E.REGIME, cfg, teacher_seed)
    windows, frequencies = E.BM.make_stream(cfg, stream_seed)
    batches, sources, _, test = E.BM.materialize(
        teacher, cfg, windows, stream_seed
    )

    capture = {}
    original_groups = V4._window_pool_groups_aligned
    original_select = V4._select
    original_activation = V4._activation_train_path
    original_spectral = V4._spectral_pareto_path
    # The deployed model never feeds buffer admission, codebook refresh, or
    # arrival counts in v4.  Candidate construction at non-final refreshes is
    # therefore dead computation for this final-state diagnosis.  Skip it and
    # run the exact path on refresh 12 only; assert that the expected refresh
    # count is reached below.
    expected_fits = int(math.ceil(len(batches) / 20.0))
    path_calls = {"activation": 0, "spectral": 0}

    def wrapped_groups(*args, **kwargs):
        result = original_groups(*args, **kwargs)
        capture["train_grouped"] = result[0]
        capture["audit_grouped"] = result[1]
        capture["centers"] = result[3]
        return result

    def wrapped_select(candidates, *args, **kwargs):
        capture["candidates"] = list(candidates)
        capture["active"] = np.asarray(args[3], dtype=int).copy()
        return original_select(candidates, *args, **kwargs)

    def wrapped_activation(*args, **kwargs):
        path_calls["activation"] += 1
        if path_calls["activation"] < expected_fits:
            return []
        return original_activation(*args, **kwargs)

    def wrapped_spectral(*args, **kwargs):
        path_calls["spectral"] += 1
        if path_calls["spectral"] < expected_fits:
            return []
        return original_spectral(*args, **kwargs)

    V4._window_pool_groups_aligned = wrapped_groups
    V4._select = wrapped_select
    V4._activation_train_path = wrapped_activation
    V4._spectral_pareto_path = wrapped_spectral
    try:
        model, stats = V4.run_sgcr_v4(cfg, batches, LEARNER)
    finally:
        V4._window_pool_groups_aligned = original_groups
        V4._select = original_select
        V4._activation_train_path = original_activation
        V4._spectral_pareto_path = original_spectral
    if path_calls != {"activation": expected_fits, "spectral": expected_fits}:
        raise RuntimeError(f"unexpected refresh count: {path_calls}")

    train_grouped = capture["train_grouped"]
    audit_grouped = V4._unique_grouped(capture["audit_grouped"])
    groups = len(train_grouped)
    train_table, train_purity, train_concentration = _contingency(
        train_grouped, windows, cfg["batch"], groups, sources
    )
    audit_table, audit_purity, audit_concentration = _contingency(
        audit_grouped, windows, cfg["batch"], groups, sources
    )

    q0 = np.asarray(stats["arrival_frequencies"], dtype=float)
    retained = np.asarray(stats["train_group_sizes"], dtype=float)
    retained /= retained.sum()
    cell_tv = float(0.5 * np.abs(q0 - retained).sum())
    max_log_ratio = float(np.max(np.abs(np.log((retained + 1e-12) / (q0 + 1e-12)))))
    coherence = np.asarray(stats["cell_coherence"], dtype=float)
    active = capture["active"]
    singular = _weighted_teacher_spectrum(
        train_grouped, q0, cfg.get("ridge", 1e-6)
    )
    rank = int(cfg["R"])
    spectral_gap = float(singular[rank - 1] - singular[rank])
    spectral_gap_rel = float(spectral_gap / (singular[rank - 1] + 1e-12))

    models = {name: candidate for name, candidate in capture["candidates"]}
    baseline = models["frequency_baseline"]
    candidate_metrics = _candidate_metrics(stats)
    spectral_name = candidate_metrics["best_spectral_name"]
    selected_mean, selected_tail = _mean_tail(model, sources, frequencies, test)
    baseline_mean, baseline_tail = _mean_tail(
        baseline, sources, frequencies, test
    )
    psr = E.P.run_psr_lora(cfg, batches, LEARNER)[0][-1]
    psr_mean, psr_tail = _mean_tail(psr, sources, frequencies, test)
    if spectral_name is not None:
        spectral_mean, spectral_tail = _mean_tail(
            models[spectral_name], sources, frequencies, test
        )
    else:
        spectral_mean, spectral_tail = np.nan, np.nan
    retention = {
        source: E.BM.source_retention(model, *test[source])
        for source in sources
    }
    worst_source = min(retention, key=retention.get)
    worst_column = sources.index(worst_source)
    worst_cell = int(np.argmax(audit_table[:, worst_column]))
    worst_source_recall = float(
        audit_table[worst_cell, worst_column]
        / max(1, audit_table[:, worst_column].sum())
    )
    worst_source_precision = float(
        audit_table[worst_cell, worst_column]
        / max(1, audit_table[worst_cell].sum())
    )

    result = {
        "teacher": int(teacher_seed),
        "stream": int(stream_seed),
        "is_reported_catastrophe": (teacher_seed, stream_seed) in CATASTROPHES,
        "selected": stats["selected_candidate"],
        "round": _selected_round(stats["selected_candidate"]),
        "mean": float(selected_mean),
        "tail": float(selected_tail),
        "psr_mean": float(psr_mean),
        "psr_tail": float(psr_tail),
        "tail_delta_psr": float(selected_tail - psr_tail),
        "q0_mean": float(baseline_mean),
        "q0_tail": float(baseline_tail),
        "tail_delta_q0": float(selected_tail - baseline_tail),
        "spectral_mean": float(spectral_mean),
        "spectral_tail": float(spectral_tail),
        "spectral_tail_delta_psr": float(spectral_tail - psr_tail),
        "spectral_tail_delta_selected": float(spectral_tail - selected_tail),
        "coherence_min": float(coherence.min()),
        "coherence_median": float(np.median(coherence)),
        "coherence_cv": float(coherence.std() / (coherence.mean() + 1e-12)),
        "active_cells": int(len(active)),
        "train_support_min": int(min(stats["train_group_sizes"])),
        "audit_support_min": int(min(stats["audit_group_sizes"])),
        "arrival_retained_tv": cell_tv,
        "arrival_retained_max_log_ratio": max_log_ratio,
        "spectral_gap": spectral_gap,
        "spectral_gap_rel": spectral_gap_rel,
        "train_oracle_purity": train_purity,
        "audit_oracle_purity": audit_purity,
        "train_oracle_source_concentration": train_concentration,
        "audit_oracle_source_concentration": audit_concentration,
        "audit_oracle_nmi": _normalized_mi(audit_table),
        "oracle_worst_source": int(worst_source),
        "oracle_worst_source_cell_recall": worst_source_recall,
        "oracle_worst_source_cell_precision": worst_source_precision,
        **candidate_metrics,
    }
    return result


def _print_row(row):
    print(
        f"t{row['teacher']}/s{row['stream']} "
        f"cat={int(row['is_reported_catastrophe'])} "
        f"dP={row['tail_delta_psr']:+.4f} dQ={row['tail_delta_q0']:+.4f} "
        f"sel={row['selected']} ucbM={row['paired_mean_ucb']:+.4f} "
        f"gainW={row['worst_ucb_gain']:+.4f} selgap={row['selection_margin']:.4g} "
        f"specAdv={row['activation_over_spectral']:+.4f} "
        f"coh={row['coherence_min']:.3f}/{row['coherence_median']:.3f} "
        f"sup={row['train_support_min']}/{row['audit_support_min']} "
        f"TV={row['arrival_retained_tv']:.3f} "
        f"sgap={row['spectral_gap_rel']:.3f} "
        f"pur={row['audit_oracle_purity']:.3f} "
        f"NMI={row['audit_oracle_nmi']:.3f} "
        f"worstSrc={row['oracle_worst_source']} "
        f"rec/pre={row['oracle_worst_source_cell_recall']:.2f}/"
        f"{row['oracle_worst_source_cell_precision']:.2f}"
    )


def _match(catastrophe, positives, used):
    keys = [
        "paired_mean_ucb", "worst_ucb_gain", "selection_margin",
        "coherence_min", "coherence_median", "arrival_retained_tv",
        "spectral_gap_rel",
    ]
    pool = [row for row in positives if (row["teacher"], row["stream"]) not in used]
    preferred = [row for row in pool if row["teacher"] == catastrophe["teacher"]]
    if preferred:
        pool = preferred
    matrix = np.asarray([[row[key] for key in keys] for row in positives])
    scale = np.nanstd(matrix, axis=0) + 1e-8
    target = np.asarray([catastrophe[key] for key in keys])
    def distance(row):
        value = np.asarray([row[key] for key in keys])
        family_penalty = 2.0 * (
            row["selected"].split("_")[0]
            != catastrophe["selected"].split("_")[0]
        )
        round_penalty = abs(row["round"] - catastrophe["round"]) / 24.0
        return float(np.linalg.norm((value - target) / scale) + family_penalty + round_penalty)
    return min(pool, key=distance)


def main():
    rows = []
    for teacher_seed in TEACHERS:
        for stream_seed in STREAMS:
            row = _diagnose(teacher_seed, stream_seed)
            rows.append(row)
            _print_row(row)

    catastrophes = [row for row in rows if row["is_reported_catastrophe"]]
    positives = [row for row in rows if row["tail_delta_psr"] > 0.0]
    used = set()
    print("\nMATCHED POSITIVES (learner-observable nearest neighbor, same teacher preferred)")
    for catastrophe in catastrophes:
        match = _match(catastrophe, positives, used)
        used.add((match["teacher"], match["stream"]))
        print(
            f"cat t{catastrophe['teacher']}/s{catastrophe['stream']} -> "
            f"pos t{match['teacher']}/s{match['stream']}"
        )
        _print_row(catastrophe)
        _print_row(match)

    print("\nGROUP SUMMARY")
    matched = [
        next(row for row in rows if (row["teacher"], row["stream"]) == key)
        for key in used
    ]
    keys = [
        "paired_mean_ucb", "mean_gate_slack", "worst_ucb_gain",
        "selection_margin", "activation_over_spectral", "coherence_min",
        "coherence_median", "coherence_cv", "train_support_min",
        "audit_support_min", "arrival_retained_tv",
        "arrival_retained_max_log_ratio", "spectral_gap_rel",
        "audit_oracle_purity", "audit_oracle_source_concentration",
        "audit_oracle_nmi", "oracle_worst_source_cell_recall",
        "oracle_worst_source_cell_precision", "tail_delta_q0",
    ]
    for key in keys:
        cat_value = np.mean([row[key] for row in catastrophes])
        pos_value = np.mean([row[key] for row in matched])
        print(f"{key:42s} catastrophe={cat_value:+.5f} matched={pos_value:+.5f}")


if __name__ == "__main__":
    main()
