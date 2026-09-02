"""DEVELOPMENT ONLY: same-memory spectral hybrid for SGCR-v2.

This file deliberately leaves ``explore_sgcr_v2.py`` and its locked
confirmatory script untouched.  It reuses the v2 streaming stores and adds only
transient candidate computations at each fit:

* the existing empirical-RRR path, which re-estimates the full-rank teacher for
  every cell-weight vector; and
* a fixed-teacher spectral path, which estimates one pooled teacher at the
  arrival-frequency weights and changes only the rank-R spectral allocation.

Both paths use exactly the same train coreset and the same disjoint audit
coreset.  No additional raw example is retained.  ``confidence_z`` optionally
scores candidates by the worst cell's empirical upper-confidence loss; this is
an exploratory stability gate, not a population guarantee.

The wrapper temporarily replaces v2's internal selector while one run executes.
It is intentionally single-threaded research scaffolding; a production version
should pass the selector explicitly instead of monkey-patching the module.
"""

from __future__ import annotations

import numpy as np

import explore_gcdr as E
import explore_sgcr_v2 as V


_V2_SELECTOR = V._select_same_pool


def _pooled_teacher(grouped, weights, ridge):
    """Full-rank weighted LS teacher, estimated once for the spectral path."""
    xs, ys, sample_weights = [], [], []
    for weight, entries in zip(weights, grouped):
        X, Y = V._xy(entries)
        xs.append(X)
        ys.append(Y)
        sample_weights.append(np.full(len(entries), weight / len(entries)))
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    w = np.concatenate(sample_weights)
    w /= w.sum()
    cxx = X.T @ (w[:, None] * X) + ridge * np.eye(X.shape[1])
    cyx = Y.T @ (w[:, None] * X)
    return np.linalg.solve(cxx, cyx.T).T


def _fixed_teacher_rrr(teacher, covariances, weights, rank, ridge):
    """Rank allocation under weighted input geometry with a fixed teacher."""
    covariance = sum(
        float(weight) * sigma
        for weight, sigma in zip(weights, covariances)
    ) + ridge * np.eye(teacher.shape[1])
    root, inv_root = E._sqrt_and_inv_sqrt(covariance)
    whitened = teacher @ root
    left, singular, right = np.linalg.svd(whitened, full_matrices=False)
    truncated = (
        left[:, :rank] * singular[:rank][None, :]
    ) @ right[:rank]
    return truncated @ inv_root


def _group_nmse_and_se(model, grouped):
    """Cell NMSE and an example-level plug-in standard error."""
    means, errors = [], []
    for entries in grouped:
        X, Y = V._xy(entries)
        squared_error = np.sum((X @ model.T - Y) ** 2, axis=1)
        target_energy = np.sum(Y ** 2, axis=1)
        scale = float(target_energy.mean()) + 1e-12
        normalized = squared_error / scale
        means.append(float(normalized.mean()))
        if len(normalized) > 1:
            errors.append(float(normalized.std(ddof=1) / np.sqrt(len(normalized))))
        else:
            errors.append(np.inf)
    return np.asarray(means), np.asarray(errors)


def _select_hybrid(train_grouped, audit_grouped, frequencies, robust_mask,
                   rank, ridge, rounds, eta, mean_margin, min_audit_gain,
                   blend_grid, *, confidence_z, allowed_families):
    """Generate empirical and fixed-teacher candidates from one train pool."""
    q0 = np.asarray(frequencies, dtype=float)
    q0 /= q0.sum()
    if any(not entries for entries in train_grouped):
        raise RuntimeError("refresh must keep every train cell non-empty")
    if any(not entries for entries in audit_grouped):
        # Preserve v2's conservative degenerate-stream behavior.
        return _V2_SELECTOR(
            train_grouped, audit_grouped, frequencies, robust_mask,
            rank, ridge, rounds, eta, mean_margin, min_audit_gain, blend_grid
        )

    robust_mask = np.asarray(robust_mask, dtype=bool)
    active = np.flatnonzero(robust_mask)
    if len(active) < 2:
        robust_mask[:] = True
        active = np.flatnonzero(robust_mask)
    active_mass = float(q0[active].sum())

    baseline = V._fit_groups(train_grouped, q0, rank, ridge)
    baseline_loss, baseline_se = _group_nmse_and_se(baseline, audit_grouped)
    baseline_mean = float(q0 @ baseline_loss)
    baseline_worst = float(baseline_loss[active].max())
    baseline_score = float(
        np.max(baseline_loss[active] + confidence_z * baseline_se[active])
    )

    fixed_teacher = _pooled_teacher(train_grouped, q0, ridge)
    covariances = []
    for entries in train_grouped:
        X, _ = V._xy(entries)
        covariances.append((X.T @ X) / len(X))

    best = baseline
    best_weights = q0.copy()
    best_round = 0
    best_candidate = "frequency_baseline"
    best_mean = baseline_mean
    best_worst = baseline_worst
    best_score = baseline_score
    feasible = 1
    feasible_by_family = {"baseline": 1, "empirical": 0, "spectral": 0}

    def spectral(weights):
        return _fixed_teacher_rrr(
            fixed_teacher, covariances, weights, rank, ridge
        )

    def consider(candidate, candidate_weights, candidate_name,
                 candidate_round, family):
        nonlocal best, best_weights, best_candidate, best_round
        nonlocal best_mean, best_worst, best_score, feasible
        if family not in allowed_families:
            return
        audit_loss, audit_se = _group_nmse_and_se(candidate, audit_grouped)
        candidate_mean = float(q0 @ audit_loss)
        candidate_worst = float(audit_loss[active].max())
        candidate_score = float(
            np.max(audit_loss[active] + confidence_z * audit_se[active])
        )
        if candidate_mean <= baseline_mean + mean_margin:
            feasible += 1
            feasible_by_family[family] += 1
            if candidate_score < best_score - min_audit_gain:
                best = candidate
                best_weights = np.asarray(candidate_weights).copy()
                best_candidate = candidate_name
                best_round = candidate_round
                best_mean = candidate_mean
                best_worst = candidate_worst
                best_score = candidate_score

    uniform_active = np.full(len(active), active_mass / len(active))
    for index, rho in enumerate(blend_grid, start=1):
        weights = q0.copy()
        weights[active] = (
            (1.0 - float(rho)) * q0[active]
            + float(rho) * uniform_active
        )
        consider(
            V._fit_groups(train_grouped, weights, rank, ridge), weights,
            f"empirical_frequency_uniform_{float(rho):.3f}", -index,
            "empirical"
        )
        consider(
            spectral(weights), weights,
            f"spectral_frequency_uniform_{float(rho):.3f}", -index,
            "spectral"
        )

    # Keep v2's empirical Hedge path and evaluate the fixed-teacher operator at
    # each identical weight vector.  This isolates mapping re-estimation from
    # rank reallocation without adding a separately tuned dual trajectory.
    dual = q0[active] / active_mass
    previous = baseline
    for step in range(1, rounds + 1):
        train_loss = V._group_nmse(previous, train_grouped)[active]
        scaled = train_loss / (train_loss.mean() + 1e-12)
        logits = np.log(dual + 1e-300) + eta * scaled
        logits -= logits.max()
        dual = np.exp(logits)
        dual /= dual.sum()
        weights = q0.copy()
        weights[active] = active_mass * dual
        empirical = V._fit_groups(train_grouped, weights, rank, ridge)
        consider(
            empirical, weights, f"empirical_hedge_{step}", step,
            "empirical"
        )
        consider(
            spectral(weights), weights, f"spectral_hedge_{step}", step,
            "spectral"
        )
        previous = empirical

    return best, {
        "selected_round": best_round,
        "selected_candidate": best_candidate,
        "selected_family": (
            "spectral" if best_candidate.startswith("spectral_")
            else "empirical" if best_candidate.startswith("empirical_")
            else "baseline"
        ),
        "confidence_z": float(confidence_z),
        "baseline_audit_mean_nmse": baseline_mean,
        "baseline_audit_worst_nmse": baseline_worst,
        "baseline_audit_worst_ucb": baseline_score,
        "selected_audit_mean_nmse": best_mean,
        "selected_audit_worst_nmse": best_worst,
        "selected_audit_worst_ucb": best_score,
        "feasible_candidates": feasible,
        "feasible_by_family": feasible_by_family,
        "robust_groups": active.tolist(),
        "selected_weights": best_weights.tolist(),
    }


def run_sgcr_v3(cfg, batches, seed, *, confidence_z=0.0,
                allowed_families=("empirical", "spectral"), **kwargs):
    """Run the v2 stream with the development-only hybrid selector."""
    original = V._select_same_pool

    def selector(*args, **selector_kwargs):
        return _select_hybrid(
            *args, **selector_kwargs, confidence_z=float(confidence_z),
            allowed_families=tuple(allowed_families)
        )

    V._select_same_pool = selector
    try:
        model, stats = V.run_sgcr_v2(cfg, batches, seed, **kwargs)
    finally:
        V._select_same_pool = original
    stats["development_only"] = True
    stats["v3_extra_persistent_raw_entries"] = 0
    stats["v3_extra_state_note"] = (
        "fixed teacher/covariances/candidates are transient fit workspace"
    )
    return model, stats


def _split_window_rows(entry, seed, train_fraction):
    """Deterministic permanent row roles for a retained window slot."""
    count = len(entry["X"])
    # The split is a pure function of learner seed and stream window id.  A
    # duplicated U/G slot therefore cannot put the same row in both roles.
    rng = np.random.default_rng(
        (int(seed) * 1_000_003 + int(entry["window_id"]) * 97_409 + 31)
        % (2 ** 32)
    )
    order = rng.permutation(count)
    cut = min(count - 1, max(1, int(np.floor(count * train_fraction))))
    return order[:cut], order[cut:]


def _window_pool_groups(items, seed, train_fraction, groups):
    """Transient activation cells over row-disjoint views of one window pool."""
    train_entries, audit_entries, train_windows = [], [], []
    for slot, item in enumerate(items):
        train_rows, audit_rows = _split_window_rows(item, seed, train_fraction)
        train_windows.append({
            "X": item["X"][train_rows],
            "Y": item["Y"][train_rows],
        })
        for role_rows, destination in (
            (train_rows, train_entries), (audit_rows, audit_entries)
        ):
            for row in role_rows:
                destination.append({
                    "x": item["X"][row],
                    "y": item["Y"][row],
                    "id": (int(item["window_id"]), int(row)),
                    "slot": int(slot),
                })

    train_x, _ = V._xy(train_entries)
    train_labels, centers = V._cluster(
        V._activation_signature(train_x), groups
    )
    audit_labels = V._assign(audit_entries, centers)
    train_grouped = [[] for _ in range(groups)]
    audit_grouped = [[] for _ in range(groups)]
    for entry, label in zip(train_entries, train_labels):
        train_grouped[int(label)].append(entry)
    for entry, label in zip(audit_entries, audit_labels):
        audit_grouped[int(label)].append(entry)
    return train_grouped, audit_grouped, train_windows, centers


def _choose_shared_pool_candidate(candidates, audit_grouped, q0, active,
                                  mean_margin, confidence_z, allowed_candidates):
    """Audit-select q0/SGCR/spectral-window candidates with a mean guard."""
    baseline = candidates[0]
    base_loss, base_se = _group_nmse_and_se(baseline[1], audit_grouped)
    base_mean = float(q0 @ base_loss)
    rows = []
    for name, model in candidates:
        loss, error = _group_nmse_and_se(model, audit_grouped)
        mean = float(q0 @ loss)
        worst = float(loss[active].max())
        score = float(np.max(
            loss[active] + confidence_z * error[active]
        ))
        rows.append({
            "name": name, "model": model, "mean": mean,
            "worst": worst, "score": score,
            "feasible": bool(mean <= base_mean + mean_margin),
        })
    feasible = [
        row for row in rows
        if row["feasible"] and row["name"] in allowed_candidates
    ]
    selected = min(feasible, key=lambda row: (row["score"], row["name"]))
    stats = {
        "selected_candidate": selected["name"],
        "selected_audit_mean_nmse": selected["mean"],
        "selected_audit_worst_nmse": selected["worst"],
        "selected_audit_worst_ucb": selected["score"],
        "baseline_audit_mean_nmse": base_mean,
        "baseline_audit_worst_nmse": float(base_loss[active].max()),
        "candidate_audit": [
            {key: row[key] for key in (
                "name", "mean", "worst", "score", "feasible"
            )}
            for row in rows
        ],
    }
    return selected["model"], stats


def run_sgcr_v3_shared(cfg, batches, seed, *, uniform_windows=3,
                       diverse_windows=11, train_fraction=0.50,
                       refresh_every=20, groups=None, rounds=24, eta=0.1,
                       mean_margin=0.01, coherence_ratio=0.60,
                       confidence_z=0.0,
                       allowed_candidates=("frequency_baseline",
                                           "activation_sgcr",
                                           "window_spectral")):
    """DEVELOPMENT shared-window hybrid; zero extra persistent raw examples.

    One 14-window pool is maintained by PSR's uniform+input-diverse rule.  Its
    rows have deterministic train/audit roles.  The train view produces both an
    activation-cell SGCR candidate and a PSR-style window-spectral candidate;
    the audit view selects between them and the frequency baseline.
    """
    d = batches[0][0].shape[1]
    groups = int(2 * cfg["R"] if groups is None else groups)
    total_slots = int(uniform_windows + diverse_windows)
    if total_slots < 2:
        raise ValueError("shared pool needs at least two window slots")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")

    numeric_budget = (
        (cfg["buf_u"] + cfg["buf_g"]) * cfg["batch"] * 2 * d
    )
    # Logical persistent state: raw X/Y slots, diverse input directions, one id
    # per slot, and two reservoir/clock counters.  Cell signatures and all model
    # candidates are transient fit workspace.
    persistent_numeric_floats = (
        total_slots * cfg["batch"] * 2 * d
        + diverse_windows * d + total_slots + 2
    )
    if persistent_numeric_floats > numeric_budget:
        raise ValueError("shared window pool exceeds the inherited numeric budget")

    rng = np.random.default_rng(seed + 41)
    uniform = E.P.H.Reservoir(int(uniform_windows), rng)
    diverse = []
    current_model = np.zeros((d, d))
    last_stats = None
    fit_count = 0

    for window_index, (X, Y) in enumerate(batches):
        base = {
            "X": X.copy(), "Y": Y.copy(),
            "window_id": int(window_index),
        }
        uniform.offer(base)
        diverse_entry = dict(base)
        diverse_entry["idir"] = E.P._indir(X)
        E.P._div_offer(diverse, diverse_entry, int(diverse_windows))

        should_fit = (
            (window_index + 1) % refresh_every == 0
            or window_index + 1 == len(batches)
        )
        if not should_fit or len(uniform.items) + len(diverse) < groups:
            continue

        items = list(uniform.items) + list(diverse)
        train_grouped, audit_grouped, train_windows, centers = (
            _window_pool_groups(items, seed, train_fraction, groups)
        )
        if any(not entries for entries in audit_grouped):
            # Extremely small/degenerate audit support: deterministic baseline.
            q0 = np.asarray([len(x) for x in train_grouped], dtype=float)
            q0 /= q0.sum()
            current_model = V._fit_groups(
                train_grouped, q0, cfg["R"], cfg.get("ridge", 1e-6)
            )
            last_stats = {
                "selected_candidate": "frequency_baseline_missing_audit_support",
                "missing_audit_groups": [
                    i for i, entries in enumerate(audit_grouped) if not entries
                ],
            }
            fit_count += 1
            continue

        q0 = np.asarray([len(x) for x in train_grouped], dtype=float)
        q0 /= q0.sum()
        coherence = []
        for group, entries in enumerate(train_grouped):
            group_x, _ = V._xy(entries)
            similarity = V._activation_signature(group_x) @ centers[group]
            coherence.append(float(similarity.mean()))
        coherence = np.asarray(coherence)
        active = np.flatnonzero(
            coherence >= coherence_ratio * np.median(coherence)
        )
        if len(active) < 2:
            active = np.arange(groups)

        q0_model = V._fit_groups(
            train_grouped, q0, cfg["R"], cfg.get("ridge", 1e-6)
        )
        sgcr_model, sgcr_stats = _V2_SELECTOR(
            train_grouped, audit_grouped, q0, np.isin(np.arange(groups), active),
            cfg["R"], cfg.get("ridge", 1e-6), rounds, eta, mean_margin,
            0.0, (0.10, 0.20, 0.35, 0.50, 0.70, 1.00)
        )
        spectral_model, spectral_round = E.P._minimax_rrr(
            train_windows, d, cfg["R"],
            int(E.P.DEFAULT["rounds"]), float(E.P.DEFAULT["mw_eta"]),
            True, cfg.get("ridge", 1e-6)
        )
        current_model, choose_stats = _choose_shared_pool_candidate(
            [
                ("frequency_baseline", q0_model),
                ("activation_sgcr", sgcr_model),
                ("window_spectral", spectral_model),
            ],
            audit_grouped, q0, active, mean_margin, confidence_z,
            tuple(allowed_candidates)
        )
        last_stats = dict(choose_stats)
        last_stats.update({
            "window": window_index + 1,
            "sgcr_inner_candidate": sgcr_stats["selected_candidate"],
            "spectral_train_round": int(spectral_round),
            "cell_coherence": coherence.tolist(),
            "train_group_sizes": [len(x) for x in train_grouped],
            "audit_group_sizes": [len(x) for x in audit_grouped],
        })
        fit_count += 1

    if last_stats is None:
        raise RuntimeError("stream never populated enough shared slots to fit")
    last_stats.update({
        "development_only": True,
        "fit_count": int(fit_count),
        "window_slots": int(total_slots),
        "uniform_windows": int(uniform_windows),
        "diverse_windows": int(diverse_windows),
        "train_fraction": float(train_fraction),
        "numeric_budget_floats": int(numeric_budget),
        "persistent_numeric_floats": int(persistent_numeric_floats),
        "v3_extra_persistent_raw_entries": 0,
        "persistent_note": (
            "one shared bounded window pool; clusters and candidates transient"
        ),
    })
    return current_model, last_stats
