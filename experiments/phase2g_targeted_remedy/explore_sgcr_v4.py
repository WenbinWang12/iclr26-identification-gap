"""DEVELOPMENT ONLY: shared-window SGCR with a spectral Pareto path.

This module does not modify or import any confirmatory seed protocol.  It keeps
one bounded uniform+input-diverse window pool, gives every retained row a
deterministic train/audit role, and generates all candidates from the train
view.  The audit view is used only for model selection.

Relative to the first shared-pool prototype, the window-spectral arm is not a
single train-worst endpoint.  We retain round 0 through round 12 of the
multiplicative-weights trajectory and fixed-grid interpolants from the uniform
window weights q0 to every trajectory weight.  All operators are fresh rank-R
spectral solves under one fixed pooled teacher.

Selection has two label-free, audit-only safeguards:

1. q0 is an unconditional feasible fallback;
2. a challenger must satisfy a paired example-level upper confidence bound on
   mean-NMSE degradation relative to q0 before competing on worst-cell UCB.

The window pool is the only persistent raw state.  Codebooks, covariances,
candidate operators, and audit statistics are transient fit workspace.  This
is a development implementation, not a population confidence guarantee.
"""

from __future__ import annotations

import numpy as np

import explore_gcdr as E
import explore_sgcr_v2 as V
import explore_sgcr_v3 as W


def _window_pool_groups_aligned(items, seed, train_fraction, groups,
                                old_centers):
    """Build transient row views while preserving persistent cell identity."""
    train_entries, audit_entries, train_windows = [], [], []
    for slot, item in enumerate(items):
        train_rows, audit_rows = W._split_window_rows(
            item, seed, train_fraction
        )
        train_windows.append({
            "X": item["X"][train_rows],
            "Y": item["Y"][train_rows],
        })
        for rows, destination in (
            (train_rows, train_entries), (audit_rows, audit_entries)
        ):
            for row in rows:
                destination.append({
                    "x": item["X"][row],
                    "y": item["Y"][row],
                    "id": (int(item["window_id"]), int(row)),
                    "slot": int(slot),
                })

    train_x, _ = V._xy(train_entries)
    raw_labels, raw_centers = V._cluster(
        V._activation_signature(train_x), groups
    )
    if old_centers is None:
        centers = raw_centers
        train_labels = raw_labels
    else:
        centers, new_to_old = V._align_centers(old_centers, raw_centers)
        train_labels = new_to_old[raw_labels]
    audit_labels = V._assign(audit_entries, centers)
    train_grouped = [[] for _ in range(groups)]
    audit_grouped = [[] for _ in range(groups)]
    for entry, label in zip(train_entries, train_labels):
        train_grouped[int(label)].append(entry)
    for entry, label in zip(audit_entries, audit_labels):
        audit_grouped[int(label)].append(entry)
    return train_grouped, audit_grouped, train_windows, centers


def _unique_grouped(grouped):
    """Collapse a U/G duplicate slot for audit uncertainty calculations."""
    output = []
    for entries in grouped:
        seen = set()
        unique = []
        for entry in entries:
            identifier = entry.get("id")
            if identifier in seen:
                continue
            seen.add(identifier)
            unique.append(entry)
        output.append(unique)
    return output


def _loss_samples(model, entries):
    """Per-example loss normalized so its mean equals cell NMSE."""
    X, Y = V._xy(entries)
    residual = np.sum((X @ model.T - Y) ** 2, axis=1)
    target = np.sum(Y ** 2, axis=1)
    return residual / (float(target.mean()) + 1e-12)


def _candidate_audit(model, baseline, audit_grouped, q0, active, z_value):
    losses, errors = [], []
    paired_means, paired_errors = [], []
    for entries in audit_grouped:
        candidate_loss = _loss_samples(model, entries)
        baseline_loss = _loss_samples(baseline, entries)
        difference = candidate_loss - baseline_loss
        losses.append(float(candidate_loss.mean()))
        errors.append(
            float(candidate_loss.std(ddof=1) / np.sqrt(len(candidate_loss)))
            if len(candidate_loss) > 1 else np.inf
        )
        paired_means.append(float(difference.mean()))
        paired_errors.append(
            float(difference.std(ddof=1) / np.sqrt(len(difference)))
            if len(difference) > 1 else np.inf
        )
    losses = np.asarray(losses)
    errors = np.asarray(errors)
    paired_means = np.asarray(paired_means)
    paired_errors = np.asarray(paired_errors)
    mean = float(q0 @ losses)
    mean_delta = float(q0 @ paired_means)
    # Cell reservoirs are disjoint.  This plug-in SE combines their paired
    # estimates; it is a stability penalty, not a formal finite-population CI.
    mean_delta_se = float(np.sqrt(np.sum((q0 * paired_errors) ** 2)))
    mean_delta_ucb = mean_delta + float(z_value) * mean_delta_se
    worst = float(losses[active].max())
    worst_ucb = float(
        np.max(losses[active] + float(z_value) * errors[active])
    )
    return {
        "mean": mean,
        "mean_delta": mean_delta,
        "mean_delta_se": mean_delta_se,
        "mean_delta_ucb": mean_delta_ucb,
        "worst": worst,
        "worst_ucb": worst_ucb,
    }


def _spectral_pareto_path(train_windows, d, rank, ridge, rounds, eta,
                          rho_grid):
    """Return q0, MW rounds 0..rounds, and q0-to-round interpolants."""
    teacher = E.P._pooled_teacher(train_windows, d, ridge)
    covariances = [E.P.H.second_moment(item["X"]) for item in train_windows]
    count = len(train_windows)
    q0 = np.full(count, 1.0 / count)
    weights = q0.copy()
    candidates = []
    seen = set()

    def append(name, candidate_weights):
        normalized = np.asarray(candidate_weights, dtype=float)
        normalized /= normalized.sum()
        key = np.round(normalized, 12).tobytes()
        if key in seen:
            return
        seen.add(key)
        candidates.append((
            name,
            E.P.H.weighted_rrr(
                teacher, covariances, normalized, rank
            ),
        ))

    append("window_spectral_q0", q0)
    for step in range(int(rounds) + 1):
        for rho in rho_grid:
            blended = (1.0 - float(rho)) * q0 + float(rho) * weights
            append(
                f"window_spectral_r{step:02d}_rho{float(rho):.2f}",
                blended,
            )
        endpoint = E.P.H.weighted_rrr(
            teacher, covariances, weights, rank
        )
        if step == int(rounds):
            break
        train_loss = np.asarray([
            E.P._loss(endpoint, item["X"], item["Y"])
            for item in train_windows
        ])
        scaled = train_loss / (train_loss.mean() + 1e-12)
        logits = np.log(weights + 1e-300) + float(eta) * scaled
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
    return candidates


def _activation_train_path(train_grouped, q0, active, rank, ridge, rounds,
                           eta, blend_grid):
    """Generate the activation-cell path using TRAIN losses only.

    Candidate generation never receives the audit partition.  The outer
    selector is consequently the sole reader of audit X/Y.
    """
    q0 = np.asarray(q0, dtype=float)
    q0 /= q0.sum()
    active = np.asarray(active, dtype=int)
    active_mass = float(q0[active].sum())
    candidates = []

    uniform_active = np.full(len(active), active_mass / len(active))
    for rho in blend_grid:
        weights = q0.copy()
        weights[active] = (
            (1.0 - float(rho)) * q0[active]
            + float(rho) * uniform_active
        )
        candidates.append((
            f"activation_uniform_rho{float(rho):.2f}",
            V._fit_groups(train_grouped, weights, rank, ridge),
        ))

    dual = q0[active] / active_mass
    previous = V._fit_groups(train_grouped, q0, rank, ridge)
    for step in range(1, int(rounds) + 1):
        train_loss = V._group_nmse(previous, train_grouped)[active]
        scaled = train_loss / (train_loss.mean() + 1e-12)
        logits = np.log(dual + 1e-300) + float(eta) * scaled
        logits -= logits.max()
        dual = np.exp(logits)
        dual /= dual.sum()
        weights = q0.copy()
        weights[active] = active_mass * dual
        previous = V._fit_groups(train_grouped, weights, rank, ridge)
        candidates.append((
            f"activation_hedge_{step:02d}", previous
        ))
    return candidates


def _select(candidates, baseline, audit_grouped, q0, active, mean_margin,
            z_value):
    """Paired mean-UCB hard constraint, then minimum worst-cell UCB."""
    rows = []
    for name, model in candidates:
        audit = _candidate_audit(
            model, baseline, audit_grouped, q0, active, z_value
        )
        # q0 remains feasible even under numerical noise or singleton cells.
        feasible = bool(
            name == "frequency_baseline"
            or audit["mean_delta_ucb"] <= float(mean_margin)
        )
        rows.append({"name": name, "model": model,
                     "feasible": feasible, **audit})
    feasible_rows = [row for row in rows if row["feasible"]]
    assert feasible_rows, "unconditional q0 fallback was lost"
    selected = min(
        feasible_rows,
        key=lambda row: (row["worst_ucb"], row["mean_delta_ucb"], row["name"]),
    )
    summary = [
        {key: row[key] for key in (
            "name", "feasible", "mean", "mean_delta",
            "mean_delta_ucb", "worst", "worst_ucb"
        )}
        for row in rows
    ]
    return selected["model"], {
        "selected_candidate": selected["name"],
        "selected_audit_mean_nmse": selected["mean"],
        "selected_audit_mean_delta": selected["mean_delta"],
        "selected_audit_mean_delta_ucb": selected["mean_delta_ucb"],
        "selected_audit_worst_nmse": selected["worst"],
        "selected_audit_worst_ucb": selected["worst_ucb"],
        "feasible_candidates": int(sum(row["feasible"] for row in rows)),
        "candidate_count": len(rows),
        "candidate_audit": summary,
    }


def run_sgcr_v4(cfg, batches, seed, *, uniform_windows=3,
                diverse_windows=11, train_fraction=0.50,
                refresh_every=20, groups=None,
                activation_rounds=24, activation_eta=0.1,
                spectral_rounds=12, spectral_eta=1.0,
                rho_grid=(0.25, 0.50, 0.75, 1.00),
                mean_margin=0.01, coherence_ratio=0.60,
                ucb_z=1.645):
    """Run the fixed-budget shared-window spectral Pareto hybrid."""
    d = batches[0][0].shape[1]
    groups = int(2 * cfg["R"] if groups is None else groups)
    total_slots = int(uniform_windows + diverse_windows)
    numeric_budget = (
        (cfg["buf_u"] + cfg["buf_g"]) * cfg["batch"] * 2 * d
    )
    persistent_numeric_floats = (
        total_slots * cfg["batch"] * 2 * d
        + int(diverse_windows) * d + total_slots + 2
        + groups * d * d + groups
    )
    if persistent_numeric_floats > numeric_budget:
        raise ValueError("shared window pool exceeds inherited numeric budget")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between zero and one")

    rng = np.random.default_rng(seed + 41)
    uniform = E.P.H.Reservoir(int(uniform_windows), rng)
    diverse = []
    current_model = np.zeros((d, d))
    last_stats = None
    fit_count = 0
    centers = None
    arrival_counts = None

    for window_index, (X, Y) in enumerate(batches):
        entry = {
            "X": X.copy(), "Y": Y.copy(),
            "window_id": int(window_index),
        }
        uniform.offer(entry)
        diverse_entry = dict(entry)
        diverse_entry["idir"] = E.P._indir(X)
        E.P._div_offer(diverse, diverse_entry, int(diverse_windows))

        # Once the warmup codebook exists, count each actual TRAIN arrival once.
        # This state, unlike the deliberately rebalanced retained pool, estimates
        # stream frequency and therefore defines the audit mean constraint.
        if centers is not None:
            train_rows, _ = W._split_window_rows(
                entry, seed, train_fraction
            )
            incoming = [
                {"x": X[row], "y": Y[row]}
                for row in train_rows
            ]
            incoming_labels = V._assign(incoming, centers)
            arrival_counts += np.bincount(
                incoming_labels, minlength=groups
            )

        should_fit = (
            (window_index + 1) % int(refresh_every) == 0
            or window_index + 1 == len(batches)
        )
        if not should_fit or len(uniform.items) + len(diverse) < groups:
            continue

        items = list(uniform.items) + list(diverse)
        train_grouped, audit_grouped, train_windows, centers = (
            _window_pool_groups_aligned(
                items, seed, train_fraction, groups, centers
            )
        )
        if arrival_counts is None:
            # The first codebook is learned after warmup; past non-retained rows
            # are unavailable by design.  Initialize from its retained train view,
            # then count every subsequent stream arrival exactly once.
            arrival_counts = np.asarray(
                [len(x) for x in train_grouped], dtype=float
            )
        # Duplicate U/G slots remain legitimate training weights, but must not
        # masquerade as independent observations in the uncertainty gate.
        audit_unique = _unique_grouped(audit_grouped)
        if any(not entries for entries in audit_unique):
            q0 = arrival_counts.copy()
            q0 /= q0.sum()
            current_model = V._fit_groups(
                train_grouped, q0, cfg["R"], cfg.get("ridge", 1e-6)
            )
            last_stats = {
                "selected_candidate": "frequency_baseline_missing_audit_support",
                "feasible_candidates": 1,
                "candidate_count": 1,
            }
            fit_count += 1
            continue

        q0 = arrival_counts.copy()
        q0 /= q0.sum()
        coherence = []
        for group, entries in enumerate(train_grouped):
            group_x, _ = V._xy(entries)
            similarity = V._activation_signature(group_x) @ centers[group]
            coherence.append(float(similarity.mean()))
        coherence = np.asarray(coherence)
        active = np.flatnonzero(
            coherence >= float(coherence_ratio) * np.median(coherence)
        )
        if len(active) < 2:
            active = np.arange(groups)
        baseline = V._fit_groups(
            train_grouped, q0, cfg["R"], cfg.get("ridge", 1e-6)
        )
        activation = _activation_train_path(
            train_grouped, q0, active, cfg["R"],
            cfg.get("ridge", 1e-6), int(activation_rounds),
            float(activation_eta),
            (0.10, 0.20, 0.35, 0.50, 0.70, 1.00),
        )
        spectral = _spectral_pareto_path(
            train_windows, d, cfg["R"], cfg.get("ridge", 1e-6),
            int(spectral_rounds), float(spectral_eta), tuple(rho_grid),
        )
        candidates = [
            ("frequency_baseline", baseline),
            *activation,
            *spectral,
        ]
        current_model, selection_stats = _select(
            candidates, baseline, audit_unique, q0, active,
            float(mean_margin), float(ucb_z),
        )
        last_stats = dict(selection_stats)
        last_stats.update({
            "window": window_index + 1,
            "activation_candidate_count": len(activation),
            "cell_coherence": coherence.tolist(),
            "train_group_sizes": [len(x) for x in train_grouped],
            "audit_group_sizes": [len(x) for x in audit_unique],
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
        "numeric_budget_floats": int(numeric_budget),
        "persistent_numeric_floats": int(persistent_numeric_floats),
        "extra_persistent_raw_entries": 0,
        "ucb_z": float(ucb_z),
        "mean_margin": float(mean_margin),
        "spectral_rounds": int(spectral_rounds),
        "rho_grid": tuple(float(x) for x in rho_grid),
        "arrival_frequencies": (arrival_counts / arrival_counts.sum()).tolist(),
        "persistent_note": (
            "one shared bounded window pool plus aligned activation centers and "
            "train-only arrival counts; all Pareto candidates transient"
        ),
    })
    return current_model, last_stats
