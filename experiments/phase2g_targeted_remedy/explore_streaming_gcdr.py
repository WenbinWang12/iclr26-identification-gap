"""Streaming group-discovery + stratified-memory remedy (development only).

Unlike ``explore_gcdr.py``, this implementation discovers behavior groups while
the stream is arriving and gives each discovered group its own fixed quota.  It
therefore targets the failure observed in the first locked phase-2g test: a final
clustering pass cannot recover rare examples that buffer selection already lost.

The total raw-sample-plus-centroid float budget is no larger than the phase-2f
16-window buffer.  Candidate generation and audit selection use disjoint halves
of the retained priority reservoirs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import explore_gcdr as E  # noqa: E402


class PriorityReservoir:
    """Mergeable uniform reservoir: retain the smallest independent priorities."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.items = []

    def offer(self, entry):
        if self.capacity <= 0:
            return
        if len(self.items) < self.capacity:
            self.items.append(entry)
            return
        worst = max(range(len(self.items)), key=lambda i: self.items[i]["key"])
        if entry["key"] < self.items[worst]["key"]:
            self.items[worst] = entry

    def rebuild(self, entries):
        self.items = sorted(entries, key=lambda e: e["key"])[:self.capacity]


def _behavior_signature(M, X, Y, activation_weight=0.35, gradient_weight=1.0):
    """Gauge-free ambient gradient plus a frozen-activation fallback."""
    grad = E._gradient_signatures(M, X, Y)
    x = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    activation = np.einsum("ni,nj->nij", x, x).reshape(len(X), -1)
    activation /= np.linalg.norm(activation, axis=1, keepdims=True) + 1e-12
    signature = np.concatenate([gradient_weight * grad,
                                activation_weight * activation], axis=1)
    signature /= np.linalg.norm(signature, axis=1, keepdims=True) + 1e-12
    return signature


def _cluster_with_centers(Z, groups):
    labels = E._farthest_spherical_kmeans(Z, groups, iters=25)
    groups = int(labels.max()) + 1
    centers = np.zeros((groups, Z.shape[1]))
    for g in range(groups):
        c = Z[labels == g].mean(axis=0)
        centers[g] = c / (np.linalg.norm(c) + 1e-12)
    return labels, centers


def _assign(M, entries, centers, activation_weight, gradient_weight):
    if not entries:
        return np.empty(0, int)
    X = np.stack([e["x"] for e in entries])
    Y = np.stack([e["y"] for e in entries])
    Z = _behavior_signature(M, X, Y, activation_weight, gradient_weight)
    return np.argmax(Z @ centers.T, axis=1)


def _xy(entries):
    return (np.stack([e["x"] for e in entries]),
            np.stack([e["y"] for e in entries]))


def _split(entries):
    """Priority order is random; alternate it into disjoint train/audit halves."""
    ordered = sorted(entries, key=lambda e: e["key"])
    return ordered[::2], ordered[1::2]


def _nmse(M, entries):
    if not entries:
        return np.inf
    X, Y = _xy(entries)
    return E._mean_nmse(M, X, Y)


def _group_nmse_entries(M, groups):
    return np.asarray([_nmse(M, entries) for entries in groups])


def _fit_weighted_groups(train_groups, weights, rank, ridge):
    Xs, Ys, ws = [], [], []
    for q, entries in zip(weights, train_groups):
        if not entries:
            continue
        X, Y = _xy(entries)
        Xs.append(X)
        Ys.append(Y)
        ws.append(np.full(len(entries), q / len(entries)))
    return E._empirical_rrr(np.concatenate(Xs), np.concatenate(Ys),
                            np.concatenate(ws), rank, ridge)


def _audit_select(global_train, global_audit, train_grouped, audit_grouped,
                  rank, ridge, rounds, eta, mean_margin):
    paired = [(tr, au) for tr, au in zip(train_grouped, audit_grouped)
              if tr and au]
    train_groups = [x[0] for x in paired]
    audit_groups = [x[1] for x in paired]
    if len(train_groups) < 2 or not global_train or not global_audit:
        fallback = global_train or global_audit
        X, Y = _xy(fallback)
        M = E._empirical_rrr(X, Y, np.ones(len(X)), rank, ridge)
        return M, {"selected_round": 0, "groups": len(train_groups)}

    Xg, Yg = _xy(global_train)
    baseline = E._empirical_rrr(Xg, Yg, np.ones(len(Xg)), rank, ridge)
    baseline_mean = _nmse(baseline, global_audit)
    candidates = [(0, baseline)]
    q = np.asarray([len(x) for x in train_groups], float)
    q /= q.sum()
    previous = baseline
    for step in range(1, rounds + 1):
        loss = _group_nmse_entries(previous, train_groups)
        logits = np.log(q + 1e-300) + eta * loss / (loss.mean() + 1e-12)
        logits -= logits.max()
        q = np.exp(logits)
        q /= q.sum()
        previous = _fit_weighted_groups(train_groups, q, rank, ridge)
        candidates.append((step, previous))

    feasible = []
    for step, M in candidates:
        mean_loss = _nmse(M, global_audit)
        worst = float(_group_nmse_entries(M, audit_groups).max())
        if mean_loss <= baseline_mean + mean_margin:
            feasible.append((worst, mean_loss, step, M))
    chosen = min(feasible, key=lambda z: z[0])
    return chosen[3], {
        "selected_round": chosen[2],
        "audit_mean_nmse": chosen[1],
        "audit_worst_group_nmse": chosen[0],
        "baseline_audit_mean_nmse": baseline_mean,
        "feasible_candidates": len(feasible),
        "groups": len(train_groups),
    }


def run_streaming_gcdr(cfg, batches, seed, *, groups=8, warmup_windows=16,
                       refresh_every=20, rounds=24, eta=0.7,
                       mean_margin=0.01, activation_weight=0.35,
                       gradient_weight=1.0, return_debug=False):
    d = batches[0][0].shape[1]
    raw_budget = (cfg["buf_u"] + cfg["buf_g"]) * cfg["batch"]
    # Two d^2 signatures per center plus one persistent d^2 reference operator
    # are charged against the raw X+Y budget (2d floats per raw example).
    metadata_floats = groups * (2 * d * d) + d * d
    metadata_equiv = int(np.ceil(metadata_floats / (2 * d)))
    usable = raw_budget - metadata_equiv
    global_train_cap = min(128, usable // 8)
    global_audit_cap = global_train_cap
    group_total = usable - global_train_cap - global_audit_cap
    quota = group_total // (2 * groups)

    rng = np.random.default_rng(seed + 901)
    pre_train = PriorityReservoir(groups * quota)
    pre_audit = PriorityReservoir(groups * quota)
    global_train = PriorityReservoir(global_train_cap)
    global_audit = PriorityReservoir(global_audit_cap)
    group_train = [PriorityReservoir(quota) for _ in range(groups)]
    group_audit = [PriorityReservoir(quota) for _ in range(groups)]
    centers = None
    reference = np.zeros((d, d))
    next_id = 0

    def refresh():
        nonlocal centers, reference
        # Centers and reference are functions of TRAIN partitions only.  Audit
        # examples are assigned to those centers but never influence them.
        retained_group_train = [z for r in group_train for z in r.items]
        center_train = global_train.items + retained_group_train
        if len(center_train) < groups:
            return
        ref_entries = global_train.items if len(global_train.items) >= d else center_train
        Xr, Yr = _xy(ref_entries)
        reference = E._empirical_rrr(Xr, Yr, np.ones(len(Xr)), cfg["R"],
                                     cfg.get("ridge", 1e-6))
        X, Y = _xy(center_train)
        Z = _behavior_signature(reference, X, Y, activation_weight, gradient_weight)
        labels, centers = _cluster_with_centers(Z, groups)
        retained_labels = _assign(reference, retained_group_train, centers,
                                  activation_weight, gradient_weight)
        for g, reservoir in enumerate(group_train):
            reservoir.rebuild([e for e, lab in zip(retained_group_train, retained_labels)
                               if lab == g])
        audit_entries = [z for r in group_audit for z in r.items]
        audit_labels = _assign(reference, audit_entries, centers, activation_weight,
                               gradient_weight)
        for g, reservoir in enumerate(group_audit):
            reservoir.rebuild([e for e, lab in zip(audit_entries, audit_labels) if lab == g])

    for window_idx, (X, Y) in enumerate(batches):
        current = []
        for x, y in zip(X, Y):
            role_draw = float(rng.random())
            entry = {"id": next_id, "key": float(rng.random()),
                     "x": x.copy(), "y": y.copy()}
            next_id += 1
            # Roles are permanent and mutually exclusive.  Their probabilities
            # match the reserved slot proportions, so every audit item is held
            # out from reference fitting, clustering, and candidate training.
            p_gt = global_train_cap / usable
            p_ga = global_audit_cap / usable
            p_group_train = (groups * quota) / usable
            if role_draw < p_gt:
                global_train.offer(entry)
            elif role_draw < p_gt + p_ga:
                global_audit.offer(entry)
            elif role_draw < p_gt + p_ga + p_group_train:
                entry["partition"] = "train"
                current.append(entry)
                if centers is None:
                    pre_train.offer(entry)
            else:
                entry["partition"] = "audit"
                current.append(entry)
                if centers is None:
                    pre_audit.offer(entry)

        if centers is None and window_idx + 1 >= warmup_windows:
            initial_train = pre_train.items
            initial_audit = pre_audit.items
            if len(initial_train) < groups:
                continue
            ref_entries = global_train.items if len(global_train.items) >= d else initial_train
            X0r, Y0r = _xy(ref_entries)
            reference = E._empirical_rrr(X0r, Y0r, np.ones(len(X0r)), cfg["R"],
                                         cfg.get("ridge", 1e-6))
            X0, Y0 = _xy(initial_train)
            Z0 = _behavior_signature(reference, X0, Y0, activation_weight,
                                     gradient_weight)
            labels, centers = _cluster_with_centers(Z0, groups)
            for g, reservoir in enumerate(group_train):
                reservoir.rebuild([e for e, lab in zip(initial_train, labels) if lab == g])
            audit_labels = _assign(reference, initial_audit, centers, activation_weight,
                                   gradient_weight)
            for g, reservoir in enumerate(group_audit):
                reservoir.rebuild([e for e, lab in zip(initial_audit, audit_labels) if lab == g])
            pre_train.items = []
            pre_audit.items = []
        elif centers is not None and current:
            labels = _assign(reference, current, centers, activation_weight,
                             gradient_weight)
            for entry, label in zip(current, labels):
                target = group_train if entry["partition"] == "train" else group_audit
                target[int(label)].offer(entry)

        if centers is not None and (window_idx + 1) % refresh_every == 0:
            refresh()

    if centers is None:
        raise RuntimeError("warmup exceeds stream length")
    refresh()
    partitions = [global_train.items, global_audit.items,
                  [z for r in group_train for z in r.items],
                  [z for r in group_audit for z in r.items]]
    id_sets = [{e["id"] for e in part} for part in partitions]
    assert all(id_sets[i].isdisjoint(id_sets[j])
               for i in range(len(id_sets)) for j in range(i + 1, len(id_sets))), \
        "train/audit memory partitions must be mutually exclusive"
    M, stats = _audit_select(global_train.items, global_audit.items,
                             [r.items for r in group_train],
                             [r.items for r in group_audit], cfg["R"],
                             cfg.get("ridge", 1e-6), rounds, eta, mean_margin)
    stats.update({"raw_budget": raw_budget, "usable_raw_samples": usable,
                  "metadata_equiv_samples": metadata_equiv,
                  "global_train_cap": global_train_cap,
                  "global_audit_cap": global_audit_cap,
                  "per_group_partition_quota": quota,
                  "stored_raw_slots": (global_train_cap + global_audit_cap
                                       + 2 * groups * quota)})
    if return_debug:
        stats["debug_group_train_ids"] = [
            [e["id"] for e in r.items] for r in group_train
        ]
        stats["debug_group_audit_ids"] = [
            [e["id"] for e in r.items] for r in group_audit
        ]
    return M, stats


def evaluate(teachers=E.DEV_TEACHERS, streams=E.DEV_STREAMS, **over):
    cfg = E._cfg()
    rows = []
    print(f"{'world':14s} {'stream':>7s} {'psr':>7s} {'cvar':>7s} "
          f"{'meanS':>7s} {'round':>5s}")
    for ts in teachers:
        teacher = E.BM.build_teacher(E.REGIME, cfg, ts)
        for ss in streams:
            windows, freqs = E.BM.make_stream(cfg, ss)
            batches, srcs, _, test = E.BM.materialize(teacher, cfg, windows, ss)
            M_s, stats = run_streaming_gcdr(cfg, batches, E.RUN_SEED, **over)
            M_p = E.P.run_psr_lora(cfg, batches, E.RUN_SEED)[0][-1]
            M_c = E.BL.run_excess_cvar(cfg, batches, E.RUN_SEED)[0][-1]
            ms, ts_ = E._mean_tail(M_s, srcs, freqs, test)
            mp, tp = E._mean_tail(M_p, srcs, freqs, test)
            mc, tc = E._mean_tail(M_c, srcs, freqs, test)
            rows.append((ms, ts_, mp, tp, mc, tc))
            print(f"t{ts % 100000:05d}/s{ss:04d} {ts_:7.3f} {tp:7.3f} {tc:7.3f} "
                  f"{ms:7.3f} {stats['selected_round']:5d}")
    a = np.asarray(rows)
    print("\nstreaming GCDR mean/tail %.4f / %.4f" % (a[:, 0].mean(), a[:, 1].mean()))
    print("PSR            mean/tail %.4f / %.4f" % (a[:, 2].mean(), a[:, 3].mean()))
    print("CVaR           mean/tail %.4f / %.4f" % (a[:, 4].mean(), a[:, 5].mean()))
    print("streaming - PSR tail %+0.4f, wins %d/%d" %
          ((a[:, 1] - a[:, 3]).mean(), np.sum(a[:, 1] > a[:, 3]), len(a)))
    return rows


if __name__ == "__main__":
    evaluate()
