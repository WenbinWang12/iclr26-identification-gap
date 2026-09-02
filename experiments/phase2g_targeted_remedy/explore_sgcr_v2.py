"""Failure-driven SGCR v2 prototype (development only).

This version repairs the main validity defects found in the first streaming
prototype:

* frozen projective-activation signatures replace noisy per-example residual
  gradients as the primary pseudo-context signal;
* ``K_max = 2R`` is a resolution budget, not the planted source count;
* each train/audit store has a per-cell minimum plus a shared overflow, so empty
  quota cannot waste memory and frequent cells retain most remaining capacity;
* the frequency baseline and robust candidates use exactly the same training
  coreset; only their cell weights differ;
* train/audit roles are permanent and disjoint;
* the model is updated at every refresh rather than only after the full stream;
* historical numeric state (keys, ids, labels, counters, and centers included)
  is explicitly charged against the phase-2f raw X+Y float budget.

The code is still a controlled linear-RRR mechanism test, not a Transformer
LoRA result.  Its audit buffer is reused across refreshes, so the gate provides
empirical buffer safety, not a reusable-holdout population theorem.
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
    """Uniform priority reservoir used only before the codebook is initialized."""

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


class MinPlusOverflowStore:
    """Per-cell minimum coverage with one globally shared priority overflow."""

    def __init__(self, capacity, groups, minimum):
        self.capacity = int(capacity)
        self.groups = int(groups)
        self.minimum = int(minimum)
        if self.groups * self.minimum > self.capacity:
            raise ValueError("minimum group quotas exceed store capacity")
        self.base = [[] for _ in range(self.groups)]
        self.overflow = []

    @property
    def overflow_capacity(self):
        # Unused minimum quota is shared, rather than becoming dead capacity.
        return self.capacity - sum(len(cell) for cell in self.base)

    def _trim_overflow(self):
        if len(self.overflow) > self.overflow_capacity:
            self.overflow.sort(key=lambda entry: entry["key"])
            self.overflow = self.overflow[:self.overflow_capacity]

    @staticmethod
    def _replace_by_priority(items, entry, capacity):
        if capacity <= 0:
            return
        if len(items) < capacity:
            items.append(entry)
            return
        worst = max(range(len(items)), key=lambda i: items[i]["key"])
        if entry["key"] < items[worst]["key"]:
            items[worst] = entry

    def offer(self, entry, label):
        label = int(label)
        entry["label"] = label
        cell = self.base[label]
        if len(cell) < self.minimum:
            cell.append(entry)
            # Growing a previously underfull base consumes one shared slot.
            self._trim_overflow()
            return
        worst = max(range(len(cell)), key=lambda i: cell[i]["key"])
        if entry["key"] < cell[worst]["key"]:
            displaced = cell[worst]
            cell[worst] = entry
            self._replace_by_priority(
                self.overflow, displaced, self.overflow_capacity
            )
        else:
            self._replace_by_priority(
                self.overflow, entry, self.overflow_capacity
            )

    @property
    def items(self):
        return [entry for cell in self.base for entry in cell] + self.overflow

    def grouped(self):
        out = [list(cell) for cell in self.base]
        for entry in self.overflow:
            out[int(entry["label"])].append(entry)
        return out

    def rebuild(self, entries, labels):
        buckets = [[] for _ in range(self.groups)]
        for entry, label in zip(entries, labels):
            entry["label"] = int(label)
            buckets[int(label)].append(entry)
        self.base = []
        leftovers = []
        for bucket in buckets:
            bucket.sort(key=lambda entry: entry["key"])
            self.base.append(bucket[:self.minimum])
            leftovers.extend(bucket[self.minimum:])
        leftovers.sort(key=lambda entry: entry["key"])
        self.overflow = leftovers[:self.overflow_capacity]
        assert len(self.items) <= self.capacity


def _activation_signature(X):
    """Sign-invariant frozen-activation geometry for the controlled benchmark."""
    x = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    signature = np.einsum("ni,nj->nij", x, x).reshape(len(X), -1)
    signature /= np.linalg.norm(signature, axis=1, keepdims=True) + 1e-12
    return signature


def _xy(entries):
    return (np.stack([entry["x"] for entry in entries]),
            np.stack([entry["y"] for entry in entries]))


def _assign(entries, centers):
    if not entries:
        return np.empty(0, dtype=int)
    X, _ = _xy(entries)
    return np.argmax(_activation_signature(X) @ centers.T, axis=1)


def _cluster(Z, groups, iterations=30):
    """Non-empty deterministic spherical k-means."""
    groups = min(int(groups), len(Z))
    first = int(np.argmax(np.linalg.norm(Z, axis=1)))
    chosen = [first]
    best = Z @ Z[first]
    for _ in range(1, groups):
        nxt = int(np.argmin(best))
        chosen.append(nxt)
        best = np.maximum(best, Z @ Z[nxt])
    centers = Z[np.asarray(chosen)].copy()
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
    labels = np.full(len(Z), -1, dtype=int)
    for _ in range(iterations):
        similarity = Z @ centers.T
        new_labels = np.argmax(similarity, axis=1)
        # Re-seed every empty cell with the point least represented by its
        # currently assigned center; this prevents silent NaN centers.
        empty = [g for g in range(groups) if not np.any(new_labels == g)]
        if empty:
            represented = similarity[np.arange(len(Z)), new_labels]
            available = np.argsort(represented)
            used = set()
            counts = np.bincount(new_labels, minlength=groups)
            for g in empty:
                # Move a poorly represented point only from a donor that keeps
                # at least one member. Taking a singleton merely moves the empty
                # cell and can leave a NaN center on duplicate-heavy streams.
                idx = next(
                    int(i) for i in available
                    if int(i) not in used and counts[new_labels[int(i)]] > 1
                )
                used.add(idx)
                donor = int(new_labels[idx])
                counts[donor] -= 1
                new_labels[idx] = g
                counts[g] += 1
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for g in range(groups):
            center = Z[labels == g].mean(axis=0)
            centers[g] = center / (np.linalg.norm(center) + 1e-12)
    return labels, centers


def _align_centers(old, new):
    """Align refreshed cells and return the new-label -> persistent-id map."""
    groups = len(old)
    similarity = old @ new.T
    # Exact bitmask assignment is O(K 2^K), cheap for K_max=2R=8, and avoids
    # both a scipy dependency and a greedy label-swap artifact.
    if groups <= 16:
        states = {0: (0.0, ())}
        for old_idx in range(groups):
            next_states = {}
            for mask, (score, perm) in states.items():
                for new_idx in range(groups):
                    bit = 1 << new_idx
                    if mask & bit:
                        continue
                    candidate = (score + similarity[old_idx, new_idx],
                                 perm + (new_idx,))
                    target = mask | bit
                    if target not in next_states or candidate[0] > next_states[target][0]:
                        next_states[target] = candidate
            states = next_states
        best_perm = states[(1 << groups) - 1][1]
        new_to_old = np.empty(groups, dtype=int)
        for old_idx, new_idx in enumerate(best_perm):
            new_to_old[new_idx] = old_idx
        return new[np.asarray(best_perm)], new_to_old
    remaining = set(range(groups))
    aligned = np.empty_like(new)
    new_to_old = np.empty(groups, dtype=int)
    for g in range(groups):
        pick = max(remaining, key=lambda j: similarity[g, j])
        aligned[g] = new[pick]
        new_to_old[pick] = g
        remaining.remove(pick)
    return aligned, new_to_old


def _nmse(M, entries):
    X, Y = _xy(entries)
    return E._mean_nmse(M, X, Y)


def _group_nmse(M, grouped):
    return np.asarray([_nmse(M, entries) for entries in grouped])


def _fit_groups(grouped, weights, rank, ridge):
    xs, ys, sample_weights = [], [], []
    for weight, entries in zip(weights, grouped):
        X, Y = _xy(entries)
        xs.append(X)
        ys.append(Y)
        sample_weights.append(np.full(len(entries), weight / len(entries)))
    return E._empirical_rrr(
        np.concatenate(xs), np.concatenate(ys),
        np.concatenate(sample_weights), rank, ridge
    )


def _select_same_pool(train_grouped, audit_grouped, frequencies, robust_mask,
                      rank, ridge, rounds, eta, mean_margin, min_audit_gain,
                      blend_grid):
    """Generate all candidates from one train pool; select on disjoint audit."""
    q0 = np.asarray(frequencies, dtype=float)
    q0 /= q0.sum()
    if any(not entries for entries in train_grouped):
        raise RuntimeError("refresh must keep every train cell non-empty")
    if any(not entries for entries in audit_grouped):
        # A finite audit split can miss an observable cell. In that event it is
        # invalid to tune a robust path on the remaining cells, so return the
        # train-only frequency baseline instead of deleting the rare cell or
        # training on audit. This path is primarily a degenerate-stream guard.
        baseline = _fit_groups(train_grouped, q0, rank, ridge)
        supported = [g for g, entries in enumerate(audit_grouped) if entries]
        if supported:
            audit_loss = np.asarray([
                _nmse(baseline, audit_grouped[g]) for g in supported
            ])
            support_weight = q0[supported]
            support_weight /= support_weight.sum()
            audit_mean = float(support_weight @ audit_loss)
            audit_worst = float(audit_loss.max())
        else:
            audit_mean = np.nan
            audit_worst = np.nan
        return baseline, {
            "selected_round": 0,
            "selected_candidate": "frequency_baseline_missing_audit_support",
            "baseline_audit_mean_nmse": audit_mean,
            "baseline_audit_worst_nmse": audit_worst,
            "selected_audit_mean_nmse": audit_mean,
            "selected_audit_worst_nmse": audit_worst,
            "feasible_candidates": 1,
            "robust_groups": supported,
            "selected_weights": q0.tolist(),
            "missing_audit_groups": [
                g for g, entries in enumerate(audit_grouped) if not entries
            ],
        }
    robust_mask = np.asarray(robust_mask, dtype=bool)
    active = np.flatnonzero(robust_mask)
    if len(active) < 2:
        robust_mask[:] = True
        active = np.flatnonzero(robust_mask)
    active_mass = float(q0[active].sum())
    baseline = _fit_groups(train_grouped, q0, rank, ridge)
    baseline_groups = _group_nmse(baseline, audit_grouped)
    baseline_mean = float(q0 @ baseline_groups)
    baseline_worst = float(baseline_groups[active].max())
    best = baseline
    best_weights = q0.copy()
    best_round = 0
    best_candidate = "frequency_baseline"
    best_mean = baseline_mean
    best_worst = baseline_worst
    feasible = 1

    def consider(candidate, candidate_weights, candidate_name, candidate_round):
        nonlocal best, best_weights, best_candidate, best_round
        nonlocal best_mean, best_worst, feasible
        audit_loss = _group_nmse(candidate, audit_grouped)
        candidate_mean = float(q0 @ audit_loss)
        candidate_worst = float(audit_loss[active].max())
        if candidate_mean <= baseline_mean + mean_margin:
            feasible += 1
            if candidate_worst < best_worst - min_audit_gain:
                best = candidate
                best_weights = np.asarray(candidate_weights).copy()
                best_candidate = candidate_name
                best_round = candidate_round
                best_mean = candidate_mean
                best_worst = candidate_worst

    # A deterministic Pareto path uses the same coreset and only flattens the
    # empirical cell-frequency weights.  It avoids relying on large, discontinuous
    # Hedge jumps near a low-rank spectral switch.
    uniform_active = np.full(len(active), active_mass / len(active))
    for index, rho in enumerate(blend_grid, start=1):
        blend = q0.copy()
        blend[active] = ((1.0 - float(rho)) * q0[active]
                         + float(rho) * uniform_active)
        consider(_fit_groups(train_grouped, blend, rank, ridge), blend,
                 f"frequency_uniform_{float(rho):.3f}", -index)

    dual = q0[active] / active_mass
    previous = baseline
    for step in range(1, rounds + 1):
        train_loss = _group_nmse(previous, train_grouped)[active]
        scaled = train_loss / (train_loss.mean() + 1e-12)
        logits = np.log(dual + 1e-300) + eta * scaled
        logits -= logits.max()
        dual = np.exp(logits)
        dual /= dual.sum()
        candidate_weights = q0.copy()
        candidate_weights[active] = active_mass * dual
        candidate = _fit_groups(train_grouped, candidate_weights, rank, ridge)
        consider(candidate, candidate_weights, f"hedge_{step}", step)
        previous = candidate

    return best, {
        "selected_round": best_round,
        "selected_candidate": best_candidate,
        "baseline_audit_mean_nmse": baseline_mean,
        "baseline_audit_worst_nmse": baseline_worst,
        "selected_audit_mean_nmse": best_mean,
        "selected_audit_worst_nmse": best_worst,
        "feasible_candidates": feasible,
        "robust_groups": active.tolist(),
        "selected_weights": best_weights.tolist(),
    }


def _cell_coherence(store, centers):
    """Mean train-signature similarity; a label-free cell reliability proxy."""
    grouped = store.grouped()
    coherence = []
    for group, entries in enumerate(grouped):
        X, _ = _xy(entries)
        similarity = _activation_signature(X) @ centers[group]
        coherence.append(float(similarity.mean()))
    return np.asarray(coherence)


def _initialize(pre_train, pre_audit, groups, train_capacity, audit_capacity,
                minimum):
    """Initialize codebook/stores in a helper so large scratch dies on return."""
    if len(pre_train.items) < groups or len(pre_audit.items) < groups:
        raise RuntimeError("warmup has insufficient train/audit support")
    X, _ = _xy(pre_train.items)
    train_labels, centers = _cluster(_activation_signature(X), groups)
    audit_labels = _assign(pre_audit.items, centers)
    train_store = MinPlusOverflowStore(train_capacity, groups, minimum)
    audit_store = MinPlusOverflowStore(audit_capacity, groups, minimum)
    train_store.rebuild(pre_train.items, train_labels)
    audit_store.rebuild(pre_audit.items, audit_labels)
    # This frequency state generates candidates, so it is train-only. Audit
    # labels are used solely to build the held-out selection reservoirs.
    counts = np.bincount(train_labels, minlength=groups).astype(float)
    return centers, train_store, audit_store, counts


def _refresh(centers, train_store, audit_store):
    """Refresh on train only, align identities, and rebalance shared overflow."""
    train_entries = train_store.items
    X, _ = _xy(train_entries)
    raw_labels, raw_centers = _cluster(_activation_signature(X), len(centers))
    centers, new_to_old = _align_centers(centers, raw_centers)
    # Preserve the non-empty labels produced by _cluster. Re-running argmax on
    # duplicate centers collapses every tie to cell zero and recreates empties.
    train_labels = new_to_old[raw_labels]
    audit_entries = audit_store.items
    audit_labels = _assign(audit_entries, centers)
    train_store.rebuild(train_entries, train_labels)
    audit_store.rebuild(audit_entries, audit_labels)
    return centers


def run_sgcr_v2(cfg, batches, seed, *, max_groups=None, warmup_windows=16,
                refresh_every=20, minimum_per_cell=32, rounds=24, eta=0.1,
                mean_margin=0.01, min_audit_gain=0.0,
                blend_grid=(0.10, 0.20, 0.35, 0.50, 0.70, 1.00),
                coherence_ratio=0.60,
                audit_fraction=0.50, refit_on_all=False,
                return_debug=False, return_frequency_baseline=False):
    d = batches[0][0].shape[1]
    groups = int(2 * cfg["R"] if max_groups is None else max_groups)
    raw_slots = (cfg["buf_u"] + cfg["buf_g"]) * cfg["batch"]
    numeric_budget = raw_slots * (2 * d)
    # One center per cell; arrival counts and fixed bookkeeping are persistent.
    metadata_floats = groups * d * d + 4 * groups + 1
    # x, y, priority key, integer id, and cell label are all charged as one
    # float-sized scalar each. Python-container overhead is excluded from this
    # logical payload ledger and must be measured separately in system results.
    numeric_per_entry = 2 * d + 3
    retained_capacity = (numeric_budget - metadata_floats) // numeric_per_entry
    audit_capacity = int(np.floor(retained_capacity * audit_fraction))
    train_capacity = int(retained_capacity - audit_capacity)
    minimum = min(int(minimum_per_cell), train_capacity // groups,
                  audit_capacity // groups)
    if minimum < 1:
        raise ValueError("budget cannot support one train/audit item per cell")

    rng = np.random.default_rng(seed + 1901)
    pre_train = PriorityReservoir(train_capacity)
    pre_audit = PriorityReservoir(audit_capacity)
    centers = None
    train_store = None
    audit_store = None
    train_arrival_counts = None
    next_id = 0
    current_model = None
    last_fit_window = 0
    last_stats = None
    fit_count = 0
    baseline_selected_count = 0
    debug_fit_stats = [] if return_debug else None
    train_probability = train_capacity / (train_capacity + audit_capacity)

    for window_idx, (X_raw, Y_raw) in enumerate(batches):
        # The generator concatenates sources inside a window. The permutation is
        # part of the learner and prevents accidental use of row order.
        order = rng.permutation(len(X_raw))
        current_train, current_audit = [], []
        for row in order:
            entry = {
                "id": int(next_id),
                "key": float(rng.random()),
                "x": X_raw[row].copy(),
                "y": Y_raw[row].copy(),
            }
            next_id += 1
            if rng.random() < train_probability:
                current_train.append(entry)
                if centers is None:
                    pre_train.offer(entry)
            else:
                current_audit.append(entry)
                if centers is None:
                    pre_audit.offer(entry)

        if centers is None and window_idx + 1 >= warmup_windows:
            centers, train_store, audit_store, train_arrival_counts = _initialize(
                pre_train, pre_audit, groups, train_capacity, audit_capacity,
                minimum
            )
            pre_train.items = []
            pre_audit.items = []
            # Current entries are already represented in the warmup reservoirs;
            # do not insert them twice.
            current_train = []
            current_audit = []
        elif centers is not None:
            train_labels = _assign(current_train, centers)
            audit_labels = _assign(current_audit, centers)
            for entry, label in zip(current_train, train_labels):
                train_store.offer(entry, label)
            for entry, label in zip(current_audit, audit_labels):
                audit_store.offer(entry, label)
            # Frequency weights are candidate-generation state and therefore use
            # TRAIN arrivals only. Audit X/Y affect selection, never candidates.
            train_arrival_counts += np.bincount(
                train_labels, minlength=groups
            )

        if centers is not None and (window_idx + 1) % refresh_every == 0:
            centers = _refresh(centers, train_store, audit_store)
            train_grouped = train_store.grouped()
            audit_grouped = audit_store.grouped()
            coherence = _cell_coherence(train_store, centers)
            robust_mask = coherence >= coherence_ratio * np.median(coherence)
            current_model, stats = _select_same_pool(
                train_grouped, audit_grouped, train_arrival_counts,
                robust_mask, cfg["R"],
                cfg.get("ridge", 1e-6), rounds, eta, mean_margin,
                min_audit_gain, blend_grid
            )
            if refit_on_all:
                all_grouped = [train + audit for train, audit in
                               zip(train_grouped, audit_grouped)]
                current_model = _fit_groups(
                    all_grouped, np.asarray(stats["selected_weights"]),
                    cfg["R"], cfg.get("ridge", 1e-6)
                )
            stats["cell_coherence"] = coherence.tolist()
            stats["refit_on_all"] = bool(refit_on_all)
            stats["window"] = window_idx + 1
            last_stats = stats
            fit_count += 1
            baseline_selected_count += int(stats["selected_round"] == 0)
            if return_debug:
                debug_fit_stats.append(dict(stats))
            last_fit_window = window_idx + 1

    if centers is None:
        raise RuntimeError("warmup exceeds stream length")
    if last_fit_window != len(batches):
        centers = _refresh(centers, train_store, audit_store)
        coherence = _cell_coherence(train_store, centers)
        robust_mask = coherence >= coherence_ratio * np.median(coherence)
        current_model, stats = _select_same_pool(
            train_store.grouped(), audit_store.grouped(), train_arrival_counts,
            robust_mask,
            cfg["R"], cfg.get("ridge", 1e-6), rounds, eta, mean_margin,
            min_audit_gain, blend_grid
        )
        if refit_on_all:
            train_grouped = train_store.grouped()
            audit_grouped = audit_store.grouped()
            all_grouped = [train + audit for train, audit in
                           zip(train_grouped, audit_grouped)]
            current_model = _fit_groups(
                all_grouped, np.asarray(stats["selected_weights"]), cfg["R"],
                cfg.get("ridge", 1e-6)
            )
        stats["cell_coherence"] = coherence.tolist()
        stats["refit_on_all"] = bool(refit_on_all)
        stats["window"] = len(batches)
        last_stats = stats
        fit_count += 1
        baseline_selected_count += int(stats["selected_round"] == 0)
        if return_debug:
            debug_fit_stats.append(dict(stats))

    train_ids = {entry["id"] for entry in train_store.items}
    audit_ids = {entry["id"] for entry in audit_store.items}
    assert train_ids.isdisjoint(audit_ids)
    actual_entries = len(train_store.items) + len(audit_store.items)
    persistent_numeric_floats = metadata_floats + actual_entries * numeric_per_entry
    assert persistent_numeric_floats <= numeric_budget
    output = dict(last_stats)
    output.update({
        "max_groups": groups,
        "minimum_per_cell": minimum,
        "train_capacity": train_capacity,
        "audit_capacity": audit_capacity,
        "actual_train_entries": len(train_store.items),
        "actual_audit_entries": len(audit_store.items),
        "train_group_sizes": [len(x) for x in train_store.grouped()],
        "audit_group_sizes": [len(x) for x in audit_store.grouped()],
        "arrival_frequencies": (
            train_arrival_counts / train_arrival_counts.sum()
        ).tolist(),
        "numeric_budget_floats": int(numeric_budget),
        "persistent_numeric_floats": int(persistent_numeric_floats),
        "fit_count": fit_count,
        "selected_round_zero_rate": baseline_selected_count / fit_count,
        "working_memory_note": (
            "clustering/signature matrices and RRR candidates are transient "
            "working memory and are not included in persistent-state floats"
        ),
    })
    if return_frequency_baseline:
        frequency_weights = train_arrival_counts / train_arrival_counts.sum()
        output["evaluation_frequency_baseline_model"] = _fit_groups(
            train_store.grouped(), frequency_weights, cfg["R"],
            cfg.get("ridge", 1e-6)
        )
    if return_debug:
        output["debug_train_ids"] = sorted(train_ids)
        output["debug_audit_ids"] = sorted(audit_ids)
        output["debug_fit_stats"] = debug_fit_stats
        output["debug_centers"] = centers.copy()
        output["debug_train_x"] = np.stack(
            [entry["x"] for entry in train_store.items]
        )
        output["debug_train_labels"] = np.asarray(
            [entry["label"] for entry in train_store.items], dtype=int
        )
        output["debug_audit_x"] = np.stack(
            [entry["x"] for entry in audit_store.items]
        )
        output["debug_audit_labels"] = np.asarray(
            [entry["label"] for entry in audit_store.items], dtype=int
        )
    return current_model, output


def evaluate(teachers=E.DEV_TEACHERS, streams=E.DEV_STREAMS, **overrides):
    cfg = E._cfg()
    rows = []
    print(f"{'world':14s} {'sgcr2':>7s} {'psr':>7s} {'mean2':>7s} "
          f"{'meanP':>7s} {'round':>5s} {'occ':>5s}")
    for teacher_seed in teachers:
        teacher = E.BM.build_teacher(E.REGIME, cfg, teacher_seed)
        for stream_seed in streams:
            windows, frequencies = E.BM.make_stream(cfg, stream_seed)
            batches, sources, _, test = E.BM.materialize(
                teacher, cfg, windows, stream_seed
            )
            model, stats = run_sgcr_v2(cfg, batches, E.RUN_SEED, **overrides)
            psr = E.P.run_psr_lora(cfg, batches, E.RUN_SEED)[0][-1]
            mean2, tail2 = E._mean_tail(model, sources, frequencies, test)
            meanp, tailp = E._mean_tail(psr, sources, frequencies, test)
            rows.append((mean2, tail2, meanp, tailp))
            occupancy = stats["actual_train_entries"] + stats["actual_audit_entries"]
            print(f"t{teacher_seed % 100000:05d}/s{stream_seed:04d} "
                  f"{tail2:7.3f} {tailp:7.3f} {mean2:7.3f} {meanp:7.3f} "
                  f"{stats['selected_round']:5d} {occupancy:5d}")
    values = np.asarray(rows)
    print("\nSGCR-v2 mean/tail %.4f / %.4f" %
          (values[:, 0].mean(), values[:, 1].mean()))
    print("PSR     mean/tail %.4f / %.4f" %
          (values[:, 2].mean(), values[:, 3].mean()))
    delta = values[:, 1] - values[:, 3]
    print("SGCR-v2 - PSR tail %+0.4f, wins %d/%d" %
          (delta.mean(), np.sum(delta > 0), len(delta)))
    return rows


if __name__ == "__main__":
    evaluate()
