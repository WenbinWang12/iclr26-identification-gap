"""Exploratory targeted remedy for the phase-2f failure cases.

This file is deliberately separate from the frozen phase-2f implementation and
its held-out protocol.  It tests one question only:

    Does discovering pseudo-groups from *example-level gradient signatures* and
    optimizing their actual empirical losses repair the mixed-window/source
    mismatch of PSR?

The learner never sees latent source labels.  It uses the same 4+12 window
working-set budget as PSR, removes duplicate windows, and then:

  1. fits the uniform empirical rank-R reduced-rank regression baseline;
  2. forms per-example operator-gradient signatures at that baseline;
  3. clusters those signatures with deterministic farthest-first spherical
     k-means;
  4. runs multiplicative weights over the discovered clusters, re-solving the
     *actual weighted empirical RRR* at every round; and
  5. returns the lowest worst-cluster-loss candidate whose empirical mean NMSE
     is within ``mean_margin`` of the uniform baseline.

This is an exploration, not confirmatory evidence.  The real neural analogue
would replace the exact RRR solve by replay AdamW on LoRA factors and use a
low-dimensional random projection of per-example LoRA gradients.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PHASE2F = HERE.parent / "phase2f_psr_lora"
if str(PHASE2F) not in sys.path:
    sys.path.insert(0, str(PHASE2F))

import benchmark_v4 as BM  # noqa: E402
import baselines as BL  # noqa: E402
import harness as H  # noqa: E402
import psr_lora as P  # noqa: E402
from compare import _mean_tail  # noqa: E402


DEV_TEACHERS = [20260701, 20260710, 20260711, 20260712, 20260713]
DEV_STREAMS = [6060, 6061]
REGIME = "capacity_limited"
RUN_SEED = 7


def _cfg():
    cfg = dict(BM.DEFAULT_CFG)
    cfg.update({"buf_u": 4, "buf_g": 12, "coverage": True,
                "reweight": True, "ridge": 1e-6})
    return cfg


def _coverage_working_set(cfg, batches, seed):
    """Reproduce phase-2f's 4+12 X-only buffer, but do not double count overlap."""
    rng = np.random.default_rng(seed + 41)
    uniform = H.Reservoir(cfg["buf_u"], rng)
    diverse = []
    for idx, (X, Y) in enumerate(batches):
        entry = {"id": idx, "X": X, "Y": Y, "idir": P._indir(X)}
        uniform.offer(entry)
        P._div_offer(diverse, entry, cfg["buf_g"])

    unique = {}
    for entry in uniform.items + diverse:
        unique[entry["id"]] = entry
    return list(unique.values())


def _sqrt_and_inv_sqrt(C):
    vals, vecs = np.linalg.eigh((C + C.T) * 0.5)
    vals = np.maximum(vals, 1e-12)
    root = (vecs * np.sqrt(vals)[None, :]) @ vecs.T
    inv_root = (vecs * (1.0 / np.sqrt(vals))[None, :]) @ vecs.T
    return root, inv_root


def _empirical_rrr(X, Y, sample_weight, rank, ridge):
    """Exact weighted empirical reduced-rank regression for row-major X and Y."""
    w = np.asarray(sample_weight, float)
    w = w / (w.sum() + 1e-30)
    Cxx = X.T @ (w[:, None] * X) + ridge * np.eye(X.shape[1])
    Cyx = Y.T @ (w[:, None] * X)
    teacher = np.linalg.solve(Cxx, Cyx.T).T
    root, inv_root = _sqrt_and_inv_sqrt(Cxx)
    whitened = teacher @ root
    U, s, Vt = np.linalg.svd(whitened, full_matrices=False)
    truncated = (U[:, :rank] * s[:rank][None, :]) @ Vt[:rank]
    return truncated @ inv_root


def _gradient_signatures(M, X, Y):
    """Per-example operator gradients, invariant to the x -> -x symmetry."""
    residual = X @ M.T - Y
    grad = np.einsum("ni,nj->nij", residual, X).reshape(len(X), -1)
    norm = np.linalg.norm(grad, axis=1, keepdims=True)
    usable = norm[:, 0] > 1e-10
    grad[usable] /= norm[usable]
    # Near-zero gradients carry no reliable grouping information.  Append the
    # normalized input outer-product signature as a deterministic fallback.
    xnorm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    xin = np.einsum("ni,nj->nij", xnorm, xnorm).reshape(len(X), -1)
    grad[~usable] = xin[~usable] / (
        np.linalg.norm(xin[~usable], axis=1, keepdims=True) + 1e-12
    )
    return grad


def _farthest_spherical_kmeans(Z, groups, iters=20):
    """Deterministic spherical k-means with farthest-first initialization."""
    n = len(Z)
    groups = min(int(groups), n)
    centers = [int(np.argmax(np.linalg.norm(Z, axis=1)))]
    best_sim = Z @ Z[centers[0]]
    for _ in range(1, groups):
        nxt = int(np.argmin(best_sim))
        centers.append(nxt)
        best_sim = np.maximum(best_sim, Z @ Z[nxt])
    C = Z[np.asarray(centers)].copy()
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    labels = np.full(n, -1, int)
    for _ in range(iters):
        new_labels = np.argmax(Z @ C.T, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for g in range(groups):
            members = Z[labels == g]
            if len(members):
                c = members.mean(axis=0)
                C[g] = c / (np.linalg.norm(c) + 1e-12)
    return labels


def _group_nmse(M, X, Y, labels, groups):
    residual = X @ M.T - Y
    out = np.empty(groups)
    for g in range(groups):
        take = labels == g
        out[g] = np.sum(residual[take] ** 2) / (np.sum(Y[take] ** 2) + 1e-12)
    return out


def _mean_nmse(M, X, Y):
    return float(np.sum((X @ M.T - Y) ** 2) / (np.sum(Y ** 2) + 1e-12))


def fit_gcdr(items, rank, ridge=1e-6, groups=8, rounds=24,
             eta=0.7, mean_margin=0.01):
    """Gradient-Clustered, mean-constrained Group-DRO RRR."""
    X = np.concatenate([e["X"] for e in items], axis=0)
    Y = np.concatenate([e["Y"] for e in items], axis=0)
    n = len(X)
    uniform = np.full(n, 1.0 / n)
    base = _empirical_rrr(X, Y, uniform, rank, ridge)
    base_mean = _mean_nmse(base, X, Y)

    signature = _gradient_signatures(base, X, Y)
    labels = _farthest_spherical_kmeans(signature, groups)
    present = np.unique(labels)
    remap = {old: new for new, old in enumerate(present)}
    labels = np.asarray([remap[x] for x in labels], int)
    groups = len(present)
    counts = np.bincount(labels, minlength=groups).astype(float)

    # Round zero is exactly the uniform-example solution.
    dual = counts / counts.sum()
    best = base
    best_worst = float(np.max(_group_nmse(base, X, Y, labels, groups)))
    best_round = 0
    feasible = 1
    for step in range(1, rounds + 1):
        losses = _group_nmse(best if step == 1 else candidate,
                             X, Y, labels, groups)
        scaled = losses / (losses.mean() + 1e-12)
        logits = np.log(dual + 1e-300) + eta * scaled
        logits -= logits.max()
        dual = np.exp(logits)
        dual /= dual.sum()
        sw = dual[labels] / counts[labels]
        candidate = _empirical_rrr(X, Y, sw, rank, ridge)
        candidate_mean = _mean_nmse(candidate, X, Y)
        candidate_groups = _group_nmse(candidate, X, Y, labels, groups)
        candidate_worst = float(candidate_groups.max())
        if candidate_mean <= base_mean + mean_margin:
            feasible += 1
            if candidate_worst < best_worst - 1e-12:
                best, best_worst, best_round = candidate, candidate_worst, step

    return best, {
        "base_mean_nmse": base_mean,
        "selected_mean_nmse": _mean_nmse(best, X, Y),
        "selected_worst_cluster_nmse": best_worst,
        "selected_round": best_round,
        "feasible_candidates": feasible,
        "cluster_counts": counts.astype(int).tolist(),
        "working_windows": len(items),
    }


def run_gcdr(cfg, batches, seed, **over):
    items = _coverage_working_set(cfg, batches, seed)
    params = {"rank": cfg["R"], "ridge": cfg.get("ridge", 1e-6)}
    params.update(over)
    return fit_gcdr(items, **params)


def evaluate(teachers=DEV_TEACHERS, streams=DEV_STREAMS, **gcdr_over):
    cfg = _cfg()
    rows = []
    print(f"{'world':14s} {'gcdr':>7s} {'psr':>7s} {'cvar':>7s} "
          f"{'meanG':>7s} {'meanP':>7s} {'round':>5s}")
    for ts in teachers:
        teacher = BM.build_teacher(REGIME, cfg, ts)
        for ss in streams:
            windows, freqs = BM.make_stream(cfg, ss)
            batches, srcs, _, test = BM.materialize(teacher, cfg, windows, ss)
            M_g, stats = run_gcdr(cfg, batches, RUN_SEED, **gcdr_over)
            M_p = P.run_psr_lora(cfg, batches, RUN_SEED)[0][-1]
            M_c = BL.run_excess_cvar(cfg, batches, RUN_SEED)[0][-1]
            mg, tg = _mean_tail(M_g, srcs, freqs, test)
            mp, tp = _mean_tail(M_p, srcs, freqs, test)
            mc, tc = _mean_tail(M_c, srcs, freqs, test)
            rows.append((ts, ss, mg, tg, mp, tp, mc, tc))
            print(f"t{ts % 100000:05d}/s{ss:04d} {tg:7.3f} {tp:7.3f} {tc:7.3f} "
                  f"{mg:7.3f} {mp:7.3f} {stats['selected_round']:5d}")

    a = np.asarray([r[2:] for r in rows])
    names = ["gcdr", "psr", "cvar"]
    print("\n--- exploratory summary ---")
    for j, name in enumerate(names):
        mean = a[:, 2 * j]
        tail = a[:, 2 * j + 1]
        print(f"{name:8s} mean {mean.mean():.4f}  tail {tail.mean():.4f}")
    dg_p = a[:, 1] - a[:, 3]
    dg_c = a[:, 1] - a[:, 5]
    print(f"GCDR - PSR tail:  {dg_p.mean():+.4f}, wins {int(np.sum(dg_p > 0))}/{len(dg_p)}")
    print(f"GCDR - CVaR tail: {dg_c.mean():+.4f}, wins {int(np.sum(dg_c > 0))}/{len(dg_c)}")
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--eta", type=float, default=0.7)
    parser.add_argument("--mean-margin", type=float, default=0.01)
    args = parser.parse_args()
    evaluate(groups=args.groups, rounds=args.rounds, eta=args.eta,
             mean_margin=args.mean_margin)
