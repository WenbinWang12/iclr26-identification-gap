"""Minimal structural tests for the DEVELOPMENT-only SGCR-v4 implementation."""

from __future__ import annotations

import inspect

import numpy as np

import explore_sgcr_v4 as V4


def _rank(model, tolerance=1e-8):
    singular = np.linalg.svd(model, compute_uv=False)
    return int(np.sum(singular > tolerance * singular[0]))


def main():
    rng = np.random.default_rng(20260827)
    d, rank, groups = 6, 2, 4
    teacher = rng.normal(size=(d, d))
    grouped = []
    for group in range(groups):
        entries = []
        for index in range(12):
            x = rng.normal(size=d)
            y = teacher @ x + 0.03 * rng.normal(size=d)
            entries.append({"x": x, "y": y, "id": (group, index)})
        grouped.append(entries)
    q0 = np.asarray([0.55, 0.20, 0.15, 0.10])

    # The activation generator has no audit argument and returns only models
    # obtained from the supplied training cells.
    assert "audit" not in inspect.signature(V4._activation_train_path).parameters
    activation = V4._activation_train_path(
        grouped, q0, np.arange(groups), rank, 1e-6,
        rounds=3, eta=0.1, blend_grid=(0.5, 1.0),
    )
    assert len(activation) == 5
    assert all(_rank(model) <= rank for _, model in activation)

    windows = []
    for _ in range(6):
        X = rng.normal(size=(16, d))
        Y = X @ teacher.T + 0.03 * rng.normal(size=(16, d))
        windows.append({"X": X, "Y": Y})
    spectral = V4._spectral_pareto_path(
        windows, d, rank, 1e-6, rounds=2, eta=1.0,
        rho_grid=(0.5, 1.0),
    )
    assert len(spectral) == 5  # q0 plus two nonduplicate weights per later round
    assert all(_rank(model) <= rank for _, model in spectral)

    baseline = activation[0][1]
    deliberately_bad = 100.0 * np.eye(d)
    selected, stats = V4._select(
        [("frequency_baseline", baseline), ("bad", deliberately_bad)],
        baseline, grouped, q0, np.arange(groups),
        mean_margin=-1.0, z_value=1.645,
    )
    assert stats["selected_candidate"] == "frequency_baseline"
    assert np.array_equal(selected, baseline)
    assert stats["feasible_candidates"] == 1
    print("SGCR-v4 minimal self-test: PASS")


if __name__ == "__main__":
    main()

