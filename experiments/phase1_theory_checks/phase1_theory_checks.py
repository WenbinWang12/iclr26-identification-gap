"""Deterministic NumPy checks for the conditional Phase-1 theory.

The probes evaluate losses and constraints independently of their analytic
references. They are theorem regression checks, not neural-network evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import io
import json
import math
import platform
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "notes" / "phase1_theory_protocol.md"
SOURCE_PATH = HERE / "phase1_theory_checks.py"
TEST_PATH = HERE / "test_phase1.py"
EXACT_ATOL = 2e-11
PROJECTOR_ATOL = 1e-9
RANK_REL_TOL = 1e-10


def _as_json(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(observed: float, expected: float, atol: float = EXACT_ATOL, rtol: float = 0.0) -> tuple[bool, float, float]:
    error = abs(float(observed) - float(expected))
    tolerance = float(atol + rtol * max(abs(float(expected)), 1.0))
    return error <= tolerance, error, tolerance


def _add_gate(
    gates: list[dict[str, Any]],
    name: str,
    observed: Any,
    expected: Any,
    passed: bool,
    tolerance: float,
    notes: str = "",
    error: float | None = None,
    relation: str = "eq",
) -> None:
    if relation not in {"eq", "le", "ge"}:
        raise ValueError(f"unsupported gate relation: {relation}")
    numeric = isinstance(observed, (int, float, np.number)) and isinstance(expected, (int, float, np.number))
    if error is None and numeric and relation == "eq":
        error = abs(float(observed) - float(expected))
    if numeric:
        if relation == "eq":
            violation = max(0.0, abs(float(observed) - float(expected)) - float(tolerance))
        elif relation == "le":
            violation = max(0.0, float(observed) - float(expected) - float(tolerance))
        else:
            violation = max(0.0, float(expected) - float(observed) - float(tolerance))
    else:
        violation = 0.0 if passed else 1.0
    gates.append(
        {
            "name": name,
            "observed": _as_json(observed),
            "expected": _as_json(expected),
            "passed": bool(passed),
            "tolerance": float(tolerance),
            "relation": relation,
            "abs_error": None if relation != "eq" or error is None else float(error),
            "violation": float(violation),
            "notes": notes,
        }
    )


def _rank(matrix: np.ndarray, rel_tol: float = RANK_REL_TOL) -> int:
    values = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    if values.size == 0:
        return 0
    cutoff = rel_tol * max(1.0, float(values[0]))
    return int(np.count_nonzero(values > cutoff))


def _null_basis(matrix: np.ndarray, rel_tol: float = RANK_REL_TOL) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    dimension = matrix.shape[1]
    if matrix.shape[0] == 0:
        return np.eye(dimension)
    _, values, vh = np.linalg.svd(matrix, full_matrices=True)
    cutoff = rel_tol * max(1.0, float(values[0]) if values.size else 0.0)
    rank = int(np.count_nonzero(values > cutoff))
    return vh[rank:].T.copy()


def _null_projector(matrix: np.ndarray, rel_tol: float = RANK_REL_TOL) -> np.ndarray:
    basis = _null_basis(matrix, rel_tol)
    return basis @ basis.T


def _safe_projector(matrix: np.ndarray, rel_tol: float = RANK_REL_TOL) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    cutoff = rel_tol * max(1.0, float(np.max(np.abs(values))) if values.size else 0.0)
    selected = values <= cutoff
    if not np.any(selected):
        return np.zeros_like(matrix)
    basis = vectors[:, selected]
    return basis @ basis.T


def _givens(n: int, i: int, j: int, c: float, s: float) -> np.ndarray:
    result = np.eye(n)
    result[i, i] = c
    result[j, j] = c
    result[i, j] = -s
    result[j, i] = s
    return result


def _fixed_orthogonal(n: int, rotations: Iterable[tuple[int, int, float, float]]) -> np.ndarray:
    result = np.eye(n)
    for i, j, c, s in rotations:
        result = result @ _givens(n, i, j, c, s)
    return result


def _record_case(case: str, gates: list[dict[str, Any]], trace: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    for gate in gates:
        gate["case"] = case
    return {"case": case, "gates": gates}, {"case": case, **trace}


def check_taylor() -> tuple[dict[str, Any], dict[str, Any]]:
    p = 4
    hessian = np.diag([1.0, 2.0, 3.0, 4.0])
    rho = 0.7
    u = np.array([1.0, 2.0, -2.0, 0.0]) / 3.0
    gradient = np.array([0.0, 0.0, 0.3, 0.0])
    e4 = np.eye(p)[3]
    radii = 2.0 ** (-np.arange(7, dtype=float))

    def loss(x: np.ndarray) -> float:
        return float(gradient @ x + 0.5 * x @ hessian @ x + (rho / 6.0) * (u @ x) ** 3)

    def remainder(direction: np.ndarray, radius: float) -> tuple[float, float, float]:
        displacement = radius * direction
        observed = loss(displacement) - loss(np.zeros(p)) - gradient @ displacement - 0.5 * displacement @ hessian @ displacement
        expected = (rho / 6.0) * (u @ displacement) ** 3
        bound = (rho / 6.0) * np.linalg.norm(displacement) ** 3
        return observed, expected, bound

    gates: list[dict[str, Any]] = []
    trace: dict[str, Any] = {"rho": rho, "u": u, "radii": radii, "directions": {}}
    max_identity = 0.0
    max_bound_excess = -math.inf
    parallel_ratios: list[float] = []
    directions = {
        "parallel": u,
        "orthogonal": e4,
        "mixed": (u + e4) / np.sqrt(2.0),
    }
    for label, direction in directions.items():
        rows = []
        for radius in radii:
            observed, expected, bound = remainder(direction, float(radius))
            ok, error, tolerance = _close(observed, expected, 3e-12)
            _add_gate(gates, f"{label}_remainder_identity_r{radius:g}", observed, expected, ok, tolerance, error=error)
            max_identity = max(max_identity, error)
            max_bound_excess = max(max_bound_excess, abs(observed) - bound)
            if label == "parallel":
                parallel_ratios.append(abs(observed) / bound)
            rows.append({"radius": radius, "observed": observed, "expected": expected, "bound": bound})
        trace["directions"][label] = rows
    _add_gate(gates, "remainder_identity_max_error", max_identity, 0.0, max_identity <= 3e-12, 3e-12)
    _add_gate(gates, "remainder_bound_max_excess", max_bound_excess, 0.0, max_bound_excess <= 3e-12, 3e-12, relation="le")
    tight_error = max(abs(value - 1.0) for value in parallel_ratios)
    slope_values = [abs(remainder(u, float(radius))[0]) for radius in radii]
    slope = float(np.polyfit(np.log(radii), np.log(slope_values), 1)[0])
    _add_gate(gates, "parallel_cubic_tightness", tight_error, 0.0, tight_error <= 3e-12, 3e-12)
    _add_gate(gates, "parallel_loglog_slope", slope, 3.0, abs(slope - 3.0) <= 3e-10, 3e-10)

    basis = np.eye(p)[:, :2]
    safe_projection = basis.T @ gradient
    bad_gradient = gradient + np.array([0.2, 0.0, 0.0, 0.0])
    bad_projection = basis.T @ bad_gradient
    safe_linear = [float(gradient @ basis[:, index]) for index in range(basis.shape[1])]
    _add_gate(gates, "projected_stationarity_safe_norm", np.linalg.norm(safe_projection), 0.0, np.linalg.norm(safe_projection) <= 3e-12, 3e-12)
    _add_gate(gates, "projected_stationarity_safe_linear_max", max(abs(value) for value in safe_linear), 0.0, max(abs(value) for value in safe_linear) <= 3e-12, 3e-12)
    _add_gate(gates, "projected_stationarity_nonzero_control", np.linalg.norm(bad_projection), 0.2, abs(np.linalg.norm(bad_projection) - 0.2) <= 3e-12, 3e-12)
    trace.update({"parallel_ratios": parallel_ratios, "loglog_slope": slope, "safe_projection": safe_projection, "bad_projection": bad_projection})
    return _record_case("taylor_stationarity", gates, trace)


def check_sequential() -> tuple[dict[str, Any], dict[str, Any]]:
    hessian = np.diag([1.0, 2.0, 3.0])
    e1, e2, e3 = np.eye(3)

    def old_loss(point: np.ndarray) -> float:
        return float(0.5 * point @ hessian @ point)

    paths = {
        "aligned": [e1, e1, e1, e1],
        "orthogonal": [e1, e2, e3],
        "cancelling": [e1, -e1],
        "recurrent": [e1, e2, -e2, e2],
    }
    gates: list[dict[str, Any]] = []
    trace: dict[str, Any] = {"paths": {}}
    for label, updates in paths.items():
        point = np.zeros(3)
        previous_point = point.copy()
        previous_loss = old_loss(point)
        rows = []
        for step, update in enumerate(updates, start=1):
            point = point + update
            current_loss = old_loss(point)
            direct_increment = current_loss - previous_loss
            cross = float(previous_point @ hessian @ update)
            quadratic = float(0.5 * update @ hessian @ update)
            rows.append(
                {
                    "step": step,
                    "point": point.copy(),
                    "forgetting": current_loss,
                    "increment": direct_increment,
                    "cross": cross,
                    "quadratic": quadratic,
                }
            )
            ok, error, tolerance = _close(direct_increment, cross + quadratic)
            _add_gate(gates, f"{label}_increment_identity_{step}", direct_increment, cross + quadratic, ok, tolerance, error=error)
            previous_point = point.copy()
            previous_loss = current_loss
        total_update = np.sum(updates, axis=0)
        path_value = old_loss(point)
        expected_path = float(0.5 * total_update @ hessian @ total_update)
        ok, error, tolerance = _close(path_value, expected_path)
        _add_gate(gates, f"{label}_path_identity", path_value, expected_path, ok, tolerance, error=error)
        trace["paths"][label] = rows

    aligned = [row["forgetting"] for row in trace["paths"]["aligned"]]
    orthogonal_cross = [abs(row["cross"]) for row in trace["paths"]["orthogonal"]]
    cancelling = [row["forgetting"] for row in trace["paths"]["cancelling"]]
    recurrent = [row["forgetting"] for row in trace["paths"]["recurrent"]]
    _add_gate(gates, "aligned_expected_final", aligned[-1], 8.0, abs(aligned[-1] - 8.0) <= EXACT_ATOL, EXACT_ATOL)
    _add_gate(gates, "aligned_strictly_increasing", min(np.diff(aligned)), 0.5, min(np.diff(aligned)) >= 0.5 - EXACT_ATOL, EXACT_ATOL, relation="ge")
    _add_gate(gates, "orthogonal_cross_zero", max(orthogonal_cross), 0.0, max(orthogonal_cross) <= EXACT_ATOL, EXACT_ATOL)
    _add_gate(gates, "cancelling_returns_zero", cancelling[-1], 0.0, abs(cancelling[-1]) <= EXACT_ATOL, EXACT_ATOL)
    _add_gate(gates, "cancelling_negative_increment", cancelling[-1] - cancelling[-2], -0.5, abs(cancelling[-1] - cancelling[-2] + 0.5) <= EXACT_ATOL, EXACT_ATOL)
    _add_gate(gates, "recurrent_returns_to_first_point", recurrent[2], recurrent[0], abs(recurrent[2] - recurrent[0]) <= EXACT_ATOL, EXACT_ATOL)
    _add_gate(gates, "recurrent_returns_to_second_point", recurrent[3], recurrent[1], abs(recurrent[3] - recurrent[1]) <= EXACT_ATOL, EXACT_ATOL)

    means = np.array(
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1], [0.05, 0.05, 0.0]],
        dtype=float,
    )
    directions = np.array(
        [[0.2, 0.0, 0.0], [0.0, 0.15, 0.0], [0.0, 0.0, 0.1], [0.1, -0.1, 0.05]],
        dtype=float,
    )
    enumerated_values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(means)):
        final_point = np.sum(means + np.asarray(signs)[:, None] * directions, axis=0)
        enumerated_values.append(old_loss(final_point))
    enumerated_mean = float(np.mean(enumerated_values))
    mean_sum = np.sum(means, axis=0)
    innovation_expected = float(0.5 * mean_sum @ hessian @ mean_sum + 0.5 * sum(vector @ hessian @ vector for vector in directions))
    ok, error, tolerance = _close(enumerated_mean, innovation_expected, 3e-12)
    _add_gate(gates, "independent_innovation_exact_expectation", enumerated_mean, innovation_expected, ok, tolerance, error=error)
    _add_gate(gates, "independent_innovation_frozen_value", enumerated_mean, 0.125, abs(enumerated_mean - 0.125) <= 3e-12, 3e-12)

    # This is deliberately outside the quadratic-old-loss assumption.
    eta = 0.1
    first = e1.copy()
    second = 2.0 * e1
    quartic_loss = lambda point: float(0.5 * point @ hessian @ point + eta / 4.0 * np.sum(point**4))
    observed_quartic_increment = quartic_loss(second) - quartic_loss(first)
    quadratic_prediction = float(first @ hessian @ e1 + 0.5 * e1 @ hessian @ e1)
    guard_gap = abs(observed_quartic_increment - quadratic_prediction)
    _add_gate(
        gates,
        "nonquadratic_increment_guard",
        guard_gap,
        0.1,
        guard_gap >= 0.1,
        0.0,
        "out-of-assumption control must not pass as an exact quadratic identity",
        relation="ge",
    )
    trace.update({"innovation_enumerated_mean": enumerated_mean, "innovation_expected": innovation_expected, "quartic_guard_gap": guard_gap})
    return _record_case("sequential_innovations", gates, trace)


def _spectral_factors() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_rotation = _fixed_orthogonal(5, [(0, 1, 0.8, 0.6), (1, 2, 0.6, 0.8), (3, 4, 0.8, -0.6)])
    right_rotation = _fixed_orthogonal(4, [(0, 1, 0.8, -0.6), (2, 3, 0.6, 0.8)])
    left_factor = left_rotation @ np.diag(np.sqrt([1.0, 2.0, 3.0, 4.0, 5.0])) @ left_rotation.T
    right_factor = right_rotation @ np.diag(np.sqrt([1.0, 1.5, 2.5, 4.0])) @ right_rotation.T
    return left_factor @ left_factor.T, right_factor @ right_factor.T, left_factor, right_factor


def check_decomposition_and_spectral() -> tuple[dict[str, Any], dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    trace: dict[str, Any] = {}

    # C4: an exact three-way accounting identity with all terms nonzero.
    target_one = np.diag([3.0, 0.0])
    target_two = np.diag([1.0, 2.0])
    full_optimum = 0.5 * (target_one + target_two)
    rank_optimum = np.diag([2.0, 0.0])
    online = np.diag([0.0, 1.0])

    def risk(target: np.ndarray, point: np.ndarray) -> float:
        return float(0.5 * np.sum((point - target) ** 2))

    targets = (target_one, target_two)
    total_gap = 0.5 * sum(risk(target, online) - risk(target, target) for target in targets)
    conflict = 0.5 * sum(risk(target, full_optimum) - risk(target, target) for target in targets)
    capacity = 0.5 * sum(risk(target, rank_optimum) - risk(target, full_optimum) for target in targets)
    online_gap = 0.5 * sum(risk(target, online) - risk(target, rank_optimum) for target in targets)
    accounting_error = abs(total_gap - conflict - capacity - online_gap)
    _add_gate(gates, "decomposition_identity", total_gap, conflict + capacity + online_gap, accounting_error <= EXACT_ATOL, EXACT_ATOL, error=accounting_error)
    _add_gate(gates, "decomposition_total_frozen_value", total_gap, 3.0, abs(total_gap - 3.0) <= EXACT_ATOL, EXACT_ATOL)
    _add_gate(gates, "decomposition_conflict_frozen_value", conflict, 1.0, abs(conflict - 1.0) <= EXACT_ATOL, EXACT_ATOL)
    _add_gate(gates, "decomposition_capacity_frozen_value", capacity, 0.5, abs(capacity - 0.5) <= EXACT_ATOL, EXACT_ATOL)
    _add_gate(gates, "decomposition_online_frozen_value", online_gap, 1.5, abs(online_gap - 1.5) <= EXACT_ATOL, EXACT_ATOL)
    _add_gate(gates, "decomposition_conflict_nonnegative", conflict, 0.0, conflict >= -EXACT_ATOL, EXACT_ATOL, relation="ge")
    _add_gate(gates, "decomposition_capacity_nonnegative", capacity, 0.0, capacity >= -EXACT_ATOL, EXACT_ATOL, relation="ge")
    _add_gate(gates, "decomposition_online_nonnegative", online_gap, 0.0, online_gap >= -EXACT_ATOL, EXACT_ATOL, relation="ge")
    full_gradient_norm = float(np.linalg.norm(full_optimum - 0.5 * (target_one + target_two)))
    rank_residual = float(0.5 * np.sum((rank_optimum - full_optimum) ** 2))
    _add_gate(gates, "decomposition_full_optimum_stationarity", full_gradient_norm, 0.0, full_gradient_norm <= EXACT_ATOL, EXACT_ATOL)
    _add_gate(gates, "decomposition_rank_optimum_membership", _rank(rank_optimum), 1, _rank(rank_optimum) <= 1, 0.0, relation="le")
    _add_gate(gates, "decomposition_online_membership", _rank(online), 1, _rank(online) <= 1, 0.0, relation="le")
    _add_gate(gates, "decomposition_rank_optimal_tail", rank_residual, 0.5, abs(rank_residual - 0.5) <= EXACT_ATOL, EXACT_ATOL)

    # C5/C9: weighted SVD under a fixed Kronecker quadratic.
    cout, cin, cout_half, cin_half = _spectral_factors()
    left = _fixed_orthogonal(5, [(0, 1, 0.8, 0.6), (2, 3, 0.6, -0.8), (3, 4, 0.8, 0.6)])
    right = _fixed_orthogonal(4, [(0, 1, 0.6, 0.8), (1, 2, 0.8, -0.6), (2, 3, 0.8, 0.6)])
    singular_values = np.array([4.0, 2.0, 1.0, 0.25])
    y_star = left[:, :4] @ np.diag(singular_values) @ right.T
    x_star = np.linalg.solve(cout_half, y_star) @ np.linalg.inv(cin_half)
    measured_losses = []
    spectral_rows = []
    euclidean_gaps = []
    for rank in range(5):
        left_svd, values, right_svd = np.linalg.svd(y_star, full_matrices=False)
        y_rank = left_svd[:, :rank] @ np.diag(values[:rank]) @ right_svd[:rank, :] if rank else np.zeros_like(y_star)
        x_rank = np.linalg.solve(cout_half, y_rank) @ np.linalg.inv(cin_half)
        weighted_residual = cout_half @ (x_rank - x_star) @ cin_half
        measured = float(0.5 * np.sum(weighted_residual**2))
        expected = float(0.5 * np.sum(singular_values[rank:] ** 2))
        ok, error, tolerance = _close(measured, expected, EXACT_ATOL)
        _add_gate(gates, f"spectral_tail_rank_{rank}", measured, expected, ok, tolerance, error=error)
        numerical_rank = _rank(x_rank)
        _add_gate(gates, f"spectral_rank_bound_{rank}", numerical_rank, rank, numerical_rank <= rank, 0.0, relation="le")
        measured_losses.append(measured)
        spectral_rows.append({"rank": rank, "singular_values": values, "measured": measured, "expected": expected, "numerical_rank": numerical_rank})

        eu_left, eu_values, eu_right = np.linalg.svd(x_star, full_matrices=False)
        eu_rank = eu_left[:, :rank] @ np.diag(eu_values[:rank]) @ eu_right[:rank, :] if rank else np.zeros_like(x_star)
        eu_residual = cout_half @ (eu_rank - x_star) @ cin_half
        euclidean_loss = float(0.5 * np.sum(eu_residual**2))
        euclidean_gaps.append(euclidean_loss - measured)
        _add_gate(gates, f"weighted_svd_not_worse_than_euclidean_{rank}", measured, euclidean_loss, measured <= euclidean_loss + EXACT_ATOL, EXACT_ATOL, relation="le")

        if rank:
            # Directly score deterministic rank-r competitors in whitened space.
            # Use the frozen construction, not SVD vectors whose signs may
            # differ across LAPACK implementations.
            base_a = left[:, :rank] @ np.diag(np.sqrt(singular_values[:rank]))
            base_b = right[:, :rank] @ np.diag(np.sqrt(singular_values[:rank]))
            perturb_a = np.sin(np.arange(5 * rank, dtype=float).reshape(5, rank) + 1.0)
            perturb_b = np.cos(np.arange(4 * rank, dtype=float).reshape(4, rank) + 0.5)
            for epsilon in (0.01, 0.05, 0.2):
                candidate_y = (base_a + epsilon * perturb_a) @ (base_b + epsilon * perturb_b).T
                candidate_x = np.linalg.solve(cout_half, candidate_y) @ np.linalg.inv(cin_half)
                candidate_residual = cout_half @ (candidate_x - x_star) @ cin_half
                candidate_loss = float(0.5 * np.sum(candidate_residual**2))
                _add_gate(gates, f"rank_competitor_{rank}_{epsilon:g}", candidate_loss - measured, 0.0, candidate_loss + EXACT_ATOL >= measured, EXACT_ATOL, relation="ge")
    singular_value_error = float(np.max(np.abs(np.linalg.svd(y_star, compute_uv=False) - singular_values)))
    _add_gate(gates, "spectral_singular_values", singular_value_error, 0.0, singular_value_error <= EXACT_ATOL, EXACT_ATOL)
    _add_gate(gates, "finite_budget_lower_bound", measured_losses[1], 1.0, measured_losses[1] >= 1.0 - EXACT_ATOL, EXACT_ATOL, relation="ge")
    _add_gate(gates, "weighted_strict_control", max(euclidean_gaps), 1e-6, max(euclidean_gaps) >= 1e-6, 0.0, relation="ge")

    layer_values = [np.array([4.0, 1.0]), np.array([3.0, 0.5]), np.array([2.0, 1.5])]
    allocations = []
    for allocation in itertools.product(range(3), repeat=3):
        if sum(allocation) <= 3:
            residual = 0.5 * sum(float(np.sum(values[rank:] ** 2)) for values, rank in zip(layer_values, allocation))
            allocations.append((allocation, residual))
    best_allocation, best_residual = min(allocations, key=lambda item: item[1])
    _add_gate(gates, "global_layer_allocation_residual", best_residual, 1.75, abs(best_residual - 1.75) <= EXACT_ATOL, EXACT_ATOL)
    _add_gate(gates, "global_layer_allocation_top_values", best_allocation, (1, 1, 1), best_allocation == (1, 1, 1), 0.0)

    innovation_ranks = []
    innovation_tails = []
    running = np.zeros_like(y_star)
    for index, value in enumerate(singular_values):
        running = running + value * np.outer(left[:, index], right[:, index])
        innovation_ranks.append(_rank(running))
        running_values = np.linalg.svd(running, compute_uv=False)
        measured_tail = float(0.5 * np.sum(running_values[2:] ** 2))
        expected_tail = float(0.5 * np.sum(singular_values[2 : index + 1] ** 2))
        innovation_tails.append(measured_tail)
        _add_gate(
            gates,
            f"known_innovation_rank2_tail_{index + 1}",
            measured_tail,
            expected_tail,
            abs(measured_tail - expected_tail) <= EXACT_ATOL,
            EXACT_ATOL,
        )
    _add_gate(gates, "known_innovation_rank_growth", innovation_ranks, [1, 2, 3, 4], innovation_ranks == [1, 2, 3, 4], 0.0)
    trace.update(
        {
            "decomposition": {"total_gap": total_gap, "conflict": conflict, "capacity": capacity, "online": online_gap},
            "spectral": spectral_rows,
            "euclidean_gaps": euclidean_gaps,
            "global_allocation": {"best": best_allocation, "residual": best_residual},
            "innovation_ranks": innovation_ranks,
            "innovation_tails": innovation_tails,
        }
    )
    return _record_case("decomposition_spectral_capacity", gates, trace)


def _sensitive_projector(matrix: np.ndarray, threshold: float) -> np.ndarray:
    matrix = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    operator_norm = float(np.max(values)) if values.size else 0.0
    if operator_norm <= 0.0:
        return np.zeros_like(matrix)
    selected = values > threshold * operator_norm
    if not np.any(selected):
        return np.zeros_like(matrix)
    basis = vectors[:, selected]
    return basis @ basis.T


def _intersection_projector(projectors: list[np.ndarray], dimension: int) -> np.ndarray:
    rows = []
    for projector in projectors:
        values, vectors = np.linalg.eigh((projector + projector.T) / 2.0)
        active = values > 0.5
        if np.any(active):
            rows.append(vectors[:, active].T)
    stacked = np.vstack(rows) if rows else np.empty((0, dimension))
    return _null_projector(stacked)


def check_safe_space() -> tuple[dict[str, Any], dict[str, Any]]:
    dimension = 6
    eye = np.eye(dimension)
    e1, e2, e3, e4, e5, e6 = eye
    directions: list[np.ndarray | None] = [
        e1,
        e2,
        (e3 + e4) / np.sqrt(2.0),
        e2,
        None,
        (e3 - e4) / np.sqrt(2.0),
    ]
    weights = [1.0, 0.5, 2.0, 3.0, 1.0, 1.5]
    expected_nullities = [6, 5, 4, 3, 3, 3, 2]
    gates: list[dict[str, Any]] = []
    trace: dict[str, Any] = {"ambient": [], "dictionary": []}

    accumulated = np.zeros((dimension, dimension))
    sensitivity_rows: list[np.ndarray] = []
    previous_projector = np.eye(dimension)
    observed_nullities = [dimension]
    for step, (direction, weight) in enumerate(zip(directions, weights), start=1):
        if direction is not None:
            accumulated += weight * np.outer(direction, direction)
            sensitivity_rows.append(np.sqrt(weight) * direction)
        cumulative_projector = _safe_projector(accumulated)
        stacked = np.asarray(sensitivity_rows).reshape((-1, dimension)) if sensitivity_rows else np.empty((0, dimension))
        independent_projector = _null_projector(stacked)
        nullity = int(round(np.trace(cumulative_projector)))
        observed_nullities.append(nullity)
        containment = float(np.linalg.norm((np.eye(dimension) - previous_projector) @ cumulative_projector, ord="fro"))
        difference = float(np.linalg.norm(cumulative_projector - independent_projector, ord="fro"))
        _add_gate(gates, f"ambient_projector_match_{step}", difference, 0.0, difference <= PROJECTOR_ATOL, PROJECTOR_ATOL, relation="le")
        _add_gate(gates, f"ambient_nesting_{step}", containment, 0.0, containment <= PROJECTOR_ATOL, PROJECTOR_ATOL, relation="le")
        trace["ambient"].append({"step": step, "nullity": nullity, "containment": containment, "projector": cumulative_projector})
        previous_projector = cumulative_projector
    _add_gate(gates, "ambient_expected_nullities", observed_nullities, expected_nullities, observed_nullities == expected_nullities, 0.0)
    _add_gate(gates, "ambient_repeated_direction_noop", observed_nullities[4], observed_nullities[3], observed_nullities[4] == observed_nullities[3], 0.0)
    _add_gate(gates, "ambient_zero_matrix_noop", observed_nullities[5], observed_nullities[4], observed_nullities[5] == observed_nullities[4], 0.0)

    dictionary = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ],
        dtype=float,
    )
    dictionary_rank = _rank(dictionary)
    _add_gate(gates, "dictionary_full_column_rank", dictionary_rank, 4, dictionary_rank == 4, 0.0)
    coefficient_expected = [4, 3, 2, 1, 1, 1, 0]
    coefficient_nullities = [4]
    accumulated = np.zeros((dimension, dimension))
    sensitivity_rows = []
    previous_projector = np.eye(4)
    for step, (direction, weight) in enumerate(zip(directions, weights), start=1):
        if direction is not None:
            accumulated += weight * np.outer(direction, direction)
            sensitivity_rows.append(np.sqrt(weight) * direction)
        gram = dictionary.T @ accumulated @ dictionary
        coefficient_projector = _safe_projector(gram)
        preimage_rows = np.asarray(sensitivity_rows) @ dictionary if sensitivity_rows else np.empty((0, dictionary.shape[1]))
        independent_preimage = _null_projector(preimage_rows)
        nullity = int(round(np.trace(coefficient_projector)))
        coefficient_nullities.append(nullity)
        containment = float(np.linalg.norm((np.eye(4) - previous_projector) @ coefficient_projector, ord="fro"))
        difference = float(np.linalg.norm(coefficient_projector - independent_preimage, ord="fro"))
        _add_gate(gates, f"coefficient_projector_match_{step}", difference, 0.0, difference <= PROJECTOR_ATOL, PROJECTOR_ATOL, relation="le")
        _add_gate(gates, f"coefficient_nesting_{step}", containment, 0.0, containment <= PROJECTOR_ATOL, PROJECTOR_ATOL, relation="le")
        trace["dictionary"].append({"step": step, "nullity": nullity, "containment": containment})
        previous_projector = coefficient_projector
    _add_gate(gates, "coefficient_expected_nullities", coefficient_nullities, coefficient_expected, coefficient_nullities == coefficient_expected, 0.0)

    rotation = _fixed_orthogonal(6, [(0, 1, 0.8, 0.6), (1, 2, 0.6, -0.8), (3, 4, 0.8, 0.6)])
    base = rotation @ np.diag([10.0, 2.0, 0.1, 0.0, 0.0, 0.0]) @ rotation.T
    threshold = 0.15
    base_sensitive = _sensitive_projector(base, threshold)
    scale_errors = []
    for scale in (1e-6, 1e6):
        scaled_sensitive = _sensitive_projector(scale * base, threshold)
        error = float(np.linalg.norm(scaled_sensitive - base_sensitive, ord="fro"))
        scale_errors.append(error)
        _add_gate(gates, f"approximate_scale_invariance_{scale:g}", error, 0.0, error <= PROJECTOR_ATOL, PROJECTOR_ATOL, relation="le")
    repeated_sensitive = _sensitive_projector(7.0 * base, threshold)
    _add_gate(gates, "approximate_repeated_span_noop", float(np.linalg.norm(repeated_sensitive - base_sensitive, ord="fro")), 0.0, np.linalg.norm(repeated_sensitive - base_sensitive, ord="fro") <= PROJECTOR_ATOL, PROJECTOR_ATOL, relation="le")
    new_sensitive = _sensitive_projector(4.0 * np.outer(e6, e6), threshold)
    contracted = _intersection_projector([base_sensitive, new_sensitive], dimension)
    expected_trace = float(dimension - np.trace(base_sensitive) - 1.0)
    _add_gate(gates, "approximate_new_span_contracts", float(np.trace(contracted)), expected_trace, abs(float(np.trace(contracted)) - expected_trace) <= PROJECTOR_ATOL, PROJECTOR_ATOL)
    eigenvalues = np.array([10.0, 2.0, 0.1, 0.0, 0.0, 0.0])
    threshold_margin = min(abs(eigenvalues[1] / eigenvalues[0] - threshold), abs(eigenvalues[2] / eigenvalues[0] - threshold))
    _add_gate(gates, "approximate_threshold_margin", threshold_margin, 0.04, threshold_margin >= 0.04, 0.0, relation="ge")

    signed_sum = np.array([[1.0]]) + np.array([[-1.0]])
    signed_guard = _rank(np.array([[1.0]])) == 1 and _rank(signed_sum) == 0
    _add_gate(gates, "non_psd_signed_accumulation_guard", signed_guard, True, signed_guard, 0.0, "negative weights are outside the PSD/additive assumption")

    gamma = 1.0
    early_curvature = np.diag([1.0, 0.0, 0.0])
    early_gain = np.array([1.0, 1.0, 0.0])
    null_component = _null_projector(early_curvature) @ early_gain
    free_update = gamma * null_component / (early_gain @ null_component)
    free_price = float(0.5 * free_update @ early_curvature @ free_update)
    _add_gate(gates, "useful_update_zero_price", free_price, 0.0, abs(free_price) <= EXACT_ATOL, EXACT_ATOL)
    _add_gate(gates, "useful_update_free_gain", float(early_gain @ free_update), gamma, abs(float(early_gain @ free_update) - gamma) <= EXACT_ATOL, EXACT_ATOL)
    full_curvature = np.diag([1.0, 2.0, 4.0])
    full_gain = np.array([1.0, 1.0, 0.0])
    denominator = float(full_gain @ np.linalg.pinv(full_curvature) @ full_gain)
    expected_price = gamma**2 / (2.0 * denominator)
    positive_update = gamma * np.linalg.pinv(full_curvature) @ full_gain / denominator
    measured_price = float(0.5 * positive_update @ full_curvature @ positive_update)
    _add_gate(gates, "useful_update_positive_price", measured_price, expected_price, abs(measured_price - expected_price) <= EXACT_ATOL, EXACT_ATOL)
    _add_gate(gates, "useful_update_positive_gain", float(full_gain @ positive_update), gamma, abs(float(full_gain @ positive_update) - gamma) <= EXACT_ATOL, EXACT_ATOL)
    trace.update({"nullities": observed_nullities, "coefficient_nullities": coefficient_nullities, "scale_errors": scale_errors, "price": {"free": free_price, "positive": expected_price}})
    return _record_case("safe_space_and_price", gates, trace)


def _support_utility(matrix: np.ndarray, vector: np.ndarray, support: tuple[int, ...]) -> float:
    if not support:
        return 0.0
    indices = np.asarray(support, dtype=int)
    submatrix = matrix[np.ix_(indices, indices)]
    subvector = vector[indices]
    return float(subvector @ np.linalg.solve(submatrix, subvector))


def check_block_allocation() -> tuple[dict[str, Any], dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    trace: dict[str, Any] = {"rho": {}}
    supports = [support for size in (1, 2) for support in itertools.combinations(range(3), size)]
    vector = np.array([1.0, 1.0, 0.9])
    diagonal = np.eye(3)
    diagonal_scores = vector**2
    selected = tuple(np.argsort(-diagonal_scores, kind="stable")[:2].tolist())
    diagonal_utilities = {support: _support_utility(diagonal, vector, support) for support in supports}
    best_diagonal = max(diagonal_utilities.values())
    _add_gate(gates, "block_diagonal_topk_support", selected, (0, 1), selected == (0, 1), 0.0)
    _add_gate(gates, "block_diagonal_topk_optimal", diagonal_utilities[selected], best_diagonal, abs(diagonal_utilities[selected] - best_diagonal) <= EXACT_ATOL, EXACT_ATOL)

    for rho in (0.0, 0.2, 0.9):
        matrix = np.array([[1.0, rho, 0.0], [rho, 1.0, 0.0], [0.0, 0.0, 1.0]])
        utilities = {support: _support_utility(matrix, vector, support) for support in supports}
        costs = {support: 1.0 / (2.0 * utility) for support, utility in utilities.items()}
        best_support = min(costs, key=costs.get)
        selected_cost = costs[selected]
        optimal_cost = costs[best_support]
        eigenvalues = np.linalg.eigvalsh(matrix)
        delta = float(np.max(np.abs(eigenvalues - 1.0)))
        factor = (1.0 + delta) / (1.0 - delta) if delta < 1.0 else math.inf
        ratio = selected_cost / optimal_cost
        trace["rho"][str(rho)] = {"selected": selected, "best_support": best_support, "utilities": utilities, "costs": costs, "eigenvalues": eigenvalues, "delta": delta, "factor": factor, "ratio": ratio}
        _add_gate(gates, f"cross_block_rho{rho:g}_positive_definite", float(np.min(eigenvalues)), EXACT_ATOL, float(np.min(eigenvalues)) >= EXACT_ATOL, 0.0, relation="ge")
        _add_gate(gates, f"cross_block_rho{rho:g}_computed_delta", delta, rho, abs(delta - rho) <= EXACT_ATOL, EXACT_ATOL)
        if rho == 0.0:
            _add_gate(gates, "block_control_rho0_optimal", selected_cost, optimal_cost, abs(selected_cost - optimal_cost) <= EXACT_ATOL, EXACT_ATOL)
        else:
            _add_gate(gates, f"cross_block_rho{rho:g}_strictly_better_exists", selected_cost - optimal_cost, EXACT_ATOL, selected_cost - optimal_cost >= EXACT_ATOL, 0.0, relation="ge")
            _add_gate(gates, f"cross_block_rho{rho:g}_spectral_factor", ratio, factor, ratio <= factor + EXACT_ATOL, EXACT_ATOL, relation="le")
        if rho == 0.9:
            _add_gate(gates, "cross_block_rho09_selected_utility", utilities[(0, 1)], 20.0 / 19.0, abs(utilities[(0, 1)] - 20.0 / 19.0) <= EXACT_ATOL, EXACT_ATOL)
            _add_gate(gates, "cross_block_rho09_alternative_utility", utilities[(0, 2)], 1.81, abs(utilities[(0, 2)] - 1.81) <= EXACT_ATOL, EXACT_ATOL)
            _add_gate(gates, "cross_block_rho09_selected_cost", selected_cost, 19.0 / 40.0, abs(selected_cost - 19.0 / 40.0) <= EXACT_ATOL, EXACT_ATOL)
            _add_gate(gates, "cross_block_rho09_alternative_cost", optimal_cost, 50.0 / 181.0, abs(optimal_cost - 50.0 / 181.0) <= EXACT_ATOL, EXACT_ATOL)
            _add_gate(gates, "cross_block_rho09_delta", delta, 0.9, abs(delta - 0.9) <= EXACT_ATOL, EXACT_ATOL)
            _add_gate(gates, "cross_block_rho09_factor", factor, 19.0, abs(factor - 19.0) <= EXACT_ATOL, EXACT_ATOL)
            _add_gate(gates, "cross_block_rho09_tie", utilities[(0, 2)], utilities[(1, 2)], abs(utilities[(0, 2)] - utilities[(1, 2)]) <= EXACT_ATOL, EXACT_ATOL)
    return _record_case("block_allocation", gates, trace)


def run_all_checks() -> dict[str, Any]:
    case_results = [
        check_taylor(),
        check_sequential(),
        check_decomposition_and_spectral(),
        check_safe_space(),
        check_block_allocation(),
    ]
    case_summaries = [result[0] for result in case_results]
    traces = {result[1]["case"]: result[1] for result in case_results}
    gates = [gate for case in case_summaries for gate in case["gates"]]
    failed = [gate for gate in gates if not gate["passed"]]
    return {
        "schema_version": 1,
        "stage": "phase1_theory_falsification",
        "claim_boundary": "conditional_theory_only",
        "required_gate_count": len(gates),
        "failed_gate_count": len(failed),
        "all_required_pass": not failed,
        "cases": [
            {
                "case": case["case"],
                "gate_count": len(case["gates"]),
                "failed_count": sum(not gate["passed"] for gate in case["gates"]),
            }
            for case in case_summaries
        ],
        "gates": gates,
        "traces": traces,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_as_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_run(output_dir: Path, run_role: str) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_all_checks()
    summary = {key: value for key, value in result.items() if key != "traces"}
    summary["headline"] = {
        "decomposition_total_gap": result["traces"]["decomposition_spectral_capacity"]["decomposition"]["total_gap"],
        "innovation_expected": result["traces"]["sequential_innovations"]["innovation_expected"],
        "cross_block_rho09_ratio": result["traces"]["block_allocation"]["rho"]["0.9"]["ratio"],
    }
    traces_path = output_dir / "traces.json"
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "per_case.csv"
    _write_json(traces_path, result["traces"])
    _write_json(summary_path, summary)
    fields = ["case", "name", "relation", "observed", "expected", "passed", "tolerance", "abs_error", "violation", "notes"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for gate in result["gates"]:
            writer.writerow({field: _as_json(gate.get(field)) for field in fields})

    output_hashes = {path.name: _sha256(path) for path in (traces_path, summary_path, csv_path)}
    config_buffer = io.StringIO()
    with redirect_stdout(config_buffer):
        np.show_config()
    manifest = {
        "schema_version": 1,
        "stage": "phase1_theory_falsification",
        "run_role": run_role,
        "source_sha256": _sha256(SOURCE_PATH),
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "test_sha256": _sha256(TEST_PATH),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_config": config_buffer.getvalue(),
        "command": " ".join(sys.argv),
        "resource_ledger": {
            "gpu_used": False,
            "optimizer_state_bytes": 0,
            "replay_bytes": 0,
            "external_model_bytes": 0,
        },
        "output_hashes": output_hashes,
        "all_required_pass": result["all_required_pass"],
        "claim_boundary": "conditional_theory_only",
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {"summary": summary, "manifest": manifest, "output_dir": str(output_dir)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-role", default="primary_frozen")
    args = parser.parse_args(argv)
    try:
        result = write_run(args.output_dir, args.run_role)
    except Exception as exc:  # pragma: no cover - CLI failure path
        print(f"phase1 run failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_dir": result["output_dir"],
                "all_required_pass": result["summary"]["all_required_pass"],
                "failed_gate_count": result["summary"]["failed_gate_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["summary"]["all_required_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
