"""Deterministic fixed-capacity replay controllers for the Phase-2 bridge.

Only :class:`ReplayState` persists historical learner data across windows.
Selection plans and priority ordering records are transient controller staging
and are measured separately from the frozen 93,488-byte replay allocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cmp_to_key
import hashlib
import sys
from typing import Any, Sequence

import numpy as np


B_MAX = 352
T_MAX = 12
D = 32
REPLAY_BYTES = (
    B_MAX * D * np.dtype(np.float32).itemsize * 2
    + B_MAX * np.dtype(np.float64).itemsize
    + B_MAX * np.dtype(np.uint8).itemsize
    + T_MAX * np.dtype(np.uint64).itemsize * 2
    + np.dtype(np.uint64).itemsize * 2
)


@dataclass
class ReplayState:
    """The complete persistent learner-facing replay allocation."""

    x_slots: np.ndarray = field(default_factory=lambda: np.zeros((B_MAX, D), dtype=np.float32))
    y_slots: np.ndarray = field(default_factory=lambda: np.zeros((B_MAX, D), dtype=np.float32))
    priority_slots: np.ndarray = field(default_factory=lambda: np.ones(B_MAX, dtype=np.float64))
    occupied: np.ndarray = field(default_factory=lambda: np.zeros(B_MAX, dtype=np.uint8))
    window_counts: np.ndarray = field(default_factory=lambda: np.zeros(T_MAX, dtype=np.uint64))
    window_offsets: np.ndarray = field(default_factory=lambda: np.zeros(T_MAX, dtype=np.uint64))
    write_ptr: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.uint64))
    eligible_count: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.uint64))

    def __post_init__(self) -> None:
        expected = {
            "x_slots": ((B_MAX, D), np.float32),
            "y_slots": ((B_MAX, D), np.float32),
            "priority_slots": ((B_MAX,), np.float64),
            "occupied": ((B_MAX,), np.uint8),
            "window_counts": ((T_MAX,), np.uint64),
            "window_offsets": ((T_MAX,), np.uint64),
            "write_ptr": ((1,), np.uint64),
            "eligible_count": ((1,), np.uint64),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != dtype:
                raise TypeError(f"{name} must have shape {shape} and dtype {dtype}")
        if self.allocated_bytes != REPLAY_BYTES:
            raise AssertionError("fixed replay byte allocation changed")

    @property
    def allocated_bytes(self) -> int:
        return int(
            self.x_slots.nbytes
            + self.y_slots.nbytes
            + self.priority_slots.nbytes
            + self.occupied.nbytes
            + self.window_counts.nbytes
            + self.window_offsets.nbytes
            + self.write_ptr.nbytes
            + self.eligible_count.nbytes
        )

    @property
    def n_occupied(self) -> int:
        return int(np.count_nonzero(self.occupied))

    @property
    def occupied_slots(self) -> np.ndarray:
        return np.flatnonzero(self.occupied).astype(np.int64, copy=False)

    def clear(self) -> None:
        self.x_slots.fill(0)
        self.y_slots.fill(0)
        self.priority_slots.fill(1.0)
        self.occupied.fill(0)
        self.window_counts.fill(0)
        self.window_offsets.fill(0)
        self.write_ptr.fill(0)
        self.eligible_count.fill(0)


@dataclass(frozen=True)
class ReplayTransition:
    completed_window: int
    history_windows: int
    capacity: int
    eligible_count: int
    occupied_count: int
    unique_count: int
    duplicate_count: int
    replacement_count: int
    priority_draws: int
    comparison_count: int
    priority_tie_count: int
    selected_priority_order_sha256: str


def buffer_capacity(history_windows: int) -> int:
    if history_windows < 0:
        raise ValueError("history_windows must be non-negative")
    return 0 if history_windows == 0 else B_MAX


def clock_quotas(history_windows: int) -> np.ndarray:
    """Return the exact quota vector for windows 1..H."""

    if history_windows < 1 or history_windows > T_MAX - 1:
        raise ValueError("clock quota history must be in [1, 11]")
    base, remainder = divmod(B_MAX, history_windows)
    result = np.full(history_windows, base, dtype=np.uint64)
    if remainder:
        result[:remainder] += 1
    if int(result.sum()) != B_MAX:
        raise AssertionError("clock quotas do not sum to B_MAX")
    return result


def _deep_size(value: Any, seen: set[int] | None = None) -> int:
    """Conservative Python-container size for transient instrumentation."""

    seen = set() if seen is None else seen
    identifier = id(value)
    if identifier in seen:
        return 0
    seen.add(identifier)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_deep_size(key, seen) + _deep_size(item, seen) for key, item in value.items())
    elif isinstance(value, (tuple, list, set, frozenset)):
        size += sum(_deep_size(item, seen) for item in value)
    return int(size)


def _ordered_new(priorities: np.ndarray) -> tuple[list[int], int]:
    """Sort one new window by (priority, local_index) and count comparisons."""

    comparisons = 0

    def compare(left: int, right: int) -> int:
        nonlocal comparisons
        comparisons += 1
        left_key = (float(priorities[left]), int(left))
        right_key = (float(priorities[right]), int(right))
        return (left_key > right_key) - (left_key < right_key)

    return sorted(range(int(priorities.size)), key=cmp_to_key(compare)), comparisons


def _priority_order_sha256(priorities: Sequence[float], window_counts: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(priorities, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(window_counts, dtype="<u8").tobytes(order="C"))
    return digest.hexdigest()


class _ReplayController:
    def __init__(self, *, regime_code: int = 0, rank_code: int = 0) -> None:
        self.state = ReplayState()
        self.regime_code = int(regime_code)
        self.rank_code = int(rank_code)
        self.priority_draws = 0
        self.comparison_count = 0
        self.replacement_count = 0
        self.priority_tie_count = 0
        self.transient_records_peak = 0
        self.transient_python_bytes_peak = 0
        self.transient_numpy_bytes_peak = 0
        self.persistent_sidecar_records = 0
        self.persistent_sidecar_bytes = 0

    @property
    def n_occupied(self) -> int:
        return self.state.n_occupied

    @property
    def x_slots(self) -> np.ndarray:
        return self.state.x_slots

    @property
    def y_slots(self) -> np.ndarray:
        return self.state.y_slots

    def _draw_priorities(
        self,
        window: int,
        x: np.ndarray,
        y: np.ndarray,
        priority_rng: np.random.Generator,
        split_code: int,
    ) -> tuple[np.ndarray, list[int], int]:
        if window < 1 or window > T_MAX - 1:
            raise ValueError("only completed windows 1..11 can be inserted")
        if int(split_code) != 0:
            raise ValueError("only train split records may enter replay")
        if np.asarray(x).shape != (1024, D) or np.asarray(y).shape != (1024, D):
            raise ValueError("completed windows must contain exactly (1024, 32) x/y")
        if np.asarray(x).dtype != np.float32 or np.asarray(y).dtype != np.float32:
            raise TypeError("replay inputs must be the exact float32 training arrays")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise FloatingPointError("non-finite replay candidate")
        priorities = priority_rng.random(1024, dtype=np.float64)
        order, comparisons = _ordered_new(priorities)
        self.priority_draws += 1024
        self.transient_records_peak = max(self.transient_records_peak, len(order))
        self.transient_python_bytes_peak = max(
            self.transient_python_bytes_peak,
            _deep_size(order),
        )
        self.transient_numpy_bytes_peak = max(
            self.transient_numpy_bytes_peak,
            int(priorities.nbytes + B_MAX * np.dtype(np.float64).itemsize),
        )
        return priorities, order, comparisons

    def sample_indices(self, draw_rng: np.random.Generator, size: int = 64) -> np.ndarray | None:
        occupied = self.state.occupied_slots
        if occupied.size == 0:
            return None
        replace = bool(occupied.size < size)
        return np.asarray(
            draw_rng.choice(occupied, size=size, replace=replace, shuffle=True),
            dtype=np.int64,
        )

    def snapshot(self) -> dict[str, np.ndarray | int]:
        return {
            "x_slots": self.state.x_slots.copy(),
            "y_slots": self.state.y_slots.copy(),
            "priority_slots": self.state.priority_slots.copy(),
            "occupied": self.state.occupied.copy(),
            "window_counts": self.state.window_counts.copy(),
            "window_offsets": self.state.window_offsets.copy(),
            "write_ptr": self.state.write_ptr.copy(),
            "eligible_count": self.state.eligible_count.copy(),
            "priority_draws": self.priority_draws,
            "comparison_count": self.comparison_count,
            "replacement_count": self.replacement_count,
            "priority_tie_count": self.priority_tie_count,
        }


class ClockBalancedReplay(_ReplayController):
    """Window-stratified replay using only the fixed ReplayState arrays."""

    def insert_completed_window(
        self,
        window: int,
        x: np.ndarray,
        y: np.ndarray,
        priority_rng: np.random.Generator,
        *,
        split_code: int = 0,
    ) -> ReplayTransition:
        priorities, order, comparisons = self._draw_priorities(
            window, x, y, priority_rng, split_code
        )
        old_occupied = self.state.n_occupied
        old_counts = self.state.window_counts.copy()
        old_offsets = self.state.window_offsets.copy()
        quotas = clock_quotas(window)
        cursor = 0

        # Every existing quota is non-increasing as H grows.  Moving ranges in
        # ascending window/source order is therefore an overlap-safe left shift.
        for history_window in range(1, window):
            quota = int(quotas[history_window - 1])
            source = int(old_offsets[history_window - 1])
            if int(old_counts[history_window - 1]) < quota:
                raise AssertionError("clock quota increased for an existing window")
            for offset in range(quota):
                destination = cursor + offset
                source_slot = source + offset
                self.state.x_slots[destination] = self.state.x_slots[source_slot]
                self.state.y_slots[destination] = self.state.y_slots[source_slot]
                self.state.priority_slots[destination] = self.state.priority_slots[source_slot]
            cursor += quota

        new_quota = int(quotas[window - 1])
        selected_new = order[:new_quota]
        for offset, local_index in enumerate(selected_new, start=cursor):
            self.state.x_slots[offset] = x[local_index]
            self.state.y_slots[offset] = y[local_index]
            self.state.priority_slots[offset] = priorities[local_index]
        cursor += new_quota
        if cursor != B_MAX:
            raise AssertionError("clock insertion did not fill the fixed capacity")

        self.state.occupied.fill(1)
        self.state.window_counts.fill(0)
        self.state.window_offsets.fill(0)
        offset = 0
        for index, quota_value in enumerate(quotas.tolist()):
            self.state.window_counts[index] = int(quota_value)
            self.state.window_offsets[index] = offset
            offset += int(quota_value)
        self.state.write_ptr.fill(0)
        self.state.eligible_count[0] = 1024 * window

        replacement_count = 0 if old_occupied == 0 else old_occupied - int(np.sum(quotas[:-1]))
        tie_count = int(priorities.size - np.unique(priorities).size)
        self.comparison_count += comparisons
        self.replacement_count += replacement_count
        self.priority_tie_count += tie_count
        selected_priorities = self.state.priority_slots[:B_MAX].tolist()
        return ReplayTransition(
            completed_window=window,
            history_windows=window,
            capacity=B_MAX,
            eligible_count=1024 * window,
            occupied_count=self.state.n_occupied,
            unique_count=B_MAX,
            duplicate_count=0,
            replacement_count=replacement_count,
            priority_draws=1024,
            comparison_count=self.comparison_count,
            priority_tie_count=tie_count,
            selected_priority_order_sha256=_priority_order_sha256(
                selected_priorities, self.state.window_counts
            ),
        )


class ReservoirReplay(_ReplayController):
    """Global priority reservoir with stable exact-key tie handling."""

    def insert_completed_window(
        self,
        window: int,
        x: np.ndarray,
        y: np.ndarray,
        priority_rng: np.random.Generator,
        *,
        split_code: int = 0,
    ) -> ReplayTransition:
        priorities, new_order, comparisons = self._draw_priorities(
            window, x, y, priority_rng, split_code
        )
        old_count = self.state.n_occupied
        old_priority_values = self.state.priority_slots[:old_count].copy()
        combined_priorities = set(float(value) for value in priorities.tolist())
        combined_total = len(priorities) + old_count
        combined_priorities.update(float(value) for value in old_priority_values.tolist())
        tie_count = int(combined_total - len(combined_priorities))
        plan: list[tuple[str, int]] = []
        old_index = 0
        new_position = 0
        merge_comparisons = 0
        while len(plan) < B_MAX:
            if old_index >= old_count:
                plan.append(("new", int(new_order[new_position])))
                new_position += 1
                continue
            if new_position >= len(new_order):
                plan.append(("old", old_index))
                old_index += 1
                continue
            new_index = int(new_order[new_position])
            old_priority = float(old_priority_values[old_index])
            new_priority = float(priorities[new_index])
            merge_comparisons += 1
            # All old global keys have a smaller window field than every new
            # key.  Stable old-first equality is therefore the exact key rule.
            if old_priority <= new_priority:
                plan.append(("old", old_index))
                old_index += 1
            else:
                plan.append(("new", new_index))
                new_position += 1

        # A retained old record can only move right in the stable merge.  Copy
        # old sources in descending order, then populate all new destinations.
        for destination in range(B_MAX - 1, -1, -1):
            source_kind, source_index = plan[destination]
            if source_kind == "old":
                if destination < source_index:
                    raise AssertionError("reservoir merge attempted an unsafe left move")
                self.state.x_slots[destination] = self.state.x_slots[source_index]
                self.state.y_slots[destination] = self.state.y_slots[source_index]
                self.state.priority_slots[destination] = self.state.priority_slots[source_index]
        for destination, (source_kind, source_index) in enumerate(plan):
            if source_kind == "new":
                self.state.x_slots[destination] = x[source_index]
                self.state.y_slots[destination] = y[source_index]
                self.state.priority_slots[destination] = priorities[source_index]

        self.state.occupied.fill(1)
        self.state.window_counts.fill(0)
        self.state.window_offsets.fill(0)
        self.state.write_ptr.fill(0)
        self.state.eligible_count[0] = 1024 * window
        selected_new_count = sum(kind == "new" for kind, _ in plan)
        replacement_count = 0 if old_count == 0 else int(selected_new_count)
        comparisons += merge_comparisons
        self.comparison_count += comparisons
        self.replacement_count += replacement_count
        self.priority_tie_count += tie_count
        self.transient_records_peak = max(self.transient_records_peak, len(new_order) + len(plan))
        self.transient_numpy_bytes_peak = max(
            self.transient_numpy_bytes_peak,
            int(priorities.nbytes + old_priority_values.nbytes + B_MAX * np.dtype(np.float64).itemsize),
        )
        self.transient_python_bytes_peak = max(
            self.transient_python_bytes_peak,
            _deep_size((new_order, plan, combined_priorities)),
        )
        selected_priorities = self.state.priority_slots[:B_MAX].tolist()
        return ReplayTransition(
            completed_window=window,
            history_windows=window,
            capacity=B_MAX,
            eligible_count=1024 * window,
            occupied_count=self.state.n_occupied,
            unique_count=B_MAX,
            duplicate_count=0,
            replacement_count=replacement_count,
            priority_draws=1024,
            comparison_count=self.comparison_count,
            priority_tie_count=tie_count,
            selected_priority_order_sha256=_priority_order_sha256(
                selected_priorities, self.state.window_counts
            ),
        )


__all__ = [
    "B_MAX",
    "D",
    "REPLAY_BYTES",
    "T_MAX",
    "ClockBalancedReplay",
    "ReplayState",
    "ReplayTransition",
    "ReservoirReplay",
    "buffer_capacity",
    "clock_quotas",
]
