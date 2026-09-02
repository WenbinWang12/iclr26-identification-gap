"""Phase-2Y: Adaptive rank allocation conditioned on measured forgetting.

KEY INSIGHT from Phase-2K/2L analysis:
- Forgetting is task-heterogeneous: MNLI +17.5pp residual, RTE +11.5, BoolQA +6.9
- Offset repairs only 45% of large forgetting (>8pp stratum)
- The residual is representation-level → needs rank, not just offset

HYPOTHESIS:
- Phase-2W showed uniform rank scaling is flat (±1.30pp noise)
- But allocating rank CONDITIONAL on measured forgetting might work
- Budget-neutral: Σr_t = 120 (same as 15 tasks × r=8 baseline)

METHOD:
- At task t, measure residual forgetting on previous tasks using PSR examples
- Allocate rank proportional to forgetting: tasks with deep forgetting get more rank
- Constraint: total rank budget = 120 (redistribute, don't grow)

CHALLENGE:
- PEFT LoRA r is per-module, not per-task
- Need task-conditional routing OR progressive rank growth within fixed budget
- This module implements the rank allocation policy; LoRA wiring TBD
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional


def measure_residual_forgetting(
    task_name: str,
    logits: np.ndarray,
    labels: np.ndarray,
    per_task_offset: Optional[np.ndarray] = None,
) -> float:
    """Measure residual forgetting after best offset is applied.

    Returns balanced accuracy loss relative to some baseline (e.g., post-train accuracy).
    This is the "deep forgetting" that offsets cannot repair.

    Args:
        task_name: Task identifier
        logits: [n, K] verbalizer logits
        labels: [n] true labels
        per_task_offset: [K] optimal offset for this task (if None, compute it)

    Returns:
        Residual forgetting in pp (positive = forgetting, negative = gain)
    """
    from experiments.phase2j_offset_conflict.offsets import (
        balanced_accuracy, fit_offset, gauge_fix
    )

    # Baseline: raw accuracy (no offset)
    acc_raw = balanced_accuracy(logits, labels, None)

    # Best offset
    if per_task_offset is None:
        per_task_offset = gauge_fix(fit_offset(logits, labels))

    acc_offset = balanced_accuracy(logits, labels, per_task_offset)

    # Residual: what offset cannot repair
    # (In practice, this would be measured against post-train accuracy, not raw)
    residual = acc_raw - acc_offset  # Simplified; proper version needs baseline

    return float(residual * 100.0)  # Return in percentage points


def allocate_ranks(
    forgetting_per_task: Dict[str, float],
    total_budget: int = 120,
    base_rank: int = 4,
    sensitivity: float = 0.5,
) -> Dict[str, int]:
    """Allocate rank budget proportional to measured forgetting.

    Args:
        forgetting_per_task: {task_name: residual_forgetting_pp}
        total_budget: Total rank units to allocate (default 120 = 15 tasks × 8)
        base_rank: Minimum rank per task (default 4)
        sensitivity: How much to weight forgetting (0=uniform, 1=proportional)

    Returns:
        {task_name: allocated_rank}

    Constraint: sum(allocated_rank) == total_budget
    """
    tasks = sorted(forgetting_per_task.keys())
    n_tasks = len(tasks)

    if n_tasks == 0:
        return {}

    # Ensure base_rank × n_tasks <= total_budget
    min_budget = base_rank * n_tasks
    if min_budget > total_budget:
        raise ValueError(
            f"Cannot allocate base_rank={base_rank} to {n_tasks} tasks "
            f"within budget={total_budget} (needs {min_budget})"
        )

    # Uniform allocation (baseline)
    if sensitivity == 0.0:
        uniform = total_budget // n_tasks
        allocated = {t: uniform for t in tasks}
        # Handle rounding remainder
        remainder = total_budget - sum(allocated.values())
        for i in range(remainder):
            allocated[tasks[i]] += 1
        return allocated

    # Forgetting-conditional allocation
    forgetting = np.array([forgetting_per_task[t] for t in tasks])

    # Normalize: shift so min=0, then scale
    forgetting_shifted = forgetting - forgetting.min()
    if forgetting_shifted.max() == 0:
        # All tasks have same forgetting → uniform
        return allocate_ranks(forgetting_per_task, total_budget, base_rank, sensitivity=0.0)

    forgetting_scaled = forgetting_shifted / forgetting_shifted.max()

    # Budget allocation: base + (total - base*n) * (forgetting_weight)
    flexible_budget = total_budget - min_budget
    weights = sensitivity * forgetting_scaled + (1 - sensitivity) * np.ones(n_tasks)
    weights = weights / weights.sum()

    extra = (flexible_budget * weights).astype(int)
    allocated_ranks = base_rank + extra

    # Adjust for rounding to hit exact budget
    current_sum = allocated_ranks.sum()
    diff = total_budget - current_sum

    if diff > 0:
        # Add to tasks with highest fractional remainder
        fractional = flexible_budget * weights - extra
        top_indices = np.argsort(-fractional)[:diff]
        allocated_ranks[top_indices] += 1
    elif diff < 0:
        # Remove from tasks with lowest fractional remainder
        fractional = flexible_budget * weights - extra
        bottom_indices = np.argsort(fractional)[:-diff]
        allocated_ranks[bottom_indices] -= 1

    # Ensure no task drops below base_rank
    allocated_ranks = np.maximum(allocated_ranks, base_rank)

    # Final adjustment to hit budget exactly
    while allocated_ranks.sum() > total_budget:
        # Remove from tasks above base_rank, prioritizing low forgetting
        above_base = allocated_ranks > base_rank
        if not above_base.any():
            break
        candidates = np.where(above_base)[0]
        victim = candidates[np.argmin(forgetting[candidates])]
        allocated_ranks[victim] -= 1

    return {t: int(allocated_ranks[i]) for i, t in enumerate(tasks)}


def validate_allocation(allocated: Dict[str, int], total_budget: int) -> None:
    """Assert allocation constraints."""
    total = sum(allocated.values())
    assert total == total_budget, f"Budget violation: {total} != {total_budget}"
    assert all(r > 0 for r in allocated.values()), "All ranks must be positive"


# Example usage and test
if __name__ == "__main__":
    # Simulate forgetting measurements from Phase-2K analysis
    # High forgetting: MNLI, RTE, BoolQA (>8pp residual)
    # Medium: others (2-8pp)
    # Low: remaining (<2pp)

    forgetting = {
        "MNLI": 17.5,
        "RTE": 11.5,
        "BoolQA": 6.9,
        "SST-2": 5.2,
        "IMDB": 4.8,
        "MultiRC": 3.1,
        "QQP": 2.5,
        "COPA": 1.9,
        "WiC": 1.2,
        "ReCoRD": 0.8,
        "WSC": 0.5,
        "CB": 0.3,
        "Headline": 0.2,
        "WG": 0.1,
        "Yelp": 0.0,
    }

    print("Forgetting-conditional rank allocation (Phase-2Y)")
    print("=" * 60)

    for sens in [0.0, 0.3, 0.5, 0.8]:
        allocated = allocate_ranks(forgetting, total_budget=120, base_rank=4, sensitivity=sens)
        validate_allocation(allocated, 120)

        print(f"\nSensitivity={sens:.1f}")
        print(f"  Total budget: {sum(allocated.values())}")
        print(f"  Top-3 tasks:")
        top3 = sorted(allocated.items(), key=lambda x: -x[1])[:3]
        for task, rank in top3:
            print(f"    {task:12s}: r={rank:2d} (forgetting={forgetting[task]:.1f}pp)")
        print(f"  Bottom-3 tasks:")
        bottom3 = sorted(allocated.items(), key=lambda x: x[1])[:3]
        for task, rank in bottom3:
            print(f"    {task:12s}: r={rank:2d} (forgetting={forgetting[task]:.1f}pp)")
