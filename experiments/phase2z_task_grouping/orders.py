"""RUNBOOK 3B: task-order permutations for the order-sensitivity study.

The paper's 2x2 is measured on exactly one stream order (Order-4 / O-LoRA long
sequence). Continual-learning results are known to be order-sensitive, so this
module defines additional orders over the SAME 15 tasks. Only the TRAINING
sequence changes; the eligibility filter, splits, offsets, and scoring are
order-independent (they act per task on the final model), so any 2x2 difference
across orders is a pure order effect.

Orders are defined as explicit name permutations so a run is fully reproducible
from its `--order` string alone. `canonical` is ORDER4_TASKS as-is. `reverse` is
its reversal. `shuffleA`/`shuffleB` are two fixed permutations (written out, not
RNG-generated, so no seed dependence and no Math.random-style nondeterminism).
Every order is asserted at import time to be a permutation of the canonical set.
"""

from __future__ import annotations

from experiments.phase2i_anchored_cvar.order4_data import ORDER4_TASK_NAMES

CANONICAL = list(ORDER4_TASK_NAMES)

# Two fixed permutations chosen to interleave families differently from the
# canonical order (which clusters NLI early, sentiment/topic late). Written out
# explicitly for reproducibility.
_SHUFFLE_A = [
    "DBpedia", "WiC", "IMDB", "MNLI", "Yahoo", "COPA", "SST-2", "RTE",
    "AGNews", "BoolQA", "Amazon", "CB", "MultiRC", "Yelp", "QQP",
]
_SHUFFLE_B = [
    "QQP", "AGNews", "CB", "Yelp", "MNLI", "SST-2", "MultiRC", "DBpedia",
    "WiC", "BoolQA", "IMDB", "COPA", "Yahoo", "RTE", "Amazon",
]

ORDERS = {
    "canonical": CANONICAL,
    "reverse": list(reversed(CANONICAL)),
    "shuffleA": _SHUFFLE_A,
    "shuffleB": _SHUFFLE_B,
}

# Fail loudly if any order is not a permutation of the canonical task set.
_canon = set(CANONICAL)
for _name, _seq in ORDERS.items():
    if sorted(_seq) != sorted(CANONICAL):
        missing = _canon - set(_seq)
        extra = set(_seq) - _canon
        raise ValueError(
            f"order '{_name}' is not a permutation of the 15 tasks: "
            f"missing={sorted(missing)} extra={sorted(extra)}")


def order_index(order_name):
    """Return {task_name: position} for the requested order (for sorting)."""
    if order_name not in ORDERS:
        raise KeyError(f"unknown order '{order_name}'; choices={sorted(ORDERS)}")
    return {name: i for i, name in enumerate(ORDERS[order_name])}


def reorder_task_list(task_list, order_name):
    """Sort an eligible-task list [(TaskSpec, rarest), ...] by the given order.

    Tasks absent from the order string keep canonical relative position after
    the named ones (defensive; every order here is complete so this is a no-op).
    """
    idx = order_index(order_name)
    fallback = len(idx)
    return sorted(task_list, key=lambda tr: idx.get(tr[0].name, fallback))
