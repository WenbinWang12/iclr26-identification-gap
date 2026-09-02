"""Acquisition-conditioned T5-small Order-4 oracle-headroom panel.

This runner repairs the acquisition/consolidation confound in the original
five-arm feasibility panel.  At every task both arms first receive an identical
current-only acquisition phase.  A train-derived held-out audit then measures
acquisition, the historical adversary is refreshed, and an exposure-matched
consolidation phase runs.  Current anchors are admitted only afterwards.  The
two locked arms differ only in the historical replay adversary:

``uniform_er_rank4``
    Deployable uniform experience replay; its controller view contains no task
    ID.
``task_id_oracle``
    A diagnostic worst-task DRO adversary with privileged training task IDs.

Both arms retain the same uniform rank-four T5 q/v LoRA mask, the same number
of acquisition and consolidation optimizer steps, and the same record budget.
The official test files are loaded only for an exact 15-task Order-4 run with a
real T5-small checkpoint, after every acquisition guard passes, cross-arm
resource parity is revalidated, and both arms have durable
``TRAINING_COMPLETED`` sentinels.  Prefix, smoke, and tiny-random runs are
intrinsically test-sealed.  Test is then used once for final reporting, never
for selection, gating, replay weights, early stopping, or checkpoints.

The default acquisition schedule is LR 1e-3, at most 96 update examples per
class, one epoch except MultiRC (two).  QQP/BoolQA one-epoch and MultiRC
two-epoch choices reproduce the row-held-out rescue selections; they are a
frozen development schedule rather than independent strict confirmation.  This
runner uses normalized-question connected components for QQP, normalized
passages for BoolQA, and paragraph+question groups for MultiRC as atomic split
units.  It is an oracle-headroom diagnostic, not a claimed implementation of
O-LoRA or OA-Adapter.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

import numpy as np
import torch
from torch import Tensor

try:
    from .buffers import (
        AnchorRecord,
        BufferRole,
        ControllerRecord,
        HistoryBuffer,
        assert_disjoint_history,
    )
    from .objectives import (
        detached_cvar_weights,
        detached_worst_group_dro_weights,
        differentiable_anchored_regret,
        per_example_target_token_nll,
    )
    from .order4_data import (
        OFFICIAL_COMMIT,
        OFFICIAL_REPOSITORY,
        ORDER4_TASK_NAMES,
        TASK_BY_NAME,
        OfficialExample,
        download_official_data,
        load_official_examples,
        load_sealed_evaluation_data,
        normalize_answer,
        normalized_em,
    )
    from .run_order4_panel import select_evaluation_examples
    from .t5_rank_bank import T5GlobalRankBank
except ImportError:  # pragma: no cover - direct ``python file.py`` use.
    from buffers import (  # type: ignore
        AnchorRecord,
        BufferRole,
        ControllerRecord,
        HistoryBuffer,
        assert_disjoint_history,
    )
    from objectives import (  # type: ignore
        detached_cvar_weights,
        detached_worst_group_dro_weights,
        differentiable_anchored_regret,
        per_example_target_token_nll,
    )
    from order4_data import (  # type: ignore
        OFFICIAL_COMMIT,
        OFFICIAL_REPOSITORY,
        ORDER4_TASK_NAMES,
        TASK_BY_NAME,
        OfficialExample,
        download_official_data,
        load_official_examples,
        load_sealed_evaluation_data,
        normalize_answer,
        normalized_em,
    )
    from run_order4_panel import select_evaluation_examples  # type: ignore
    from t5_rank_bank import T5GlobalRankBank  # type: ignore


LOCKED_ARMS: tuple[str, ...] = ("uniform_er_rank4", "task_id_oracle")
ORACLE_ARM = "task_id_oracle"
FORMAT_VERSION = "phase2i.acquisition-conditioned-headroom.v4"


@dataclass(frozen=True)
class HeadroomConfig:
    data_root: str
    model_path: str
    output_root: str
    task_names: tuple[str, ...]
    arms: tuple[str, ...]
    seed: int
    update_cap_per_class: int
    risk_per_class: int
    acquisition_audit_examples: int
    min_update_per_class: int
    eval_per_task: int
    batch_size: int
    replay_batch_size: int
    default_acquisition_epochs: int
    acquisition_epochs_by_task: Mapping[str, int]
    consolidation_epochs: int
    lr: float
    max_source_length: int
    max_target_length: int
    cvar_alpha: float
    risk_fraction: float
    min_audit_acquisition: float
    min_anchor_loss_gain: float
    risk_buffer_records: int
    audit_buffer_records: int
    fail_on_acquisition: bool
    stop_after_task: int | None
    tiny_random_model: bool
    smoke: bool
    device: str


@dataclass(frozen=True)
class StreamTask:
    name: str
    update: tuple[OfficialExample, ...]
    risk: tuple[OfficialExample, ...]
    audit: tuple[OfficialExample, ...]
    effective_risk_per_class: int
    effective_audit_per_class: int
    effective_update_per_class: int
    excluded_train_count: int
    natural_label_prior: Mapping[str, float]
    train_label_counts: Mapping[str, int]
    train_distribution_sha256: str
    group_key_by_example_id: Mapping[str, str]
    audit_group_selection_policy: str
    audit_group_priority_seed: int

    def assert_valid(self) -> None:
        groups = (
            {item.example_id for item in self.update},
            {item.example_id for item in self.risk},
            {item.example_id for item in self.audit},
        )
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise AssertionError("update/risk/audit partitions overlap")
        if not all(item.subset == "train" for part in (self.update, self.risk, self.audit) for item in part):
            raise AssertionError("all streaming partitions must be train-derived")


@dataclass(frozen=True)
class ReplayPlan:
    records: tuple[AnchorRecord | ControllerRecord, ...]
    probabilities: tuple[float, ...] | None
    eligible_count: int
    mode: str
    controller_forward_examples: int
    controller_forward_batches: int
    controller_forward_source_tokens: int
    controller_forward_target_tokens: int


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(examples: Sequence[OfficialExample]) -> str:
    payload = json.dumps(
        [item.example_id for item in examples], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_seed(base_seed: int, *parts: object) -> int:
    material = "\0".join([str(int(base_seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _partition_key(example: OfficialExample, seed: int, purpose: str) -> str:
    material = f"acq-conditioned\0{seed}\0{purpose}\0{example.example_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


SEMANTIC_GROUP_KEY_VERSION = "headroom-semantic-group.v2"


def _strictly_above(value: float, threshold: float) -> bool:
    """A strict gate with a tiny equality guard for serialized floats."""

    return bool(
        value > threshold
        and not math.isclose(value, threshold, rel_tol=0.0, abs_tol=1e-12)
    )


def _at_least(value: float, threshold: float) -> bool:
    """Inclusive guard robust to subtraction noise at the exact boundary."""

    return bool(
        value >= threshold
        or math.isclose(value, threshold, rel_tol=0.0, abs_tol=1e-12)
    )


def _normalize_group_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def _split_once(value: str, marker: str, *, task_name: str) -> tuple[str, str]:
    head, separator, tail = value.partition(marker)
    if not separator:
        raise ValueError(f"{task_name} sentence lacks {marker.strip()!r} field")
    return head, tail


def semantic_group_keys(
    examples: Sequence[OfficialExample], *, task_name: str
) -> dict[str, str]:
    """Full-corpus semantic components, equivalent to strict rescue-v2.

    QQP uses connected components in the normalized-question graph; BoolQA
    groups every question sharing a normalized passage; MultiRC groups every
    candidate sharing normalized paragraph+question.  Other tasks remain
    content-addressed row units.
    """

    raw: dict[str, str] = {}
    if task_name == "MultiRC":
        for example in examples:
            body, candidate = _split_once(
                example.sentence, "\ncandidate answer: ", task_name=task_name
            )
            paragraph, question = _split_once(
                body, "\nquestion: ", task_name=task_name
            )
            if not paragraph.startswith("paragraph: ") or not candidate:
                raise ValueError("MultiRC sentence has malformed semantic fields")
            raw[example.example_id] = json.dumps(
                {
                    "paragraph": _normalize_group_text(paragraph[len("paragraph: ") :]),
                    "question": _normalize_group_text(question),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
    elif task_name == "BoolQA":
        for example in examples:
            question, passage = _split_once(
                example.sentence, "\npassage: ", task_name=task_name
            )
            if not question.startswith("question: ") or not passage:
                raise ValueError("BoolQA sentence has malformed semantic fields")
            raw[example.example_id] = _normalize_group_text(passage)
    elif task_name == "QQP":
        parent: dict[str, str] = {}
        pairs: list[tuple[OfficialExample, str, str]] = []

        def find(node: str) -> str:
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                return
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

        for example in examples:
            first, second = _split_once(
                example.sentence, "\nsecond sentence: ", task_name=task_name
            )
            if not first.startswith("first sentence: ") or not second:
                raise ValueError("QQP sentence has malformed semantic fields")
            q1 = _normalize_group_text(first[len("first sentence: ") :])
            q2 = _normalize_group_text(second)
            union(q1, q2)
            pairs.append((example, q1, q2))
        nodes_by_root: dict[str, list[str]] = {}
        for node in sorted(parent):
            nodes_by_root.setdefault(find(node), []).append(node)
        payload_by_root = {
            root: json.dumps(nodes, ensure_ascii=False, separators=(",", ":"))
            for root, nodes in nodes_by_root.items()
        }
        for example, q1, _q2 in pairs:
            raw[example.example_id] = payload_by_root[find(q1)]
    else:
        raw = {example.example_id: example.example_id for example in examples}
    return {
        example_id: hashlib.sha256(
            f"{SEMANTIC_GROUP_KEY_VERSION}\0{task_name}\0{payload}".encode("utf-8")
        ).hexdigest()
        for example_id, payload in raw.items()
    }


def _semantic_partition_key(task_name: str, group_key: str, seed: int) -> str:
    return hashlib.sha256(
        f"headroom-semantic\0{seed}\0{task_name}\0{group_key}".encode("utf-8")
    ).hexdigest()


AUDIT_GROUP_SELECTION_POLICY = "hash_random_stratified_exact_sparse_dp.v2"
CAPACITY_GROUP_SELECTION_POLICY = "size_first_capacity_greedy.v1"


def _take_exact_group_partition(
    remaining: Mapping[str, tuple[OfficialExample, ...]],
    *,
    task_name: str,
    labels: Sequence[str],
    quota_per_class: int,
    seed: int,
    selection_policy: str = CAPACITY_GROUP_SELECTION_POLICY,
) -> tuple[tuple[OfficialExample, ...], dict[str, tuple[OfficialExample, ...]]]:
    deficits = {label: quota_per_class for label in labels}
    available = dict(remaining)
    chosen: list[OfficialExample] = []
    if selection_policy == CAPACITY_GROUP_SELECTION_POLICY:
        ordered = sorted(
            available.items(),
            key=lambda item: (
                -len(item[1]),
                _semantic_partition_key(task_name, item[0], seed),
                item[0],
            ),
        )
        for group_key, group in ordered:
            counts = {
                label: sum(row.label == label for row in group) for label in labels
            }
            if all(counts[label] <= deficits[label] for label in labels):
                chosen.extend(group)
                del available[group_key]
                for label in labels:
                    deficits[label] -= counts[label]
                if not any(deficits.values()):
                    break
    elif selection_policy == AUDIT_GROUP_SELECTION_POLICY:
        # Audit groups are ordered only by a seeded hash, never by group size.
        # Multi-row representation is targeted from the complete corpus within
        # each label, a sparse DP chooses whole multi-row groups, and seeded-
        # hash singleton fillers recover the exact label quotas.  This avoids
        # both the old -len(group) priority and the implicit "reach the target
        # fastest" preference for large groups.  Atomic-row tasks take a linear
        # fast path.
        ordered = sorted(
            available.items(),
            key=lambda item: (
                _semantic_partition_key(task_name, item[0], seed),
                item[0],
            ),
        )
        label_index = {label: index for index, label in enumerate(labels)}
        target = tuple(quota_per_class for _ in labels)
        vectors = tuple(
            tuple(
                sum(row.label == label for row in group)
                for label in labels
            )
            for _group_key, group in ordered
        )
        singleton_indices = [
            index for index, vector in enumerate(vectors) if sum(vector) == 1
        ]
        multirow_indices = [
            index for index, vector in enumerate(vectors) if sum(vector) > 1
        ]
        if not multirow_indices:
            selected_indices: list[int] = []
            filled = [0 for _ in labels]
            for index in singleton_indices:
                vector = vectors[index]
                label_position = vector.index(1)
                if filled[label_position] < quota_per_class:
                    selected_indices.append(index)
                    filled[label_position] += 1
                if tuple(filled) == target:
                    break
        else:
            corpus_label_rows = tuple(
                sum(vector[position] for vector in vectors)
                for position in range(len(labels))
            )
            multirow_label_rows = tuple(
                sum(vectors[index][position] for index in multirow_indices)
                for position in range(len(labels))
            )
            singleton_label_rows = tuple(
                sum(vectors[index][position] for index in singleton_indices)
                for position in range(len(labels))
            )
            target_multirow_rows = tuple(
                min(
                    quota_per_class,
                    int(
                        math.floor(
                            quota_per_class
                            * multirow_label_rows[position]
                            / corpus_label_rows[position]
                            + 0.5
                        )
                    ),
                )
                for position in range(len(labels))
            )

            zero = tuple(0 for _ in labels)
            paths: dict[tuple[int, ...], tuple[int, ...]] = {zero: ()}
            for index in multirow_indices:
                vector = vectors[index]
                if any(vector[position] > target[position] for position in range(len(labels))):
                    continue
                prior_states = tuple(paths.items())
                for state, path in prior_states:
                    candidate = tuple(
                        state[position] + vector[position]
                        for position in range(len(labels))
                    )
                    if any(
                        candidate[position] > target[position]
                        for position in range(len(labels))
                    ):
                        continue
                    candidate_path = (*path, index)
                    existing_path = paths.get(candidate)
                    # Indices are the seeded-hash group ranks.  Lexicographic
                    # path order therefore implements explicit random group
                    # priority without any size-based tie-break.
                    if existing_path is None or candidate_path < existing_path:
                        paths[candidate] = candidate_path
                if len(paths) > 1_000_000:
                    raise RuntimeError(
                        f"{task_name} audit exact-quota DP exceeded one million states"
                    )
            feasible_states = [
                state
                for state in paths
                if all(
                    target[position] - state[position]
                    <= singleton_label_rows[position]
                    for position in range(len(labels))
                )
            ]
            if not feasible_states:
                raise ValueError(
                    f"{task_name} semantic groups cannot leave singleton exact-quota fillers"
                )
            best_state = min(
                feasible_states,
                key=lambda state: (
                    sum(
                        abs(state[position] - target_multirow_rows[position])
                        for position in range(len(labels))
                    ),
                    sum(
                        max(0, state[position] - target_multirow_rows[position])
                        for position in range(len(labels))
                    ),
                    paths[state],
                    state,
                ),
            )
            selected_indices = list(paths[best_state])
            filled = list(best_state)
            for index in singleton_indices:
                label_position = vectors[index].index(1)
                if filled[label_position] < quota_per_class:
                    selected_indices.append(index)
                    filled[label_position] += 1
                if tuple(filled) == target:
                    break
        for index in selected_indices:
            group_key, group = ordered[index]
            chosen.extend(group)
            del available[group_key]
            for label, position in label_index.items():
                deficits[label] -= vectors[index][position]
    else:
        raise ValueError(f"unknown semantic-group selection policy {selection_policy!r}")
    if any(deficits.values()):
        raise ValueError(
            f"{task_name} semantic groups cannot meet exact quota "
            f"{quota_per_class}; deficits={deficits}"
        )
    return tuple(sorted(chosen, key=lambda row: row.source_index)), available


def parse_task_selection(value: str) -> tuple[str, ...]:
    stripped = value.strip()
    if stripped.lower() in {"all", "all15", "15"}:
        return ORDER4_TASK_NAMES
    if stripped.isdigit():
        count = int(stripped)
        if not 1 <= count <= len(ORDER4_TASK_NAMES):
            raise ValueError(f"task prefix must be in [1, {len(ORDER4_TASK_NAMES)}]")
        return ORDER4_TASK_NAMES[:count]
    lookup = {name.lower(): name for name in ORDER4_TASK_NAMES}
    requested = tuple(part.strip() for part in stripped.split(",") if part.strip())
    try:
        canonical = tuple(lookup[part.lower()] for part in requested)
    except KeyError as error:
        raise ValueError(f"unknown Order-4 task {error.args[0]!r}") from error
    if not canonical or canonical != ORDER4_TASK_NAMES[: len(canonical)]:
        raise ValueError("--tasks must be a non-empty chronology-preserving Order-4 prefix")
    return canonical


def parse_arm_selection(values: Sequence[str]) -> tuple[str, ...]:
    flat = [part.strip() for value in values for part in value.split(",") if part.strip()]
    if len(flat) == 1 and flat[0].lower() == "all":
        return LOCKED_ARMS
    if not flat or len(flat) != len(set(flat)):
        raise ValueError("--arms must be non-empty and contain no duplicates")
    unknown = sorted(set(flat).difference(LOCKED_ARMS))
    if unknown:
        raise ValueError(f"unknown arms {unknown}; allowed={LOCKED_ARMS}")
    return tuple(arm for arm in LOCKED_ARMS if arm in set(flat))


def build_stream_task(
    examples: Sequence[OfficialExample],
    *,
    task_name: str,
    update_cap_per_class: int,
    risk_per_class: int,
    acquisition_audit_examples: int,
    min_update_per_class: int,
    seed: int,
) -> StreamTask:
    """Make disjoint train partitions, adapting held-out quotas for tiny classes.

    The requested risk:audit ratio is retained as closely as possible.  A task
    uses one balanced held-out quota for every label; this prevents a rare
    class (CB/neutral in the pinned data) from disappearing while keeping audit
    EM class-balanced.  Update is capped independently after held-out removal.
    """

    integers = (
        update_cap_per_class,
        risk_per_class,
        acquisition_audit_examples,
        min_update_per_class,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in integers):
        raise ValueError("partition counts must be positive integers")
    if not examples or any(item.subset != "train" for item in examples):
        raise ValueError("stream preparation accepts non-empty train examples only")
    if any(item.task_name != task_name for item in examples):
        raise ValueError("stream examples must all belong to task_name")
    labels = TASK_BY_NAME[task_name].labels
    label_counts = {label: 0 for label in labels}
    for item in examples:
        if item.label not in label_counts:
            raise ValueError(f"unexpected label {item.label!r} for {task_name}")
        label_counts[item.label] += 1
    total = sum(label_counts.values())
    natural_prior = {label: label_counts[label] / total for label in labels}
    distribution_payload = json.dumps(
        {
            "task": task_name,
            "counts": label_counts,
            "total": total,
            "ordered_example_ids_sha256": _ids_sha256(
                tuple(sorted(examples, key=lambda item: item.source_index))
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    group_key_by_id = semantic_group_keys(examples, task_name=task_name)
    semantic_groups: dict[str, list[OfficialExample]] = {}
    for item in examples:
        semantic_groups.setdefault(group_key_by_id[item.example_id], []).append(item)
    remaining: dict[str, tuple[OfficialExample, ...]] = {
        key: tuple(sorted(rows, key=lambda item: item.source_index))
        for key, rows in semantic_groups.items()
    }
    desired_audit_quota = max(1, acquisition_audit_examples // len(labels))
    audit_quota = min(
        desired_audit_quota,
        min(label_counts.values()) - min_update_per_class - 1,
    )
    if audit_quota < 1:
        raise ValueError(f"{task_name} cannot retain disjoint audit/risk/update rows")
    audit_priority_seed = _stable_seed(seed, "audit")
    audit_tuple, remaining = _take_exact_group_partition(
        remaining,
        task_name=task_name,
        labels=labels,
        quota_per_class=audit_quota,
        seed=audit_priority_seed,
        selection_policy=AUDIT_GROUP_SELECTION_POLICY,
    )
    remaining_counts = {
        label: sum(row.label == label for rows in remaining.values() for row in rows)
        for label in labels
    }
    risk_quota = min(
        risk_per_class,
        min(remaining_counts.values()) - min(min_update_per_class, update_cap_per_class),
    )
    if risk_quota < 1:
        raise ValueError(
            f"{task_name} balanced audit leaves no disjoint risk/update support: "
            f"{remaining_counts}"
        )
    risk_tuple, remaining = _take_exact_group_partition(
        remaining,
        task_name=task_name,
        labels=labels,
        quota_per_class=risk_quota,
        seed=_stable_seed(seed, "risk"),
    )
    post_risk_counts = {
        label: sum(row.label == label for rows in remaining.values() for row in rows)
        for label in labels
    }
    effective_update_target = min(update_cap_per_class, min(post_risk_counts.values()))
    if effective_update_target < min(min_update_per_class, update_cap_per_class):
        raise ValueError(
            f"{task_name} cannot retain the minimum balanced update support: {post_risk_counts}"
        )
    update_tuple, _remaining = _take_exact_group_partition(
        remaining,
        task_name=task_name,
        labels=labels,
        quota_per_class=effective_update_target,
        seed=_stable_seed(seed, "update"),
    )
    assigned = len(update_tuple) + len(risk_tuple) + len(audit_tuple)
    task = StreamTask(
        name=task_name,
        update=update_tuple,
        risk=risk_tuple,
        audit=audit_tuple,
        effective_risk_per_class=risk_quota,
        effective_audit_per_class=audit_quota,
        effective_update_per_class=effective_update_target,
        excluded_train_count=len(examples) - assigned,
        natural_label_prior=natural_prior,
        train_label_counts=label_counts,
        train_distribution_sha256=hashlib.sha256(distribution_payload).hexdigest(),
        group_key_by_example_id=group_key_by_id,
        audit_group_selection_policy=AUDIT_GROUP_SELECTION_POLICY,
        audit_group_priority_seed=audit_priority_seed,
    )
    task.assert_valid()
    semantic_sets = [
        {group_key_by_id[item.example_id] for item in partition}
        for partition in (task.update, task.risk, task.audit)
    ]
    if semantic_sets[0] & semantic_sets[1] or semantic_sets[0] & semantic_sets[2] or semantic_sets[1] & semantic_sets[2]:
        raise AssertionError("semantic groups overlap across train partitions")
    return task


def prepare_train_stream(config: HeadroomConfig) -> tuple[StreamTask, ...]:
    """The only pre-training loader: its split argument is a constant train value."""

    tasks: list[StreamTask] = []
    for index, task_name in enumerate(config.task_names):
        examples = load_official_examples(config.data_root, task_name, "train", verify=True)
        tasks.append(
            build_stream_task(
                examples,
                task_name=task_name,
                update_cap_per_class=config.update_cap_per_class,
                risk_per_class=config.risk_per_class,
                acquisition_audit_examples=config.acquisition_audit_examples,
                min_update_per_class=config.min_update_per_class,
                seed=_stable_seed(config.seed, "partition", index, task_name),
            )
        )
    return tuple(tasks)


def _chunks(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _generic_field_aware_ids(
    tokenizer: Any,
    record: OfficialExample | AnchorRecord | ControllerRecord,
    max_source_length: int,
) -> list[int]:
    """Field-aware replay packing inferred from text, without a task-ID input."""

    prompt = record.prompt
    answer = "\nAnswer:"
    if not prompt.endswith(answer):
        return [int(value) for value in tokenizer(prompt, truncation=True, max_length=max_source_length)["input_ids"]]
    body = prompt[: -len(answer)]
    components: tuple[str, str, str] | None = None
    paragraph_marker = "paragraph: "
    question_marker = "\nquestion: "
    candidate_marker = "\ncandidate answer: "
    if paragraph_marker in body and question_marker in body and candidate_marker in body:
        head, marker, paragraph_tail = body.rpartition(paragraph_marker)
        paragraph, separator, question_candidate = paragraph_tail.rpartition(question_marker)
        question, separator2, candidate = question_candidate.rpartition(candidate_marker)
        if marker and separator and separator2:
            components = (
                head + paragraph_marker,
                paragraph,
                question_marker + question + candidate_marker + candidate + answer,
            )
    passage_marker = "\npassage: "
    if components is None and passage_marker in body and "question: " in body:
        head_question, separator, passage = body.rpartition(passage_marker)
        if separator:
            components = (head_question + passage_marker, passage, answer)
    if components is None:
        return [int(value) for value in tokenizer(prompt, truncation=True, max_length=max_source_length)["input_ids"]]
    head, flexible, suffix = components
    head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
    flexible_ids = tokenizer(flexible, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    special = int(tokenizer.num_special_tokens_to_add(pair=False))
    available = max_source_length - len(head_ids) - len(suffix_ids) - special
    if available < 0:
        raise ValueError("protected replay fields exceed max_source_length")
    return [
        int(value)
        for value in tokenizer.build_inputs_with_special_tokens(
            [*head_ids, *flexible_ids[:available], *suffix_ids]
        )
    ]


def _encode_official_sources(
    tokenizer: Any,
    examples: Sequence[OfficialExample],
    *,
    max_source_length: int,
) -> dict[str, Tensor]:
    rows = [
        _generic_field_aware_ids(tokenizer, example, max_source_length)
        for example in examples
    ]
    width = max(len(row) for row in rows)
    input_ids = torch.full(
        (len(rows), width), int(tokenizer.pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for row_index, row in enumerate(rows):
        input_ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask[row_index, : len(row)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _source_token_length_report(
    tokenizer: Any,
    examples: Sequence[OfficialExample],
    *,
    max_source_length: int,
) -> dict[str, Any]:
    lengths: list[int] = []
    for batch in _chunks(examples, 256):
        encoded = tokenizer(
            [example.prompt for example in batch],
            add_special_tokens=True,
            truncation=False,
            return_length=True,
        )
        reported = encoded.get("length")
        lengths.extend(
            int(value)
            for value in (
                reported
                if reported is not None
                else [len(ids) for ids in encoded["input_ids"]]
            )
        )
    values = np.asarray(lengths, dtype=np.int64)
    return {
        "example_count": len(examples),
        "untruncated_tokens": {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": int(values.max()),
        },
        "max_source_length": max_source_length,
        "would_truncate_count": int((values > max_source_length).sum()),
        "would_truncate_rate": float((values > max_source_length).mean()),
        "packing": "local field-aware; protected BoolQA question and MultiRC question/candidate suffix",
    }


def _encode_records(
    tokenizer: Any,
    records: Sequence[OfficialExample | AnchorRecord | ControllerRecord],
    *,
    config: HeadroomConfig,
) -> dict[str, Tensor]:
    if not records:
        raise ValueError("cannot encode an empty record batch")
    rows: list[list[int]] = []
    for record in records:
        rows.append(_generic_field_aware_ids(tokenizer, record, config.max_source_length))
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), int(tokenizer.pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for row_index, row in enumerate(rows):
        input_ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask[row_index, : len(row)] = 1
    targets = tokenizer(
        text_target=[record.label if isinstance(record, OfficialExample) else record.target for record in records],
        padding=True,
        truncation=True,
        max_length=config.max_target_length,
        return_tensors="pt",
    )
    labels = targets["input_ids"].masked_fill(targets["input_ids"].eq(tokenizer.pad_token_id), -100)
    device = torch.device(config.device)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


def _per_example_losses(
    bank: T5GlobalRankBank,
    tokenizer: Any,
    records: Sequence[OfficialExample | AnchorRecord | ControllerRecord],
    *,
    config: HeadroomConfig,
    forward_ledger: dict[str, int] | None = None,
) -> tuple[float, ...]:
    if not records:
        return ()
    was_training = bank.training
    bank.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in _chunks(records, config.batch_size):
            encoded = _encode_records(tokenizer, batch, config=config)
            output = bank(**encoded)
            values = per_example_target_token_nll(output.logits, encoded["labels"])
            losses.extend(float(value) for value in values.detach().cpu())
            if forward_ledger is not None:
                forward_ledger["examples"] += len(batch)
                forward_ledger["batches"] += 1
                forward_ledger["source_tokens"] += int(encoded["attention_mask"].sum())
                forward_ledger["target_tokens"] += int(encoded["labels"].ne(-100).sum())
    bank.train(was_training)
    return tuple(losses)


def natural_prior_score(
    per_class_recall: Mapping[str, float | None],
    natural_label_prior: Mapping[str, float],
) -> float:
    if set(per_class_recall) != set(natural_label_prior) or any(
        value is None for value in per_class_recall.values()
    ):
        raise ValueError("natural-prior score requires every official label")
    if not math.isclose(
        sum(float(value) for value in natural_label_prior.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("natural label prior must sum to one")
    return float(
        sum(
            float(natural_label_prior[label]) * float(per_class_recall[label])
            for label in per_class_recall
        )
    )


def evaluate_free_em(
    bank: T5GlobalRankBank,
    tokenizer: Any,
    examples: Sequence[OfficialExample],
    *,
    config: HeadroomConfig,
    natural_label_prior: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("evaluation requires non-empty examples")
    was_training = bank.training
    bank.eval()
    predictions: list[str] = []
    with torch.no_grad():
        for batch in _chunks(examples, config.batch_size):
            encoded = _encode_official_sources(
                tokenizer,
                batch,
                max_source_length=config.max_source_length,
            )
            generated = bank.generate(
                input_ids=encoded["input_ids"].to(torch.device(config.device)),
                attention_mask=encoded["attention_mask"].to(torch.device(config.device)),
                max_new_tokens=config.max_target_length,
                num_beams=1,
                do_sample=False,
            )
            predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    bank.train(was_training)
    references = [item.label for item in examples]
    labels = {normalize_answer(label) for label in TASK_BY_NAME[examples[0].task_name].labels}
    normalized = [normalize_answer(value) for value in predictions]
    recalls: dict[str, float | None] = {}
    for label in TASK_BY_NAME[examples[0].task_name].labels:
        indices = [index for index, reference in enumerate(references) if reference == label]
        recalls[label] = (
            float(
                sum(normalized[index] == normalize_answer(references[index]) for index in indices)
                / len(indices)
            )
            if indices
            else None
        )
    observed_recalls = [value for value in recalls.values() if value is not None]
    natural_prior_em = None
    if natural_label_prior is not None:
        natural_prior_em = natural_prior_score(recalls, natural_label_prior)
    return {
        "normalized_em": normalized_em(predictions, references) / 100.0,
        "predictions": predictions,
        "references": references,
        "example_ids": [item.example_id for item in examples],
        "per_class_recall": recalls,
        "balanced_accuracy_auxiliary": (
            float(np.mean(observed_recalls))
            if len(observed_recalls) == len(recalls)
            else None
        ),
        "worst_class_recall_auxiliary": (
            float(min(observed_recalls))
            if len(observed_recalls) == len(recalls)
            else None
        ),
        "valid_label_rate": float(sum(value in labels for value in normalized) / len(normalized)),
        "natural_prior_em": natural_prior_em,
        "natural_label_prior": (
            None
            if natural_label_prior is None
            else {label: float(natural_label_prior[label]) for label in recalls}
        ),
    }


def acquisition_gate(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    if before.get("natural_prior_em") is None or after.get("natural_prior_em") is None:
        raise ValueError("acquisition gate requires natural-prior EM")
    before_recalls = before.get("per_class_recall")
    after_recalls = after.get("per_class_recall")
    if not isinstance(before_recalls, Mapping) or not isinstance(after_recalls, Mapping):
        raise ValueError("acquisition gate requires per-class recalls")
    if set(before_recalls) != set(after_recalls) or any(
        before_recalls[label] is None or after_recalls[label] is None
        for label in before_recalls
    ):
        raise ValueError("acquisition gate requires matched observed classes")
    gain = float(after["natural_prior_em"] - before["natural_prior_em"])
    class_deltas = {
        label: float(after_recalls[label]) - float(before_recalls[label])
        for label in before_recalls
    }
    minimum = min(class_deltas.values())
    guards = {
        "natural_prior_gain_strictly_greater_than_threshold": _strictly_above(
            gain, threshold
        ),
        "post_valid_label_rate_at_least_0.99": _at_least(
            float(after["valid_label_rate"]), 0.99
        ),
        "min_per_class_recall_delta_at_least_minus_0.05": _at_least(
            minimum, -0.05
        ),
    }
    return {
        "natural_prior_gain": gain,
        "per_class_recall_delta": class_deltas,
        "min_per_class_recall_delta": minimum,
        "guards": guards,
        "pass": all(guards.values()),
    }


def _build_model(config: HeadroomConfig, tokenizer: Any) -> T5GlobalRankBank:
    from transformers import AutoModelForSeq2SeqLM, T5Config, T5ForConditionalGeneration

    _set_seed(config.seed)
    if config.tiny_random_model:
        model = T5ForConditionalGeneration(
            T5Config(
                vocab_size=len(tokenizer),
                d_model=32,
                d_kv=8,
                d_ff=64,
                num_layers=1,
                num_decoder_layers=1,
                num_heads=4,
                dropout_rate=0.0,
                decoder_start_token_id=tokenizer.pad_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path, local_files_only=True)
    model.to(torch.device(config.device))
    return T5GlobalRankBank(
        model,
        max_rank=8,
        initial_rank=4,
        alpha=16.0,
        reference_rank=4,
        lora_dropout=0.1,
        require_t5_small=not config.tiny_random_model,
    )


def _adapter_fingerprint(bank: T5GlobalRankBank) -> str:
    digest = hashlib.sha256()
    for name, adapter in zip(bank.adapter_names, bank.adapters):
        digest.update(name.encode("utf-8"))
        digest.update(adapter.active_mask.detach().cpu().numpy().tobytes())
        for atom in adapter.atoms:
            digest.update(atom.a.detach().cpu().contiguous().numpy().tobytes())
            digest.update(atom.b.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sequence_mean_prior_weights(
    labels: Sequence[str],
    *,
    natural_label_prior: Mapping[str, float],
    sampler_q: Mapping[str, float],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Importance weights pi(y)/q(y), never batch-renormalized."""

    if set(natural_label_prior) != set(sampler_q):
        raise ValueError("natural prior and sampler q labels differ")
    values = [
        float(natural_label_prior[label]) / float(sampler_q[label])
        for label in labels
    ]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("sequence_mean_prior weights must be finite and positive")
    return torch.tensor(values, dtype=dtype, device=device)


def train_acquisition(
    bank: T5GlobalRankBank,
    tokenizer: Any,
    examples: Sequence[OfficialExample],
    *,
    task_index: int,
    acquisition_epochs: int,
    natural_label_prior: Mapping[str, float],
    config: HeadroomConfig,
) -> dict[str, Any]:
    """Current-only training with a fresh per-task optimizer."""

    optimizer = torch.optim.AdamW(tuple(bank.all_atom_parameters()), lr=config.lr, weight_decay=0.0)
    official_labels = TASK_BY_NAME[examples[0].task_name].labels
    update_counts = {
        label: sum(example.label == label for example in examples)
        for label in official_labels
    }
    if len(set(update_counts.values())) != 1:
        raise ValueError(
            "sequence_mean_prior protocol requires an exactly balanced update sampler"
        )
    update_total = sum(update_counts.values())
    sampler_q = {label: update_counts[label] / update_total for label in official_labels}
    ledger: dict[str, Any] = {
        "optimizer_steps": 0,
        "gradient_example_exposures": 0,
        "source_tokens": 0,
        "target_tokens": 0,
        "sampler_label_counts_per_epoch": update_counts,
        "sampler_q": sampler_q,
        "natural_label_prior": dict(natural_label_prior),
        "per_class": {
            label: {"row_exposures": 0, "objective_weight_sum": 0.0}
            for label in official_labels
        },
    }
    epoch_losses: list[float] = []
    bank.train()
    for epoch in range(acquisition_epochs):
        _set_seed(_stable_seed(config.seed, "acquisition", task_index, epoch))
        rng = np.random.Generator(np.random.PCG64(_stable_seed(config.seed, "order", task_index, epoch)))
        ordered = tuple(examples[int(index)] for index in rng.permutation(len(examples)))
        losses: list[float] = []
        for batch in _chunks(ordered, config.batch_size):
            encoded = _encode_records(tokenizer, batch, config=config)
            output = bank(**encoded)
            sequence_nll = per_example_target_token_nll(output.logits, encoded["labels"])
            example_weights = sequence_mean_prior_weights(
                [example.label for example in batch],
                natural_label_prior=natural_label_prior,
                sampler_q=sampler_q,
                dtype=sequence_nll.dtype,
                device=sequence_nll.device,
            )
            loss = torch.mean(example_weights * sequence_nll)
            if loss is None or loss.ndim != 0 or not bool(torch.isfinite(loss.detach())):
                raise RuntimeError("T5 did not return a finite token-level CE loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tuple(bank.active_atom_parameters()), 1.0)
            optimizer.step()
            bank.assert_exact_budget()
            losses.append(float(loss.detach().cpu()))
            ledger["optimizer_steps"] += 1
            ledger["gradient_example_exposures"] += len(batch)
            ledger["source_tokens"] += int(encoded["attention_mask"].sum())
            ledger["target_tokens"] += int(encoded["labels"].ne(-100).sum())
            for example, weight in zip(batch, example_weights.detach().cpu().tolist()):
                ledger["per_class"][example.label]["row_exposures"] += 1
                ledger["per_class"][example.label]["objective_weight_sum"] += float(weight)
        epoch_losses.append(float(np.mean(losses)))
    return {
        "phase": "current_only_acquisition",
        "optimizer_reset_at_phase_start": True,
        "optimizer": "AdamW",
        "lr": config.lr,
        "weight_decay": 0.0,
        "epochs": acquisition_epochs,
        "loss_objective": "sequence_mean_prior",
        "loss_reduction": "mean(pi(y)/q(y) * per-example target-sequence mean NLL); no batch renormalization",
        "epoch_mean_objective": epoch_losses,
        "resource_ledger": ledger,
        "historical_gradient_examples": 0,
        "audit_gradient_examples": 0,
    }


def _make_anchors(
    examples: Sequence[OfficialExample],
    pre: Sequence[float],
    post: Sequence[float],
    *,
    task_name: str,
    include_task_id: bool,
    task_gate_pass: bool,
    min_gain: float,
) -> tuple[AnchorRecord, ...]:
    if not (len(examples) == len(pre) == len(post)):
        raise ValueError("anchor vectors differ in length")
    return tuple(
        AnchorRecord(
            example_id=example.example_id,
            prompt=example.prompt,
            target=example.label,
            loss_pre=float(before),
            loss_post=float(after),
            eligible=bool(task_gate_pass and float(before) - float(after) >= min_gain),
            task_id=task_name if include_task_id else None,
        )
        for example, before, after in zip(examples, pre, post)
    )


def build_replay_plan(
    arm: str,
    buffer: HistoryBuffer,
    bank: T5GlobalRankBank,
    tokenizer: Any,
    *,
    config: HeadroomConfig,
) -> ReplayPlan:
    resident = buffer.records_for_evaluation()
    if arm == ORACLE_ARM:
        if any(record.task_id is None for record in resident):
            raise RuntimeError("oracle buffer lacks task IDs")
        records: tuple[AnchorRecord | ControllerRecord, ...] = tuple(resident)
    else:
        records = tuple(record.controller_view() for record in resident)
        if any(hasattr(record, "task_id") for record in records):
            raise RuntimeError("group-free controller view leaked task IDs")
    if not records:
        return ReplayPlan((), None, 0, "empty", 0, 0, 0, 0)
    forward_ledger = {"examples": 0, "batches": 0, "source_tokens": 0, "target_tokens": 0}
    now = torch.tensor(
        _per_example_losses(
            bank,
            tokenizer,
            records,
            config=config,
            forward_ledger=forward_ledger,
        ),
        dtype=torch.float32,
    )
    if arm != ORACLE_ARM:
        return ReplayPlan(
            records,
            None,
            sum(record.eligible for record in records),
            "group_free_uniform_er_full_buffer_scored_then_weights_ignored",
            forward_ledger["examples"],
            forward_ledger["batches"],
            forward_ledger["source_tokens"],
            forward_ledger["target_tokens"],
        )
    anchored = differentiable_anchored_regret(
        now,
        [record.loss_pre for record in records],
        [record.loss_post for record in records],
        eligibility=[record.eligible for record in records],
        min_acquired_gain=config.min_anchor_loss_gain,
    )
    eligible = int(anchored.eligible.sum())
    if eligible == 0:
        return ReplayPlan(
            records,
            None,
            0,
            "uniform_fallback_no_eligible_anchor",
            forward_ledger["examples"],
            forward_ledger["batches"],
            forward_ledger["source_tokens"],
            forward_ledger["target_tokens"],
        )
    probabilities = detached_worst_group_dro_weights(
        anchored.regret,
        [record.task_id for record in records],  # type: ignore[attr-defined]
        eligible=anchored.eligible,
    )
    mode = "task_id_worst_group_dro"
    return ReplayPlan(
        records,
        tuple(float(value) for value in probabilities.cpu()),
        eligible,
        mode,
        forward_ledger["examples"],
        forward_ledger["batches"],
        forward_ledger["source_tokens"],
        forward_ledger["target_tokens"],
    )


def _draw_indices(
    count: int,
    size: int,
    rng: np.random.Generator,
    probabilities: Sequence[float] | None = None,
) -> list[int]:
    if size <= 0:
        return []
    p = None if probabilities is None else np.asarray(probabilities, dtype=np.float64)
    if p is not None:
        total = float(p.sum())
        p = None if total <= 0 else p / total
    return [int(index) for index in rng.choice(count, size=size, replace=True, p=p)]


def train_consolidation(
    arm: str,
    bank: T5GlobalRankBank,
    tokenizer: Any,
    plan: ReplayPlan,
    current_examples: Sequence[OfficialExample],
    *,
    matched_steps_per_epoch: int,
    task_index: int,
    natural_label_prior: Mapping[str, float],
    config: HeadroomConfig,
) -> dict[str, Any]:
    """Current+historical consolidation with arm-matched step budgets."""

    requested_steps = matched_steps_per_epoch * config.consolidation_epochs
    if not current_examples:
        raise RuntimeError("consolidation requires current update examples")
    optimizer = torch.optim.AdamW(tuple(bank.all_atom_parameters()), lr=config.lr, weight_decay=0.0)
    rng = np.random.Generator(np.random.PCG64(_stable_seed(config.seed, "consolidation", task_index)))
    losses: list[float] = []
    official_labels = TASK_BY_NAME[current_examples[0].task_name].labels
    current_counts = {
        label: sum(example.label == label for example in current_examples)
        for label in official_labels
    }
    if len(set(current_counts.values())) != 1:
        raise ValueError("consolidation current sampler must remain exactly balanced")
    current_total = sum(current_counts.values())
    sampler_q = {label: current_counts[label] / current_total for label in official_labels}
    ledger: dict[str, Any] = {
        "optimizer_steps": 0,
        "current_example_exposures": 0,
        "replay_example_exposures": 0,
        "current_source_tokens": 0,
        "current_target_tokens": 0,
        "replay_source_tokens": 0,
        "replay_target_tokens": 0,
        "sampler_q": sampler_q,
        "natural_label_prior": dict(natural_label_prior),
        "per_class_current": {
            label: {"row_exposures": 0, "objective_weight_sum": 0.0}
            for label in official_labels
        },
    }
    bank.train()
    current_batches: list[Sequence[OfficialExample]] = []
    for epoch in range(config.consolidation_epochs):
        order_rng = np.random.Generator(
            np.random.PCG64(_stable_seed(config.seed, "consolidation-order", task_index, epoch))
        )
        ordered = tuple(current_examples[int(index)] for index in order_rng.permutation(len(current_examples)))
        current_batches.extend(_chunks(ordered, config.batch_size))
    if len(current_batches) != requested_steps:
        raise RuntimeError("consolidation current-batch schedule does not match its step ledger")
    for current in current_batches:
        if not plan.records:
            uniform_size, risk_size = 0, 0
        elif plan.probabilities is None:
            uniform_size, risk_size = config.replay_batch_size, 0
        else:
            risk_size = max(1, int(round(config.replay_batch_size * config.risk_fraction)))
            if config.replay_batch_size > 1:
                risk_size = min(risk_size, config.replay_batch_size - 1)
            uniform_size = config.replay_batch_size - risk_size
        uniform = tuple(plan.records[index] for index in _draw_indices(len(plan.records), uniform_size, rng))
        risk = tuple(
            plan.records[index]
            for index in _draw_indices(len(plan.records), risk_size, rng, plan.probabilities)
        )
        records = (*current, *uniform, *risk)
        encoded = _encode_records(tokenizer, records, config=config)
        output = bank(**encoded)
        nll = per_example_target_token_nll(output.logits, encoded["labels"])
        current_count = len(current)
        current_weights = sequence_mean_prior_weights(
            [example.label for example in current],
            natural_label_prior=natural_label_prior,
            sampler_q=sampler_q,
            dtype=nll.dtype,
            device=nll.device,
        )
        current_loss = torch.mean(current_weights * nll[:current_count])
        terms: list[tuple[Tensor, float]] = []
        cursor = current_count
        if uniform:
            terms.append((nll[cursor : cursor + len(uniform)].mean(), 1.0 - config.risk_fraction if risk else 1.0))
            cursor += len(uniform)
        if risk:
            # The oracle changes only which historical rows are sampled.  Once
            # sampled, both arms optimize the identical per-example sequence
            # NLL; using anchored regret here would confound allocation with a
            # different loss scale/objective.
            terms.append((nll[cursor:].mean(), config.risk_fraction))
        if terms:
            weight = sum(item[1] for item in terms)
            replay_loss = sum(term * (coefficient / weight) for term, coefficient in terms)
            objective = 0.5 * current_loss + 0.5 * replay_loss
        else:
            objective = current_loss
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(tuple(bank.active_atom_parameters()), 1.0)
        optimizer.step()
        bank.assert_exact_budget()
        losses.append(float(objective.detach().cpu()))
        ledger["optimizer_steps"] += 1
        ledger["current_example_exposures"] += len(current)
        ledger["replay_example_exposures"] += len(uniform) + len(risk)
        source_counts = encoded["attention_mask"].sum(dim=1)
        target_counts = encoded["labels"].ne(-100).sum(dim=1)
        ledger["current_source_tokens"] += int(source_counts[:current_count].sum())
        ledger["current_target_tokens"] += int(target_counts[:current_count].sum())
        ledger["replay_source_tokens"] += int(source_counts[current_count:].sum())
        ledger["replay_target_tokens"] += int(target_counts[current_count:].sum())
        for example, weight_value in zip(current, current_weights.detach().cpu().tolist()):
            ledger["per_class_current"][example.label]["row_exposures"] += 1
            ledger["per_class_current"][example.label]["objective_weight_sum"] += float(weight_value)
    return {
        "phase": "current_plus_historical_consolidation",
        "optimizer_reset_at_phase_start": True,
        "optimizer": "AdamW",
        "lr": config.lr,
        "weight_decay": 0.0,
        "epochs_equivalent": config.consolidation_epochs,
        "matched_steps_per_epoch": matched_steps_per_epoch,
        "mean_objective": float(np.mean(losses)),
        "loss_objective": "per-example sequence-mean NLL throughout",
        "current_objective": "sequence_mean_prior pi(y)/q(y), no batch renormalization",
        "uniform_replay_objective": "unweighted mean per-example sequence NLL",
        "risk_replay_objective": (
            "unweighted mean per-example sequence NLL after adversarial sampling; "
            "identical loss family to uniform replay"
        ),
        "resource_ledger": ledger,
        "plan": asdict(plan),
        "audit_gradient_examples": 0,
        "current_update_gradient_examples": ledger["current_example_exposures"],
    }


def _audit_regret(
    bank: T5GlobalRankBank,
    tokenizer: Any,
    buffer: HistoryBuffer,
    *,
    config: HeadroomConfig,
) -> dict[str, Any]:
    records = buffer.records_for_evaluation()
    if not records:
        return {"record_count": 0, "eligible_count": 0, "gradient_access": False}
    now = torch.tensor(_per_example_losses(bank, tokenizer, records, config=config), dtype=torch.float32)
    anchored = differentiable_anchored_regret(
        now,
        [record.loss_pre for record in records],
        [record.loss_post for record in records],
        eligibility=[record.eligible for record in records],
        min_acquired_gain=config.min_anchor_loss_gain,
    )
    values = anchored.tail_values.detach().cpu()
    result: dict[str, Any] = {
        "record_count": len(records),
        "eligible_count": int(values.numel()),
        "gradient_access": False,
    }
    if values.numel():
        weights = detached_cvar_weights(values, config.cvar_alpha)
        result.update(
            mean_anchored_regret=float(values.mean()),
            max_anchored_regret=float(values.max()),
            empirical_cvar=float((weights * values).sum()),
        )
    return result


def _semantic_group_size_report(group_keys_by_row: Iterable[str]) -> dict[str, Any]:
    sizes_by_group: dict[str, int] = {}
    for group_key in group_keys_by_row:
        sizes_by_group[group_key] = sizes_by_group.get(group_key, 0) + 1
    histogram: dict[int, int] = {}
    for size in sizes_by_group.values():
        histogram[size] = histogram.get(size, 0) + 1
    row_count = sum(sizes_by_group.values())
    multirow_group_count = sum(count for size, count in histogram.items() if size > 1)
    rows_in_multirow_groups = sum(
        size * count for size, count in histogram.items() if size > 1
    )
    return {
        "row_count": row_count,
        "group_count": len(sizes_by_group),
        "group_size_histogram": {
            str(size): histogram[size] for size in sorted(histogram)
        },
        "row_count_by_group_size": {
            str(size): size * histogram[size] for size in sorted(histogram)
        },
        "multirow_group_count": multirow_group_count,
        "rows_in_multirow_groups": rows_in_multirow_groups,
        "rows_in_multirow_groups_rate": (
            rows_in_multirow_groups / row_count if row_count else 0.0
        ),
        "mean_rows_per_group": (
            row_count / len(sizes_by_group) if sizes_by_group else 0.0
        ),
    }


def _split_manifest(task: StreamTask) -> dict[str, Any]:
    def part(examples: Sequence[OfficialExample]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for item in examples:
            counts[item.label] = counts.get(item.label, 0) + 1
        group_keys = sorted(
            {task.group_key_by_example_id[item.example_id] for item in examples}
        )
        group_size_report = _semantic_group_size_report(
            task.group_key_by_example_id[item.example_id] for item in examples
        )
        return {
            "count": len(examples),
            "per_class": counts,
            "ordered_ids_sha256": _ids_sha256(examples),
            "example_ids": [item.example_id for item in examples],
            "source_subset": "train",
            "semantic_group_count": len(group_keys),
            "semantic_group_size_report": group_size_report,
            "semantic_group_keys_sha256": hashlib.sha256(
                json.dumps(group_keys, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
    audit_group_report = _semantic_group_size_report(
        task.group_key_by_example_id[item.example_id] for item in task.audit
    )
    complete_train_group_report = _semantic_group_size_report(
        task.group_key_by_example_id.values()
    )
    return {
        "task": task.name,
        "update": part(task.update),
        "risk_train": part(task.risk),
        "audit": part(task.audit),
        "effective_risk_per_class": task.effective_risk_per_class,
        "effective_audit_per_class": task.effective_audit_per_class,
        "effective_update_per_class": task.effective_update_per_class,
        "acquisition_audit_sampling": (
            "complete-corpus label-conditional multi-row representation target, "
            "seeded SHA-256 hash-random priority within strata, deterministic whole-"
            "group sparse subset DP, and singleton exact-quota fill; no descending-"
            "size or fastest-target priority; primary EM is reweighted by the "
            "complete pinned-train natural label prior"
        ),
        "audit_group_sampling_audit": {
            "selection_policy": task.audit_group_selection_policy,
            "priority_seed": task.audit_group_priority_seed,
            "priority_features": ["seeded_group_hash", "group_key_tiebreak"],
            "representativeness_constraint": (
                "label-conditional multi-row row counts target the complete train corpus; "
                "singletons fill the remaining exact label quotas"
            ),
            "forbidden_priority_features": [
                "descending_group_size_ordering",
                "fastest_exact_target_completion",
            ],
            "audit": audit_group_report,
            "complete_train_corpus": complete_train_group_report,
            "audit_minus_corpus_multirow_row_rate": (
                audit_group_report["rows_in_multirow_groups_rate"]
                - complete_train_group_report["rows_in_multirow_groups_rate"]
            ),
        },
        "train_label_distribution": {
            "counts": dict(task.train_label_counts),
            "natural_label_prior": dict(task.natural_label_prior),
            "sha256": task.train_distribution_sha256,
            "source": "complete pinned train.json before any cap/split",
        },
        "excluded_train_count": task.excluded_train_count,
        "pairwise_disjoint": True,
        "semantic_group_disjoint": True,
        "semantic_group_policy": (
            "QQP normalized-question connected components; BoolQA normalized "
            "passage; MultiRC normalized paragraph+question; row atomic otherwise"
        ),
    }


def official_evaluation_eligible(
    *,
    completed_task_count: int,
    total_task_count: int,
    stop_after_task: int | None,
    gate_passes: Sequence[bool],
    task_names: Sequence[str],
    smoke: bool,
    tiny_random_model: bool,
) -> bool:
    """Admit only a real-model, exact full-Order-4 run with every gate passed."""

    return bool(
        stop_after_task is None
        and tuple(task_names) == tuple(ORDER4_TASK_NAMES)
        and not smoke
        and not tiny_random_model
        and completed_task_count == total_task_count
        and total_task_count == len(ORDER4_TASK_NAMES)
        and len(gate_passes) == total_task_count
        and all(gate_passes)
    )


def train_arm(
    arm: str,
    tasks: Sequence[StreamTask],
    tokenizer: Any,
    *,
    config: HeadroomConfig,
    expected_initial_fingerprint: str | None,
) -> tuple[str, dict[str, Any]]:
    arm_dir = Path(config.output_root) / arm
    arm_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    bank = _build_model(config, tokenizer)
    initial_fingerprint = _adapter_fingerprint(bank)
    if expected_initial_fingerprint is not None and initial_fingerprint != expected_initial_fingerprint:
        raise RuntimeError("initial adapter reservoirs differ across arms")
    payload = bank.payload_audit()
    if not payload.exact_budget or (not config.tiny_random_model and payload.active_atoms != 144):
        raise RuntimeError("canonical uniform rank-four LoRA payload is not exact")
    risk_buffer = HistoryBuffer(
        role=BufferRole.RISK_TRAIN,
        max_records=config.risk_buffer_records,
        base_seed=config.seed,
        stream_id="acq-conditioned:risk",
        allow_task_ids=arm == ORACLE_ARM,
    )
    audit_buffer = HistoryBuffer(
        role=BufferRole.AUDIT,
        max_records=config.audit_buffer_records,
        base_seed=config.seed,
        stream_id="acq-conditioned:audit",
        allow_task_ids=True,
    )
    _atomic_json(
        arm_dir / "manifest.json",
        {
            "format": FORMAT_VERSION,
            "status": "training",
            "arm": arm,
            "initial_adapter_fingerprint": initial_fingerprint,
            "initial_payload": payload.to_dict(),
            "controller_task_id_access": arm == ORACLE_ARM,
            "official_test_access": False,
            "rank_policy": "uniform rank 4 on every q/v projection; frozen mask",
        },
    )
    base_checkpoint = arm_dir / "base_adapter.pt"
    _atomic_torch_save(base_checkpoint, bank.compact_state_dict())
    base_checkpoint_info = {
        "path": str(base_checkpoint.resolve()),
        "bytes": base_checkpoint.stat().st_size,
        "sha256": _sha256_file(base_checkpoint),
        "timing": "before task 1; official test not accessed",
    }
    stages: list[dict[str, Any]] = []
    total = {
        "acquisition_optimizer_steps": 0,
        "consolidation_optimizer_steps": 0,
        "acquisition_current_example_exposures": 0,
        "consolidation_current_example_exposures": 0,
        "consolidation_replay_example_exposures": 0,
        "controller_forward_examples": 0,
        "controller_forward_batches": 0,
        "acquisition_source_tokens": 0,
        "acquisition_target_tokens": 0,
        "consolidation_current_source_tokens": 0,
        "consolidation_current_target_tokens": 0,
        "consolidation_replay_source_tokens": 0,
        "consolidation_replay_target_tokens": 0,
        "controller_forward_source_tokens": 0,
        "controller_forward_target_tokens": 0,
    }
    for task_index, task in enumerate(tasks):
        stage_started = time.perf_counter()
        stage_dir = arm_dir / f"stage_{task_index + 1:02d}_{task.name}"
        stage_dir.mkdir(parents=True, exist_ok=False)
        if task_index == 0:
            incoming_checkpoint_info = {
                **base_checkpoint_info,
                "role": "incoming_pre_task",
                "source": "initial base adapter before task 1",
                "source_stage_index": None,
            }
        else:
            previous = stages[-1]["deployed_checkpoint"]
            incoming_checkpoint_info = {
                **previous,
                "role": "incoming_pre_task",
                "source": "previous stage post-consolidation deployed adapter",
                "source_stage_index": task_index - 1,
            }
        before_audit = evaluate_free_em(
            bank,
            tokenizer,
            task.audit,
            config=config,
            natural_label_prior=task.natural_label_prior,
        )
        pre_risk = _per_example_losses(bank, tokenizer, task.risk, config=config)
        pre_audit_losses = _per_example_losses(bank, tokenizer, task.audit, config=config)
        acquisition_epochs = int(config.acquisition_epochs_by_task[task.name])
        acquisition = train_acquisition(
            bank,
            tokenizer,
            task.update,
            task_index=task_index,
            acquisition_epochs=acquisition_epochs,
            natural_label_prior=task.natural_label_prior,
            config=config,
        )
        after_audit = evaluate_free_em(
            bank,
            tokenizer,
            task.audit,
            config=config,
            natural_label_prior=task.natural_label_prior,
        )
        post_risk = _per_example_losses(bank, tokenizer, task.risk, config=config)
        post_audit_losses = _per_example_losses(bank, tokenizer, task.audit, config=config)
        immediate_checkpoint = stage_dir / "phase_a_immediate_adapter.pt"
        _atomic_torch_save(immediate_checkpoint, bank.compact_state_dict())
        immediate_checkpoint_info = {
            "path": str(immediate_checkpoint.resolve()),
            "bytes": immediate_checkpoint.stat().st_size,
            "sha256": _sha256_file(immediate_checkpoint),
            "timing": "after current-only Phase A; before risk refresh/consolidation",
        }
        gate = acquisition_gate(
            before_audit,
            after_audit,
            threshold=config.min_audit_acquisition,
        )
        gain = float(gate["natural_prior_gain"])
        class_deltas = dict(gate["per_class_recall_delta"])
        min_class_delta = float(gate["min_per_class_recall_delta"])
        gate_pass = bool(gate["pass"])
        risk_records = _make_anchors(
            task.risk,
            pre_risk,
            post_risk,
            task_name=task.name,
            include_task_id=arm == ORACLE_ARM,
            task_gate_pass=gate_pass,
            min_gain=config.min_anchor_loss_gain,
        )
        audit_records = _make_anchors(
            task.audit,
            pre_audit_losses,
            post_audit_losses,
            task_name=task.name,
            include_task_id=True,
            task_gate_pass=gate_pass,
            min_gain=config.min_anchor_loss_gain,
        )
        # Refresh the adversary on historical risk only at the Phase-A model.
        # Current-task anchors are admitted only after consolidation, so the
        # two arms isolate allocation over exactly the same past support.
        plan = build_replay_plan(arm, risk_buffer, bank, tokenizer, config=config)
        current_ids = {item.example_id for item in task.risk}
        if current_ids.intersection(record.example_id for record in plan.records):
            raise RuntimeError("current anchors entered the historical replay plan early")
        acquisition_steps_per_epoch = math.ceil(len(task.update) / config.batch_size)
        consolidation = train_consolidation(
            arm,
            bank,
            tokenizer,
            plan,
            task.update,
            matched_steps_per_epoch=acquisition_steps_per_epoch,
            task_index=task_index,
            natural_label_prior=task.natural_label_prior,
            config=config,
        )
        risk_decisions = risk_buffer.extend(risk_records)
        audit_decisions = audit_buffer.extend(audit_records)
        assert_disjoint_history(risk_buffer, audit_buffer)
        acquisition_ledger = acquisition["resource_ledger"]
        consolidation_ledger = consolidation["resource_ledger"]
        total["acquisition_optimizer_steps"] += int(acquisition_ledger["optimizer_steps"])
        total["consolidation_optimizer_steps"] += int(consolidation_ledger["optimizer_steps"])
        total["acquisition_current_example_exposures"] += int(
            acquisition_ledger["gradient_example_exposures"]
        )
        total["consolidation_current_example_exposures"] += int(
            consolidation_ledger["current_example_exposures"]
        )
        total["consolidation_replay_example_exposures"] += int(
            consolidation_ledger["replay_example_exposures"]
        )
        total["controller_forward_examples"] += int(plan.controller_forward_examples)
        total["controller_forward_batches"] += int(plan.controller_forward_batches)
        total["acquisition_source_tokens"] += int(acquisition_ledger["source_tokens"])
        total["acquisition_target_tokens"] += int(acquisition_ledger["target_tokens"])
        for key in (
            "current_source_tokens",
            "current_target_tokens",
            "replay_source_tokens",
            "replay_target_tokens",
        ):
            total[f"consolidation_{key}"] += int(consolidation_ledger[key])
        total["controller_forward_source_tokens"] += int(
            plan.controller_forward_source_tokens
        )
        total["controller_forward_target_tokens"] += int(
            plan.controller_forward_target_tokens
        )
        checkpoint = stage_dir / "adapter.pt"
        _atomic_torch_save(checkpoint, bank.compact_state_dict())
        deployed_checkpoint_info = {
            "path": str(checkpoint.resolve()),
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256_file(checkpoint),
            "timing": "after Phase-B consolidation; deployed into the next task",
            "role": "post_consolidation_deployed",
            "stage_index": task_index,
        }
        stage = {
            "stage_index": task_index,
            "task": task.name,
            "chronology": [item.name for item in tasks[: task_index + 1]],
            "train_partition": _split_manifest(task),
            "acquisition_gate": {
                "partition": (
                    "train-derived, exactly class-balanced, strict semantic-group-disjoint audit"
                ),
                "metric": (
                    "primary: free-generation per-class recall reweighted by complete "
                    "pinned-train natural label prior"
                ),
                "pre": before_audit,
                "post": after_audit,
                "natural_prior_gain": gain,
                "balanced_gain_auxiliary": float(
                    after_audit["balanced_accuracy_auxiliary"]
                    - before_audit["balanced_accuracy_auxiliary"]
                ),
                "per_class_recall_delta": class_deltas,
                "min_per_class_recall_delta": min_class_delta,
                "threshold": config.min_audit_acquisition,
                "guards": gate["guards"],
                "pass": gate_pass,
                "used_for": (
                    "anchor eligibility, fail-closed training abort, and official-test "
                    "admission; never hyperparameter or epoch selection"
                ),
                "auxiliary_only": [
                    "balanced_accuracy",
                    "absolute worst_class_recall (the recall delta floor is a hard guard)",
                ],
            },
            "incoming_checkpoint": incoming_checkpoint_info,
            "phase_a_immediate_checkpoint": immediate_checkpoint_info,
            "deployed_checkpoint": deployed_checkpoint_info,
            "acquisition_training": acquisition,
            "consolidation_training": consolidation,
            "anchor_admission_timing": "after consolidation; replay plan contains historical tasks only",
            "eligible_new_anchors": {
                "risk": sum(record.eligible for record in risk_records),
                "audit": sum(record.eligible for record in audit_records),
            },
            "audit_after_consolidation": _audit_regret(bank, tokenizer, audit_buffer, config=config),
            "reservoir_decisions": {
                "risk_accepted": sum(decision.accepted for decision in risk_decisions),
                "audit_accepted": sum(decision.accepted for decision in audit_decisions),
            },
            "risk_buffer_ledger": asdict(risk_buffer.byte_ledger()),
            "audit_buffer_ledger": asdict(audit_buffer.byte_ledger()),
            "payload": bank.payload_audit().to_dict(),
            "checkpoint": {
                "path": checkpoint.name,
                "bytes": checkpoint.stat().st_size,
                "sha256": deployed_checkpoint_info["sha256"],
            },
            "official_test_access": False,
            "wall_seconds": time.perf_counter() - stage_started,
        }
        _atomic_json(stage_dir / "stage.json", stage)
        _atomic_json(stage_dir / "risk_buffer.json", risk_buffer.state_dict())
        _atomic_json(stage_dir / "audit_buffer.json", audit_buffer.state_dict())
        stages.append(stage)
        print(
            f"[{arm}] {task_index + 1}/{len(tasks)} {task.name}: "
            f"audit acquisition={gain:+.4f}, gate={'PASS' if gate_pass else 'FAIL'}",
            flush=True,
        )
        if config.fail_on_acquisition and not gate_pass:
            raise RuntimeError(
                f"{arm}/{task.name} acquisition gate failed ({gain:+.4f}); "
                "headroom is invalid and official test remains sealed"
            )
        if config.stop_after_task is not None and task_index + 1 >= config.stop_after_task:
            break
    final_checkpoint = arm_dir / "final_adapter.pt"
    _atomic_torch_save(final_checkpoint, bank.compact_state_dict())
    fully_trained = len(stages) == len(tasks) and config.stop_after_task is None
    all_acquisition_gates_pass = all(
        stage["acquisition_gate"]["pass"] for stage in stages
    )
    official_test_eligible = official_evaluation_eligible(
        completed_task_count=len(stages),
        total_task_count=len(tasks),
        stop_after_task=config.stop_after_task,
        gate_passes=[bool(stage["acquisition_gate"]["pass"]) for stage in stages],
        task_names=config.task_names,
        smoke=config.smoke,
        tiny_random_model=config.tiny_random_model,
    )
    official_scope_eligible = bool(
        tuple(config.task_names) == tuple(ORDER4_TASK_NAMES)
        and not config.smoke
        and not config.tiny_random_model
    )
    training_result = {
        "format": FORMAT_VERSION,
        "status": (
            "training_completed"
            if official_test_eligible
            else "stopped_after_task"
            if not fully_trained
            else "training_completed_acquisition_failed_test_sealed"
            if not all_acquisition_gates_pass
            else "training_completed_nonofficial_test_sealed"
        ),
        "arm": arm,
        "base_checkpoint": base_checkpoint_info,
        "stages": [
            {
                "task": stage["task"],
                "natural_prior_acquisition_gain": stage["acquisition_gate"][
                    "natural_prior_gain"
                ],
                "acquisition_pass": stage["acquisition_gate"]["pass"],
                "acquisition_optimizer_steps": stage["acquisition_training"]["resource_ledger"]["optimizer_steps"],
                "consolidation_optimizer_steps": stage["consolidation_training"]["resource_ledger"]["optimizer_steps"],
                "acquisition_current_example_exposures": stage["acquisition_training"][
                    "resource_ledger"
                ]["gradient_example_exposures"],
                "consolidation_current_example_exposures": stage[
                    "consolidation_training"
                ]["resource_ledger"]["current_example_exposures"],
                "consolidation_replay_example_exposures": stage[
                    "consolidation_training"
                ]["resource_ledger"]["replay_example_exposures"],
                "controller_forward_examples": stage["consolidation_training"][
                    "plan"
                ]["controller_forward_examples"],
                "controller_forward_batches": stage["consolidation_training"][
                    "plan"
                ]["controller_forward_batches"],
                "payload": stage["payload"],
                "incoming_checkpoint": stage["incoming_checkpoint"],
                "phase_a_immediate_checkpoint": stage[
                    "phase_a_immediate_checkpoint"
                ],
                "deployed_checkpoint": stage["deployed_checkpoint"],
            }
            for stage in stages
        ],
        "total_training_budget": total,
        "all_acquisition_gates_pass": all_acquisition_gates_pass,
        "final_payload": bank.payload_audit().to_dict(),
        "final_checkpoint": {
            "path": str(final_checkpoint.resolve()),
            "bytes": final_checkpoint.stat().st_size,
            "sha256": _sha256_file(final_checkpoint),
        },
        "official_test_access": False,
        "official_test_admission": {
            "eligible": official_test_eligible,
            "exact_full_order_required": list(ORDER4_TASK_NAMES),
            "selected_order_is_exact_full_order": tuple(config.task_names)
            == tuple(ORDER4_TASK_NAMES),
            "smoke_forbidden": True,
            "tiny_random_model_forbidden": True,
            "scope_eligible": official_scope_eligible,
        },
        "automatic_stage_resume_implemented": False,
        "stop_after_task": config.stop_after_task,
        "restart_policy": (
            "--stop-after-task creates an auditable prefix only; a later invocation "
            "does not resume and must use a fresh output root"
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(arm_dir / "training_result.json", training_result)
    if official_test_eligible:
        (arm_dir / "TRAINING_COMPLETED").write_text(
            "training completed; official test not accessed\n", encoding="utf-8"
        )
    elif fully_trained and not all_acquisition_gates_pass:
        (arm_dir / "ACQUISITION_FAILED_TEST_SEALED").write_text(
            "training diagnostic completed; at least one acquisition gate failed; official test sealed\n",
            encoding="utf-8",
        )
    elif fully_trained:
        (arm_dir / "NONOFFICIAL_RUN_TEST_SEALED").write_text(
            "selected prefix, smoke, or tiny-random diagnostic completed; official test sealed\n",
            encoding="utf-8",
        )
    else:
        (arm_dir / "STOPPED_AFTER_TASK").write_text(
            f"stopped after {len(stages)} tasks; automatic resume is not implemented\n",
            encoding="utf-8",
        )
    del bank
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return initial_fingerprint, training_result


def assert_matched_training_budgets(
    results: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if tuple(results) != LOCKED_ARMS:
        raise ValueError("canonical headroom requires both locked arms")
    references = results[LOCKED_ARMS[0]]
    reference_payload = references["final_payload"]
    reference_budget = references["total_training_budget"]
    matched_fields = (
        "acquisition_optimizer_steps",
        "consolidation_optimizer_steps",
        "acquisition_current_example_exposures",
        "consolidation_current_example_exposures",
        "consolidation_replay_example_exposures",
        "controller_forward_examples",
        "controller_forward_batches",
    )
    token_fields = tuple(
        key for key in reference_budget if key.endswith("_tokens")
    )
    for arm in LOCKED_ARMS[1:]:
        candidate = results[arm]
        candidate_budget = candidate["total_training_budget"]
        for field in matched_fields:
            if candidate_budget[field] != reference_budget[field]:
                raise RuntimeError(f"arms have unequal matched resource field {field}")
        candidate_steps = [
            (
                item["acquisition_optimizer_steps"],
                item["consolidation_optimizer_steps"],
                item["acquisition_current_example_exposures"],
                item["consolidation_current_example_exposures"],
                item["consolidation_replay_example_exposures"],
                item["controller_forward_examples"],
                item["controller_forward_batches"],
            )
            for item in candidate["stages"]
        ]
        reference_stage_steps = [
            (
                item["acquisition_optimizer_steps"],
                item["consolidation_optimizer_steps"],
                item["acquisition_current_example_exposures"],
                item["consolidation_current_example_exposures"],
                item["consolidation_replay_example_exposures"],
                item["controller_forward_examples"],
                item["controller_forward_batches"],
            )
            for item in references["stages"]
        ]
        if candidate_steps != reference_stage_steps:
            raise RuntimeError("arms have unequal per-task optimizer-step budgets")
        if [item["task"] for item in candidate["stages"]] != [
            item["task"] for item in references["stages"]
        ]:
            raise RuntimeError("arms have unequal stage task identities")
        for reference_stage, candidate_stage in zip(
            references["stages"], candidate["stages"]
        ):
            for field in ("active_atoms", "active_trainable_scalars", "exact_budget"):
                if candidate_stage["payload"][field] != reference_stage["payload"][field]:
                    raise RuntimeError(
                        f"arms have unequal stage LoRA payload field {field}"
                    )
        for field in ("active_atoms", "active_trainable_scalars", "exact_budget"):
            if candidate["final_payload"][field] != reference_payload[field]:
                raise RuntimeError(f"arms have unequal LoRA payload field {field}")
    token_totals = {
        arm: {
            field: int(results[arm]["total_training_budget"][field])
            for field in token_fields
        }
        for arm in LOCKED_ARMS
    }
    token_differences = {
        field: token_totals[ORACLE_ARM][field] - token_totals[LOCKED_ARMS[0]][field]
        for field in token_fields
    }
    token_count_matched = all(value == 0 for value in token_differences.values())
    return {
        "optimizer_steps_matched": True,
        "current_example_exposures_matched": True,
        "replay_example_exposures_matched": True,
        "controller_forward_examples_matched": True,
        "controller_forward_batches_matched": True,
        "lora_payload_matched": True,
        "token_totals": token_totals,
        "oracle_minus_uniform_token_differences": token_differences,
        "token_count_matched": token_count_matched,
        "flop_matched": False,
        "claim": (
            "step/example and aggregate non-pad token-count matched; no FLOP-matched "
            "claim because padded shapes, attention cost, and controller arithmetic are not instrumented"
            if token_count_matched
            else "step/example matched only; replay sampling changed aggregate non-pad token counts; no token- or FLOP-matched claim"
        ),
    }


def _tail_mean(values: Sequence[float], alpha: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.ndim != 1 or ordered.size == 0 or not 0.0 < alpha <= 1.0:
        raise ValueError("tail mean requires non-empty values and alpha in (0,1]")
    mass = alpha * ordered.size
    full = int(math.floor(mass))
    fraction = mass - full
    total = float(ordered[:full].sum())
    if fraction:
        total += fraction * float(ordered[full])
    return total / mass


def offline_retention_vectors(
    incoming_pre_scores: Sequence[float],
    immediate_scores: Sequence[float],
    final_scores: Sequence[float],
    *,
    denominator_floor: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    incoming = np.asarray(incoming_pre_scores, dtype=np.float64)
    immediate = np.asarray(immediate_scores, dtype=np.float64)
    final = np.asarray(final_scores, dtype=np.float64)
    if (
        incoming.ndim != 1
        or incoming.size == 0
        or incoming.shape != immediate.shape
        or incoming.shape != final.shape
        or not np.isfinite(np.concatenate((incoming, immediate, final))).all()
        or denominator_floor <= 0
    ):
        raise ValueError("offline incoming/immediate/final score vectors are invalid")
    acquisition = immediate - incoming
    bwt = final - immediate
    retention = (final - incoming) / np.maximum(acquisition, denominator_floor)
    return acquisition, bwt, retention


def evaluate_all_final_checkpoints(
    training_results: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
    *,
    config: HeadroomConfig,
) -> dict[str, Any]:
    """The executable's sole official-test access point.

    This function fails closed unless a real-model exact full-Order-4 run
    completed every gate and the two arm resource ledgers match.
    No return value from it is consumed by any training or selection function.
    """

    if (
        tuple(config.task_names) != tuple(ORDER4_TASK_NAMES)
        or config.smoke
        or config.tiny_random_model
    ):
        raise RuntimeError(
            "official test is sealed for task prefixes, smoke runs, and tiny-random models"
        )
    for arm in config.arms:
        if training_results[arm].get("status") != "training_completed" or not bool(
            training_results[arm].get("all_acquisition_gates_pass")
        ):
            raise RuntimeError(
                "official test is sealed until every train-derived acquisition gate passes"
            )
        stage_tasks = tuple(
            stage.get("task") for stage in training_results[arm].get("stages", ())
        )
        if stage_tasks != tuple(config.task_names) or not all(
            bool(stage.get("acquisition_pass"))
            for stage in training_results[arm].get("stages", ())
        ):
            raise RuntimeError(
                "official test is sealed until every exact Order-4 stage is present and passed"
            )
    # Recompute this assertion here rather than trusting arm sentinels or main's
    # earlier call.  Arm sentinels are written before the cross-arm comparison,
    # so a failed comparison must remain incapable of opening sealed data.
    assert_matched_training_budgets(training_results)
    for arm in config.arms:
        sentinel = Path(config.output_root) / arm / "TRAINING_COMPLETED"
        if not sentinel.is_file():
            raise RuntimeError("official test cannot open before all arms complete training")

    def validated_checkpoint_path(
        checkpoint_info: Mapping[str, Any], *, description: str
    ) -> Path:
        try:
            checkpoint = Path(str(checkpoint_info["path"]))
            expected_sha256 = str(checkpoint_info["sha256"])
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                f"offline evaluation checkpoint provenance is incomplete: {description}"
            ) from error
        if (
            not checkpoint.is_file()
            or _sha256_file(checkpoint) != expected_sha256
        ):
            raise RuntimeError(
                f"offline evaluation checkpoint hash mismatch: {description}: {checkpoint}"
            )
        return checkpoint

    def checkpoint_identity(checkpoint_info: Mapping[str, Any]) -> tuple[str, str]:
        try:
            path = str(Path(str(checkpoint_info["path"])).resolve())
            sha256 = str(checkpoint_info["sha256"])
        except (KeyError, TypeError) as error:
            raise RuntimeError("checkpoint-chain provenance is incomplete") from error
        return path, sha256

    # Preflight the complete initial/incoming/immediate/deployed/final panel and
    # its chronology before the first sealed-data read.  A broken artifact must
    # not consume official test access and only then discover that evaluation
    # cannot be completed.
    for arm in config.arms:
        training = training_results[arm]
        validated_checkpoint_path(
            training["base_checkpoint"], description=f"{arm}/base"
        )
        for stage_index, stage in enumerate(training["stages"]):
            expected_incoming = (
                training["base_checkpoint"]
                if stage_index == 0
                else training["stages"][stage_index - 1]["deployed_checkpoint"]
            )
            if checkpoint_identity(stage["incoming_checkpoint"]) != checkpoint_identity(
                expected_incoming
            ):
                raise RuntimeError(
                    f"incoming checkpoint chain mismatch: {arm}/{stage['task']}"
                )
            validated_checkpoint_path(
                stage["incoming_checkpoint"],
                description=f"{arm}/{stage['task']}/incoming_pre",
            )
            validated_checkpoint_path(
                stage["phase_a_immediate_checkpoint"],
                description=f"{arm}/{stage['task']}/phase_a_immediate",
            )
            validated_checkpoint_path(
                stage["deployed_checkpoint"],
                description=f"{arm}/{stage['task']}/post_consolidation_deployed",
            )
        validated_checkpoint_path(
            training["final_checkpoint"], description=f"{arm}/final"
        )

    sealed: dict[str, Any] = {}
    for task_index, task_name in enumerate(config.task_names):
        loaded = load_sealed_evaluation_data(config.data_root, task_name)
        labels = TASK_BY_NAME[task_name].labels
        full_test_counts = {
            label: sum(example.label == label for example in loaded.examples)
            for label in labels
        }
        full_test_total = sum(full_test_counts.values())
        full_test_prior = {
            label: full_test_counts[label] / full_test_total for label in labels
        }
        examples = select_evaluation_examples(
            loaded.examples,
            config.eval_per_task,
            seed=_stable_seed(config.seed, "final-test", task_index, task_name),
        )
        sealed[task_name] = {
            "examples": examples,
            "excluded_unlabeled_count": loaded.excluded_unlabeled_count,
            "excluded_unlabeled_sha256": loaded.excluded_unlabeled_sha256,
            "full_labeled_test_counts": full_test_counts,
            "full_labeled_test_natural_prior": full_test_prior,
        }
    def load_bank(checkpoint_info: Mapping[str, Any]) -> T5GlobalRankBank:
        checkpoint = validated_checkpoint_path(
            checkpoint_info, description="post-preflight load"
        )
        loaded_bank = _build_model(config, tokenizer)
        loaded_bank.load_compact_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False)
        )
        return loaded_bank

    def evaluate_tasks(
        bank: T5GlobalRankBank, names: Sequence[str]
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for task_name in names:
            item = sealed[task_name]
            metric = evaluate_free_em(
                bank,
                tokenizer,
                item["examples"],
                config=config,
                natural_label_prior=item["full_labeled_test_natural_prior"],
            )
            metric.update(
                source_subset="test",
                selected_ids_sha256=_ids_sha256(item["examples"]),
                excluded_unlabeled_count=item["excluded_unlabeled_count"],
                excluded_unlabeled_sha256=item["excluded_unlabeled_sha256"],
                full_labeled_test_counts=item["full_labeled_test_counts"],
                score_interpretation=(
                    "normalized_em is direct EM on an approximately class-balanced "
                    "subsample; natural_prior_em reweights its per-class recalls by "
                    "the complete labeled official-test prior"
                ),
            )
            output[task_name] = metric
        return output

    arm_results: dict[str, Any] = {}
    for arm in config.arms:
        training = training_results[arm]
        base_bank = load_bank(training["base_checkpoint"])
        initial_base_evaluations = evaluate_tasks(base_bank, config.task_names)
        del base_bank
        gc.collect()

        incoming_evaluations: dict[str, Any] = {}
        immediate_evaluations: dict[str, Any] = {}
        for stage in training["stages"]:
            task_name = str(stage["task"])
            incoming_bank = load_bank(stage["incoming_checkpoint"])
            incoming_evaluations.update(evaluate_tasks(incoming_bank, (task_name,)))
            del incoming_bank
            gc.collect()
            immediate_bank = load_bank(stage["phase_a_immediate_checkpoint"])
            immediate_evaluations.update(evaluate_tasks(immediate_bank, (task_name,)))
            del immediate_bank
            gc.collect()

        final_bank = load_bank(training["final_checkpoint"])
        final_evaluations = evaluate_tasks(final_bank, config.task_names)
        payload = final_bank.payload_audit().to_dict()
        del final_bank
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        initial_base_scores = np.asarray(
            [
                initial_base_evaluations[name]["natural_prior_em"]
                for name in config.task_names
            ],
            dtype=np.float64,
        )
        incoming_scores = np.asarray(
            [incoming_evaluations[name]["natural_prior_em"] for name in config.task_names],
            dtype=np.float64,
        )
        immediate_scores = np.asarray(
            [immediate_evaluations[name]["natural_prior_em"] for name in config.task_names],
            dtype=np.float64,
        )
        final_scores = np.asarray(
            [final_evaluations[name]["natural_prior_em"] for name in config.task_names],
            dtype=np.float64,
        )
        raw_balanced_final_scores = np.asarray(
            [final_evaluations[name]["normalized_em"] for name in config.task_names],
            dtype=np.float64,
        )
        acquisition, bwt, retention = offline_retention_vectors(
            incoming_scores, immediate_scores, final_scores
        )
        (
            initial_base_acquisition_auxiliary,
            _initial_base_bwt_duplicate,
            initial_base_retention_auxiliary,
        ) = offline_retention_vectors(
            initial_base_scores, immediate_scores, final_scores
        )
        task_metrics = {
            name: {
                "incoming_pre_t": float(incoming_scores[index]),
                "initial_pretrained_base_auxiliary": float(
                    initial_base_scores[index]
                ),
                "phase_a_immediate": float(immediate_scores[index]),
                "final": float(final_scores[index]),
                "final_balanced_subsample_em": float(raw_balanced_final_scores[index]),
                "acquisition": float(acquisition[index]),
                "bwt": float(bwt[index]),
                "normalized_retention": float(retention[index]),
                "initial_base_acquisition_auxiliary": float(
                    initial_base_acquisition_auxiliary[index]
                ),
                "initial_base_normalized_retention_auxiliary": float(
                    initial_base_retention_auxiliary[index]
                ),
            }
            for index, name in enumerate(config.task_names)
        }
        arm_results[arm] = {
            "arm": arm,
            "official_test_initial_pretrained_base_auxiliary": initial_base_evaluations,
            "official_test_incoming_pre_t": incoming_evaluations,
            "official_test_phase_a_immediate": immediate_evaluations,
            "official_test_final": final_evaluations,
            "task_metrics": task_metrics,
            "summary": {
                "mean_final_natural_prior_reweighted_em": float(final_scores.mean()),
                "worst_task_final_natural_prior_reweighted_em": float(final_scores.min()),
                "mean_final_balanced_subsample_em": float(
                    raw_balanced_final_scores.mean()
                ),
                "mean_acquisition": float(acquisition.mean()),
                "mean_bwt": float(bwt.mean()),
                "min_normalized_retention": float(retention.min()),
                "worst_two_normalized_retention": float(
                    np.sort(retention)[: min(2, len(retention))].mean()
                ),
                "lower_tail_normalized_retention": _tail_mean(
                    retention.tolist(), config.cvar_alpha
                ),
                "mean_initial_base_acquisition_auxiliary": float(
                    initial_base_acquisition_auxiliary.mean()
                ),
                "lower_tail_initial_base_normalized_retention_auxiliary": _tail_mean(
                    initial_base_retention_auxiliary.tolist(), config.cvar_alpha
                ),
            },
            "retention_definition": (
                "natural-prior-reweighted official-subsample scores: "
                "R_t=(final_t-incoming_pre_t)/max(phase_a_post_t-incoming_pre_t,0.05); "
                "incoming_pre_t is initial before task 1 and the previous stage's "
                "post-consolidation deployed checkpoint thereafter"
            ),
            "initial_base_retention_status": (
                "auxiliary only; never used for primary retention or oracle headroom"
            ),
            "test_access_timing": "offline, after every arm TRAINING_COMPLETED",
            "test_used_for_selection_or_training": False,
            "payload": payload,
        }
        _atomic_json(Path(config.output_root) / arm / "result.json", arm_results[arm])
        (Path(config.output_root) / arm / "COMPLETED").write_text(
            "offline initial/incoming/immediate/final evaluation completed\n",
            encoding="utf-8",
        )
    group_free = arm_results[LOCKED_ARMS[0]]["summary"]
    oracle = arm_results[ORACLE_ARM]["summary"]
    headroom = {
        "definition": (
            "task-ID oracle DRO minus group-free uniform ER at identical "
            "optimizer-step/example-exposure/LoRA-rank budgets"
        ),
        "lower_tail_normalized_retention": (
            oracle["lower_tail_normalized_retention"]
            - group_free["lower_tail_normalized_retention"]
        ),
        "min_normalized_retention": (
            oracle["min_normalized_retention"]
            - group_free["min_normalized_retention"]
        ),
        "mean_bwt": oracle["mean_bwt"] - group_free["mean_bwt"],
        "final_natural_prior_reweighted_em_gap_not_retention": (
            oracle["mean_final_natural_prior_reweighted_em"]
            - group_free["mean_final_natural_prior_reweighted_em"]
        ),
        "raw_balanced_subsample_final_em_gap_not_retention": (
            oracle["mean_final_balanced_subsample_em"]
            - group_free["mean_final_balanced_subsample_em"]
        ),
        "raw_accuracy_warning": (
            "Final-score gaps are reported separately and are not retention metrics; "
            "direct normalized_em is on the approximately class-balanced selected panel."
        ),
    }
    return {"arms": arm_results, "oracle_headroom": headroom}


def _model_provenance(model_path: str) -> dict[str, Any]:
    root = Path(model_path).resolve()
    files = {}
    for name in ("config.json", "generation_config.json", "pytorch_model.bin", "model.safetensors", "spiece.model", "tokenizer.json", "tokenizer_config.json"):
        path = root / name
        if path.is_file():
            files[name] = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    return {"local_path": str(root), "files": files}


def _environment(config: HeadroomConfig) -> dict[str, Any]:
    versions = {}
    for package in ("torch", "transformers", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "device": config.device,
        "cuda_available": torch.cuda.is_available(),
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_epoch_overrides(
    values: Sequence[str], task_names: Sequence[str], default_epochs: int
) -> dict[str, int]:
    schedule = {task: int(default_epochs) for task in task_names}
    lookup = {task.lower(): task for task in ORDER4_TASK_NAMES}
    seen: set[str] = set()
    for value in values:
        for assignment in (part.strip() for part in value.split(",") if part.strip()):
            if "=" not in assignment:
                raise ValueError("epoch overrides must use Task=epochs")
            raw_task, raw_epochs = assignment.split("=", 1)
            try:
                task = lookup[raw_task.strip().lower()]
                epochs = int(raw_epochs)
            except (KeyError, ValueError) as error:
                raise ValueError(f"invalid acquisition epoch override {assignment!r}") from error
            if task not in schedule:
                continue
            if task in seen or epochs <= 0:
                raise ValueError(f"invalid or duplicate acquisition epoch override {assignment!r}")
            seen.add(task)
            schedule[task] = epochs
    return schedule


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--arms", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--update-cap-per-class", type=_positive_int, default=96)
    parser.add_argument("--risk-per-class", type=_positive_int, default=16)
    parser.add_argument("--acquisition-audit-examples", type=_positive_int, default=64)
    parser.add_argument("--min-update-per-class", type=_positive_int, default=4)
    parser.add_argument("--eval-per-task", type=_positive_int, default=200)
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument("--replay-batch-size", type=_positive_int, default=4)
    parser.add_argument("--default-acquisition-epochs", type=_positive_int, default=1)
    parser.add_argument(
        "--acquisition-epoch-overrides",
        nargs="+",
        default=["MultiRC=2"],
        help="task-specific frozen schedule, e.g. MultiRC=2",
    )
    parser.add_argument("--consolidation-epochs", type=_positive_int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-source-length", type=_positive_int, default=512)
    parser.add_argument("--max-target-length", type=_positive_int, default=50)
    parser.add_argument("--risk-buffer-records", type=_positive_int, default=512)
    parser.add_argument("--audit-buffer-records", type=_positive_int, default=512)
    parser.add_argument("--allow-failed-acquisition", action="store_true")
    parser.add_argument(
        "--stop-after-task",
        help=(
            "stop after this Order-4 prefix count or task name; writes prefix "
            "checkpoints but does not implement automatic resume or open official test"
        ),
    )
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--tiny-random-model", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> HeadroomConfig:
    tasks = parse_task_selection(args.tasks)
    arms = parse_arm_selection(args.arms)
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise ValueError("model-path must be an existing local model directory")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    epoch_schedule = parse_epoch_overrides(
        args.acquisition_epoch_overrides, tasks, args.default_acquisition_epochs
    )
    stop_after_task: int | None = None
    if args.stop_after_task is not None:
        raw_stop = str(args.stop_after_task).strip()
        if raw_stop.isdigit():
            stop_after_task = int(raw_stop)
        else:
            lookup = {task.lower(): index + 1 for index, task in enumerate(tasks)}
            try:
                stop_after_task = lookup[raw_stop.lower()]
            except KeyError as error:
                raise ValueError("--stop-after-task must name a selected prefix task") from error
        if not 1 <= stop_after_task <= len(tasks):
            raise ValueError("--stop-after-task lies outside the selected task prefix")
    values = {
        "update_cap_per_class": args.update_cap_per_class,
        "risk_per_class": args.risk_per_class,
        "acquisition_audit_examples": args.acquisition_audit_examples,
        "min_update_per_class": args.min_update_per_class,
        "eval_per_task": args.eval_per_task,
        "batch_size": args.batch_size,
        "replay_batch_size": args.replay_batch_size,
        "consolidation_epochs": args.consolidation_epochs,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
    }
    if args.smoke:
        values.update(
            update_cap_per_class=min(values["update_cap_per_class"], 4),
            risk_per_class=min(values["risk_per_class"], 1),
            acquisition_audit_examples=min(values["acquisition_audit_examples"], 4),
            min_update_per_class=min(values["min_update_per_class"], 2),
            eval_per_task=min(values["eval_per_task"], 4),
            batch_size=min(values["batch_size"], 2),
            replay_batch_size=min(values["replay_batch_size"], 2),
            consolidation_epochs=1,
            max_source_length=min(values["max_source_length"], 64),
            max_target_length=min(values["max_target_length"], 8),
        )
    official_capable_scope = bool(
        tuple(tasks) == tuple(ORDER4_TASK_NAMES)
        and not args.smoke
        and not args.tiny_random_model
    )
    required_eval_examples = max(len(TASK_BY_NAME[task].labels) for task in tasks)
    if official_capable_scope and values["eval_per_task"] < required_eval_examples:
        raise ValueError(
            "--eval-per-task must be at least the maximum selected label count "
            f"({required_eval_examples}) for official natural-prior reweighting"
        )
    if values["replay_batch_size"] < 2:
        raise ValueError("replay-batch-size must be at least 2 for uniform+risk replay")
    return HeadroomConfig(
        data_root=str(Path(args.data_root).expanduser().resolve()),
        model_path=str(model_path),
        output_root=str(Path(args.output_root).expanduser().resolve()),
        task_names=tasks,
        arms=arms,
        seed=args.seed,
        default_acquisition_epochs=(1 if args.smoke else args.default_acquisition_epochs),
        acquisition_epochs_by_task=(
            {task: 1 for task in tasks} if args.smoke else epoch_schedule
        ),
        lr=float(args.lr),
        cvar_alpha=0.2,
        risk_fraction=0.5,
        min_audit_acquisition=0.05,
        min_anchor_loss_gain=0.05,
        risk_buffer_records=args.risk_buffer_records,
        audit_buffer_records=args.audit_buffer_records,
        fail_on_acquisition=not bool(args.allow_failed_acquisition),
        stop_after_task=stop_after_task,
        tiny_random_model=bool(args.tiny_random_model),
        smoke=bool(args.smoke),
        device="cuda" if torch.cuda.is_available() else "cpu",
        **values,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    if config.arms != LOCKED_ARMS:
        parser.error("oracle headroom requires --arms all (both locked arms)")
    output_root = Path(config.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing non-empty output root {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if args.download_data:
        download_official_data(config.data_root, config.task_names)
    tasks = prepare_train_stream(config)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True, local_files_only=True)
    runner_path = Path(__file__).resolve()
    root_manifest = {
        "format": FORMAT_VERSION,
        "status": "training",
        "config": asdict(config),
        "official_protocol": {
            "repository": OFFICIAL_REPOSITORY,
            "commit": OFFICIAL_COMMIT,
            "order": list(config.task_names),
            "official_test_requires_exact_order": list(ORDER4_TASK_NAMES),
            "selected_order_is_exact_full_order": tuple(config.task_names)
            == tuple(ORDER4_TASK_NAMES),
            "smoke_and_tiny_random_test_access": "forbidden",
            "model_scope": "T5-small feasibility; not T5-large paper-number comparability",
        },
        "causal_protocol": {
            "phase_order": [
                "current-only acquisition",
                "measure acquisition anchors and refresh historical risk",
                "current+historical consolidation",
                "admit current anchors",
            ],
            "fixed_acquisition_schedule": {
                "lr": config.lr,
                "update_cap_per_class": config.update_cap_per_class,
                "epochs_by_task": dict(config.acquisition_epochs_by_task),
                "source": (
                    "QQP/BoolQA=1 and MultiRC=2 reproduce row-held-out rescue selections; "
                    "other tasks=1 official-development schedule; no official test selection"
                ),
                "qualification": (
                    "the old rescue confirmations were not a jointly strict prerequisite; "
                    "this runner draws an exactly balanced, semantic-group-disjoint audit "
                    "and reweights per-class recall by the complete-train natural prior"
                ),
            },
            "optimizer_reset": "fresh at every task acquisition and consolidation phase",
            "source_packing": "field-aware 512; protected BoolQA question and MultiRC question/candidate suffix",
            "test_access": (
                "only for exact full Order-4, non-smoke, non-tiny T5-small runs, "
                "after all gates, recomputed cross-arm resource parity, and "
                "all TRAINING_COMPLETED sentinels"
            ),
            "test_used_for_selection": False,
            "audit_gradient_access": False,
            "acquisition_loss": (
                "sequence_mean_prior: mean(pi(y)/q(y) * per-example target-sequence "
                "mean NLL), pi from complete train and q from exact balanced update"
            ),
            "gate": (
                "natural-prior gain strictly >0.05; post valid-label rate >=0.99; "
                "minimum per-class recall delta >=-0.05"
            ),
            "resume": (
                "automatic stage resume is not implemented; --stop-after-task writes "
                "an auditable prefix and keeps official test sealed"
            ),
        },
        "comparison": {
            "uniform_er_rank4": "deployable uniform replay without task IDs",
            "task_id_oracle": "privileged worst-task DRO replay",
            "matched": [
                "optimizer steps per task/phase",
                "current and replay example exposures",
                "full-buffer controller forward examples",
                "LoRA rank and active atoms",
                "record cap",
            ],
            "token_and_flop_claim": (
                "aggregate non-pad token-count equality is reported separately; no "
                "FLOP-matched claim without padded-shape/attention/controller instrumentation"
            ),
        },
        "splits": {task.name: _split_manifest(task) for task in tasks},
        "source_token_lengths": {
            task.name: {
                part: _source_token_length_report(
                    tokenizer,
                    examples,
                    max_source_length=config.max_source_length,
                )
                for part, examples in (("update", task.update), ("risk_train", task.risk), ("audit", task.audit))
            }
            for task in tasks
        },
        "provenance": {
            "runner": {"path": str(runner_path), "sha256": _sha256_file(runner_path)},
            "model": _model_provenance(config.model_path),
            "environment": _environment(config),
            "rng": "SHA-256-derived seeds with numpy PCG64; arm-independent partition/admission streams",
        },
    }
    _atomic_json(output_root / "manifest.json", root_manifest)
    expected: str | None = None
    training_results: dict[str, dict[str, Any]] = {}
    for arm in config.arms:
        expected, result = train_arm(
            arm,
            tasks,
            tokenizer,
            config=config,
            expected_initial_fingerprint=expected,
        )
        training_results[arm] = result
    fairness = assert_matched_training_budgets(training_results)
    _atomic_json(output_root / "resource_fairness.json", fairness)
    if any(result["status"] != "training_completed" for result in training_results.values()):
        stopped = any(
            result["status"] == "stopped_after_task"
            for result in training_results.values()
        )
        acquisition_failed = any(
            result["status"]
            == "training_completed_acquisition_failed_test_sealed"
            for result in training_results.values()
        )
        if stopped:
            root_manifest["status"] = "stopped_after_task_test_still_sealed"
            sentinel_name = "STOPPED_AFTER_TASK"
            sentinel_text = (
                "auditable prefix completed; automatic resume is not implemented; "
                "official test sealed\n"
            )
        elif acquisition_failed:
            root_manifest["status"] = "acquisition_failed_test_still_sealed"
            sentinel_name = "ACQUISITION_FAILED_TEST_SEALED"
            sentinel_text = (
                "training diagnostics completed; acquisition prerequisite failed; "
                "official test sealed\n"
            )
        else:
            root_manifest["status"] = "nonofficial_run_test_still_sealed"
            sentinel_name = "NONOFFICIAL_RUN_TEST_SEALED"
            sentinel_text = (
                "prefix, smoke, or tiny-random diagnostic completed; official test sealed\n"
            )
        root_manifest["resource_fairness"] = fairness
        _atomic_json(output_root / "manifest.json", root_manifest)
        (output_root / sentinel_name).write_text(sentinel_text, encoding="utf-8")
        return 0
    root_manifest["status"] = "training_completed_test_still_sealed"
    root_manifest["matched_training_budget_verified"] = True
    root_manifest["resource_fairness"] = fairness
    _atomic_json(output_root / "manifest.json", root_manifest)
    final = evaluate_all_final_checkpoints(training_results, tokenizer, config=config)
    result = {
        "format": FORMAT_VERSION,
        "status": "completed",
        "task_names": list(config.task_names),
        "matched_training_budget_verified": True,
        "resource_fairness": fairness,
        **final,
    }
    _atomic_json(output_root / "result.json", result)
    root_manifest["status"] = "completed"
    root_manifest["oracle_headroom"] = final["oracle_headroom"]
    _atomic_json(output_root / "manifest.json", root_manifest)
    (output_root / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
