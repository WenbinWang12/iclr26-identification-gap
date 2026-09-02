"""Executable T5-small Order-4 five-arm Phase-2I headroom panel.

This runner is deliberately a *feasibility panel*, not an implementation of
O-LoRA or OA-Adapter.  It locks the official O-LoRA Order-4 data, prompts, and
chronology while comparing five controllers under one physical q/v LoRA atom
budget.  The adaptive controllers use a deterministic, one-atom gradient/norm
proxy exchange.  Every exchange is labelled as such in the output; it is not
presented as the paper's final measured shortlist/probe controller.

Chronology and leakage boundaries are enforced in code:

* risk-train and audit examples are hash-split from ``train.json``;
* acquisition ``loss_pre`` is measured before the task's update epoch;
* audit examples are never exposed to a gradient helper;
* group-free controller views contain no task IDs;
* ``test.json`` is touched only by the evaluator; and
* ``COMPLETED`` is written only after all arm artifacts and ``result.json``.

The default model is a local or Hugging Face T5-small checkpoint.  The
``--tiny-random-model`` option exists solely for integration tests; such runs
are marked non-canonical and do not have the 144-atom T5-small budget.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

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
        OfficialExample,
        download_official_data,
        load_sealed_evaluation_data,
        normalized_em,
        prepare_task_partitions,
    )
    from .panel_report import (
        LOCKED_ARMS,
        aggregate_completed_run,
        summarize_arm,
        write_summary,
    )
    from .t5_rank_bank import T5GlobalRankBank
except ImportError:  # pragma: no cover - permits direct ``python file.py`` use.
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
        OfficialExample,
        download_official_data,
        load_sealed_evaluation_data,
        normalized_em,
        prepare_task_partitions,
    )
    from panel_report import (  # type: ignore
        LOCKED_ARMS,
        aggregate_completed_run,
        summarize_arm,
        write_summary,
    )
    from t5_rank_bank import T5GlobalRankBank  # type: ignore


ADAPTIVE_RANK_ARMS = frozenset({"mean_rank_er", "oracle_dro_rank", "cvar_rank"})
ORACLE_ARMS = frozenset({"oracle_dro_rank"})
CVAR_ARMS = frozenset({"cvar_uniform", "cvar_rank"})
UNIFORM_RANK_ARMS = frozenset({"fixed_er", "cvar_uniform"})
FORMAT_VERSION = "phase2i.order4-five-arm-panel.v1"


@dataclass(frozen=True)
class PanelConfig:
    data_root: str
    model_path: str
    output_root: str
    task_names: tuple[str, ...]
    arms: tuple[str, ...]
    seed: int
    per_class_cap: int
    risk_per_class: int
    audit_per_class: int
    eval_per_task: int
    batch_size: int
    replay_batch_size: int
    risk_buffer_records: int
    audit_buffer_records: int
    risk_buffer_bytes: int | None
    audit_buffer_bytes: int | None
    lr: float
    max_source_length: int
    max_target_length: int
    cvar_alpha: float
    min_acquired_gain: float
    risk_fraction: float
    max_rank: int
    initial_rank: int
    lora_alpha: float
    reference_rank: int
    tiny_random_model: bool
    smoke: bool
    device: str


@dataclass(frozen=True)
class PreparedTask:
    name: str
    update: tuple[OfficialExample, ...]
    risk: tuple[OfficialExample, ...]
    audit: tuple[OfficialExample, ...]
    evaluation: tuple[OfficialExample, ...]
    excluded_unlabeled_test_count: int
    excluded_unlabeled_test_sha256: str


@dataclass(frozen=True)
class RiskPlan:
    """Detached full-buffer adversarial distribution for one task episode."""

    records: tuple[AnchorRecord | ControllerRecord, ...]
    probabilities: tuple[float, ...] | None
    eligible_count: int
    mode: str
    refreshed: str


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


def _stable_seed(base_seed: int, *parts: object) -> int:
    material = "\0".join([str(int(base_seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_task_selection(value: str) -> tuple[str, ...]:
    """Parse ``all``, a numeric prefix length, or an exact Order-4 prefix."""

    stripped = value.strip()
    if stripped.lower() in {"all", "all15", "15"}:
        return ORDER4_TASK_NAMES
    if stripped.isdigit():
        count = int(stripped)
        if not 1 <= count <= len(ORDER4_TASK_NAMES):
            raise ValueError(f"task prefix must lie in [1, {len(ORDER4_TASK_NAMES)}]")
        return ORDER4_TASK_NAMES[:count]
    requested = tuple(item.strip() for item in stripped.split(",") if item.strip())
    if not requested:
        raise ValueError("--tasks cannot be empty")
    canonical_lookup = {name.lower(): name for name in ORDER4_TASK_NAMES}
    try:
        canonical = tuple(canonical_lookup[item.lower()] for item in requested)
    except KeyError as error:
        raise ValueError(f"unknown Order-4 task {error.args[0]!r}") from error
    expected = ORDER4_TASK_NAMES[: len(canonical)]
    if canonical != expected:
        raise ValueError(
            "--tasks must be a chronology-preserving Order-4 prefix; "
            f"expected {expected}, got {canonical}"
        )
    return canonical


def parse_arm_selection(values: Sequence[str]) -> tuple[str, ...]:
    flattened = [part.strip() for value in values for part in value.split(",") if part.strip()]
    if len(flattened) == 1 and flattened[0].lower() == "all":
        return LOCKED_ARMS
    unknown = sorted(set(flattened).difference(LOCKED_ARMS))
    if unknown:
        raise ValueError(f"unknown arms: {unknown}; allowed={LOCKED_ARMS}")
    if len(set(flattened)) != len(flattened) or not flattened:
        raise ValueError("--arms must be non-empty and contain no duplicates")
    chosen = set(flattened)
    return tuple(arm for arm in LOCKED_ARMS if arm in chosen)


def _example_rank(example: OfficialExample, seed: int, purpose: str) -> str:
    material = f"{seed}\0{purpose}\0{example.example_id}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def select_evaluation_examples(
    examples: Sequence[OfficialExample], count: int, *, seed: int
) -> tuple[OfficialExample, ...]:
    """Select a deterministic, approximately class-balanced sealed test subset."""

    if count <= 0:
        raise ValueError("eval_per_task must be positive")
    if not examples:
        raise ValueError("cannot select from an empty test split")
    groups: dict[str, list[OfficialExample]] = {}
    for example in examples:
        if example.subset != "test":
            raise ValueError("evaluation examples must come only from test.json")
        groups.setdefault(example.label, []).append(example)
    labels = sorted(groups)
    target = min(int(count), len(examples))
    base, remainder = divmod(target, len(labels))
    selected: list[OfficialExample] = []
    leftovers: list[OfficialExample] = []
    for label_index, label in enumerate(labels):
        ranked = sorted(
            groups[label],
            key=lambda item: (_example_rank(item, seed, "sealed-eval"), item.example_id),
        )
        quota = base + int(label_index < remainder)
        selected.extend(ranked[:quota])
        leftovers.extend(ranked[quota:])
    if len(selected) < target:
        leftovers.sort(
            key=lambda item: (_example_rank(item, seed, "sealed-eval-fill"), item.example_id)
        )
        selected.extend(leftovers[: target - len(selected)])
    return tuple(sorted(selected, key=lambda item: item.source_index))


def prepare_data(config: PanelConfig) -> tuple[PreparedTask, ...]:
    prepared: list[PreparedTask] = []
    for task_index, task_name in enumerate(config.task_names):
        partitions = prepare_task_partitions(
            config.data_root,
            task_name,
            cap_per_class=config.per_class_cap,
            risk_per_class=config.risk_per_class,
            audit_per_class=config.audit_per_class,
            seed=_stable_seed(config.seed, "partition", task_index, task_name),
        )
        sealed_test = load_sealed_evaluation_data(config.data_root, task_name)
        evaluation = select_evaluation_examples(
            sealed_test.examples,
            config.eval_per_task,
            seed=_stable_seed(config.seed, "evaluation", task_index, task_name),
        )
        prepared.append(
            PreparedTask(
                name=task_name,
                update=partitions.update,
                risk=partitions.risk,
                audit=partitions.audit,
                evaluation=evaluation,
                excluded_unlabeled_test_count=sealed_test.excluded_unlabeled_count,
                excluded_unlabeled_test_sha256=sealed_test.excluded_unlabeled_sha256,
            )
        )
    return tuple(prepared)


def _chunks(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _tokenize(
    tokenizer: Any,
    prompts: Sequence[str],
    targets: Sequence[str],
    *,
    max_source_length: int,
    max_target_length: int,
    device: torch.device,
) -> dict[str, Tensor]:
    encoded = tokenizer(
        list(prompts),
        padding=True,
        truncation=True,
        max_length=max_source_length,
        return_tensors="pt",
    )
    encoded_targets = tokenizer(
        text_target=list(targets),
        padding=True,
        truncation=True,
        max_length=max_target_length,
        return_tensors="pt",
    )
    labels = encoded_targets["input_ids"]
    labels = labels.masked_fill(labels.eq(tokenizer.pad_token_id), -100)
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "labels": labels.to(device),
    }


def _per_example_losses(
    bank: T5GlobalRankBank,
    tokenizer: Any,
    records: Sequence[OfficialExample | AnchorRecord | ControllerRecord],
    *,
    config: PanelConfig,
) -> tuple[float, ...]:
    if not records:
        return ()
    device = torch.device(config.device)
    was_training = bank.training
    bank.eval()
    values: list[float] = []
    with torch.no_grad():
        for batch in _chunks(records, config.batch_size):
            prompts = [record.prompt for record in batch]
            targets = [record.label if isinstance(record, OfficialExample) else record.target for record in batch]
            encoded = _tokenize(
                tokenizer,
                prompts,
                targets,
                max_source_length=config.max_source_length,
                max_target_length=config.max_target_length,
                device=device,
            )
            output = bank(**encoded)
            nll = per_example_target_token_nll(output.logits, encoded["labels"])
            values.extend(float(value) for value in nll.detach().cpu())
    bank.train(was_training)
    return tuple(values)


def evaluate_em(
    bank: T5GlobalRankBank,
    tokenizer: Any,
    examples: Sequence[OfficialExample],
    *,
    config: PanelConfig,
) -> tuple[float, tuple[str, ...]]:
    if not examples:
        raise ValueError("evaluation subset is empty")
    device = torch.device(config.device)
    was_training = bank.training
    bank.eval()
    predictions: list[str] = []
    with torch.no_grad():
        for batch in _chunks(examples, config.batch_size):
            encoded = tokenizer(
                [example.prompt for example in batch],
                padding=True,
                truncation=True,
                max_length=config.max_source_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            generated = bank.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_target_length,
                num_beams=1,
                do_sample=False,
            )
            predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    bank.train(was_training)
    references = [example.label for example in examples]
    return normalized_em(predictions, references) / 100.0, tuple(predictions)


def _controller_records(
    records: Sequence[AnchorRecord], *, oracle: bool
) -> tuple[AnchorRecord | ControllerRecord, ...]:
    if oracle:
        if any(record.task_id is None for record in records):
            raise RuntimeError("oracle buffer record is missing its task ID")
        return tuple(records)
    views = tuple(record.controller_view() for record in records)
    if any(hasattr(view, "task_id") for view in views):
        raise RuntimeError("group-free controller view leaked task metadata")
    return views


def build_risk_plan(
    arm: str,
    risk_buffer: HistoryBuffer,
    bank: T5GlobalRankBank,
    tokenizer: Any,
    *,
    config: PanelConfig,
) -> RiskPlan:
    resident = risk_buffer.records_for_evaluation()
    controller_records = _controller_records(resident, oracle=arm in ORACLE_ARMS)
    if not controller_records:
        return RiskPlan((), None, 0, "empty", "once_per_task_pre_update")
    if arm not in ORACLE_ARMS and arm not in CVAR_ARMS:
        return RiskPlan(
            controller_records,
            None,
            sum(record.eligible for record in controller_records),
            "uniform_er",
            "once_per_task_pre_update",
        )

    loss_now = torch.tensor(
        _per_example_losses(bank, tokenizer, controller_records, config=config),
        dtype=torch.float32,
    )
    anchored = differentiable_anchored_regret(
        loss_now,
        [record.loss_pre for record in controller_records],
        [record.loss_post for record in controller_records],
        eligibility=[record.eligible for record in controller_records],
        min_acquired_gain=config.min_acquired_gain,
    )
    eligible_count = int(anchored.eligible.sum().item())
    if eligible_count == 0:
        return RiskPlan(
            controller_records,
            None,
            0,
            "uniform_fallback_no_eligible_anchor",
            "once_per_task_pre_update",
        )
    if arm in ORACLE_ARMS:
        assert all(isinstance(record, AnchorRecord) for record in controller_records)
        probabilities = detached_worst_group_dro_weights(
            anchored.regret,
            [record.task_id for record in controller_records],  # type: ignore[attr-defined]
            eligible=anchored.eligible,
        )
        mode = "task_id_oracle_worst_group_dro"
    else:
        probabilities = detached_cvar_weights(
            anchored.regret,
            config.cvar_alpha,
            eligible=anchored.eligible,
        )
        mode = "group_free_empirical_cvar"
    return RiskPlan(
        controller_records,
        tuple(float(value) for value in probabilities.cpu()),
        eligible_count,
        mode,
        "once_per_task_pre_update",
    )


def _sample_indices(
    count: int,
    size: int,
    *,
    rng: np.random.Generator,
    probabilities: Sequence[float] | None = None,
) -> list[int]:
    if count <= 0 or size <= 0:
        return []
    p = None if probabilities is None else np.asarray(probabilities, dtype=np.float64)
    if p is not None:
        if p.shape != (count,) or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("invalid replay probabilities")
        total = float(p.sum())
        if total <= 0:
            p = None
        else:
            p = p / total
    return [int(index) for index in rng.choice(count, size=size, replace=True, p=p)]


def _training_replay_records(
    arm: str,
    plan: RiskPlan,
    *,
    replay_batch_size: int,
    risk_fraction: float,
    rng: np.random.Generator,
) -> tuple[
    tuple[AnchorRecord | ControllerRecord, ...],
    tuple[AnchorRecord | ControllerRecord, ...],
]:
    """Return separate uniform-coverage and adversarial-risk samples."""

    if not plan.records or replay_batch_size <= 0:
        return (), ()
    if arm not in ORACLE_ARMS and arm not in CVAR_ARMS:
        uniform_size = replay_batch_size
        risk_size = 0
    elif plan.probabilities is None:
        uniform_size = replay_batch_size
        risk_size = 0
    else:
        risk_size = max(1, int(round(replay_batch_size * risk_fraction)))
        risk_size = min(risk_size, replay_batch_size - 1) if replay_batch_size > 1 else 1
        uniform_size = replay_batch_size - risk_size
    uniform_indices = _sample_indices(len(plan.records), uniform_size, rng=rng)
    risk_indices = _sample_indices(
        len(plan.records), risk_size, rng=rng, probabilities=plan.probabilities
    )
    return (
        tuple(plan.records[index] for index in uniform_indices),
        tuple(plan.records[index] for index in risk_indices),
    )


def _accumulate_gradient_scores(bank: T5GlobalRankBank, scores: Tensor) -> None:
    with torch.no_grad():
        for layer_index, adapter in enumerate(bank.adapters):
            squared = torch.zeros((), dtype=torch.float64)
            for atom_index in adapter.active_indices():
                atom = adapter.atoms[atom_index]
                for parameter in (atom.a, atom.b):
                    if parameter.grad is not None:
                        squared += parameter.grad.detach().double().square().sum().cpu()
            scores[layer_index] += squared.sqrt()


def proxy_rank_exchange(
    bank: T5GlobalRankBank,
    optimizer: torch.optim.Optimizer,
    gradient_scores: Tensor,
    *,
    arm: str,
    stage_index: int,
) -> dict[str, Any]:
    """Apply one deterministic exact-budget proxy exchange when justified.

    This intentionally performs no audit-set tuning and no claim of measured
    counterfactual acceptance.  The recipient is the highest accumulated
    training-objective gradient layer; the donor is the lowest score layer and
    contributes its minimum update-norm active atom.
    """

    event: dict[str, Any] = {
        "stage_index": stage_index,
        "arm": arm,
        "mechanism": "one_atom_combined_training_gradient_norm_proxy",
        "measured_probe": False,
        "accepted": False,
        "reason": "uniform_rank_policy",
    }
    if arm not in ADAPTIVE_RANK_ARMS:
        return event
    if gradient_scores.numel() != len(bank.adapters):
        raise ValueError("gradient score vector does not match adapter count")
    scores = gradient_scores.detach().double().cpu()
    ranks = torch.tensor([adapter.active_rank for adapter in bank.adapters])
    donor_candidates = [index for index, rank in enumerate(ranks.tolist()) if rank > 1]
    recipient_candidates = [
        index for index, rank in enumerate(ranks.tolist()) if rank < bank.max_rank
    ]
    if not donor_candidates or not recipient_candidates:
        event["reason"] = "no_feasible_exact_budget_exchange"
        return event
    donor = min(donor_candidates, key=lambda index: (float(scores[index]), index))
    recipient_pool = [index for index in recipient_candidates if index != donor]
    if not recipient_pool:
        event["reason"] = "no_distinct_recipient"
        return event
    recipient = max(recipient_pool, key=lambda index: (float(scores[index]), -index))
    donor_score = float(scores[donor])
    recipient_score = float(scores[recipient])
    event.update(
        {
            "donor_layer": bank.adapter_names[donor],
            "recipient_layer": bank.adapter_names[recipient],
            "donor_score": donor_score,
            "recipient_score": recipient_score,
        }
    )
    if not math.isfinite(donor_score + recipient_score) or recipient_score <= donor_score:
        event["reason"] = "non_positive_proxy_margin"
        return event

    donor_adapter = bank.adapters[donor]
    recipient_adapter = bank.adapters[recipient]
    active_indices = donor_adapter.active_indices()
    utilities = donor_adapter.atom_utility(active_only=True).detach().cpu().tolist()
    donor_slot = min(
        zip(active_indices, utilities), key=lambda item: (float(item[1]), int(item[0]))
    )[0]
    recipient_slot = next(
        index for index in range(bank.max_rank) if not bool(recipient_adapter.active_mask[index])
    )
    bank.swap_atom_slots(
        remove=(donor, int(donor_slot)),
        add=(recipient, int(recipient_slot)),
        optimizer=optimizer,
        zero_new=True,
    )
    bank.assert_exact_budget()
    event.update(
        {
            "accepted": True,
            "reason": "positive_proxy_margin",
            "donor_slot": int(donor_slot),
            "recipient_slot": int(recipient_slot),
            "proxy_margin": recipient_score - donor_score,
        }
    )
    return event


def train_one_epoch(
    arm: str,
    bank: T5GlobalRankBank,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    update_examples: Sequence[OfficialExample],
    risk_plan: RiskPlan,
    *,
    config: PanelConfig,
    task_index: int,
) -> tuple[dict[str, Any], Tensor]:
    if not update_examples:
        raise ValueError("each task must retain at least one update example")
    order_rng = np.random.Generator(
        np.random.PCG64(_stable_seed(config.seed, "update-order", task_index))
    )
    replay_rng = np.random.Generator(
        np.random.PCG64(_stable_seed(config.seed, "replay-draws", task_index))
    )
    order = order_rng.permutation(len(update_examples)).tolist()
    ordered = tuple(update_examples[int(index)] for index in order)
    gradient_scores = torch.zeros(len(bank.adapters), dtype=torch.float64)
    losses: list[float] = []
    replay_steps = 0
    resource_ledger = {
        "current_examples": 0,
        "replay_examples": 0,
        "current_source_tokens": 0,
        "current_target_tokens": 0,
        "replay_source_tokens": 0,
        "replay_target_tokens": 0,
    }
    bank.train()
    for current in _chunks(ordered, config.batch_size):
        uniform_records, risk_records = _training_replay_records(
            arm,
            risk_plan,
            replay_batch_size=config.replay_batch_size,
            risk_fraction=config.risk_fraction,
            rng=replay_rng,
        )
        replay = (*uniform_records, *risk_records)
        all_prompts = [example.prompt for example in current] + [record.prompt for record in replay]
        all_targets = [example.label for example in current] + [record.target for record in replay]
        encoded = _tokenize(
            tokenizer,
            all_prompts,
            all_targets,
            max_source_length=config.max_source_length,
            max_target_length=config.max_target_length,
            device=torch.device(config.device),
        )
        output = bank(**encoded)
        nll = per_example_target_token_nll(output.logits, encoded["labels"])
        current_count = len(current)
        resource_ledger["current_examples"] += current_count
        resource_ledger["replay_examples"] += len(replay)
        source_counts = encoded["attention_mask"].sum(dim=1)
        target_counts = encoded["labels"].ne(-100).sum(dim=1)
        resource_ledger["current_source_tokens"] += int(source_counts[:current_count].sum())
        resource_ledger["current_target_tokens"] += int(target_counts[:current_count].sum())
        resource_ledger["replay_source_tokens"] += int(source_counts[current_count:].sum())
        resource_ledger["replay_target_tokens"] += int(target_counts[current_count:].sum())
        current_loss = nll[:current_count].mean()
        if not replay:
            objective = current_loss
        else:
            cursor = current_count
            replay_terms: list[Tensor] = []
            replay_term_weights: list[float] = []
            if uniform_records:
                uniform_nll = nll[cursor : cursor + len(uniform_records)]
                replay_terms.append(uniform_nll.mean())
                replay_term_weights.append(1.0 - config.risk_fraction if risk_records else 1.0)
                cursor += len(uniform_records)
            if risk_records:
                risk_nll = nll[cursor : cursor + len(risk_records)]
                anchored = differentiable_anchored_regret(
                    risk_nll,
                    [record.loss_pre for record in risk_records],
                    [record.loss_post for record in risk_records],
                    eligibility=[record.eligible for record in risk_records],
                    min_acquired_gain=config.min_acquired_gain,
                )
                # Risk samples are drawn from the full detached CVaR/DRO best
                # response, so their unweighted mean is a stochastic estimate.
                replay_terms.append(anchored.regret.mean())
                replay_term_weights.append(config.risk_fraction)
            total_weight = sum(replay_term_weights)
            replay_loss = sum(
                term * (weight / total_weight)
                for term, weight in zip(replay_terms, replay_term_weights)
            )
            objective = 0.5 * current_loss + 0.5 * replay_loss
            replay_steps += 1
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        _accumulate_gradient_scores(bank, gradient_scores)
        torch.nn.utils.clip_grad_norm_(tuple(bank.active_atom_parameters()), 1.0)
        optimizer.step()
        bank.assert_exact_budget()
        losses.append(float(objective.detach().cpu()))
    return (
        {
            "epoch_count": 1,
            "optimizer": "AdamW",
            "constant_lr": config.lr,
            "weight_decay": 0.0,
            "steps": len(losses),
            "replay_steps": replay_steps,
            "mean_training_objective": float(np.mean(losses)),
            "resource_ledger": resource_ledger,
            "risk_plan": asdict(risk_plan),
            "risk_plan_record_count": len(risk_plan.records),
        },
        gradient_scores,
    )


def _make_anchor_records(
    examples: Sequence[OfficialExample],
    pre: Mapping[str, float],
    post: Mapping[str, float],
    *,
    task_name: str,
    include_task_id: bool,
    min_acquired_gain: float,
) -> tuple[AnchorRecord, ...]:
    output: list[AnchorRecord] = []
    for example in examples:
        acquired = float(pre[example.example_id]) - float(post[example.example_id])
        output.append(
            AnchorRecord(
                example_id=example.example_id,
                prompt=example.prompt,
                target=example.label,
                loss_pre=float(pre[example.example_id]),
                loss_post=float(post[example.example_id]),
                eligible=acquired >= min_acquired_gain,
                task_id=task_name if include_task_id else None,
            )
        )
    return tuple(output)


def audit_diagnostics(
    bank: T5GlobalRankBank,
    tokenizer: Any,
    audit_buffer: HistoryBuffer,
    *,
    config: PanelConfig,
) -> dict[str, Any]:
    records = audit_buffer.records_for_evaluation()
    if not records:
        return {"record_count": 0, "eligible_count": 0}
    now = torch.tensor(
        _per_example_losses(bank, tokenizer, records, config=config), dtype=torch.float32
    )
    anchored = differentiable_anchored_regret(
        now,
        [record.loss_pre for record in records],
        [record.loss_post for record in records],
        eligibility=[record.eligible for record in records],
        min_acquired_gain=config.min_acquired_gain,
    )
    eligible = anchored.tail_values.detach().cpu()
    result: dict[str, Any] = {
        "record_count": len(records),
        "eligible_count": int(eligible.numel()),
        "gradient_access": False,
    }
    if eligible.numel():
        weights = detached_cvar_weights(eligible, config.cvar_alpha)
        result.update(
            {
                "mean_anchored_regret": float(eligible.mean()),
                "max_anchored_regret": float(eligible.max()),
                "empirical_cvar": float((weights * eligible).sum()),
            }
        )
    return result


def _rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def _adapter_fingerprint(bank: T5GlobalRankBank) -> str:
    digest = hashlib.sha256()
    for name, adapter in zip(bank.adapter_names, bank.adapters):
        digest.update(name.encode("utf-8"))
        digest.update(bytes(adapter.active_mask.detach().cpu().numpy()))
        for atom in adapter.atoms:
            digest.update(atom.a.detach().cpu().contiguous().numpy().tobytes())
            digest.update(atom.b.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _build_model(config: PanelConfig, tokenizer: Any) -> T5GlobalRankBank:
    from transformers import AutoModelForSeq2SeqLM, T5Config, T5ForConditionalGeneration

    _set_seed(config.seed)
    if config.tiny_random_model:
        tiny_config = T5Config(
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
        model = T5ForConditionalGeneration(tiny_config)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_path,
            local_files_only=Path(config.model_path).expanduser().exists(),
        )
    model.to(torch.device(config.device))
    bank = T5GlobalRankBank(
        model,
        max_rank=config.max_rank,
        initial_rank=config.initial_rank,
        alpha=config.lora_alpha,
        reference_rank=config.reference_rank,
        require_t5_small=not config.tiny_random_model,
    )
    return bank


def _model_provenance(model_path: str) -> dict[str, Any]:
    path = Path(model_path).expanduser()
    if not path.exists():
        return {"identifier": model_path, "local": False}
    resolved = path.resolve()
    files: list[dict[str, Any]] = []
    candidates = (
        "config.json",
        "generation_config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    for name in candidates:
        candidate = resolved / name
        if candidate.is_file():
            files.append(
                {"name": name, "size": candidate.stat().st_size, "sha256": _sha256_file(candidate)}
            )
    return {"identifier": str(resolved), "local": True, "files": files}


def _environment_manifest(config: PanelConfig) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "numpy", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "device": config.device,
    }


def _save_stage(
    stage_dir: Path,
    bank: T5GlobalRankBank,
    risk_buffer: HistoryBuffer,
    audit_buffer: HistoryBuffer,
    *,
    stage_payload: Mapping[str, Any],
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = stage_dir / "adapter.pt"
    _atomic_torch_save(checkpoint, bank.compact_state_dict())
    _atomic_json(stage_dir / "stage.json", dict(stage_payload))
    _atomic_json(stage_dir / "payload.json", bank.compact_manifest())
    _atomic_json(stage_dir / "ranks.json", bank.layer_ranks())
    _atomic_json(stage_dir / "risk_buffer.json", risk_buffer.state_dict())
    _atomic_json(stage_dir / "audit_buffer.json", audit_buffer.state_dict())
    _atomic_json(
        stage_dir / "checkpoint.json",
        {
            "file": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256_file(checkpoint),
            "compact_active_adapter_only": True,
        },
    )


def run_arm(
    arm: str,
    prepared: Sequence[PreparedTask],
    tokenizer: Any,
    *,
    config: PanelConfig,
    expected_initial_fingerprint: str | None,
    cached_base: tuple[float, ...] | None,
) -> tuple[str, tuple[float, ...]]:
    arm_dir = Path(config.output_root) / arm
    if arm_dir.exists() and any(arm_dir.iterdir()):
        raise FileExistsError(
            f"refusing to mix a new run with existing artifacts in {arm_dir}; "
            "choose a fresh --output-root"
        )
    arm_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    bank = _build_model(config, tokenizer)
    initial_fingerprint = _adapter_fingerprint(bank)
    if expected_initial_fingerprint is not None and initial_fingerprint != expected_initial_fingerprint:
        raise RuntimeError("LoRA reservoir initialization differs across panel arms")
    initial_audit = bank.payload_audit()
    if not initial_audit.exact_budget:
        raise RuntimeError("initial adapter payload does not satisfy the exact budget")
    if not config.tiny_random_model and initial_audit.active_atoms != 144:
        raise RuntimeError(f"canonical T5-small panel requires 144 atoms, got {initial_audit.active_atoms}")

    risk_buffer = HistoryBuffer(
        role=BufferRole.RISK_TRAIN,
        max_records=config.risk_buffer_records,
        base_seed=config.seed,
        # Keep reservoir admission draws arm-independent.  Controller policy
        # changes replay weights, not which RNG stream admits history.
        stream_id="order4:risk",
        byte_budget=config.risk_buffer_bytes,
        allow_task_ids=arm in ORACLE_ARMS,
    )
    audit_buffer = HistoryBuffer(
        role=BufferRole.AUDIT,
        max_records=config.audit_buffer_records,
        base_seed=config.seed,
        stream_id="order4:audit",
        byte_budget=config.audit_buffer_bytes,
        allow_task_ids=True,
    )
    optimizer = torch.optim.AdamW(
        tuple(bank.all_atom_parameters()), lr=config.lr, weight_decay=0.0
    )
    _atomic_json(
        arm_dir / "manifest.json",
        {
            "format": FORMAT_VERSION,
            "status": "running",
            "arm": arm,
            "config": asdict(config),
            "initial_adapter_fingerprint": initial_fingerprint,
            "initial_payload": initial_audit.to_dict(),
            "rank_mechanism": (
                "deterministic one-atom combined-training-gradient/norm proxy; "
                "not the final measured shortlist/probe controller"
                if arm in ADAPTIVE_RANK_ARMS
                else "uniform rank-4 mask frozen for the entire stream"
            ),
            "test_usage": "stage-end evaluator only",
            "audit_gradient_access": False,
            "group_free_controller_has_task_ids": False if arm not in ORACLE_ARMS else None,
        },
    )

    if cached_base is None:
        base_scores = tuple(
            evaluate_em(bank, tokenizer, task.evaluation, config=config)[0] for task in prepared
        )
    else:
        base_scores = cached_base
    _atomic_json(
        arm_dir / "base_scores.json",
        {"task_names": [task.name for task in prepared], "scores": base_scores},
    )

    immediate_scores: list[float] = []
    stage_evaluations: list[dict[str, Any]] = []
    exchange_log: list[dict[str, Any]] = []
    for task_index, task in enumerate(prepared):
        stage_started = time.perf_counter()
        anchor_examples = (*task.risk, *task.audit)
        pre_values = _per_example_losses(bank, tokenizer, anchor_examples, config=config)
        pre = {example.example_id: value for example, value in zip(anchor_examples, pre_values)}
        # The adversary is refreshed on the complete historical risk buffer.
        # It cannot include this task's risk-train examples until after their
        # acquisition anchors have been measured around this update.
        plan = build_risk_plan(arm, risk_buffer, bank, tokenizer, config=config)
        # Reset dropout/model RNG independently of base-evaluation caching and
        # earlier rank activation.  Current-batch stochastic schedules are
        # therefore arm-matched even when controllers choose different slots.
        _set_seed(_stable_seed(config.seed, "training", task_index))
        training, gradient_scores = train_one_epoch(
            arm,
            bank,
            tokenizer,
            optimizer,
            (*task.update, *task.risk),
            plan,
            config=config,
            task_index=task_index,
        )
        # The deployed stage checkpoint includes the rank decision.  Therefore
        # post anchors, audit diagnostics, and immediate EM all measure the
        # exact same function; exchange self-damage is not deferred and then
        # mislabelled as next-stage forgetting.
        _set_seed(_stable_seed(config.seed, "rank-exchange", task_index))
        exchange = proxy_rank_exchange(
            bank,
            optimizer,
            gradient_scores,
            arm=arm,
            stage_index=task_index,
        )
        exchange_log.append(exchange)
        if arm in UNIFORM_RANK_ARMS and any(
            rank != config.initial_rank for rank in bank.layer_ranks().values()
        ):
            raise RuntimeError(f"uniform arm {arm} changed its rank mask")
        post_values = _per_example_losses(bank, tokenizer, anchor_examples, config=config)
        post = {example.example_id: value for example, value in zip(anchor_examples, post_values)}
        risk_records = _make_anchor_records(
            task.risk,
            pre,
            post,
            task_name=task.name,
            include_task_id=arm in ORACLE_ARMS,
            min_acquired_gain=config.min_acquired_gain,
        )
        audit_records = _make_anchor_records(
            task.audit,
            pre,
            post,
            task_name=task.name,
            include_task_id=True,
            min_acquired_gain=config.min_acquired_gain,
        )
        risk_decisions = risk_buffer.extend(risk_records)
        audit_decisions = audit_buffer.extend(audit_records)
        assert_disjoint_history(risk_buffer, audit_buffer)
        if not risk_buffer.byte_ledger().within_budget or not audit_buffer.byte_ledger().within_budget:
            raise RuntimeError("a persistent history buffer exceeded its byte budget")

        scores: dict[str, float] = {}
        predictions: dict[str, tuple[str, ...]] = {}
        for seen_task in prepared[: task_index + 1]:
            score, task_predictions = evaluate_em(
                bank, tokenizer, seen_task.evaluation, config=config
            )
            scores[seen_task.name] = score
            predictions[seen_task.name] = task_predictions
        immediate_scores.append(scores[task.name])
        audit = audit_diagnostics(bank, tokenizer, audit_buffer, config=config)
        stage_payload = {
            "stage_index": task_index,
            "task": task.name,
            "chronology": [item.name for item in prepared[: task_index + 1]],
            "partition_counts": {
                "update": len(task.update),
                "risk_train": len(task.risk),
                "audit": len(task.audit),
                "sealed_test_evaluation": len(task.evaluation),
                "excluded_unlabeled_test": task.excluded_unlabeled_test_count,
                "excluded_unlabeled_test_sha256": task.excluded_unlabeled_test_sha256,
            },
            "training": training,
            "gradient_access_counts": {
                "current_update": len(task.update),
                "current_risk_train": len(task.risk),
                "historical_replay_resident": len(plan.records),
                "audit": 0,
            },
            "acquisition": {
                "risk_eligible": sum(record.eligible for record in risk_records),
                "audit_eligible": sum(record.eligible for record in audit_records),
                "measured_pre_before_update": True,
                "measured_post_after_update": True,
            },
            "rank_exchange": exchange,
            "evaluation_scores": scores,
            "evaluation_predictions": predictions,
            "audit_diagnostics": audit,
            "payload_audit": bank.payload_audit().to_dict(),
            "risk_buffer_ledger": asdict(risk_buffer.byte_ledger()),
            "audit_buffer_ledger": asdict(audit_buffer.byte_ledger()),
            "reservoir_decisions": {
                "risk_train": {
                    action: sum(decision.action == action for decision in risk_decisions)
                    for action in (
                        "append", "replace", "skip_reservoir", "skip_byte_budget"
                    )
                },
                "audit": {
                    action: sum(decision.action == action for decision in audit_decisions)
                    for action in (
                        "append", "replace", "skip_reservoir", "skip_byte_budget"
                    )
                },
            },
            "resource": {
                "wall_seconds": time.perf_counter() - stage_started,
                "rss_bytes": _rss_bytes(),
                "training_examples_and_tokens": training["resource_ledger"],
                "cuda_peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
                ),
            },
        }
        stage_evaluations.append({"task": task.name, "scores": scores})
        _save_stage(
            arm_dir / f"stage_{task_index + 1:02d}_{task.name}",
            bank,
            risk_buffer,
            audit_buffer,
            stage_payload=stage_payload,
        )
        print(
            f"[{arm}] {task_index + 1}/{len(prepared)} {task.name}: "
            f"diag={scores[task.name]:.4f}, wall={stage_payload['resource']['wall_seconds']:.1f}s",
            flush=True,
        )

    final_scores = tuple(
        float(stage_evaluations[-1]["scores"][task.name]) for task in prepared
    )
    result = {
        "format": FORMAT_VERSION,
        "status": "completed",
        "arm": arm,
        "task_names": [task.name for task in prepared],
        "base_scores": base_scores,
        "immediate_scores": immediate_scores,
        "final_scores": final_scores,
        "task_tail_alpha": config.cvar_alpha,
        "stage_evaluations": stage_evaluations,
        "rank_exchanges": exchange_log,
        "final_payload": bank.payload_audit().to_dict(),
        "final_ranks": bank.layer_ranks(),
        "resource": {
            "wall_seconds": time.perf_counter() - started,
            "rss_bytes": _rss_bytes(),
        },
    }
    # Validate all vectors and normalized-retention denominators before marking
    # an arm complete.  A zero acquisition denominator is a real failed panel,
    # not something to hide behind a sentinel.
    summary = summarize_arm(
        arm=arm,
        task_names=result["task_names"],
        base_scores=base_scores,
        immediate_scores=immediate_scores,
        final_scores=final_scores,
        tail_alpha=config.cvar_alpha,
    )
    result["summary"] = asdict(summary)
    _atomic_json(arm_dir / "result.json", result)
    _atomic_json(
        arm_dir / "manifest.json",
        {
            "format": FORMAT_VERSION,
            "status": "completed",
            "arm": arm,
            "config": asdict(config),
            "initial_adapter_fingerprint": initial_fingerprint,
            "initial_payload": initial_audit.to_dict(),
            "final_payload": bank.payload_audit().to_dict(),
            "rank_mechanism": (
                "deterministic one-atom combined-training-gradient/norm proxy; "
                "not the final measured shortlist/probe controller"
                if arm in ADAPTIVE_RANK_ARMS
                else "uniform rank-4 mask frozen for the entire stream"
            ),
            "test_usage": "stage-end evaluator only",
            "audit_gradient_access": False,
            "group_free_controller_has_task_ids": False if arm not in ORACLE_ARMS else None,
        },
    )
    (arm_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return initial_fingerprint, base_scores


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--per-class-cap", type=_positive_int, default=200)
    parser.add_argument("--risk-per-class", type=_positive_int, default=10)
    parser.add_argument("--audit-per-class", type=_positive_int, default=10)
    parser.add_argument("--eval-per-task", type=_positive_int, default=200)
    parser.add_argument(
        "--tasks",
        default="all",
        help="all, an integer prefix length, or a comma-separated exact Order-4 prefix",
    )
    parser.add_argument(
        "--arms", nargs="+", default=["all"], help="all or one/more locked arm names"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--replay-batch-size", type=_positive_int, default=8)
    parser.add_argument("--risk-buffer-records", type=_positive_int, default=512)
    parser.add_argument("--audit-buffer-records", type=_positive_int, default=256)
    parser.add_argument("--risk-buffer-bytes", type=_positive_int)
    parser.add_argument("--audit-buffer-bytes", type=_positive_int)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-source-length", type=_positive_int, default=512)
    parser.add_argument("--max-target-length", type=_positive_int, default=50)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tiny-random-model", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> PanelConfig:
    tasks = parse_task_selection(args.tasks)
    arms = parse_arm_selection(args.arms)
    per_class_cap = args.per_class_cap
    risk_per_class = args.risk_per_class
    audit_per_class = args.audit_per_class
    eval_per_task = args.eval_per_task
    batch_size = args.batch_size
    replay_batch_size = args.replay_batch_size
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length
    if args.smoke:
        # Smoke changes workload, never chronology.  Use ``--tasks 2 --smoke``
        # for the two-task integration test; ``--tasks all --smoke`` retains 15.
        per_class_cap = min(per_class_cap, 4)
        risk_per_class = min(risk_per_class, 1)
        audit_per_class = min(audit_per_class, 1)
        eval_per_task = min(eval_per_task, 4)
        batch_size = min(batch_size, 2)
        replay_batch_size = min(replay_batch_size, 2)
        max_source_length = min(max_source_length, 64)
        max_target_length = min(max_target_length, 8)
    if per_class_cap <= risk_per_class + audit_per_class:
        raise ValueError("per-class-cap must exceed risk-per-class + audit-per-class")
    if replay_batch_size < 2 and any(
        arm in ORACLE_ARMS or arm in CVAR_ARMS for arm in arms
    ):
        raise ValueError(
            "adaptive risk arms require replay-batch-size >= 2 so uniform "
            "coverage and adversarial replay are both present"
        )
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return PanelConfig(
        data_root=str(Path(args.data_root).expanduser().resolve()),
        model_path=args.model_path,
        output_root=str(Path(args.output_root).expanduser().resolve()),
        task_names=tasks,
        arms=arms,
        seed=args.seed,
        per_class_cap=per_class_cap,
        risk_per_class=risk_per_class,
        audit_per_class=audit_per_class,
        eval_per_task=eval_per_task,
        batch_size=batch_size,
        replay_batch_size=replay_batch_size,
        risk_buffer_records=args.risk_buffer_records,
        audit_buffer_records=args.audit_buffer_records,
        risk_buffer_bytes=args.risk_buffer_bytes,
        audit_buffer_bytes=args.audit_buffer_bytes,
        lr=float(args.lr),
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        cvar_alpha=0.2,
        min_acquired_gain=0.05,
        risk_fraction=0.5,
        max_rank=8,
        initial_rank=4,
        lora_alpha=16.0,
        reference_rank=4,
        tiny_random_model=bool(args.tiny_random_model),
        smoke=bool(args.smoke),
        device=device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    output_root = Path(config.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"refusing to mix attempts in non-empty output root {output_root}; "
            "choose a fresh --output-root"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    if args.download_data:
        download_official_data(config.data_root, config.task_names)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        use_fast=True,
        local_files_only=Path(config.model_path).expanduser().exists(),
    )
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise RuntimeError("T5 tokenizer must define pad and EOS tokens")
    prepared = prepare_data(config)
    root_manifest = {
        "format": FORMAT_VERSION,
        "status": "running",
        "config": asdict(config),
        "official_protocol": {
            "repository": OFFICIAL_REPOSITORY,
            "commit": OFFICIAL_COMMIT,
            "order": list(config.task_names),
            "epochs_per_task": 1,
            "prompt_and_label_source": "pinned O-LoRA repository",
            "paper_comparability": (
                "T5-small feasibility probe; not T5-large paper-number comparability"
            ),
        },
        "method_disclosure": {
            "adaptive_rank": (
                "deterministic exact-budget one-atom combined-training-gradient/norm proxy"
            ),
            "measured_shortlist_probe_implemented": False,
            "risk_distribution_refresh": "complete risk buffer once per task before update",
            "uniform_replay_present_in_every_arm": True,
            "audit_gradient_access": False,
            "rng_schedule": "arm-matched reset before every training episode and rank exchange",
        },
        "environment": _environment_manifest(config),
        "model_provenance": _model_provenance(config.model_path),
        "partitions": {
            task.name: {
                "update": len(task.update),
                "risk_train": len(task.risk),
                "audit": len(task.audit),
                "sealed_test_evaluation": len(task.evaluation),
                "excluded_unlabeled_test": task.excluded_unlabeled_test_count,
                "excluded_unlabeled_test_sha256": task.excluded_unlabeled_test_sha256,
            }
            for task in prepared
        },
    }
    _atomic_json(output_root / "manifest.json", root_manifest)
    expected_fingerprint: str | None = None
    cached_base: tuple[float, ...] | None = None
    for arm in config.arms:
        expected_fingerprint, cached_base = run_arm(
            arm,
            prepared,
            tokenizer,
            config=config,
            expected_initial_fingerprint=expected_fingerprint,
            cached_base=cached_base,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if config.arms == LOCKED_ARMS:
        summaries = aggregate_completed_run(output_root)
        write_summary(output_root, summaries)
    else:
        partial = {}
        for arm in config.arms:
            with (output_root / arm / "result.json").open("r", encoding="utf-8") as handle:
                partial[arm] = json.load(handle)["summary"]
        _atomic_json(output_root / "partial_panel_summary.json", partial)
    root_manifest["status"] = "completed"
    root_manifest["completed_arms"] = list(config.arms)
    _atomic_json(output_root / "manifest.json", root_manifest)
    (output_root / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
