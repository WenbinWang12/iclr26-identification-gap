"""Synthetic integrity tests for neural objectives and history isolation."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buffers import (  # noqa: E402
    AnchorRecord,
    BufferRole,
    ControllerRecord,
    HistoryBuffer,
    assert_disjoint_history,
)
from objectives import (  # noqa: E402
    detached_cvar_weights,
    detached_worst_group_dro_weights,
    differentiable_anchored_regret,
    groupfree_anchored_cvar_objective,
    mixed_replay_weights,
    oracle_anchored_dro_objective,
    per_example_target_token_nll,
)


def _record(index: int, *, task_id=None, prompt_suffix: str = "") -> AnchorRecord:
    return AnchorRecord(
        example_id=f"example-{index}",
        prompt=f"问题 {index}{prompt_suffix}",
        target="True",
        loss_pre=1.0 + index / 100.0,
        loss_post=0.2,
        eligible=True,
        task_id=task_id,
    )


def test_target_token_nll_is_length_normalized_and_differentiable():
    logits = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0], [3.0, -1.0]],
            [[2.0, 0.0], [0.0, 2.0], [3.0, -1.0]],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[0, 1, -100], [0, -100, -100]])
    got = per_example_target_token_nll(logits, labels)
    one_token = torch.nn.functional.cross_entropy(logits[0, 0:1], labels[0, 0:1])
    assert torch.allclose(got, torch.stack([one_token, one_token]))
    got.sum().backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, 2]).item() == 0


def test_target_token_nll_rejects_empty_and_invalid_targets():
    logits = torch.zeros(1, 2, 3)
    for labels in (torch.tensor([[-100, -100]]), torch.tensor([[0, 3]])):
        try:
            per_example_target_token_nll(logits, labels)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid target labels were accepted")


def test_anchored_regret_only_differentiates_reliably_acquired_examples():
    now = torch.tensor([5.2, 1.0, 0.3], requires_grad=True)
    out = differentiable_anchored_regret(
        now,
        loss_pre=[5.0, 1.0, 1.0],
        loss_post=[4.98, 0.2, 0.2],
        eligibility=[True, True, False],
    )
    assert out.eligible.tolist() == [False, True, False]
    assert torch.allclose(out.regret, torch.tensor([0.0, 1.0, 0.0]))
    out.regret.sum().backward()
    assert now.grad is not None
    assert now.grad[0].item() == 0.0 and now.grad[2].item() == 0.0
    assert now.grad[1].item() > 0.0


def test_cvar_best_response_is_exact_fractional_and_detached():
    values = torch.tensor([1.0, 2.0, 3.0, 10.0], requires_grad=True)
    q = detached_cvar_weights(values, alpha=0.375)
    expected = (10.0 + 0.5 * 3.0) / 1.5
    assert not q.requires_grad
    assert torch.isclose(q.sum(), torch.tensor(1.0))
    assert q.max().item() <= 1.0 / (0.375 * 4) + 1e-7
    assert np.isclose(float((q @ values).detach()), expected)
    (q @ values).backward()
    assert torch.allclose(values.grad, q)

    masked = detached_cvar_weights(values.detach(), 0.5, eligible=[False, True, True, False])
    assert torch.allclose(masked, torch.tensor([0.0, 0.0, 1.0, 0.0]))


def test_task_oracle_selects_worst_group_and_groupfree_uses_no_ids():
    values = torch.tensor([0.8, 1.0, 0.2, 0.4], requires_grad=True)
    q_oracle = detached_worst_group_dro_weights(values, ["a", "a", "b", "b"])
    assert torch.allclose(q_oracle, torch.tensor([0.5, 0.5, 0.0, 0.0]))
    assert not q_oracle.requires_grad

    oracle = oracle_anchored_dro_objective(
        values,
        loss_pre=[1.2] * 4,
        loss_post=[0.2] * 4,
        group_ids=["a", "a", "b", "b"],
    )
    assert torch.allclose(oracle.weights, q_oracle)
    groupfree = groupfree_anchored_cvar_objective(
        values,
        loss_pre=[1.2] * 4,
        loss_post=[0.2] * 4,
        alpha=0.5,
    )
    assert groupfree.weights[1] > 0 and groupfree.weights[0] > 0
    assert torch.count_nonzero(groupfree.weights).item() == 2


def test_mixed_replay_weights_preserve_coverage_and_risk_mass():
    risk = torch.tensor([0.0, 0.25, 0.75])
    got = mixed_replay_weights(risk, risk_fraction=0.6)
    expected = 0.4 * torch.full((3,), 1.0 / 3.0) + 0.6 * risk
    assert torch.allclose(got, expected)
    assert not got.requires_grad and torch.isclose(got.sum(), torch.tensor(1.0))


def test_canonical_ledger_counts_utf8_and_roundtrips_exactly():
    buffer = HistoryBuffer(
        role=BufferRole.AUDIT,
        max_records=3,
        base_seed=7,
        stream_id="audit",
        byte_budget=10_000,
    )
    buffer.offer(_record(0, task_id="CB"))
    payload = buffer.canonical_bytes()
    assert len(payload) == buffer.byte_ledger().total_bytes
    assert "问题".encode("utf-8") in payload
    assert json.loads(payload.decode("utf-8"))["role"] == "audit"

    restored = HistoryBuffer.from_state_dict(buffer.state_dict())
    assert restored.canonical_bytes() == payload
    assert restored.records_for_evaluation() == buffer.records_for_evaluation()


def test_reservoir_membership_is_deterministic_and_sampling_independent():
    def build(*, interleave_sampling: bool) -> HistoryBuffer:
        history = HistoryBuffer(
            role="risk_train",
            max_records=4,
            base_seed=123,
            stream_id="risk",
        )
        for index in range(30):
            history.offer(_record(index))
            if interleave_sampling and len(history) >= 2:
                history.sample_for_gradient(2)
        return history

    untouched = build(interleave_sampling=False)
    sampled = build(interleave_sampling=True)
    ids_a = [record.example_id for record in untouched.records_for_evaluation()]
    ids_b = [record.example_id for record in sampled.records_for_evaluation()]
    assert ids_a == ids_b
    assert untouched.seen_count == sampled.seen_count == 30

    repeated = build(interleave_sampling=False)
    assert repeated.canonical_bytes() == untouched.canonical_bytes()


def test_byte_budget_is_hard_and_reports_rejection():
    history = HistoryBuffer(
        role="risk_train",
        max_records=10,
        base_seed=1,
        stream_id="small-envelope",
        byte_budget=1_000,
    )
    decision = history.offer(_record(0, prompt_suffix="x" * 5_000))
    assert not decision.accepted and decision.action == "skip_byte_budget"
    ledger = history.byte_ledger()
    assert ledger.within_budget and ledger.total_bytes <= 1_000
    assert ledger.record_count == 0 and ledger.seen_count == 1


def test_audit_gradient_sampling_and_group_metadata_fail_closed():
    audit = HistoryBuffer(
        role="audit", max_records=2, base_seed=4, stream_id="audit"
    )
    audit.offer(_record(1, task_id="MNLI"))
    try:
        audit.sample_for_gradient(1)
    except PermissionError:
        pass
    else:
        raise AssertionError("audit data entered a gradient sampler")

    groupfree = HistoryBuffer(
        role="risk_train", max_records=2, base_seed=4, stream_id="risk"
    )
    try:
        groupfree.offer(_record(2, task_id="MNLI"))
    except PermissionError:
        pass
    else:
        raise AssertionError("group-free history accepted a task ID")
    groupfree.offer(_record(2))
    sampled = groupfree.sample_for_gradient(1)
    assert isinstance(sampled[0], ControllerRecord)
    assert not hasattr(sampled[0], "task_id")

    oracle = HistoryBuffer(
        role="risk_train",
        max_records=2,
        base_seed=4,
        stream_id="oracle",
        allow_task_ids=True,
    )
    oracle.offer(_record(3, task_id="MNLI"))
    assert isinstance(oracle.sample_for_gradient(1, oracle=True)[0], AnchorRecord)


def test_risk_and_audit_buffers_must_be_disjoint():
    risk = HistoryBuffer(
        role="risk_train", max_records=2, base_seed=8, stream_id="risk"
    )
    audit = HistoryBuffer(
        role="audit", max_records=2, base_seed=8, stream_id="audit"
    )
    risk.offer(_record(1))
    audit.offer(_record(2, task_id="CB"))
    assert_disjoint_history(risk, audit)
    audit.offer(_record(1, task_id="CB"))
    try:
        assert_disjoint_history(risk, audit)
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping risk/audit history was accepted")


if __name__ == "__main__":
    test_target_token_nll_is_length_normalized_and_differentiable()
    test_target_token_nll_rejects_empty_and_invalid_targets()
    test_anchored_regret_only_differentiates_reliably_acquired_examples()
    test_cvar_best_response_is_exact_fractional_and_detached()
    test_task_oracle_selects_worst_group_and_groupfree_uses_no_ids()
    test_mixed_replay_weights_preserve_coverage_and_risk_mass()
    test_canonical_ledger_counts_utf8_and_roundtrips_exactly()
    test_reservoir_membership_is_deterministic_and_sampling_independent()
    test_byte_budget_is_hard_and_reports_rejection()
    test_audit_gradient_sampling_and_group_metadata_fail_closed()
    test_risk_and_audit_buffers_must_be_disjoint()
    print("test_objectives_buffers OK")
