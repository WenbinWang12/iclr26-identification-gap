from __future__ import annotations

import ast
import copy
import inspect
from types import SimpleNamespace

import pytest
import torch

from experiments.phase2i_anchored_cvar.buffers import (
    AnchorRecord,
    BufferRole,
    HistoryBuffer,
)
from experiments.phase2i_anchored_cvar.order4_data import (
    ORDER4_TASK_NAMES,
    OfficialExample,
)
from experiments.phase2i_anchored_cvar import run_acquisition_conditioned_headroom as runner


def _example(
    task: str,
    index: int,
    label: str,
    *,
    sentence: str | None = None,
) -> OfficialExample:
    content = sentence or f"sentence {index}"
    return OfficialExample(
        task_name=task,
        category=task,
        dataset=task,
        subset="train",
        source_index=index,
        example_id=f"{task}-{index}",
        sentence=content,
        label=label,
        prompt=f"HEADER\n{content}\nAnswer:",
    )


def test_task_and_arm_protocol_is_locked() -> None:
    assert runner.parse_task_selection("2") == ORDER4_TASK_NAMES[:2]
    assert runner.parse_task_selection("MNLI,CB") == ORDER4_TASK_NAMES[:2]
    with pytest.raises(ValueError, match="chronology-preserving"):
        runner.parse_task_selection("MNLI,WiC")
    assert runner.parse_arm_selection(["all"]) == (
        "uniform_er_rank4",
        "task_id_oracle",
    )


def test_frozen_epoch_schedule_supports_multirc_override(tmp_path) -> None:
    args = runner.build_argument_parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--model-path",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    config = runner._config_from_args(args)
    assert config.acquisition_epochs_by_task["QQP"] == 1
    assert config.acquisition_epochs_by_task["BoolQA"] == 1
    assert config.acquisition_epochs_by_task["MultiRC"] == 2
    stopped_args = runner.build_argument_parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--model-path",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "stopped"),
            "--tasks",
            "2",
            "--stop-after-task",
            "MNLI",
        ]
    )
    assert runner._config_from_args(stopped_args).stop_after_task == 1


def test_multirc_semantic_groups_are_atomic_across_partitions() -> None:
    examples = []
    for group_index in range(30):
        prefix = f"paragraph: paragraph {group_index}\nquestion: question {group_index}?"
        for candidate_index, label in enumerate(("False", "True")):
            examples.append(
                _example(
                    "MultiRC",
                    group_index * 10 + candidate_index,
                    label,
                    sentence=prefix + f"\ncandidate answer: candidate {candidate_index}",
                )
            )
    task = runner.build_stream_task(
        tuple(examples),
        task_name="MultiRC",
        update_cap_per_class=6,
        risk_per_class=2,
        acquisition_audit_examples=8,
        min_update_per_class=2,
        seed=9,
    )
    group_sets = [
        {task.group_key_by_example_id[item.example_id] for item in partition}
        for partition in (task.update, task.risk, task.audit)
    ]
    assert not group_sets[0] & group_sets[1]
    assert not group_sets[0] & group_sets[2]
    assert not group_sets[1] & group_sets[2]
    # Every selected question contributes both candidate rows.
    assert all(len(partition) % 2 == 0 for partition in (task.update, task.risk, task.audit))


def _qqp(index: int, label: str, first: str, second: str) -> OfficialExample:
    return _example(
        "QQP",
        index,
        label,
        sentence=f"first sentence: {first}\nsecond sentence: {second}",
    )


def test_strict_semantic_groups_cover_qqp_components_and_boolqa_passages() -> None:
    qqp = (
        _qqp(0, "False", "Question A", "Question B"),
        _qqp(1, "True", " question b ", "Question C"),
        _qqp(2, "False", "Independent", "Other"),
    )
    qqp_keys = runner.semantic_group_keys(qqp, task_name="QQP")
    assert qqp_keys[qqp[0].example_id] == qqp_keys[qqp[1].example_id]
    assert qqp_keys[qqp[0].example_id] != qqp_keys[qqp[2].example_id]

    boolqa = (
        _example(
            "BoolQA",
            0,
            "True",
            sentence="question: first?\npassage: Shared   Passage",
        ),
        _example(
            "BoolQA",
            1,
            "False",
            sentence="question: second?\npassage: shared passage",
        ),
    )
    bool_keys = runner.semantic_group_keys(boolqa, task_name="BoolQA")
    assert bool_keys[boolqa[0].example_id] == bool_keys[boolqa[1].example_id]


def test_audit_exact_quota_uses_hash_priority_not_group_size() -> None:
    large_rows = tuple(
        _example("BoolQA", index, label)
        for index, label in enumerate(("False", "False", "True", "True"))
    )
    singleton_rows = tuple(
        _example("BoolQA", 100 + index, label)
        for index, label in enumerate(("False",) * 4 + ("True",) * 4)
    )
    remaining = {"large": large_rows}
    remaining.update(
        {f"singleton-{index}": (row,) for index, row in enumerate(singleton_rows)}
    )
    # Pick a deterministic seed where hash priority puts the large group after
    # all singleton groups.  A size-first selector would still take it first.
    seed = next(
        candidate
        for candidate in range(10_000)
        if max(
            remaining,
            key=lambda key: runner._semantic_partition_key(
                "BoolQA", key, candidate
            ),
        )
        == "large"
    )
    audit, _ = runner._take_exact_group_partition(
        remaining,
        task_name="BoolQA",
        labels=("False", "True"),
        quota_per_class=4,
        seed=seed,
        selection_policy=runner.AUDIT_GROUP_SELECTION_POLICY,
    )
    repeated, _ = runner._take_exact_group_partition(
        remaining,
        task_name="BoolQA",
        labels=("False", "True"),
        quota_per_class=4,
        seed=seed,
        selection_policy=runner.AUDIT_GROUP_SELECTION_POLICY,
    )
    capacity_first, _ = runner._take_exact_group_partition(
        remaining,
        task_name="BoolQA",
        labels=("False", "True"),
        quota_per_class=4,
        seed=seed,
        selection_policy=runner.CAPACITY_GROUP_SELECTION_POLICY,
    )
    large_ids = {row.example_id for row in large_rows}
    assert [row.example_id for row in audit] == [row.example_id for row in repeated]
    assert not large_ids.intersection(row.example_id for row in audit)
    assert large_ids.issubset(row.example_id for row in capacity_first)
    assert {
        label: sum(row.label == label for row in audit)
        for label in ("False", "True")
    } == {"False": 4, "True": 4}


def test_balanced_audit_reports_complete_train_natural_prior() -> None:
    examples = tuple(
        [_qqp(index, "False", f"F{index}a", f"F{index}b") for index in range(90)]
        + [
            _qqp(100 + index, "True", f"T{index}a", f"T{index}b")
            for index in range(30)
        ]
    )
    task = runner.build_stream_task(
        examples,
        task_name="QQP",
        update_cap_per_class=8,
        risk_per_class=2,
        acquisition_audit_examples=30,
        min_update_per_class=4,
        seed=4,
    )
    counts = {
        label: sum(item.label == label for item in task.audit)
        for label in ("False", "True")
    }
    assert counts == {"False": 15, "True": 15}
    assert task.natural_label_prior == {"False": 0.75, "True": 0.25}
    update_counts = {
        label: sum(item.label == label for item in task.update)
        for label in ("False", "True")
    }
    assert update_counts == {"False": 8, "True": 8}
    split_manifest = runner._split_manifest(task)
    group_audit = split_manifest["audit_group_sampling_audit"]
    assert group_audit["selection_policy"] == runner.AUDIT_GROUP_SELECTION_POLICY
    assert "descending_group_size_ordering" in group_audit[
        "forbidden_priority_features"
    ]
    assert group_audit["audit"]["group_count"] == 30
    assert group_audit["audit"]["group_size_histogram"] == {"1": 30}
    assert group_audit["complete_train_corpus"]["group_size_histogram"] == {"1": 120}


def test_group_free_plan_scores_full_buffer_but_ignores_task_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = HistoryBuffer(
        role=BufferRole.RISK_TRAIN,
        max_records=4,
        base_seed=1,
        stream_id="test",
        allow_task_ids=False,
    )
    buffer.offer(AnchorRecord("x", "prompt", "True", 1.0, 0.5, True, None))
    def fake_losses(*args, forward_ledger=None, **kwargs):
        assert forward_ledger is not None
        forward_ledger.update(examples=1, batches=1, source_tokens=7, target_tokens=2)
        return (0.7,)

    monkeypatch.setattr(runner, "_per_example_losses", fake_losses)
    plan = runner.build_replay_plan(
        "uniform_er_rank4", buffer, None, None, config=None  # type: ignore[arg-type]
    )
    assert plan.mode.startswith("group_free_uniform_er")
    assert plan.probabilities is None
    assert not hasattr(plan.records[0], "task_id")
    assert plan.controller_forward_examples == 1
    assert plan.controller_forward_source_tokens == 7


def test_sequence_mean_prior_weights_match_pi_over_balanced_q() -> None:
    weights = runner.sequence_mean_prior_weights(
        ["False", "True"],
        natural_label_prior={"False": 0.75, "True": 0.25},
        sampler_q={"False": 0.5, "True": 0.5},
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert weights.tolist() == pytest.approx([1.5, 0.5])
    assert not runner._strictly_above(0.05, 0.05)
    assert runner._strictly_above(0.050001, 0.05)
    assert runner._at_least(-0.050000000000000044, -0.05)
    assert not runner._at_least(-0.050001, -0.05)


def test_gate_enforces_strict_gain_validity_and_class_floor() -> None:
    before = {
        "natural_prior_em": 0.50,
        "valid_label_rate": 1.0,
        "per_class_recall": {"False": 0.4, "True": 0.6},
    }
    passing = {
        "natural_prior_em": 0.551,
        "valid_label_rate": 0.99,
        "per_class_recall": {"False": 0.5, "True": 0.551},
    }
    assert runner.acquisition_gate(before, passing, threshold=0.05)["pass"]
    boundary = dict(
        passing,
        per_class_recall={"False": 0.5, "True": 0.5499999999999999},
    )
    assert runner.acquisition_gate(before, boundary, threshold=0.05)["pass"]
    equality = dict(passing, natural_prior_em=0.55)
    assert not runner.acquisition_gate(before, equality, threshold=0.05)["pass"]
    invalid = dict(passing, valid_label_rate=0.989)
    assert not runner.acquisition_gate(before, invalid, threshold=0.05)["pass"]
    harmed = dict(
        passing,
        per_class_recall={"False": 0.6, "True": 0.549},
    )
    assert not runner.acquisition_gate(before, harmed, threshold=0.05)["pass"]


def test_only_real_full_order_can_open_official_test() -> None:
    assert not runner.official_evaluation_eligible(
        completed_task_count=2,
        total_task_count=2,
        stop_after_task=2,
        gate_passes=[True, True],
        task_names=ORDER4_TASK_NAMES[:2],
        smoke=False,
        tiny_random_model=False,
    )
    assert not runner.official_evaluation_eligible(
        completed_task_count=len(ORDER4_TASK_NAMES),
        total_task_count=len(ORDER4_TASK_NAMES),
        stop_after_task=None,
        gate_passes=[True] * (len(ORDER4_TASK_NAMES) - 1) + [False],
        task_names=ORDER4_TASK_NAMES,
        smoke=False,
        tiny_random_model=False,
    )
    # A passing selected prefix is still a train-only diagnostic.
    assert not runner.official_evaluation_eligible(
        completed_task_count=1,
        total_task_count=1,
        stop_after_task=None,
        gate_passes=[True],
        task_names=ORDER4_TASK_NAMES[:1],
        smoke=False,
        tiny_random_model=False,
    )
    # Passing smoke and tiny-random runs are intrinsically test-sealed.
    for smoke, tiny in ((True, False), (False, True), (True, True)):
        assert not runner.official_evaluation_eligible(
            completed_task_count=len(ORDER4_TASK_NAMES),
            total_task_count=len(ORDER4_TASK_NAMES),
            stop_after_task=None,
            gate_passes=[True] * len(ORDER4_TASK_NAMES),
            task_names=ORDER4_TASK_NAMES,
            smoke=smoke,
            tiny_random_model=tiny,
        )
    assert runner.official_evaluation_eligible(
        completed_task_count=len(ORDER4_TASK_NAMES),
        total_task_count=len(ORDER4_TASK_NAMES),
        stop_after_task=None,
        gate_passes=[True] * len(ORDER4_TASK_NAMES),
        task_names=ORDER4_TASK_NAMES,
        smoke=False,
        tiny_random_model=False,
    )


def test_explicit_full_order_stop_keeps_official_test_sealed() -> None:
    assert not runner.official_evaluation_eligible(
        completed_task_count=len(ORDER4_TASK_NAMES),
        total_task_count=len(ORDER4_TASK_NAMES),
        stop_after_task=len(ORDER4_TASK_NAMES),
        gate_passes=[True] * len(ORDER4_TASK_NAMES),
        task_names=ORDER4_TASK_NAMES,
        smoke=False,
        tiny_random_model=False,
    )


def test_natural_prior_reweight_and_task_incoming_retention_formula() -> None:
    assert runner.natural_prior_score(
        {"False": 0.8, "True": 0.4}, {"False": 0.75, "True": 0.25}
    ) == pytest.approx(0.7)
    acquisition, bwt, retention = runner.offline_retention_vectors(
        [0.1, 0.4], [0.6, 0.6], [0.5, 0.5]
    )
    assert acquisition.tolist() == pytest.approx([0.5, 0.2])
    assert bwt.tolist() == pytest.approx([-0.1, -0.1])
    assert retention.tolist() == pytest.approx([0.8, 0.5])


def test_consolidation_uses_same_sequence_nll_after_both_sampling_paths() -> None:
    source = inspect.getsource(runner.train_consolidation)
    assert "nll[cursor:].mean()" in source
    assert "anchored.regret.mean()" not in source


def test_budget_matcher_rejects_unequal_steps() -> None:
    payload = {
        "active_atoms": 144,
        "active_trainable_scalars": 147456,
        "exact_budget": True,
    }
    base = {
        "final_payload": payload,
        "total_training_budget": {
            "acquisition_optimizer_steps": 10,
            "consolidation_optimizer_steps": 5,
            "acquisition_current_example_exposures": 40,
            "consolidation_current_example_exposures": 20,
            "consolidation_replay_example_exposures": 16,
            "controller_forward_examples": 8,
            "controller_forward_batches": 2,
            "acquisition_source_tokens": 100,
            "acquisition_target_tokens": 40,
            "consolidation_current_source_tokens": 50,
            "consolidation_current_target_tokens": 20,
            "consolidation_replay_source_tokens": 55,
            "consolidation_replay_target_tokens": 16,
            "controller_forward_source_tokens": 30,
            "controller_forward_target_tokens": 8,
        },
        "stages": [
            {
                "acquisition_optimizer_steps": 10,
                "consolidation_optimizer_steps": 5,
                "acquisition_current_example_exposures": 40,
                "consolidation_current_example_exposures": 20,
                "consolidation_replay_example_exposures": 16,
                "controller_forward_examples": 8,
                "controller_forward_batches": 2,
                "task": "MNLI",
                "payload": payload,
            }
        ],
    }
    fairness = runner.assert_matched_training_budgets(
        {"uniform_er_rank4": base, "task_id_oracle": dict(base)}
    )
    assert fairness["token_count_matched"]
    assert not fairness["flop_matched"]
    token_different = copy.deepcopy(base)
    token_different["total_training_budget"]["consolidation_replay_source_tokens"] += 3
    fairness = runner.assert_matched_training_budgets(
        {"uniform_er_rank4": base, "task_id_oracle": token_different}
    )
    assert not fairness["token_count_matched"]
    assert not fairness["flop_matched"]
    assert "no token- or FLOP-matched claim" in fairness["claim"]
    bad = dict(base)
    bad["total_training_budget"] = dict(base["total_training_budget"])
    bad["total_training_budget"]["consolidation_replay_example_exposures"] = 17
    with pytest.raises(RuntimeError, match="unequal matched resource"):
        runner.assert_matched_training_budgets(
            {"uniform_er_rank4": base, "task_id_oracle": bad}
        )


def test_official_test_loader_is_confined_to_post_training_evaluator() -> None:
    tree = ast.parse(inspect.getsource(runner))
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_sealed_evaluation_data"
    ]
    assert len(calls) == 1
    owner = parents[calls[0]]
    while owner is not tree and not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
        owner = parents[owner]
    assert isinstance(owner, ast.FunctionDef)
    assert owner.name == "evaluate_all_final_checkpoints"
    source = inspect.getsource(runner.evaluate_all_final_checkpoints)
    assert "base_checkpoint" in source
    assert "incoming_checkpoint" in source
    assert "phase_a_immediate_checkpoint" in source
    assert "initial_pretrained_base_auxiliary" in source
    assert "normalized_retention" in source


def test_official_evaluator_rejects_any_failed_acquisition_before_file_access() -> None:
    failed = {
        arm: {"status": "training_completed_acquisition_failed_test_sealed", "all_acquisition_gates_pass": False}
        for arm in runner.LOCKED_ARMS
    }
    config = SimpleNamespace(
        arms=runner.LOCKED_ARMS,
        output_root="missing",
        task_names=ORDER4_TASK_NAMES,
        smoke=False,
        tiny_random_model=False,
    )
    with pytest.raises(RuntimeError, match="sealed until every"):
        runner.evaluate_all_final_checkpoints(
            failed,
            None,
            config=config,  # type: ignore[arg-type]
        )


def test_official_evaluator_rechecks_cross_arm_resources_before_loader(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "active_atoms": 144,
        "active_trainable_scalars": 147456,
        "exact_budget": True,
    }
    stages = [
        {
            "task": task,
            "acquisition_pass": True,
            "acquisition_optimizer_steps": 1,
            "consolidation_optimizer_steps": 1,
            "acquisition_current_example_exposures": 2,
            "consolidation_current_example_exposures": 2,
            "consolidation_replay_example_exposures": 2,
            "controller_forward_examples": 2,
            "controller_forward_batches": 1,
            "payload": payload,
        }
        for task in ORDER4_TASK_NAMES
    ]
    totals = {
        "acquisition_optimizer_steps": len(stages),
        "consolidation_optimizer_steps": len(stages),
        "acquisition_current_example_exposures": 2 * len(stages),
        "consolidation_current_example_exposures": 2 * len(stages),
        "consolidation_replay_example_exposures": 2 * len(stages),
        "controller_forward_examples": 2 * len(stages),
        "controller_forward_batches": len(stages),
        "acquisition_source_tokens": 100,
        "acquisition_target_tokens": 20,
        "consolidation_current_source_tokens": 100,
        "consolidation_current_target_tokens": 20,
        "consolidation_replay_source_tokens": 100,
        "consolidation_replay_target_tokens": 20,
        "controller_forward_source_tokens": 100,
        "controller_forward_target_tokens": 20,
    }
    uniform = {
        "status": "training_completed",
        "all_acquisition_gates_pass": True,
        "stages": stages,
        "total_training_budget": totals,
        "final_payload": payload,
    }
    oracle = copy.deepcopy(uniform)
    oracle["total_training_budget"]["consolidation_replay_example_exposures"] += 1
    opened = False

    def forbidden_loader(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("sealed loader must not be reached")

    monkeypatch.setattr(runner, "load_sealed_evaluation_data", forbidden_loader)
    config = SimpleNamespace(
        arms=runner.LOCKED_ARMS,
        output_root=str(tmp_path),
        task_names=ORDER4_TASK_NAMES,
        smoke=False,
        tiny_random_model=False,
    )
    with pytest.raises(RuntimeError, match="unequal matched resource"):
        runner.evaluate_all_final_checkpoints(
            {runner.LOCKED_ARMS[0]: uniform, runner.LOCKED_ARMS[1]: oracle},
            None,
            config=config,  # type: ignore[arg-type]
        )
    assert not opened


def test_official_evaluator_preflights_every_checkpoint_before_loader(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = {"path": str(tmp_path / "missing.pt"), "sha256": "0" * 64}
    stages = [
        {
            "task": task,
            "acquisition_pass": True,
            "incoming_checkpoint": missing,
            "phase_a_immediate_checkpoint": missing,
            "deployed_checkpoint": missing,
        }
        for task in ORDER4_TASK_NAMES
    ]
    result = {
        "status": "training_completed",
        "all_acquisition_gates_pass": True,
        "stages": stages,
        "base_checkpoint": missing,
        "final_checkpoint": missing,
    }
    for arm in runner.LOCKED_ARMS:
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        (arm_dir / "TRAINING_COMPLETED").write_text("ok\n", encoding="utf-8")
    opened = False

    def forbidden_loader(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("sealed loader must not be reached")

    monkeypatch.setattr(runner, "load_sealed_evaluation_data", forbidden_loader)
    monkeypatch.setattr(runner, "assert_matched_training_budgets", lambda results: {})
    config = SimpleNamespace(
        arms=runner.LOCKED_ARMS,
        output_root=str(tmp_path),
        task_names=ORDER4_TASK_NAMES,
        smoke=False,
        tiny_random_model=False,
    )
    with pytest.raises(RuntimeError, match="checkpoint hash mismatch"):
        runner.evaluate_all_final_checkpoints(
            {runner.LOCKED_ARMS[0]: result, runner.LOCKED_ARMS[1]: copy.deepcopy(result)},
            None,
            config=config,  # type: ignore[arg-type]
        )
    assert not opened


def test_synthetic_pass_path_uses_task_incoming_not_global_initial_base(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "active_atoms": 144,
        "active_trainable_scalars": 147456,
        "exact_budget": True,
    }
    checkpoint_infos = {}
    for marker in ("base", "incoming_late", "immediate", "final"):
        path = tmp_path / f"{marker}.pt"
        torch.save({"marker": marker}, path)
        checkpoint_infos[marker] = {
            "path": str(path),
            "sha256": runner._sha256_file(path),
        }

    class FakeBank:
        marker = ""

        def load_compact_state_dict(self, state) -> None:
            self.marker = state["marker"]

        def payload_audit(self):
            return SimpleNamespace(to_dict=lambda: payload)

    build_count = 0

    def fake_build(*args, **kwargs):
        nonlocal build_count
        build_count += 1
        return FakeBank()

    opened_tasks: list[str] = []

    def fake_sealed_loader(data_root, task_name):
        opened_tasks.append(task_name)
        examples = tuple(
            OfficialExample(
                task_name=task_name,
                category=task_name,
                dataset=task_name,
                subset="test",
                source_index=index,
                example_id=f"{task_name}-test-{index}",
                sentence=f"sentence {index}",
                label=label,
                prompt=f"HEADER\nsentence {index}\nAnswer:",
            )
            for index, label in enumerate(runner.TASK_BY_NAME[task_name].labels)
        )
        return SimpleNamespace(
            examples=examples,
            excluded_unlabeled_count=0,
            excluded_unlabeled_sha256="0" * 64,
        )

    scores = {"base": 0.1, "incoming_late": 0.4, "immediate": 0.6, "final": 0.5}

    def fake_evaluate(bank, *args, **kwargs):
        score = scores[bank.marker]
        return {"natural_prior_em": score, "normalized_em": score}

    monkeypatch.setattr(runner, "_build_model", fake_build)
    monkeypatch.setattr(runner, "load_sealed_evaluation_data", fake_sealed_loader)
    monkeypatch.setattr(runner, "evaluate_free_em", fake_evaluate)

    for arm in runner.LOCKED_ARMS:
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        (arm_dir / "TRAINING_COMPLETED").write_text("ok\n", encoding="utf-8")

    stages = [
        {
            "task": task,
            "acquisition_pass": True,
            "acquisition_optimizer_steps": 1,
            "consolidation_optimizer_steps": 1,
            "acquisition_current_example_exposures": 2,
            "consolidation_current_example_exposures": 2,
            "consolidation_replay_example_exposures": 2,
            "controller_forward_examples": 2,
            "controller_forward_batches": 1,
            "payload": payload,
            "incoming_checkpoint": (
                checkpoint_infos["base"]
                if index == 0
                else checkpoint_infos["incoming_late"]
            ),
            "phase_a_immediate_checkpoint": checkpoint_infos["immediate"],
            "deployed_checkpoint": checkpoint_infos["incoming_late"],
        }
        for index, task in enumerate(ORDER4_TASK_NAMES)
    ]
    totals = {
        "acquisition_optimizer_steps": len(stages),
        "consolidation_optimizer_steps": len(stages),
        "acquisition_current_example_exposures": 2 * len(stages),
        "consolidation_current_example_exposures": 2 * len(stages),
        "consolidation_replay_example_exposures": 2 * len(stages),
        "controller_forward_examples": 2 * len(stages),
        "controller_forward_batches": len(stages),
        "acquisition_source_tokens": 100,
        "acquisition_target_tokens": 20,
        "consolidation_current_source_tokens": 100,
        "consolidation_current_target_tokens": 20,
        "consolidation_replay_source_tokens": 100,
        "consolidation_replay_target_tokens": 20,
        "controller_forward_source_tokens": 100,
        "controller_forward_target_tokens": 20,
    }
    one_result = {
        "status": "training_completed",
        "all_acquisition_gates_pass": True,
        "stages": stages,
        "total_training_budget": totals,
        "final_payload": payload,
        "base_checkpoint": checkpoint_infos["base"],
        "final_checkpoint": checkpoint_infos["final"],
    }
    training_results = {
        runner.LOCKED_ARMS[0]: one_result,
        runner.LOCKED_ARMS[1]: copy.deepcopy(one_result),
    }
    config = SimpleNamespace(
        arms=runner.LOCKED_ARMS,
        output_root=str(tmp_path),
        task_names=ORDER4_TASK_NAMES,
        smoke=False,
        tiny_random_model=False,
        data_root="synthetic",
        eval_per_task=max(
            len(runner.TASK_BY_NAME[task].labels) for task in ORDER4_TASK_NAMES
        ),
        seed=42,
        cvar_alpha=0.2,
    )
    result = runner.evaluate_all_final_checkpoints(
        training_results,
        None,
        config=config,  # type: ignore[arg-type]
    )
    assert opened_tasks == list(ORDER4_TASK_NAMES)
    assert build_count == 2 * (2 * len(ORDER4_TASK_NAMES) + 2)
    for arm in runner.LOCKED_ARMS:
        task_metrics = result["arms"][arm]["task_metrics"]
        assert task_metrics[ORDER4_TASK_NAMES[0]]["incoming_pre_t"] == pytest.approx(0.1)
        assert task_metrics[ORDER4_TASK_NAMES[0]]["normalized_retention"] == pytest.approx(0.8)
        assert task_metrics[ORDER4_TASK_NAMES[1]]["incoming_pre_t"] == pytest.approx(0.4)
        assert task_metrics[ORDER4_TASK_NAMES[1]]["normalized_retention"] == pytest.approx(0.5)
        assert task_metrics[ORDER4_TASK_NAMES[1]][
            "initial_base_normalized_retention_auxiliary"
        ] == pytest.approx(0.8)
        assert result["arms"][arm]["summary"][
            "lower_tail_normalized_retention"
        ] == pytest.approx(0.5)
    assert result["oracle_headroom"]["lower_tail_normalized_retention"] == pytest.approx(0.0)
