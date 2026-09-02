from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
from pathlib import Path

import pytest
import torch

from experiments.phase2i_anchored_cvar.order4_data import OfficialExample
from experiments.phase2i_anchored_cvar import run_acquisition_rescue as rescue


def _examples(task: str = "QQP", subset: str = "train") -> tuple[OfficialExample, ...]:
    labels = ("False", "True")
    output = []
    for label_index, label in enumerate(labels):
        for local_index in range(12):
            index = label_index * 100 + local_index
            output.append(
                OfficialExample(
                    task_name=task,
                    category="QQP",
                    dataset=task,
                    subset=subset,
                    source_index=index,
                    example_id=f"{task}-{label}-{local_index}",
                    sentence=f"sentence {index}",
                    label=label,
                    prompt=f"prompt {index}",
                )
            )
    return tuple(output)


def test_three_way_split_is_deterministic_disjoint_and_nested() -> None:
    examples = _examples()
    first = rescue.build_rescue_split(
        examples,
        task_name="QQP",
        max_update_per_class=4,
        tune_audit_per_class=2,
        confirm_audit_per_class=3,
        seed=17,
    )
    second = rescue.build_rescue_split(
        tuple(reversed(examples)),
        task_name="QQP",
        max_update_per_class=4,
        tune_audit_per_class=2,
        confirm_audit_per_class=3,
        seed=17,
    )
    first.assert_disjoint()
    assert first == second
    assert len(first.update_pool) == 8
    assert len(first.tune_audit) == 4
    assert len(first.confirm_audit) == 6
    cap_two = {example.example_id for example in first.update_for_cap(2)}
    cap_four = {example.example_id for example in first.update_for_cap(4)}
    assert cap_two < cap_four
    assert cap_four.isdisjoint(example.example_id for example in first.tune_audit)
    assert cap_four.isdisjoint(example.example_id for example in first.confirm_audit)


def _multirc_example(
    index: int, *, paragraph: str, question: str, candidate: str, label: str
) -> OfficialExample:
    sentence = (
        f"paragraph: {paragraph}\nquestion: {question}"
        f"\ncandidate answer: {candidate}"
    )
    return OfficialExample(
        task_name="MultiRC",
        category="MultiRC",
        dataset="MultiRC",
        subset="train",
        source_index=index,
        example_id=f"multirc-semantic-{index}",
        sentence=sentence,
        label=label,
        prompt=f"prompt {index}",
    )


def test_semantic_split_keeps_all_multirc_candidates_in_one_partition() -> None:
    examples = [
        _multirc_example(
            0,
            paragraph="shared context",
            question="shared question?",
            candidate="candidate A",
            label="False",
        ),
        _multirc_example(
            1,
            paragraph="shared  context",  # whitespace normalization is semantic
            question="shared question?",
            candidate="candidate B",
            label="True",
        ),
    ]
    index = 2
    # Exact class quotas remain feasible after assigning the two-row group.
    for label in ("False", "True"):
        for local in range(5):
            examples.append(
                _multirc_example(
                    index,
                    paragraph=f"unique {label} context {local}",
                    question=f"unique question {local}?",
                    candidate="only candidate",
                    label=label,
                )
            )
            index += 1

    split = rescue.build_rescue_split(
        examples,
        task_name="MultiRC",
        max_update_per_class=1,
        tune_audit_per_class=1,
        confirm_audit_per_class=1,
        seed=23,
        split_unit="semantic_group",
    )
    rebuilt = rescue.build_rescue_split(
        tuple(reversed(examples)),
        task_name="MultiRC",
        max_update_per_class=1,
        tune_audit_per_class=1,
        confirm_audit_per_class=1,
        seed=23,
        split_unit="semantic_group",
    )
    assert split == rebuilt
    partitions = {
        "update": {item.example_id for item in split.update_pool},
        "tune": {item.example_id for item in split.tune_audit},
        "confirm": {item.example_id for item in split.confirm_audit},
    }
    owners = [
        name
        for name, ids in partitions.items()
        if {"multirc-semantic-0", "multirc-semantic-1"} & ids
    ]
    assert len(owners) <= 1
    if owners:
        assert {
            "multirc-semantic-0",
            "multirc-semantic-1",
        }.issubset(partitions[owners[0]])
    manifest = rescue._split_manifest(split)
    assert manifest["split_unit"] == "semantic_group"
    assert manifest["group_key_version"] == rescue.SEMANTIC_GROUP_KEY_VERSION
    assert manifest["semantic_group_sampling"]["selection_algorithm_version"] == (
        rescue.SEMANTIC_SELECTION_ALGORITHM_VERSION
    )
    assert "group_size_histogram" in manifest["confirm_audit"]
    assert "effective_group_count" in manifest["confirm_audit"]
    assert manifest["semantic_group_sampling"]["corpus"]["group_count"] == 11
    assert manifest["semantic_group_pairwise_overlap_counts"] == {
        "update_tune": 0,
        "update_confirm": 0,
        "tune_confirm": 0,
    }
    for partition in (split.update_pool, split.tune_audit, split.confirm_audit):
        assert {item.label for item in partition} == {"False", "True"}


def test_hash_priority_exact_solver_does_not_systematically_take_large_group() -> None:
    examples = [
        _multirc_example(
            0,
            paragraph="large shared context",
            question="large shared question?",
            candidate="false A",
            label="False",
        ),
        _multirc_example(
            1,
            paragraph="large shared context",
            question="large shared question?",
            candidate="false B",
            label="False",
        ),
        _multirc_example(
            2,
            paragraph="large shared context",
            question="large shared question?",
            candidate="true A",
            label="True",
        ),
        _multirc_example(
            3,
            paragraph="large shared context",
            question="large shared question?",
            candidate="true B",
            label="True",
        ),
    ]
    for index in range(4, 28):
        label = "False" if index % 2 == 0 else "True"
        examples.append(
            _multirc_example(
                index,
                paragraph=f"singleton context {index}",
                question=f"singleton question {index}?",
                candidate="candidate",
                label=label,
            )
        )
    keys = rescue._semantic_group_keys(examples, task_name="MultiRC")
    groups: dict[str, list[OfficialExample]] = {}
    for example in examples:
        groups.setdefault(keys[example.example_id], []).append(example)
    atomic_groups = {key: tuple(value) for key, value in groups.items()}
    large_ids = {f"multirc-semantic-{index}" for index in range(4)}

    selected_large = 0
    for seed in range(32):
        selected, _remaining = rescue._take_exact_group_partition(
            atomic_groups,
            task_name="MultiRC",
            labels=("False", "True"),
            quota_per_class=2,
            seed=seed,
        )
        selected_ids = {example.example_id for example in selected}
        assert len([example for example in selected if example.label == "False"]) == 2
        assert len([example for example in selected if example.label == "True"]) == 2
        assert not (selected_ids & large_ids) or large_ids.issubset(selected_ids)
        selected_large += int(large_ids.issubset(selected_ids))

    # v1 selected the largest feasible group for every seed.  Hash priority
    # sometimes reaches the exact quota with singleton groups first.
    assert 0 < selected_large < 32


@pytest.mark.parametrize("task_name", ("QQP", "BoolQA", "MultiRC"))
def test_real_pinned_train_semantic_dry_split_is_exact_and_disjoint(
    task_name: str,
) -> None:
    data_root = Path("D:/phase2i_order4_data")
    if not data_root.is_dir():
        pytest.skip("local pinned Order-4 data mirror is unavailable")
    examples = rescue.load_official_examples(
        data_root, task_name, "train", verify=True
    )
    split = rescue.build_rescue_split(
        examples,
        task_name=task_name,
        max_update_per_class=8,
        tune_audit_per_class=4,
        confirm_audit_per_class=4,
        seed=rescue._stable_seed(42, task_name, "three-way-split"),
        split_unit="semantic_group",
    )
    split.assert_disjoint()
    for partition, quota in (
        (split.update_pool, 8),
        (split.tune_audit, 4),
        (split.confirm_audit, 4),
    ):
        assert {
            label: sum(example.label == label for example in partition)
            for label in rescue.TASK_BY_NAME[task_name].labels
        } == {label: quota for label in rescue.TASK_BY_NAME[task_name].labels}
    manifest = rescue._split_manifest(split)
    assert manifest["semantic_group_pairwise_disjoint"] is True
    assert manifest["semantic_group_sampling"]["corpus"]["row_count"] == len(
        examples
    )


def test_legacy_row_manifest_schema_remains_hash_compatible() -> None:
    split = rescue.build_rescue_split(
        _examples(),
        task_name="QQP",
        max_update_per_class=2,
        tune_audit_per_class=1,
        confirm_audit_per_class=1,
        seed=5,
    )
    manifest = rescue._split_manifest(split)
    assert "split_unit" not in manifest
    assert "group_key_version" not in manifest


def test_legacy_semantic_algorithm_can_rebuild_old_manifest_schema() -> None:
    examples = []
    for index in range(12):
        label = "False" if index % 2 == 0 else "True"
        examples.append(
            _multirc_example(
                index,
                paragraph=f"legacy singleton {index}",
                question=f"legacy question {index}?",
                candidate="candidate",
                label=label,
            )
        )
    split = rescue.build_rescue_split(
        examples,
        task_name="MultiRC",
        max_update_per_class=2,
        tune_audit_per_class=1,
        confirm_audit_per_class=1,
        seed=7,
        split_unit="semantic_group",
        semantic_selection_algorithm=rescue.LEGACY_SEMANTIC_SELECTION_ALGORITHM,
    )
    manifest = rescue._split_manifest(split)
    assert "semantic_group_sampling" not in manifest
    assert "group_size_histogram" not in manifest["confirm_audit"]


def test_qqp_transitive_shared_question_component_cannot_cross_partitions() -> None:
    def qqp(index: int, first: str, second: str, label: str) -> OfficialExample:
        sentence = f"first sentence: {first}\nsecond sentence: {second}"
        return OfficialExample(
            task_name="QQP",
            category="QQP",
            dataset="QQP",
            subset="train",
            source_index=index,
            example_id=f"qqp-semantic-{index}",
            sentence=sentence,
            label=label,
            prompt=f"prompt {index}",
        )

    examples = [
        qqp(0, "question A", "question B", "False"),
        qqp(1, "question b", "question C", "True"),
        qqp(2, "question C", "question D", "False"),
    ]
    index = 3
    for label in ("False", "True"):
        for local in range(10):
            examples.append(
                qqp(
                    index,
                    f"unique {label} left {local}",
                    f"unique {label} right {local}",
                    label,
                )
            )
            index += 1
    split = rescue.build_rescue_split(
        examples,
        task_name="QQP",
        max_update_per_class=2,
        tune_audit_per_class=2,
        confirm_audit_per_class=2,
        seed=31,
        split_unit="semantic_group",
    )
    partition_ids = [
        {item.example_id for item in partition}
        for partition in (split.update_pool, split.tune_audit, split.confirm_audit)
    ]
    component = {"qqp-semantic-0", "qqp-semantic-1", "qqp-semantic-2"}
    owners = [ids for ids in partition_ids if ids & component]
    assert len(owners) <= 1
    if owners:
        assert component.issubset(owners[0])
    cap_one = {item.example_id for item in split.update_for_cap(1)}
    cap_two = {item.example_id for item in split.update_for_cap(2)}
    assert cap_one < cap_two


def test_prior_manifest_row_exclusion_expands_to_full_multirc_confirm_group(
    tmp_path: Path,
) -> None:
    examples = [
        _multirc_example(
            0,
            paragraph="shared context",
            question="shared question?",
            candidate="candidate A",
            label="False",
        ),
        _multirc_example(
            1,
            paragraph="shared context",
            question="shared question?",
            candidate="candidate B",
            label="True",
        ),
    ]
    index = 2
    for label in ("False", "True"):
        for local in range(8):
            examples.append(
                _multirc_example(
                    index,
                    paragraph=f"fresh {label} context {local}",
                    question=f"fresh question {local}?",
                    candidate="candidate",
                    label=label,
                )
            )
            index += 1
    prior_path = tmp_path / "prior_split.json"
    prior_path.write_text(
        json.dumps(
            {
                "task": "MultiRC",
                "update_pool": {"example_ids": ["multirc-semantic-0"]},
                "tune_audit": {"example_ids": ["multirc-semantic-2"]},
                "confirm_audit": {"example_ids": ["multirc-semantic-10"]},
            }
        ),
        encoding="utf-8",
    )
    requested, sources = rescue.load_prior_touched_ids(
        [str(prior_path)], task_name="MultiRC"
    )
    split = rescue.build_rescue_split(
        examples,
        task_name="MultiRC",
        max_update_per_class=1,
        tune_audit_per_class=1,
        confirm_audit_per_class=1,
        seed=41,
        split_unit="semantic_group",
        prior_touched_example_ids=requested,
        freshness_sources=sources,
    )
    assert "multirc-semantic-0" in split.prior_touched_example_ids
    # The second candidate was not listed in the old row manifest but shares
    # paragraph+question, so strict freshness excludes it too.
    assert "multirc-semantic-1" in split.prior_touched_example_ids
    confirm_ids = {item.example_id for item in split.confirm_audit}
    assert confirm_ids.isdisjoint(split.prior_touched_example_ids)
    freshness = rescue._split_manifest(split)[
        "fresh_confirm_against_prior_splits"
    ]
    assert freshness["requested_example_id_count"] == 3
    assert freshness["expanded_prior_touched_row_count"] == 4
    assert freshness["confirm_vs_prior_touched_group_overlap"] == 0
    assert freshness["partition_overlap_with_prior_touched_groups"][
        "confirm_audit"
    ] == 0


def test_access_only_ledger_excludes_preallocated_confirm_and_locks_content(
    tmp_path: Path,
) -> None:
    examples = _examples()
    source_split = rescue.build_rescue_split(
        examples,
        task_name="QQP",
        max_update_per_class=2,
        tune_audit_per_class=1,
        confirm_audit_per_class=2,
        seed=13,
    )
    source_path = tmp_path / "source_split.json"
    source_path.write_text(
        json.dumps(rescue._split_manifest(source_split)), encoding="utf-8"
    )
    update_evidence = tmp_path / "candidate_metrics.json"
    tune_evidence = tmp_path / "base_tune.json"
    tune_ids = [example.example_id for example in source_split.tune_audit]
    base_tune = {"example_ids": tune_ids, "source": "train.json audit partition"}
    tune_evidence.write_text(json.dumps(base_tune), encoding="utf-8")
    update_evidence.write_text(
        json.dumps(
            {
                "candidate_id": "unit-candidate",
                "candidate": {"cap_per_class": 2, "epochs": 1},
                "training": {
                    "completed_epochs": 1,
                    "resource_ledger": {
                        "gradient_examples_per_epoch": 4,
                        "gradient_example_exposures": 4,
                        "sampler_label_counts_per_epoch": {"False": 2, "True": 2},
                        "per_class": {
                            "False": {"row_exposures": 2},
                            "True": {"row_exposures": 2},
                        },
                        "tune_audit_gradient_examples": 0,
                        "confirm_audit_gradient_examples": 0,
                    },
                },
                "base_tune": base_tune,
                "post_tune": {"example_ids": tune_ids},
                "selection_partition": "tune_audit",
                "confirm_audit_accessed": False,
            }
        ),
        encoding="utf-8",
    )

    ledger = rescue.build_access_only_freshness_ledger(
        source_path,
        train_examples=examples,
        task_name="QQP",
        candidate_metrics_evidence_path=update_evidence,
        base_tune_evidence_path=tune_evidence,
    )
    ledger_path = tmp_path / "access_only.json"
    rescue._atomic_json(ledger_path, ledger)

    requested, sources = rescue.load_prior_touched_ids(
        [str(ledger_path)], task_name="QQP", train_examples=examples
    )
    expected_accessed = {
        example.example_id
        for example in (*source_split.update_pool, *source_split.tune_audit)
    }
    preallocated_confirm = {
        example.example_id for example in source_split.confirm_audit
    }
    assert requested == expected_accessed
    assert requested.isdisjoint(preallocated_confirm)
    assert ledger["confirm_audit"]["example_ids"] == []
    assert ledger["confirm_audit"]["status"] == "runner-attested-not-model-accessed"
    assert "not independently observable" in ledger["confirm_audit"][
        "nonaccess_claim_type"
    ]
    assert ledger["confirm_audit"]["id_metadata_read_by_ledger_builder"] is True
    assert ledger["confirm_audit"]["preallocated_but_not_touched_count"] == 4
    assert sources[0]["format"] == rescue.ACCESS_ONLY_FRESHNESS_FORMAT
    assert sources[0]["partition_row_counts"] == {
        "update_pool": 4,
        "tune_audit": 2,
        "confirm_audit": 0,
    }

    changed_examples = list(examples)
    changed_examples[0] = replace(changed_examples[0], prompt="content drift")
    with pytest.raises(ValueError, match="pinned-train content drifted"):
        rescue.load_prior_touched_ids(
            [str(ledger_path)],
            task_name="QQP",
            train_examples=changed_examples,
        )

    source_bytes = source_path.read_bytes()
    source_path.write_bytes(source_bytes + b"\n")
    with pytest.raises(ValueError, match="source split byte count drifted"):
        rescue.load_prior_touched_ids(
            [str(ledger_path)], task_name="QQP", train_examples=examples
        )
    source_path.write_bytes(source_bytes)

    evidence_bytes = update_evidence.read_bytes()
    update_evidence.write_bytes(evidence_bytes + b"\n")
    with pytest.raises(ValueError, match="candidate metrics evidence byte count drifted"):
        rescue.load_prior_touched_ids(
            [str(ledger_path)], task_name="QQP", train_examples=examples
        )
    update_evidence.write_bytes(evidence_bytes)

    tampered = json.loads(ledger_path.read_text(encoding="utf-8"))
    tampered["update_pool"]["example_ids"].reverse()
    ledger_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        rescue.load_prior_touched_ids(
            [str(ledger_path)], task_name="QQP", train_examples=examples
        )

    unsigned = dict(tampered)
    unsigned.pop("ledger_payload_sha256")
    tampered["ledger_payload_sha256"] = rescue._canonical_payload_sha256(unsigned)
    ledger_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="update_pool differs from source split"):
        rescue.load_prior_touched_ids(
            [str(ledger_path)], task_name="QQP", train_examples=examples
        )

    bad_metrics = json.loads(evidence_bytes.decode("utf-8"))
    bad_metrics["confirm_audit_accessed"] = True
    bad_evidence_path = tmp_path / "bad_candidate_metrics.json"
    bad_evidence_path.write_text(json.dumps(bad_metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="selection-only confirm access"):
        rescue.build_access_only_freshness_ledger(
            source_path,
            train_examples=examples,
            task_name="QQP",
            candidate_metrics_evidence_path=bad_evidence_path,
            base_tune_evidence_path=tune_evidence,
        )


def test_fresh_confirm_manifest_requires_semantic_group_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="require split_unit='semantic_group'"):
        rescue.build_rescue_split(
            _examples(),
            task_name="QQP",
            max_update_per_class=1,
            tune_audit_per_class=1,
            confirm_audit_per_class=1,
            seed=1,
            prior_touched_example_ids={"QQP-False-0"},
        )


def test_split_rejects_any_non_train_example() -> None:
    with pytest.raises(ValueError, match="train.json examples only"):
        rescue.build_rescue_split(
            _examples(subset="heldout"),
            task_name="QQP",
            max_update_per_class=2,
            tune_audit_per_class=1,
            confirm_audit_per_class=1,
            seed=1,
        )


def test_executable_has_exactly_one_constant_train_loader_call() -> None:
    source = inspect.getsource(rescue)
    assert "load_sealed_evaluation_data" not in source
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_official_examples"
    ]
    assert len(calls) == 1
    assert len(calls[0].args) >= 3
    split_argument = calls[0].args[2]
    assert isinstance(split_argument, ast.Constant)
    assert split_argument.value == "train"


def test_search_then_confirmation_defers_confirm_inspection_until_selection_file() -> None:
    source = inspect.getsource(rescue.run_task_search)
    tree = ast.parse(source)
    selection_writes = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_atomic_json"
        and any(
            isinstance(value, ast.Constant) and value.value == "selection.json"
            for argument in node.args
            for value in ast.walk(argument)
        )
    ]
    confirm_length_reads = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "source_token_length_report"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Attribute)
        and isinstance(node.args[1].value, ast.Name)
        and node.args[1].value.id == "split"
        and node.args[1].attr == "confirm_audit"
    ]
    assert len(selection_writes) == 1
    assert len(confirm_length_reads) == 1
    assert confirm_length_reads[0] > selection_writes[0]


def _measurement(
    spec: rescue.CandidateSpec,
    *,
    base: float,
    post: float,
) -> rescue.CandidateMeasurement:
    return rescue.CandidateMeasurement(
        candidate=spec,
        base_tune_em=base,
        post_tune_em=post,
        base_tune_label_constrained_accuracy=0.5,
        post_tune_label_constrained_accuracy=0.6,
        update_examples_per_epoch=spec.cap_per_class * 2,
        checkpoint=f"{spec.candidate_id}.pt",
    )


def test_selection_uses_free_generation_gain_then_resource_tiebreak() -> None:
    low_gain = _measurement(
        rescue.CandidateSpec(1e-3, 1, 2), base=0.4, post=0.5
    )
    expensive = _measurement(
        rescue.CandidateSpec(3e-3, 3, 4), base=0.4, post=0.6
    )
    efficient = _measurement(
        rescue.CandidateSpec(1e-3, 1, 4), base=0.4, post=0.6
    )
    assert rescue.select_candidate([low_gain, expensive, efficient]) == efficient


def test_candidate_grid_is_unique_and_epoch_checkpoints_share_lr_cap() -> None:
    grid = rescue.candidate_grid([1e-3, 1e-3], [1, 3, 5], [4])
    assert len(grid) == 3
    assert [candidate.epochs for candidate in grid] == [1, 3, 5]
    assert len({candidate.candidate_id for candidate in grid}) == 3


def test_candidate_grid_crosses_loss_objectives_and_ids_content_address_them() -> None:
    grid = rescue.candidate_grid(
        [1e-3],
        [1, 2],
        [96],
        rescue.LOSS_OBJECTIVES,
        "balanced",
    )
    assert len(grid) == 6
    assert {candidate.loss_objective for candidate in grid} == set(
        rescue.LOSS_OBJECTIVES
    )
    assert len({candidate.candidate_id for candidate in grid}) == 6
    assert all("balanced" in candidate.candidate_id for candidate in grid)


def test_selection_only_objective_plan_defers_token_ce_control() -> None:
    mixed = rescue.selection_objective_plan(
        ("hf_token_ce", "sequence_mean_balanced"), selection_only=True
    )
    assert mixed.requested_objectives == (
        "hf_token_ce",
        "sequence_mean_balanced",
    )
    assert mixed.selection_objectives == ("sequence_mean_balanced",)
    assert mixed.train_matched_control_after_selection

    implicit_control = rescue.selection_objective_plan(
        ("sequence_mean_prior",), selection_only=True
    )
    assert implicit_control.selection_objectives == ("sequence_mean_prior",)
    assert implicit_control.train_matched_control_after_selection

    legacy_default = rescue.selection_objective_plan(
        ("hf_token_ce",), selection_only=True
    )
    assert legacy_default.selection_objectives == ("hf_token_ce",)
    assert not legacy_default.train_matched_control_after_selection

    combined = rescue.selection_objective_plan(
        ("hf_token_ce", "sequence_mean_balanced"), selection_only=False
    )
    assert combined.selection_objectives == (
        "hf_token_ce",
        "sequence_mean_balanced",
    )
    assert not combined.train_matched_control_after_selection


def test_post_selection_control_is_ineligible_sealed_and_fully_accounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    examples = _examples()
    split = rescue.build_rescue_split(
        examples,
        task_name="QQP",
        max_update_per_class=2,
        tune_audit_per_class=2,
        confirm_audit_per_class=2,
        seed=19,
    )
    distribution = rescue.pinned_train_label_distribution(
        examples, task_name="QQP"
    )
    config = rescue.RescueConfig(
        data_root=str(tmp_path),
        model_path=str(tmp_path),
        output_root=str(tmp_path / "out"),
        task_names=("QQP",),
        learning_rates=(1e-3,),
        epochs=(1, 2),
        caps_per_class=(2,),
        tune_audit_per_class=2,
        confirm_audit_per_class=2,
        seed=7,
        batch_size=2,
        max_source_length=32,
        max_target_length=8,
        truncation_mode="official_right",
        min_rescue_gain=0.05,
        tiny_random_model=True,
        selection_only=True,
        confirm_selection=None,
        device="cpu",
        loss_objectives=("hf_token_ce", "sequence_mean_balanced"),
    )
    candidates = rescue.candidate_grid(
        config.learning_rates,
        config.epochs,
        config.caps_per_class,
        ("sequence_mean_balanced",),
        config.update_sampler,
    )
    tune_ids = {example.example_id for example in split.tune_audit}
    confirm_ids = {example.example_id for example in split.confirm_audit}
    events: list[tuple[str, int, bool]] = []
    evaluation_ids: list[set[str]] = []

    class FakePayload:
        exact_budget = True
        active_atoms = 144

        @staticmethod
        def to_dict():
            return {"exact_budget": True, "active_atoms": 144}

    class FakeBank:
        def __init__(self) -> None:
            self.parameter = torch.nn.Parameter(torch.tensor(0.0))
            self.trained_candidate = None

        def all_atom_parameters(self):
            return (self.parameter,)

        def compact_state_dict(self):
            return {"adapter": self.parameter.detach().clone()}

        @staticmethod
        def payload_audit():
            return FakePayload()

    def fake_train(
        bank,
        tokenizer,
        update_examples,
        candidate,
        optimizer,
        ledger,
        *,
        task_name,
        config,
        start_epoch,
        stop_epoch,
        natural_label_prior,
    ):
        del tokenizer, optimizer, natural_label_prior
        selection_lock_exists = (
            Path(config.output_root) / task_name / "selection_lock.json"
        ).is_file()
        events.append(
            (candidate.loss_objective, stop_epoch, selection_lock_exists)
        )
        if candidate.loss_objective == "hf_token_ce":
            assert selection_lock_exists
        bank.trained_candidate = candidate
        epoch_count = stop_epoch - start_epoch
        counts = {
            label: sum(example.label == label for example in update_examples)
            for label in rescue.TASK_BY_NAME[task_name].labels
        }
        ledger["sampler_label_counts_per_epoch"] = counts
        ledger["sampler_q"] = {label: 0.5 for label in counts}
        ledger["natural_label_prior"] = dict(distribution["natural_label_prior"])
        exposures = len(update_examples) * epoch_count
        ledger["gradient_example_exposures"] += exposures
        ledger["source_tokens"] += exposures * 3
        ledger["target_tokens"] += exposures * 2
        ledger["optimizer_steps"] += (
            (len(update_examples) + config.batch_size - 1) // config.batch_size
        ) * epoch_count
        for label, count in counts.items():
            class_exposures = count * epoch_count
            ledger["per_class"][label]["row_exposures"] += class_exposures
            ledger["per_class"][label]["target_tokens"] += class_exposures * 2
            ledger["per_class"][label]["objective_weight_sum"] += float(
                class_exposures
            )
        return [0.25] * epoch_count

    def fake_evaluate(bank, tokenizer, audit_examples, *, config, natural_label_prior):
        del tokenizer, config, natural_label_prior
        ids = {example.example_id for example in audit_examples}
        evaluation_ids.append(ids)
        assert ids == tune_ids
        assert ids.isdisjoint(confirm_ids)
        candidate = bank.trained_candidate
        if candidate is None:
            score = 0.40
        else:
            assert candidate.loss_objective == "sequence_mean_balanced"
            score = 0.50 if candidate.epochs == 1 else 0.60
        return {
            "normalized_em": score,
            "balanced_em": score,
            "natural_prior_em": score,
            "free_generation_valid_label_rate": 1.0,
            "per_class_recall": {"False": score, "True": score},
            "label_constrained_sequence_nll_accuracy": score,
        }

    monkeypatch.setattr(rescue, "_build_model", lambda config, tokenizer: FakeBank())
    monkeypatch.setattr(rescue, "_adapter_fingerprint", lambda bank: "same-init")
    monkeypatch.setattr(rescue, "train_update_segment", fake_train)
    monkeypatch.setattr(rescue, "evaluate_train_audit", fake_evaluate)
    monkeypatch.setattr(
        rescue,
        "source_token_length_report",
        lambda tokenizer, examples, **kwargs: {"count": len(examples)},
    )

    result = rescue.run_task_search(
        "QQP",
        split,
        candidates,
        tokenizer=object(),
        config=config,
        perform_confirmation=False,
        train_label_distribution=distribution,
    )

    selection = result["selection"]
    selected_id = selection["selected_candidate_id"]
    assert selection["selected"]["candidate"]["loss_objective"] == (
        "sequence_mean_balanced"
    )
    assert selection["selection_candidate_count"] == 2
    assert all(
        item["candidate"]["loss_objective"] != "hf_token_ce"
        for item in selection["candidate_registry"]
        if item["selection_eligible"]
    )
    controls = [
        item
        for item in selection["candidate_registry"]
        if not item["selection_eligible"]
    ]
    assert len(controls) == 1
    control = controls[0]
    assert control["candidate"]["loss_objective"] == "hf_token_ce"
    assert control["trained_after_selection_lock"]
    assert not control["confirm_audit_accessed"]
    assert control["resource_ledger_validation"]["status"] == "pass"
    assert result["selection_candidate_count"] == 2
    assert result["trained_candidate_count"] == 3
    assert result["post_selection_control_count"] == 1

    lock_path = Path(selection["selection_lock"]["path"])
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["selected_candidate_id"] == selected_id
    assert lock["selection_candidate_count"] == 2
    assert not lock["control_candidate_metrics_available"]
    assert not lock["confirm_audit_accessed"]
    bundle = selection["confirmation_bundle"]
    assert bundle["control_excluded_from_selection"]
    assert bundle["control_training_phase"] == (
        "post_selection_lock_pre_confirmation"
    )
    matched = bundle["matched_hf_token_ce"]
    assert matched["candidate_id"] == control["candidate_id"]
    assert not matched["selection_eligible"]
    for field in ("learning_rate", "epochs", "cap_per_class", "update_sampler"):
        assert matched["candidate"][field] == selection["selected"]["candidate"][
            field
        ]

    # Both selection trajectories run before the sole matched control; the
    # latter observes the durable lock and no evaluation ever sees confirm rows.
    assert [event[0] for event in events] == [
        "sequence_mean_balanced",
        "sequence_mean_balanced",
        "hf_token_ce",
    ]
    assert events[-1][2]
    assert len(evaluation_ids) == 3
    assert all(ids == tune_ids for ids in evaluation_ids)


def test_matched_control_preflight_rejects_identity_registry_and_lock_tampering(
    tmp_path: Path,
) -> None:
    selected_spec = rescue.CandidateSpec(
        1e-3, 2, 8, "sequence_mean_balanced", "balanced"
    )
    control_spec = rescue.CandidateSpec(
        1e-3, 2, 8, "hf_token_ce", "balanced"
    )
    selected_checkpoint = tmp_path / "selected.pt"
    control_checkpoint = tmp_path / "control.pt"
    selected_checkpoint.write_bytes(b"selected")
    control_checkpoint.write_bytes(b"control")
    selected_sha = rescue._sha256_file(selected_checkpoint)
    control_sha = rescue._sha256_file(control_checkpoint)
    selected_registry = {
        "candidate_id": selected_spec.candidate_id,
        "candidate": dict(selected_spec.__dict__),
        "checkpoint": str(selected_checkpoint),
        "checkpoint_sha256": selected_sha,
        "metrics": str(tmp_path / "selected-metrics.json"),
        "metrics_sha256": "a" * 64,
        "registry_role": "selection_candidate",
        "selection_eligible": True,
        "trained_after_selection_lock": False,
        "confirm_audit_accessed": False,
    }
    lock_payload = {
        "selected_candidate_id": selected_spec.candidate_id,
        "selected_checkpoint_sha256": selected_sha,
        "selection_candidate_ids": [selected_spec.candidate_id],
        "selection_candidate_registry_sha256": rescue._canonical_payload_sha256(
            [selected_registry]
        ),
        "confirm_audit_accessed": False,
    }
    lock_path = tmp_path / "selection_lock.json"
    rescue._atomic_json(lock_path, lock_payload)
    lock_sha = rescue._sha256_file(lock_path)
    control_registry = {
        "candidate_id": control_spec.candidate_id,
        "candidate": dict(control_spec.__dict__),
        "checkpoint": str(control_checkpoint),
        "checkpoint_sha256": control_sha,
        "registry_role": "post_selection_matched_hf_token_ce_control",
        "selection_eligible": False,
        "trained_after_selection_lock": True,
        "confirm_audit_accessed": False,
        "selection_lock_sha256": lock_sha,
    }
    selection = {
        "format": rescue.FORMAT_VERSION,
        "gate_version": "v2",
        "selected_candidate_id": selected_spec.candidate_id,
        "selected": {
            "candidate": dict(selected_spec.__dict__),
            "checkpoint": str(selected_checkpoint),
            "checkpoint_sha256": selected_sha,
        },
        "candidate_registry": [selected_registry, control_registry],
        "selection_lock": {"path": str(lock_path), "sha256": lock_sha},
        "confirmation_bundle": {
            "selected": {
                "candidate_id": selected_spec.candidate_id,
                "checkpoint": str(selected_checkpoint),
                "checkpoint_sha256": selected_sha,
            },
            "matched_hf_token_ce": {
                "candidate_id": control_spec.candidate_id,
                "candidate": dict(control_spec.__dict__),
                "checkpoint": str(control_checkpoint),
                "checkpoint_sha256": control_sha,
                "same_as_selected": False,
                "selection_eligible": False,
                "trained_after_selection_lock": True,
                "confirm_audit_accessed": False,
                "selection_lock_sha256": lock_sha,
            },
            "selection_lock": {"path": str(lock_path), "sha256": lock_sha},
            "control_training_phase": "post_selection_lock_pre_confirmation",
        },
    }
    valid = rescue._preflight_selection_confirmation(selection)
    assert valid["control_checkpoint"] == control_checkpoint.resolve()

    def cloned():
        return json.loads(json.dumps(selection))

    downgraded = cloned()
    downgraded.pop("gate_version")
    with pytest.raises(ValueError, match="non-downgradable v2 gate"):
        rescue._preflight_selection_confirmation(downgraded)

    wrong_objective = cloned()
    wrong_objective["confirmation_bundle"]["matched_hf_token_ce"][
        "candidate"
    ]["loss_objective"] = "sequence_mean_prior"
    with pytest.raises(ValueError, match="candidate|objective"):
        rescue._preflight_selection_confirmation(wrong_objective)

    mismatched_epoch = cloned()
    bad_spec = rescue.CandidateSpec(1e-3, 3, 8, "hf_token_ce", "balanced")
    matched = mismatched_epoch["confirmation_bundle"]["matched_hf_token_ce"]
    matched["candidate"] = dict(bad_spec.__dict__)
    matched["candidate_id"] = bad_spec.candidate_id
    mismatched_epoch["candidate_registry"][1]["candidate"] = dict(
        bad_spec.__dict__
    )
    mismatched_epoch["candidate_registry"][1]["candidate_id"] = (
        bad_spec.candidate_id
    )
    with pytest.raises(ValueError, match="mismatched epochs"):
        rescue._preflight_selection_confirmation(mismatched_epoch)

    forged_same = cloned()
    forged_same["confirmation_bundle"]["matched_hf_token_ce"][
        "same_as_selected"
    ] = True
    with pytest.raises(ValueError, match="same_as_selected"):
        rescue._preflight_selection_confirmation(forged_same)

    bad_registry = cloned()
    bad_registry["candidate_registry"][1]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="registry checkpoint"):
        rescue._preflight_selection_confirmation(bad_registry)

    bad_lock = cloned()
    bad_lock["confirmation_bundle"]["matched_hf_token_ce"][
        "selection_lock_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="selection-lock SHA"):
        rescue._preflight_selection_confirmation(bad_lock)

    # Older v2 grid controls had no redundant candidate/role/phase fields;
    # registry-backed validation remains compatible with that schema.
    legacy_v2 = cloned()
    legacy_v2["selection_lock"] = None
    legacy_bundle = legacy_v2["confirmation_bundle"]
    legacy_bundle.pop("selection_lock")
    legacy_bundle.pop("control_training_phase")
    legacy_matched = legacy_bundle["matched_hf_token_ce"]
    for key in (
        "candidate",
        "selection_eligible",
        "trained_after_selection_lock",
        "confirm_audit_accessed",
        "selection_lock_sha256",
    ):
        legacy_matched.pop(key)
    legacy_registry = legacy_v2["candidate_registry"][1]
    for key in (
        "registry_role",
        "selection_eligible",
        "trained_after_selection_lock",
        "confirm_audit_accessed",
        "selection_lock_sha256",
    ):
        legacy_registry.pop(key)
    assert rescue._preflight_selection_confirmation(legacy_v2)[
        "control_checkpoint"
    ] == control_checkpoint.resolve()


def test_atomic_json_payload_hash_matches_platform_file_bytes(tmp_path: Path) -> None:
    payload = {"nested": {"value": 1}, "rows": ["a", "b"]}
    path = tmp_path / "payload.json"
    rescue._atomic_json(path, payload)
    assert rescue._atomic_json_payload_sha256(payload) == rescue._sha256_file(path)


@pytest.mark.parametrize("artifact_version", ("v1", "v2_prebundle"))
def test_legacy_confirmation_schemas_still_complete_without_control_bundle(
    artifact_version: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / f"{artifact_version}.pt"
    torch.save({}, checkpoint)
    split = rescue.build_rescue_split(
        _examples(),
        task_name="QQP",
        max_update_per_class=2,
        tune_audit_per_class=2,
        confirm_audit_per_class=2,
        seed=29,
    )
    config = rescue.RescueConfig(
        data_root=str(tmp_path),
        model_path=str(tmp_path),
        output_root=str(tmp_path / "out"),
        task_names=("QQP",),
        learning_rates=(1e-3,),
        epochs=(1,),
        caps_per_class=(2,),
        tune_audit_per_class=2,
        confirm_audit_per_class=2,
        seed=7,
        batch_size=2,
        max_source_length=32,
        max_target_length=8,
        truncation_mode="official_right",
        min_rescue_gain=0.05,
        tiny_random_model=True,
        selection_only=False,
        confirm_selection=None,
        device="cpu",
    )

    class FakePayload:
        @staticmethod
        def to_dict():
            return {"active_atoms": 144}

    class FakeBank:
        loaded = False

        def load_compact_state_dict(self, state):
            del state
            self.loaded = True

        @staticmethod
        def payload_audit():
            return FakePayload()

    def fake_evaluate(bank, tokenizer, examples, *, config, natural_label_prior):
        del tokenizer, examples, config, natural_label_prior
        score = 0.60 if bank.loaded else 0.40
        return {
            "normalized_em": score,
            "balanced_em": score,
            "natural_prior_em": score,
            "free_generation_valid_label_rate": 1.0,
            "per_class_recall": {"False": score, "True": score},
            "label_constrained_sequence_nll_accuracy": score,
        }

    monkeypatch.setattr(rescue, "_build_model", lambda config, tokenizer: FakeBank())
    monkeypatch.setattr(rescue, "evaluate_train_audit", fake_evaluate)
    selection = {
        "format": (
            rescue.LEGACY_FORMAT_VERSION
            if artifact_version == "v1"
            else rescue.FORMAT_VERSION
        ),
        "task": "QQP",
        "selected_candidate_id": "legacy-selected",
        "selected": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": rescue._sha256_file(checkpoint),
        },
        "model_protocol": {},
        "min_rescue_gain": 0.05,
    }
    if artifact_version == "v2_prebundle":
        spec = rescue.CandidateSpec(
            1e-3, 1, 2, "sequence_mean_balanced", "balanced"
        )
        selection["selected_candidate_id"] = spec.candidate_id
        selection["selected"]["candidate"] = dict(spec.__dict__)
        selection["gate_version"] = "v2"
        selection["pinned_train_label_distribution"] = (
            rescue.pinned_train_label_distribution(_examples(), task_name="QQP")
        )

    confirmation = rescue.evaluate_selected_confirmation(
        selection,
        split,
        tokenizer=object(),
        config=config,
        selection_file_preexisted=True,
    )
    assert confirmation["confirm_rescue_pass"]
    assert confirmation["matched_hf_token_ce_control"] is None
    assert confirmation["gate_version"] == (
        "legacy_v1" if artifact_version == "v1" else "v2"
    )


def test_resource_ledger_validation_fails_closed_on_sampler_and_weight_mass() -> None:
    examples = _examples()[:2] + _examples()[12:14]
    candidate = rescue.CandidateSpec(
        1e-3, 1, 2, "sequence_mean_prior", "balanced"
    )
    prior = {"False": 0.8, "True": 0.2}
    ledger = rescue._new_resource_ledger("QQP", examples)
    ledger.update(
        {
            "gradient_example_exposures": 4,
            "source_tokens": 12,
            "target_tokens": 8,
            "optimizer_steps": 2,
            "sampler_label_counts_per_epoch": {"False": 2, "True": 2},
            "sampler_q": {"False": 0.5, "True": 0.5},
            "natural_label_prior": prior,
        }
    )
    ledger["per_class"]["False"].update(
        {"row_exposures": 2, "target_tokens": 4, "objective_weight_sum": 3.2}
    )
    ledger["per_class"]["True"].update(
        {"row_exposures": 2, "target_tokens": 4, "objective_weight_sum": 0.8}
    )
    assert rescue._validate_completed_resource_ledger(
        ledger,
        candidate,
        examples,
        task_name="QQP",
        batch_size=2,
        natural_label_prior=prior,
    )["status"] == "pass"

    missing_sampler = json.loads(json.dumps(ledger))
    missing_sampler.pop("sampler_q")
    with pytest.raises(RuntimeError, match="sampler/prior schema"):
        rescue._validate_completed_resource_ledger(
            missing_sampler,
            candidate,
            examples,
            task_name="QQP",
            batch_size=2,
            natural_label_prior=prior,
        )
    tiny_mass = json.loads(json.dumps(ledger))
    tiny_mass["per_class"]["False"]["objective_weight_sum"] = 1e-300
    with pytest.raises(RuntimeError, match="objective mass"):
        rescue._validate_completed_resource_ledger(
            tiny_mass,
            candidate,
            examples,
            task_name="QQP",
            batch_size=2,
            natural_label_prior=prior,
        )


def test_pinned_train_prior_is_derived_and_content_addressed() -> None:
    examples = _examples()
    distribution = rescue.pinned_train_label_distribution(
        examples, task_name="QQP"
    )
    assert distribution["label_counts"] == {"False": 12, "True": 12}
    assert distribution["natural_label_prior"] == {"False": 0.5, "True": 0.5}
    assert len(distribution["ordered_id_label_rows_sha256"]) == 64
    assert len(distribution["statistics_sha256"]) == 64
    changed = list(examples)
    changed[0] = OfficialExample(
        **{**changed[0].__dict__, "example_id": "content-changed"}
    )
    assert rescue.pinned_train_label_distribution(
        changed, task_name="QQP"
    )["ordered_id_label_rows_sha256"] != distribution[
        "ordered_id_label_rows_sha256"
    ]


def test_natural_and_balanced_metrics_are_both_explicit() -> None:
    metrics = rescue.free_generation_classification_metrics(
        ["False", "False", "False", "True"],
        ["False", "False", "True", "True"],
        labels=("False", "True"),
        natural_label_prior={"False": 0.8, "True": 0.2},
    )
    assert metrics["per_class_recall"] == {"False": 1.0, "True": 0.5}
    assert metrics["balanced_em"] == pytest.approx(0.75)
    assert metrics["natural_prior_em"] == pytest.approx(0.9)
    assert metrics["worst_class_recall"] == pytest.approx(0.5)
    assert metrics["free_generation_valid_label_rate"] == 1.0


def test_sequence_prior_weights_are_pi_over_q_without_batch_normalization() -> None:
    weights = rescue.sequence_objective_weights(
        "sequence_mean_prior",
        ["False", "True"],
        natural_label_prior={"False": 0.8, "True": 0.2},
        sampler_q={"False": 0.5, "True": 0.5},
    )
    assert torch.equal(weights, torch.tensor([1.6, 0.4]))
    # The batch sum happens to be two only when this batch mirrors q; the
    # implementation returns raw pi/q rather than normalizing each batch.
    one_class = rescue.sequence_objective_weights(
        "sequence_mean_prior",
        ["False"],
        natural_label_prior={"False": 0.8, "True": 0.2},
        sampler_q={"False": 0.5, "True": 0.5},
    )
    assert one_class.item() == pytest.approx(1.6)


def test_v2_gate_is_strict_and_checks_validity_and_class_regression() -> None:
    spec = rescue.CandidateSpec(
        1e-3, 1, 2, "sequence_mean_prior", "balanced"
    )

    def measurement(gain: float, valid: float, true_delta: float):
        return rescue.CandidateMeasurement(
            candidate=spec,
            base_tune_em=0.5,
            post_tune_em=0.5 + gain,
            base_tune_label_constrained_accuracy=0.5,
            post_tune_label_constrained_accuracy=0.5,
            update_examples_per_epoch=4,
            checkpoint="unused.pt",
            base_tune_natural_prior_em=0.5,
            post_tune_natural_prior_em=0.5 + gain,
            post_tune_valid_rate=valid,
            base_tune_per_class_recall={"False": 0.5, "True": 0.5},
            post_tune_per_class_recall={
                "False": 0.6,
                "True": 0.5 + true_delta,
            },
        )

    assert not measurement(0.05, 1.0, 0.0).v2_gate_pass(0.05)
    assert not measurement(0.06, 0.98, 0.0).v2_gate_pass(0.05)
    assert not measurement(0.06, 1.0, -0.051).v2_gate_pass(0.05)
    assert measurement(0.06, 1.0, -0.05).v2_gate_pass(0.05)


def test_selection_prefers_candidate_that_passes_all_v2_guards() -> None:
    def item(objective: str, gain: float, valid: float) -> rescue.CandidateMeasurement:
        return rescue.CandidateMeasurement(
            candidate=rescue.CandidateSpec(1e-3, 1, 4, objective, "balanced"),
            base_tune_em=0.5,
            post_tune_em=0.5 + gain,
            base_tune_label_constrained_accuracy=0.5,
            post_tune_label_constrained_accuracy=0.5,
            update_examples_per_epoch=8,
            checkpoint="unused.pt",
            base_tune_natural_prior_em=0.5,
            post_tune_natural_prior_em=0.5 + gain,
            post_tune_valid_rate=valid,
            base_tune_per_class_recall={"False": 0.5, "True": 0.5},
            post_tune_per_class_recall={"False": 0.55, "True": 0.55},
        )

    invalid_large_gain = item("hf_token_ce", 0.2, 0.98)
    valid_smaller_gain = item("sequence_mean_prior", 0.06, 1.0)
    assert rescue.select_candidate(
        [invalid_large_gain, valid_smaller_gain], min_gain=0.05
    ) == valid_smaller_gain


def test_legacy_v1_selection_can_load_but_is_distinguishable(tmp_path: Path) -> None:
    artifact = tmp_path / "selection.json"
    artifact.write_text(
        json.dumps(
            {
                "format": rescue.LEGACY_FORMAT_VERSION,
                "task": "BoolQA",
                "split_protocol": {},
                "evaluation_protocol": {},
            }
        ),
        encoding="utf-8",
    )
    loaded = rescue.load_selection_artifact(artifact)
    assert loaded["format"] == rescue.LEGACY_FORMAT_VERSION
    assert loaded.get("gate_version") != "v2"


def test_supported_task_selection_is_locked() -> None:
    assert rescue.parse_tasks(["all"]) == ("QQP", "BoolQA", "MultiRC")
    assert rescue.parse_tasks(["multirc,qqp"]) == ("QQP", "MultiRC")
    with pytest.raises(ValueError, match="unsupported task"):
        rescue.parse_tasks(["MNLI"])


def test_field_aware_components_protect_hard_task_fields() -> None:
    bool_sentence = "question: is this true\npassage: " + "long passage " * 20
    bool_example = OfficialExample(
        task_name="BoolQA",
        category="BoolQA",
        dataset="BoolQA",
        subset="train",
        source_index=0,
        example_id="bool-0",
        sentence=bool_sentence,
        label="True",
        prompt="HEADER\n" + bool_sentence + "\nAnswer:",
    )
    head, flexible, suffix = rescue._field_aware_components(bool_example)
    assert "question: is this true" in head
    assert flexible.startswith("long passage")
    assert suffix == "\nAnswer:"

    multi_sentence = (
        "paragraph: "
        + "long paragraph " * 20
        + "\nquestion: what happened?\ncandidate answer: something"
    )
    multi_example = OfficialExample(
        task_name="MultiRC",
        category="MultiRC",
        dataset="MultiRC",
        subset="train",
        source_index=0,
        example_id="multi-0",
        sentence=multi_sentence,
        label="False",
        prompt="HEADER\n" + multi_sentence + "\nAnswer:",
    )
    head, flexible, suffix = rescue._field_aware_components(multi_example)
    assert head.endswith("paragraph: ")
    assert flexible.startswith("long paragraph")
    assert "question: what happened?" in suffix
    assert "candidate answer: something" in suffix
    assert suffix.endswith("\nAnswer:")


class _CharacterTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        **kwargs,
    ):
        assert isinstance(text, str)
        ids = [ord(character) + 1 for character in text]
        if add_special_tokens:
            ids.append(1)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def num_special_tokens_to_add(self, pair=False):
        return 1

    def build_inputs_with_special_tokens(self, content):
        return [*content, 1]


def test_field_aware_ids_respect_budget_and_keep_multirc_suffix() -> None:
    sentence = (
        "paragraph: "
        + "P" * 200
        + "\nquestion: Q?\ncandidate answer: C"
    )
    example = OfficialExample(
        task_name="MultiRC",
        category="MultiRC",
        dataset="MultiRC",
        subset="train",
        source_index=0,
        example_id="multi-budget",
        sentence=sentence,
        label="True",
        prompt="H\n" + sentence + "\nAnswer:",
    )
    tokenizer = _CharacterTokenizer()
    ids = rescue._field_aware_ids(tokenizer, example, 80)
    encoded_suffix = tokenizer(
        "\nquestion: Q?\ncandidate answer: C\nAnswer:",
        add_special_tokens=False,
    )["input_ids"]
    assert len(ids) == 80
    assert ids[-1] == 1
    assert ids[-1 - len(encoded_suffix) : -1] == encoded_suffix
