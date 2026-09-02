from __future__ import annotations

import pytest

from experiments.phase2i_anchored_cvar.order4_data import (
    ORDER4_TASK_NAMES,
    OfficialExample,
)
from experiments.phase2i_anchored_cvar.run_order4_panel import (
    _config_from_args,
    build_argument_parser,
    parse_arm_selection,
    parse_task_selection,
    select_evaluation_examples,
)


def _example(index: int, label: str) -> OfficialExample:
    return OfficialExample(
        task_name="MNLI",
        category="NLI",
        dataset="MNLI",
        subset="test",
        source_index=index,
        example_id=f"id-{index}",
        sentence=f"sentence {index}",
        label=label,
        prompt=f"prompt {index}",
    )


def test_task_selection_only_accepts_order4_prefix() -> None:
    assert parse_task_selection("2") == ORDER4_TASK_NAMES[:2]
    assert parse_task_selection("MNLI,CB") == ORDER4_TASK_NAMES[:2]
    assert parse_task_selection("all") == ORDER4_TASK_NAMES
    with pytest.raises(ValueError, match="chronology-preserving"):
        parse_task_selection("MNLI,WiC")


def test_arm_selection_is_returned_in_locked_order() -> None:
    assert parse_arm_selection(["cvar_rank,fixed_er"]) == ("fixed_er", "cvar_rank")
    with pytest.raises(ValueError, match="unknown arms"):
        parse_arm_selection(["not_an_arm"])


def test_smoke_preserves_selected_prefix_and_locks_small_caps(tmp_path) -> None:
    args = build_argument_parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--model-path",
            str(tmp_path / "model"),
            "--output-root",
            str(tmp_path / "out"),
            "--tasks",
            "all",
            "--smoke",
        ]
    )
    config = _config_from_args(args)
    assert config.task_names == ORDER4_TASK_NAMES
    assert config.per_class_cap == 4
    assert config.risk_per_class == 1
    assert config.audit_per_class == 1
    assert config.eval_per_task == 4


def test_sealed_eval_selection_is_deterministic_and_balanced() -> None:
    examples = tuple(
        _example(index, "a" if index < 6 else "b") for index in range(12)
    )
    first = select_evaluation_examples(examples, 4, seed=7)
    second = select_evaluation_examples(tuple(reversed(examples)), 4, seed=7)
    assert first == second
    assert {example.label for example in first} == {"a", "b"}
    assert len(first) == 4

