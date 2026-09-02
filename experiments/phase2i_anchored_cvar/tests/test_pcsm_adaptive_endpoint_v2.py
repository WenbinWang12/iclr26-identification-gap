from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

from experiments.phase2i_anchored_cvar import run_acquisition_rescue as rescue
from experiments.phase2i_anchored_cvar import run_bap_lora_rescue as bap
from experiments.phase2i_anchored_cvar import run_pcsm_adaptive_endpoint_v2 as v2
from experiments.phase2i_anchored_cvar.order4_data import OfficialExample


def _qqp_example(index: int, label: str) -> OfficialExample:
    sentence = (
        f"first sentence: unique first question {index}\n"
        f"second sentence: unique second question {index}"
    )
    return OfficialExample(
        task_name="QQP",
        category="QQP",
        dataset="QQP",
        subset="train",
        source_index=index,
        example_id=f"qqp-v2-{index}",
        sentence=sentence,
        label=label,
        prompt=f"QQP\n{sentence}\nAnswer:",
    )


def _corpus(per_class: int = 8) -> tuple[OfficialExample, ...]:
    return tuple(
        _qqp_example(label_index * 100 + local_index, label)
        for label_index, label in enumerate(("True", "False"))
        for local_index in range(per_class)
    )


def _config(tmp_path: Path, *, output_name: str = "selection") -> v2.TransferConfig:
    model_path = tmp_path / "model"
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "config.json").write_text(
        '{"model_type":"t5"}\n', encoding="utf-8"
    )
    return v2.TransferConfig(
        data_root=str(tmp_path),
        model_path=str(model_path),
        output_root=str(tmp_path / output_name),
        task_name="QQP",
        learning_rate=3e-4,
        fit_per_class=2,
        calibration_per_class=2,
        tune_audit_per_class=2,
        confirm_audit_per_class=2,
        optimizer_steps=2,
        seed=17,
        batch_size=4,
        max_source_length=32,
        max_target_length=8,
        tiny_random_model=True,
    )


def _raw_comparison(
    *, gain: float, minimum_class_delta: float, valid: float = 1.0
) -> dict[str, Any]:
    return {
        "natural_prior_gain": gain,
        "minimum_per_class_delta": minimum_class_delta,
        "post_valid_label_rate": valid,
    }


def _calibrated_comparison(
    *, gain: float, minimum_class_delta: float, auc_delta: float
) -> dict[str, Any]:
    return {
        "training_contribution_natural_prior": gain,
        "training_min_per_class_delta": minimum_class_delta,
        "margin_diagnostics": {"roc_auc_delta": auc_delta},
    }


@pytest.mark.parametrize(
    ("raw_gain", "raw_class", "valid", "auc", "cal_gain", "cal_class", "expected"),
    [
        (0.06, -0.05, 0.99, 0.01, -1.0, -1.0, v2.RAW_ENDPOINT),
        (0.06, 0.20, 1.00, 0.10, 0.50, 0.50, v2.RAW_ENDPOINT),
        (0.06, -0.06, 1.00, 0.10, 0.06, -0.05, v2.CALIBRATED_ENDPOINT),
        (0.05, -0.06, 1.00, 0.10, 0.50, 0.50, None),
        (0.06, -0.06, 0.98, 0.10, 0.50, 0.50, None),
        (0.06, -0.06, 1.00, 0.00, 0.50, 0.50, None),
        (0.06, -0.06, 1.00, 0.10, 0.05, 0.50, None),
        (0.06, -0.06, 1.00, 0.10, 0.50, -0.06, None),
    ],
)
def test_internal_endpoint_truth_table(
    tmp_path: Path,
    raw_gain: float,
    raw_class: float,
    valid: float,
    auc: float,
    cal_gain: float,
    cal_class: float,
    expected: str | None,
) -> None:
    config = _config(tmp_path)
    decision = v2.internal_endpoint_decision(
        _raw_comparison(
            gain=raw_gain, minimum_class_delta=raw_class, valid=valid
        ),
        _calibrated_comparison(
            gain=cal_gain, minimum_class_delta=cal_class, auc_delta=auc
        ),
        config=config,
    )
    assert decision["selected_endpoint"] == expected
    assert decision["endpoint_priority"] == [v2.RAW_ENDPOINT, v2.CALIBRATED_ENDPOINT]
    assert decision["endpoint_switch_count"] == 0
    fallback = decision["calibrated_internal_fallback_gate"]
    assert fallback["calibrated_metric_gate_pass"] == fallback["deployment_gate"][
        "gate_pass"
    ]
    assert fallback["deployment_gate"]["common_checks"]["raw_validity_source"] == (
        "raw_free_generation.post_valid_label_rate"
    )


def test_endpoint_float_tolerance_and_nonfinite_rejection(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def decide(raw_gain: float, raw_class: float, valid: float, auc: float):
        return v2.internal_endpoint_decision(
            _raw_comparison(
                gain=raw_gain, minimum_class_delta=raw_class, valid=valid
            ),
            _calibrated_comparison(
                gain=0.2, minimum_class_delta=0.1, auc_delta=auc
            ),
            config=config,
        )

    assert decide(0.05 + 5e-13, -0.05, 1.0, 0.1)["selected_endpoint"] is None
    assert decide(0.05 + 2e-12, -0.05, 1.0, 0.1)["selected_endpoint"] == v2.RAW_ENDPOINT
    assert decide(0.1, -0.05 - 5e-13, 1.0, 0.1)["selected_endpoint"] == v2.RAW_ENDPOINT
    assert decide(0.1, -0.05 - 2e-12, 1.0, 0.1)["selected_endpoint"] == v2.CALIBRATED_ENDPOINT
    assert decide(0.1, -0.06, 0.99 - 5e-13, 0.1)["selected_endpoint"] == v2.CALIBRATED_ENDPOINT
    assert decide(0.1, -0.06, 0.99 - 2e-12, 0.1)["selected_endpoint"] is None
    assert decide(0.1, -0.06, 1.0, 5e-13)["selected_endpoint"] is None
    assert decide(0.1, -0.06, 1.0, 2e-12)["selected_endpoint"] == v2.CALIBRATED_ENDPOINT
    for value, field in ((float("nan"), "gain"), (float("inf"), "gain")):
        with pytest.raises(ValueError, match="must be finite"):
            decide(value, -0.06, 1.0, 0.1)
    with pytest.raises(ValueError, match="must be finite"):
        decide(0.1, -0.06, 1.0, float("inf"))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("BOOTSTRAP_SEED", 43003),
        ("BOOTSTRAP_REPLICATES", 9999),
        ("BOOTSTRAP_INTERVAL_LEVEL", 0.90),
    ],
)
def test_formal_entry_rejects_uncertainty_protocol_runtime_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: int | float,
) -> None:
    monkeypatch.setattr(v2, name, value)
    with pytest.raises(RuntimeError, match="frozen uncertainty protocol drifted"):
        v2._protocol_payload(_config(tmp_path))


class _Payload:
    @staticmethod
    def to_dict() -> dict[str, object]:
        return {"exact_budget": True, "active_atoms": 144}


class _FakeBank:
    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor(0.0))
        self.training = True

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def all_atom_parameters(self):
        return (self.parameter,)

    def active_atom_parameters(self):
        return (self.parameter,)

    def assert_exact_budget(self) -> None:
        return None

    @staticmethod
    def payload_audit() -> _Payload:
        return _Payload()

    def compact_state_dict(self):
        return {"parameter": self.parameter.detach().clone()}

    def load_compact_state_dict(self, state) -> None:
        with torch.no_grad():
            self.parameter.copy_(state["parameter"])


def _raw_audit(
    examples: tuple[OfficialExample, ...] | list[OfficialExample],
    pattern: str,
    natural_prior: Mapping[str, float],
) -> dict[str, Any]:
    by_label_seen = {label: 0 for label in v2.BINARY_LABELS}
    per_label_total = {
        label: sum(item.label == label for item in examples)
        for label in v2.BINARY_LABELS
    }
    correct_fractions = {
        "base": {"True": 0.0, "False": 1.0},
        "raw_pass": {"True": 1.0, "False": 1.0},
        "fallback": {"True": 1.0, "False": 0.5},
        "gain_fail": {"True": 0.0, "False": 1.0},
    }[pattern]
    predictions: list[str] = []
    for item in examples:
        seen = by_label_seen[item.label]
        by_label_seen[item.label] += 1
        correct_count = round(correct_fractions[item.label] * per_label_total[item.label])
        correct = seen < correct_count
        predictions.append(
            item.label
            if correct
            else ("False" if item.label == "True" else "True")
        )
    references = [item.label for item in examples]
    metrics = rescue.free_generation_classification_metrics(
        predictions,
        references,
        labels=v2.BINARY_LABELS,
        natural_label_prior=natural_prior,
    )
    return {
        **metrics,
        "example_ids": [item.example_id for item in examples],
        "predictions": predictions,
        "references": references,
        "source": "tiny-test train audit",
    }


def _split_ids(
    config: v2.TransferConfig, examples: tuple[OfficialExample, ...]
) -> tuple[set[str], set[str], set[str]]:
    outer, internal, _sources, _audit = v2._build_splits(examples, config=config)
    return (
        {item.example_id for item in internal.calibration},
        {item.example_id for item in outer.tune_audit},
        {item.example_id for item in outer.confirm_audit},
    )


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: v2.TransferConfig,
    examples: tuple[OfficialExample, ...],
    stage_behaviors: Mapping[str, tuple[str, str]],
) -> tuple[list[str], set[str]]:
    internal_ids, dev_ids, confirm_ids = _split_ids(config, examples)
    accesses: list[str] = []
    selection_task_dir = Path(config.output_root) / config.task_name

    def stage_for(ids: set[str]) -> str:
        return (
            "internal"
            if ids == internal_ids
            else "development"
            if ids == dev_ids
            else "confirm"
            if ids == confirm_ids
            else "unknown"
        )

    def fake_score(bank, tokenizer, partition, *, config, natural_prior):
        del tokenizer, config
        ids = {item.example_id for item in partition}
        stage = stage_for(ids)
        accesses.append(stage)
        if stage == "development":
            assert (selection_task_dir / "internal_endpoint_lock.json").is_file()
        if stage == "confirm":
            assert (selection_task_dir / "confirmation_claim.json").is_file()
        learned = float(bank.parameter.detach()) > 0
        if stage == "internal" and learned:
            assert (
                selection_task_dir
                / "trajectory_completed_pending_internal_eval.json"
            ).is_file()
        if not learned:
            raw_pattern, margin_pattern = "base", "flat"
        else:
            raw_pattern, margin_pattern = stage_behaviors[stage]
        raw = _raw_audit(tuple(partition), raw_pattern, natural_prior)
        if margin_pattern == "flat":
            margins = [0.0 for _item in partition]
        elif margin_pattern == "separated":
            margins = [
                1.0 if item.label == "True" else -1.0 for item in partition
            ]
        elif margin_pattern == "shifted":
            # Ranking is perfect but the internal post threshold (-1) predicts
            # both classes as True, so calibrated gain fails while AUC passes.
            margins = [2.0 if item.label == "True" else 1.0 for item in partition]
        else:  # pragma: no cover - test setup guard.
            raise AssertionError(margin_pattern)
        return raw, margins

    def fake_logical_step(
        bank, tokenizer, batch, optimizer, *, config, natural_prior
    ):
        del tokenizer, optimizer, config, natural_prior
        assert bank.training is False
        with torch.no_grad():
            bank.parameter.add_(1.0)
        return 0.25, {
            "gradient_examples": len(batch),
            "physical_microbatches": 2,
            "physical_microbatch_size": 2,
            "microbatch_example_ids": [
                [item.example_id for item in batch[:2]],
                [item.example_id for item in batch[2:]],
            ],
            "source_tokens_processed": 3 * len(batch),
            "target_tokens_processed": 2 * len(batch),
            "zero_grad_calls": 1,
            "backward_calls": 2,
            "gradient_clip_calls": 1,
            "optimizer_step_calls": 1,
        }

    monkeypatch.setattr(rescue, "_build_model", lambda config, tokenizer: _FakeBank())
    monkeypatch.setattr(rescue, "_adapter_fingerprint", lambda bank: "fixed-init")
    monkeypatch.setattr(rescue, "_release_model", lambda bank: None)
    monkeypatch.setattr(
        bap,
        "_tokenizer_protocol_fingerprint",
        lambda tokenizer: {"aggregate_sha256": "fixed-tokenizer"},
    )
    monkeypatch.setattr(v2, "_accumulate_pcsm_logical_batch", fake_logical_step)
    monkeypatch.setattr(v2, "_score_partition", fake_score)
    return accesses, confirm_ids


def _install_fast_cluster_bootstrap(
    monkeypatch: pytest.MonkeyPatch, *, replicates: int
) -> None:
    original = v2.semantic_group_cluster_uncertainty

    def fast_bootstrap(*args, **kwargs):
        kwargs["replicates"] = replicates
        return original(*args, **kwargs)

    monkeypatch.setattr(v2, "semantic_group_cluster_uncertainty", fast_bootstrap)


def test_raw_priority_lock_precedes_dev_and_calibrated_cannot_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    examples = _corpus()
    accesses, _confirm_ids = _install_fake_runtime(
        monkeypatch,
        config=config,
        examples=examples,
        stage_behaviors={
            "internal": ("raw_pass", "separated"),
            "development": ("raw_pass", "shifted"),
        },
    )
    selection = v2.run_selection(config, train_examples=examples, tokenizer=object())
    assert selection["selected"]["endpoint"] == v2.RAW_ENDPOINT
    assert selection["internal_calibration"]["endpoint_decision"][
        "raw_priority_applied"
    ] is True
    assert selection["development_tune"]["qualification_pass"] is True
    assert selection["confirm_eligible"] is True
    assert selection["development_tune"]["endpoint_switch_count"] == 0
    expected_exposures = [
        item.example_id
        for _step, batch in bap.paired_fit_batches(
            v2._build_splits(examples, config=config)[1].fit,
            batch_size=4,
            steps=config.optimizer_steps,
            seed=config.seed,
        )
        for item in batch
    ]
    ledger = selection["resource_ledger"]
    assert ledger["ordered_exposure_ids_sha256"] == bap._sha256_json(
        expected_exposures
    )
    assert ledger["gradient_example_exposures"] == 8
    assert ledger["physical_microbatches"] == 4
    assert ledger["backward_calls"] == 4
    assert ledger["gradient_clip_calls"] == 2
    assert ledger["optimizer_step_calls"] == 2
    assert ledger["zero_grad_calls"] == 2
    assert ledger["post_trajectory_cleanup_zero_grad_calls"] == 1
    assert ledger["total_optimizer_zero_grad_calls"] == 3
    assert ledger["bitwise_identical_to_v1_claimed"] is False
    trajectory_path = (
        Path(config.output_root)
        / "QQP"
        / "trajectory_completed_pending_internal_eval.json"
    )
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert trajectory["status"] == "trajectory_completed_pending_internal_eval"
    assert trajectory["post_training_internal_eval_accessed"] is False
    assert trajectory["ordered_exposure_ids_sha256"] == ledger[
        "ordered_exposure_ids_sha256"
    ]
    assert trajectory["checkpoint"]["sha256"] == selection[
        "selected_checkpoint"
    ]["sha256"]
    assert "confirm" not in accesses


def test_calibrated_fallback_locks_and_passes_selected_dev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    examples = _corpus()
    accesses, _confirm_ids = _install_fake_runtime(
        monkeypatch,
        config=config,
        examples=examples,
        stage_behaviors={
            "internal": ("fallback", "separated"),
            "development": ("fallback", "separated"),
        },
    )
    selection = v2.run_selection(config, train_examples=examples, tokenizer=object())
    decision = selection["internal_calibration"]["endpoint_decision"]
    assert selection["selected"]["endpoint"] == v2.CALIBRATED_ENDPOINT
    assert decision["raw_endpoint_gate"]["class_safety_pass"] is False
    assert decision["calibrated_internal_fallback_gate"]["fallback_gate_pass"] is True
    assert selection["development_tune"]["qualification_pass"] is True
    assert selection["confirm_eligible"] is True
    assert "confirm" not in accesses


@pytest.mark.parametrize(
    ("internal_behavior", "dev_behavior", "selected"),
    [
        (("raw_pass", "separated"), ("fallback", "separated"), v2.RAW_ENDPOINT),
        (("fallback", "separated"), ("raw_pass", "shifted"), v2.CALIBRATED_ENDPOINT),
    ],
)
def test_development_cannot_switch_to_passing_alternative_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    internal_behavior: tuple[str, str],
    dev_behavior: tuple[str, str],
    selected: str,
) -> None:
    config = _config(tmp_path)
    examples = _corpus()
    accesses, _confirm_ids = _install_fake_runtime(
        monkeypatch,
        config=config,
        examples=examples,
        stage_behaviors={
            "internal": internal_behavior,
            "development": dev_behavior,
        },
    )
    selection = v2.run_selection(config, train_examples=examples, tokenizer=object())
    assert selection["selected"]["endpoint"] == selected
    assert selection["development_tune"]["qualification_pass"] is False
    assert selection["confirm_eligible"] is False
    assert selection["development_tune"]["alternative_endpoint_cannot_rescue"] is True
    assert selection["development_tune"]["endpoint_switch_count"] == 0
    assert "confirm" not in accesses


def test_absent_raw_acquisition_cannot_open_dev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    examples = _corpus()
    accesses, _confirm_ids = _install_fake_runtime(
        monkeypatch,
        config=config,
        examples=examples,
        stage_behaviors={"internal": ("gain_fail", "separated")},
    )
    failure = v2.run_selection(config, train_examples=examples, tokenizer=object())
    assert failure["status"] == "internal_endpoint_qualification_failed"
    assert failure["selected_endpoint"] is None
    assert failure["development_tune_accessed"] is False
    assert "development" not in accesses
    assert "confirm" not in accesses


def test_one_shot_calibrated_confirmation_and_cluster_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    examples = _corpus()
    accesses, _confirm_ids = _install_fake_runtime(
        monkeypatch,
        config=config,
        examples=examples,
        stage_behaviors={
            "internal": ("fallback", "separated"),
            "development": ("fallback", "separated"),
            "confirm": ("fallback", "separated"),
        },
    )
    _install_fast_cluster_bootstrap(monkeypatch, replicates=200)
    selection = v2.run_selection(config, train_examples=examples, tokenizer=object())
    selection_path = Path(config.output_root) / "QQP" / "selection.json"
    confirmation_config = replace(
        config,
        output_root=str(tmp_path / "confirmation"),
        confirm_selection=str(selection_path),
    )
    result = v2.run_confirmation(
        confirmation_config,
        selection,
        selection_path,
        train_examples=examples,
        tokenizer=object(),
    )
    assert result["selected_endpoint"] == v2.CALIBRATED_ENDPOINT
    assert result["confirm_selected_endpoint_gate_pass"] is True
    assert result["metric_is_raw_official_free_generation"] is False
    assert "not raw free-generation" in result["positive_claim_metric"]
    assert result["endpoint_switch_count"] == 0
    uncertainty = result["semantic_group_cluster_uncertainty"]
    assert uncertainty["replicates_accepted"] == 200
    assert uncertainty["seed"] == v2.BOOTSTRAP_SEED == 43002
    assert "confirm" in accesses
    claim_path = selection_path.parent / "confirmation_claim.json"
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    assert claim["status"] == "confirmation_completed"
    result_path = Path(confirmation_config.output_root) / "QQP" / "confirmation.json"
    assert claim["confirmation_result_sha256"] == rescue._sha256_file(result_path)
    with pytest.raises(RuntimeError, match="already has a confirmation claim"):
        v2.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "confirm-again")),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )


def test_confirmation_cannot_switch_from_failed_calibrated_to_passing_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    examples = _corpus()
    behaviors: dict[str, tuple[str, str]] = {
        "internal": ("fallback", "separated"),
        "development": ("fallback", "separated"),
        # Raw passes and ranking improves, but frozen calibrated thresholds
        # produce no gain.  The raw diagnostic must not rescue confirmation.
        "confirm": ("raw_pass", "shifted"),
    }
    _accesses, _confirm_ids = _install_fake_runtime(
        monkeypatch,
        config=config,
        examples=examples,
        stage_behaviors=behaviors,
    )
    _install_fast_cluster_bootstrap(monkeypatch, replicates=100)
    selection = v2.run_selection(config, train_examples=examples, tokenizer=object())
    selection_path = Path(config.output_root) / "QQP" / "selection.json"
    result = v2.run_confirmation(
        replace(
            config,
            output_root=str(tmp_path / "confirmation-no-switch"),
            confirm_selection=str(selection_path),
        ),
        selection,
        selection_path,
        train_examples=examples,
        tokenizer=object(),
    )
    assert result["selected_endpoint"] == v2.CALIBRATED_ENDPOINT
    assert result["confirm_selected_endpoint_gate_pass"] is False
    raw_gate = v2.deployment_endpoint_gate_audit(
        v2.RAW_ENDPOINT,
        result["raw_free_generation_diagnostic"],
        result["internal_frozen_calibrated_diagnostic"],
        config=config,
    )
    assert raw_gate["gate_pass"] is True
    assert result["alternative_endpoint_cannot_rescue"] is True
    assert result["endpoint_switch_count"] == 0


def test_scoring_failure_consumes_one_shot_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    examples = _corpus()
    _accesses, _confirm_ids = _install_fake_runtime(
        monkeypatch,
        config=config,
        examples=examples,
        stage_behaviors={
            "internal": ("raw_pass", "separated"),
            "development": ("raw_pass", "separated"),
        },
    )
    selection = v2.run_selection(config, train_examples=examples, tokenizer=object())
    selection_path = Path(config.output_root) / "QQP" / "selection.json"
    claim_path = selection_path.parent / "confirmation_claim.json"

    def fail_after_claim(*args, **kwargs):
        del args, kwargs
        assert claim_path.is_file()
        raise RuntimeError("synthetic scoring failure")

    monkeypatch.setattr(v2, "_score_partition", fail_after_claim)
    confirmation_config = replace(
        config,
        output_root=str(tmp_path / "failed-confirmation"),
        confirm_selection=str(selection_path),
    )
    with pytest.raises(RuntimeError, match="synthetic scoring failure"):
        v2.run_confirmation(
            confirmation_config,
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    assert claim["status"] == "claimed_immediately_before_confirm_scoring"
    with pytest.raises(RuntimeError, match="already has a confirmation claim"):
        v2.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "retry")),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )


def test_preflight_rederives_raw_priority_and_checkpoint_sha_precedes_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    examples = _corpus()
    accesses, _confirm_ids = _install_fake_runtime(
        monkeypatch,
        config=config,
        examples=examples,
        stage_behaviors={
            "internal": ("raw_pass", "separated"),
            "development": ("raw_pass", "separated"),
            "confirm": ("raw_pass", "separated"),
        },
    )
    selection = v2.run_selection(config, train_examples=examples, tokenizer=object())
    selection_path = Path(config.output_root) / "QQP" / "selection.json"
    outer, internal, _sources, _audit = v2._build_splits(examples, config=config)
    distribution = rescue.pinned_train_label_distribution(examples, task_name="QQP")
    semantic_tamper = copy.deepcopy(selection)
    semantic_tamper["internal_calibration"]["endpoint_decision"][
        "selected_endpoint"
    ] = v2.CALIBRATED_ENDPOINT
    with pytest.raises(ValueError, match="does not follow raw priority"):
        v2._preflight_confirmation(
            semantic_tamper,
            selection_path,
            outer,
            internal,
            distribution,
            config=config,
        )

    trajectory_lock = selection_path.parent / "trajectory_completed_pending_internal_eval.json"
    original_trajectory_lock = trajectory_lock.read_bytes()
    trajectory_lock.write_bytes(original_trajectory_lock + b" ")
    trajectory_confirmation_config = replace(
        config,
        output_root=str(tmp_path / "trajectory-tamper-confirmation"),
        confirm_selection=str(selection_path),
    )
    with pytest.raises(ValueError, match="trajectory-completed lock changed"):
        v2.run_confirmation(
            trajectory_confirmation_config,
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    assert not (selection_path.parent / "confirmation_claim.json").exists()
    trajectory_lock.write_bytes(original_trajectory_lock)

    checkpoint = Path(selection["selected_checkpoint"]["path"])
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(original + b"tamper")
    confirmation_config = replace(
        config,
        output_root=str(tmp_path / "confirmation"),
        confirm_selection=str(selection_path),
    )
    with pytest.raises(ValueError, match="checkpoint binding is invalid"):
        v2.run_confirmation(
            confirmation_config,
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    assert not (selection_path.parent / "confirmation_claim.json").exists()
    assert "confirm" not in accesses
    checkpoint.write_bytes(original)


def test_cluster_bootstrap_carries_whole_paired_groups() -> None:
    examples = (
        _qqp_example(0, "True"),
        _qqp_example(1, "True"),
        _qqp_example(100, "False"),
        _qqp_example(101, "False"),
    )
    natural_prior = {"True": 0.5, "False": 0.5}
    base_raw = _raw_audit(examples, "base", natural_prior)
    post_raw = _raw_audit(examples, "raw_pass", natural_prior)
    groups = {
        examples[0].example_id: "paired-group-1",
        examples[2].example_id: "paired-group-1",
        examples[1].example_id: "paired-group-2",
        examples[3].example_id: "paired-group-2",
    }
    kwargs = dict(
        base_raw=base_raw,
        post_raw=post_raw,
        base_margins=[0.0] * 4,
        post_margins=[1.0, 1.0, -1.0, -1.0],
        natural_prior=natural_prior,
        selected_endpoint=v2.RAW_ENDPOINT,
        base_threshold=0.0,
        post_threshold=0.0,
        seed=v2.BOOTSTRAP_SEED,
        replicates=100,
    )
    first = v2.semantic_group_cluster_uncertainty(examples, groups, **kwargs)
    second = v2.semantic_group_cluster_uncertainty(examples, groups, **kwargs)
    assert first == second
    assert first["semantic_group_count"] == 2
    assert first["semantic_group_size_summary"] == {
        "minimum": 2,
        "maximum": 2,
        "mean": 2.0,
    }
    assert first["replicates_accepted"] == 100
    assert first["draws_attempted"] == 100


def test_v2_dependencies_bind_final_protocol_and_leave_v1_unchanged() -> None:
    dependencies = v2._dependency_fingerprints()
    assert dependencies["adaptive_endpoint_protocol"]["sha256"] == (
        "c0bb4f10d17fe10afb30fdb499549492a2191d8e391ed3544e8610a34b51b040"
    )
    assert dependencies["transfer_v1"]["sha256"] == v2.FROZEN_TRANSFER_V1_SHA256
    assert dependencies["adaptive_endpoint_execution_lock"]["sha256"] == (
        "1051bf4de41634714430943ab7cd9b4c54e4420f71031739b96523fa0662f42d"
    )
    assert dependencies["adaptive_endpoint_v2_runner"]["sha256"] == (
        rescue._sha256_file(Path(v2.__file__)).lower()
    )
    recipe = v2._recipe_payload()
    assert recipe["training_batch_size"] == 4
    assert recipe["evaluation_microbatch_size"] == 2
    assert recipe["maximum_label_expanded_evaluation_sequences"] == 4


def test_evaluation_microbatch_preserves_order_and_does_not_change_training_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training_config = replace(_config(tmp_path), batch_size=4)
    examples = list(_corpus()[:4])
    observed_batch_sizes: list[int] = []
    observed_orders: list[list[str]] = []

    def fake_raw(bank, tokenizer, rows, *, config, natural_prior):
        del bank, tokenizer, natural_prior
        observed_batch_sizes.append(config.batch_size)
        observed_orders.append([item.example_id for item in rows])
        return {
            "example_ids": [item.example_id for item in rows],
            "references": [item.label for item in rows],
        }

    def fake_margins(bank, tokenizer, rows, *, config):
        del bank, tokenizer
        observed_batch_sizes.append(config.batch_size)
        observed_orders.append([item.example_id for item in rows])
        return [float(index) for index, _item in enumerate(rows)]

    monkeypatch.setattr(bap, "_raw_audit", fake_raw)
    monkeypatch.setattr(bap, "collect_label_margins", fake_margins)
    raw, margins = v2._score_partition(
        object(),
        object(),
        examples,
        config=training_config,
        natural_prior={"True": 0.5, "False": 0.5},
    )
    expected_order = [item.example_id for item in examples]
    assert training_config.batch_size == 4
    assert observed_batch_sizes == [2, 2]
    assert observed_orders == [expected_order, expected_order]
    assert raw["example_ids"] == expected_order
    assert margins == [0.0, 1.0, 2.0, 3.0]


def test_logical_objective_uses_two_microbatches_but_one_clip_and_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    logical_batch = (
        _qqp_example(1, "True"),
        _qqp_example(2, "False"),
        _qqp_example(3, "True"),
        _qqp_example(4, "False"),
    )

    class Bank:
        def __init__(self) -> None:
            self.parameter = torch.nn.Parameter(torch.tensor(1.0))
            self.assert_calls = 0

        def __call__(self, **encoded):
            return SimpleNamespace(
                logits=self.parameter * encoded["input_ids"].to(torch.float32)
            )

        def active_atom_parameters(self):
            return (self.parameter,)

        def assert_exact_budget(self) -> None:
            self.assert_calls += 1

    class Optimizer:
        def __init__(self, parameter: torch.nn.Parameter) -> None:
            self.parameter = parameter
            self.zero_calls = 0
            self.step_calls = 0
            self.gradient_at_step: float | None = None

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True
            self.zero_calls += 1
            self.parameter.grad = None

        def step(self) -> None:
            self.step_calls += 1
            assert self.parameter.grad is not None
            self.gradient_at_step = float(self.parameter.grad)

    def fake_tokenize(tokenizer, rows, *, config):
        del tokenizer, config
        values = torch.tensor(
            [[float(item.source_index)] for item in rows], dtype=torch.float32
        )
        return {
            "input_ids": values,
            "attention_mask": torch.ones_like(values, dtype=torch.long),
            "labels": torch.ones_like(values, dtype=torch.long),
        }

    monkeypatch.setattr(rescue, "_tokenize", fake_tokenize)
    monkeypatch.setattr(
        bap,
        "per_example_target_token_nll",
        lambda logits, labels: logits.reshape(len(labels)),
    )
    clip_calls: list[float] = []

    def fake_clip(parameters, max_norm):
        parameters = tuple(parameters)
        assert max_norm == 1.0
        assert len(parameters) == 1 and parameters[0].grad is not None
        clip_calls.append(float(parameters[0].grad))
        return torch.tensor(abs(clip_calls[-1]))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip)
    bank = Bank()
    optimizer = Optimizer(bank.parameter)
    loss, ledger = v2._accumulate_pcsm_logical_batch(
        bank,
        object(),
        logical_batch,
        optimizer,  # type: ignore[arg-type]
        config=config,
        natural_prior={"True": 0.5, "False": 0.5},
    )
    # mean([1,2,3,4]) at parameter=1; consecutive micro means are each
    # weighted by 2/4, so accumulated objective and gradient are both 2.5.
    assert loss == pytest.approx(2.5)
    assert optimizer.zero_calls == 1
    assert optimizer.step_calls == 1
    assert optimizer.gradient_at_step == pytest.approx(2.5)
    assert clip_calls == pytest.approx([2.5])
    assert bank.assert_calls == 1
    assert ledger["microbatch_example_ids"] == [
        [logical_batch[0].example_id, logical_batch[1].example_id],
        [logical_batch[2].example_id, logical_batch[3].example_id],
    ]
    assert ledger["backward_calls"] == 2
    assert ledger["gradient_clip_calls"] == 1
    assert ledger["optimizer_step_calls"] == 1
