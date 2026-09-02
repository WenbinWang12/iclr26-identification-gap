from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.phase2i_anchored_cvar import run_acquisition_rescue as rescue
from experiments.phase2i_anchored_cvar import run_bap_lora_rescue as bap
from experiments.phase2i_anchored_cvar.order4_data import OfficialExample


def _boolqa_example(index: int, label: str, passage: str) -> OfficialExample:
    sentence = f"question: question {index}?\npassage: {passage}"
    return OfficialExample(
        task_name="BoolQA",
        category="BoolQA",
        dataset="BoolQA",
        subset="train",
        source_index=index,
        example_id=f"boolqa-{index}",
        sentence=sentence,
        label=label,
        prompt=f"BoolQA\n{sentence}\nAnswer:",
    )


def _small_boolqa_corpus(per_class: int = 8) -> tuple[OfficialExample, ...]:
    output = []
    for label_index, label in enumerate(("True", "False")):
        for local_index in range(per_class):
            index = label_index * 100 + local_index
            output.append(
                _boolqa_example(index, label, f"unique {label} passage {local_index}")
            )
    return tuple(output)


def _small_config(tmp_path: Path, *, dry_run: bool = False) -> bap.BAPConfig:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_stub = model_dir / "config.json"
    if not model_stub.exists():
        model_stub.write_text('{"model_type":"t5"}\n', encoding="utf-8")
    return bap.BAPConfig(
        data_root=str(tmp_path),
        model_path=str(model_dir),
        output_root=str(tmp_path / "out"),
        task_name="BoolQA",
        learning_rate=3e-4,
        fit_per_class=2,
        calibration_per_class=1,
        tune_audit_per_class=1,
        confirm_audit_per_class=1,
        checkpoint_steps=(1, 2),
        seed=17,
        batch_size=2,
        max_source_length=32,
        max_target_length=8,
        truncation_mode="field_aware",
        tiny_random_model=True,
        dry_run=dry_run,
    )


def test_bap_loss_separates_translation_invariant_rank_from_shift_anchor() -> None:
    base = torch.tensor([1.0, -1.0, 2.0, -2.0])
    current = base.clone().requires_grad_(True)
    labels = ("True", "False", "True", "False")
    original = bap.bap_loss_components(
        current,
        base,
        labels,
        base_threshold=0.0,
        margin_scale=1.0,
        pair_temperature=1.0,
        safe_margin_cap=0.5,
    )
    translated = bap.bap_loss_components(
        current + 0.75,
        base,
        labels,
        base_threshold=0.0,
        margin_scale=1.0,
        pair_temperature=1.0,
        safe_margin_cap=0.5,
    )
    assert torch.allclose(original["pairwise_rank"], translated["pairwise_rank"])
    assert original["common_shift_anchor"].item() == pytest.approx(0.0)
    assert translated["common_shift_anchor"].item() == pytest.approx(0.75**2)
    assert original["worst_class_base_correct_hinge"].item() == pytest.approx(0.0)
    total = bap.bap_total_loss(
        translated, common_shift_weight=1.0, safe_hinge_weight=2.0
    )
    total.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()


def test_worst_class_hinge_activates_when_a_base_correct_class_crosses() -> None:
    base = torch.tensor([0.8, -0.8, 0.9, -0.9])
    # Both True rows remain safe; both False rows cross to the True side.
    current = torch.tensor([0.8, 0.4, 0.9, 0.5], requires_grad=True)
    components = bap.bap_loss_components(
        current,
        base,
        ("True", "False", "True", "False"),
        base_threshold=0.0,
        margin_scale=1.0,
        pair_temperature=1.0,
        safe_margin_cap=0.5,
    )
    assert components["worst_class_base_correct_hinge"].item() > 0
    assert components["base_correct_count"].item() == 4
    equality = bap.bap_loss_components(
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        ("True", "False"),
        base_threshold=0.0,
        margin_scale=1.0,
        pair_temperature=1.0,
        safe_margin_cap=0.5,
    )
    assert equality["base_correct_count"].item() == 2


def test_ranknet_uses_all_within_batch_true_false_pairs() -> None:
    current = torch.tensor([2.0, -1.0, 1.0, 0.5])
    components = bap.bap_loss_components(
        current,
        current.clone(),
        ("True", "False", "True", "False"),
        base_threshold=0.0,
        margin_scale=1.0,
        pair_temperature=1.0,
        safe_margin_cap=0.5,
    )
    expected_differences = torch.tensor([[3.0, 1.5], [2.0, 0.5]])
    expected = torch.nn.functional.softplus(-expected_differences).mean()
    assert torch.allclose(components["pairwise_rank"], expected)


def test_calibrated_comparison_uses_calibrated_base_as_denominator() -> None:
    references = ["True"] * 10 + ["False"] * 10
    base_margins = [0.0] * 20
    post_margins = [1.0] * 10 + [-1.0] * 10
    comparison = bap.calibrated_training_comparison(
        base_margins,
        post_margins,
        references,
        natural_prior={"True": 0.6, "False": 0.4},
        max_class_drop=0.05,
    )
    assert comparison["base_same_calibration_control"]["metrics"][
        "natural_prior_em"
    ] == pytest.approx(0.6)
    assert comparison["bap_calibrated"]["metrics"]["natural_prior_em"] == 1.0
    assert comparison["base_same_calibration_control"][
        "constraint_reference_threshold"
    ] is None
    assert comparison["bap_calibrated"]["constraint_reference_threshold"] is None
    assert comparison["training_contribution_natural_prior"] == pytest.approx(0.4)
    assert comparison["margin_diagnostics"]["base"]["roc_auc"] == pytest.approx(0.5)
    assert comparison["margin_diagnostics"]["post"]["roc_auc"] == pytest.approx(1.0)
    assert comparison["calibration_gain_is_not_training_contribution"]
    assert bap.qualification_gate(
        comparison, min_gain=0.05, max_class_drop=0.05
    )


def test_gain_decomposition_keeps_raw_and_calibration_gain_out_of_training() -> None:
    comparison = {
        "base_same_calibration_control": {
            "metrics": {"natural_prior_em": 0.70, "balanced_em": 0.68}
        },
        "bap_calibrated": {
            "metrics": {"natural_prior_em": 0.78, "balanced_em": 0.76}
        },
        "training_contribution_natural_prior": 0.08,
        "training_contribution_balanced": 0.08,
        "training_per_class_recall_delta": {"True": 0.02, "False": 0.17},
        "training_min_per_class_delta": 0.02,
    }
    result = bap._raw_gain_decomposition(
        {"natural_prior_em": 0.60, "balanced_em": 0.59},
        {"natural_prior_em": 0.69, "balanced_em": 0.67},
        comparison,
    )
    assert result["raw_free_generation"]["natural_prior_gain"] == pytest.approx(
        0.09
    )
    assert result["calibration_only_gain"]["base_natural_prior"] == pytest.approx(
        0.10
    )
    assert result["calibration_only_gain"]["post_natural_prior"] == pytest.approx(
        0.09
    )
    assert result["bap_training_contribution"]["natural_prior"] == pytest.approx(
        0.08
    )
    assert result["calibration_only_gain"][
        "not_counted_as_training_contribution"
    ]


def test_group_atomic_internal_split_is_exact_and_deterministic(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    examples = [
        _boolqa_example(0, "True", "shared passage"),
        _boolqa_example(1, "False", "shared passage"),
        _boolqa_example(2, "True", "true singleton one"),
        _boolqa_example(3, "True", "true singleton two"),
        _boolqa_example(4, "False", "false singleton one"),
        _boolqa_example(5, "False", "false singleton two"),
    ]
    keys = rescue._semantic_group_keys(examples, task_name="BoolQA")
    split = rescue.RescueSplit(
        task_name="BoolQA",
        update_by_label={
            "True": tuple(item for item in examples if item.label == "True"),
            "False": tuple(item for item in examples if item.label == "False"),
        },
        tune_audit=(),
        confirm_audit=(),
        excluded_count=0,
        split_unit="semantic_group",
        group_key_version=rescue.SEMANTIC_GROUP_KEY_VERSION,
        group_key_by_example_id=keys,
        selection_algorithm_version=rescue.SEMANTIC_SELECTION_ALGORITHM_VERSION,
    )
    first = bap.build_bap_internal_split(split, config=config)
    second = bap.build_bap_internal_split(split, config=config)
    assert first == second
    first.assert_valid(config=config)
    shared = {"boolqa-0", "boolqa-1"}
    fit_ids = {item.example_id for item in first.fit}
    calibration_ids = {item.example_id for item in first.calibration}
    assert not (shared & fit_ids and shared & calibration_ids)
    manifest = bap.bap_internal_split_manifest(first)
    assert manifest["fit"]["per_class"] == {"True": 2, "False": 2}
    assert manifest["internal_calibration"]["per_class"] == {
        "True": 1,
        "False": 1,
    }
    assert manifest["semantic_group_overlap"] == 0


def test_paired_schedule_is_balanced_deterministic_and_cycles_exactly() -> None:
    examples = _small_boolqa_corpus(per_class=4)
    first = list(
        bap.paired_fit_batches(examples, batch_size=4, steps=4, seed=11)
    )
    second = list(
        bap.paired_fit_batches(tuple(reversed(examples)), batch_size=4, steps=4, seed=11)
    )
    # Input order is part of the fit manifest, so repeatability is asserted for
    # an identical persisted partition rather than arbitrary caller order.
    repeated = list(
        bap.paired_fit_batches(examples, batch_size=4, steps=4, seed=11)
    )
    assert [[item.example_id for item in batch] for _, batch in first] == [
        [item.example_id for item in batch] for _, batch in repeated
    ]
    assert [[item.label for item in batch] for _, batch in first] == [
        ["True", "False", "True", "False"]
    ] * 4
    assert len(second) == len(first)
    # Two steps exhaust one exact cycle of four rows per class.
    first_cycle = [item for _, batch in first[:2] for item in batch]
    assert {item.example_id for item in first_cycle} == {
        item.example_id for item in examples
    }


def test_dry_run_builds_all_ledgers_without_model_or_confirm_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config(tmp_path, dry_run=True)
    examples = _small_boolqa_corpus()

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not construct a model")

    monkeypatch.setattr(rescue, "_build_model", forbidden)
    result = bap.run_selection(config, train_examples=examples)
    assert result["status"] == "dry_run_complete"
    assert not result["model_or_tokenizer_constructed"]
    assert not result["tune_audit_accessed"]
    assert not result["confirm_audit_accessed"]
    assert result["internal_split"]["fit"]["per_class"] == {
        "True": 2,
        "False": 2,
    }
    assert (Path(config.output_root) / "DRY_RUN_COMPLETE").is_file()
    persisted = json.loads(
        (Path(config.output_root) / "dry_run.json").read_text(encoding="utf-8")
    )
    assert persisted["gradient_steps"] == 0


def test_config_rejects_non_pairable_or_non_monotone_step_protocol(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="even batch_size"):
        bap.BAPConfig(
            data_root=str(tmp_path),
            model_path=str(tmp_path),
            output_root=str(tmp_path / "bad"),
            batch_size=3,
        )
    with pytest.raises(ValueError, match="checkpoint_steps"):
        bap.BAPConfig(
            data_root=str(tmp_path),
            model_path=str(tmp_path),
            output_root=str(tmp_path / "bad2"),
            checkpoint_steps=(192, 96),
        )
    with pytest.raises(ValueError, match="must not overlap locked data_root"):
        bap.BAPConfig(
            data_root=str(tmp_path / "data"),
            model_path=str(tmp_path / "model"),
            output_root=str(tmp_path / "data" / "run"),
        )


class _TinyTensorTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self) -> int:
        return 512

    @staticmethod
    def _ids(value: str, *, max_length: int | None) -> list[int]:
        ids = [2 + (ord(character) % 500) for character in value]
        ids.append(1)
        if max_length is not None:
            ids = ids[:max_length]
            if ids:
                ids[-1] = 1
        return ids

    def __call__(
        self,
        text=None,
        *,
        text_target=None,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        add_special_tokens=True,
        **kwargs,
    ):
        del kwargs, add_special_tokens
        values = text_target if text_target is not None else text
        if isinstance(values, str):
            return {
                "input_ids": self._ids(
                    values, max_length=max_length if truncation else None
                )
            }
        rows = [
            self._ids(value, max_length=max_length if truncation else None)
            for value in values
        ]
        width = max(len(row) for row in rows) if padding else None
        padded = [row + [0] * (width - len(row)) for row in rows]
        masks = [[int(token != 0) for token in row] for row in padded]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": masks}


def test_tiny_random_t5_has_differentiable_bap_margin_and_exact_budget(
    tmp_path: Path,
) -> None:
    config = bap.BAPConfig(
        data_root=str(tmp_path),
        model_path=str(tmp_path),
        output_root=str(tmp_path / "tiny"),
        fit_per_class=2,
        calibration_per_class=1,
        tune_audit_per_class=1,
        confirm_audit_per_class=1,
        checkpoint_steps=(1,),
        batch_size=2,
        max_source_length=24,
        max_target_length=8,
        truncation_mode="official_right",
        tiny_random_model=True,
    )
    tokenizer = _TinyTensorTokenizer()
    bank = rescue._build_model(bap._model_config(config), tokenizer)
    bank.eval()  # BAP enables autograd in eval mode to disable dropout.
    batch = (
        _boolqa_example(0, "True", "short true passage"),
        _boolqa_example(1, "False", "short false passage"),
    )
    margins, ledger = bap._label_margin_batch(
        bank, tokenizer, batch, config=config
    )
    repeated_margins, _ = bap._label_margin_batch(
        bank, tokenizer, batch, config=config
    )
    assert margins.shape == (2,)
    assert margins.requires_grad
    assert torch.allclose(margins, repeated_margins)
    assert ledger["gradient_examples"] == 2
    assert ledger["scored_label_sequences"] == 4
    components = bap.bap_loss_components(
        margins,
        margins.detach().clone(),
        ("True", "False"),
        base_threshold=0.0,
        margin_scale=1.0,
        pair_temperature=1.0,
        safe_margin_cap=0.5,
    )
    loss = bap.bap_total_loss(
        components, common_shift_weight=1.0, safe_hinge_weight=2.0
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in bank.active_atom_parameters())
    bank.assert_exact_budget()
    rescue._release_model(bank)


def test_fake_end_to_end_selection_then_one_shot_confirmation_keeps_confirm_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_small_config(tmp_path), run_pointwise_control=True)
    examples = _small_boolqa_corpus()
    outer, _sources = bap._build_outer_split(examples, config=config)
    confirm_ids = {item.example_id for item in outer.confirm_audit}
    accessed_ids: list[set[str]] = []
    tokenizer_protocol_hash = {"value": "fixed-tokenizer"}
    execution_environment_drift = {"enabled": False}
    threshold_fit_sizes: list[int] = []
    original_threshold_fitter = bap.fit_margin_threshold
    original_execution_fingerprint = bap._execution_environment_fingerprint

    class FakePayload:
        @staticmethod
        def to_dict():
            return {"exact_budget": True, "active_atoms": 144}

    class FakeBank:
        def __init__(self) -> None:
            self.parameter = torch.nn.Parameter(torch.tensor(0.0))
            self.training = True

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def eval(self):
            return self.train(False)

        def all_atom_parameters(self):
            return (self.parameter,)

        def active_atom_parameters(self):
            return (self.parameter,)

        def assert_exact_budget(self):
            return None

        @staticmethod
        def payload_audit():
            return FakePayload()

        def compact_state_dict(self):
            return {"parameter": self.parameter.detach().clone()}

        def load_compact_state_dict(self, state):
            with torch.no_grad():
                self.parameter.copy_(state["parameter"])

    def fake_margin_batch(bank, tokenizer, batch, *, config):
        del tokenizer, config
        signs = torch.tensor(
            [1.0 if item.label == "True" else -1.0 for item in batch]
        )
        margins = signs * bank.parameter
        return margins, {
            "gradient_examples": len(batch),
            "scored_label_sequences": 2 * len(batch),
            "source_tokens_processed": 3 * len(batch),
            "target_tokens_processed": 4 * len(batch),
        }

    def fake_collect(bank, tokenizer, audit_examples, *, config):
        del tokenizer, config
        ids = {item.example_id for item in audit_examples}
        accessed_ids.append(ids)
        value = float(bank.parameter.detach())
        return [value if item.label == "True" else -value for item in audit_examples]

    def fake_raw(bank, tokenizer, audit_examples, *, config, natural_prior):
        del tokenizer, config
        ids = {item.example_id for item in audit_examples}
        accessed_ids.append(ids)
        learned = float(bank.parameter.detach()) > 0
        score = 1.0 if learned else float(natural_prior["True"])
        return {
            "natural_prior_em": score,
            "balanced_em": 1.0 if learned else 0.5,
            "per_class_recall": (
                {"True": 1.0, "False": 1.0}
                if learned
                else {"True": 1.0, "False": 0.0}
            ),
            "free_generation_valid_label_rate": 1.0,
        }

    def fake_pointwise_step(
        bank, tokenizer, batch, optimizer, *, config, natural_prior
    ):
        del tokenizer, optimizer, config, natural_prior
        with torch.no_grad():
            bank.parameter.add_(1.0)
        return 0.25, {
            "gradient_examples": len(batch),
            "source_tokens_processed": 3 * len(batch),
            "target_tokens_processed": 2 * len(batch),
        }

    def recording_threshold_fitter(margins, references, **kwargs):
        threshold_fit_sizes.append(len(margins))
        return original_threshold_fitter(margins, references, **kwargs)

    def controlled_execution_fingerprint(device):
        payload = original_execution_fingerprint(device)
        if execution_environment_drift["enabled"]:
            payload = {**payload, "aggregate_sha256": "drifted-environment"}
        return payload

    monkeypatch.setattr(rescue, "_build_model", lambda config, tokenizer: FakeBank())
    monkeypatch.setattr(rescue, "_adapter_fingerprint", lambda bank: "fixed-init")
    monkeypatch.setattr(rescue, "_release_model", lambda bank: None)
    monkeypatch.setattr(
        bap, "_execution_environment_fingerprint", controlled_execution_fingerprint
    )
    monkeypatch.setattr(
        bap,
        "_tokenizer_protocol_fingerprint",
        lambda tokenizer: {
            "aggregate_sha256": tokenizer_protocol_hash["value"]
        },
    )
    monkeypatch.setattr(bap, "_label_margin_batch", fake_margin_batch)
    monkeypatch.setattr(bap, "collect_label_margins", fake_collect)
    monkeypatch.setattr(bap, "_raw_audit", fake_raw)
    monkeypatch.setattr(bap, "fit_margin_threshold", recording_threshold_fitter)
    monkeypatch.setattr(
        bap, "_pointwise_sequence_prior_step", fake_pointwise_step
    )

    selection = bap.run_selection(
        config, train_examples=examples, tokenizer=object()
    )
    assert selection["selected"]["step"] == 1
    # The boundary used by the gradient hinge comes from fit (4 rows), not the
    # held-out internal calibration partition (2 rows).
    assert threshold_fit_sizes[0] == 2 * config.fit_per_class
    assert selection["resource_ledger"]["anchor_scoring_mode_matched"] is True
    assert selection["confirm_eligible"]
    assert selection["development_tune"]["did_not_select_checkpoint_or_threshold"]
    assert all(not (ids & confirm_ids) for ids in accessed_ids)
    selection_path = Path(config.output_root) / "BoolQA" / "selection.json"
    assert selection_path.is_file()
    assert (selection_path.parent / "selection_lock.json").is_file()
    protocol_lock = json.loads(
        (selection_path.parent / "protocol_lock.json").read_text(encoding="utf-8")
    )
    assert protocol_lock["external_preregistration_claimed"] is False
    assert protocol_lock["pilot_inventory"][
        "run_local_coefficient_candidate_count"
    ] == 1
    control = json.loads(
        (selection_path.parent / "pointwise_control" / "control.json").read_text(
            encoding="utf-8"
        )
    )
    assert control["selection_eligible"] is False
    assert control["exact_ordered_exposure_match"] is True
    assert (
        control["ordered_exposure_ids_sha256"]
        == selection["resource_ledger"]["ordered_exposure_ids_sha256"]
    )
    assert control["matched_bap_initialization"] is True

    confirmation_config = replace(
        config,
        output_root=str(tmp_path / "confirm"),
        confirm_selection="locked",
    )
    with pytest.raises(ValueError, match="differs from locked"):
        bap._config_from_selection(
            selection, output_root=str(tmp_path / "wrong-device"), device="cuda"
        )
    accesses_before_preflight_attacks = list(accessed_ids)
    tampered_selection = json.loads(json.dumps(selection))
    tampered_selection["selected"]["bap_threshold"] += 1.0
    with pytest.raises(ValueError, match="in-memory selection differs"):
        bap.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "memory-tamper")),
            tampered_selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    with pytest.raises(ValueError, match="scientific config differs"):
        bap.run_confirmation(
            replace(
                confirmation_config,
                output_root=str(tmp_path / "config-tamper"),
                max_source_length=confirmation_config.max_source_length + 1,
            ),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    tampered_examples = (
        replace(examples[0], prompt=examples[0].prompt + " tampered"),
        *examples[1:],
    )
    with pytest.raises(ValueError, match="train corpus content changed"):
        bap.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "corpus-tamper")),
            selection,
            selection_path,
            train_examples=tampered_examples,
            tokenizer=object(),
        )
    execution_environment_drift["enabled"] = True
    with pytest.raises(ValueError, match="execution environment drifted"):
        bap.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "env-tamper")),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    execution_environment_drift["enabled"] = False
    assert accessed_ids == accesses_before_preflight_attacks
    assert not (selection_path.parent / "confirmation_claim.json").exists()

    protocol_lock_path = selection_path.parent / "protocol_lock.json"
    original_protocol_lock = protocol_lock_path.read_text(encoding="utf-8")
    protocol_lock_path.write_text(original_protocol_lock + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="protocol lock path or hash changed"):
        bap.run_confirmation(
            replace(
                confirmation_config, output_root=str(tmp_path / "protocol-tamper")
            ),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    assert accessed_ids == accesses_before_preflight_attacks
    assert not (selection_path.parent / "confirmation_claim.json").exists()
    protocol_lock_path.write_text(original_protocol_lock, encoding="utf-8")

    selected_checkpoint = Path(selection["selected_checkpoint"]["path"])
    original_checkpoint = selected_checkpoint.read_bytes()
    selected_checkpoint.write_bytes(original_checkpoint + b"tamper")
    with pytest.raises(ValueError, match="selected checkpoint SHA-256 mismatch"):
        bap.run_confirmation(
            replace(
                confirmation_config, output_root=str(tmp_path / "checkpoint-tamper")
            ),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    assert accessed_ids == accesses_before_preflight_attacks
    assert not (selection_path.parent / "confirmation_claim.json").exists()
    selected_checkpoint.write_bytes(original_checkpoint)

    control_path = selection_path.parent / "pointwise_control" / "control.json"
    original_control_text = control_path.read_text(encoding="utf-8")
    tampered_control = json.loads(original_control_text)
    tampered_control["ordered_exposure_ids_sha256"] = "0" * 64
    control_path.write_text(json.dumps(tampered_control), encoding="utf-8")
    accesses_before_tamper_attempt = list(accessed_ids)
    with pytest.raises(ValueError, match="bind the pointwise control"):
        bap.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "tamper-confirm")),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    assert accessed_ids == accesses_before_tamper_attempt
    assert not (selection_path.parent / "confirmation_claim.json").exists()
    control_path.write_text(original_control_text, encoding="utf-8")

    model_stub = Path(config.model_path) / "config.json"
    original_model_text = model_stub.read_text(encoding="utf-8")
    model_stub.write_text('{"model_type":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        bap.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "model-tamper")),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    assert accessed_ids == accesses_before_tamper_attempt
    assert not (selection_path.parent / "confirmation_claim.json").exists()
    model_stub.write_text(original_model_text, encoding="utf-8")

    tokenizer_protocol_hash["value"] = "changed-tokenizer"
    with pytest.raises(ValueError, match="tokenizer protocol changed"):
        bap.run_confirmation(
            replace(
                confirmation_config, output_root=str(tmp_path / "tokenizer-tamper")
            ),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    assert accessed_ids == accesses_before_tamper_attempt
    assert not (selection_path.parent / "confirmation_claim.json").exists()
    tokenizer_protocol_hash["value"] = "fixed-tokenizer"

    result = bap.run_confirmation(
        confirmation_config,
        selection,
        selection_path,
        train_examples=examples,
        tokenizer=object(),
    )
    assert result["confirm_gate_pass"]
    assert result["comparison"]["training_contribution_natural_prior"] > 0.05
    assert any(ids == confirm_ids for ids in accessed_ids)
    assert result["calibration_gain_counted_as_training_contribution"] is False
    control_confirm = result["pointwise_control_confirmation"]
    assert control_confirm["selection_eligible"] is False
    assert control_confirm["exact_resource_match"]["same_initial_adapter"] is True
    assert (
        control_confirm["exact_resource_match"]["ordered_exposure_ids_sha256"]
        == selection["resource_ledger"]["ordered_exposure_ids_sha256"]
    )

    with pytest.raises(RuntimeError, match="already has a confirmation claim"):
        bap.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "confirm2")),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
