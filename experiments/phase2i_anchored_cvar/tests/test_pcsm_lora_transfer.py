from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.phase2i_anchored_cvar import run_acquisition_rescue as rescue
from experiments.phase2i_anchored_cvar import run_bap_lora_rescue as bap
from experiments.phase2i_anchored_cvar import run_pcsm_lora_rescue as pcsm
from experiments.phase2i_anchored_cvar import run_pcsm_lora_transfer as transfer
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
        example_id=f"qqp-{index}",
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


def _config(tmp_path: Path) -> transfer.TransferConfig:
    model_path = tmp_path / "model"
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "config.json").write_text(
        '{"model_type":"t5"}\n', encoding="utf-8"
    )
    return transfer.TransferConfig(
        data_root=str(tmp_path),
        model_path=str(model_path),
        output_root=str(tmp_path / "selection"),
        task_name="QQP",
        learning_rate=3e-4,
        fit_per_class=2,
        calibration_per_class=1,
        tune_audit_per_class=1,
        confirm_audit_per_class=1,
        optimizer_steps=2,
        seed=17,
        batch_size=2,
        max_source_length=32,
        max_target_length=8,
        tiny_random_model=True,
    )


def _raw(
    *, natural: float, true_recall: float, false_recall: float, valid: float = 1.0
) -> dict[str, object]:
    return {
        "natural_prior_em": natural,
        "balanced_em": (true_recall + false_recall) / 2,
        "per_class_recall": {"True": true_recall, "False": false_recall},
        "worst_class_recall": min(true_recall, false_recall),
        "free_generation_valid_label_rate": valid,
    }


def test_transfer_recipe_is_exact_and_has_no_target_recipe_hparam_checkpoint_tuning(
    tmp_path: Path,
) -> None:
    source = tmp_path / "prior-split.json"
    source.write_text("{}", encoding="utf-8")
    config = transfer.TransferConfig(
        data_root=str(tmp_path / "data"),
        model_path=str(tmp_path / "model"),
        output_root=str(tmp_path / "out"),
        task_name="QQP",
        fresh_confirm_against_manifests=(str(source),),
    )
    recipe = transfer._recipe_payload()
    protocol = transfer._protocol_payload(config)
    assert recipe["origin"].startswith("BoolQA PCSM-LoRA v1")
    assert recipe["target_task_recipe_hparam_checkpoint_tuning"] is False
    assert recipe["target_task_gradient_adaptation"] is True
    assert recipe["target_task_internal_threshold_calibration"] is True
    assert recipe["learning_rate"] == 3e-4
    assert recipe["eligible_checkpoint_steps"] == [384]
    assert recipe["max_source_length"] == 512
    assert recipe["max_target_length"] == 50
    assert recipe["truncation_mode"] == "field_aware"
    assert protocol["selection_eligible_arm_count"] == 1
    assert protocol["target_task_recipe_hparam_checkpoint_tuning"] is False
    assert protocol["target_task_gradient_adaptation"] is True
    assert protocol["target_task_internal_threshold_calibration"] is True
    assert protocol["freshness_policy"]["hardcoded_unreviewed_target_paths"] is False
    assert transfer._model_config(config).loss_objectives == (pcsm.OBJECTIVE,)
    dependencies = transfer._dependency_fingerprints()
    assert dependencies["pcsm_v1"]["sha256"] == transfer.FROZEN_PCSM_SHA256
    multirc = replace(
        config, task_name="MultiRC", output_root=str(tmp_path / "multirc-out")
    )
    assert multirc.task_name == "MultiRC"
    tiny_config = _config(tmp_path / "tiny-split")
    outer, _internal, _sources, _audit = transfer._build_splits(
        _corpus(), config=tiny_config
    )
    source_sha = rescue._sha256_file(source)
    freshness_audit = transfer._validate_freshness_sources(
        (
            {
                "path": str(source.resolve()),
                "sha256": source_sha,
                "task": "QQP",
            },
        ),
        outer,
        config=config,
    )
    assert freshness_audit["semantic_group_strict"] is True
    assert freshness_audit["source_file_bindings"][0]["bytes"] == 2
    with pytest.raises(ValueError, match="frozen; drifted fields"):
        replace(config, learning_rate=1e-3)
    with pytest.raises(ValueError, match="at least one freshness"):
        replace(config, fresh_confirm_against_manifests=())
    with pytest.raises(ValueError, match="must be one of"):
        transfer.TransferConfig(
            data_root=str(tmp_path / "data"),
            model_path=str(tmp_path / "model"),
            output_root=str(tmp_path / "other"),
            task_name="BoolQA",
            fresh_confirm_against_manifests=(str(source),),
        )


def test_optional_task_registry_binds_recipe_and_cli_freshness_sha(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.json"
    source_path = tmp_path / "canonical-qqp-split.json"
    source_sha = "a" * 64
    payload = {
        "format": transfer.TASK_PROTOCOL_REGISTRY_FORMAT,
        "tasks": {
            "QQP": {
                "source_recipe_sha256": bap._sha256_json(
                    transfer._recipe_payload()
                ),
                "semantic_group_required": True,
                "freshness_source_sha256s": [source_sha],
            }
        },
    }
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    config = transfer.TransferConfig(
        data_root=str(tmp_path / "data"),
        model_path=str(tmp_path / "model"),
        output_root=str(tmp_path / "out"),
        task_name="QQP",
        fresh_confirm_against_manifests=(str(source_path),),
        task_protocol_registry=str(registry_path),
    )
    sources = (
        {
            "path": str(source_path.resolve()),
            "sha256": source_sha,
            "task": "QQP",
        },
    )
    binding = transfer._load_task_protocol_registry(config, sources)
    assert binding["status"] == "verified"
    assert binding["sha256"] == rescue._sha256_file(registry_path)
    changed = ({**sources[0], "sha256": "b" * 64},)
    with pytest.raises(ValueError, match="does not match CLI sources"):
        transfer._load_task_protocol_registry(config, changed)


class _TinyTensorTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self) -> int:
        return 512

    @staticmethod
    def _ids(value: str, *, max_length: int | None) -> list[int]:
        values = [2 + (ord(character) % 500) for character in value] + [1]
        if max_length is not None:
            values = values[:max_length]
            if values:
                values[-1] = 1
        return values

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


def test_tiny_random_transfer_step_uses_frozen_pcsm_objective(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), truncation_mode="official_right")
    tokenizer = _TinyTensorTokenizer()
    bank = rescue._build_model(transfer._model_config(config), tokenizer)
    bank.eval()
    before = rescue._adapter_fingerprint(bank)
    optimizer = torch.optim.AdamW(
        tuple(bank.all_atom_parameters()), lr=config.learning_rate, weight_decay=0.0
    )
    loss, ledger = bap._pointwise_sequence_prior_step(
        bank,
        tokenizer,
        (_qqp_example(0, "True"), _qqp_example(100, "False")),
        optimizer,
        config=transfer._as_bap_config(config),
        natural_prior={"True": 0.4, "False": 0.6},
    )
    assert bank.training is False
    assert torch.isfinite(torch.tensor(loss))
    assert ledger["gradient_examples"] == 2
    assert rescue._adapter_fingerprint(bank) != before
    bank.assert_exact_budget()
    rescue._release_model(bank)


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


def _split_ids(
    config: transfer.TransferConfig, examples: tuple[OfficialExample, ...]
) -> tuple[set[str], set[str], set[str]]:
    outer, internal, _sources, _audit = transfer._build_splits(
        examples, config=config
    )
    return (
        {item.example_id for item in internal.calibration},
        {item.example_id for item in outer.tune_audit},
        {item.example_id for item in outer.confirm_audit},
    )


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    internal_ids: set[str],
    dev_ids: set[str],
    confirm_ids: set[str],
    fail_stage: str | None,
    claim_path: Path | None = None,
    internal_lock_path: Path | None = None,
) -> list[set[str]]:
    accesses: list[set[str]] = []

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

    def fake_raw(bank, tokenizer, examples, *, config, natural_prior):
        del tokenizer, config, natural_prior
        ids = {item.example_id for item in examples}
        accesses.append(ids)
        stage = stage_for(ids)
        if stage == "confirm":
            assert claim_path is not None and claim_path.is_file()
        if stage == "development":
            assert internal_lock_path is not None and internal_lock_path.is_file()
        learned = float(bank.parameter.detach()) > 0
        if not learned:
            return _raw(natural=0.5, true_recall=1.0, false_recall=0.0)
        if fail_stage == stage:
            return _raw(natural=0.55, true_recall=1.0, false_recall=0.1)
        return _raw(natural=1.0, true_recall=1.0, false_recall=1.0)

    def fake_margins(bank, tokenizer, examples, *, config):
        del tokenizer, config
        ids = {item.example_id for item in examples}
        accesses.append(ids)
        learned = float(bank.parameter.detach()) > 0
        if learned and fail_stage == f"{stage_for(ids)}_secondary":
            return [0.0 for _item in examples]
        return [
            (1.0 if item.label == "True" else -1.0) if learned else 0.0
            for item in examples
        ]

    def fake_step(bank, tokenizer, batch, optimizer, *, config, natural_prior):
        del tokenizer, optimizer, config, natural_prior
        assert bank.training is False
        with torch.no_grad():
            bank.parameter.add_(1.0)
        return 0.25, {
            "gradient_examples": len(batch),
            "source_tokens_processed": 3 * len(batch),
            "target_tokens_processed": 2 * len(batch),
        }

    monkeypatch.setattr(rescue, "_build_model", lambda config, tokenizer: _FakeBank())
    monkeypatch.setattr(rescue, "_adapter_fingerprint", lambda bank: "fixed-init")
    monkeypatch.setattr(rescue, "_release_model", lambda bank: None)
    monkeypatch.setattr(
        bap,
        "_tokenizer_protocol_fingerprint",
        lambda tokenizer: {"aggregate_sha256": "fixed-tokenizer"},
    )
    monkeypatch.setattr(bap, "_raw_audit", fake_raw)
    monkeypatch.setattr(bap, "collect_label_margins", fake_margins)
    monkeypatch.setattr(bap, "_pointwise_sequence_prior_step", fake_step)
    return accesses


@pytest.mark.parametrize("fail_stage", ["internal", "internal_secondary"])
def test_target_internal_failure_never_accesses_development_or_confirm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_stage: str
) -> None:
    config = _config(tmp_path)
    examples = _corpus()
    internal_ids, dev_ids, confirm_ids = _split_ids(config, examples)
    accesses = _install_fake_runtime(
        monkeypatch,
        internal_ids=internal_ids,
        dev_ids=dev_ids,
        confirm_ids=confirm_ids,
        fail_stage=fail_stage,
        internal_lock_path=(
            Path(config.output_root) / "QQP" / "internal_qualification_lock.json"
        ),
    )
    result = transfer.run_selection(
        config, train_examples=examples, tokenizer=object()
    )
    assert result["status"] == "internal_qualification_failed"
    assert result["target_task_recipe_hparam_checkpoint_tuning"] is False
    assert result["target_task_gradient_adaptation"] is True
    assert result["target_task_internal_threshold_calibration"] is True
    assert result["development_tune_accessed"] is False
    assert all(not (ids & dev_ids) for ids in accesses)
    assert all(not (ids & confirm_ids) for ids in accesses)


@pytest.mark.parametrize("fail_stage", ["development", "development_secondary"])
def test_target_development_failure_never_opens_confirm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_stage: str
) -> None:
    config = _config(tmp_path)
    examples = _corpus()
    internal_ids, dev_ids, confirm_ids = _split_ids(config, examples)
    accesses = _install_fake_runtime(
        monkeypatch,
        internal_ids=internal_ids,
        dev_ids=dev_ids,
        confirm_ids=confirm_ids,
        fail_stage=fail_stage,
        internal_lock_path=(
            Path(config.output_root) / "QQP" / "internal_qualification_lock.json"
        ),
    )
    selection = transfer.run_selection(
        config, train_examples=examples, tokenizer=object()
    )
    assert selection["status"] == "selection_locked"
    assert selection["confirm_eligible"] is False
    assert selection["development_tune"]["qualification_pass"] is False
    if fail_stage.endswith("secondary"):
        assert selection["development_tune"]["primary_pass"] is True
        assert selection["development_tune"]["secondary_pass"] is False
    assert any(ids == dev_ids for ids in accesses)
    assert all(not (ids & confirm_ids) for ids in accesses)


def test_transfer_selection_and_one_shot_confirmation_are_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    examples = _corpus()
    internal_ids, dev_ids, confirm_ids = _split_ids(config, examples)
    selection_path = Path(config.output_root) / "QQP" / "selection.json"
    accesses = _install_fake_runtime(
        monkeypatch,
        internal_ids=internal_ids,
        dev_ids=dev_ids,
        confirm_ids=confirm_ids,
        fail_stage=None,
        claim_path=selection_path.parent / "confirmation_claim.json",
        internal_lock_path=selection_path.parent / "internal_qualification_lock.json",
    )
    selection = transfer.run_selection(
        config, train_examples=examples, tokenizer=object()
    )
    assert selection["format"] == transfer.FORMAT_VERSION
    assert selection["source_recipe_task"] == "BoolQA"
    assert selection["task"] == "QQP"
    assert selection["target_task_recipe_hparam_checkpoint_tuning"] is False
    assert selection["target_task_gradient_adaptation"] is True
    assert selection["target_task_internal_threshold_calibration"] is True
    assert selection["selected"]["step"] == 2
    assert selection["confirm_eligible"] is True
    assert set(selection["frozen_dependency_fingerprints"]) == {
        "pcsm_v1",
        "bap_v1",
        "shared_rescue",
        "order4_data",
        "objectives",
        "t5_rank_bank",
        "transfer_runner",
    }
    assert all(not (ids & confirm_ids) for ids in accesses)

    confirmation_config = replace(
        config,
        output_root=str(tmp_path / "confirmation"),
        confirm_selection=str(selection_path),
    )
    accesses_before = list(accesses)
    with pytest.raises(ValueError, match="scientific config differs"):
        transfer.run_confirmation(
            replace(
                confirmation_config,
                output_root=str(tmp_path / "freshness-drift"),
                fresh_confirm_against_manifests=("drifted.json",),
            ),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    protocol_path = selection_path.parent / "protocol_lock.json"
    original_protocol = protocol_path.read_bytes()
    protocol_path.write_bytes(original_protocol + b" ")
    with pytest.raises(ValueError, match="protocol lock path or SHA-256 changed"):
        transfer.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "protocol-drift")),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    protocol_path.write_bytes(original_protocol)
    checkpoint = Path(selection["selected_checkpoint"]["path"])
    original_checkpoint = checkpoint.read_bytes()
    checkpoint.write_bytes(original_checkpoint + b"tamper")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        transfer.run_confirmation(
            replace(
                confirmation_config, output_root=str(tmp_path / "checkpoint-drift")
            ),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    checkpoint.write_bytes(original_checkpoint)
    assert accesses == accesses_before
    assert not (selection_path.parent / "confirmation_claim.json").exists()

    original_dependency_fingerprints = transfer._dependency_fingerprints

    def drifted_dependency_fingerprints() -> dict[str, Any]:
        drifted = original_dependency_fingerprints()
        drifted["t5_rank_bank"] = {
            **drifted["t5_rank_bank"],
            "sha256": "0" * 64,
        }
        return drifted

    monkeypatch.setattr(
        transfer, "_dependency_fingerprints", drifted_dependency_fingerprints
    )
    with pytest.raises(ValueError, match="dependencies changed after selection"):
        transfer.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "dependency-drift")),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    assert accesses == accesses_before
    assert not (selection_path.parent / "confirmation_claim.json").exists()
    monkeypatch.setattr(
        transfer, "_dependency_fingerprints", original_dependency_fingerprints
    )

    result = transfer.run_confirmation(
        confirmation_config,
        selection,
        selection_path,
        train_examples=examples,
        tokenizer=object(),
    )
    assert result["confirm_primary_gate_pass"] is True
    assert result["target_task_recipe_hparam_checkpoint_tuning"] is False
    assert result["target_task_gradient_adaptation"] is True
    assert result["target_task_internal_threshold_calibration"] is True
    assert result["secondary_used_for_confirmation_selection_or_tuning"] is False
    assert any(ids == confirm_ids for ids in accesses)
    with pytest.raises(RuntimeError, match="already has a confirmation claim"):
        transfer.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "confirm-2")),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
