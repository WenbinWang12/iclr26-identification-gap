from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.phase2i_anchored_cvar import run_acquisition_rescue as rescue
from experiments.phase2i_anchored_cvar import run_bap_lora_rescue as bap
from experiments.phase2i_anchored_cvar import run_pcsm_lora_rescue as pcsm
from experiments.phase2i_anchored_cvar.order4_data import OfficialExample


def _example(index: int, label: str) -> OfficialExample:
    sentence = f"question: question {index}?\npassage: unique {label} passage {index}"
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


def _corpus(per_class: int = 8) -> tuple[OfficialExample, ...]:
    return tuple(
        _example(label_index * 100 + local_index, label)
        for label_index, label in enumerate(("True", "False"))
        for local_index in range(per_class)
    )


def _config(tmp_path: Path) -> pcsm.PCSMConfig:
    model_path = tmp_path / "model"
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "config.json").write_text(
        '{"model_type":"t5"}\n', encoding="utf-8"
    )
    return pcsm.PCSMConfig(
        data_root=str(tmp_path),
        model_path=str(model_path),
        output_root=str(tmp_path / "selection"),
        task_name="BoolQA",
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


def test_pcsm_primary_gate_is_raw_strict_and_secondary_is_independent() -> None:
    config = pcsm.PCSMConfig(
        data_root="data",
        model_path="model",
        output_root="out",
        task_name="BoolQA",
        fit_per_class=2,
        calibration_per_class=1,
        tune_audit_per_class=1,
        confirm_audit_per_class=1,
        optimizer_steps=2,
        batch_size=2,
        tiny_random_model=True,
    )
    base = _raw(natural=0.50, true_recall=1.0, false_recall=0.0)
    exact = _raw(natural=0.55, true_recall=1.0, false_recall=0.1)
    above = _raw(natural=0.551, true_recall=1.0, false_recall=0.102)
    assert not pcsm.primary_gate(
        pcsm.raw_official_comparison(base, exact), config=config
    )
    assert pcsm.primary_gate(
        pcsm.raw_official_comparison(base, above), config=config
    )
    invalid = dict(above)
    invalid["free_generation_valid_label_rate"] = 0.989
    assert not pcsm.primary_gate(
        pcsm.raw_official_comparison(base, invalid), config=config
    )

    secondary = {
        "training_contribution_natural_prior": 0.01,
        "training_min_per_class_delta": -0.05,
        "margin_diagnostics": {"roc_auc_delta": 0.01},
    }
    assert pcsm.secondary_gate(secondary, config=config)
    assert not pcsm.secondary_gate(
        {**secondary, "training_contribution_natural_prior": 0.0}, config=config
    )
    assert not pcsm.secondary_gate(
        {**secondary, "margin_diagnostics": {"roc_auc_delta": 0.0}},
        config=config,
    )


def test_formal_pcsm_constants_and_objective_are_locked(tmp_path: Path) -> None:
    sources = tuple(str(tmp_path / f"source-{index}.json") for index in range(4))
    config = pcsm.PCSMConfig(
        data_root=str(tmp_path / "data"),
        model_path=str(tmp_path / "model"),
        output_root=str(tmp_path / "out"),
        fresh_confirm_against_manifests=sources,
    )
    assert config.fit_cycles == 2
    assert pcsm._model_config(config).loss_objectives == (pcsm.OBJECTIVE,)
    protocol = pcsm._protocol_payload(config)
    assert protocol["fixed_trajectory"]["eligible_checkpoint_steps"] == [384]
    assert protocol["fixed_trajectory"]["candidate_count"] == 1
    assert protocol["primary_gate"]["metric"].startswith("raw official")
    assert protocol["secondary_gate"]["calibration_gain_counted_as_primary"] is False
    formal_sources = [
        {
            "path": sources[index],
            "sha256": value,
            "task": "BoolQA",
            **(
                {"format": rescue.ACCESS_ONLY_FRESHNESS_FORMAT}
                if value.startswith("c0a190")
                else {}
            ),
        }
        for index, value in enumerate(sorted(pcsm.FORMAL_FRESHNESS_SHA256S))
    ]
    pcsm._validate_formal_freshness_sources(formal_sources, config=config)
    wrong_composition = [dict(item) for item in formal_sources]
    wrong_composition[0]["format"] = rescue.ACCESS_ONLY_FRESHNESS_FORMAT
    with pytest.raises(ValueError, match="three legacy splits"):
        pcsm._validate_formal_freshness_sources(wrong_composition, config=config)
    wrong_hash = [dict(item) for item in formal_sources]
    wrong_hash[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 set changed"):
        pcsm._validate_formal_freshness_sources(wrong_hash, config=config)
    with pytest.raises(ValueError, match="fixed; drifted fields"):
        replace(config, optimizer_steps=192)
    with pytest.raises(ValueError, match="fixed; drifted fields"):
        replace(config, max_source_length=256)
    with pytest.raises(ValueError, match="exactly four freshness"):
        replace(config, fresh_confirm_against_manifests=sources[:3])


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


def test_tiny_random_pcsm_step_is_eval_mode_differentiable_and_budget_exact(
    tmp_path: Path,
) -> None:
    config = replace(_config(tmp_path), truncation_mode="official_right")
    tokenizer = _TinyTensorTokenizer()
    bank = rescue._build_model(pcsm._model_config(config), tokenizer)
    bank.eval()
    before = rescue._adapter_fingerprint(bank)
    optimizer = torch.optim.AdamW(
        tuple(bank.all_atom_parameters()), lr=config.learning_rate, weight_decay=0.0
    )
    loss, ledger = bap._pointwise_sequence_prior_step(
        bank,
        tokenizer,
        (_example(0, "True"), _example(100, "False")),
        optimizer,
        config=pcsm._as_bap_config(config),
        natural_prior={"True": 0.6, "False": 0.4},
    )
    assert bank.training is False
    assert loss == pytest.approx(float(loss))
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

    def fake_raw(bank, tokenizer, examples, *, config, natural_prior):
        del tokenizer, config, natural_prior
        ids = {item.example_id for item in examples}
        accesses.append(ids)
        learned = float(bank.parameter.detach()) > 0
        stage = (
            "internal"
            if ids == internal_ids
            else "development"
            if ids == dev_ids
            else "confirm"
            if ids == confirm_ids
            else "unknown"
        )
        if stage == "confirm":
            assert claim_path is not None and claim_path.is_file()
        if stage == "development":
            assert internal_lock_path is not None and internal_lock_path.is_file()
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
        stage = (
            "internal"
            if ids == internal_ids
            else "development"
            if ids == dev_ids
            else "confirm"
            if ids == confirm_ids
            else "unknown"
        )
        if learned and fail_stage == f"{stage}_secondary":
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


def _split_ids(
    config: pcsm.PCSMConfig, examples: tuple[OfficialExample, ...]
) -> tuple[set[str], set[str], set[str]]:
    outer, internal, _sources = pcsm._build_splits(examples, config=config)
    return (
        {item.example_id for item in internal.calibration},
        {item.example_id for item in outer.tune_audit},
        {item.example_id for item in outer.confirm_audit},
    )


@pytest.mark.parametrize("fail_stage", ["internal", "internal_secondary"])
def test_internal_failure_never_accesses_development_or_confirm(
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
            Path(config.output_root) / "BoolQA" / "internal_qualification_lock.json"
        ),
    )
    result = pcsm.run_selection(config, train_examples=examples, tokenizer=object())
    assert result["status"] == "internal_qualification_failed"
    assert result["development_tune_accessed"] is False
    assert result["confirm_eligible"] is False
    assert all(not (ids & dev_ids) for ids in accesses)
    assert all(not (ids & confirm_ids) for ids in accesses)


@pytest.mark.parametrize("fail_stage", ["development", "development_secondary"])
def test_development_failure_locks_ineligible_selection_without_confirm_access(
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
            Path(config.output_root) / "BoolQA" / "internal_qualification_lock.json"
        ),
    )
    result = pcsm.run_selection(config, train_examples=examples, tokenizer=object())
    assert result["status"] == "selection_locked"
    assert result["internal_calibration"]["qualification_pass"] is True
    if fail_stage == "development":
        assert result["development_tune"]["primary_pass"] is False
    else:
        assert result["development_tune"]["primary_pass"] is True
        assert result["development_tune"]["secondary_pass"] is False
    assert result["confirm_eligible"] is False
    assert any(ids == dev_ids for ids in accesses)
    assert all(not (ids & confirm_ids) for ids in accesses)
    assert result["resource_ledger"]["optimizer_steps"] == 2
    assert result["resource_ledger"]["gradient_example_exposures"] == 4
    assert result["resource_ledger"]["paired_exposures_per_class"] == {
        "True": 2,
        "False": 2,
    }


def test_fake_selection_and_one_shot_confirmation_are_hash_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    examples = _corpus()
    internal_ids, dev_ids, confirm_ids = _split_ids(config, examples)
    accesses = _install_fake_runtime(
        monkeypatch,
        internal_ids=internal_ids,
        dev_ids=dev_ids,
        confirm_ids=confirm_ids,
        fail_stage=None,
        claim_path=Path(config.output_root) / "BoolQA" / "confirmation_claim.json",
        internal_lock_path=(
            Path(config.output_root) / "BoolQA" / "internal_qualification_lock.json"
        ),
    )
    selection = pcsm.run_selection(
        config, train_examples=examples, tokenizer=object()
    )
    assert selection["confirm_eligible"] is True
    assert selection["selected"]["step"] == 2
    assert selection["selected"]["objective"] == pcsm.OBJECTIVE
    assert selection["resource_ledger"]["eligible_checkpoint_steps"] == [2]
    assert all(not (ids & confirm_ids) for ids in accesses)

    selection_path = Path(config.output_root) / "BoolQA" / "selection.json"
    confirmation_config = replace(
        config,
        output_root=str(tmp_path / "confirmation"),
        confirm_selection=str(selection_path),
    )
    accesses_before_attacks = list(accesses)
    tampered_memory = json.loads(json.dumps(selection))
    tampered_memory["selected"]["pcsm_threshold"] += 1.0
    with pytest.raises(ValueError, match="in-memory selection differs"):
        pcsm.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "memory-attack")),
            tampered_memory,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    assert accesses == accesses_before_attacks
    assert not (selection_path.parent / "confirmation_claim.json").exists()

    with pytest.raises(ValueError, match="scientific config differs"):
        pcsm.run_confirmation(
            replace(
                confirmation_config,
                output_root=str(tmp_path / "config-attack"),
                fresh_confirm_against_manifests=("drifted.json",),
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
        pcsm.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "corpus-attack")),
            selection,
            selection_path,
            train_examples=tampered_examples,
            tokenizer=object(),
        )
    original_environment_fingerprint = bap._execution_environment_fingerprint

    def drifted_environment(device):
        value = original_environment_fingerprint(device)
        return {**value, "aggregate_sha256": "drifted-environment"}

    monkeypatch.setattr(bap, "_execution_environment_fingerprint", drifted_environment)
    with pytest.raises(ValueError, match="execution environment drifted"):
        pcsm.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "env-attack")),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    monkeypatch.setattr(
        bap, "_execution_environment_fingerprint", original_environment_fingerprint
    )
    assert accesses == accesses_before_attacks
    assert not (selection_path.parent / "confirmation_claim.json").exists()

    selection_lock_path = selection_path.parent / "selection_lock.json"
    original_selection_lock = selection_lock_path.read_bytes()
    tampered_selection_lock = json.loads(original_selection_lock)
    tampered_selection_lock["selection_sha256"] = "0" * 64
    selection_lock_path.write_text(json.dumps(tampered_selection_lock), encoding="utf-8")
    with pytest.raises(ValueError, match="selection lock does not bind"):
        pcsm.run_confirmation(
            replace(
                confirmation_config, output_root=str(tmp_path / "selection-lock-attack")
            ),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    selection_lock_path.write_bytes(original_selection_lock)

    internal_lock_path = selection_path.parent / "internal_qualification_lock.json"
    original_internal_lock = internal_lock_path.read_bytes()
    internal_lock_path.write_bytes(original_internal_lock + b" ")
    with pytest.raises(ValueError, match="internal qualification lock path"):
        pcsm.run_confirmation(
            replace(
                confirmation_config, output_root=str(tmp_path / "internal-lock-attack")
            ),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    internal_lock_path.write_bytes(original_internal_lock)

    manifest_path = selection_path.parent.parent / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    tampered_manifest = json.loads(original_manifest)
    tampered_manifest["status"] = "tampered"
    manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest does not bind"):
        pcsm.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "manifest-attack")),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    manifest_path.write_bytes(original_manifest)
    assert accesses == accesses_before_attacks
    assert not (selection_path.parent / "confirmation_claim.json").exists()

    protocol_path = selection_path.parent / "protocol_lock.json"
    original_protocol = protocol_path.read_bytes()
    protocol_path.write_bytes(original_protocol + b" ")
    with pytest.raises(ValueError, match="protocol lock path or SHA-256 changed"):
        pcsm.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "protocol-attack")),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    protocol_path.write_bytes(original_protocol)
    assert accesses == accesses_before_attacks
    assert not (selection_path.parent / "confirmation_claim.json").exists()

    checkpoint = Path(selection["selected_checkpoint"]["path"])
    original_checkpoint = checkpoint.read_bytes()
    checkpoint.write_bytes(original_checkpoint + b"tamper")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        pcsm.run_confirmation(
            replace(
                confirmation_config, output_root=str(tmp_path / "checkpoint-attack")
            ),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    checkpoint.write_bytes(original_checkpoint)
    assert accesses == accesses_before_attacks
    assert not (selection_path.parent / "confirmation_claim.json").exists()

    model_stub = Path(config.model_path) / "config.json"
    original_model = model_stub.read_bytes()
    model_stub.write_bytes(b'{"model_type":"changed"}\n')
    with pytest.raises(ValueError, match="artifact changed"):
        pcsm.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "model-attack")),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    model_stub.write_bytes(original_model)
    monkeypatch.setattr(
        bap,
        "_tokenizer_protocol_fingerprint",
        lambda tokenizer: {"aggregate_sha256": "changed-tokenizer"},
    )
    with pytest.raises(ValueError, match="tokenizer protocol changed"):
        pcsm.run_confirmation(
            replace(
                confirmation_config, output_root=str(tmp_path / "tokenizer-attack")
            ),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
    monkeypatch.setattr(
        bap,
        "_tokenizer_protocol_fingerprint",
        lambda tokenizer: {"aggregate_sha256": "fixed-tokenizer"},
    )
    assert accesses == accesses_before_attacks
    assert not (selection_path.parent / "confirmation_claim.json").exists()

    result = pcsm.run_confirmation(
        confirmation_config,
        selection,
        selection_path,
        train_examples=examples,
        tokenizer=object(),
    )
    assert result["confirm_primary_gate_pass"] is True
    assert result["primary_raw_official"]["natural_prior_gain"] > 0.05
    assert result["secondary_used_for_confirmation_selection_or_tuning"] is False
    assert result["calibration_gain_counted_as_primary"] is False
    assert any(ids == confirm_ids for ids in accesses)
    claim = json.loads(
        (selection_path.parent / "confirmation_claim.json").read_text(
            encoding="utf-8"
        )
    )
    assert claim["status"] == "confirmation_completed"
    with pytest.raises(RuntimeError, match="already has a confirmation claim"):
        pcsm.run_confirmation(
            replace(confirmation_config, output_root=str(tmp_path / "confirmation-2")),
            selection,
            selection_path,
            train_examples=examples,
            tokenizer=object(),
        )
