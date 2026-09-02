"""Dependency-free tests for the pinned O-LoRA Order-4 data layer."""

from __future__ import annotations

from collections import Counter
import io
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from order4_data import (  # noqa: E402
    BlobSpec,
    DataIntegrityError,
    OFFICIAL_COMMIT,
    ORDER4_TASK_NAMES,
    TASK_BY_NAME,
    build_prompt,
    deterministic_per_class_cap,
    download_blob,
    git_blob_sha1,
    load_official_examples,
    load_sealed_evaluation_data,
    make_train_partitions,
    normalized_em,
    normalized_exact_match,
    official_blob_specs,
    validate_split,
    verify_blob,
)


EXPECTED_ORDER4 = (
    "MNLI",
    "CB",
    "WiC",
    "COPA",
    "QQP",
    "BoolQA",
    "RTE",
    "IMDB",
    "Yelp",
    "Amazon",
    "SST-2",
    "DBpedia",
    "AGNews",
    "MultiRC",
    "Yahoo",
)


def _write_synthetic_task(root: Path, task_name: str, split: str, records: list[dict]) -> None:
    spec = TASK_BY_NAME[task_name]
    data_path = root / spec.blob_for(split).relative_path
    labels_path = root / spec.label_file.relative_path
    data_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(records), encoding="utf-8")
    labels_path.write_text(json.dumps(spec.labels), encoding="utf-8")


def test_commit_order_and_manifest_are_locked_and_dev_free():
    assert OFFICIAL_COMMIT == "07117e1fc4a5f5ad9308a815a42cee8f46502dc8"
    assert ORDER4_TASK_NAMES == EXPECTED_ORDER4
    blobs = official_blob_specs()
    assert len(blobs) == 45
    assert len({blob.relative_path for blob in blobs}) == 45
    assert all("/dev.json" not in blob.relative_path.lower() for blob in blobs)
    assert all(
        blob.relative_path.endswith(("/train.json", "/test.json", "/labels.json"))
        for blob in blobs
    )


def test_dev_and_validation_aliases_fail_closed():
    for split in ("dev", "DEV", "val", "valid", "validation"):
        with pytest.raises(ValueError, match="prohibited"):
            validate_split(split)
    assert validate_split(" train ") == "train"
    assert validate_split("TEST") == "test"


def test_exact_prompt_including_copa_exception():
    sentence = "sentence 1: A.\nsentence 2: B."
    assert build_prompt("MNLI", sentence) == (
        "Task:NLI\n"
        "Dataset:MNLI\n"
        'What is the logical relationship between the "sentence 1" and the '
        '"sentence 2"? Choose one from the option.\n'
        "Option: neutral, entailment, contradiction \n"
        "sentence 1: A.\nsentence 2: B.\n"
        "Answer:"
    )

    copa_sentence = (
        'Which sentence is the cause of "X"? Choose one between A and B.\n'
        "A: first\nB: second"
    )
    copa_prompt = build_prompt("COPA", copa_sentence)
    assert copa_prompt == f"Task:COPA\nDataset:COPA\n{copa_sentence}\nAnswer:"
    assert "Option:" not in copa_prompt


def test_blob_download_is_atomic_verified_and_reused(tmp_path: Path):
    payload = b'[{"label":"x","sentence":"hello"}]'
    spec = BlobSpec(
        "CL_Benchmark/X/X/train.json",
        len(payload),
        git_blob_sha1(payload),
    )
    calls: list[tuple[str, float]] = []

    def opener(url: str, *, timeout: float):
        calls.append((url, timeout))
        return io.BytesIO(payload)

    path = download_blob(tmp_path, spec, opener=opener, timeout=7.0)
    assert path.read_bytes() == payload
    assert verify_blob(path, spec)
    assert len(calls) == 1

    def should_not_open(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("verified local blob should have been reused")

    assert download_blob(tmp_path, spec, opener=should_not_open) == path
    assert not path.with_name(path.name + ".part").exists()


def test_bad_download_never_replaces_existing_target(tmp_path: Path):
    expected = b"trusted bytes"
    corrupt = b"wrong bytes!!"
    spec = BlobSpec("CL_Benchmark/X/test.json", len(expected), git_blob_sha1(expected))
    target = tmp_path / spec.relative_path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"previous invalid local copy")

    def opener(url: str, *, timeout: float):
        return io.BytesIO(corrupt)

    with pytest.raises(DataIntegrityError, match="verification failed"):
        download_blob(tmp_path, spec, opener=opener)
    assert target.read_bytes() == b"previous invalid local copy"
    assert not target.with_name(target.name + ".part").exists()


def test_loading_schema_label_order_and_stable_ids(tmp_path: Path):
    records = [
        {"label": "neutral", "sentence": "sentence 1: a\nsentence 2: b"},
        {"label": "entailment", "sentence": "sentence 1: c\nsentence 2: d"},
    ]
    _write_synthetic_task(tmp_path, "MNLI", "train", records)
    first = load_official_examples(tmp_path, "MNLI", "train", verify=False)
    second = load_official_examples(tmp_path, "MNLI", "train", verify=False)
    assert first == second
    assert len({item.example_id for item in first}) == 2
    assert all(len(item.example_id) == 64 for item in first)
    assert first[0].prompt == build_prompt("MNLI", records[0]["sentence"])

    labels_path = tmp_path / TASK_BY_NAME["MNLI"].label_file.relative_path
    labels_path.write_text(
        json.dumps(["entailment", "neutral", "contradiction"]),
        encoding="utf-8",
    )
    with pytest.raises(DataIntegrityError, match="label order mismatch"):
        load_official_examples(tmp_path, "MNLI", "train", verify=False)


def test_per_class_cap_and_train_partitions_are_stable_and_disjoint(tmp_path: Path):
    records = []
    for label in ("neutral", "entailment", "contradiction"):
        records.extend(
            {"label": label, "sentence": f"{label} example {index}"}
            for index in range(7)
        )
    _write_synthetic_task(tmp_path, "MNLI", "train", records)
    examples = load_official_examples(tmp_path, "MNLI", "train", verify=False)

    capped = deterministic_per_class_cap(examples, 5, seed=42)
    reverse_capped = deterministic_per_class_cap(reversed(examples), 5, seed=42)
    assert [item.example_id for item in capped] == [
        item.example_id for item in reverse_capped
    ]
    assert Counter(item.label for item in capped) == {
        "neutral": 5,
        "entailment": 5,
        "contradiction": 5,
    }

    partitions = make_train_partitions(
        examples,
        cap_per_class=5,
        risk_per_class=1,
        audit_per_class=1,
        seed=42,
    )
    reverse = make_train_partitions(
        reversed(examples),
        cap_per_class=5,
        risk_per_class=1,
        audit_per_class=1,
        seed=42,
    )
    partitions.assert_disjoint()
    assert partitions == reverse
    assert Counter(item.label for item in partitions.update) == {
        "neutral": 3,
        "entailment": 3,
        "contradiction": 3,
    }
    assert Counter(item.label for item in partitions.risk) == {
        "neutral": 1,
        "entailment": 1,
        "contradiction": 1,
    }
    assert Counter(item.label for item in partitions.audit) == {
        "neutral": 1,
        "entailment": 1,
        "contradiction": 1,
    }


def test_test_examples_cannot_enter_risk_or_audit(tmp_path: Path):
    records = [
        {"label": "neutral", "sentence": "x"},
        {"label": "neutral", "sentence": "y"},
    ]
    _write_synthetic_task(tmp_path, "MNLI", "test", records)
    examples = load_official_examples(tmp_path, "MNLI", "test", verify=False)
    with pytest.raises(ValueError, match="train-derived"):
        make_train_partitions(
            examples,
            cap_per_class=None,
            risk_per_class=0,
            audit_per_class=0,
            seed=42,
        )


def test_sealed_evaluator_drops_only_dash_and_preserves_source_indices(tmp_path: Path):
    records = [
        {"label": "neutral", "sentence": "labelled zero"},
        {"label": "-", "sentence": "unlabelled one"},
        {"label": "entailment", "sentence": "labelled two"},
    ]
    _write_synthetic_task(tmp_path, "MNLI", "test", records)
    with pytest.raises(DataIntegrityError, match="unknown label"):
        load_official_examples(tmp_path, "MNLI", "test", verify=False)
    sealed = load_sealed_evaluation_data(tmp_path, "MNLI", verify=False)
    assert sealed.excluded_unlabeled_count == 1
    assert len(sealed.excluded_unlabeled_sha256) == 64
    assert [example.source_index for example in sealed.examples] == [0, 2]
    assert [example.label for example in sealed.examples] == ["neutral", "entailment"]

    records[1]["label"] = "not-a-real-label"
    _write_synthetic_task(tmp_path, "MNLI", "test", records)
    with pytest.raises(DataIntegrityError, match="unknown label"):
        load_sealed_evaluation_data(tmp_path, "MNLI", verify=False)


def test_normalized_em_matches_official_normalizer():
    assert normalized_exact_match(" Science-or Technology! ", "scienceor technology")
    assert normalized_em(["True.", "wrong"], ["true", "False"]) == 50.0
    with pytest.raises(ValueError, match="equal length"):
        normalized_em(["a"], [])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
