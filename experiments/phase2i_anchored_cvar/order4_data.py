"""Pinned O-LoRA Long-CL Order-4 data and prompt preparation.

This module intentionally has no dependency on :mod:`datasets`.  It mirrors
the prompt construction in O-LoRA's ``uie_dataset_lora.py`` and
``uie_collator.py`` at :data:`OFFICIAL_COMMIT`, while adding the leakage-safe
train-derived partitions required by the Phase-2I headroom panel.

The upstream repository contains several byte-identical ``dev.json`` and
``test.json`` files.  Consequently, ``dev``/``validation`` access is rejected
here by construction: the audit split is always carved out of ``train.json``
and ``test.json`` remains sealed for stage-end evaluation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import string
from typing import BinaryIO, Callable, Iterable, Mapping, Sequence
from urllib.request import urlopen


OFFICIAL_REPOSITORY = "cmnfriend/O-LoRA"
OFFICIAL_COMMIT = "07117e1fc4a5f5ad9308a815a42cee8f46502dc8"
RAW_BASE_URL = (
    f"https://raw.githubusercontent.com/{OFFICIAL_REPOSITORY}/"
    f"{OFFICIAL_COMMIT}/"
)


INSTRUCTIONS: Mapping[str, str] = {
    "NLI": (
        'What is the logical relationship between the "sentence 1" and the '
        '"sentence 2"? Choose one from the option.\n'
    ),
    "QQP": (
        'Whether the "first sentence" and the "second sentence" have the '
        "same meaning? Choose one from the option.\n"
    ),
    "SC": (
        "What is the sentiment of the following paragraph? Choose one from "
        "the option.\n"
    ),
    "TC": (
        "What is the topic of the following paragraph? Choose one from the "
        "option.\n"
    ),
    "BoolQA": (
        "According to the following passage, is the question true or false? "
        "Choose one from the option.\n"
    ),
    "MultiRC": (
        "According to the following passage and question, is the candidate "
        "answer true or false? Choose one from the option.\n"
    ),
    "WiC": (
        "Given a word and two sentences, whether the word is used with the "
        "same sense in both sentence? Choose one from the option.\n"
    ),
}


class DataIntegrityError(RuntimeError):
    """Raised when a downloaded or local file is not the pinned Git blob."""


@dataclass(frozen=True)
class BlobSpec:
    """One content-addressed file from the pinned upstream repository."""

    relative_path: str
    size: int
    git_blob_sha1: str

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe manifest path: {self.relative_path!r}")
        if self.size < 0:
            raise ValueError("blob size must be non-negative")
        if len(self.git_blob_sha1) != 40 or any(
            char not in string.hexdigits for char in self.git_blob_sha1
        ):
            raise ValueError(f"invalid Git blob SHA-1: {self.git_blob_sha1!r}")
        lower_path = self.relative_path.lower()
        if lower_path.endswith("/dev.json") or lower_path.endswith("\\dev.json"):
            raise ValueError("dev.json is prohibited in the Phase-2I manifest")

    @property
    def url(self) -> str:
        return RAW_BASE_URL + self.relative_path.replace("\\", "/")


@dataclass(frozen=True)
class TaskSpec:
    """Official task metadata and its only permitted source files."""

    name: str
    category: str
    dataset: str
    labels: tuple[str, ...]
    train: BlobSpec
    test: BlobSpec
    label_file: BlobSpec

    def blob_for(self, split: str) -> BlobSpec:
        canonical = validate_split(split)
        return self.train if canonical == "train" else self.test


@dataclass(frozen=True)
class OfficialExample:
    """A prepared example with a stable content/source identifier."""

    task_name: str
    category: str
    dataset: str
    subset: str
    source_index: int
    example_id: str
    sentence: str
    label: str
    prompt: str


@dataclass(frozen=True)
class TrainPartitions:
    """Disjoint, train-derived update/risk/audit partitions."""

    update: tuple[OfficialExample, ...]
    risk: tuple[OfficialExample, ...]
    audit: tuple[OfficialExample, ...]

    def assert_disjoint(self) -> None:
        ids = [
            {example.example_id for example in partition}
            for partition in (self.update, self.risk, self.audit)
        ]
        if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
            raise AssertionError("update/risk/audit partitions overlap")


@dataclass(frozen=True)
class SealedEvaluationData:
    """Labelled test examples plus an auditable unlabeled-row exclusion.

    The pinned MNLI test file contains rows whose target is the literal ``-``.
    They cannot define exact-match accuracy and must never be mapped to a
    class.  This report makes their evaluator-only exclusion explicit and
    content-addressed while preserving original source indices for retained
    examples.
    """

    examples: tuple[OfficialExample, ...]
    excluded_unlabeled_count: int
    excluded_unlabeled_sha256: str


def _blob(path: str, size: int, sha1: str) -> BlobSpec:
    return BlobSpec(path, size, sha1)


def _task(
    name: str,
    category: str,
    dataset: str,
    labels: Sequence[str],
    label_blob: tuple[str, int, str],
    test_blob: tuple[str, int, str],
    train_blob: tuple[str, int, str],
) -> TaskSpec:
    return TaskSpec(
        name=name,
        category=category,
        dataset=dataset,
        labels=tuple(labels),
        train=_blob(*train_blob),
        test=_blob(*test_blob),
        label_file=_blob(*label_blob),
    )


# O-LoRA Appendix Order-4 (the same sequence called Order-8 by Progressive
# Prompts).  Tuple order is part of the experimental protocol.
ORDER4_TASKS: tuple[TaskSpec, ...] = (
    _task(
        "MNLI", "NLI", "MNLI",
        ("neutral", "entailment", "contradiction"),
        ("CL_Benchmark/NLI/MNLI/labels.json", 42, "35b498247fffa71bf7be5413a4425535b5f89465"),
        ("CL_Benchmark/NLI/MNLI/test.json", 1_783_461, "0295bdfa4f323a1919451f45fa015e3fa7c7c699"),
        ("CL_Benchmark/NLI/MNLI/train.json", 713_744, "7051cb0145c7dd70572581b9a0021a68ba4099fc"),
    ),
    _task(
        "CB", "NLI", "CB",
        ("entailment", "contradiction", "neutral"),
        ("CL_Benchmark/NLI/CB/labels.json", 42, "e35c6f5565e71c4029f1a1676eeb9382b741f427"),
        ("CL_Benchmark/NLI/CB/test.json", 24_552, "48c0c2e31bedd7dde7fc7b6608260406979c33b7"),
        ("CL_Benchmark/NLI/CB/train.json", 99_109, "12fa0454815334621e517fce092b01f301ec5fd8"),
    ),
    _task(
        "WiC", "WiC", "WiC", ("True", "False"),
        ("CL_Benchmark/WiC/WiC/labels.json", 17, "c94cbaa2fc1f2f45808d8ce54752125e58a7f3f8"),
        ("CL_Benchmark/WiC/WiC/test.json", 85_336, "f8bd891b9a448ebc0f51b5ddedcb08e6bbd837f6"),
        ("CL_Benchmark/WiC/WiC/train.json", 253_640, "2a0b421cfc974e1860c5bca6fc2631f3d1a393e5"),
    ),
    _task(
        "COPA", "COPA", "COPA", ("A", "B"),
        ("CL_Benchmark/COPA/COPA/labels.json", 10, "21df2a72953868047e8c360a5cfaf21b920e52dd"),
        ("CL_Benchmark/COPA/COPA/test.json", 19_779, "0afde22eee567d7749443651d0ab8bc6bb56df17"),
        ("CL_Benchmark/COPA/COPA/train.json", 78_433, "69bc50fee02e7197ea68c60f80e76f44ece31429"),
    ),
    _task(
        "QQP", "QQP", "QQP", ("False", "True"),
        ("CL_Benchmark/QQP/QQP/labels.json", 17, "464cc7f4820eb5def8282fef5ad3ed74e54b0e21"),
        ("CL_Benchmark/QQP/QQP/test.json", 1_449_179, "1fb3020b4655aa54c74074785cc35f5ade393c3b"),
        ("CL_Benchmark/QQP/QQP/train.json", 382_085, "d0f46e45ba75dcfcae47ea380356bd695140c36b"),
    ),
    _task(
        "BoolQA", "BoolQA", "BoolQA", ("True", "False"),
        ("CL_Benchmark/BoolQA/BoolQA/labels.json", 17, "c94cbaa2fc1f2f45808d8ce54752125e58a7f3f8"),
        ("CL_Benchmark/BoolQA/BoolQA/test.json", 2_235_456, "de9b9d2b3013b78a1e71480c6617358061052671"),
        ("CL_Benchmark/BoolQA/BoolQA/train.json", 1_376_246, "14430c7210eb587b4b403eb83ab1f9cb06875dba"),
    ),
    _task(
        "RTE", "NLI", "RTE", ("contradiction", "entailment"),
        ("CL_Benchmark/NLI/RTE/labels.json", 31, "bfad21d93775c46f579cdc987ad2c1087401a043"),
        ("CL_Benchmark/NLI/RTE/test.json", 104_295, "4d78479bc8b067075c39cbe1772c662bb44a6556"),
        ("CL_Benchmark/NLI/RTE/train.json", 783_059, "64bf411495f98e34f744c1d2757c959c19d05bff"),
    ),
    _task(
        "IMDB", "SC", "IMDB", ("Good", "Bad"),
        ("CL_Benchmark/SC/IMDB/labels.json", 15, "b6a488522912d3f3003d3f490561d6d2528273a5"),
        ("CL_Benchmark/SC/IMDB/test.json", 10_220_306, "65ff1c75c533bb018aa405610e4dfc361d4635ca"),
        ("CL_Benchmark/SC/IMDB/train.json", 2_747_877, "f946665585c86c2b7a13f3277557b3aa33d9185e"),
    ),
    _task(
        "Yelp", "SC", "yelp",
        ("very negative", "negative", "neutral", "positive", "very positive"),
        ("CL_Benchmark/SC/yelp/labels.json", 69, "647f8e328a233a091faa548b0dc6fd3d7cf94d6b"),
        ("CL_Benchmark/SC/yelp/test.json", 6_023_185, "4783bfd68806a6ed714ef8bbe3306cbe492ece9c"),
        ("CL_Benchmark/SC/yelp/train.json", 3_971_368, "703671e733fcf26b0971d62446657e58b3498aa6"),
    ),
    _task(
        "Amazon", "SC", "amazon",
        ("very negative", "negative", "neutral", "positive", "very positive"),
        ("CL_Benchmark/SC/amazon/labels.json", 69, "647f8e328a233a091faa548b0dc6fd3d7cf94d6b"),
        ("CL_Benchmark/SC/amazon/test.json", 3_796_075, "06155095382ec90e80db4a19d451b52ad8e9707d"),
        ("CL_Benchmark/SC/amazon/train.json", 2_462_412, "5336e68ba1613d0bea976e77bc68a86ac6cca6ce"),
    ),
    _task(
        "SST-2", "SC", "SST-2", ("Good", "Bad"),
        ("CL_Benchmark/SC/SST-2/labels.json", 15, "b6a488522912d3f3003d3f490561d6d2528273a5"),
        ("CL_Benchmark/SC/SST-2/test.json", 127_624, "358a86380ea7df31bab9e877113ede463b891974"),
        ("CL_Benchmark/SC/SST-2/train.json", 187_855, "7565a9fd0c5f4bd7b982f3af8d82c0f50b93cdd2"),
    ),
    _task(
        "DBpedia", "TC", "dbpedia",
        (
            "Company", "Educational Institution", "Artist", "Athlete",
            "Office Holder", "Mean of Transportation", "Building",
            "Natural Place", "Village", "Animal", "Plant", "Album", "Film",
            "Written Work",
        ),
        ("CL_Benchmark/TC/dbpedia/labels.json", 194, "4c2c58499965cf20eea9374496b19610c34b63de"),
        ("CL_Benchmark/TC/dbpedia/test.json", 2_731_145, "d9cd947dd5e1df0290484564e946320d81dcaf8b"),
        ("CL_Benchmark/TC/dbpedia/train.json", 5_039_390, "fa31f7f2b3037640dc1e9e84959e507aca169549"),
    ),
    _task(
        "AGNews", "TC", "agnews",
        ("World", "Sports", "Business", "Science or Technology"),
        ("CL_Benchmark/TC/agnews/labels.json", 56, "a5da0b793e0b85cf8997b717c71a16c4836a5c7b"),
        ("CL_Benchmark/TC/agnews/test.json", 2_223_897, "6a3a6265cf5d7eda03884b7851cf542559de68f6"),
        ("CL_Benchmark/TC/agnews/train.json", 1_171_164, "22dfdf072e580c47e5c3c2d97efa5abfca853979"),
    ),
    _task(
        "MultiRC", "MultiRC", "MultiRC", ("False", "True"),
        ("CL_Benchmark/MultiRC/MultiRC/labels.json", 17, "464cc7f4820eb5def8282fef5ad3ed74e54b0e21"),
        ("CL_Benchmark/MultiRC/MultiRC/test.json", 7_993_302, "a98113c3bfffaff29ca87ea4d7e8bf09e5ec2fc3"),
        ("CL_Benchmark/MultiRC/MultiRC/train.json", 3_531_818, "66e2907b9315e71a77aa5d21540424afcff5abf3"),
    ),
    _task(
        "Yahoo", "TC", "yahoo",
        (
            "Society & Culture", "Science & Mathematics", "Health",
            "Education & Reference", "Computers & Internet", "Sports",
            "Business & Finance", "Entertainment & Music",
            "Family & Relationships", "Politics & Government",
        ),
        ("CL_Benchmark/TC/yahoo/labels.json", 213, "ed2fb4c33d1f88d4647431feebe9086070d13045"),
        ("CL_Benchmark/TC/yahoo/test.json", 4_521_852, "171cfbdce0d6e62c5ea3fa69b959b0843e6fb665"),
        ("CL_Benchmark/TC/yahoo/train.json", 5_998_676, "ba092987523a63eeb242dc4f52f623da3e837ab1"),
    ),
)

ORDER4_TASK_NAMES: tuple[str, ...] = tuple(task.name for task in ORDER4_TASKS)
TASK_BY_NAME: Mapping[str, TaskSpec] = {task.name: task for task in ORDER4_TASKS}


def validate_split(split: str) -> str:
    """Return the canonical allowed split or fail closed on dev aliases."""

    if not isinstance(split, str):
        raise TypeError("split must be a string")
    canonical = split.strip().lower()
    if canonical in {"dev", "val", "valid", "validation"}:
        raise ValueError(
            "dev/validation data are prohibited: audit data must be carved "
            "from train.json; test.json is reserved for stage-end evaluation"
        )
    if canonical not in {"train", "test"}:
        raise ValueError(f"unsupported split {split!r}; allowed: train, test")
    return canonical


def _resolve_task(task_name: str) -> TaskSpec:
    try:
        return TASK_BY_NAME[task_name]
    except KeyError as error:
        raise KeyError(
            f"unknown Order-4 task {task_name!r}; expected one of "
            f"{ORDER4_TASK_NAMES}"
        ) from error


def official_blob_specs(
    task_names: Sequence[str] = ORDER4_TASK_NAMES,
) -> tuple[BlobSpec, ...]:
    """Return the pinned train/test/label manifest in protocol order."""

    blobs: list[BlobSpec] = []
    seen: set[str] = set()
    for task_name in task_names:
        task = _resolve_task(task_name)
        for blob in (task.train, task.test, task.label_file):
            if blob.relative_path not in seen:
                blobs.append(blob)
                seen.add(blob.relative_path)
    return tuple(blobs)


def git_blob_sha1(data: bytes) -> str:
    """Compute Git's SHA-1 object ID for a byte string."""

    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def verify_blob(path: str | os.PathLike[str], spec: BlobSpec) -> bool:
    """Return whether ``path`` is exactly the pinned Git blob."""

    source = Path(path)
    try:
        if source.stat().st_size != spec.size:
            return False
        digest = hashlib.sha1(f"blob {spec.size}\0".encode("ascii"))
        with source.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest() == spec.git_blob_sha1.lower()
    except FileNotFoundError:
        return False


Opener = Callable[..., BinaryIO]


def download_blob(
    data_root: str | os.PathLike[str],
    spec: BlobSpec,
    *,
    opener: Opener = urlopen,
    timeout: float = 120.0,
) -> Path:
    """Download one blob atomically and verify size plus Git object ID.

    A valid existing file is reused.  An invalid existing target is left in
    place until a fully verified replacement has been downloaded.
    """

    root = Path(data_root).expanduser().resolve()
    destination = (root / spec.relative_path).resolve()
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise ValueError(f"manifest path escapes data root: {spec.relative_path}") from error

    if verify_blob(destination, spec):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    digest = hashlib.sha1(f"blob {spec.size}\0".encode("ascii"))
    actual_size = 0
    try:
        with opener(spec.url, timeout=timeout) as response, part.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                actual_size += len(chunk)
        actual_sha1 = digest.hexdigest()
        if actual_size != spec.size or actual_sha1 != spec.git_blob_sha1.lower():
            raise DataIntegrityError(
                f"pinned blob verification failed for {spec.relative_path}: "
                f"size={actual_size} (expected {spec.size}), "
                f"git_sha1={actual_sha1} (expected {spec.git_blob_sha1})"
            )
        os.replace(part, destination)
    finally:
        try:
            part.unlink()
        except FileNotFoundError:
            pass
    return destination


def download_official_data(
    data_root: str | os.PathLike[str],
    task_names: Sequence[str] = ORDER4_TASK_NAMES,
    *,
    opener: Opener = urlopen,
    timeout: float = 120.0,
) -> tuple[Path, ...]:
    """Materialize only pinned train/test/labels files under ``data_root``."""

    return tuple(
        download_blob(data_root, blob, opener=opener, timeout=timeout)
        for blob in official_blob_specs(task_names)
    )


def build_prompt(
    task: TaskSpec | str,
    sentence: str,
    *,
    labels: Sequence[str] | None = None,
) -> str:
    """Build the exact O-LoRA T5 input string with task/dataset prefixes."""

    spec = _resolve_task(task) if isinstance(task, str) else task
    task_labels = tuple(labels) if labels is not None else spec.labels
    prefix = f"Task:{spec.category}\nDataset:{spec.dataset}\n"
    if spec.category == "COPA":
        # Upstream COPA sentences already contain the question and A/B options.
        return prefix + sentence + "\nAnswer:"
    instruction = INSTRUCTIONS[spec.category]
    return (
        prefix
        + instruction
        + "Option: "
        + ", ".join(task_labels)
        + " \n"
        + sentence
        + "\nAnswer:"
    )


def _stable_json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _records_to_examples(
    spec: TaskSpec,
    split: str,
    records: Sequence[object],
    labels: Sequence[object],
    *,
    source_indices: Sequence[int] | None = None,
) -> tuple[OfficialExample, ...]:
    canonical_split = validate_split(split)
    if tuple(labels) != spec.labels:
        raise DataIntegrityError(
            f"label order mismatch for {spec.name}: got {tuple(labels)!r}, "
            f"expected {spec.labels!r}"
        )

    if source_indices is None:
        indices = tuple(range(len(records)))
    else:
        indices = tuple(int(index) for index in source_indices)
        if len(indices) != len(records) or len(set(indices)) != len(indices):
            raise ValueError("source_indices must be unique and match records")
        if any(index < 0 for index in indices):
            raise ValueError("source_indices must be non-negative")

    output: list[OfficialExample] = []
    for index, raw in zip(indices, records):
        if not isinstance(raw, dict):
            raise DataIntegrityError(
                f"{spec.name}/{canonical_split} record {index} is not an object"
            )
        sentence = raw.get("sentence")
        label = raw.get("label")
        if not isinstance(sentence, str) or not isinstance(label, str):
            raise DataIntegrityError(
                f"{spec.name}/{canonical_split} record {index} requires string "
                "'sentence' and 'label' fields"
            )
        if label not in spec.labels:
            raise DataIntegrityError(
                f"unknown label {label!r} in {spec.name}/{canonical_split} "
                f"record {index}"
            )
        example_id = _stable_json_hash(
            {
                "commit": OFFICIAL_COMMIT,
                "task": spec.name,
                "split": canonical_split,
                "source_index": index,
                "sentence": sentence,
                "label": label,
            }
        )
        output.append(
            OfficialExample(
                task_name=spec.name,
                category=spec.category,
                dataset=spec.dataset,
                subset=canonical_split,
                source_index=index,
                example_id=example_id,
                sentence=sentence,
                label=label,
                prompt=build_prompt(spec, sentence, labels=spec.labels),
            )
        )
    return tuple(output)


def load_official_examples(
    data_root: str | os.PathLike[str],
    task_name: str,
    split: str,
    *,
    verify: bool = True,
) -> tuple[OfficialExample, ...]:
    """Load one pinned split and validate schema, labels, and (by default) bytes."""

    spec = _resolve_task(task_name)
    canonical_split = validate_split(split)
    data_path = Path(data_root).expanduser().resolve() / spec.blob_for(canonical_split).relative_path
    labels_path = Path(data_root).expanduser().resolve() / spec.label_file.relative_path
    if verify:
        if not verify_blob(data_path, spec.blob_for(canonical_split)):
            raise DataIntegrityError(f"missing or invalid pinned data file: {data_path}")
        if not verify_blob(labels_path, spec.label_file):
            raise DataIntegrityError(f"missing or invalid pinned labels file: {labels_path}")
    try:
        with data_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        with labels_path.open("r", encoding="utf-8") as handle:
            labels = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise DataIntegrityError(f"could not parse official data for {task_name}") from error
    if not isinstance(records, list) or not isinstance(labels, list):
        raise DataIntegrityError("official data and labels JSON must both be lists")
    return _records_to_examples(spec, canonical_split, records, labels)


def load_sealed_evaluation_data(
    data_root: str | os.PathLike[str],
    task_name: str,
    *,
    verify: bool = True,
) -> SealedEvaluationData:
    """Load labelled ``test.json`` rows and explicitly drop only label ``-``.

    Training remains strict through :func:`load_official_examples`; this
    evaluator-specific entry point is the sole place where the upstream MNLI
    anomaly is tolerated.  Any other unknown label still fails closed.
    """

    spec = _resolve_task(task_name)
    data_path = Path(data_root).expanduser().resolve() / spec.test.relative_path
    labels_path = Path(data_root).expanduser().resolve() / spec.label_file.relative_path
    if verify:
        if not verify_blob(data_path, spec.test):
            raise DataIntegrityError(f"missing or invalid pinned data file: {data_path}")
        if not verify_blob(labels_path, spec.label_file):
            raise DataIntegrityError(f"missing or invalid pinned labels file: {labels_path}")
    try:
        with data_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        with labels_path.open("r", encoding="utf-8") as handle:
            labels = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise DataIntegrityError(f"could not parse official data for {task_name}") from error
    if not isinstance(records, list) or not isinstance(labels, list):
        raise DataIntegrityError("official data and labels JSON must both be lists")

    retained_records: list[object] = []
    retained_indices: list[int] = []
    excluded: list[dict[str, object]] = []
    for source_index, raw in enumerate(records):
        if isinstance(raw, dict) and raw.get("label") == "-":
            excluded.append({"source_index": source_index, "record": raw})
        else:
            retained_records.append(raw)
            retained_indices.append(source_index)
    exclusion_bytes = json.dumps(
        excluded,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    examples = _records_to_examples(
        spec,
        "test",
        retained_records,
        labels,
        source_indices=retained_indices,
    )
    return SealedEvaluationData(
        examples=examples,
        excluded_unlabeled_count=len(excluded),
        excluded_unlabeled_sha256=hashlib.sha256(exclusion_bytes).hexdigest(),
    )


def _rank_key(example: OfficialExample, *, seed: int, purpose: str) -> str:
    return _stable_json_hash(
        {
            "seed": int(seed),
            "purpose": purpose,
            "example_id": example.example_id,
        }
    )


def deterministic_per_class_cap(
    examples: Iterable[OfficialExample],
    cap_per_class: int | None,
    *,
    seed: int,
) -> tuple[OfficialExample, ...]:
    """Hash-sample at most ``cap_per_class`` examples from every label.

    Selection is independent of input iteration order and Python's randomized
    object hash.  The returned sequence is in source order for reproducible
    downstream batching.
    """

    materialized = tuple(examples)
    if cap_per_class is None:
        return tuple(sorted(materialized, key=lambda item: (item.task_name, item.source_index)))
    if not isinstance(cap_per_class, int) or isinstance(cap_per_class, bool):
        raise TypeError("cap_per_class must be an integer or None")
    if cap_per_class <= 0:
        raise ValueError("cap_per_class must be positive")

    groups: dict[tuple[str, str], list[OfficialExample]] = defaultdict(list)
    for example in materialized:
        groups[(example.task_name, example.label)].append(example)

    selected: list[OfficialExample] = []
    for group in groups.values():
        ranked = sorted(
            group,
            key=lambda item: (
                _rank_key(item, seed=seed, purpose="per-class-cap"),
                item.example_id,
            ),
        )
        selected.extend(ranked[:cap_per_class])
    return tuple(sorted(selected, key=lambda item: (item.task_name, item.source_index)))


def make_train_partitions(
    examples: Iterable[OfficialExample],
    *,
    cap_per_class: int | None,
    risk_per_class: int,
    audit_per_class: int,
    seed: int,
) -> TrainPartitions:
    """Create disjoint update/risk/audit sets exclusively from train data.

    Risk and audit quotas are applied independently within every task/label
    class after deterministic capping.  The function fails if a class cannot
    retain at least one update example, avoiding silent empty-class training.
    """

    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (risk_per_class, audit_per_class)
    ):
        raise ValueError("risk_per_class and audit_per_class must be non-negative integers")

    materialized = tuple(examples)
    non_train = [example for example in materialized if example.subset != "train"]
    if non_train:
        raise ValueError(
            "risk/audit partitions must be train-derived; received non-train "
            f"example {non_train[0].example_id} from {non_train[0].subset!r}"
        )
    capped = deterministic_per_class_cap(materialized, cap_per_class, seed=seed)
    groups: dict[tuple[str, str], list[OfficialExample]] = defaultdict(list)
    for example in capped:
        groups[(example.task_name, example.label)].append(example)

    update: list[OfficialExample] = []
    risk: list[OfficialExample] = []
    audit: list[OfficialExample] = []
    held_out = risk_per_class + audit_per_class
    for key, group in groups.items():
        if len(group) <= held_out:
            raise ValueError(
                f"class {key} has {len(group)} capped examples, but risk+audit "
                f"requires {held_out}; at least one update example is required"
            )
        ranked = sorted(
            group,
            key=lambda item: (
                _rank_key(item, seed=seed, purpose="train-partition"),
                item.example_id,
            ),
        )
        risk.extend(ranked[:risk_per_class])
        audit.extend(ranked[risk_per_class:held_out])
        update.extend(ranked[held_out:])

    sort_key = lambda item: (item.task_name, item.source_index)
    result = TrainPartitions(
        update=tuple(sorted(update, key=sort_key)),
        risk=tuple(sorted(risk, key=sort_key)),
        audit=tuple(sorted(audit, key=sort_key)),
    )
    result.assert_disjoint()
    return result


def prepare_task_partitions(
    data_root: str | os.PathLike[str],
    task_name: str,
    *,
    cap_per_class: int | None,
    risk_per_class: int,
    audit_per_class: int,
    seed: int,
    verify: bool = True,
) -> TrainPartitions:
    """Load a pinned task's train file and produce leakage-safe partitions."""

    examples = load_official_examples(data_root, task_name, "train", verify=verify)
    return make_train_partitions(
        examples,
        cap_per_class=cap_per_class,
        risk_per_class=risk_per_class,
        audit_per_class=audit_per_class,
        seed=seed,
    )


def normalize_answer(text: str) -> str:
    """Match O-LoRA's EM normalization: lower, ASCII punctuation, whitespace."""

    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    return " ".join(without_punctuation.split())


def normalized_exact_match(prediction: str, reference: str) -> bool:
    """Return the official normalized exact-match indicator."""

    return normalize_answer(prediction) == normalize_answer(reference)


def normalized_em(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Return official-style exact match on a 0--100 scale, rounded to 4 digits."""

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not references:
        raise ValueError("normalized EM is undefined for an empty sequence")
    matches = sum(
        normalized_exact_match(prediction, reference)
        for prediction, reference in zip(predictions, references)
    )
    return round(100.0 * matches / len(references), 4)


__all__ = [
    "BlobSpec",
    "DataIntegrityError",
    "INSTRUCTIONS",
    "OFFICIAL_COMMIT",
    "OFFICIAL_REPOSITORY",
    "ORDER4_TASK_NAMES",
    "ORDER4_TASKS",
    "OfficialExample",
    "RAW_BASE_URL",
    "TASK_BY_NAME",
    "TaskSpec",
    "TrainPartitions",
    "SealedEvaluationData",
    "build_prompt",
    "deterministic_per_class_cap",
    "download_blob",
    "download_official_data",
    "git_blob_sha1",
    "load_official_examples",
    "load_sealed_evaluation_data",
    "make_train_partitions",
    "normalize_answer",
    "normalized_em",
    "normalized_exact_match",
    "official_blob_specs",
    "prepare_task_partitions",
    "validate_split",
    "verify_blob",
]
