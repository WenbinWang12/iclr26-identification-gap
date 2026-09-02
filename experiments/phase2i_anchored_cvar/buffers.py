"""Byte-auditable, role-separated history buffers for Phase-2I.

The risk-train and audit roles are intentionally different types of access:
only a risk-train buffer exposes gradient sampling.  Admission and sampling use
independent deterministic random streams, so diagnostic sampling cannot change
future reservoir membership.  A canonical UTF-8 JSON representation includes
records, counters, and RNG state, making the reported byte ledger a persistent
payload measurement rather than an estimate of Python object size.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Iterable, Literal

import numpy as np


_FORMAT_VERSION = 1


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars/containers to JSON primitives."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize a payload canonically for exact persistent-byte accounting."""

    return json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class BufferRole(str, Enum):
    """A history role with an explicit optimization-access boundary."""

    RISK_TRAIN = "risk_train"
    AUDIT = "audit"


@dataclass(frozen=True)
class ControllerRecord:
    """Group-free controller view; task identity is absent by construction."""

    example_id: str
    prompt: str
    target: str
    loss_pre: float
    loss_post: float
    eligible: bool


@dataclass(frozen=True)
class AnchorRecord:
    """Canonical training-derived example and its acquisition anchors.

    ``task_id`` is evaluator/oracle-only metadata.  Group-free risk buffers
    reject non-null task IDs unless explicitly constructed with
    ``allow_task_ids=True``; their gradient sampler returns
    :class:`ControllerRecord` even when metadata is allowed for an oracle run.
    """

    example_id: str
    prompt: str
    target: str
    loss_pre: float
    loss_post: float
    eligible: bool
    task_id: str | int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id:
            raise ValueError("example_id must be a non-empty string")
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("target must be a non-empty string")
        for name, value in (("loss_pre", self.loss_pre), ("loss_post", self.loss_post)):
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.eligible, (bool, np.bool_)):
            raise TypeError("eligible must be boolean")
        if isinstance(self.task_id, bool) or (
            self.task_id is not None and not isinstance(self.task_id, (str, int))
        ):
            raise TypeError("task_id must be a string, integer, or None")

    @property
    def acquired_gain(self) -> float:
        return float(self.loss_pre) - float(self.loss_post)

    def controller_view(self) -> ControllerRecord:
        """Return the metadata-free view used by a group-free learner."""

        return ControllerRecord(
            example_id=self.example_id,
            prompt=self.prompt,
            target=self.target,
            loss_pre=float(self.loss_pre),
            loss_post=float(self.loss_post),
            eligible=bool(self.eligible),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": bool(self.eligible),
            "example_id": self.example_id,
            "loss_post": float(self.loss_post),
            "loss_pre": float(self.loss_pre),
            "prompt": self.prompt,
            "target": self.target,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnchorRecord":
        expected = {
            "eligible",
            "example_id",
            "loss_post",
            "loss_pre",
            "prompt",
            "target",
            "task_id",
        }
        if set(payload) != expected:
            raise ValueError(
                f"record fields differ from schema: expected {sorted(expected)}"
            )
        return cls(**payload)

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


@dataclass(frozen=True)
class ReservoirDecision:
    """Auditable result of one streaming admission attempt."""

    accepted: bool
    action: Literal["append", "replace", "skip_reservoir", "skip_byte_budget"]
    evicted_example_id: str | None = None


@dataclass(frozen=True)
class ByteLedger:
    """Exact canonical serialized-byte accounting for one buffer."""

    total_bytes: int
    record_bytes: int
    framing_and_metadata_bytes: int
    byte_budget: int | None
    remaining_bytes: int | None
    within_budget: bool
    record_count: int
    seen_count: int


def _stream_seed(base_seed: int, stream_id: str, purpose: str) -> int:
    material = f"phase2i\0{base_seed}\0{stream_id}\0{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def _rng_payload(rng: np.random.Generator) -> dict[str, Any]:
    """Encode PCG64 state with fixed-width integers for stable ledger length."""

    state = rng.bit_generator.state
    if state.get("bit_generator") != "PCG64":
        raise TypeError("HistoryBuffer requires numpy's PCG64 bit generator")
    core = state["state"]
    return {
        "bit_generator": "PCG64",
        "has_uint32": bool(state["has_uint32"]),
        "inc_hex": f"{int(core['inc']):032x}",
        "state_hex": f"{int(core['state']):032x}",
        "uinteger_hex": f"{int(state['uinteger']):08x}",
    }


def _restore_rng(rng: np.random.Generator, payload: dict[str, Any]) -> None:
    if payload.get("bit_generator") != "PCG64":
        raise ValueError("unsupported RNG state")
    rng.bit_generator.state = {
        "bit_generator": "PCG64",
        "state": {
            "state": int(payload["state_hex"], 16),
            "inc": int(payload["inc_hex"], 16),
        },
        "has_uint32": int(bool(payload["has_uint32"])),
        "uinteger": int(payload["uinteger_hex"], 16),
    }


class HistoryBuffer:
    """A deterministic reservoir with exact serialized-byte enforcement."""

    def __init__(
        self,
        *,
        role: BufferRole | str,
        max_records: int,
        base_seed: int,
        stream_id: str,
        byte_budget: int | None = None,
        allow_task_ids: bool | None = None,
    ) -> None:
        self.role = BufferRole(role)
        if isinstance(max_records, bool) or int(max_records) <= 0:
            raise ValueError("max_records must be a positive integer")
        if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)):
            raise TypeError("base_seed must be an integer")
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("stream_id must be a non-empty string")
        if byte_budget is not None and (
            isinstance(byte_budget, bool) or int(byte_budget) <= 0
        ):
            raise ValueError("byte_budget must be a positive integer or None")
        if allow_task_ids is None:
            allow_task_ids = self.role is BufferRole.AUDIT
        if not isinstance(allow_task_ids, bool):
            raise TypeError("allow_task_ids must be boolean")

        self.max_records = int(max_records)
        self.base_seed = int(base_seed)
        self.stream_id = stream_id
        self.byte_budget = None if byte_budget is None else int(byte_budget)
        self.allow_task_ids = allow_task_ids
        self._records: list[AnchorRecord] = []
        self._seen_count = 0
        self._skipped_reservoir = 0
        self._skipped_byte_budget = 0
        self._admission_rng = np.random.Generator(
            np.random.PCG64(_stream_seed(self.base_seed, stream_id, "admission"))
        )
        self._sampling_rng = np.random.Generator(
            np.random.PCG64(_stream_seed(self.base_seed, stream_id, "sampling"))
        )
        if self.byte_budget is not None and self.serialized_size_bytes() > self.byte_budget:
            raise ValueError("byte_budget is smaller than the empty persistent buffer")

    def __len__(self) -> int:
        return len(self._records)

    @property
    def seen_count(self) -> int:
        return self._seen_count

    def _persistent_payload(
        self, records: list[AnchorRecord] | None = None
    ) -> dict[str, Any]:
        chosen = self._records if records is None else records
        return {
            "admission_rng": _rng_payload(self._admission_rng),
            "allow_task_ids": self.allow_task_ids,
            "base_seed_hex": f"{self.base_seed % (1 << 128):032x}",
            "byte_budget": self.byte_budget,
            "format_version": _FORMAT_VERSION,
            "max_records": self.max_records,
            "records": [record.to_dict() for record in chosen],
            "role": self.role.value,
            "sampling_rng": _rng_payload(self._sampling_rng),
            "seen_count_hex": f"{self._seen_count:016x}",
            "skipped_byte_budget_hex": f"{self._skipped_byte_budget:016x}",
            "skipped_reservoir_hex": f"{self._skipped_reservoir:016x}",
            "stream_id": self.stream_id,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-safe state sufficient for bit-exact continuation."""

        return self._persistent_payload()

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> "HistoryBuffer":
        if payload.get("format_version") != _FORMAT_VERSION:
            raise ValueError("unsupported history-buffer format version")
        instance = cls(
            role=payload["role"],
            max_records=int(payload["max_records"]),
            base_seed=int(payload["base_seed_hex"], 16),
            stream_id=payload["stream_id"],
            byte_budget=payload["byte_budget"],
            allow_task_ids=bool(payload["allow_task_ids"]),
        )
        instance._records = [AnchorRecord.from_dict(item) for item in payload["records"]]
        if len(instance._records) > instance.max_records:
            raise ValueError("serialized buffer exceeds max_records")
        instance._seen_count = int(payload["seen_count_hex"], 16)
        instance._skipped_byte_budget = int(payload["skipped_byte_budget_hex"], 16)
        instance._skipped_reservoir = int(payload["skipped_reservoir_hex"], 16)
        _restore_rng(instance._admission_rng, payload["admission_rng"])
        _restore_rng(instance._sampling_rng, payload["sampling_rng"])
        instance._validate_policy()
        if instance.byte_budget is not None and (
            instance.serialized_size_bytes() > instance.byte_budget
        ):
            raise ValueError("serialized buffer exceeds its byte budget")
        return instance

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self._persistent_payload())

    def serialized_size_bytes(self, records: list[AnchorRecord] | None = None) -> int:
        return len(canonical_json_bytes(self._persistent_payload(records)))

    def byte_ledger(self) -> ByteLedger:
        record_bytes = sum(len(record.canonical_bytes()) for record in self._records)
        total = self.serialized_size_bytes()
        remaining = None if self.byte_budget is None else self.byte_budget - total
        return ByteLedger(
            total_bytes=total,
            record_bytes=record_bytes,
            framing_and_metadata_bytes=total - record_bytes,
            byte_budget=self.byte_budget,
            remaining_bytes=remaining,
            within_budget=self.byte_budget is None or total <= self.byte_budget,
            record_count=len(self),
            seen_count=self._seen_count,
        )

    def _validate_policy(self) -> None:
        ids = [record.example_id for record in self._records]
        if len(ids) != len(set(ids)):
            raise ValueError("a history buffer cannot contain duplicate example IDs")
        if not self.allow_task_ids and any(
            record.task_id is not None for record in self._records
        ):
            raise PermissionError("this group-free buffer forbids task metadata")

    def _fits(self, records: list[AnchorRecord]) -> bool:
        return self.byte_budget is None or self.serialized_size_bytes(records) <= self.byte_budget

    def offer(self, record: AnchorRecord) -> ReservoirDecision:
        """Offer one stream record using deterministic Algorithm-R admission.

        The byte budget is a hard secondary constraint.  A proposed append or
        replacement that would exceed it is rejected and explicitly counted.
        Variable-size rejection means byte-constrained membership is not an
        unbiased record reservoir; the ledger exposes such rejections.
        """

        if not isinstance(record, AnchorRecord):
            raise TypeError("record must be an AnchorRecord")
        if not self.allow_task_ids and record.task_id is not None:
            raise PermissionError("task_id is forbidden in a group-free history buffer")
        if any(existing.example_id == record.example_id for existing in self._records):
            raise ValueError(f"duplicate resident example_id: {record.example_id}")

        self._seen_count += 1
        if len(self._records) < self.max_records:
            candidate = [*self._records, record]
            if self._fits(candidate):
                self._records = candidate
                return ReservoirDecision(True, "append")
            self._skipped_byte_budget += 1
            return ReservoirDecision(False, "skip_byte_budget")

        slot = int(self._admission_rng.integers(0, self._seen_count))
        if slot >= self.max_records:
            self._skipped_reservoir += 1
            return ReservoirDecision(False, "skip_reservoir")
        candidate = list(self._records)
        evicted = candidate[slot]
        candidate[slot] = record
        if not self._fits(candidate):
            self._skipped_byte_budget += 1
            return ReservoirDecision(False, "skip_byte_budget")
        self._records = candidate
        return ReservoirDecision(True, "replace", evicted.example_id)

    def extend(self, records: Iterable[AnchorRecord]) -> tuple[ReservoirDecision, ...]:
        return tuple(self.offer(record) for record in records)

    def sample_for_gradient(
        self,
        batch_size: int,
        *,
        replace: bool = False,
        oracle: bool = False,
    ) -> tuple[ControllerRecord, ...] | tuple[AnchorRecord, ...]:
        """Sample risk-train records; audit buffers always fail closed."""

        if self.role is not BufferRole.RISK_TRAIN:
            raise PermissionError("audit records may never be sampled for gradients")
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")
        batch_size = int(batch_size)
        if not self._records:
            raise ValueError("cannot sample an empty history buffer")
        if not replace and batch_size > len(self._records):
            raise ValueError("batch_size exceeds buffer size without replacement")
        if oracle:
            if not self.allow_task_ids:
                raise PermissionError("oracle sampling was not enabled for this buffer")
            if any(record.task_id is None for record in self._records):
                raise ValueError("oracle sampling requires a task_id on every record")

        indices = self._sampling_rng.choice(
            len(self._records), size=batch_size, replace=replace
        ).tolist()
        selected = tuple(self._records[int(index)] for index in indices)
        if oracle:
            return selected
        return tuple(record.controller_view() for record in selected)

    def records_for_evaluation(self) -> tuple[AnchorRecord, ...]:
        """Return immutable records for evaluator/audit computation only."""

        return tuple(self._records)


def assert_disjoint_history(
    risk_train: HistoryBuffer, audit: HistoryBuffer
) -> None:
    """Validate role and example-ID separation of a two-buffer envelope."""

    if risk_train.role is not BufferRole.RISK_TRAIN:
        raise ValueError("risk_train must have the risk_train role")
    if audit.role is not BufferRole.AUDIT:
        raise ValueError("audit must have the audit role")
    risk_ids = {record.example_id for record in risk_train.records_for_evaluation()}
    audit_ids = {record.example_id for record in audit.records_for_evaluation()}
    overlap = sorted(risk_ids & audit_ids)
    if overlap:
        raise ValueError(f"risk-train and audit buffers overlap: {overlap[:5]}")


__all__ = [
    "AnchorRecord",
    "BufferRole",
    "ByteLedger",
    "ControllerRecord",
    "HistoryBuffer",
    "ReservoirDecision",
    "assert_disjoint_history",
    "canonical_json_bytes",
]
