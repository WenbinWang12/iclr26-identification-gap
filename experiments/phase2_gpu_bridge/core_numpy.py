"""NumPy-only core for the planted Phase-2 GPU bridge.

This module owns the protocol-visible data, oracle, sanity, bootstrap, and
identity-hash logic.  It deliberately has no dependency on PyTorch so that
local CPU checks can run in the small NumPy environment.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_SHA256 = "99893D34AB9B47057906B499843C4236A146242FCDAECBA5A019B12CCE06B55E"
STREAM_SEED = 20260723
BASE_SEED = (20260723, 0)
BOOTSTRAP_SEED = (90920260723, 0)
REGIMES = ("compatible", "stress")
RANKS = (2, 4, 8)
SPLITS = ("train", "eval")
ARMS = (
    "offline_rank_R_oracle",
    "sequential_dense_rank_R",
    "clock_balanced_replay_rank_R",
    "reservoir_random_replay_rank_R",
)
BASE_CHILD_NAMES = ("W0", "compatible_U", "compatible_V", "stress_U", "stress_V")
STREAM_CHILD_NAMES = (
    "train_signs",
    "eval_signs",
    "train_permutations",
    "eval_permutations",
    "clock_insert",
    "reservoir_priority",
    "clock_replay_draws",
    "reservoir_replay_draws",
    "factor_init",
)
PRIMARY_SEEDS = (
    53,
    71,
    89,
    107,
    131,
    149,
    167,
    191,
    211,
    233,
    257,
    277,
    307,
    331,
    353,
    379,
    401,
    431,
    457,
    487,
)
CALIBRATION_SEEDS = (11, 23, 37)


def _as_json(value: Any) -> Any:
    """Convert NumPy/dataclass values to deterministic JSON-compatible data."""

    if dataclasses.is_dataclass(value):
        return _as_json(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return [_as_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _as_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_as_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the protocol's sorted-key, UTF-8 canonical JSON representation."""

    return json.dumps(
        _as_json(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _little_endian_array(array: np.ndarray) -> tuple[np.ndarray, np.dtype]:
    value = np.asarray(array)
    dtype = np.dtype(value.dtype).newbyteorder("<")
    return np.ascontiguousarray(value, dtype=dtype), dtype


def update_array_hash(digest: "hashlib._Hash", tag: str, array: np.ndarray) -> None:
    """Hash an array with an explicit tag, shape, little-endian dtype, and C order."""

    value, dtype = _little_endian_array(np.asarray(array))
    header = {
        "tag": str(tag),
        "dtype": dtype.str,
        "shape": list(value.shape),
        "order": "C",
    }
    digest.update(canonical_json_bytes(header))
    digest.update(value.tobytes(order="C"))


def hash_named_arrays(named_arrays: Iterable[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for tag, array in named_arrays:
        update_array_hash(digest, tag, array)
    return digest.hexdigest().upper()


def protocol_hash(path: str | Path) -> str:
    return sha256_file(path)


def assert_protocol_hash(path: str | Path, expected: str = PROTOCOL_SHA256) -> str:
    observed = protocol_hash(path)
    if observed != expected.upper():
        raise ValueError(f"protocol hash mismatch: expected {expected}, observed {observed}")
    return observed


@dataclass(frozen=True)
class BridgeConfig:
    """Frozen protocol constants.

    The dataclass accepts explicit values for small non-citable unit probes, but
    ``is_protocol_locked`` must be true before a primary/confirmation run.
    """

    d_in: int = 32
    d_out: int = 32
    T: int = 12
    ranks: tuple[int, ...] = RANKS
    n_train: int = 1024
    n_eval: int = 2048
    replay_capacity: int = 352
    batch_size: int = 128
    replay_batch: int = 64
    updates_per_window: int = 100
    stream_seed: int = STREAM_SEED
    base_seed: tuple[int, int] = BASE_SEED
    bootstrap_seed: tuple[int, int] = BOOTSTRAP_SEED
    bootstrap_draws: int = 10000
    lr: float = 3e-2
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip: float = 5.0
    sanity_steps: int = 512
    sanity_step_size: float = 0.5
    protocol_sha256: str = PROTOCOL_SHA256
    primary_seeds: tuple[int, ...] = PRIMARY_SEEDS
    calibration_seeds: tuple[int, ...] = CALIBRATION_SEEDS

    def __post_init__(self) -> None:
        if self.d_in != 32 or self.d_out != 32:
            raise ValueError("Phase-2 uses d_in=d_out=32")
        if self.T <= 0 or self.n_train <= 0 or self.n_eval <= 0:
            raise ValueError("dimensions, T, and sample counts must be positive")
        if tuple(self.ranks) != tuple(sorted(self.ranks)) or any(rank <= 0 for rank in self.ranks):
            raise ValueError("ranks must be a sorted tuple of positive integers")
        if self.n_train % 2 or self.n_eval % 2:
            raise ValueError("all frozen sign blocks must have even counts")
        if self.batch_size != 128 or self.replay_batch != 64:
            raise ValueError("frozen batch composition is 128 current / 64 replay")
        if self.replay_capacity != 352:
            raise ValueError("frozen replay capacity is 352")
        if self.updates_per_window != 100:
            raise ValueError("frozen update count is 100 per window")
        if self.bootstrap_draws <= 0:
            raise ValueError("bootstrap_draws must be positive")

    @property
    def is_protocol_locked(self) -> bool:
        return self == type(self)()

    @property
    def fixed_replay_bytes(self) -> int:
        return (
            self.replay_capacity * self.d_in * np.dtype("<f4").itemsize * 2
            + self.replay_capacity * np.dtype("<f8").itemsize
            + self.replay_capacity * np.dtype("u1").itemsize
            + 2 * self.T * np.dtype("<u8").itemsize
            + 2 * np.dtype("<u8").itemsize
        )

    def payload(self) -> dict[str, Any]:
        return {
            **_as_json(dataclasses.asdict(self)),
            "base_child_names": list(BASE_CHILD_NAMES),
            "stream_child_names": list(STREAM_CHILD_NAMES),
            "rng": {"bit_generator": "PCG64", "normal_order": "C", "uniform_dtype": "<f8"},
            "hash_serialization": {"arrays": "little-endian contiguous C", "json": "sorted UTF-8"},
            "adamw": {
                "amsgrad": False,
                "foreach": False,
                "fused": False,
                "maximize": False,
                "capturable": False,
                "differentiable": False,
            },
        }

    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


@dataclass(frozen=True)
class BaseDraws:
    W0: np.ndarray
    compatible_U: np.ndarray
    compatible_V: np.ndarray
    stress_U: np.ndarray
    stress_V: np.ndarray

    def __post_init__(self) -> None:
        for name in BASE_CHILD_NAMES:
            array = getattr(self, name)
            if np.asarray(array).dtype != np.dtype("<f8"):
                raise ValueError(f"{name} must be float64")
            if not np.asarray(array).flags.c_contiguous:
                raise ValueError(f"{name} must be C contiguous")

    def basis(self, regime: str, rank: int) -> tuple[np.ndarray, np.ndarray]:
        if regime == "compatible":
            if rank not in RANKS:
                raise ValueError(f"unsupported rank: {rank}")
            return self.compatible_U[:, :rank], self.compatible_V[:, :rank]
        if regime == "stress":
            if rank not in RANKS:
                raise ValueError(f"unsupported rank: {rank}")
            return self.stress_U, self.stress_V
        raise ValueError(f"unsupported regime: {regime}")

    def sha256(self) -> str:
        return hash_named_arrays((name, getattr(self, name)) for name in BASE_CHILD_NAMES)


@dataclass
class SeedStreams:
    seed: int
    train_signs: np.random.Generator
    eval_signs: np.random.Generator
    train_permutations: np.random.Generator
    eval_permutations: np.random.Generator
    clock_insert: np.random.Generator
    reservoir_priority: np.random.Generator
    clock_replay_draws: np.random.Generator
    reservoir_replay_draws: np.random.Generator
    factor_init: np.random.Generator

    def named(self) -> dict[str, np.random.Generator]:
        return {name: getattr(self, name) for name in STREAM_CHILD_NAMES}


@dataclass(frozen=True)
class FactorInit:
    """One deterministic, shared float32 LoRA initialization for a cell.

    ``G64`` is retained so the exact random draw can be included in provenance
    hashes.  Training consumes ``A32`` and ``B32``; the latter is all zeros.
    """

    regime: str
    rank: int
    G64: np.ndarray
    A64: np.ndarray
    A32: np.ndarray
    B32: np.ndarray

    def __post_init__(self) -> None:
        if self.regime not in REGIMES or self.rank not in RANKS:
            raise ValueError("factor initialization has an unsupported cell")
        rank = int(self.rank)
        if self.G64.shape != (rank, 32) or self.A64.shape != (rank, 32):
            raise ValueError("float64 factor arrays must have shape (R,32)")
        if self.A32.shape != (rank, 32) or self.B32.shape != (32, rank):
            raise ValueError("deployed factor arrays have incompatible shapes")
        if self.G64.dtype != np.dtype("<f8") or self.A64.dtype != np.dtype("<f8"):
            raise ValueError("G64 and A64 must be little-endian float64")
        if self.A32.dtype != np.dtype("<f4") or self.B32.dtype != np.dtype("<f4"):
            raise ValueError("A32 and B32 must be little-endian float32")
        if not np.all(np.isfinite(self.G64)) or not np.all(np.isfinite(self.A64)):
            raise ValueError("factor initialization contains non-finite values")
        if not np.array_equal(self.B32, np.zeros_like(self.B32)):
            raise ValueError("B32 must be initialized to zeros")

    def sha256(self) -> str:
        return hash_named_arrays(
            (
                ("G64", self.G64),
                ("A64", self.A64),
                ("A32", self.A32),
                ("B32", self.B32),
            )
        )


def _freeze(array: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(array)
    value.setflags(write=False)
    return value


def _canonicalize_columns(matrix: np.ndarray) -> np.ndarray:
    result = np.array(matrix, dtype="<f8", order="C", copy=True)
    for column in range(result.shape[1]):
        row = int(np.argmax(np.abs(result[:, column])))
        if result[row, column] < 0.0:
            result[:, column] *= -1.0
    return _freeze(result)


def _canonicalize_rows(matrix: np.ndarray) -> np.ndarray:
    result = np.array(matrix, dtype="<f8", order="C", copy=True)
    for row in range(result.shape[0]):
        column = int(np.argmax(np.abs(result[row, :])))
        if result[row, column] < 0.0:
            result[row, :] *= -1.0
    return _freeze(result)


def make_base_draws(config: BridgeConfig | None = None) -> BaseDraws:
    config = config or BridgeConfig()
    children = np.random.SeedSequence(list(config.base_seed)).spawn(5)
    generators = [np.random.Generator(np.random.PCG64(child)) for child in children]
    w0 = generators[0].standard_normal((config.d_out, config.d_in), dtype=np.float64) / math.sqrt(config.d_in)
    comp_u, _ = np.linalg.qr(
        generators[1].standard_normal((config.d_out, 8), dtype=np.float64), mode="reduced"
    )
    comp_v, _ = np.linalg.qr(
        generators[2].standard_normal((config.d_in, 8), dtype=np.float64), mode="reduced"
    )
    stress_u, _ = np.linalg.qr(
        generators[3].standard_normal((config.d_out, 12), dtype=np.float64), mode="reduced"
    )
    stress_v, _ = np.linalg.qr(
        generators[4].standard_normal((config.d_in, 12), dtype=np.float64), mode="reduced"
    )
    return BaseDraws(
        _freeze(np.asarray(w0, dtype="<f8", order="C")),
        _canonicalize_columns(comp_u),
        _canonicalize_columns(comp_v),
        _canonicalize_columns(stress_u),
        _canonicalize_columns(stress_v),
    )


def make_seed_streams(seed: int, config: BridgeConfig | None = None) -> SeedStreams:
    config = config or BridgeConfig()
    children = np.random.SeedSequence([config.stream_seed, int(seed)]).spawn(9)
    generators = [np.random.Generator(np.random.PCG64(child)) for child in children]
    return SeedStreams(int(seed), *generators)


def make_factor_initializations(
    streams: SeedStreams, config: BridgeConfig | None = None
) -> dict[tuple[str, int], FactorInit]:
    """Consume the factor child in the frozen compatible-then-stress order."""

    config = config or BridgeConfig()
    generator = streams.factor_init
    initializations: dict[tuple[str, int], FactorInit] = {}
    for regime in REGIMES:
        for rank in RANKS:
            draw = np.asarray(
                generator.standard_normal((rank, config.d_in), dtype=np.float64),
                dtype="<f8",
                order="C",
            )
            q, _ = np.linalg.qr(draw.T, mode="reduced")
            a64 = _canonicalize_rows(q.T)
            a32 = _freeze(np.asarray(a64, dtype="<f4", order="C"))
            b32 = _freeze(np.zeros((config.d_out, rank), dtype="<f4", order="C"))
            initializations[(regime, rank)] = FactorInit(
                regime,
                int(rank),
                _freeze(draw),
                a64,
                a32,
                b32,
            )
    return initializations


def factor_init(
    streams: SeedStreams,
    regime: str,
    rank: int,
    config: BridgeConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Consume one frozen factor draw and return ``(B0, A0)`` in float64.

    The child stream is consumed in compatible ``R=2,4,8`` order followed by
    stress ``R=2,4,8`` order by the caller.  The same result is shared by all
    three trainable arms for a cell; the model casts it once to float32.
    """

    config = config or BridgeConfig()
    if regime not in REGIMES or rank not in RANKS:
        raise ValueError("unsupported factor initialization cell")
    if rank > config.d_in or rank > config.d_out:
        raise ValueError("rank exceeds factor dimensions")
    generator = streams.factor_init
    matrix = generator.standard_normal((rank, config.d_in), dtype=np.float64)
    q, _ = np.linalg.qr(matrix.T, mode="reduced")
    a0 = _canonicalize_rows(q.T)
    b0 = _freeze(np.zeros((config.d_out, rank), dtype="<f8", order="C"))
    return b0, a0


def mode_count(regime: str, rank: int) -> int:
    if regime == "compatible":
        return int(rank)
    if regime == "stress":
        return 12
    raise ValueError(f"unsupported regime: {regime}")


def mode_schedule(regime: str, rank: int, window: int) -> tuple[int, int | None]:
    if window < 1:
        raise ValueError("window is one-based")
    count = mode_count(regime, rank)
    current = (window - 1) % count
    previous = None if window == 1 else (window - 2) % count
    return current, previous


def alpha_values(regime: str, rank: int) -> np.ndarray:
    count = mode_count(regime, rank)
    return np.asarray([1.0 + 0.02 * (count - (index + 1)) for index in range(count)], dtype="<f8")


def _mode_counts(config: BridgeConfig, regime: str, rank: int, window: int, split: str) -> np.ndarray:
    count = mode_count(regime, rank)
    result = np.zeros(count, dtype=np.int64)
    current, previous = mode_schedule(regime, rank, window)
    total = config.n_train if split == "train" else config.n_eval
    if window == 1:
        result[current] = total
    else:
        result[current] = (3 * total) // 4
        result[int(previous)] = total // 4
    return result


def _balanced_signs(rng: np.random.Generator, count: int) -> np.ndarray:
    if count <= 0 or count % 2:
        raise ValueError("sign block count must be a positive even integer")
    base = np.concatenate(
        (np.ones(count // 2, dtype=np.int8), -np.ones(count // 2, dtype=np.int8))
    )
    return _freeze(base[rng.permutation(count)])


def _record_key(regime: str, rank: int, window: int, split: str, local_index: int) -> tuple[int, int, int, int, int]:
    regime_code = 0 if regime == "compatible" else 1
    rank_code = {2: 0, 4: 1, 8: 2}[int(rank)]
    split_code = 0 if split == "train" else 1
    return regime_code, rank_code, int(window), split_code, int(local_index)


@dataclass(frozen=True)
class RecordBlock:
    """Canonical mode-block records plus the one frozen storage permutation."""

    x64: np.ndarray
    y64: np.ndarray
    x32: np.ndarray
    y32: np.ndarray
    mode_ids: np.ndarray
    signs: np.ndarray
    record_ids: np.ndarray
    global_keys: np.ndarray
    storage_permutation: np.ndarray

    def __post_init__(self) -> None:
        n = int(self.x64.shape[0])
        if self.x64.shape != self.y64.shape or self.x64.shape[1] != 32:
            raise ValueError("record x/y must have shape [n,32]")
        if self.x32.shape != self.x64.shape or self.y32.shape != self.x64.shape:
            raise ValueError("float32 casts must preserve record shape")
        if any(array.shape[0] != n for array in (self.mode_ids, self.signs, self.record_ids, self.global_keys, self.storage_permutation)):
            raise ValueError("record metadata length mismatch")

    @property
    def count(self) -> int:
        return int(self.x64.shape[0])

    def learner_view(self, order: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        indices = self.storage_permutation if order is None else np.asarray(order, dtype=np.int64)
        return self.x32[indices], self.y32[indices]


@dataclass(frozen=True)
class WindowRecords:
    window: int
    train: RecordBlock
    eval: RecordBlock
    pi_t: np.ndarray


@dataclass(frozen=True)
class SeedDataset:
    seed: int
    cells: Mapping[tuple[str, int], tuple[WindowRecords, ...]]
    data_sha256: str
    split_sha256: str

    def cell(self, regime: str, rank: int) -> tuple[WindowRecords, ...]:
        return self.cells[(regime, int(rank))]


def _build_block(
    base: BaseDraws,
    config: BridgeConfig,
    streams: SeedStreams,
    regime: str,
    rank: int,
    window: int,
    split: str,
) -> RecordBlock:
    U, V = base.basis(regime, rank)
    alpha = alpha_values(regime, rank)
    counts = _mode_counts(config, regime, rank, window, split)
    sign_rng = streams.train_signs if split == "train" else streams.eval_signs
    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    chunks_modes: list[np.ndarray] = []
    chunks_signs: list[np.ndarray] = []
    for mode, count in enumerate(counts.tolist()):
        if count == 0:
            continue
        signs = _balanced_signs(sign_rng, int(count))
        x = signs.astype("<f8")[:, None] * V[:, mode][None, :]
        y = base.W0 @ x.T
        y = (y + alpha[mode] * signs.astype("<f8")[None, :] * U[:, mode][:, None]).T
        chunks_x.append(np.asarray(x, dtype="<f8", order="C"))
        chunks_y.append(np.asarray(y, dtype="<f8", order="C"))
        chunks_modes.append(np.full(count, mode, dtype=np.int64))
        chunks_signs.append(np.asarray(signs, dtype=np.int8))
    x64 = np.concatenate(chunks_x, axis=0)
    y64 = np.concatenate(chunks_y, axis=0)
    mode_ids = np.concatenate(chunks_modes, axis=0)
    signs = np.concatenate(chunks_signs, axis=0)
    n = int(x64.shape[0])
    record_ids = np.arange(n, dtype=np.int64)
    split_rng = streams.train_permutations if split == "train" else streams.eval_permutations
    permutation = np.asarray(split_rng.permutation(n), dtype=np.int64)
    keys = np.asarray(
        [_record_key(regime, rank, window, split, int(local)) for local in record_ids], dtype=np.int64
    )
    return RecordBlock(
        _freeze(x64),
        _freeze(y64),
        _freeze(np.asarray(x64, dtype="<f4", order="C")),
        _freeze(np.asarray(y64, dtype="<f4", order="C")),
        _freeze(mode_ids),
        _freeze(signs),
        _freeze(record_ids),
        _freeze(keys),
        _freeze(permutation),
    )


def _dataset_hashes(cells: Mapping[tuple[str, int], tuple[WindowRecords, ...]]) -> tuple[str, str]:
    data_digest = hashlib.sha256()
    split_digest = hashlib.sha256()
    for regime in REGIMES:
        for rank in RANKS:
            for window in cells[(regime, rank)]:
                for split, block in (("train", window.train), ("eval", window.eval)):
                    prefix = f"{regime}:{rank}:{window.window}:{split}"
                    for tag, array in (
                        ("x64", block.x64),
                        ("y64", block.y64),
                        ("x32", block.x32),
                        ("y32", block.y32),
                        ("mode_ids", block.mode_ids),
                        ("signs", block.signs),
                        ("record_ids", block.record_ids),
                        ("global_keys", block.global_keys),
                        ("storage_permutation", block.storage_permutation),
                    ):
                        update_array_hash(data_digest, f"{prefix}:{tag}", array)
                    for tag, array in (
                        ("mode_ids", block.mode_ids),
                        ("signs", block.signs),
                        ("record_ids", block.record_ids),
                        ("global_keys", block.global_keys),
                        ("storage_permutation", block.storage_permutation),
                    ):
                        update_array_hash(split_digest, f"{prefix}:{tag}", array)
                update_array_hash(data_digest, f"{regime}:{rank}:{window.window}:pi_t", window.pi_t)
                update_array_hash(split_digest, f"{regime}:{rank}:{window.window}:pi_t", window.pi_t)
    return data_digest.hexdigest().upper(), split_digest.hexdigest().upper()


def build_seed_dataset(
    base: BaseDraws, seed: int, config: BridgeConfig | None = None, streams: SeedStreams | None = None
) -> SeedDataset:
    config = config or BridgeConfig()
    streams = streams or make_seed_streams(seed, config)
    if streams.seed != int(seed):
        raise ValueError("streams.seed does not match requested dataset seed")
    cells: dict[tuple[str, int], tuple[WindowRecords, ...]] = {}
    # The nested order is part of the protocol's RNG contract.
    for regime in REGIMES:
        for rank in RANKS:
            windows: list[WindowRecords] = []
            for window in range(1, config.T + 1):
                train = _build_block(base, config, streams, regime, rank, window, "train")
                evaluation = _build_block(base, config, streams, regime, rank, window, "eval")
                windows.append(WindowRecords(window, train, evaluation, _freeze(train.storage_permutation.copy())))
            cells[(regime, rank)] = tuple(windows)
    data_hash, split_hash = _dataset_hashes(cells)
    return SeedDataset(int(seed), cells, data_hash, split_hash)


def build_all_seed_datasets(
    base: BaseDraws, seeds: Sequence[int], config: BridgeConfig | None = None
) -> tuple[SeedDataset, ...]:
    config = config or BridgeConfig()
    return tuple(build_seed_dataset(base, int(seed), config) for seed in seeds)


def prefix_mode_weights(regime: str, rank: int, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("window is one-based")
    count = mode_count(regime, rank)
    counts = np.zeros(count, dtype=np.float64)
    for current_window in range(1, window + 1):
        current, previous = mode_schedule(regime, rank, current_window)
        if current_window == 1:
            counts[current] += 1.0
        else:
            counts[current] += 0.75
            counts[int(previous)] += 0.25
    return np.asarray(counts / float(window), dtype="<f8")


def current_mode_weights(regime: str, rank: int, window: int) -> np.ndarray:
    count = mode_count(regime, rank)
    weights = np.zeros(count, dtype=np.float64)
    current, previous = mode_schedule(regime, rank, window)
    if window == 1:
        weights[current] = 1.0
    else:
        weights[current] = 0.75
        weights[int(previous)] = 0.25
    return np.asarray(weights, dtype="<f8")


def modal_matrix(U: np.ndarray, V: np.ndarray, alpha: np.ndarray, indices: Iterable[int] | None = None) -> np.ndarray:
    selected = range(len(alpha)) if indices is None else tuple(int(index) for index in indices)
    result = np.zeros((U.shape[0], V.shape[0]), dtype=np.float64)
    for index in selected:
        result += float(alpha[index]) * np.outer(U[:, index], V[:, index])
    return np.asarray(result, dtype="<f8", order="C")


def population_risk(
    delta: np.ndarray, U: np.ndarray, V: np.ndarray, alpha: np.ndarray, weights: np.ndarray
) -> float:
    delta = np.asarray(delta, dtype=np.float64)
    residual = delta @ V - U * np.asarray(alpha, dtype=np.float64)[None, :]
    return float(0.5 * np.sum(np.asarray(weights, dtype=np.float64) * np.sum(residual * residual, axis=0)))


def weighted_modal_indices(alpha: np.ndarray, weights: np.ndarray, rank: int) -> tuple[int, ...]:
    positive = [index for index, weight in enumerate(np.asarray(weights)) if float(weight) > 0.0]
    scores = np.asarray(alpha, dtype=np.float64) ** 2 * np.asarray(weights, dtype=np.float64)
    return tuple(sorted(positive, key=lambda index: (-float(scores[index]), int(index)))[: min(rank, len(positive))])


def weighted_modal_svd(
    U: np.ndarray,
    V: np.ndarray,
    alpha: np.ndarray,
    weights: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray]:
    """Return the weighted modal SVD reference and its canonical refit.

    The SVD is taken on ``Y=sum_j alpha_j*sqrt(q_j) u_j v_j^T``.  Its
    singular values only select directions; retained modes are refit to their
    planted amplitudes, as required by the population-risk oracle.
    """

    alpha_array = np.asarray(alpha, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    positive = np.flatnonzero(weight_array > 0.0)
    weighted = np.zeros((U.shape[0], V.shape[0]), dtype=np.float64)
    for index in positive.tolist():
        weighted += (
            float(alpha_array[index])
            * math.sqrt(float(weight_array[index]))
            * np.outer(U[:, index], V[:, index])
        )
    left, singular_values, right_t = np.linalg.svd(weighted, full_matrices=False)
    if not (
        np.all(np.isfinite(left))
        and np.all(np.isfinite(singular_values))
        and np.all(np.isfinite(right_t))
    ):
        raise FloatingPointError("weighted modal SVD returned non-finite values")

    scale = max(1.0, float(np.linalg.norm(weighted, ord="fro")))
    reconstruction = (left * singular_values[None, :]) @ right_t
    if float(np.linalg.norm(reconstruction - weighted, ord="fro")) > 1e-11 * scale:
        raise AssertionError("weighted modal SVD does not reconstruct the weighted matrix")

    # Recover the canonical modal coefficients from the numerical matrix rather
    # than reusing the analytic alpha^2*q ordering.  This also exposes any
    # off-modal numerical contamination in the independent SVD construction.
    canonical_matrix = U.T @ weighted @ V
    canonical_coefficients = np.abs(np.diag(canonical_matrix))
    off_modal = canonical_matrix - np.diag(np.diag(canonical_matrix))
    if float(np.linalg.norm(off_modal, ord="fro")) > 1e-11 * scale:
        raise AssertionError("weighted matrix is not diagonal in the planted modal basis")

    positive_coefficients = canonical_coefficients[positive]
    expected_spectrum = np.sort(positive_coefficients)[::-1]
    observed_spectrum = singular_values[: positive.size]
    if not np.allclose(observed_spectrum, expected_spectrum, rtol=1e-10, atol=1e-10):
        raise AssertionError("weighted modal SVD spectrum disagrees with canonical projections")

    retained = min(int(rank), int(positive.size))
    if retained == 0:
        selected = ()
    elif retained == int(positive.size):
        selected = tuple(
            sorted(positive.tolist(), key=lambda index: (-float(canonical_coefficients[index]), int(index)))
        )
    else:
        cutoff = float(singular_values[retained - 1])
        tie_tolerance = 1e-10 * max(1.0, float(singular_values[0]))
        above = [
            int(index)
            for index in positive.tolist()
            if float(canonical_coefficients[index]) > cutoff + tie_tolerance
        ]
        tied = [
            int(index)
            for index in positive.tolist()
            if abs(float(canonical_coefficients[index]) - cutoff) <= tie_tolerance
        ]

        # Map the complete numerical degenerate singular subspace back to the
        # planted canonical pairs before applying the frozen lower-index rule.
        tie_positions = np.flatnonzero(np.abs(singular_values - cutoff) <= tie_tolerance)
        left_tie = left[:, tie_positions]
        right_tie = right_t[tie_positions, :]
        mapped_ties: list[int] = []
        for index in tied:
            left_projection = float(np.sum((left_tie.T @ U[:, index]) ** 2))
            right_projection = float(np.sum((right_tie @ V[:, index]) ** 2))
            if left_projection * right_projection < 1.0 - 1e-8:
                raise AssertionError("degenerate SVD subspace did not map to a canonical modal pair")
            mapped_ties.append(index)
        above.sort(key=lambda index: (-float(canonical_coefficients[index]), int(index)))
        mapped_ties.sort()
        selected = tuple(above + mapped_ties[: retained - len(above)])
        if len(selected) != retained:
            raise AssertionError("weighted modal SVD did not recover the requested canonical rank")

    refit = modal_matrix(U, V, alpha_array, selected)
    return np.asarray(refit, dtype="<f8", order="C"), selected, np.asarray(singular_values, dtype="<f8")


@dataclass(frozen=True)
class OraclePrefix:
    regime: str
    rank: int
    window: int
    q: np.ndarray
    p: np.ndarray
    alpha: np.ndarray
    selected: tuple[int, ...]
    delta_inf: np.ndarray
    delta_rank: np.ndarray
    loss_inf: float
    loss_rank: float
    capacity_tail: float
    D_t: float
    D_current: float


def build_prefix_oracle(base: BaseDraws, regime: str, rank: int, window: int) -> OraclePrefix:
    U, V = base.basis(regime, rank)
    alpha = alpha_values(regime, rank)
    q = prefix_mode_weights(regime, rank, window)
    p = current_mode_weights(regime, rank, window)
    selected = weighted_modal_indices(alpha, q, rank)
    delta_inf = modal_matrix(U, V, alpha, np.flatnonzero(q > 0.0))
    delta_rank = modal_matrix(U, V, alpha, selected)
    zero = np.zeros_like(delta_inf)
    loss_inf = population_risk(delta_inf, U, V, alpha, q)
    loss_rank = population_risk(delta_rank, U, V, alpha, q)
    capacity_tail = 0.5 * float(np.sum(q[[j for j in range(len(q)) if j not in selected]] * alpha[[j for j in range(len(q)) if j not in selected]] ** 2))
    D_t = population_risk(zero, U, V, alpha, q) - loss_inf
    D_current = population_risk(zero, U, V, alpha, p) - population_risk(delta_inf, U, V, alpha, p)
    return OraclePrefix(
        regime,
        int(rank),
        int(window),
        _freeze(q),
        _freeze(p),
        _freeze(alpha),
        selected,
        _freeze(delta_inf),
        _freeze(delta_rank),
        float(loss_inf),
        float(loss_rank),
        float(capacity_tail),
        float(D_t),
        float(D_current),
    )


def build_oracles(base: BaseDraws, config: BridgeConfig | None = None) -> dict[tuple[str, int, int], OraclePrefix]:
    config = config or BridgeConfig()
    return {
        (regime, rank, window): build_prefix_oracle(base, regime, rank, window)
        for regime in REGIMES
        for rank in RANKS
        for window in range(1, config.T + 1)
    }


@dataclass(frozen=True)
class SanityPrefix:
    oracle: OraclePrefix
    final_delta: np.ndarray
    pre_selected: tuple[tuple[int, ...], ...]
    post_support: tuple[tuple[int, ...], ...]
    residual_norms: tuple[float, ...]
    taus: tuple[float, ...]
    final_normalized_error: float
    q_min_keep: float


def _population_gradient(delta: np.ndarray, U: np.ndarray, V: np.ndarray, alpha: np.ndarray, weights: np.ndarray) -> np.ndarray:
    residual = delta @ V - U * alpha[None, :]
    return np.asarray((residual * weights[None, :]) @ V.T, dtype=np.float64)


def _modal_coefficients(delta: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    return np.asarray(np.diag(U.T @ delta @ V), dtype=np.float64)


def _modal_truncate(delta: np.ndarray, U: np.ndarray, V: np.ndarray, weights: np.ndarray, rank: int) -> tuple[np.ndarray, tuple[int, ...]]:
    # Validate the full numerical SVD, then express its projection in the
    # canonical planted modal basis so a cutoff tie cannot inherit an arbitrary
    # backend-specific singular-vector rotation.
    coefficients = _modal_coefficients(delta, U, V)
    positive = [index for index, weight in enumerate(weights) if float(weight) > 0.0]
    selected = tuple(sorted(positive, key=lambda index: (-abs(float(coefficients[index])), int(index)))[: min(rank, len(positive))])
    left, singular_values, right = np.linalg.svd(
        np.asarray(delta, dtype=np.float64), full_matrices=False
    )
    scale = max(1.0, float(np.linalg.norm(delta, ord="fro")))
    reconstruction = (left * singular_values[None, :]) @ right
    if float(np.linalg.norm(reconstruction - delta, ord="fro")) > 1e-11 * scale:
        raise AssertionError("sanity SVD does not reconstruct its input")
    modal_part = modal_matrix(U, V, coefficients)
    if float(np.linalg.norm(delta - modal_part, ord="fro")) > 1e-11 * scale:
        raise AssertionError("sanity iterate left the planted modal span")
    expected_spectrum = np.sort(np.abs(coefficients[positive]))[::-1]
    if not np.allclose(
        singular_values[: len(expected_spectrum)],
        expected_spectrum,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise AssertionError("sanity SVD spectrum disagrees with canonical modal coefficients")
    result = modal_matrix(U, V, coefficients, selected)
    return np.asarray(result, dtype="<f8", order="C"), selected


def run_sanity_projection(
    base: BaseDraws,
    regime: str,
    rank: int,
    window: int,
    config: BridgeConfig | None = None,
) -> SanityPrefix:
    config = config or BridgeConfig()
    oracle = build_prefix_oracle(base, regime, rank, window)
    U, V = base.basis(regime, rank)
    delta = np.zeros((config.d_out, config.d_in), dtype=np.float64)
    pre_selected: list[tuple[int, ...]] = []
    post_support: list[tuple[int, ...]] = []
    residual_norms: list[float] = []
    taus: list[float] = []
    for _ in range(config.sanity_steps):
        delta_pre = delta - config.sanity_step_size * _population_gradient(delta, U, V, oracle.alpha, oracle.q)
        delta, selected = _modal_truncate(delta_pre, U, V, oracle.q, rank)
        coefficients = _modal_coefficients(delta, U, V)
        modal_part = np.zeros_like(delta)
        for index, coefficient in enumerate(coefficients):
            modal_part += float(coefficient) * np.outer(U[:, index], V[:, index])
        residual = delta - modal_part
        tau = 1e-12 * max(1.0, float(np.linalg.norm(delta, ord="fro")))
        support = tuple(index for index, weight in enumerate(oracle.q) if weight > 0.0 and abs(coefficients[index]) > tau)
        pre_selected.append(selected)
        post_support.append(support)
        residual_norms.append(float(np.linalg.norm(residual, ord="fro")))
        taus.append(float(tau))
    final_error = (population_risk(delta, U, V, oracle.alpha, oracle.q) - oracle.loss_rank) / oracle.D_t
    q_min_keep = min((float(oracle.q[index]) for index in oracle.selected), default=0.0)
    return SanityPrefix(
        oracle,
        _freeze(delta),
        tuple(pre_selected),
        tuple(post_support),
        tuple(residual_norms),
        tuple(taus),
        float(final_error),
        float(q_min_keep),
    )


def run_all_sanity(base: BaseDraws, config: BridgeConfig | None = None) -> dict[tuple[str, int, int], SanityPrefix]:
    config = config or BridgeConfig()
    return {
        (regime, rank, window): run_sanity_projection(base, regime, rank, window, config)
        for regime in REGIMES
        for rank in RANKS
        for window in range(1, config.T + 1)
    }


def q_min_keep(sanity: Mapping[tuple[str, int, int], SanityPrefix]) -> float:
    values = [result.q_min_keep for result in sanity.values() if result.q_min_keep > 0.0]
    return min(values) if values else 0.0


def sanity_contraction_bound(q_min: float, steps: int = 512, step_size: float = 0.5) -> float:
    return float((1.0 - step_size * q_min) ** (2 * steps))


def bootstrap_index_matrices(config: BridgeConfig | None = None, valid_count: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    config = config or BridgeConfig()
    children = np.random.SeedSequence(list(config.bootstrap_seed)).spawn(2)
    all_rng = np.random.Generator(np.random.PCG64(children[0]))
    valid_rng = np.random.Generator(np.random.PCG64(children[1]))
    all_indices = all_rng.integers(
        0,
        20,
        size=(config.bootstrap_draws, 20),
        dtype=np.int64,
        endpoint=False,
    )
    valid_indices = None
    if valid_count is not None and valid_count > 0:
        valid_indices = valid_rng.integers(
            0,
            int(valid_count),
            size=(config.bootstrap_draws, int(valid_count)),
            dtype=np.int64,
            endpoint=False,
        )
    return _freeze(all_indices), None if valid_indices is None else _freeze(valid_indices)


def bootstrap_mean_replicates(values: Sequence[float], indices: np.ndarray) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.shape != (20,):
        raise ValueError("all-seed bootstrap statistics require exactly 20 values")
    index_array = np.asarray(indices, dtype=np.int64)
    if index_array.ndim != 2 or index_array.shape[1] != 20:
        raise ValueError("all-seed index matrix must have shape [draws,20]")
    return np.mean(values_array[index_array], axis=1)


def bootstrap_gap_fraction_replicates(values: Sequence[float], indices: np.ndarray) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    index_array = np.asarray(indices, dtype=np.int64)
    if values_array.ndim != 1 or values_array.size == 0:
        raise ValueError("valid gap values must be non-empty")
    if index_array.ndim != 2 or index_array.shape[1] != values_array.size:
        raise ValueError("valid-gap index matrix must have shape [draws,m]")
    return np.median(values_array[index_array], axis=1)


def one_sided_lower_quantile(replicates: Sequence[float], quantile: float = 0.05) -> float:
    return float(np.quantile(np.asarray(replicates, dtype=np.float64), quantile, method="linear"))


def bootstrap_summary(
    all_seed_values: Mapping[str, Sequence[float]], valid_gap_values: Sequence[float] | None = None, config: BridgeConfig | None = None
) -> dict[str, Any]:
    config = config or BridgeConfig()
    all_indices, valid_indices = bootstrap_index_matrices(config, None if valid_gap_values is None else len(valid_gap_values))
    all_results: dict[str, Any] = {}
    for name, values in all_seed_values.items():
        replicates = bootstrap_mean_replicates(values, all_indices)
        all_results[str(name)] = {
            "seed_values": [float(value) for value in values],
            "mean": float(np.mean(np.asarray(values, dtype=np.float64))),
            "lower_bound_05": one_sided_lower_quantile(replicates),
            "draws": int(config.bootstrap_draws),
            "sample_size": 20,
            "quantile": 0.05,
            "quantile_method": "linear",
            "resampling_unit": "paired primary seeds",
        }
    if valid_gap_values is None or len(valid_gap_values) == 0:
        all_results["gap_fraction"] = {
            "valid_count": 0,
            "draws": 0,
            "statistic": None,
            "status": "invalid",
        }
    else:
        assert valid_indices is not None
        replicates = bootstrap_gap_fraction_replicates(valid_gap_values, valid_indices)
        all_results["gap_fraction"] = {
            "valid_count": int(len(valid_gap_values)),
            "draws": int(config.bootstrap_draws),
            "sample_size": int(len(valid_gap_values)),
            "median": float(np.median(np.asarray(valid_gap_values, dtype=np.float64))),
            "bootstrap_median_mean": float(np.mean(replicates)),
            "bootstrap_lower_bound_05": one_sided_lower_quantile(replicates),
            "quantile": 0.05,
            "quantile_method": "linear",
            "resampling_unit": "valid positive-gap seeds",
            "status": "valid",
        }
    return {
        "seed_sequence": list(config.bootstrap_seed),
        "draws": int(config.bootstrap_draws),
        "all_seed_index_shape": list(all_indices.shape),
        "valid_gap_index_shape": None if valid_indices is None else list(valid_indices.shape),
        "all_seed_indices_sha256": hash_named_arrays((("all_seed_indices", all_indices),)),
        "valid_gap_indices_sha256": None if valid_indices is None else hash_named_arrays((("valid_gap_indices", valid_indices),)),
        "quantile": 0.05,
        "quantile_method": "linear",
        "statistics": all_results,
    }


def environment_payload() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


__all__ = [
    "ARMS",
    "BASE_CHILD_NAMES",
    "BASE_SEED",
    "BOOTSTRAP_SEED",
    "BridgeConfig",
    "BaseDraws",
    "CALIBRATION_SEEDS",
    "FactorInit",
    "OraclePrefix",
    "PRIMARY_SEEDS",
    "PROTOCOL_SHA256",
    "RANKS",
    "REGIMES",
    "RecordBlock",
    "SPLITS",
    "SeedDataset",
    "SeedStreams",
    "SanityPrefix",
    "STREAM_CHILD_NAMES",
    "STREAM_SEED",
    "WindowRecords",
    "alpha_values",
    "assert_protocol_hash",
    "bootstrap_gap_fraction_replicates",
    "bootstrap_index_matrices",
    "bootstrap_mean_replicates",
    "bootstrap_summary",
    "build_all_seed_datasets",
    "build_oracles",
    "build_prefix_oracle",
    "build_seed_dataset",
    "canonical_json_bytes",
    "current_mode_weights",
    "environment_payload",
    "factor_init",
    "hash_named_arrays",
    "make_base_draws",
    "make_factor_initializations",
    "make_seed_streams",
    "modal_matrix",
    "mode_count",
    "mode_schedule",
    "one_sided_lower_quantile",
    "population_risk",
    "prefix_mode_weights",
    "protocol_hash",
    "q_min_keep",
    "run_all_sanity",
    "run_sanity_projection",
    "sanity_contraction_bound",
    "sha256_bytes",
    "sha256_file",
    "update_array_hash",
    "weighted_modal_svd",
    "weighted_modal_indices",
]
