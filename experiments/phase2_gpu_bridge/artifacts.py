"""Canonical hashing and small artifact writers for Phase-2 runs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _little_endian_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and sys.byteorder == "big"):
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    elif array.dtype.byteorder == "=" and array.dtype.kind not in "OSU":
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(array)


def canonical_array_bytes(array: np.ndarray, *, label: str = "") -> bytes:
    array = _little_endian_array(np.asarray(array))
    header = canonical_json_bytes(
        {
            "label": label,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "order": "C",
        }
    )
    return len(header).to_bytes(8, "little") + header + array.tobytes(order="C")


def hash_arrays(named_arrays: Iterable[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for label, array in named_arrays:
        payload = canonical_array_bytes(array, label=label)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _atomic_write(path: Path, writer: Any) -> None:
    """Write and fsync a same-directory temporary file before replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_json(path: Path, payload: Any) -> str:
    text = json.dumps(
        jsonable(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    _atomic_write(path, lambda handle: handle.write(text))
    return sha256_file(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    names = list(fieldnames)

    def _write(handle: Any) -> None:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            extra = set(row) - set(names)
            if extra:
                raise ValueError(f"CSV row contains fields outside the declared schema: {sorted(extra)!r}")
            writer.writerow({field: jsonable(row.get(field)) for field in names})
    _atomic_write(path, _write)
    return sha256_file(path)


def file_hashes(directory: Path, names: Sequence[str]) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in names if (directory / name).is_file()}


def base_runtime_metadata() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def torch_runtime_metadata() -> dict[str, Any]:
    """Collect torch/CUDA metadata without importing torch on CPU-only hosts."""

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on host
        return {"torch_available": False, "torch_import_error": type(exc).__name__}
    metadata: dict[str, Any] = {
        "hostname": platform.node(),
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "environment_controls": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "MKL_DYNAMIC",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
    }
    if metadata["cuda_available"]:
        metadata["device_count"] = int(torch.cuda.device_count())
        metadata["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            devices = []
            for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
                if len(row) != 3:
                    raise RuntimeError("unexpected nvidia-smi device row")
                devices.append(
                    {
                        "index": int(row[0]),
                        "name": row[1].strip(),
                        "driver_version": row[2].strip(),
                    }
                )
            metadata["nvidia_smi_devices"] = devices
        except Exception as exc:  # pragma: no cover - host dependent
            metadata["nvidia_smi_error"] = type(exc).__name__
    return metadata


__all__ = [
    "base_runtime_metadata",
    "canonical_array_bytes",
    "canonical_json_bytes",
    "file_hashes",
    "hash_arrays",
    "hash_json",
    "jsonable",
    "sha256_bytes",
    "sha256_file",
    "torch_runtime_metadata",
    "write_csv",
    "write_json",
]
