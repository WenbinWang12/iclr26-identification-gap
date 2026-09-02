"""Lazy PyTorch training path for the Phase-2 planted linear-LoRA bridge."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


def configure_environment() -> None:
    """Set the protocol's process-wide numerical controls before torch import."""

    values = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MKL_DYNAMIC": "FALSE",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for key, value in values.items():
        os.environ[key] = value


def import_torch() -> Any:
    configure_environment()
    import torch  # type: ignore

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch


def require_cuda(torch: Any, device: str) -> Any:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return resolved


def make_model(torch: Any, w0_32: np.ndarray, a_32: np.ndarray) -> Any:
    """Create a small module without any stochastic torch initialization."""

    class PlantedLoRA(torch.nn.Module):
        def __init__(self, base: np.ndarray, init_a: np.ndarray) -> None:
            super().__init__()
            base_tensor = torch.as_tensor(np.asarray(base, dtype=np.float32), dtype=torch.float32)
            init_tensor = torch.as_tensor(np.asarray(init_a, dtype=np.float32), dtype=torch.float32)
            self.register_buffer("W0", base_tensor, persistent=True)
            # B follows the protocol's deployed shape [d_out, R].
            self.B = torch.nn.Parameter(torch.zeros((base_tensor.shape[0], init_tensor.shape[0]), dtype=torch.float32))
            self.A = torch.nn.Parameter(init_tensor.clone())

        def forward(self, x: Any) -> Any:
            return x @ self.W0.transpose(0, 1) + (x @ self.A.transpose(0, 1)) @ self.B.transpose(0, 1)

    return PlantedLoRA(w0_32, a_32)


def make_optimizer(torch: Any, model: Any) -> Any:
    return torch.optim.AdamW(
        [model.B, model.A],
        lr=3e-2,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        foreach=False,
        fused=False,
        maximize=False,
        capturable=False,
        differentiable=False,
    )


def exact_batch_loss(torch: Any, model: Any, x32: Any, y32: Any) -> Any:
    if int(x32.shape[0]) != 128 or int(y32.shape[0]) != 128:
        raise ValueError("the protocol loss is defined for exactly 128 examples")
    residual = model(x32) - y32
    return residual.square().sum() / (2.0 * 128.0)


@dataclass(frozen=True)
class BatchTrace:
    update: int
    loss: float
    grad_norm: float
    replay_count: int
    replay_unique_count: int
    replay_duplicate_count: int
    replay_sample_seconds: float
    replay_assembly_seconds: float
    device_staging_seconds: float
    model_input_names: tuple[str, str]
    model_input_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    model_input_dtypes: tuple[str, str]
    latent_metadata_tensor_count: int
    replay_sampler_argument_names: tuple[str, str] | tuple[()]


def train_window(
    torch: Any,
    model: Any,
    optimizer: Any,
    *,
    x_current: np.ndarray,
    y_current: np.ndarray,
    pi_t: np.ndarray,
    replay_state: Any | None,
    replay_rng: np.random.Generator | None,
    device: Any,
    updates: int = 100,
    trace: list[BatchTrace] | None = None,
) -> list[BatchTrace]:
    """Run exactly one protocol window, preserving the fixed gather order."""

    if x_current.shape != (1024, 32) or y_current.shape != (1024, 32):
        raise ValueError("current window arrays must have shape (1024, 32)")
    if np.asarray(pi_t).shape != (1024,):
        raise ValueError("pi_t must contain exactly 1024 stored indices")
    if trace is None:
        trace = []
    model.train()
    for update in range(updates):
        current_indices = np.asarray([pi_t[(128 * update + offset) % 1024] for offset in range(128)], dtype=np.int64)
        replay_count = 0
        replay_unique_count = 0
        replay_duplicate_count = 0
        replay_sample_seconds = 0.0
        replay_assembly_seconds = 0.0
        sampler_argument_names: tuple[str, str] | tuple[()] = ()
        if replay_state is not None and replay_rng is not None and replay_state.n_occupied > 0:
            sample_started = time.perf_counter()
            replay_indices = replay_state.sample_indices(replay_rng, size=64)
            replay_sample_seconds = time.perf_counter() - sample_started
            assert replay_indices is not None
            sampler_argument_names = ("draw_rng", "size")
            assembly_started = time.perf_counter()
            replay_unique_count = int(np.unique(replay_indices).size)
            replay_duplicate_count = int(replay_indices.size - replay_unique_count)
            current_indices = current_indices[:64]
            x_batch_np = np.concatenate((x_current[current_indices], replay_state.x_slots[replay_indices]), axis=0)
            y_batch_np = np.concatenate((y_current[current_indices], replay_state.y_slots[replay_indices]), axis=0)
            replay_assembly_seconds = time.perf_counter() - assembly_started
            replay_count = 64
        else:
            x_batch_np = x_current[current_indices]
            y_batch_np = y_current[current_indices]
        staging_started = time.perf_counter()
        x_batch = torch.as_tensor(np.ascontiguousarray(x_batch_np), dtype=torch.float32, device=device)
        y_batch = torch.as_tensor(np.ascontiguousarray(y_batch_np), dtype=torch.float32, device=device)
        device_staging_seconds = time.perf_counter() - staging_started
        optimizer.zero_grad(set_to_none=True)
        loss = exact_batch_loss(torch, model, x_batch, y_batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([model.B, model.A], max_norm=5.0, norm_type=2.0, error_if_nonfinite=True, foreach=False)
        optimizer.step()
        trace.append(
            BatchTrace(
                update=update,
                loss=float(loss.detach().cpu()),
                grad_norm=float(grad_norm.detach().cpu()),
                replay_count=replay_count,
                replay_unique_count=replay_unique_count,
                replay_duplicate_count=replay_duplicate_count,
                replay_sample_seconds=float(replay_sample_seconds),
                replay_assembly_seconds=float(replay_assembly_seconds),
                device_staging_seconds=float(device_staging_seconds),
                model_input_names=("x", "y"),
                model_input_shapes=(tuple(int(value) for value in x_batch.shape), tuple(int(value) for value in y_batch.shape)),
                model_input_dtypes=(str(x_batch.dtype), str(y_batch.dtype)),
                latent_metadata_tensor_count=0,
                replay_sampler_argument_names=sampler_argument_names,
            )
        )
    return trace


def model_delta64(model: Any) -> np.ndarray:
    """Copy a trained module to NumPy float64 without retaining torch tensors."""

    b = model.B.detach().to(device="cpu", dtype=getattr(model.B, "dtype", None)).numpy().astype(np.float64, copy=False)
    a = model.A.detach().to(device="cpu", dtype=getattr(model.A, "dtype", None)).numpy().astype(np.float64, copy=False)
    return b @ a


__all__ = [
    "BatchTrace",
    "configure_environment",
    "exact_batch_loss",
    "import_torch",
    "make_model",
    "make_optimizer",
    "model_delta64",
    "require_cuda",
    "train_window",
]
