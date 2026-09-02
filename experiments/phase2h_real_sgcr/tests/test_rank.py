"""Phase-2H F3 LoRA-viability unit test: rank bound, frozen base, no-op init.

Runnable as pytest OR as ``python tests/test_rank.py``.
Imports numpy + torch + stdlib only; no transformers, no network.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lora_bert import LoRALinear


def test_base_is_buffer_not_param():
    base = nn.Linear(12, 7)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    # weight/bias are buffers -> not trainable, not in .parameters().
    param_ids = {id(p) for p in lora.parameters()}
    assert id(lora.weight) not in param_ids
    assert id(lora.bias) not in param_ids
    buf_names = {n for n, _ in lora.named_buffers()}
    assert "weight" in buf_names and "bias" in buf_names
    # only the two LoRA factors are trainable.
    trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
    assert set(trainable) == {"lora_A", "lora_B"}, trainable


def test_noop_at_init():
    torch.manual_seed(0)
    base = nn.Linear(12, 7)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(5, 12)
    with torch.no_grad():
        got = lora(x)
        want = torch.nn.functional.linear(x, base.weight, base.bias)
    assert torch.allclose(got, want, atol=1e-6), "adapter is not a no-op at init"


def test_output_shape():
    lora = LoRALinear(nn.Linear(12, 7), rank=4)
    out = lora(torch.randn(3, 4, 12))
    assert out.shape == (3, 4, 7), out.shape


def test_rank_bounded_after_steps():
    torch.manual_seed(1)
    lora = LoRALinear(nn.Linear(16, 9), rank=4, alpha=8.0)
    opt = torch.optim.AdamW(lora.trainable_parameters(), lr=1e-2)
    for _ in range(25):
        x = torch.randn(8, 16)
        target = torch.randn(8, 9)
        loss = ((lora(x) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        # base must never receive a gradient.
        assert lora.weight.grad is None
        assert lora.bias.grad is None
        opt.step()
    assert lora.delta_numerical_rank() <= 4, lora.delta_numerical_rank()
    # after training B is nonzero -> adapter is no longer a no-op.
    assert lora.lora_B.abs().sum().item() > 0


def run_all():
    test_base_is_buffer_not_param()
    test_noop_at_init()
    test_output_shape()
    test_rank_bounded_after_steps()
    print("OK")


if __name__ == "__main__":
    run_all()
