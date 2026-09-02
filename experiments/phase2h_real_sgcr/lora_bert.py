"""Manually audited LoRA wrapper for a frozen Transformer linear projection.

STATUS: development scaffold.  `peft` is deliberately NOT used: the protocol
(notes/phase2h_real_sgcr_protocol.md) requires that every trainable parameter be
explicitly accounted for, that numerical rank never exceed R, and that no frozen
base parameter ever receives a gradient.  This module implements exactly that and
is unit-tested standalone (no `transformers` needed) via tests/test_rank.py and
tests/test_no_source_leak.py.

LoRA form (protocol Model-and-genuine-LoRA-path):
    W(x) = W0(x) + (alpha / rank) * (x @ A^T) @ B^T
with W0 frozen, A in R^{rank x in}, B in R^{out x rank}, rank=4, alpha=8, dropout=0.
A ~ small Gaussian, B = 0 at init (standard LoRA init: the adapter starts as a
no-op so warmup begins from the pretrained function).

`attach_lora_to_bert` wraps the query and value nn.Linear of every attention layer
of a HuggingFace BERT (imported lazily so this file loads without transformers).
"""
from __future__ import annotations

import contextlib
from typing import List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen base linear + trainable low-rank update.  Base weight/bias are
    registered as non-trainable buffers so they can never receive a gradient."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        # Freeze the base: store as buffers, NOT Parameters -> no grad, not in
        # .parameters() that the optimizer sees.
        self.register_buffer("weight", base.weight.detach().clone())
        if base.bias is not None:
            self.register_buffer("bias", base.bias.detach().clone())
        else:
            self.bias = None

        # Trainable low-rank factors.  B=0 => adapter is a no-op at init.
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)

        # Runtime toggle: when False the module is exactly the frozen base (used
        # to compute the LoRA-disabled stable signature, protocol "Stable
        # activation cells").  This does not detach the factors from autograd;
        # it just skips their contribution, so signatures are gradient-free when
        # called under no_grad and never influence training when re-enabled.
        self.lora_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = torch.nn.functional.linear(x, self.weight, self.bias)
        if not self.lora_enabled:
            return base_out
        # (x @ A^T) @ B^T, scaled.  delta has rank <= self.rank by construction.
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def delta_numerical_rank(self, tol: float = 1e-6) -> int:
        """Numerical rank of the effective update B@A.  Must be <= self.rank; a
        value above self.rank is an integrity failure (protocol)."""
        delta = self.lora_B @ self.lora_A  # (out, in)
        if delta.numel() == 0:
            return 0
        sv = torch.linalg.svdvals(delta.float())
        thresh = tol * (sv[0] if sv.numel() else torch.tensor(0.0))
        return int((sv > thresh).sum().item())

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [self.lora_A, self.lora_B]


def attach_lora_to_bert(model, rank: int = 4, alpha: float = 8.0) -> List[str]:
    """Replace query & value nn.Linear in every BERT self-attention with LoRALinear
    and freeze all remaining base parameters.  Returns the list of wrapped module
    paths.  `model` is a transformers BertModel/BertForSequenceClassification; the
    import is the caller's responsibility (kept out of this module so it loads
    without transformers installed)."""
    wrapped: List[str] = []
    for name, module in list(model.named_modules()):
        # BERT self-attention exposes .query, .key, .value as nn.Linear.
        if module.__class__.__name__.endswith("SelfAttention"):
            for proj in ("query", "value"):
                lin = getattr(module, proj, None)
                if isinstance(lin, nn.Linear):
                    setattr(module, proj, LoRALinear(lin, rank=rank, alpha=alpha))
                    wrapped.append(f"{name}.{proj}")
    if not wrapped:
        raise RuntimeError("no BERT SelfAttention query/value linears found to wrap")

    # Freeze everything that is not a LoRA factor.
    lora_param_ids = set()
    for m in model.modules():
        if isinstance(m, LoRALinear):
            lora_param_ids.update(id(p) for p in m.trainable_parameters())
    for p in model.parameters():
        p.requires_grad_(id(p) in lora_param_ids)
    return wrapped


@contextlib.contextmanager
def lora_disabled(model):
    """Temporarily disable every LoRALinear adapter (restore prior state on
    exit).  Used to compute the LoRA-disabled stable signature and the F3
    "disabling LoRA changes logits" check."""
    mods = [m for m in model.modules() if isinstance(m, LoRALinear)]
    prev = [m.lora_enabled for m in mods]
    try:
        for m in mods:
            m.lora_enabled = False
        yield
    finally:
        for m, p in zip(mods, prev):
            m.lora_enabled = p


def integrity_report(model) -> dict:
    """Collect the quantities the protocol requires logging every window:
    trainable names, per-module numerical rank of B@A, and a flag if any
    non-LoRA parameter is trainable (an integrity failure)."""
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    ranks = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            ranks[name] = m.delta_numerical_rank()
    # any trainable param that is not a lora factor?
    lora_names = set()
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            lora_names.add(f"{name}.lora_A")
            lora_names.add(f"{name}.lora_B")
    stray = [n for n in trainable_names if n not in lora_names]
    return {
        "trainable_names": trainable_names,
        "numerical_ranks": ranks,
        "max_rank": max(ranks.values()) if ranks else 0,
        "stray_trainable": stray,  # must be empty
    }
