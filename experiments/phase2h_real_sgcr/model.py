"""Phase-2H model: frozen bert-mini + rank-4 LoRA (q,v) + 2-class pooled head.

STATUS: development scaffold.  Implements the "Model and genuine LoRA path" and
"Stable activation cells" sections of notes/phase2h_real_sgcr_protocol.md.

2026-08-27: backbone is prajjwal1/bert-mini (L=4, hidden=256) after the single
predeclared fallback from bert-tiny (L=2, hidden=128) following the D2 F1 failure.

- frozen pretrained bert-mini backbone (all base weights buffers via lora_bert);
- rank-4 alpha-8 LoRA on the query and value projection of every attention layer
  (4 layers x 2 projections x (4*256 + 256*4) = 16,384 trainable LoRA params);
- a two-class linear head over attention-mask mean-pooled final hidden states;
- the LoRA-disabled, warmup-mean-subtracted, L2-normalized mean-pooled hidden as
  the HIDDEN-d source-free stable signature;
- warmup training (head + LoRA), head frozen permanently at the warmup boundary,
  after which LoRA factors are the only trainable model parameters.

Integrity (protocol): trainable names, grad norms, factor norms, and numerical
rank of every B@A are logged; numerical rank > 4, a gradient on a frozen
base/head parameter after warmup, or a zero/non-finite update is a failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from lora_bert import (LoRALinear, attach_lora_to_bert, integrity_report,
                       lora_disabled)

# 2026-08-27: backbone switched from prajjwal1/bert-tiny (L=2, hidden=128) to
# prajjwal1/bert-mini (L=4, hidden=256).  This is the SINGLE predeclared
# representation fallback permitted by the protocol after the D2 F1 failure
# (source-free observability).  All BERT-tiny D1/D2 outcomes are BURNED
# (development-only, non-citable) -- see D2_RESULT.md.  The 86,272-byte historical
# envelope is unchanged; only the signature dimension grows 128 -> 256, which is
# charged in buffers._SIGNATURE_DIM.
MODEL_ID = "prajjwal1/bert-mini"
HIDDEN = 256
N_CLASSES = 2
RANK = 4
ALPHA = 8.0


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class BertTinyLoRA(nn.Module):
    """Frozen bert-mini + LoRA(q,v) + mask-mean-pooled 2-class head.

    The base BERT is loaded once; ``attach_lora_to_bert`` freezes every non-LoRA
    base parameter.  The head is a fresh trainable Linear(HIDDEN, 2); it is
    trained during warmup and then frozen by ``freeze_head``."""

    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.wrapped_paths = attach_lora_to_bert(bert, rank=RANK, alpha=ALPHA)
        self.head = nn.Linear(HIDDEN, N_CLASSES)
        self._head_frozen = False

    # -- forward helpers ---------------------------------------------------- #
    def _pooled(self, input_ids, attention_mask) -> torch.Tensor:
        """Attention-mask mean-pool of the final hidden states."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                 # (B, L, H)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, L, 1)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom                          # (B, H)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        return self.head(self._pooled(input_ids, attention_mask))

    @torch.no_grad()
    def signature(self, input_ids, attention_mask,
                  warmup_mean: Optional[torch.Tensor] = None) -> torch.Tensor:
        """source-free stable signature (HIDDEN-d): LoRA-disabled, eval-mode,
        mask-mean-pooled final hidden, minus the train-role warmup mean, then L2
        normalized.  Recomputed from stored tokens; never stored per replay
        example (protocol)."""
        was_training = self.training
        self.eval()
        with lora_disabled(self.bert):
            pooled = self._pooled(input_ids, attention_mask)
        if warmup_mean is not None:
            pooled = pooled - warmup_mean
        sig = torch.nn.functional.normalize(pooled, p=2.0, dim=-1)
        if was_training:
            self.train()
        return sig

    # -- parameter groups / freezing ---------------------------------------- #
    def lora_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for m in self.bert.modules():
            if isinstance(m, LoRALinear):
                params.extend(m.trainable_parameters())
        return params

    def head_parameters(self) -> List[nn.Parameter]:
        return list(self.head.parameters())

    def freeze_head(self) -> None:
        """Permanently freeze the classification head (warmup boundary)."""
        for p in self.head.parameters():
            p.requires_grad_(False)
        self._head_frozen = True

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Currently-trainable parameters (head only trainable pre-freeze)."""
        params = self.lora_parameters()
        if not self._head_frozen:
            params = self.head_parameters() + params
        return params

    # -- integrity ---------------------------------------------------------- #
    def integrity(self) -> Dict:
        rep = integrity_report(self.bert)
        rep["head_frozen"] = self._head_frozen
        # After warmup, head params must NOT be trainable; if head is frozen and
        # any head param has requires_grad, that's a failure.
        head_trainable = [n for n, p in self.head.named_parameters() if p.requires_grad]
        rep["head_trainable"] = head_trainable
        if self._head_frozen and head_trainable:
            rep["stray_trainable"] = rep.get("stray_trainable", []) + [
                f"head.{n}" for n in head_trainable
            ]
        return rep


def load_model(device: str = "cpu", seed: Optional[int] = None) -> BertTinyLoRA:
    """Load frozen BERT-tiny and wrap it.  Deterministic head init when seed set."""
    from transformers import AutoModel
    if seed is not None:
        torch.manual_seed(seed)
    bert = AutoModel.from_pretrained(MODEL_ID)
    model = BertTinyLoRA(bert).to(device)
    return model


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
def collate(records: List[Dict], device: str = "cpu") -> Tuple[torch.Tensor, ...]:
    """Stack canonical records into (input_ids long, attention_mask long, labels
    long).  Casts int32/uint8 storage dtypes up to the long dtype the BERT
    embedding / loss expect; the STORAGE stays int32/uint8 (byte budget)."""
    input_ids = torch.from_numpy(
        np.stack([np.asarray(r["input_ids"], dtype=np.int64) for r in records])
    ).to(device)
    attention_mask = torch.from_numpy(
        np.stack([np.asarray(r["attention_mask"], dtype=np.int64) for r in records])
    ).to(device)
    labels = torch.from_numpy(
        np.array([int(r["label"]) for r in records], dtype=np.int64)
    ).to(device)
    return input_ids, attention_mask, labels


# --------------------------------------------------------------------------- #
# Optimizer + warmup
# --------------------------------------------------------------------------- #
@dataclass
class OptimConfig:
    lora_lr: float = 5e-4
    head_lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    batch_size: int = 16


def make_optimizer(model: BertTinyLoRA, cfg: OptimConfig) -> torch.optim.Optimizer:
    """AdamW with separate LoRA / head learning rates.  Head group is included
    only while the head is trainable (warmup)."""
    groups = [{"params": model.lora_parameters(), "lr": cfg.lora_lr}]
    if not model._head_frozen:
        groups.append({"params": model.head_parameters(), "lr": cfg.head_lr})
    return torch.optim.AdamW(groups, betas=cfg.betas, eps=cfg.eps,
                             weight_decay=cfg.weight_decay)


def _iter_minibatches(records, batch_size, rng):
    idx = rng.permutation(len(records))
    for i in range(0, len(records), batch_size):
        yield [records[int(j)] for j in idx[i:i + batch_size]]


def train_step(model, optimizer, batch_records, cfg, device="cpu") -> float:
    """One optimizer step on a batch; returns the scalar loss.  Applies grad
    clipping to the currently-trainable parameters."""
    model.train()
    input_ids, attention_mask, labels = collate(batch_records, device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids, attention_mask)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)
    optimizer.step()
    return float(loss.detach().cpu())


def warmup_train(model: BertTinyLoRA, warmup_records: List[Dict], cfg: OptimConfig,
                 epochs: int = 1, seed: int = 0, device: str = "cpu") -> Dict:
    """Train head + LoRA on the six warmup windows, then freeze the head and
    return the train-role warmup mean (LoRA-disabled pooled hidden) used for
    signatures.  This is a COMMON initialization stage shared by all arms; the
    caller clones the resulting state into each arm."""
    optimizer = make_optimizer(model, cfg)
    rng = np.random.default_rng(seed)
    losses: List[float] = []
    for _ in range(epochs):
        for batch in _iter_minibatches(warmup_records, cfg.batch_size, rng):
            losses.append(train_step(model, optimizer, batch, cfg, device))

    # Warmup mean of the LoRA-disabled pooled hidden over the warmup records.
    warmup_mean = _compute_pooled_mean(model, warmup_records, device)
    model.freeze_head()
    return {"losses": losses, "warmup_mean": warmup_mean,
            "integrity": model.integrity()}


import copy


def snapshot(model: BertTinyLoRA, optimizer=None) -> Dict:
    """Deep-copy the LoRA/head state and (optionally) optimizer state so a branch
    can be restored exactly (protocol: "restoring both model and optimizer state
    of the selected branch").  Only trainable-relevant state is copied; the
    frozen base is shared (buffers, never mutated)."""
    snap = {
        "lora": copy.deepcopy({n: p.detach().clone()
                               for n, p in model.bert.named_parameters()
                               if p.requires_grad}),
        "head": copy.deepcopy(model.head.state_dict()),
        "head_frozen": model._head_frozen,
    }
    if optimizer is not None:
        snap["optim"] = copy.deepcopy(optimizer.state_dict())
    return snap


def restore(model: BertTinyLoRA, snap: Dict, optimizer=None) -> None:
    """Restore a snapshot in-place."""
    with torch.no_grad():
        named = dict(model.bert.named_parameters())
        for n, v in snap["lora"].items():
            named[n].copy_(v)
    model.head.load_state_dict(snap["head"])
    model._head_frozen = snap["head_frozen"]
    if optimizer is not None and "optim" in snap:
        optimizer.load_state_dict(snap["optim"])


@torch.no_grad()
def _compute_pooled_mean(model, records, device="cpu", batch_size=64) -> torch.Tensor:
    """Mean over records of the LoRA-disabled mask-mean-pooled final hidden."""
    model.eval()
    total = torch.zeros(HIDDEN, device=device)
    n = 0
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        input_ids, attention_mask, _ = collate(chunk, device)
        with lora_disabled(model.bert):
            pooled = model._pooled(input_ids, attention_mask)
        total += pooled.sum(dim=0)
        n += pooled.shape[0]
    return total / max(n, 1)
