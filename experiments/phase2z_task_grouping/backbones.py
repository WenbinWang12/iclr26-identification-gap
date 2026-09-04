"""RUNBOOK 3B: backbone abstraction for the orders x backbones study.

The Order-4 harness (run_probe.collect_logits / train_one_task) is written for a
T5 SEQ2SEQ model: it scores the first decoder step after decoder_start, and
trains with `labels=`. A decoder-only (causal) LM needs a different code path:
the restricted-argmax score reads the logits at the LAST prompt position (the
next-token distribution), and training is next-token LM loss over prompt+label
with the prompt positions masked out. This module encapsulates BOTH so the rest
of the grouping code is backbone-agnostic.

Each backbone exposes:
  load(model_name, dtype)          -> (base_model, tokenizer, family)
  target_modules(family)           -> LoRA target module names for that family
  collect_logits(model, tok, examples, task, device, ...) -> (n, K) np.ndarray
  train_one_task(model, tok, examples, device, args, optimizer) -> [losses]

`family` is 'seq2seq' or 'causal'. The verbalizer coordinate is the FIRST
sub-word piece id of each label, identical to run_probe.first_piece_ids, so the
offset machinery is unchanged across backbones.
"""

from __future__ import annotations

import numpy as np
import torch

from experiments.phase2j_offset_conflict.run_probe import (
    first_piece_ids,
    log,
)


# ---------------------------------------------------------------------------
# LoRA target modules per architecture family. Names differ across backbones.
# ---------------------------------------------------------------------------
_TARGETS = {
    # T5: attention q/v projections (matches run_grouping/run_combined).
    "t5": ["q", "v"],
    # GPT-2 family: fused attention projection.
    "gpt2": ["c_attn"],
    # LLaMA / Mistral / Qwen-style: separate q/v projections.
    "llama": ["q_proj", "v_proj"],
}


def resolve_family(model_name):
    """Map a HF model id to (arch_family, lora_key)."""
    n = model_name.lower()
    if "t5" in n:
        return "seq2seq", "t5"
    if "gpt2" in n or "gpt-2" in n:
        return "causal", "gpt2"
    # default causal: llama/mistral/qwen/pythia use q_proj/v_proj or gpt-neox
    if "neox" in n or "pythia" in n:
        return "causal", "llama"  # gpt-neox also exposes query_key_value; see note
    return "causal", "llama"


def target_modules(lora_key):
    if lora_key not in _TARGETS:
        raise KeyError(f"no LoRA targets registered for '{lora_key}'; "
                       f"known={sorted(_TARGETS)}")
    return list(_TARGETS[lora_key])


def load(model_name, dtype):
    """Load the base model + tokenizer and report the architecture family."""
    from transformers import AutoTokenizer
    family, lora_key = resolve_family(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if family == "seq2seq":
        from transformers import AutoModelForSeq2SeqLM
        base = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None)
    else:
        from transformers import AutoModelForCausalLM
        # Causal LMs often lack a pad token; reuse EOS so batching works.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None)
        # Left-pad for causal generation/scoring so the last real token is at -1.
        tokenizer.padding_side = "left"
    log(f"backbone: {model_name} family={family} lora_key={lora_key} "
        f"targets={target_modules(lora_key)}")
    return base, tokenizer, family, lora_key


@torch.no_grad()
def collect_logits(model, tokenizer, examples, task, device, family, *,
                   max_source, batch_size):
    """(n, K) first-piece logits, dispatched by architecture family.

    seq2seq: first decoder-step logits after decoder_start (== run_probe).
    causal : next-token logits at the last prompt position (left-padded, so the
             last real token is at index -1 for every row).
    """
    columns = list(first_piece_ids(tokenizer, task.labels))
    model.eval()
    chunks = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        enc = tokenizer([ex.prompt for ex in batch], max_length=max_source,
                        truncation=True, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        if family == "seq2seq":
            start_ids = torch.full(
                (len(batch), 1), model.config.decoder_start_token_id,
                dtype=torch.long, device=device)
            logits = model(**enc, decoder_input_ids=start_ids).logits[:, 0, :]
        else:
            # left-padded: last column is the final real token for all rows.
            logits = model(**enc).logits[:, -1, :]
        chunks.append(logits[:, columns].float().cpu())
    return torch.cat(chunks, dim=0).numpy()


def train_one_task(model, tokenizer, examples, device, family, *, args,
                   optimizer):
    """Train `args.epochs` over one task; loss dispatched by family.

    seq2seq: encoder-decoder cross-entropy with `labels=` (== run_probe path,
             reused directly to stay byte-identical for T5).
    causal : next-token LM loss over prompt+label, prompt positions masked to
             -100 so only the label tokens are supervised.
    """
    if family == "seq2seq":
        from experiments.phase2j_offset_conflict.run_probe import (
            train_one_task as t5_train)
        return t5_train(model, tokenizer, examples, device,
                        args=args, optimizer=optimizer)

    # ---- causal LM training ----
    model.train()
    epochs = max(1, int(getattr(args, "epochs", 1) or 1))
    ordered = []
    for epoch in range(epochs):
        order = np.random.default_rng([args.seed, epoch]).permutation(len(examples))
        ordered.extend(examples[i] for i in order)
    micro = args.batch_size
    per_step = micro * args.grad_accum
    losses = []
    pad_id = tokenizer.pad_token_id
    for start in range(0, len(ordered), per_step):
        window = ordered[start:start + per_step]
        if not window:
            break
        while True:
            optimizer.zero_grad(set_to_none=True)
            pieces = [window[i:i + micro] for i in range(0, len(window), micro)]
            try:
                total = 0.0
                for batch in pieces:
                    # Build prompt+label sequences; supervise only label tokens.
                    input_ids, attn, label_ids = _causal_batch(
                        tokenizer, batch, args.max_source, args.max_target,
                        device)
                    scale = len(batch) / len(window)
                    out = model(input_ids=input_ids, attention_mask=attn,
                                labels=label_ids)
                    loss = out.loss * scale
                    loss.backward()
                    total += loss.detach().item()
                break
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if micro <= 1:
                    raise
                micro = max(1, micro // 2)
                log("causal train OOM -> micro-batch %d (eff batch fixed)" % micro)
        optimizer.step()
        losses.append(total)
    return losses


def _causal_batch(tokenizer, batch, max_source, max_target, device):
    """Right-padded prompt+label tensors with prompt tokens masked in labels.

    Training uses right padding (labels align to inputs); scoring uses left
    padding (set on the tokenizer at load). We flip padding_side locally here so
    the two never interfere.
    """
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        seqs, label_masks = [], []
        for ex in batch:
            p_ids = tokenizer(ex.prompt, max_length=max_source, truncation=True,
                              add_special_tokens=True)["input_ids"]
            l_ids = tokenizer(" " + ex.label, max_length=max_target,
                              truncation=True,
                              add_special_tokens=False)["input_ids"]
            eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
            ids = p_ids + l_ids + eos
            mask = [-100] * len(p_ids) + l_ids + eos
            seqs.append(ids)
            label_masks.append(mask)
        width = max(len(s) for s in seqs)
        pad_id = tokenizer.pad_token_id
        input_ids, attn, labels = [], [], []
        for ids, mask in zip(seqs, label_masks):
            pad = width - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            labels.append(mask + [-100] * pad)
        return (torch.tensor(input_ids, device=device),
                torch.tensor(attn, device=device),
                torch.tensor(labels, device=device))
    finally:
        tokenizer.padding_side = prev_side
