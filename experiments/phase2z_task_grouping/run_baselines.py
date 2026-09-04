"""Phase-2Z-H (RUNBOOK 3D): published continual-PEFT baselines at MATCHED budget.

The paper's method is a fixed-budget allocation (shared rank-8 vs 4×rank-2 +
per-scope offset). To place it against the literature, this script runs
established continual-PEFT baselines on the IDENTICAL harness: same Order-4
stream, same eligible tasks, same risk/audit splits, same seeds, same 2.36M
trainable-scalar budget (one rank-8 LoRA on q/v), same restricted-argmax balanced
accuracy on the audit split, `test.json` never opened. Only the TRAINING rule
differs, so any accuracy difference is a pure algorithm effect at fixed budget.

Baselines implemented here (each faithful to its published mechanism, adapted to
this harness — see notes/phase2zh_baselines_protocol.md for the exact mapping
and for what is deliberately NOT claimed):

  seqft   Sequential fine-tuning of ONE rank-8 LoRA over the stream. The naive
          continual baseline (no anti-forgetting). Lower bound.
  olora   O-LoRA (Wang et al., 2023): sequential LoRA with an orthogonality
          penalty pushing each task's LoRA-A subspace orthogonal to the frozen
          accumulated subspace of previous tasks. Same single rank-8 adapter;
          the only addition is the O-LoRA regularizer during training.
  ewc     EWC-style anchor regularization on the LoRA params: a quadratic pull
          toward the post-previous-task weights, diagonal-Fisher weighted. A
          representative regularization-CL baseline at the same budget.

NOT implemented here (require components outside this harness; scoped out in the
protocol, NOT stubbed and NOT reported as if run): E2-LoRA's drift-energy rank
pool, NSR's replay buffer. The RUNBOOK marks these for the collaborator.

Output: baselines_{method}_s{seed}.json, same per-task audit-accuracy schema as
the method runs so analyze_baselines.py can table them side by side.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_HOME", "/mnt/data/wenbin/iclr26/models/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # noqa: E402
from peft import LoraConfig, get_peft_model, TaskType  # noqa: E402

from experiments.phase2i_anchored_cvar.order4_data import (  # noqa: E402
    ORDER4_TASKS,
    prepare_task_partitions,
)
from experiments.phase2j_offset_conflict.offsets import (  # noqa: E402
    balanced_accuracy,
)
from experiments.phase2j_offset_conflict.run_probe import (  # noqa: E402
    collect_logits,
    log,
)
from experiments.phase2z_task_grouping.run_grouping import (  # noqa: E402
    eligible_tasks,
    make_optimizer,
)


def build_single_lora(base_model, *, rank, lora_alpha, lora_dropout):
    """One rank-`rank` LoRA on q/v == the paper's shared-adapter budget."""
    cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=rank, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, target_modules=["q", "v"])
    return get_peft_model(base_model, cfg, adapter_name="default")


def _lora_A_matrices(model, adapter="default"):
    """Yield the LoRA-A weight tensors (shape (r, in)) for the active adapter.

    Works with the standard PEFT LoRA layout where each target module exposes
    `lora_A[adapter].weight`. Robust to modules that lack the adapter key.
    """
    for module in model.modules():
        A = getattr(module, "lora_A", None)
        if A is None:
            continue
        try:
            layer = A[adapter]
        except (KeyError, TypeError):
            continue
        w = getattr(layer, "weight", None)
        if w is not None:
            yield module, w


def olora_penalty(model, prev_bases, adapter="default"):
    """O-LoRA orthogonality penalty: sum_l ||A_l^cur (A_l^prev)^T||_F^2.

    `prev_bases` maps a module id to the row-stacked A matrices of all previous
    tasks (a frozen tensor). Pushes the current task's LoRA-A rows orthogonal to
    the span of previous tasks' rows, per the O-LoRA objective. Returns a scalar
    tensor (0.0 for the first task, when there is no previous basis).
    """
    total = None
    for module, w_cur in _lora_A_matrices(model, adapter):
        prev = prev_bases.get(id(module))
        if prev is None:
            continue
        # w_cur: (r, in); prev: (r_prev, in). Overlap matrix (r, r_prev).
        overlap = w_cur @ prev.to(w_cur.dtype).t()
        term = (overlap ** 2).sum()
        total = term if total is None else total + term
    if total is None:
        return torch.zeros((), requires_grad=True)
    return total


def snapshot_A(model, adapter="default"):
    """Detached copy of each module's current LoRA-A (for O-LoRA accumulation)."""
    return {id(m): w.detach().clone() for m, w in _lora_A_matrices(model, adapter)}


def snapshot_params(model):
    """Detached copy of all trainable params (for EWC anchor)."""
    return {n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad}


def train_task_regularized(model, tokenizer, examples, device, *, args,
                           optimizer, method, prev_bases, anchor, fisher):
    """One task's training with an optional CL regularizer added to the loss.

    method='seqft' : plain LM loss (naive sequential).
    method='olora' : + args.olora_lambda * orthogonality penalty vs prev_bases.
    method='ewc'   : + 0.5*args.ewc_lambda * sum fisher*(theta-anchor)^2.
    The base LM loss/OOM handling mirrors run_probe.train_one_task exactly.
    """
    model.train()
    epochs = max(1, int(getattr(args, "epochs", 1) or 1))
    ordered = []
    for epoch in range(epochs):
        order = np.random.default_rng([args.seed, epoch]).permutation(len(examples))
        ordered.extend(examples[i] for i in order)
    micro = args.batch_size
    per_step = micro * args.grad_accum
    losses = []
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
                    enc = tokenizer([e.prompt for e in batch],
                                    max_length=args.max_source, truncation=True,
                                    padding=True, return_tensors="pt")
                    lab = tokenizer([e.label for e in batch],
                                    max_length=args.max_target, truncation=True,
                                    padding=True, return_tensors="pt")
                    label_ids = lab["input_ids"].clone()
                    label_ids[label_ids == tokenizer.pad_token_id] = -100
                    enc = {k: v.to(device) for k, v in enc.items()}
                    scale = len(batch) / len(window)
                    loss = model(**enc, labels=label_ids.to(device)).loss * scale
                    loss.backward()
                    total += loss.detach().item()
                # Regularizer added once per optimizer step (not per micro-batch).
                reg = None
                if method == "olora" and prev_bases:
                    reg = args.olora_lambda * olora_penalty(model, prev_bases)
                elif method == "ewc" and anchor is not None:
                    acc = None
                    for n, p in model.named_parameters():
                        if not p.requires_grad or n not in anchor:
                            continue
                        f = fisher.get(n)
                        d = (p - anchor[n]) ** 2
                        term = (f * d).sum() if f is not None else d.sum()
                        acc = term if acc is None else acc + term
                    if acc is not None:
                        reg = 0.5 * args.ewc_lambda * acc
                if reg is not None and reg.requires_grad:
                    reg.backward()
                    total += float(reg.detach().item())
                break
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if micro <= 1:
                    raise
                micro = max(1, micro // 2)
                log("baseline train OOM -> micro-batch %d" % micro)
        optimizer.step()
        losses.append(total)
    return losses


def estimate_fisher(model, tokenizer, examples, device, *, args, max_batches=8):
    """Diagonal Fisher (squared-grad expectation) of trainable params on a task,
    for the EWC anchor. Uses up to `max_batches` micro-batches of the task's own
    update split; cheap and standard."""
    model.eval()
    fisher = {n: torch.zeros_like(p)
              for n, p in model.named_parameters() if p.requires_grad}
    n_used = 0
    for start in range(0, min(len(examples), max_batches * args.batch_size),
                       args.batch_size):
        batch = examples[start:start + args.batch_size]
        if not batch:
            break
        enc = tokenizer([e.prompt for e in batch], max_length=args.max_source,
                        truncation=True, padding=True, return_tensors="pt")
        lab = tokenizer([e.label for e in batch], max_length=args.max_target,
                        truncation=True, padding=True, return_tensors="pt")
        label_ids = lab["input_ids"].clone()
        label_ids[label_ids == tokenizer.pad_token_id] = -100
        enc = {k: v.to(device) for k, v in enc.items()}
        model.zero_grad(set_to_none=True)
        loss = model(**enc, labels=label_ids.to(device)).loss
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        n_used += 1
    model.zero_grad(set_to_none=True)
    if n_used:
        for n in fisher:
            fisher[n] /= n_used
    return fisher


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default="google-t5/t5-large")
    ap.add_argument("--method", required=True,
                    choices=["seqft", "olora", "ewc"])

    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--cap-per-class", type=int, default=400)
    ap.add_argument("--max-source", type=int, default=512)
    ap.add_argument("--max-target", type=int, default=128)

    ap.add_argument("--rank", type=int, default=8)  # matched budget
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--olora-lambda", type=float, default=0.5)
    ap.add_argument("--ewc-lambda", type=float, default=1.0)

    ap.add_argument("--risk-per-class", type=int, default=32)
    ap.add_argument("--audit-per-class", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    log("=" * 80)
    log(f"Phase-2Z-H (RUNBOOK 3D): baseline={args.method} at matched rank-{args.rank}")
    log(f"  seed={args.seed}")
    log("=" * 80)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    base = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = build_single_lora(base, rank=args.rank, lora_alpha=args.lora_alpha,
                              lora_dropout=args.lora_dropout)

    task_list = eligible_tasks(tokenizer, args.data_root, ORDER4_TASKS)
    log(f"Eligible tasks: {[t.name for t, _ in task_list]}")

    prev_bases = {}   # O-LoRA accumulated previous A rows, per module id
    anchor = None     # EWC anchor params (previous task)
    fisher = {}       # EWC diagonal Fisher (accumulated)

    for task, rarest in task_list:
        task_risk = min(args.risk_per_class, (rarest - 1) // 2)
        task_audit = min(args.audit_per_class, rarest - 1 - task_risk)
        parts = prepare_task_partitions(
            args.data_root, task.name, cap_per_class=args.cap_per_class,
            risk_per_class=task_risk, audit_per_class=task_audit, seed=args.seed)
        train_args = SimpleNamespace(
            batch_size=args.batch_size, grad_accum=args.grad_accum, lr=args.lr,
            epochs=args.epochs, max_source=args.max_source,
            max_target=args.max_target, seed=args.seed,
            olora_lambda=args.olora_lambda, ewc_lambda=args.ewc_lambda)
        optimizer = make_optimizer(model, args.lr)
        log(f"  train {task.name} ({len(parts.update)} ex) method={args.method}")
        train_task_regularized(
            model, tokenizer, list(parts.update), device, args=train_args,
            optimizer=optimizer, method=args.method, prev_bases=prev_bases,
            anchor=anchor, fisher=fisher)

        # Update CL state AFTER the task.
        if args.method == "olora":
            for mid, w in snapshot_A(model).items():
                prev = prev_bases.get(mid)
                prev_bases[mid] = w if prev is None else torch.cat([prev, w], dim=0)
        elif args.method == "ewc":
            new_fisher = estimate_fisher(model, tokenizer, list(parts.update),
                                         device, args=train_args)
            for n, f in new_fisher.items():
                fisher[n] = f if n not in fisher else fisher[n] + f
            anchor = snapshot_params(model)

    # Score every task on its audit split with the FINAL model.
    log("### Final scoring ###")
    results = []
    for task, rarest in task_list:
        task_risk = min(args.risk_per_class, (rarest - 1) // 2)
        task_audit = min(args.audit_per_class, rarest - 1 - task_risk)
        parts = prepare_task_partitions(
            args.data_root, task.name, cap_per_class=args.cap_per_class,
            risk_per_class=task_risk, audit_per_class=task_audit, seed=args.seed)
        labels = np.array([task.labels.index(ex.label) for ex in parts.audit])
        L = collect_logits(model, tokenizer, list(parts.audit), task, device,
                           max_source=args.max_source,
                           batch_size=args.eval_batch_size)
        results.append({
            "task": task.name,
            "R": float(balanced_accuracy(L, labels, None)),
            "n_audit": int(len(parts.audit)), "K": int(len(task.labels)),
        })
        log(f"{task.name:12s} R={results[-1]['R']:.4f}")

    payload = {"method": args.method, "seed": args.seed, "rank": args.rank,
               "rows": results}
    fname = f"baselines_{args.method}_s{args.seed}.json"
    with open(out_dir / fname, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"[{args.method}] mean R = {np.mean([r['R'] for r in results]):.4f}")
    log(f"Saved {fname}")


if __name__ == "__main__":
    main()
