"""Phase-2J wiring smoke: does the real T5+LoRA stack produce the quantities
the offset-conflict probe needs?

This is deliberately tiny (a few dozen steps, two tasks) and proves plumbing,
not science.  It answers six questions, each of which would silently corrupt
the probe if wrong:

1. Do the pinned Order-4 files load and partition disjointly on this box?
2. Does LoRA attach to the intended modules, and is only LoRA trainable?
3. Are the verbalizer strings single-token under this tokenizer?  (The whole
   offset formalism assumes one decoding step decides the label.)
4. Can we read first-decode-step logits restricted to verbalizer tokens, and
   do they move under a few optimizer steps?
5. Do the offsets.py primitives run on real logits and return a sane Delta_id?
6. What is the measured throughput, so the probe's GPU-hour budget is real?

Nothing here touches test.json.  All data comes from train-derived partitions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

# This box has no direct route to huggingface.co; the working mirror and the
# shared-disk cache location must be set before transformers is imported.
os.environ.setdefault("HF_HOME", "/mnt/data/wenbin/iclr26/models/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.phase2i_anchored_cvar import order4_data as od  # noqa: E402
from experiments.phase2j_offset_conflict.offsets import (  # noqa: E402
    TaskLogits,
    identification_gap,
)

RESULT: dict[str, object] = {}


def check(name: str, ok: bool, detail: object = "") -> bool:
    """Record and print one smoke assertion without aborting the run."""

    RESULT.setdefault("checks", {})[name] = {"ok": bool(ok), "detail": detail}
    print("[%s] %s  %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    return bool(ok)


def verbalizer_token_ids(tokenizer, labels) -> dict[str, int]:
    """Map each verbalizer string to its FIRST decoded token id.

    T5 puts no BOS on the target side, so the first token of the label encoding
    is exactly what the first decode step must emit.  Order-4 verbalizers are
    not all single-token under T5's sentencepiece (``False`` is three pieces),
    so the precondition the offset formalism actually needs is weaker and is
    checked separately: *distinct first tokens within a task*.  Given distinct
    first tokens, greedy decoding's first step already determines which label
    is produced, which is the decision the offset acts on.
    """

    mapping: dict[str, int] = {}
    for label in labels:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        mapping[label] = int(ids[0])
    return mapping


def encode_batch(tokenizer, prompts, targets, device, max_source, max_target):
    enc = tokenizer(
        list(prompts),
        max_length=max_source,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    lab = tokenizer(
        list(targets),
        max_length=max_target,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    label_ids = lab["input_ids"].clone()
    label_ids[label_ids == tokenizer.pad_token_id] = -100
    return (
        {k: v.to(device) for k, v in enc.items()},
        label_ids.to(device),
    )


@torch.no_grad()
def first_step_logits(model, tokenizer, examples, task, device, max_source, batch_size):
    """Return per-example logits over the task's verbalizer tokens.

    Only the first decoder step is scored: the decoder is fed the single
    start token, so the returned row is the distribution the model would
    argmax over to emit its first label token.
    """

    verb = verbalizer_token_ids(tokenizer, task.labels)
    columns = [verb[label] for label in task.labels]
    out = []
    model.eval()
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        # OfficialExample.prompt is the pinned O-LoRA input string already.
        enc = tokenizer(
            [example.prompt for example in chunk],
            max_length=max_source,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        start_ids = torch.full(
            (len(chunk), 1),
            model.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        logits = model(**enc, decoder_input_ids=start_ids).logits[:, 0, :]
        out.append(logits[:, columns].float().cpu())
    return torch.cat(out, dim=0).numpy() if out else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/mnt/data/wenbin/iclr26/data/order4")
    # Canonical hub ids: the legacy "t5-large" alias resolves through a redirect
    # the mirror does not serve, so name the org explicitly.
    parser.add_argument("--model", default="google-t5/t5-large")
    parser.add_argument("--out", default="/mnt/data/wenbin/iclr26/runs/phase2j_smoke")
    parser.add_argument("--tasks", default="WiC,QQP")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--eval-n", type=int, default=64)
    parser.add_argument("--max-source", type=int, default=512)
    parser.add_argument("--max-target", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    # The box is shared and often leaves only a few GiB per card, so the weight
    # dtype has to be selectable.  LoRA parameters stay fp32 either way.
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "bfloat16"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULT["args"] = vars(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    check("cuda-available", device == "cuda", torch.cuda.get_device_name(0) if device == "cuda" else "cpu")

    # --- 1. data ---------------------------------------------------------
    task_names = [name.strip() for name in args.tasks.split(",") if name.strip()]
    partitions, specs = {}, {}
    for name in task_names:
        spec = next(task for task in od.ORDER4_TASKS if task.name == name)
        parts = od.prepare_task_partitions(
            args.data_root,
            name,
            cap_per_class=256,
            risk_per_class=64,
            audit_per_class=64,
            seed=args.seed,
        )
        parts.assert_disjoint()
        partitions[name] = parts
        specs[name] = spec
    check(
        "data-load-disjoint",
        True,
        {n: (len(p.update), len(p.risk), len(p.audit)) for n, p in partitions.items()},
    )

    # --- 2. model + LoRA -------------------------------------------------
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(args.model, legacy=False)
    weight_dtype = getattr(torch, args.dtype)
    model = T5ForConditionalGeneration.from_pretrained(args.model, dtype=weight_dtype)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora)
    model.to(device)
    # Keep the trainable LoRA parameters in fp32 even when the frozen backbone
    # is bf16: bf16 AdamW on rank-8 factors loses too much precision to trust
    # the small logit shifts this probe measures.
    if weight_dtype is not torch.float32:
        for _, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()

    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    n_lora = sum(1 for n, _ in trainable if "lora_" in n)
    check(
        "lora-attached",
        n_lora > 0 and all("lora_" in n for n, _ in trainable),
        {
            "lora_tensors": n_lora,
            "trainable": sum(c for _, c in trainable),
            "total": total,
            "pct": round(100.0 * sum(c for _, c in trainable) / total, 4),
        },
    )

    # --- 3. verbalizers must be single-token ------------------------------
    verb_report, multi = {}, []
    for name, spec in specs.items():
        entry = {}
        for label in spec.labels:
            ids = tokenizer(label, add_special_tokens=False)["input_ids"]
            entry[label] = {"ids": [int(i) for i in ids], "n": len(ids)}
            if len(ids) != 1:
                multi.append("%s/%s(%d)" % (name, label, len(ids)))
        verb_report[name] = entry
    RESULT["verbalizers"] = verb_report
    RESULT["multi_piece_verbalizers"] = multi
    # Informational, not a gate: multi-piece labels are fine as long as the
    # first pieces separate.  Recorded because it constrains how the probe may
    # describe "the" logit of a label.
    print("multi-piece verbalizers: %s" % (multi or "none"), flush=True)

    # The real precondition: distinct first tokens within a task, otherwise the
    # first decode step cannot separate the labels and no offset on it can.
    distinct = {}
    for name, spec in specs.items():
        first = [verb_report[name][label]["ids"][0] for label in spec.labels]
        distinct[name] = len(set(first)) == len(first)
    check("verbalizer-first-tokens-distinct", all(distinct.values()), distinct)

    # --- 4/5. logits before and after a few steps ------------------------
    first_task = task_names[0]
    spec = specs[first_task]
    eval_examples = list(partitions[first_task].audit)[: args.eval_n]

    t0 = time.time()
    logits_before = first_step_logits(
        model, tokenizer, eval_examples, spec, device, args.max_source, args.batch_size
    )
    eval_seconds = time.time() - t0
    check(
        "logits-readable",
        logits_before is not None and logits_before.shape == (len(eval_examples), len(spec.labels)),
        {"shape": None if logits_before is None else list(logits_before.shape)},
    )

    update = list(partitions[first_task].update)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    model.train()
    losses, step_times = [], []
    cursor = 0
    for step in range(args.steps):
        step_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            chunk = []
            while len(chunk) < args.batch_size:
                chunk.append(update[cursor % len(update)])
                cursor += 1
            enc, label_ids = encode_batch(
                tokenizer,
                [example.prompt for example in chunk],
                [example.label for example in chunk],
                device,
                args.max_source,
                args.max_target,
            )
            loss = model(**enc, labels=label_ids).loss / args.grad_accum
            loss.backward()
            losses.append(loss.detach().item() * args.grad_accum)
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.time() - step_start)

    logits_after = first_step_logits(
        model, tokenizer, eval_examples, spec, device, args.max_source, args.batch_size
    )
    moved = float(np.abs(logits_after - logits_before).mean())
    check("logits-move-under-training", moved > 1e-3, {"mean_abs_delta": round(moved, 5)})
    check(
        "loss-finite-and-decreasing",
        all(np.isfinite(losses)) and np.mean(losses[-5:]) < np.mean(losses[:5]),
        {"first5": round(float(np.mean(losses[:5])), 4),
         "last5": round(float(np.mean(losses[-5:])), 4)},
    )

    # --- 5. offsets.py primitives on real logits --------------------------
    label_index = {label: i for i, label in enumerate(spec.labels)}
    y = np.array([label_index[example.label] for example in eval_examples])
    check(
        "eval-labels-in-verbalizer",
        len(y) == len(eval_examples),
        {"classes": sorted(set(int(v) for v in y))},
    )

    tl_after = TaskLogits(first_task, tuple(spec.labels), logits_after, y)
    from experiments.phase2j_offset_conflict.offsets import fit_offset, fit_shared_offset

    per_task = fit_offset(logits_after, y)
    shared = fit_shared_offset([tl_after])
    gap = identification_gap(tl_after, per_task_offset=per_task,
                             shared_offset=shared)
    RESULT["identification_gap_single_task"] = {k: round(float(v), 5) for k, v in gap.items()}
    # With a single fitted task the shared offset can match the oracle, so the
    # only invariant worth asserting is the ordering, not a positive Delta_id.
    check("gap-ordering", gap["R_orc"] >= gap["R_shr"] - 1e-9
          and gap["R_orc"] >= gap["R_raw"] - 1e-9, RESULT["identification_gap_single_task"])

    # --- 6. throughput ----------------------------------------------------
    median_step = float(np.median(step_times))
    examples_per_step = args.batch_size * args.grad_accum
    RESULT["throughput"] = {
        "dtype": args.dtype,
        "median_step_seconds": round(median_step, 4),
        "train_examples_per_second": round(examples_per_step / median_step, 2),
        "eval_examples_per_second": round(len(eval_examples) / eval_seconds, 2),
        "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)
        if device == "cuda" else None,
    }
    print("throughput " + json.dumps(RESULT["throughput"]), flush=True)

    checks = RESULT["checks"]
    RESULT["all_passed"] = all(entry["ok"] for entry in checks.values())
    (out_dir / "smoke_report.json").write_text(
        json.dumps(RESULT, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("WIRING_SMOKE %s report=%s"
          % ("PASS" if RESULT["all_passed"] else "FAIL", out_dir / "smoke_report.json"),
          flush=True)
    return 0 if RESULT["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
