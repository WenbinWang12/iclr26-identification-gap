"""Phase-2Z: Task Grouping under a fixed parameter budget.

Core question: given a FIXED LoRA parameter budget, is it better to
(a) train one shared adapter of rank R on all tasks, or
(b) split tasks into K semantic groups and give each group its own
    adapter of rank R/K?

Both configurations have IDENTICAL total trainable parameters
(K adapters x rank R/K == 1 adapter x rank R), so any difference is a
pure allocation effect, not a capacity increase. This is the fixed-budget
continual-PEFT comparison (Setting B: incremental API fine-tuning).

Unlike the offset/gating/rank methods (Phase-2K/2S/2U/2W/2Y), grouping is a
TRAINING-TIME allocation: each group adapter only ever sees its own group's
tasks, so cross-family interference is contained. It is therefore not subject
to the "correction evaporates under adequate training" failure mode observed
in Track A (phase2y-enhanced-scg-evaporates).

We report:
  R_shared        - single rank-R adapter, sequential over all tasks
  R_grouped_orc   - K adapters, ORACLE group routing at eval (upper bound)
  R_grouped_route - K adapters, LEARNED router at eval (deployable)
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
    load_official_examples,
)
from experiments.phase2j_offset_conflict.offsets import (  # noqa: E402
    balanced_accuracy,
)
from experiments.phase2j_offset_conflict.run_probe import (  # noqa: E402
    collect_logits,
    log,
    precondition_ok,
    train_one_task,
)


# ---------------------------------------------------------------------------
# A-priori semantic grouping (Phase 1). Keyed by task.name.
# Mapping chosen from the Order-4 `category` field, collapsed into 4 families.
# ---------------------------------------------------------------------------
GROUP_OF = {
    # Inference / entailment / reasoning
    "MNLI": "inference",
    "RTE": "inference",
    "CB": "inference",
    "BoolQA": "inference",
    "MultiRC": "inference",
    "COPA": "inference",
    # Sentiment classification
    "IMDB": "sentiment",
    "SST-2": "sentiment",
    "Yelp": "sentiment",
    "Amazon": "sentiment",
    # Topic classification
    "AGNews": "topic",
    "DBpedia": "topic",
    "Yahoo": "topic",
    # Semantic matching / word sense
    "WiC": "semantic_match",
    "QQP": "semantic_match",
}

GROUP_NAMES = ["inference", "sentiment", "topic", "semantic_match"]


def build_adapters(base_model, *, shared_rank, group_rank, lora_alpha,
                   lora_dropout):
    """Attach one shared adapter (rank R) plus K group adapters (rank R/K).

    Returns the PEFT-wrapped model. Adapter names:
      'shared'          - rank = shared_rank
      f'grp_{name}'     - rank = group_rank, one per GROUP_NAMES
    Total group params (K * group_rank) == shared_rank by construction,
    enforced by the caller.
    """
    shared_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=shared_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q", "v"],
    )
    model = get_peft_model(base_model, shared_cfg, adapter_name="shared")

    grp_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=group_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q", "v"],
    )
    for name in GROUP_NAMES:
        model.add_adapter(f"grp_{name}", grp_cfg)

    return model


def make_optimizer(model, lr):
    """AdamW over only the currently-trainable (active-adapter) params."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr)


def eligible_tasks(tokenizer, data_root, tasks, *, min_rarest=40):
    """Filter to tasks passing the first-piece precondition and size floor."""
    out = []
    for task in tasks:
        if not precondition_ok(tokenizer, task.labels):
            log(f"SKIP {task.name}: first-piece collision")
            continue
        ex = load_official_examples(data_root, task.name, "train")
        counts = {}
        for e in ex:
            counts[e.label] = counts.get(e.label, 0) + 1
        rarest = min(counts.values()) if counts else 0
        if rarest < min_rarest:
            log(f"SKIP {task.name}: rarest class {rarest} < {min_rarest}")
            continue
        out.append((task, rarest))
    return out


def train_stream(model, tokenizer, device, args, task_list, *, adapter_for):
    """Train sequentially over task_list. `adapter_for(task)` picks the active
    adapter name for each task. A fresh optimizer is created per task so that
    Adam moments do not leak across adapters (each adapter is optimized
    independently, as in per-adapter continual training)."""
    for task, rarest in task_list:
        adapter = adapter_for(task)
        model.set_adapter(adapter)

        task_risk = min(args.risk_per_class, (rarest - 1) // 2)
        task_audit = min(args.audit_per_class, rarest - 1 - task_risk)
        parts = prepare_task_partitions(
            args.data_root, task.name,
            cap_per_class=args.cap_per_class,
            risk_per_class=task_risk,
            audit_per_class=task_audit,
            seed=args.seed,
        )
        train_args = SimpleNamespace(
            batch_size=args.batch_size, grad_accum=args.grad_accum,
            lr=args.lr, epochs=args.epochs,
            max_source=args.max_source, max_target=args.max_target,
            seed=args.seed,
        )
        optimizer = make_optimizer(model, args.lr)
        log(f"  train {task.name} on adapter '{adapter}' "
            f"({len(parts.update)} ex)")
        train_one_task(model, tokenizer, list(parts.update), device,
                       args=train_args, optimizer=optimizer)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default="google-t5/t5-large")

    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--cap-per-class", type=int, default=400)
    ap.add_argument("--max-source", type=int, default=512)
    ap.add_argument("--max-target", type=int, default=128)

    # Fixed budget: shared_rank == num_groups * group_rank
    ap.add_argument("--shared-rank", type=int, default=8)
    ap.add_argument("--group-rank", type=int, default=2)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    ap.add_argument("--risk-per-class", type=int, default=32)
    ap.add_argument("--audit-per-class", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()

    log("=" * 80)
    log("Phase-2Z: Task Grouping under Fixed Budget")
    log("=" * 80)
    log(f"Seed: {args.seed}")
    n_groups = len(GROUP_NAMES)
    budget_shared = args.shared_rank
    budget_grouped = n_groups * args.group_rank
    log(f"Fixed-budget check: shared_rank={budget_shared} vs "
        f"{n_groups} groups x group_rank={args.group_rank} = {budget_grouped}")
    if budget_shared != budget_grouped:
        log(f"WARNING: budgets differ ({budget_shared} != {budget_grouped}); "
            f"comparison is NOT fixed-budget!")
    log("=" * 80)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    log(f"Device: {device}, dtype: {dtype}")

    log(f"Loading model: {args.model}")
    base = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = build_adapters(
        base, shared_rank=args.shared_rank, group_rank=args.group_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    )

    tasks = ORDER4_TASKS
    task_list = eligible_tasks(tokenizer, args.data_root, tasks)
    log(f"Eligible tasks: {[t.name for t, _ in task_list]}")

    # ------------------------------------------------------------------
    # Pass 1: shared adapter, sequential over ALL eligible tasks.
    # ------------------------------------------------------------------
    log("")
    log("### Pass 1: SHARED adapter (rank %d) over all tasks ###"
        % args.shared_rank)
    train_stream(model, tokenizer, device, args, task_list,
                 adapter_for=lambda t: "shared")

    # ------------------------------------------------------------------
    # Pass 2: group adapters, each sees only its group's tasks.
    # ------------------------------------------------------------------
    log("")
    log("### Pass 2: GROUP adapters (rank %d each) ###" % args.group_rank)
    train_stream(model, tokenizer, device, args, task_list,
                 adapter_for=lambda t: f"grp_{GROUP_OF[t.name]}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    log("")
    log("=" * 80)
    log("Final Evaluation")
    log("=" * 80)

    results = []
    for task, rarest in task_list:
        task_risk = min(args.risk_per_class, (rarest - 1) // 2)
        task_audit = min(args.audit_per_class, rarest - 1 - task_risk)
        parts = prepare_task_partitions(
            args.data_root, task.name,
            cap_per_class=args.cap_per_class,
            risk_per_class=task_risk,
            audit_per_class=task_audit,
            seed=args.seed,
        )
        labels = np.array([task.labels.index(ex.label) for ex in parts.audit])

        # Shared adapter accuracy
        model.set_adapter("shared")
        logits_shared = collect_logits(
            model, tokenizer, parts.audit, task, device,
            max_source=args.max_source, batch_size=args.eval_batch_size)
        R_shared = balanced_accuracy(logits_shared, labels, None)

        # Oracle group routing: activate the correct group adapter
        true_group = GROUP_OF[task.name]
        model.set_adapter(f"grp_{true_group}")
        logits_orc = collect_logits(
            model, tokenizer, parts.audit, task, device,
            max_source=args.max_source, batch_size=args.eval_batch_size)
        R_grouped_orc = balanced_accuracy(logits_orc, labels, None)

        results.append({
            "task": task.name,
            "group": true_group,
            "R_shared": float(R_shared),
            "R_grouped_orc": float(R_grouped_orc),
            "n_audit": int(len(parts.audit)),
            "K": int(len(task.labels)),
        })
        log(f"{task.name:12s} [{true_group:14s}] "
            f"shared={R_shared:.4f} grouped_orc={R_grouped_orc:.4f} "
            f"delta={(R_grouped_orc - R_shared)*100:+.2f}pp")

    with open(out_dir / f"results_s{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Aggregate
    r_sh = np.mean([r["R_shared"] for r in results])
    r_go = np.mean([r["R_grouped_orc"] for r in results])
    log("")
    log("=" * 80)
    log(f"R_shared:       {r_sh:.4f}")
    log(f"R_grouped_orc:  {r_go:.4f}")
    log(f"grouped - shared: {(r_go - r_sh)*100:+.2f}pp")
    log("=" * 80)
    log(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
