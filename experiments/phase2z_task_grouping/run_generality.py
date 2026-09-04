"""Phase-2Z-F (RUNBOOK 3B): the 2x2 across task orders and backbones.

run_combined.py measures the 2x2 on ONE order (canonical Order-4) and ONE
backbone (T5-large). This script reruns the identical 2x2 under a chosen
`--order` (orders.py) and `--backbone` model id (backbones.py), so the four
effect signs can be checked for generality. Only the TRAINING order and the
model change; eligibility, splits, per-scope offsets, gauge-fixing, and scoring
are byte-identical to run_combined. Any change in the 2x2 is therefore a pure
order/backbone effect.

Output: generality_{order}_{tag}_s{seed}.json, same per-task schema as
combined_s*.json plus an "order"/"backbone"/"family" header row. Aggregated by
analyze_generality.py across (order, backbone) cells.
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
from peft import LoraConfig, get_peft_model, TaskType  # noqa: E402

from experiments.phase2i_anchored_cvar.order4_data import (  # noqa: E402
    ORDER4_TASKS,
    prepare_task_partitions,
)
from experiments.phase2j_offset_conflict.offsets import (  # noqa: E402
    balanced_accuracy,
    TaskLogits,
)
from experiments.phase2j_offset_conflict.run_probe import log  # noqa: E402
from experiments.phase2z_task_grouping.run_grouping import (  # noqa: E402
    GROUP_OF,
    GROUP_NAMES,
    make_optimizer,
    eligible_tasks,
)
from experiments.phase2z_task_grouping.run_combined import (  # noqa: E402
    fit_scope_offsets,
)
from experiments.phase2z_task_grouping.orders import reorder_task_list  # noqa: E402
from experiments.phase2z_task_grouping import backbones  # noqa: E402


def build_adapters_bb(base_model, *, family, shared_rank, group_rank,
                      lora_alpha, lora_dropout, lora_key):
    """Backbone-aware build_adapters: same shared+K-group wiring as
    run_grouping.build_adapters, but with target modules from the backbone."""
    tt = TaskType.SEQ_2_SEQ_LM if family == "seq2seq" else TaskType.CAUSAL_LM
    tgt = backbones.target_modules(lora_key)
    shared_cfg = LoraConfig(task_type=tt, r=shared_rank, lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout, target_modules=tgt)
    model = get_peft_model(base_model, shared_cfg, adapter_name="shared")
    grp_cfg = LoraConfig(task_type=tt, r=group_rank, lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout, target_modules=tgt)
    for name in GROUP_NAMES:
        model.add_adapter(f"grp_{name}", grp_cfg)
    return model


def train_stream_bb(model, tokenizer, device, family, args, task_list,
                    *, adapter_for):
    """train_stream (run_grouping) but routed through the backbone trainer."""
    for task, rarest in task_list:
        model.set_adapter(adapter_for(task))
        task_risk = min(args.risk_per_class, (rarest - 1) // 2)
        task_audit = min(args.audit_per_class, rarest - 1 - task_risk)
        parts = prepare_task_partitions(
            args.data_root, task.name, cap_per_class=args.cap_per_class,
            risk_per_class=task_risk, audit_per_class=task_audit, seed=args.seed)
        train_args = SimpleNamespace(
            batch_size=args.batch_size, grad_accum=args.grad_accum, lr=args.lr,
            epochs=args.epochs, max_source=args.max_source,
            max_target=args.max_target, seed=args.seed)
        optimizer = make_optimizer(model, args.lr)
        log(f"  train {task.name} on '{adapter_for(task)}' ({len(parts.update)} ex)")
        backbones.train_one_task(model, tokenizer, list(parts.update), device,
                                 family, args=train_args, optimizer=optimizer)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default="google-t5/t5-large")
    ap.add_argument("--order", default="canonical")
    ap.add_argument("--tag", default=None,
                    help="short backbone tag for filenames (default: derived)")

    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--cap-per-class", type=int, default=400)
    ap.add_argument("--max-source", type=int, default=512)
    ap.add_argument("--max-target", type=int, default=128)

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
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    log("=" * 80)
    log("Phase-2Z-F (RUNBOOK 3B): 2x2 across orders x backbones")
    log(f"  order={args.order}  backbone={args.model}  tag={tag}  seed={args.seed}")
    n_groups = len(GROUP_NAMES)
    if args.shared_rank != n_groups * args.group_rank:
        log("WARNING: budgets differ; NOT a fixed-budget comparison!")
    log("=" * 80)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    base, tokenizer, family, lora_key = backbones.load(args.model, dtype)
    model = build_adapters_bb(
        base, family=family, shared_rank=args.shared_rank,
        group_rank=args.group_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, lora_key=lora_key)

    task_list = eligible_tasks(tokenizer, args.data_root, ORDER4_TASKS)
    task_list = reorder_task_list(task_list, args.order)
    log(f"Train order [{args.order}]: {[t.name for t, _ in task_list]}")

    log("### Pass 1: SHARED adapter ###")
    train_stream_bb(model, tokenizer, device, family, args, task_list,
                    adapter_for=lambda t: "shared")
    log("### Pass 2: GROUP adapters ###")
    train_stream_bb(model, tokenizer, device, family, args, task_list,
                    adapter_for=lambda t: f"grp_{GROUP_OF[t.name]}")

    # Collect risk (offset fit) + audit (scoring) logits under both regimes.
    risk_shared, risk_grouped, audit_cache = {}, {}, {}
    for task, rarest in task_list:
        task_risk = min(args.risk_per_class, (rarest - 1) // 2)
        task_audit = min(args.audit_per_class, rarest - 1 - task_risk)
        parts = prepare_task_partitions(
            args.data_root, task.name, cap_per_class=args.cap_per_class,
            risk_per_class=task_risk, audit_per_class=task_audit, seed=args.seed)
        labels_risk = np.array([task.labels.index(ex.label) for ex in parts.risk])
        labels_audit = np.array([task.labels.index(ex.label) for ex in parts.audit])
        audit_cache[task.name] = (task, list(parts.audit), labels_audit)

        model.set_adapter("shared")
        lr_ = backbones.collect_logits(model, tokenizer, list(parts.risk), task,
                                       device, family, max_source=args.max_source,
                                       batch_size=args.eval_batch_size)
        risk_shared[task.name] = TaskLogits(task.name, tuple(task.labels), lr_,
                                            labels_risk)
        model.set_adapter(f"grp_{GROUP_OF[task.name]}")
        lg_ = backbones.collect_logits(model, tokenizer, list(parts.risk), task,
                                       device, family, max_source=args.max_source,
                                       batch_size=args.eval_batch_size)
        risk_grouped[task.name] = TaskLogits(task.name, tuple(task.labels), lg_,
                                             labels_risk)

    off_shared = fit_scope_offsets(risk_shared)
    off_grouped = fit_scope_offsets(risk_grouped)

    results = []
    for task, rarest in task_list:
        _, audit_ex, labels = audit_cache[task.name]
        model.set_adapter("shared")
        L_sh = backbones.collect_logits(model, tokenizer, audit_ex, task, device,
                                        family, max_source=args.max_source,
                                        batch_size=args.eval_batch_size)
        model.set_adapter(f"grp_{GROUP_OF[task.name]}")
        L_gp = backbones.collect_logits(model, tokenizer, audit_ex, task, device,
                                        family, max_source=args.max_source,
                                        batch_size=args.eval_batch_size)
        results.append({
            "task": task.name, "group": GROUP_OF[task.name],
            "R_sh": float(balanced_accuracy(L_sh, labels, None)),
            "R_sh_off": float(balanced_accuracy(L_sh, labels, off_shared[task.name])),
            "R_gp": float(balanced_accuracy(L_gp, labels, None)),
            "R_gp_off": float(balanced_accuracy(L_gp, labels, off_grouped[task.name])),
            "n_audit": int(len(audit_ex)), "K": int(len(task.labels)),
        })

    payload = {
        "order": args.order, "backbone": args.model, "family": family,
        "tag": tag, "seed": args.seed, "rows": results,
    }
    fname = f"generality_{args.order}_{tag}_s{args.seed}.json"
    with open(out_dir / fname, "w") as f:
        json.dump(payload, f, indent=2)

    def m(k):
        return float(np.mean([r[k] for r in results]))
    log("")
    log("=" * 80)
    log(f"[{args.order} / {tag}]  R_sh={m('R_sh'):.4f}  R_sh_off={m('R_sh_off'):.4f}"
        f"  R_gp={m('R_gp'):.4f}  R_gp_off={m('R_gp_off'):.4f}")
    log(f"  e2e (gp_off - sh)          : {(m('R_gp_off')-m('R_sh'))*100:+.2f}pp")
    log(f"  repr (gp_off - sh_off)     : {(m('R_gp_off')-m('R_sh_off'))*100:+.2f}pp")
    log(f"  group_alone (gp - sh)      : {(m('R_gp')-m('R_sh'))*100:+.2f}pp")
    log(f"  offset_alone (sh_off - sh) : {(m('R_sh_off')-m('R_sh'))*100:+.2f}pp")
    log("=" * 80)
    log(f"Saved {fname}")


if __name__ == "__main__":
    main()
