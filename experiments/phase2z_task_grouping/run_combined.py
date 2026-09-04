"""Phase-2Z-C: Combined two-constraint method under a fixed budget.

The paper identifies TWO capacity constraints in task-agnostic continual PEFT:
  * output layer   -- the identification gap, recoverable by a per-scope offset
                      (free at inference: reads only the option list, 0 stored
                      floats beyond a tiny table, no task index);
  * representation  -- deep forgetting in the adapter, the paper's self-declared
                      largest open gap (limitation #9), untouched by the three
                      penalty/anchoring interventions it tested (limitation #4).

This script proposes and measures a method that attacks BOTH at a FIXED
parameter budget:
  representation  <- task-family GROUPING adapters (K groups, rank R/K each;
                     total params == one shared rank-R adapter, exactly);
  output          <- a per-scope offset applied on top (the paper's +1.51pp).

The clean fixed-budget comparison is the 2x2:

                        no offset        + per-scope offset
    shared  (rank R)    R_sh             R_sh_off
    grouped (K x R/K)   R_gp             R_gp_off      <- proposed method

Every cell has identical trainable-parameter budget. The offset column adds
only a small per-scope table (<= |union of verbalizer tokens| scalars), never a
task index, never re-read training examples at test time beyond the frozen risk
split used to fit the offset (disjoint from the scored audit split).

The representation-layer effect of grouping is (R_gp_off - R_sh_off); the
end-to-end method gain over the shared+no-offset baseline is (R_gp_off - R_sh).
All numbers trace to this frozen run; nothing is derived or assumed.
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
    fit_offset,
    fit_shared_offset,
    gauge_fix,
    TaskLogits,
)
from experiments.phase2j_offset_conflict.run_probe import (  # noqa: E402
    collect_logits,
    log,
    precondition_ok,
    train_one_task,
)

# Reuse the a-priori grouping from run_grouping (single source of truth).
from experiments.phase2z_task_grouping.run_grouping import (  # noqa: E402
    GROUP_OF,
    GROUP_NAMES,
    build_adapters,
    make_optimizer,
    eligible_tasks,
    train_stream,
)


def fit_scope_offsets(task_logits_risk):
    """Fit one gauge-fixed offset per scope (verbalizer-keyed) on risk logits.

    task_logits_risk: dict task_name -> TaskLogits (on the risk split).
    Returns dict task_name -> np.ndarray offset (aligned to that task's
    verbalizer), matching the paper's per-scope rung: scope-mates share one
    offset minimizing worst-task loss; a singleton scope's offset is that
    task's own optimum (Chebyshev centre of a 1-element set == the element).
    """
    # Group tasks by their verbalizer tuple (the scope key).
    scopes = {}
    for name, tl in task_logits_risk.items():
        scopes.setdefault(tuple(tl.verbalizer), []).append(tl)

    offset_for = {}
    for key, members in scopes.items():
        if len(members) == 1:
            tl = members[0]
            off = gauge_fix(fit_offset(tl.logits, tl.labels))
            offset_for[tl.task] = off
        else:
            shared_tok = fit_shared_offset(members)  # token -> scalar
            for tl in members:
                sub = np.array([shared_tok.get(t, 0.0) for t in tl.verbalizer])
                offset_for[tl.task] = gauge_fix(sub)
    return offset_for


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
    log("Phase-2Z-C: Combined two-constraint method (grouping + per-scope offset)")
    log("=" * 80)
    n_groups = len(GROUP_NAMES)
    log(f"Seed: {args.seed}")
    log(f"Fixed-budget: shared_rank={args.shared_rank} vs "
        f"{n_groups} x group_rank={args.group_rank} = {n_groups*args.group_rank}")
    if args.shared_rank != n_groups * args.group_rank:
        log("WARNING: budgets differ; NOT a fixed-budget comparison!")
    log("=" * 80)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    log(f"Device: {device}, dtype: {dtype}")

    base = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = build_adapters(
        base, shared_rank=args.shared_rank, group_rank=args.group_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    )

    task_list = eligible_tasks(tokenizer, args.data_root, ORDER4_TASKS)
    log(f"Eligible tasks: {[t.name for t, _ in task_list]}")

    log("")
    log("### Pass 1: SHARED adapter (rank %d) over all tasks ###" % args.shared_rank)
    train_stream(model, tokenizer, device, args, task_list,
                 adapter_for=lambda t: "shared")

    log("")
    log("### Pass 2: GROUP adapters (rank %d each) ###" % args.group_rank)
    train_stream(model, tokenizer, device, args, task_list,
                 adapter_for=lambda t: f"grp_{GROUP_OF[t.name]}")

    # ------------------------------------------------------------------
    # Collect risk + audit logits under BOTH adapter regimes.
    # ------------------------------------------------------------------
    log("")
    log("### Collecting logits (risk for offset fit, audit for scoring) ###")

    risk_shared, risk_grouped = {}, {}
    audit_cache = {}  # task -> (audit_examples, labels)
    for task, rarest in task_list:
        task_risk = min(args.risk_per_class, (rarest - 1) // 2)
        task_audit = min(args.audit_per_class, rarest - 1 - task_risk)
        parts = prepare_task_partitions(
            args.data_root, task.name,
            cap_per_class=args.cap_per_class,
            risk_per_class=task_risk, audit_per_class=task_audit,
            seed=args.seed,
        )
        labels_risk = np.array([task.labels.index(ex.label) for ex in parts.risk])
        labels_audit = np.array([task.labels.index(ex.label) for ex in parts.audit])
        audit_cache[task.name] = (task, parts.audit, labels_audit)

        model.set_adapter("shared")
        lr = collect_logits(model, tokenizer, parts.risk, task, device,
                            max_source=args.max_source,
                            batch_size=args.eval_batch_size)
        risk_shared[task.name] = TaskLogits(task.name, tuple(task.labels),
                                            lr, labels_risk)

        model.set_adapter(f"grp_{GROUP_OF[task.name]}")
        lg = collect_logits(model, tokenizer, parts.risk, task, device,
                            max_source=args.max_source,
                            batch_size=args.eval_batch_size)
        risk_grouped[task.name] = TaskLogits(task.name, tuple(task.labels),
                                             lg, labels_risk)

    # Fit per-scope offsets separately for each adapter regime (offsets must
    # match the model they are applied to).
    off_shared = fit_scope_offsets(risk_shared)
    off_grouped = fit_scope_offsets(risk_grouped)

    # ------------------------------------------------------------------
    # Score the 2x2 on the audit split.
    # ------------------------------------------------------------------
    log("")
    log("=" * 80)
    log("Final Evaluation: 2x2 (adapter regime x offset)")
    log("=" * 80)

    results = []
    for task, rarest in task_list:
        _, audit_ex, labels = audit_cache[task.name]

        model.set_adapter("shared")
        L_sh = collect_logits(model, tokenizer, audit_ex, task, device,
                              max_source=args.max_source,
                              batch_size=args.eval_batch_size)
        model.set_adapter(f"grp_{GROUP_OF[task.name]}")
        L_gp = collect_logits(model, tokenizer, audit_ex, task, device,
                              max_source=args.max_source,
                              batch_size=args.eval_batch_size)

        R_sh = balanced_accuracy(L_sh, labels, None)
        R_sh_off = balanced_accuracy(L_sh, labels, off_shared[task.name])
        R_gp = balanced_accuracy(L_gp, labels, None)
        R_gp_off = balanced_accuracy(L_gp, labels, off_grouped[task.name])

        results.append({
            "task": task.name,
            "group": GROUP_OF[task.name],
            "R_sh": float(R_sh),
            "R_sh_off": float(R_sh_off),
            "R_gp": float(R_gp),
            "R_gp_off": float(R_gp_off),
            "n_audit": int(len(audit_ex)),
            "K": int(len(task.labels)),
        })
        log(f"{task.name:12s} [{GROUP_OF[task.name]:14s}] "
            f"sh={R_sh:.4f} sh+off={R_sh_off:.4f} "
            f"gp={R_gp:.4f} gp+off={R_gp_off:.4f}")

    with open(out_dir / f"results_s{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2)

    def m(k):
        return float(np.mean([r[k] for r in results]))
    log("")
    log("=" * 80)
    log(f"R_sh      (shared, no offset)      : {m('R_sh'):.4f}")
    log(f"R_sh_off  (shared + per-scope off) : {m('R_sh_off'):.4f}")
    log(f"R_gp      (grouped, no offset)     : {m('R_gp'):.4f}")
    log(f"R_gp_off  (grouped + per-scope off): {m('R_gp_off'):.4f}   <- METHOD")
    log("-" * 80)
    log(f"grouping repr-layer gain (gp_off - sh_off): "
        f"{(m('R_gp_off')-m('R_sh_off'))*100:+.2f}pp")
    log(f"end-to-end method gain   (gp_off - sh)    : "
        f"{(m('R_gp_off')-m('R_sh'))*100:+.2f}pp")
    log(f"grouping alone           (gp - sh)        : "
        f"{(m('R_gp')-m('R_sh'))*100:+.2f}pp")
    log(f"offset alone on shared   (sh_off - sh)    : "
        f"{(m('R_sh_off')-m('R_sh'))*100:+.2f}pp")
    log("=" * 80)
    log(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
