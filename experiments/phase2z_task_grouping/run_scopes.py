"""Phase-2Z-G (RUNBOOK 3C): the 2x2 on a genuine K_S>=3 multi-task scope.

The paper's per-scope offset (Prop 3) is only *theoretically* closed for binary
scopes (d = K-1 = 1); for d >= 2 the worst-case m-quantization radius q_m is the
open object. But in the eligible Order-4 stream every K>=3 task
(MNLI 3-way, AGNews 4-way, DBpedia 14-way, Yahoo 10-way) has a UNIQUE verbalizer,
so each is a SINGLETON scope: its shared offset trivially equals its own optimum
and q_m == 0. The d >= 2 regime is therefore never exercised by the paper.

This script builds a genuine multi-task, non-binary scope and measures the method
there. Two changes from run_combined, both logged and nothing else touched:

  1. Scopes are keyed by the label *set* (frozenset), not the ordered tuple, so
     tasks that share the SAME 3+ labels in a different order couple into one
     scope. (fit_shared_offset is already token-indexed, so this is correct.)
  2. The eligibility floor is relaxed to `--scope-min-rarest` (default 12) so CB
     (3-way NLI, rare class ~16, excluded by the paper's floor of 40) enters and
     joins MNLI in a K_S = 3 NLI scope. CB's small rare class ⇒ small risk/audit
     for CB only; flagged per task, never averaged away.

Reported: per-scope q_m (m = 1.. , reusing offsets.conflict_statistics) for every
multi-task scope, and the full 2x2 restricted to the K_S >= 3 subset (plus the
all-task 2x2 for reference). All numbers trace to this frozen run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_HOME", "/mnt/data/wenbin/iclr26/models/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # noqa: E402

from experiments.phase2i_anchored_cvar.order4_data import (  # noqa: E402
    ORDER4_TASKS,
    prepare_task_partitions,
)
from experiments.phase2j_offset_conflict.offsets import (  # noqa: E402
    balanced_accuracy,
    conflict_statistics,
    fit_offset,
    fit_shared_offset,
    gauge_fix,
    TaskLogits,
)
from experiments.phase2j_offset_conflict.run_probe import (  # noqa: E402
    collect_logits,
    log,
    first_piece_ids,
)
from experiments.phase2z_task_grouping.run_grouping import (  # noqa: E402
    GROUP_OF,
    GROUP_NAMES,
    build_adapters,
    eligible_tasks,
    train_stream,
)


def fit_scope_offsets_by_set(task_logits):
    """fit_scope_offsets, but scopes are keyed by the label SET, not the ordered
    tuple, so same-labels-different-order tasks (MNLI/CB) genuinely couple.

    Returns (offset_for, scope_of): offset per task (aligned to its own
    verbalizer order) and the scope key (sorted-label tuple) per task.
    """
    scopes = {}
    for name, tl in task_logits.items():
        scopes.setdefault(tuple(sorted(tl.verbalizer)), []).append(tl)

    offset_for, scope_of = {}, {}
    for key, members in scopes.items():
        if len(members) == 1:
            tl = members[0]
            offset_for[tl.task] = gauge_fix(fit_offset(tl.logits, tl.labels))
        else:
            shared_tok = fit_shared_offset(members)
            for tl in members:
                sub = np.array([shared_tok.get(t, 0.0) for t in tl.verbalizer])
                offset_for[tl.task] = gauge_fix(sub)
        for tl in members:
            scope_of[tl.task] = key
    return offset_for, scope_of


def per_task_optima(task_logits):
    """Per-task oracle offsets (fit_offset) for the q_m geometry."""
    return {name: gauge_fix(fit_offset(tl.logits, tl.labels))
            for name, tl in task_logits.items()}


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
    # 3C: relaxed floor to admit CB (3-way NLI, rare class ~16) so the NLI scope
    # has >= 2 members and becomes a genuine K_S >= 3 multi-task scope.
    ap.add_argument("--scope-min-rarest", type=int, default=12)
    return ap.parse_args()


def main():
    args = parse_args()
    log("=" * 80)
    log("Phase-2Z-G (RUNBOOK 3C): 2x2 on a genuine K_S>=3 multi-task scope")
    log(f"  seed={args.seed}  scope-min-rarest={args.scope_min_rarest}")
    n_groups = len(GROUP_NAMES)
    if args.shared_rank != n_groups * args.group_rank:
        log("WARNING: budgets differ; NOT a fixed-budget comparison!")
    log("=" * 80)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    base = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = build_adapters(
        base, shared_rank=args.shared_rank, group_rank=args.group_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

    # Relaxed floor: admits CB into the NLI 3-way scope; all other tasks already
    # clear it, so the only stream change vs the paper is the added CB task.
    task_list = eligible_tasks(tokenizer, args.data_root, ORDER4_TASKS,
                               min_rarest=args.scope_min_rarest)
    names = [t.name for t, _ in task_list]
    log(f"Eligible (relaxed) tasks [{len(names)}]: {names}")
    if "CB" not in names:
        log("NOTE: CB not admitted (precondition/size); no 3-way NLI multi-scope "
            "will form. Reporting whatever multi-task scopes exist.")

    log("### Pass 1: SHARED adapter ###")
    train_stream(model, tokenizer, device, args, task_list,
                 adapter_for=lambda t: "shared")
    log("### Pass 2: GROUP adapters ###")
    train_stream(model, tokenizer, device, args, task_list,
                 adapter_for=lambda t: f"grp_{GROUP_OF[t.name]}")

    risk_shared, risk_grouped, audit_cache = {}, {}, {}
    verbalizers = {}
    for task, rarest in task_list:
        task_risk = min(args.risk_per_class, (rarest - 1) // 2)
        task_audit = min(args.audit_per_class, rarest - 1 - task_risk)
        parts = prepare_task_partitions(
            args.data_root, task.name, cap_per_class=args.cap_per_class,
            risk_per_class=task_risk, audit_per_class=task_audit, seed=args.seed)
        labels_risk = np.array([task.labels.index(ex.label) for ex in parts.risk])
        labels_audit = np.array([task.labels.index(ex.label) for ex in parts.audit])
        audit_cache[task.name] = (task, list(parts.audit), labels_audit)
        verbalizers[task.name] = tuple(task.labels)

        model.set_adapter("shared")
        lr_ = collect_logits(model, tokenizer, list(parts.risk), task, device,
                             max_source=args.max_source,
                             batch_size=args.eval_batch_size)
        risk_shared[task.name] = TaskLogits(task.name, tuple(task.labels),
                                            lr_, labels_risk)
        model.set_adapter(f"grp_{GROUP_OF[task.name]}")
        lg_ = collect_logits(model, tokenizer, list(parts.risk), task, device,
                             max_source=args.max_source,
                             batch_size=args.eval_batch_size)
        risk_grouped[task.name] = TaskLogits(task.name, tuple(task.labels),
                                             lg_, labels_risk)

    off_shared, scope_of = fit_scope_offsets_by_set(risk_shared)
    off_grouped, _ = fit_scope_offsets_by_set(risk_grouped)

    # Scope geometry (q_m) from per-task optima, under each regime. q_m is a
    # property of the trained model's per-task optima, so it differs by regime.
    opt_shared = per_task_optima(risk_shared)
    opt_grouped = per_task_optima(risk_grouped)
    conf_shared = conflict_statistics(opt_shared, verbalizers)
    conf_grouped = conflict_statistics(opt_grouped, verbalizers)

    # scope size (K_S) and membership
    scope_members = {}
    for name, key in scope_of.items():
        scope_members.setdefault(key, []).append(name)
    ksge3_multi = {
        key: mem for key, mem in scope_members.items()
        if len(key) >= 3 and len(mem) >= 2
    }
    log(f"K_S>=3 multi-task scopes: "
        f"{[('|'.join(k), sorted(v)) for k, v in ksge3_multi.items()]}")

    rows = []
    for task, rarest in task_list:
        _, audit_ex, labels = audit_cache[task.name]
        key = scope_of[task.name]
        model.set_adapter("shared")
        L_sh = collect_logits(model, tokenizer, audit_ex, task, device,
                              max_source=args.max_source,
                              batch_size=args.eval_batch_size)
        model.set_adapter(f"grp_{GROUP_OF[task.name]}")
        L_gp = collect_logits(model, tokenizer, audit_ex, task, device,
                              max_source=args.max_source,
                              batch_size=args.eval_batch_size)
        rows.append({
            "task": task.name, "group": GROUP_OF[task.name],
            "scope": "|".join(key), "K_S": int(len(key)),
            "scope_size": int(len(scope_members[key])),
            "in_ksge3_multi": key in ksge3_multi,
            "R_sh": float(balanced_accuracy(L_sh, labels, None)),
            "R_sh_off": float(balanced_accuracy(L_sh, labels, off_shared[task.name])),
            "R_gp": float(balanced_accuracy(L_gp, labels, None)),
            "R_gp_off": float(balanced_accuracy(L_gp, labels, off_grouped[task.name])),
            "n_audit": int(len(audit_ex)), "K": int(len(task.labels)),
        })

    payload = {
        "seed": args.seed, "scope_min_rarest": args.scope_min_rarest,
        "rows": rows,
        "ksge3_multi_scopes": {"|".join(k): sorted(v)
                               for k, v in ksge3_multi.items()},
        "quantization_radius_shared": conf_shared["quantization_radius"],
        "quantization_radius_grouped": conf_grouped["quantization_radius"],
        "omega_shared": conf_shared["omega"],
        "omega_grouped": conf_grouped["omega"],
    }
    fname = f"scopes_s{args.seed}.json"
    with open(out_dir / fname, "w") as f:
        json.dump(payload, f, indent=2)

    def agg(subset, k):
        xs = [r[k] for r in subset]
        return float(np.mean(xs)) if xs else float("nan")

    ksge3 = [r for r in rows if r["in_ksge3_multi"]]
    log("")
    log("=" * 80)
    for label, subset in (("ALL tasks", rows), ("K_S>=3 multi subset", ksge3)):
        if not subset:
            log(f"[{label}] empty")
            continue
        log(f"[{label}] n={len(subset)}  "
            f"R_sh={agg(subset,'R_sh'):.4f} R_sh_off={agg(subset,'R_sh_off'):.4f} "
            f"R_gp={agg(subset,'R_gp'):.4f} R_gp_off={agg(subset,'R_gp_off'):.4f}")
        log(f"    e2e (gp_off-sh)={100*(agg(subset,'R_gp_off')-agg(subset,'R_sh')):+.2f}pp  "
            f"offset_alone (sh_off-sh)={100*(agg(subset,'R_sh_off')-agg(subset,'R_sh')):+.2f}pp")
    log("  per-scope q_m (shared regime):")
    for sc, radii in conf_shared["quantization_radius"].items():
        log(f"    {sc}: " + "  ".join(f"q_{m}={r:.4f}" for m, r in radii.items()))
    log("=" * 80)
    log(f"Saved {fname}")


if __name__ == "__main__":
    main()
