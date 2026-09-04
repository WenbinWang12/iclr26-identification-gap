"""Phase-2Z-E (RUNBOOK 3A): deployable family router + stored-offset 2x2.

The combined 2x2 (run_combined.py) and the de-stale study (run_destale.py) both
score the grouped arm at TWO upper bounds:
  * family routing uses the task's TRUE group at eval (oracle routing);
  * the offset column is fit against the FINAL model (refit), not stored.
This script replaces BOTH ceilings with deployable rules and measures the cost:

  routing   <- a TASK-FREE nearest-class-mean (NCM) router. After training we
               store, per eligible task, the mean pooled BASE-encoder state over
               its risk split (a "class mean" in encoder space; the base encoder
               is adapter-independent, so the feature a query is routed on does
               not depend on which adapter would be chosen). At eval each query
               is routed, with NO task index, to the family of the nearest stored
               class mean, and scored under that family's adapter.
  offset    <- the deployable STORED per-scope offset (fit at each task's
               boundary and frozen, run_destale's SSO), not the refit upper bound.

We report, per task, the no-offset and stored-offset accuracy under three
routing rules -- shared (one adapter, no routing), grouped-oracle (true family),
grouped-route (NCM router) -- plus the refit-offset upper bound for reference.
The headline is the routing cost R_grouped_orc - R_grouped_route (per task and
mean) and whether the fully deployable cell (grouped-route + stored offset) still
beats the shared/no-offset baseline. A negative or mixed result here is a valid,
reportable outcome (see notes protocol): we do NOT fall back to the oracle number.

Every number traces to this frozen run.
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
)
from experiments.phase2j_offset_conflict.run_probe import (  # noqa: E402
    collect_logits,
    log,
)
from experiments.phase2z_task_grouping.run_grouping import (  # noqa: E402
    GROUP_OF,
    GROUP_NAMES,
    build_adapters,
    eligible_tasks,
)
from experiments.phase2z_task_grouping.run_destale import (  # noqa: E402
    run_regime,
    centres_from_table,
    score_with_centres,
)


# ---------------------------------------------------------------------------
# Task-free nearest-class-mean router on BASE-encoder features.
# ---------------------------------------------------------------------------
@torch.no_grad()
def pooled_base_encoder(model, tokenizer, examples, device, *, max_source,
                        batch_size):
    """Mean-pooled encoder last-hidden-state per example, ADAPTERS DISABLED.

    Routing must be adapter-independent and task-free: it decides which family
    adapter to use, so it cannot depend on any family adapter. We therefore read
    the base (LoRA-disabled) encoder. Pooling is masked mean over real tokens.
    """
    model.eval()
    encoder = model.get_encoder()
    vecs = []
    with model.disable_adapter():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            enc = tokenizer(
                [ex.prompt for ex in batch],
                max_length=max_source, truncation=True, padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = encoder(input_ids=enc["input_ids"],
                          attention_mask=enc["attention_mask"])
            h = out.last_hidden_state.float()             # (b, L, d)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            vecs.append(pooled.cpu().numpy())
    return np.concatenate(vecs, axis=0)


def build_prototypes(model, tokenizer, device, args, task_list, parts_cache):
    """Per-task class mean of base-encoder states over its risk split.

    Returns (protos, proto_family): protos[name] -> (d,) centroid; proto_family
    maps each task name to its family. Routing is NCM over these prototypes; the
    winning task's family is the routed adapter.
    """
    protos, proto_family = {}, {}
    for task, _ in task_list:
        parts = parts_cache[task.name]
        V = pooled_base_encoder(model, tokenizer, list(parts.risk), device,
                                max_source=args.max_source,
                                batch_size=args.eval_batch_size)
        protos[task.name] = V.mean(axis=0)
        proto_family[task.name] = GROUP_OF[task.name]
        log(f"  prototype {task.name:12s} [{proto_family[task.name]:14s}] "
            f"n_risk={len(parts.risk)}")
    return protos, proto_family


def route_families(query_vecs, protos, proto_family):
    """NCM route: for each query vector, family of the nearest stored prototype."""
    names = list(protos.keys())
    P = np.stack([protos[n] for n in names])              # (T, d)
    d = np.linalg.norm(query_vecs[:, None, :] - P[None, :, :], axis=-1)
    nearest = d.argmin(axis=1)
    return [proto_family[names[i]] for i in nearest], [names[i] for i in nearest]


def routed_logits(model, tokenizer, examples, task, device, fams, *,
                  max_source, batch_size):
    """Verbalizer logits with a possibly-different family adapter per example."""
    L = np.zeros((len(examples), len(task.labels)), dtype=np.float64)
    for fam in sorted(set(fams)):
        idx = [i for i, f in enumerate(fams) if f == fam]
        sub = [examples[i] for i in idx]
        model.set_adapter(f"grp_{fam}")
        Ls = collect_logits(model, tokenizer, sub, task, device,
                            max_source=max_source, batch_size=batch_size)
        for j, i in enumerate(idx):
            L[i] = Ls[j]
    return L


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
    log("Phase-2Z-E (RUNBOOK 3A): deployable router + stored-offset 2x2")
    log("=" * 80)
    n_groups = len(GROUP_NAMES)
    log(f"Seed: {args.seed}")
    log(f"Fixed budget: shared_rank={args.shared_rank} vs "
        f"{n_groups} x group_rank={args.group_rank} = {n_groups*args.group_rank}")
    if args.shared_rank != n_groups * args.group_rank:
        log("WARNING: budgets differ; NOT a fixed-budget comparison!")
    log("=" * 80)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

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

    # Fixed splits per task (same seed -> identical to combined/destale).
    parts_cache = {}
    for task, rarest in task_list:
        task_risk = min(args.risk_per_class, (rarest - 1) // 2)
        task_audit = min(args.audit_per_class, rarest - 1 - task_risk)
        parts_cache[task.name] = prepare_task_partitions(
            args.data_root, task.name, cap_per_class=args.cap_per_class,
            risk_per_class=task_risk, audit_per_class=task_audit, seed=args.seed)

    # ------------------------------------------------------------------
    # Train both regimes; collect stored (boundary) + refit (final) offsets.
    # run_regime trains in place, so after both calls every adapter holds its
    # final trained state (shared and group adapters are independent).
    # ------------------------------------------------------------------
    sso_sh, refit_sh, audit_sh = run_regime(
        model, tokenizer, device, args, task_list,
        adapter_for=lambda t: "shared", regime="shared")
    sso_gp, refit_gp, audit_gp = run_regime(
        model, tokenizer, device, args, task_list,
        adapter_for=lambda t: f"grp_{GROUP_OF[t.name]}", regime="grouped")

    centres = {
        ("shared", "stored"): centres_from_table(sso_sh.table),
        ("shared", "refit"): centres_from_table(refit_sh),
        ("grouped", "stored"): centres_from_table(sso_gp.table),
        ("grouped", "refit"): centres_from_table(refit_gp),
    }

    # ------------------------------------------------------------------
    # Build the deployable router from base-encoder class means.
    # ------------------------------------------------------------------
    log("")
    log("### Building NCM router prototypes (base encoder, risk splits) ###")
    protos, proto_family = build_prototypes(
        model, tokenizer, device, args, task_list, parts_cache)

    # ------------------------------------------------------------------
    # Score the deployable 2x2 (+ oracle/refit ceilings) per task.
    # ------------------------------------------------------------------
    log("")
    log("=" * 80)
    log("Final Evaluation: deployable routing x stored offset (+ ceilings)")
    log("=" * 80)

    results = []
    for task, rarest in task_list:
        parts = parts_cache[task.name]
        audit_ex = list(parts.audit)
        verb = tuple(task.labels)
        _, L_sh, labels = audit_sh[task.name]              # shared (final) logits
        _, L_orc, labels_g = audit_gp[task.name]           # grouped-oracle logits
        assert np.array_equal(labels, labels_g)

        # Route each audit query with no task index; score under routed family.
        qv = pooled_base_encoder(model, tokenizer, audit_ex, device,
                                 max_source=args.max_source,
                                 batch_size=args.eval_batch_size)
        fams, routed_task = route_families(qv, protos, proto_family)
        L_route = routed_logits(model, tokenizer, audit_ex, task, device, fams,
                                max_source=args.max_source,
                                batch_size=args.eval_batch_size)
        true_family = GROUP_OF[task.name]
        route_acc = float(np.mean([f == true_family for f in fams]))

        def sc(L, regime, source):
            return score_with_centres(centres[(regime, source)], task, verb,
                                      L, labels)

        row = {
            "task": task.name, "group": true_family,
            "n_audit": int(len(audit_ex)), "K": int(len(task.labels)),
            "route_acc_to_true_family": route_acc,
            # no offset
            "R_shared": float(balanced_accuracy(L_sh, labels, None)),
            "R_grouped_orc": float(balanced_accuracy(L_orc, labels, None)),
            "R_grouped_route": float(balanced_accuracy(L_route, labels, None)),
            # + stored offset (deployable)
            "R_shared_stored": float(sc(L_sh, "shared", "stored")),
            "R_grouped_orc_stored": float(sc(L_orc, "grouped", "stored")),
            "R_grouped_route_stored": float(sc(L_route, "grouped", "stored")),
            # + refit offset (upper bound, for reference)
            "R_shared_refit": float(sc(L_sh, "shared", "refit")),
            "R_grouped_orc_refit": float(sc(L_orc, "grouped", "refit")),
            "R_grouped_route_refit": float(sc(L_route, "grouped", "refit")),
        }
        results.append(row)
        log(f"{task.name:12s} route->true={route_acc:.2f}  "
            f"orc={row['R_grouped_orc']:.4f} route={row['R_grouped_route']:.4f} "
            f"(gap={ (row['R_grouped_orc']-row['R_grouped_route'])*100:+.2f}pp)  "
            f"route+stored={row['R_grouped_route_stored']:.4f}")

    with open(out_dir / f"router_s{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2)

    def m(k):
        return float(np.mean([r[k] for r in results])) * 100
    log("")
    log("=" * 80)
    log("AGGREGATE (mean over tasks, this seed)")
    log(f"  route->true-family accuracy       : {m('route_acc_to_true_family'):.1f}%")
    log("  --- no offset ---")
    log(f"  R_shared                          : {m('R_shared'):.2f}")
    log(f"  R_grouped_orc  (upper bound)      : {m('R_grouped_orc'):.2f}")
    log(f"  R_grouped_route (deployable)      : {m('R_grouped_route'):.2f}")
    log(f"  routing cost (orc - route)        : "
        f"{m('R_grouped_orc')-m('R_grouped_route'):+.2f}pp")
    log("  --- + stored offset (deployable) ---")
    log(f"  R_shared_stored                   : {m('R_shared_stored'):.2f}")
    log(f"  R_grouped_route_stored (METHOD)   : {m('R_grouped_route_stored'):.2f}")
    log(f"  deployable e2e (route+stored - shared/no-off): "
        f"{m('R_grouped_route_stored')-m('R_shared'):+.2f}pp")
    log("  --- ceilings for reference ---")
    log(f"  R_grouped_orc_refit (both upper)  : {m('R_grouped_orc_refit'):.2f}")
    log("=" * 80)
    log(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
