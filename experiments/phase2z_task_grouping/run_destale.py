"""Phase-2Z-D: does grouping de-stale the STORED offset?

The combined method (run_combined.py) scored its offset column at an UPPER BOUND:
every offset was fit against the *final* model on each task's risk split. That is
the rehearsal ceiling of Section sec:gap-real, not a deployable rule. The
deployable rule is SSO (Section sec:sso): fit each task's optimum at the moment it
finishes training, store it, and never look back -- which pays a staleness cost
because the model keeps moving afterwards. On a single shared adapter that cost was
5.08pp BELOW the no-offset floor.

The paper argues (Section sec:method, and limitations) that GROUPING should soften
this: a family adapter is perturbed only by its own family's later tasks, so a
stored per-family optimum drifts less than one stored against a monolithic adapter
that every later task moves. This script MEASURES that argument.

Design: one training pass per adapter regime. Within each pass, at the moment task
g finishes training we collect g's risk logits under the CURRENT model and fit +
store its optimum (the deployable "stored" source). After all tasks finish we
collect every task's risk logits under the FINAL model and fit (the upper-bound
"refit" source). Both sources are aggregated by the SAME deployment rule -- the
scope's Chebyshev centre (StreamingScopeOffsets + chebyshev_centre, m=1) -- and
scored on the SAME held-out audit split. Only the FIT TIME differs, so the gap
   staleness = R_refit - R_stored
is pure staleness with aggregation, routing and gauge held identical.

We report, per regime R in {shared, grouped}:
  R_none      no offset
  R_stored    deployable SSO: offset fit at each task's boundary, stored
  R_refit     upper bound: offset fit against the final model (== combined column)
The paper's claim is that (R_refit - R_stored) is much smaller under grouped than
under shared -- i.e. grouping turns the offset column from an upper bound into a
deployable one. Every number traces to this frozen run; nothing is assumed.
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

from experiments.phase2i_anchored_cvar.order4_data import (  # noqa: E402
    ORDER4_TASKS,
    prepare_task_partitions,
)
from experiments.phase2j_offset_conflict.offsets import (  # noqa: E402
    balanced_accuracy,
    fit_offset,
    gauge_fix,
)
from experiments.phase2j_offset_conflict.run_probe import (  # noqa: E402
    collect_logits,
    log,
    train_one_task,
)
from experiments.phase2z_task_grouping.run_grouping import (  # noqa: E402
    GROUP_OF,
    GROUP_NAMES,
    build_adapters,
    make_optimizer,
    eligible_tasks,
)
from experiments.phase2p_sso.sso import (  # noqa: E402
    StreamingScopeOffsets,
    chebyshev_centre,
)
from experiments.phase2k_qoc.qoc import TaskOptima, scope_key  # noqa: E402


def first_pieces(tokenizer, labels):
    """First sub-word piece id of each label string (TaskOptima coordinate)."""
    out = []
    for lab in labels:
        ids = tokenizer.encode(lab, add_special_tokens=False)
        out.append(int(ids[0]) if ids else -1)
    return tuple(out)


def centres_from_table(table):
    """scope key -> gauge-fixed Chebyshev centre of that scope's stored optima."""
    out = {}
    for scope, members in table.items():
        X = np.stack([e.aligned() for e in members])
        out[scope] = chebyshev_centre(X)
    return out


def score_with_centres(centres, task, labels_verb, L_audit, labels_audit):
    """Balanced acc applying the query scope's centre, re-aligned to this task's
    verbalizer order (centre is stored in canonical scope order)."""
    scope = scope_key(labels_verb)
    centre = centres.get(scope)
    if centre is None:
        return balanced_accuracy(L_audit, labels_audit, None)
    # centre is in canonical (sorted) scope order; map back to task label order.
    canon = list(scope)
    off = np.array([centre[canon.index(lab)] for lab in labels_verb])
    return balanced_accuracy(L_audit, labels_audit, gauge_fix(off))


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


def run_regime(model, tokenizer, device, args, task_list, adapter_for, regime):
    """Train one regime; collect STORED offsets at each task boundary and REFIT
    offsets after all training. Returns (sso_stored, risk_final, audit_cache)."""
    sso = StreamingScopeOffsets()          # deployable: fit-at-boundary + store
    parts_cache = {}                        # task -> partitions (fixed splits)

    log("")
    log(f"### Regime '{regime}': train + stream stored offsets ###")
    for task, rarest in task_list:
        adapter = adapter_for(task)
        model.set_adapter(adapter)

        task_risk = min(args.risk_per_class, (rarest - 1) // 2)
        task_audit = min(args.audit_per_class, rarest - 1 - task_risk)
        parts = prepare_task_partitions(
            args.data_root, task.name,
            cap_per_class=args.cap_per_class,
            risk_per_class=task_risk, audit_per_class=task_audit,
            seed=args.seed,
        )
        parts_cache[task.name] = parts
        train_args = SimpleNamespace(
            batch_size=args.batch_size, grad_accum=args.grad_accum,
            lr=args.lr, epochs=args.epochs,
            max_source=args.max_source, max_target=args.max_target,
            seed=args.seed,
        )
        optimizer = make_optimizer(model, args.lr)
        log(f"  train {task.name} on '{adapter}' ({len(parts.update)} ex)")
        train_one_task(model, tokenizer, list(parts.update), device,
                       args=train_args, optimizer=optimizer)

        # STORED source: fit this task's optimum NOW, against the model as it
        # stands at this task's boundary, on its own risk split. Store; move on.
        labels_risk = np.array(
            [task.labels.index(ex.label) for ex in parts.risk])
        L_risk_now = collect_logits(model, tokenizer, parts.risk, task, device,
                                    max_source=args.max_source,
                                    batch_size=args.eval_batch_size)
        sso.record(task.name, list(task.labels),
                   first_pieces(tokenizer, task.labels),
                   L_risk_now, labels_risk)

    # REFIT source + audit logits: collect under the FINAL model.
    log(f"### Regime '{regime}': refit offsets against FINAL model ###")
    refit_table = {}
    audit_cache = {}
    for task, rarest in task_list:
        parts = parts_cache[task.name]
        model.set_adapter(adapter_for(task))
        labels_risk = np.array(
            [task.labels.index(ex.label) for ex in parts.risk])
        labels_audit = np.array(
            [task.labels.index(ex.label) for ex in parts.audit])

        L_risk_fin = collect_logits(model, tokenizer, parts.risk, task, device,
                                    max_source=args.max_source,
                                    batch_size=args.eval_batch_size)
        opt = gauge_fix(fit_offset(L_risk_fin, labels_risk))
        entry = TaskOptima(task=task.name, labels=tuple(task.labels),
                           pieces=first_pieces(tokenizer, task.labels),
                           optimum=opt)
        refit_table.setdefault(entry.scope, []).append(entry)

        L_audit = collect_logits(model, tokenizer, parts.audit, task, device,
                                 max_source=args.max_source,
                                 batch_size=args.eval_batch_size)
        audit_cache[task.name] = (task, L_audit, labels_audit)

    return sso, refit_table, audit_cache


def main():
    args = parse_args()
    log("=" * 80)
    log("Phase-2Z-D: grouping de-stales the stored offset?")
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

    regimes = {
        "shared": lambda t: "shared",
        "grouped": lambda t: f"grp_{GROUP_OF[t.name]}",
    }

    all_results = {}
    for regime, adapter_for in regimes.items():
        sso, refit_table, audit_cache = run_regime(
            model, tokenizer, device, args, task_list, adapter_for, regime)

        centres_stored = centres_from_table(sso.table)
        centres_refit = centres_from_table(refit_table)

        rows = []
        for task, rarest in task_list:
            _, L_audit, labels_audit = audit_cache[task.name]
            verb = tuple(task.labels)
            R_none = balanced_accuracy(L_audit, labels_audit, None)
            R_stored = score_with_centres(centres_stored, task, verb,
                                          L_audit, labels_audit)
            R_refit = score_with_centres(centres_refit, task, verb,
                                         L_audit, labels_audit)
            rows.append({
                "task": task.name, "group": GROUP_OF[task.name],
                "regime": regime,
                "R_none": float(R_none),
                "R_stored": float(R_stored),
                "R_refit": float(R_refit),
                "n_audit": int(len(L_audit)), "K": int(len(task.labels)),
            })
            log(f"[{regime:7s}] {task.name:12s} none={R_none:.4f} "
                f"stored={R_stored:.4f} refit={R_refit:.4f} "
                f"staleness={ (R_refit-R_stored)*100:+.2f}pp")

        all_results[regime] = {
            "rows": rows,
            "stored_floats": int(sso.stored_floats()),
            "n_entries": int(sso.n_entries()),
        }

        def m(k):
            return float(np.mean([r[k] for r in rows])) * 100
        log("")
        log(f"--- regime '{regime}' aggregate ---")
        log(f"  R_none   = {m('R_none'):.2f}")
        log(f"  R_stored = {m('R_stored'):.2f}   (deployable SSO)")
        log(f"  R_refit  = {m('R_refit'):.2f}   (upper bound)")
        log(f"  stored - none  = {m('R_stored')-m('R_none'):+.2f}pp")
        log(f"  staleness (refit - stored) = {m('R_refit')-m('R_stored'):+.2f}pp")
        log(f"  stored_floats = {all_results[regime]['stored_floats']}")

    with open(out_dir / f"destale_s{args.seed}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Headline: does grouping shrink staleness?
    def agg(regime, k):
        return float(np.mean([r[k] for r in all_results[regime]["rows"]])) * 100
    st_sh = agg("shared", "R_refit") - agg("shared", "R_stored")
    st_gp = agg("grouped", "R_refit") - agg("grouped", "R_stored")
    dep_sh = agg("shared", "R_stored") - agg("shared", "R_none")
    dep_gp = agg("grouped", "R_stored") - agg("grouped", "R_none")
    log("")
    log("=" * 80)
    log("HEADLINE: staleness of the stored offset, shared vs grouped")
    log(f"  shared  staleness = {st_sh:+.2f}pp   deployable gain (stored-none) = {dep_sh:+.2f}pp")
    log(f"  grouped staleness = {st_gp:+.2f}pp   deployable gain (stored-none) = {dep_gp:+.2f}pp")
    log(f"  grouping shrinks staleness by {st_sh - st_gp:+.2f}pp")
    log("=" * 80)
    log(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
