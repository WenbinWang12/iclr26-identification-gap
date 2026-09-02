"""Phase-2Y-A: Two-stage composite (PSR + SCG) - Fast track.

Stage 1: PSR source selection with coverage buffer (Phase-2F remedy)
Stage 2: SCG offset correction with task-adaptive thresholds (Phase-2U mechanism)

This is the fast-track version without adaptive rank allocation.
If successful, provides positive result with simpler method.
If fails, confirms adaptive rank (Track B) is necessary.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

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
    ORDER4_TASK_NAMES,
    TASK_BY_NAME,
    prepare_task_partitions,
    load_official_examples,
)
from experiments.phase2j_offset_conflict.offsets import (  # noqa: E402
    balanced_accuracy,
    fit_offset,
    gauge_fix,
)
from experiments.phase2j_offset_conflict.run_probe import (  # noqa: E402
    collect_logits,
    first_piece_ids,
    log,
    precondition_ok,
    train_one_task,
    with_oom_retry,
)


def psr_select_sources(examples, task, *, cover_first: int, cover_rest: int,
                       total_budget: int, seed: int) -> List:
    """PSR source selection with coverage buffer (Phase-2F remedy).

    Guarantees minimum representation of rare classes:
    - cover_first: minimum from first rare class
    - cover_rest: minimum from remaining rare classes
    - Remaining budget: uniform reservoir
    """
    import random
    rng = random.Random(seed)

    # Group by class
    by_class = {}
    for ex in examples:
        by_class.setdefault(ex.label, []).append(ex)

    # Sort classes by frequency (ascending = rare first)
    sorted_classes = sorted(by_class.keys(), key=lambda c: len(by_class[c]))

    selected = []

    # Cover first rare class
    if len(sorted_classes) > 0:
        first_rare = sorted_classes[0]
        take = min(cover_first, len(by_class[first_rare]))
        selected.extend(rng.sample(by_class[first_rare], take))

    # Cover remaining rare classes
    for cls in sorted_classes[1:]:
        take = min(cover_rest, len(by_class[cls]))
        selected.extend(rng.sample(by_class[cls], take))

    # Fill remaining budget with uniform reservoir
    remaining_budget = total_budget - len(selected)
    if remaining_budget > 0:
        pool = [ex for ex in examples if ex not in selected]
        if len(pool) > remaining_budget:
            selected.extend(rng.sample(pool, remaining_budget))
        else:
            selected.extend(pool)

    return selected[:total_budget]


def fit_scg_threshold(logits, labels, tau_grid, worst_threshold: float = -1.0):
    """Fit task-adaptive SCG threshold on validation examples.

    Searches tau_grid for threshold that maximizes mean gain
    while keeping worst >= worst_threshold.
    """
    from experiments.phase2s_sio.sio import bc_offset

    best_tau = None
    best_mean = -np.inf

    # Raw accuracy
    acc_raw = balanced_accuracy(logits, labels, None)

    for tau in tau_grid:
        gains = []
        for i in range(len(logits)):
            # Leave-one-out: fit BC on all except i, test on i
            idx = [j for j in range(len(logits)) if j != i]
            if len(idx) < 2:
                continue

            # BC offset
            offset_bc = bc_offset(logits[idx])

            # Decision statistic for test example
            if logits.shape[1] == 2:  # binary
                s = logits[i, 1] - logits[i, 0]
                m_abs = abs(s)

                # Apply SCG gate
                if m_abs >= tau:
                    offset = offset_bc
                else:
                    offset = None

                # Measure gain
                acc_with = balanced_accuracy(logits[i:i+1], labels[i:i+1], offset)
                gain = (acc_with - acc_raw) * 100
                gains.append(gain)

        if not gains:
            continue

        mean_gain = np.mean(gains)
        worst_gain = np.min(gains)

        # Accept if worst >= threshold and mean is best so far
        if worst_gain >= worst_threshold and mean_gain > best_mean:
            best_tau = tau
            best_mean = mean_gain

    return best_tau if best_tau is not None else tau_grid[-1]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)

    # Basic
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default="google-t5/t5-large")

    # Training
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--cap-per-class", type=int, default=400)
    ap.add_argument("--update-cap-per-class", type=int, default=400)
    ap.add_argument("--max-source", type=int, default=512)
    ap.add_argument("--max-target", type=int, default=128)

    # LoRA
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    # PSR (Stage 1)
    ap.add_argument("--psr-cover-first", type=int, default=4)
    ap.add_argument("--psr-cover-rest", type=int, default=12)
    ap.add_argument("--psr-budget", type=int, default=16)

    # SCG (Stage 2)
    ap.add_argument("--scg-tau-grid", default="0.8,1.0,1.2,1.4,1.6,2.0")
    ap.add_argument("--scg-worst-threshold", type=float, default=-1.0)

    # Eval
    ap.add_argument("--risk-per-class", type=int, default=64)
    ap.add_argument("--audit-per-class", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=4)

    return ap.parse_args()


def main():
    args = parse_args()

    log("=" * 80)
    log("Phase-2Y-A: Two-Stage Composite (PSR + SCG)")
    log("=" * 80)
    log(f"Seed: {args.seed}")
    log(f"Output: {args.out}")
    log(f"Stage 1 (PSR): cover_first={args.psr_cover_first}, "
        f"cover_rest={args.psr_cover_rest}, budget={args.psr_budget}")
    tau_grid = [float(x) for x in args.scg_tau_grid.split(',')]
    log(f"Stage 2 (SCG): tau_grid={tau_grid}, worst_threshold={args.scg_worst_threshold}")
    log("=" * 80)

    # Create output
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    log(f"Loading model: {args.model}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load tasks
    log(f"Using Order-4 task specifications")
    tasks = ORDER4_TASKS
    log(f"Loaded {len(tasks)} tasks")

    # PSR store and SCG thresholds
    psr_store = {}  # task_name -> examples
    scg_thresholds = {}  # task_name -> tau

    # Training loop
    for task_idx, task in enumerate(tasks, 1):
        log("")
        log(f"=== Task {task_idx}/{len(tasks)}: {task.name} ===")

        # Check precondition
        if not precondition_ok(tokenizer, task.labels):
            log(f"SKIP {task.name}: first-piece collision")
            continue

        # Determine adaptive budget based on rarest class
        task_examples = load_official_examples(args.data_root, task.name, "train")
        class_counts = {}
        for ex in task_examples:
            class_counts[ex.label] = class_counts.get(ex.label, 0) + 1
        rarest_count = min(class_counts.values()) if class_counts else 0

        # Skip if too small (same as phase2j protocol)
        if rarest_count < 40:
            log(f"SKIP {task.name}: rarest class has only {rarest_count} examples (< 40)")
            continue

        task_risk = min(args.risk_per_class, (rarest_count - 1) // 2)
        task_audit = min(args.audit_per_class, rarest_count - 1 - task_risk)
        log(f"Adaptive budget: risk={task_risk}, audit={task_audit} (rarest={rarest_count})")

        # Load data partitions
        parts = prepare_task_partitions(
            args.data_root,
            task.name,
            cap_per_class=args.cap_per_class,
            risk_per_class=task_risk,
            audit_per_class=task_audit,
            seed=args.seed
        )

        # Stage 1: PSR source selection
        psr_examples = psr_select_sources(
            parts.update, task,
            cover_first=args.psr_cover_first,
            cover_rest=args.psr_cover_rest,
            total_budget=args.psr_budget,
            seed=args.seed + task_idx
        )
        psr_store[task.name] = psr_examples
        log(f"PSR selected {len(psr_examples)} examples")

        # Train with PSR rehearsal
        train_examples = list(parts.update) + [ex for stored in psr_store.values() for ex in stored]
        log(f"Training on {len(parts.update)} new + {len(train_examples) - len(parts.update)} rehearsal")

        # Create args namespace for train_one_task
        from types import SimpleNamespace
        train_args = SimpleNamespace(
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            epochs=args.epochs,
            max_source=args.max_source,
            max_target=args.max_target,
            seed=args.seed,
        )

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        train_one_task(
            model, tokenizer, train_examples, device,
            args=train_args, optimizer=optimizer
        )

        # Stage 2: Fit SCG threshold on PSR examples
        if len(psr_examples) >= 4:  # Need minimum examples
            psr_logits = collect_logits(
                model, tokenizer, psr_examples, task, device,
                max_source=args.max_source,
                batch_size=args.eval_batch_size
            )
            psr_labels = np.array([task.labels.index(ex.label) for ex in psr_examples])

            tau = fit_scg_threshold(
                psr_logits, psr_labels, tau_grid,
                worst_threshold=args.scg_worst_threshold
            )
            scg_thresholds[task.name] = tau
            log(f"SCG threshold fitted: tau={tau:.2f}")
        else:
            scg_thresholds[task.name] = tau_grid[-1]  # default to highest
            log(f"SCG threshold default: tau={tau_grid[-1]:.2f} (insufficient examples)")

    # Save PSR store and SCG thresholds
    with open(out_dir / f"psr_store_s{args.seed}.pkl", "wb") as f:
        pickle.dump(psr_store, f)

    with open(out_dir / f"scg_thresholds_s{args.seed}.json", "w") as f:
        json.dump(scg_thresholds, f, indent=2)

    # Final evaluation on all tasks
    log("")
    log("=" * 80)
    log("Final Evaluation")
    log("=" * 80)

    results = []
    for task in tasks:
        if not precondition_ok(tokenizer, task.labels):
            continue

        # Same adaptive budget logic
        task_examples = load_official_examples(args.data_root, task.name, "train")
        class_counts = {}
        for ex in task_examples:
            class_counts[ex.label] = class_counts.get(ex.label, 0) + 1
        rarest_count = min(class_counts.values()) if class_counts else 0

        if rarest_count < 40:
            continue

        task_risk = min(args.risk_per_class, (rarest_count - 1) // 2)
        task_audit = min(args.audit_per_class, rarest_count - 1 - task_risk)

        parts = prepare_task_partitions(
            args.data_root,
            task.name,
            cap_per_class=args.cap_per_class,
            risk_per_class=task_risk,
            audit_per_class=task_audit,
            seed=args.seed
        )

        # Collect logits on audit split
        logits = collect_logits(
            model, tokenizer, parts.audit, task, device,
            max_source=args.max_source,
            batch_size=args.eval_batch_size
        )
        labels = np.array([task.labels.index(ex.label) for ex in parts.audit])

        # Reference levels
        per_task_offset = gauge_fix(fit_offset(logits, labels))
        R_raw = balanced_accuracy(logits, labels, None)
        R_orc = balanced_accuracy(logits, labels, per_task_offset)

        # Stage 2: Apply SCG
        from experiments.phase2s_sio.sio import bc_offset
        tau = scg_thresholds.get(task.name, tau_grid[-1])

        # For each batch in audit, apply SCG
        # Simplified: apply to full audit (in practice would be per-batch)
        bc_off = bc_offset(logits)

        # Decision statistic (binary tasks only for now)
        if logits.shape[1] == 2:
            s = logits[:, 1] - logits[:, 0]
            m = np.mean(s)
            m_abs = abs(m)

            if m_abs >= tau:
                scg_offset = bc_off
            else:
                scg_offset = None

            R_scg = balanced_accuracy(logits, labels, scg_offset)
        else:
            # Multi-class: use BC without gate for now
            R_scg = balanced_accuracy(logits, labels, bc_off)

        result = {
            "task": task.name,
            "R_raw": float(R_raw),
            "R_orc": float(R_orc),
            "R_scg": float(R_scg),
            "tau": float(tau),
            "n_audit": len(parts.audit),
            "K": len(task.labels),
        }
        results.append(result)

        log(f"{task.name:12s}: R_raw={R_raw:.4f}, R_orc={R_orc:.4f}, "
            f"R_scg={R_scg:.4f} (tau={tau:.2f})")

    # Save results
    with open(out_dir / f"results_s{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary statistics
    r_raw_mean = np.mean([r["R_raw"] for r in results])
    r_orc_mean = np.mean([r["R_orc"] for r in results])
    r_scg_mean = np.mean([r["R_scg"] for r in results])

    log("")
    log("=" * 80)
    log("Summary (mean across tasks)")
    log("=" * 80)
    log(f"R_raw:  {r_raw_mean:.4f}")
    log(f"R_orc:  {r_orc_mean:.4f}")
    log(f"R_scg:  {r_scg_mean:.4f}")
    log(f"SCG vs raw:    {(r_scg_mean - r_raw_mean)*100:+.2f}pp")
    log(f"SCG vs oracle: {(r_scg_mean - r_orc_mean)*100:+.2f}pp")
    log("=" * 80)
    log(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
