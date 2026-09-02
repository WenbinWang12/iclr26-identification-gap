"""Phase-2Y: Three-stage composite method (PSR + Adaptive Rank + SCG).

Stage 1: PSR source selection with coverage buffer (from Phase-2F remedy)
Stage 2: Adaptive rank allocation conditional on measured forgetting
Stage 3: SCG offset correction with task-adaptive thresholds

Protocol: notes/phase2y_composite_protocol.md
Frozen before any Phase-2Y run exists, after Phase-2X verdict (paper safe).
"""

from __future__ import annotations

import argparse
import json
import os
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

from experiments.phase2i_anchored_cvar import order4_data as od  # noqa: E402
from experiments.phase2j_offset_conflict.offsets import (  # noqa: E402
    TaskLogits,
    balanced_accuracy,
    fit_offset,
    fit_shared_offset,
    gauge_fix,
)
from experiments.phase2j_offset_conflict.run_probe import (  # noqa: E402
    collect_logits,
    first_piece_ids,
    log,
    precondition_ok,
    with_oom_retry,
)
from experiments.phase2y_composite.adaptive_rank import (  # noqa: E402
    allocate_ranks,
    measure_residual_forgetting,
)


def parse_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__)

    # Basic config
    ap.add_argument("--data-root", required=True, help="Order-4 data root directory")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--seed", type=int, default=1, help="Random seed")
    ap.add_argument("--model", default="google-t5/t5-large", help="Base model")

    # LoRA base config
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="q,v", help="LoRA target modules")

    # Training config
    ap.add_argument("--epochs", type=int, default=3, help="Epochs per task")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--cap-per-class", type=int, default=400)
    ap.add_argument("--update-cap-per-class", type=int, default=400)
    ap.add_argument("--max-source", type=int, default=512)
    ap.add_argument("--max-target", type=int, default=128)

    # Stage 1: PSR config (from Phase-2F remedy)
    ap.add_argument("--psr-cover-first", type=int, default=4,
                    help="Stage 1: minimum coverage for first rare class")
    ap.add_argument("--psr-cover-rest", type=int, default=12,
                    help="Stage 1: minimum coverage for remaining rare classes")
    ap.add_argument("--psr-examples-per-task", type=int, default=16,
                    help="Stage 1: PSR examples stored per task")

    # Stage 2: Adaptive Rank config
    ap.add_argument("--rank-budget", type=int, default=120,
                    help="Stage 2: total rank budget (15 tasks × 8 = 120)")
    ap.add_argument("--rank-base", type=int, default=4,
                    help="Stage 2: minimum rank per task")
    ap.add_argument("--rank-sensitivity", type=float, default=0.5,
                    help="Stage 2: forgetting sensitivity λ∈[0,1], 0=uniform, 1=proportional")

    # Stage 3: SCG config
    ap.add_argument("--scg-tau-grid", default="0.8,1.0,1.2,1.4,1.6,2.0",
                    help="Stage 3: τ_m grid for search")
    ap.add_argument("--scg-worst-threshold", type=float, default=-1.0,
                    help="Stage 3: worst-case threshold for τ_m selection")

    # Evaluation config
    ap.add_argument("--risk-per-class", type=int, default=64)
    ap.add_argument("--audit-per-class", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=4)

    return ap.parse_args()


def train_one_task_adaptive_rank(
    model, tokenizer, examples, task, device, *,
    rank: int,  # allocated rank for this task
    batch_size: int,
    grad_accum: int,
    lr: float,
    epochs: int,
    max_source: int,
    max_target: int,
) -> dict:
    """Train one task with allocated rank (Stage 2)."""
    # TODO: Implement adaptive rank training
    # For now, placeholder using standard training
    log(f"Training {task.name} with allocated rank={rank}")

    # Placeholder: would need to actually configure LoRA with this rank
    # This requires modifying PEFT model after initialization
    raise NotImplementedError("Adaptive rank LoRA training not yet implemented")


def main():
    """Phase-2Y main training loop."""
    args = parse_args()

    log("=" * 80)
    log("Phase-2Y: Composite Method (PSR + Adaptive Rank + SCG)")
    log("=" * 80)
    log(f"Seed: {args.seed}")
    log(f"Output: {args.out}")
    log("")
    log(f"Stage 1 (PSR): cover_first={args.psr_cover_first}, "
        f"cover_rest={args.psr_cover_rest}, examples_per_task={args.psr_examples_per_task}")
    log(f"Stage 2 (Adaptive Rank): budget={args.rank_budget}, "
        f"base={args.rank_base}, sensitivity={args.rank_sensitivity}")
    log(f"Stage 3 (SCG): tau_grid={args.scg_tau_grid}, "
        f"worst_threshold={args.scg_worst_threshold}")
    log("=" * 80)

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    log(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # NOTE: Phase-2Y requires dynamic rank allocation
    # This is a PLACEHOLDER - full implementation requires:
    # 1. Task-conditional LoRA parameter masking, OR
    # 2. Progressive rank growth within fixed budget, OR
    # 3. Mixture-of-ranks architecture
    # For now, we abort with clear TODO

    log("=" * 80)
    log("IMPLEMENTATION STATUS:")
    log("Phase-2Y requires adaptive rank LoRA, which needs one of:")
    log("  1. Task-conditional parameter masking in PEFT")
    log("  2. Progressive rank growth (complex state management)")
    log("  3. Mixture-of-ranks architecture")
    log("")
    log("Current status: Protocol frozen, adaptive_rank logic tested,")
    log("but LoRA integration not yet implemented.")
    log("")
    log("Next steps:")
    log("  1. Choose implementation strategy (recommend #1: masking)")
    log("  2. Implement LoRA rank allocation hooks")
    log("  3. Integrate PSR buffer from Phase-2F")
    log("  4. Integrate SCG threshold fitting from Phase-2U")
    log("=" * 80)

    raise NotImplementedError(
        "Phase-2Y adaptive rank LoRA integration pending. "
        "See protocol: notes/phase2y_composite_protocol.md"
    )


if __name__ == "__main__":
    main()
