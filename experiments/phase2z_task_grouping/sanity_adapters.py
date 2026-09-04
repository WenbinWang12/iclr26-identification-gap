"""10-second sanity check of the multi-adapter mechanics on CPU.

Verifies, BEFORE burning GPU hours, that:
  1. get_peft_model + add_adapter creates 1 shared + 4 group adapters
  2. set_adapter(name) makes ONLY that adapter's params require_grad
     (so per-adapter training does not leak gradients across adapters)
  3. the fixed-budget identity holds: params(shared) == sum_k params(grp_k)
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

MODEL = os.environ.get("SANITY_MODEL", "google-t5/t5-small")
GROUPS = ["inference", "sentiment", "topic", "semantic_match"]


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainable_by_substr(model, substr):
    return sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and substr in n)


def main():
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL, torch_dtype=torch.float32)

    shared_cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8,
                            lora_alpha=32, lora_dropout=0.05,
                            target_modules=["q", "v"])
    model = get_peft_model(base, shared_cfg, adapter_name="shared")

    grp_cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=2,
                         lora_alpha=32, lora_dropout=0.05,
                         target_modules=["q", "v"])
    for g in GROUPS:
        model.add_adapter(f"grp_{g}", grp_cfg)

    print("Adapters present:", list(model.peft_config.keys()))

    # --- Check per-adapter param counts ---
    # NOTE: T5's tied token embedding is literally named `model.shared`, so a
    # naive `.shared.` substring match ALSO catches the 16.4M-param embedding.
    # Restrict to LoRA params (name contains 'lora_') to count only adapters.
    def adapter_numel(substr):
        return sum(p.numel() for n, p in model.named_parameters()
                   if substr in n and "lora_" in n)

    shared_n = adapter_numel(".shared.")
    grp_ns = {g: adapter_numel(f".grp_{g}.") for g in GROUPS}
    print(f"shared LoRA params:        {shared_n}")
    print(f"per-group LoRA params:     {grp_ns}")
    print(f"sum of group LoRA params:  {sum(grp_ns.values())}")
    ok_budget = (shared_n == sum(grp_ns.values()))
    print(f"[BUDGET] shared == sum(groups): {ok_budget}")

    # --- Check set_adapter isolates trainable params ---
    results = {}
    for target in ["shared"] + [f"grp_{g}" for g in GROUPS]:
        model.set_adapter(target)
        model.train()
        tot = count_trainable(model)
        sub = ".shared." if target == "shared" else f".{target}."
        on_target = trainable_by_substr(model, sub)
        off_target = tot - on_target
        results[target] = (tot, on_target, off_target)
        print(f"set_adapter({target:18s}): trainable={tot:7d}  "
              f"on_target={on_target:7d}  off_target={off_target:7d}")

    all_isolated = all(off == 0 for _, _, off in results.values())
    print(f"[ISOLATION] set_adapter isolates trainable params: {all_isolated}")

    print()
    if ok_budget and all_isolated:
        print("SANITY_OK")
    else:
        print("SANITY_FAIL: adapter mechanics do NOT match assumptions; "
              "run_grouping.py needs a manual requires_grad fix.")


if __name__ == "__main__":
    main()
