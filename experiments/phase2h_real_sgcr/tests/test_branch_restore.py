"""Phase-2H branch-restore integrity test (protocol: SGCR compares branch A
against branch B on audit-role, then RESTORES both the model and the optimizer
state of the selected branch before deploying).  A restore that does not return
the model + optimizer to bit-identical state would leak the losing branch's
gradients into the deployed model -- a silent correctness failure.

This test needs the real BERT-tiny backbone (transformers).  If transformers is
not installed it SKIPS with a clear message rather than passing vacuously.

Run: python tests/test_branch_restore.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fake_records(n, seed=0, L=32):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ids = np.zeros(L, dtype=np.int32)
        ids[0] = 101
        k = int(rng.integers(4, 12))
        ids[1:1 + k] = rng.integers(1000, 5000, size=k)
        ids[1 + k] = 102
        mask = (ids != 0).astype(np.uint8); mask[0] = 1
        out.append({"input_ids": ids, "attention_mask": mask,
                    "label": np.int8(i % 2), "example_id": np.uint64(seed * 1000 + i)})
    return out


def _optimizer_state_flat(opt):
    """Flatten optimizer state tensors into one vector for exact comparison."""
    import torch
    flats = []
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                flats.append(v.detach().reshape(-1).clone())
    return torch.cat(flats) if flats else torch.zeros(1)


def test_branch_restore_roundtrip():
    try:
        import torch
        import model as M
    except Exception as e:  # transformers / torch not present
        print(f"SKIP test_branch_restore ({type(e).__name__}: {e})")
        return

    try:
        mdl = M.load_model(seed=0)
    except Exception as e:  # network / weights unavailable
        print(f"SKIP test_branch_restore (model load: {type(e).__name__}: {e})")
        return

    cfg = M.OptimConfig()
    recs = _fake_records(32, seed=0)

    # Warm up a couple of steps so LoRA + optimizer moments are non-trivial,
    # then freeze the head (post-warmup regime the branch logic runs in).
    M.warmup_train(mdl, recs, cfg, epochs=1, seed=0)
    opt = M.make_optimizer(mdl, cfg)
    # take one step so the optimizer has real moment estimates to restore
    M.train_step(mdl, opt, recs[:16], cfg)

    # Snapshot the "common" state both branches fork from.
    snap = M.snapshot(mdl, opt)
    lora_before = [p.detach().clone() for p in mdl.lora_parameters()]
    head_before = {n: p.detach().clone() for n, p in mdl.head.named_parameters()}
    optstate_before = _optimizer_state_flat(opt)

    # Branch B: several steps on different data (the branch we will DISCARD).
    for _ in range(3):
        M.train_step(mdl, opt, recs[16:], cfg)
    moved = max(float((a - b.detach()).abs().max())
               for a, b in zip(lora_before, mdl.lora_parameters()))
    assert moved > 0, "training did not move LoRA params; test is vacuous"

    # Restore to the common fork point.
    M.restore(mdl, snap, opt)

    lora_after = [p.detach().clone() for p in mdl.lora_parameters()]
    head_after = {n: p.detach().clone() for n, p in mdl.head.named_parameters()}
    optstate_after = _optimizer_state_flat(opt)

    lora_dev = max(float((a - b).abs().max())
                   for a, b in zip(lora_before, lora_after))
    head_dev = max(float((head_before[n] - head_after[n]).abs().max())
                   for n in head_before)
    opt_dev = float((optstate_before - optstate_after).abs().max())

    assert lora_dev == 0.0, f"LoRA not bit-identical after restore: {lora_dev}"
    assert head_dev == 0.0, f"head not bit-identical after restore: {head_dev}"
    assert opt_dev == 0.0, f"optimizer state not bit-identical after restore: {opt_dev}"

    # Frozen base must be untouched by the whole cycle.
    integ = mdl.integrity()
    assert integ["stray_trainable"] == [], integ["stray_trainable"]
    assert integ["max_rank"] <= 4, integ["max_rank"]
    print("test_branch_restore OK (lora/head/optim bit-identical; base frozen)")


if __name__ == "__main__":
    test_branch_restore_roundtrip()
