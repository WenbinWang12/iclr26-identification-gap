"""带 VLA 锚定的单任务训练循环。

与 `phase2l_additivity.train_with_penalty` 同构：语义逐行照抄 Phase-2K 的
`train_one_task`（每轮重排的种子派生、OOM 降 micro 而有效 batch 不变的归一、
step 划窗），只多一项锚定损失，否则各臂不可比。

**不需要保存快照副本。**协议 §1 说锚点目标来自"上一个任务边界的模型 θ̄"——
而在本任务开训**之前**那一刻，活着的模型就**是** θ̄。所以 `precompute` 在
训练循环开始前调用即可，额外显存为零，存储也天然是 O(1)。

锚定项**按 micro-batch 加**（与任务损失同样按样本数加权），因为它是数据相关的
期望，和 Phase-2L 那个只与参数有关的正交惩罚不同——后者按 step 加一次才对。
"""

from __future__ import annotations

import numpy as np
import torch

from experiments.phase2j_offset_conflict.run_probe import log


def train_one_task_with_anchor(model, tokenizer, examples, device, *,
                               args, optimizer, anchor=None) -> dict:
    """训练一个任务，返回逐 step 的任务损失与锚定损失。

    `anchor` 为 None、λ_a=0 或 V_{t-1} 为空时，本函数在数值上等价于
    Phase-2K 的 `train_one_task`（协议 §6 的两条反伪影检查）。
    """

    model.train()
    epochs = max(1, int(getattr(args, "epochs", 1) or 1))
    # 同时携带样本在原始序列里的下标，锚点缓存按这个下标取行。
    ordered: list[tuple[int, object]] = []
    for epoch in range(epochs):
        order = np.random.default_rng([args.seed, epoch]).permutation(len(examples))
        ordered.extend((int(i), examples[i]) for i in order)
    micro = args.batch_size
    per_step = micro * args.grad_accum
    task_losses: list[float] = []
    anchor_losses: list[float] = []

    for start in range(0, len(ordered), per_step):
        window = ordered[start : start + per_step]
        if not window:
            break
        while True:
            optimizer.zero_grad(set_to_none=True)
            pieces = [window[i : i + micro] for i in range(0, len(window), micro)]
            try:
                total = 0.0
                anchor_total = 0.0
                for chunk in pieces:
                    indices = [i for i, _ in chunk]
                    batch = [e for _, e in chunk]
                    enc = tokenizer(
                        [example.prompt for example in batch],
                        max_length=args.max_source,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    lab = tokenizer(
                        [example.label for example in batch],
                        max_length=args.max_target,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    label_ids = lab["input_ids"].clone()
                    label_ids[label_ids == tokenizer.pad_token_id] = -100
                    enc = {k: v.to(device) for k, v in enc.items()}
                    scale = len(batch) / len(window)
                    loss = model(**enc, labels=label_ids.to(device)).loss * scale
                    if anchor is not None and anchor.is_active():
                        pen = anchor.penalty(model, indices, enc, device) * scale
                        loss = loss + pen
                        anchor_total += float(pen.detach())
                    loss.backward()
                    total += float(loss.detach())
                break
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if micro <= 1:
                    raise
                micro = max(1, micro // 2)
                log("训练 OOM -> micro-batch 降为 %d（有效 batch 不变）" % micro)
        optimizer.step()
        task_losses.append(total)
        anchor_losses.append(anchor_total)

    return {"task_losses": task_losses, "anchor_losses": anchor_losses}
