"""带正交惩罚的单任务训练循环。

这是 `phase2j_offset_conflict.run_probe.train_one_task` 的**唯一**改动版本：
在每个 micro-batch 的任务损失之外，按 step 加一次正交惩罚。除此之外
（每轮重排的种子派生、OOM 降 micro 而有效 batch 不变的归一、step 划窗）
逐行照抄语义，否则 Phase-2L 的 BASE 臂与 Phase-2K 的 none 臂不可比。

惩罚**按 step 加一次**，不是按 micro-batch 加一次：惩罚只与参数有关、
与数据无关，按 micro 加会让它随 grad_accum 被放大 grad_accum 倍，
于是 λ 的含义随 batch 配置漂移。任务损失按样本数加权（`scale`），
惩罚不加权，这样 λ 相对"每 step 一次任务损失"是固定比例。
"""

from __future__ import annotations

import numpy as np
import torch

from experiments.phase2j_offset_conflict.run_probe import log


def train_one_task_with_penalty(model, tokenizer, examples, device, *,
                                args, optimizer, history=None) -> dict:
    """训练一个任务，返回逐 step 的任务损失与惩罚值。

    `history` 为 None 或 λ=0 时，本函数在数值上等价于 Phase-2K 的
    `train_one_task`（协议 §6 第二条反伪影检查）。
    """

    model.train()
    epochs = max(1, int(getattr(args, "epochs", 1) or 1))
    ordered: list = []
    for epoch in range(epochs):
        order = np.random.default_rng([args.seed, epoch]).permutation(len(examples))
        ordered.extend(examples[i] for i in order)
    micro = args.batch_size
    per_step = micro * args.grad_accum
    task_losses: list[float] = []
    penalties: list[float] = []

    for start in range(0, len(ordered), per_step):
        window = ordered[start : start + per_step]
        if not window:
            break
        while True:
            optimizer.zero_grad(set_to_none=True)
            pieces = [window[i : i + micro] for i in range(0, len(window), micro)]
            try:
                total = 0.0
                for batch in pieces:
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
                    loss.backward()
                    total += loss.detach().item()
                # 惩罚：每 step 一次，与 micro 拆分无关。
                pen_value = 0.0
                if history is not None and history.lam > 0.0 and history.n_history():
                    pen = history.penalty(model)
                    pen.backward()
                    pen_value = float(pen.detach())
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
        penalties.append(pen_value)

    return {"task_losses": task_losses, "penalties": penalties}
