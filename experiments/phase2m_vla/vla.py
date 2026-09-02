"""VLA —— Verbalizer-Logit Anchoring。锚住真正在漂的那个量。

见 `notes/phase2m_vla_protocol.md`（冻结 SHA b3225028…）§0：实测遗忘的 **78%**
可被一个逐任务的 verbalizer logit 偏移修掉，也就是说固定 LoRA 预算下遗忘的主体
是输出层上的**系统性平移**。Phase-2L 的正交惩罚约束 `A`，而优化器一直在更新 `B`，
所以产生旧任务 verbalizer logits 的那个函数根本没被约束——它买到的是容量而非保持
（6 seed、2 个 λ，peak 涨 2.5–3.0pp 而 later 掉 0.65–0.87pp）。

VLA 直接把旧任务 verbalizer 上的 logit 分布锚在上一个任务边界的快照上：

    L = L_task + λ_a · KL( p_θ̄(y | x)[V_{t-1}] ‖ p_θ(y | x)[V_{t-1}] )

* **数据无关**：只在**当前任务**的输入上评估，不存不放任何旧任务样本。
* **O(1) 存储**：快照是上一个模型（已编码所有更早任务），不是每任务一份。
* **只锚 V_{t-1}**：当前任务自己的 verbalizer 不锚，否则直接与 `L_task` 对抗。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class VerbalizerAnchor:
    """维护已见 verbalizer 首片 id 的并集，并按需预计算锚点目标。

    目标在每个任务**开训前算一次**并缓存（协议 §1），训练时只读缓存，所以
    额外开销是每任务一次评估遍，不是每步两次前向。
    """

    def __init__(self, *, lam: float, temperature: float = 2.0) -> None:
        if lam < 0:
            raise ValueError("lambda must be >= 0, got %r" % (lam,))
        if temperature <= 0:
            raise ValueError("temperature must be > 0, got %r" % (temperature,))
        self.lam = float(lam)
        self.temperature = float(temperature)
        # 已见任务的 verbalizer 首片 id（词表下标），按加入顺序去重。
        self.seen_columns: list[int] = []
        self._targets: torch.Tensor | None = None
        self._columns: list[int] = []

    def observe(self, piece_ids) -> None:
        """任务训完后登记它的 verbalizer 首片，供**后续**任务作为锚定列。"""
        for piece in piece_ids:
            value = int(piece)
            if value not in self.seen_columns:
                self.seen_columns.append(value)

    def active_columns(self) -> list[int]:
        """当前锚定的列 = V_{t-1}，即训练本任务时**已见**的那些首片。"""
        return list(self.seen_columns)

    def is_active(self) -> bool:
        return self.lam > 0.0 and bool(self.seen_columns)

    @torch.no_grad()
    def precompute(self, snapshot, tokenizer, examples, device, *,
                   max_source: int, batch_size: int) -> int:
        """用冻结快照在**当前任务**输入上算锚点目标，缓存起来。

        `snapshot` 必须是 eval 模式且不回传梯度的模型（调用方负责用
        `disable_adapter` 或独立副本给出上一个边界的参数）。返回缓存的行数。
        """

        columns = self.active_columns()
        self._columns = columns
        if not columns or self.lam == 0.0:
            self._targets = None
            return 0

        snapshot.eval()
        chunks = []
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            enc = tokenizer(
                [example.prompt for example in batch],
                max_length=max_source, truncation=True, padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            start_ids = torch.full(
                (len(batch), 1), snapshot.config.decoder_start_token_id,
                dtype=torch.long, device=device,
            )
            logits = snapshot(**enc, decoder_input_ids=start_ids).logits[:, 0, :]
            chunks.append(logits[:, columns].float().cpu())
        self._targets = torch.cat(chunks, dim=0)
        return int(self._targets.shape[0])

    def penalty(self, model, batch_indices, enc, device) -> torch.Tensor:
        """本批样本上的 KL 锚定项，已乘 λ_a。

        `batch_indices` 是这些样本在 `precompute` 传入序列里的下标——顺序必须
        对齐，否则锚的是别的样本，这一点由测试钉住。
        """

        if self._targets is None or not self.is_active():
            return torch.zeros((), device=device, dtype=torch.float32)

        start_ids = torch.full(
            (len(batch_indices), 1), model.config.decoder_start_token_id,
            dtype=torch.long, device=device,
        )
        logits = model(**enc, decoder_input_ids=start_ids).logits[:, 0, :]
        current = logits[:, self._columns].float()
        target = self._targets[batch_indices].to(device)

        tau = self.temperature
        # KL(teacher ‖ student)：teacher 是常量，故等价于交叉熵到 teacher 分布。
        # 乘 tau² 是蒸馏惯例，保证梯度量级不随 tau 缩放。
        teacher = F.softmax(target / tau, dim=-1)
        student = F.log_softmax(current / tau, dim=-1)
        kl = F.kl_div(student, teacher, reduction="batchmean") * (tau ** 2)
        return self.lam * kl

    def cached_rows(self) -> int:
        return 0 if self._targets is None else int(self._targets.shape[0])
