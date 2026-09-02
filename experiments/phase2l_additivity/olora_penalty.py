"""O-LoRA 形式的正交惩罚，作为**训练时**的表示层约束。

Phase-2L 只改训练循环里的这一项，度量路径（`fit_offset` / `gauge_fix` /
`scoped_shared_offsets` / `fit_scope_codebooks` / 首片受限 argmax /
risk 拟合-audit 打分）逐字复用 Phase-2J/2K，否则两臂不可比。见
`notes/phase2l_additivity_protocol.md` §1（冻结 SHA 8156a706…）。

惩罚形式（协议 §2）：

    L = L_task + λ · Σ_{t < 当前} ‖ A_current  A_tᵀ ‖_F²

历史 `A_t` 是 detach 过的快照，不回传梯度。这是我们在自己 harness 里对
O-LoRA 正则子的**再实现**，不是跑作者放出的代码，也不复现其报告数字；
协议 §2 明确禁止任何"复现/超过 O-LoRA"的措辞。
"""

from __future__ import annotations

import torch


def lora_a_matrices(model) -> dict[str, torch.Tensor]:
    """取出所有可训练的 LoRA A 矩阵，按参数名索引。

    peft 把 A 存成 `lora_A.<adapter>.weight`，形状 (r, in_features)。
    只收 `requires_grad` 的，避免把历史快照或已冻结层混进来。
    """

    out = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "lora_A" in name:
            out[name] = parameter
    return out


class OrthogonalHistory:
    """累积历史任务的 LoRA A 子空间，并给出正交惩罚。

    每个任务训完调用 `snapshot(model)`，把当前 A 存为 detach 的常量。
    `penalty(model)` 返回当前 A 与所有历史快照的 Frobenius 内积平方和。

    任务 1 上惩罚**按构造为 0**（历史为空），这是协议 §6 的一条反伪影检查，
    在 `penalty` 里用返回零张量体现，并由测试钉住。
    """

    def __init__(self, *, lam: float) -> None:
        if lam < 0:
            raise ValueError("lambda must be >= 0, got %r" % (lam,))
        self.lam = float(lam)
        self.history: list[dict[str, torch.Tensor]] = []

    def snapshot(self, model) -> None:
        """任务边界处冻结当前 A。用 float32 存，避免 bf16 累积误差。"""

        frozen = {name: parameter.detach().clone().float()
                  for name, parameter in lora_a_matrices(model).items()}
        if not frozen:
            raise RuntimeError("no trainable lora_A found; check target_modules")
        self.history.append(frozen)

    def penalty(self, model) -> torch.Tensor:
        """Σ_t ‖ A A_tᵀ ‖_F²，乘上 λ。

        λ = 0 时直接短路返回 0，这样"λ=0 且代码路径激活"在数值上必须等于
        `--cl-method none`（协议 §6 第二条检查），惩罚只通过系数进入。
        """

        current = lora_a_matrices(model)
        if not current:
            raise RuntimeError("no trainable lora_A found; check target_modules")
        device = next(iter(current.values())).device
        if self.lam == 0.0 or not self.history:
            return torch.zeros((), device=device, dtype=torch.float32)

        total = torch.zeros((), device=device, dtype=torch.float32)
        for past in self.history:
            for name, matrix in current.items():
                old = past.get(name)
                if old is None:
                    # 层集合在任务间不应变化；变了就是接线错误，不静默跳过。
                    raise RuntimeError("lora_A layer set changed: missing %s" % name)
                product = matrix.float() @ old.to(matrix.device).T
                total = total + (product ** 2).sum()
        return self.lam * total

    def n_history(self) -> int:
        return len(self.history)
