"""Gauge-Fixed logit Anchoring（GFA），协议 `notes/phase2n_gauge_fixed_anchor_protocol.md` §1。

与 Phase-2M 的 VLA 的唯一区别在**锚哪个量**：

* VLA 锚 `softmax(z_{V_{t-1}} / tau)`——对所有历史 verbalizer 列做归一化。这个量
  是**规范相关**的：给某个 scope 的所有列加一个常数会改变它，而不改变任何一次
  受限 argmax。Phase-2M §A2.3 测到 VLA 的全部代价（−4.78pp）都被免费的
  per-scope 常数移除（移除比例 1.059），即代价整个落在这个没人读的方向上。
* GFA 锚 `P_0 z_S`，即每个 scope 内部**减去均值**后的对比向量。加常数它不变，
  受限 argmax 也不变，所以它携带的正是决策读到的那部分、且不多携带任何东西。
  这也正是 `theory_output_layer_capacity_v1.md` §2 的 Prop 2a 所陈述的对象。

用平方误差而不是 KL：`g_S` 是未归一化的对比向量，不是分布；套 softmax 会把
归一化——也就是规范——重新引进来，那就退回 VLA 了。

单任务 scope（`K_S = 1`）被**排除**：此时 `g_S ≡ 0` 恒成立，没有任何梯度，
纳进来只会稀释均值。这就是 Prop 1 的另一种说法。
"""

from __future__ import annotations

import torch


class GaugeFixedAnchor:
    """按 scope 记住 verbalizer 列，锚定 zero-mean 投影后的 logit 对比。

    存储 O(1)：只在任务边界前用**当下活着的模型**算一遍目标，不留任何模型副本。
    """

    def __init__(self, lam: float) -> None:
        self.lam = float(lam)
        #: scope 名 -> 该 scope 的 verbalizer first-piece 列 id（去重、保序）
        self.scopes: dict[str, list[int]] = {}
        #: 上一次 precompute 的缓存：行号 -> {scope: 目标向量}
        self._targets: dict[int, dict[str, torch.Tensor]] = {}

    # ---- 历史登记 ----

    def observe(self, scope: str, piece_ids) -> None:
        """在一个任务**训练之后**登记它的 scope。任务 1 训练前 `scopes` 为空。"""
        seen, out = set(), []
        for i in piece_ids:
            i = int(i)
            if i not in seen:
                seen.add(i)
                out.append(i)
        if not out:
            return
        prev = self.scopes.get(scope)
        if prev is None:
            self.scopes[scope] = out
        else:
            # 同一个 scope 被第二个任务共享：列**集合**必须一致，否则 scope 的
            # 定义（共享 verbalizer）就被违反了，应当炸而不是静默合并。
            #
            # 比的是集合而非序列：MNLI 与 CB 共享 verbalizer，但两边 `task.labels`
            # 的顺序不同（[7163,3,27252] vs [3,27252,7163]），列集合完全相同。
            # 顺序在这里无所谓——`gauge_fix` 减的是均值，`‖·‖²` 是对称的，两者都
            # 与列的排列无关；真正要拦的是集合不同，那才意味着不是同一个
            # verbalizer。第一版比了序列，冒烟跑在任务 2 就炸了。
            #
            # 保留**首次登记的顺序**：目标缓存已经按那个顺序算好了，换序会让
            # 目标与当前 logits 逐元素错位。
            if set(prev) != set(out):
                raise RuntimeError(
                    "scope %r already registered with columns %s but task "
                    "supplies %s; a scope is by definition one verbalizer"
                    % (scope, prev, out))

    def active_scopes(self) -> list[str]:
        """参与锚定的 scope：已登记且**非单任务**（`K_S ≥ 2`）。协议 §1。"""
        return [s for s, cols in sorted(self.scopes.items()) if len(cols) >= 2]

    def is_active(self) -> bool:
        return self.lam != 0.0 and bool(self.active_scopes())

    # ---- 目标 ----

    @staticmethod
    def gauge_fix(z: torch.Tensor) -> torch.Tensor:
        """零均值投影 `P_0`。加常数不变，受限 argmax 不变。

        **在 float64 里算均值和减法。** 数学上不变性是精确的，但 float32 里
        `z + c` 与 `mean(z + c)` 相减是灾难性抵消：c = 42 时惩罚会漂 1.3e-6，
        刚好越过协议 §6 冻结的 1e-6 阈值。float64 把残差压到 1e-15 量级。
        这里的向量长度是 `K_S`（个位数），float64 的开销可以忽略，所以正确的做法
        是让实现去满足冻结的阈值，而不是把阈值放宽到实现能过。
        """
        z64 = z.to(torch.float64)
        return z64 - z64.mean(dim=-1, keepdim=True)

    @torch.no_grad()
    def precompute(self, snapshot, tokenizer, examples, device, *,
                   max_source: int = 256, batch_size: int = 8) -> int:
        """在任务开训**前**算目标。此刻活着的模型就是 θ̄，所以不需要副本。

        返回缓存的行数。`examples` 只来自当前任务 `D_t`——历史任务的样本一个不读。
        """
        self._targets = {}
        scopes = self.active_scopes()
        if self.lam == 0.0 or not scopes:
            return 0

        cols = {s: torch.tensor(self.scopes[s], device=device, dtype=torch.long)
                for s in scopes}
        rows = 0
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]
            logits = _first_step_logits(snapshot, tokenizer, chunk, device,
                                        max_source=max_source)
            for j in range(logits.shape[0]):
                per_scope = {}
                for s in scopes:
                    z = logits[j].index_select(0, cols[s])
                    per_scope[s] = self.gauge_fix(z).detach().clone()
                self._targets[start + j] = per_scope
                rows += 1
        return rows

    # ---- 惩罚 ----

    def penalty(self, model, indices, encoded, device) -> torch.Tensor:
        """`λ_g · mean_S mean_x ‖P_0 z_S(θ) − P_0 z_S(θ̄)‖²`。

        `indices` 是这个 micro-batch 每一行在 precompute 缓存里的行号，顺序必须和
        `encoded` 一致——洗牌时要把 (index, example) 一起带走。
        """
        scopes = self.active_scopes()
        if self.lam == 0.0 or not scopes or not self._targets:
            return torch.zeros((), device=device)

        logits = _first_step_logits_grad(model, encoded, device)
        cols = {s: torch.tensor(self.scopes[s], device=device, dtype=torch.long)
                for s in scopes}

        # total 用 float64 累加，与 gauge_fix 的精度一致。
        total = torch.zeros((), device=device, dtype=torch.float64)
        counted = 0
        for j, idx in enumerate(indices):
            target = self._targets.get(int(idx))
            if target is None:
                continue
            for s in scopes:
                z = logits[j].index_select(0, cols[s])
                diff = self.gauge_fix(z) - target[s]
                total = total + (diff * diff).sum()
            counted += 1
        if counted == 0:
            return torch.zeros((), device=device)
        # 回到模型的 dtype 再交给优化器：float64 只用于规范投影本身。
        out = self.lam * total / (counted * len(scopes))
        return out.to(logits.dtype)


def _first_step_logits(model, tokenizer, examples, device, *, max_source: int):
    """teacher 侧：第一个解码步的 vocab logits，no_grad。

    用 `example.prompt`——和 Phase-2M 的 `vla.precompute` 逐字同一个字段，
    这样两期锚的是同一批输入表示，方法之间的差别只在"锚哪个量"上。
    """
    enc = tokenizer([e.prompt for e in examples], return_tensors="pt",
                    padding=True, truncation=True, max_length=max_source).to(device)
    dec = torch.zeros((enc["input_ids"].shape[0], 1), dtype=torch.long, device=device)
    out = model(**enc, decoder_input_ids=dec)
    return out.logits[:, 0, :].float()


def _first_step_logits_grad(model, encoded, device):
    """student 侧：同一个量，但保留梯度。`encoded` 已在 device 上。"""
    ids = encoded["input_ids"]
    dec = torch.zeros((ids.shape[0], 1), dtype=torch.long, device=device)
    out = model(input_ids=ids, attention_mask=encoded["attention_mask"],
                decoder_input_ids=dec)
    return out.logits[:, 0, :].float()
