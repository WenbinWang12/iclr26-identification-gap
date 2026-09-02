"""Streaming Scope-Offset table（SSO），协议 `notes/phase2p_streaming_scope_offset_protocol.md` §1。

与 Phase-2K 的 `R_shr_scoped` 的唯一区别在**表是什么时候拟合的**：

* `R_shr_scoped`：在每个 stage 用**所有已见任务**的 risk logits 重新拟合。那是
  rehearsal——它重读旧任务的样本。协议 §0 记的就是这个缺口。
* SSO：任务 `t` 训完那一刻，只用 `partitions[t].risk` 拟合 `b_t*`，存进
  `table[S(t)]`，然后**再也不回头**。推理时按域取 Chebyshev 中心（m=1）或最近
  码字（m=4）。存的是 `Σ_S |table[S]|·(K_S−1)` 个浮点数，Order-4 上约 20 个。

这里**不新增正则**：训练路径是 `--cl-method none`，逐字与基线相同。2N §5 结局 4
禁止第四个锚，SSO 改的只是输出层在推理时做什么。

这个文件不重新实现几何：Chebyshev 中心就是 `build_codebook(X, 1)`（Prop 2 的几何
半，无条件成立），路由复用 `route_prototype`。复用而不是复制，是因为「m=1 是
Chebyshev 中心」这件事是 Prop 2 的内容，两份实现漂移一次，定理与代码的对应就没了。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from experiments.phase2j_offset_conflict.offsets import (
    balanced_accuracy,
    fit_offset,
    gauge_fix,
)
from experiments.phase2k_qoc.qoc import (
    ScopeCodebook,
    TaskOptima,
    build_codebook,
    scope_key,
)


@dataclass
class StreamingScopeOffsets:
    """按域累积 per-task 最优偏移的流式表。

    `record` 在每个任务边界调用一次，**只吃当前任务自己的 risk logits**。
    表里除了偏移向量本身什么都不存：没有样本、没有 logits、没有旧任务张量。
    """

    #: scope -> 该域下按登记顺序排列的 TaskOptima
    table: dict[tuple[str, ...], list[TaskOptima]] = field(default_factory=dict)

    #: scope -> task -> (状态均值, 该均值背后的样本数)。只给 m>1 的路由器用；
    #: m=1 不需要路由，见 `codebooks` 的注释。
    prototype_state: dict[tuple[str, ...], dict[str, tuple[np.ndarray, int]]] = \
        field(default_factory=dict)

    def record(self, task: str, labels, pieces, logits: np.ndarray,
               targets: np.ndarray, *, objective=balanced_accuracy,
               state_mean: np.ndarray | None = None,
               state_count: int = 0) -> TaskOptima:
        """登记任务 `task` 的最优偏移。协议 §1 第 1-2 步。

        `logits`/`targets` **必须**来自 `partitions[task].risk`，且只来自它。
        调用点由 `tests/test_wiring.py` 静态钉住；这里再断一次形状，因为把
        多个任务的 logits 拼起来传进来是唯一能让这个类偷看旧数据的方式。
        """
        logits = np.asarray(logits, dtype=np.float64)
        targets = np.asarray(targets)
        if logits.ndim != 2:
            raise ValueError("logits must be 2-D (n, K)")
        if logits.shape[0] != targets.shape[0]:
            raise ValueError("logits and targets disagree on n")
        if logits.shape[1] != len(labels):
            raise ValueError(
                "logits width %d != len(labels) %d; SSO fits one task at a time "
                "and pooling tasks would make the table rehearsal"
                % (logits.shape[1], len(labels)))

        optimum = fit_offset(logits, targets, objective=objective)
        entry = TaskOptima(task=task, labels=tuple(labels),
                           pieces=tuple(int(p) for p in pieces),
                           optimum=gauge_fix(optimum))
        self.table.setdefault(entry.scope, []).append(entry)
        if state_mean is not None:
            # 只存**均值**和计数，不存状态矩阵：否则 d_model × n 就是一个 rehearsal
            # 缓冲区。计数是为了让簇原型能按样本数加权，和 2K 的
            # `np.concatenate(...).mean(0)` 在数学上一致。
            self.prototype_state.setdefault(entry.scope, {})[task] = (
                np.asarray(state_mean, dtype=np.float64).copy(), int(state_count))
        return entry

    # ---- 表的规模：协议 §3 判据 P6 ----

    def stored_floats(self) -> int:
        """`Σ_S |table[S]| · (K_S − 1)`。gauge-fix 掉了一个自由度，所以是 K−1。"""
        return sum(len(members) * (len(scope) - 1)
                   for scope, members in self.table.items())

    def n_entries(self) -> int:
        return sum(len(v) for v in self.table.values())

    def router_floats(self) -> int:
        """路由器状态的浮点数，**不在协议 §3 P6 的 `stored_floats` 公式里**。

        P6 的公式 `Σ_S |table[S]|·(K_S−1)` 只覆盖偏移表。冻结协议时我没想到
        m>1 的 prototype 路由需要额外状态：2K 版本靠重读所有成员任务的 risk
        状态算簇质心，那是 rehearsal，流式版只能改为每任务存一个 d_model 均值。
        所以 `SSO(m=1)` 是纯 `stored_floats`（Order-4 上约 20 个，且**不需要
        路由器**），而 `SSO(m=4)` 额外要这么多。两个数分开上报，不合并、不掩盖。
        """
        return sum(int(mean.size) for per in self.prototype_state.values()
                   for mean, _ in per.values())

    # ---- 推理侧 ----

    def codebooks(self, m: int) -> dict[tuple[str, ...], ScopeCodebook]:
        """按域建 m-码本。`m = 1` 时中心就是 Chebyshev 中心（Prop 2 几何半）。

        不调 `fit_scope_codebooks`：那个函数按域分组一份 optima 列表，而这里的
        分组已经是表的结构本身，再分一次只会多一条可以漂移的路径。

        `m = 1` 时 `centres` 只有一行，任何路由器都只能选它（`route_prototype`
        在 `prototypes is None` 时返回全 0 下标），所以 m=1 **不需要路由器状态**。
        """
        books = {}
        for scope, members in self.table.items():
            X = np.stack([e.aligned() for e in members])
            centres, radius, assign = build_codebook(X, m)
            book = ScopeCodebook(
                scope=scope, members=[e.task for e in members],
                centres=centres, radius=radius,
                assignment={e.task: assign[i] for i, e in enumerate(members)},
            )
            if m > 1:
                self._attach_prototypes(book, scope)
            books[scope] = book
        return books

    def _attach_prototypes(self, book: ScopeCodebook,
                           scope: tuple[str, ...]) -> None:
        """簇原型 = 该簇成员任务状态均值的**按样本数加权**均值。

        等于 2K 的 `np.concatenate([states_for(n) for n in members]).mean(0)`，
        但只用已存的每任务均值和计数，不重读任何旧任务的样本。
        """
        per = self.prototype_state.get(scope)
        if not per:
            return
        d = next(iter(per.values()))[0].size
        vectors, counts = [], []
        for j in range(book.centres.shape[0]):
            members = [n for n, idx in book.assignment.items() if idx == j]
            num = np.zeros(d, dtype=np.float64)
            den = 0
            for n in members:
                if n in per:
                    mean, cnt = per[n]
                    num += mean * cnt
                    den += cnt
            vectors.append(num / den if den else np.zeros(d))
            counts.append(int(den))
        book.prototypes = np.stack(vectors)
        book.prototype_counts = counts


def chebyshev_centre(X: np.ndarray) -> np.ndarray:
    """一组最优偏移的 Chebyshev 中心，gauge-fixed。

    直接走 `build_codebook(X, 1)`，因为「m=1 的 minimax 码本中心 = Chebyshev
    中心」正是 Prop 2 的几何半在说的事。二元域上它必须等于标量对比的
    `(min + max) / 2`；协议 §6 用闭式把这一点钉在测试里。
    """
    centres, _, _ = build_codebook(np.asarray(X, dtype=np.float64), 1)
    return gauge_fix(centres[0])
