"""QOC —— Quantized Offset Codebook，`notes/method_qoc_v1.md` 的实现。

方法是理论 Prop 3 的构造性对偶：理论给出「m 个任务无关偏移」必然损失的下界
q_m，这里就按 q_m 的定义去**构造**那 m 个偏移，使覆盖半径按构造等于 q_m。

两级容量，按代价排序：

1. **词表域划分（免费）**——提示词里本来就写着 `Option: ...`，所以按**活跃标签
   集**索引偏移是读输入、不是读 task ID。这让 `neutral` 在 NLI 域和情感域取不同
   值，而单个全局 b 做不到。
2. **域内偏移码本（花预算）**——同一个域里标签集相同，输入里的选项列表分不开
   成员任务，这正是 q_m 生效的地方。每个域保留至多 m 个 gauge-fix 偏移。

路由器是可能失效的那一半，四种全报（含 oracle 仅作上界参照）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools

import numpy as np

from experiments.phase2j_offset_conflict.offsets import (
    balanced_accuracy,
    fit_offset,
    gauge_fix,
)


def scope_key(labels) -> tuple[str, ...]:
    """活跃标签集的规范键。

    按标签**字符串**排序而非首片 id：域是从提示词里的选项列表读出来的，那是
    字符串层面的东西。首片碰撞是**域内**偏移向量的坐标问题，由
    `TaskOptima.pieces` 承担，两者不能混。
    """

    return tuple(sorted(labels))


@dataclass(frozen=True)
class TaskOptima:
    """一个任务累积下来的最优偏移，连同它的坐标信息。"""

    task: str
    labels: tuple[str, ...]
    pieces: tuple[int, ...]          # 每个标签的首片 id
    optimum: np.ndarray              # 按 labels 顺序，已 gauge-fix

    @property
    def scope(self) -> tuple[str, ...]:
        return scope_key(self.labels)

    def aligned(self) -> np.ndarray:
        """按域的规范标签顺序重排的最优偏移。"""
        order = [self.labels.index(label) for label in self.scope]
        return gauge_fix(self.optimum[order])


def _candidate_centres(X: np.ndarray) -> list[np.ndarray]:
    """k-center 的候选中心：各点本身，加上所有成对中点。

    这与 `offsets._worst_case_quantization_radius` 用的候选集**完全一致**，
    码本的覆盖半径才能按构造等于那里报的 q_m，而不是「近似等于」。

    **这个候选集只在 `d = 1` 或 `n ≤ 2` 时含真正的最优中心**
    （`notes/theory_prop3_m2_proof.md` §6）。`d ≥ 2` 且 `n ≥ 3` 时最优中心可以是
    三点及以上的 Chebyshev 中心，不在「点∪成对中点」里：边长 1 的正三角形、m=1，
    本枚举给 0.8660，真值是外心半径 1/√3 = 0.5774，高估 1.5 倍。
    于是本函数返回的半径在 d ≥ 2 时是 q_m 的**上界**，方向对我们有利
    （下界命题里高估半径 = 少宣称），但不要称它为「精确」。
    Order-4 里被打分的共享域全是 K = 2，即 d = 1，故实际报出的数是精确的。
    """

    cands = [X[i] for i in range(X.shape[0])]
    for i, j in itertools.combinations(range(X.shape[0]), 2):
        cands.append((X[i] + X[j]) / 2.0)
    return cands


def build_codebook(X: np.ndarray, m: int) -> tuple[np.ndarray, float, list[int]]:
    """minimax(k-center) 码本；`d = 1` 或 `n ≤ 2` 时精确，否则是上界。

    返回 (centres (m', K), 覆盖半径, 每点分配到的中心下标)。
    枚举是可行的：Order-4 里同一个域最多 4 个任务，候选数 ≤ 4 + 6 = 10。
    点数不足 m 时退化为「每个点一个中心」，半径 0。
    精确性范围见 `_candidate_centres` 的注释。
    """

    n = X.shape[0]
    if n == 0:
        raise ValueError("codebook needs at least one point")
    if m >= n:
        return X.copy(), 0.0, list(range(n))

    cands = _candidate_centres(X)
    best_radius, best_combo = float("inf"), None
    for combo in itertools.combinations(range(len(cands)), m):
        C = np.stack([cands[i] for i in combo])
        d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=-1).min(axis=1)
        radius = float(d.max())
        if radius < best_radius - 1e-12:
            best_radius, best_combo = radius, combo
    C = np.stack([cands[i] for i in best_combo])
    assign = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=-1).argmin(axis=1)
    # 中心也要 gauge-fix：中点的均值不一定为 0。
    C = np.stack([gauge_fix(c) for c in C])
    return C, best_radius, [int(a) for a in assign]


@dataclass
class ScopeCodebook:
    """一个词表域的码本，以及它的路由所需状态。"""

    scope: tuple[str, ...]
    members: list[str]
    centres: np.ndarray                        # (m', K)，列序 = scope
    radius: float                              # 按构造 = q_{m'}
    assignment: dict[str, int]                 # task -> 中心下标
    prototypes: np.ndarray | None = None       # (m', d)，剥离表头的诚实版
    # 保留 `Task:`/`Dataset:` 表头的版本，**只用于量化身份泄漏幅度**，
    # 不构成任何方法证据：官方提示词把任务名写在输入里，读它的路由器等于 oracle。
    prototypes_official: np.ndarray | None = None
    prototype_counts: list[int] = field(default_factory=list)

    def offset_for(self, labels, index: int) -> np.ndarray:
        """把第 index 个中心映射回给定任务的标签顺序。"""
        order = [self.scope.index(label) for label in labels]
        return gauge_fix(self.centres[index][order])


def fit_scope_codebooks(optima, m: int) -> dict[tuple[str, ...], ScopeCodebook]:
    """按域分组，逐域建精确 k-center 码本。"""

    groups: dict[tuple[str, ...], list[TaskOptima]] = {}
    for entry in optima:
        groups.setdefault(entry.scope, []).append(entry)

    books = {}
    for scope, members in groups.items():
        X = np.stack([entry.aligned() for entry in members])
        centres, radius, assign = build_codebook(X, m)
        books[scope] = ScopeCodebook(
            scope=scope,
            members=[entry.task for entry in members],
            centres=centres,
            radius=radius,
            assignment={entry.task: assign[i] for i, entry in enumerate(members)},
        )
    return books


# ---------------------------------------------------------------- 路由器

def route_single(book: ScopeCodebook, logits: np.ndarray, labels,
                 states: np.ndarray | None = None) -> np.ndarray:
    """m=1 参照：始终用第 0 个中心。"""
    return np.zeros(logits.shape[0], dtype=int)


def _margins(book: ScopeCodebook, logits: np.ndarray, labels) -> np.ndarray:
    """(n, m)：每个样本在每个中心下的 top1−top2 间隔。"""
    n, m = logits.shape[0], book.centres.shape[0]
    out = np.empty((n, m))
    for j in range(m):
        shifted = logits + book.offset_for(labels, j)[None, :]
        part = np.partition(shifted, -2, axis=1)
        out[:, j] = part[:, -1] - part[:, -2]
    return out


def route_confidence(book: ScopeCodebook, logits: np.ndarray, labels,
                     states: np.ndarray | None = None) -> np.ndarray:
    """逐样本取最大间隔的中心。**实测退化，保留以记录该失效。**

    原始直觉是「正确的偏移把样本推离决策边界」。实测（seed 1，T5-large）它与
    oracle 的一致率在 `{False,True}` 域上是 0.000–0.008，即系统性地**反着选**。
    原因是可以事后想清楚的：间隔最大的偏移就是把所有样本最粗暴地推向某一个类的
    那个偏移，因为此时 top1 被抬高、top2 被压低，间隔单调增大。所以逐样本
    最大间隔奖励的是最极端的中心，而不是最匹配的那个 —— 这个判据是退化的。

    保留它并如实报告，因为它是「无参数路由」最自然的第一个想法。
    """
    return _margins(book, logits, labels).argmax(axis=1)


def route_batch_margin(book: ScopeCodebook, logits: np.ndarray, labels,
                       states: np.ndarray | None = None) -> np.ndarray:
    """整批共用一个中心：取批内平均间隔最大者。**带直推假设。**

    修掉逐样本版本的退化：单个样本的间隔可以被极端偏移拉大，但整批样本的
    *平均*间隔不行 —— 把所有样本推向同一类，会让本属另一类的样本间隔变小。

    代价必须说清楚：这假设查询是**按任务成批到达**的（批内同分布），属于直推
    设定，比逐样本路由弱。它不读任务身份也不读标签，但它读了「这一批是同一个
    任务」这个事实。任何用它的数字都必须标注这个假设。
    """
    scores = _margins(book, logits, labels).mean(axis=0)
    return np.full(logits.shape[0], int(scores.argmax()), dtype=int)


def _nearest_prototype(prototypes, logits: np.ndarray,
                       states: np.ndarray | None) -> np.ndarray:
    if states is None or prototypes is None:
        return np.zeros(logits.shape[0], dtype=int)
    A = states / (np.linalg.norm(states, axis=1, keepdims=True) + 1e-12)
    B = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-12)
    return (A @ B.T).argmax(axis=1)


def route_prototype(book: ScopeCodebook, logits: np.ndarray, labels,
                    states: np.ndarray | None = None) -> np.ndarray:
    """最近原型（诚实版）：输入已剥掉 `Task:`/`Dataset:` 表头。"""
    return _nearest_prototype(book.prototypes, logits, states)


def route_prototype_official(book: ScopeCodebook, logits: np.ndarray, labels,
                             states: np.ndarray | None = None) -> np.ndarray:
    """最近原型（**泄漏版**，仅用于量化泄漏幅度，不是方法）。

    官方提示词字面写着 `Dataset:WiC`，所以这个路由能直接从文本里读出任务身份。
    报告它的唯一目的是说明「读输入的路由器有多容易变成 oracle」。
    """
    return _nearest_prototype(book.prototypes_official, logits, states)


ROUTERS = {
    "single": route_single,
    "confidence": route_confidence,
    "batch_margin": route_batch_margin,
    "prototype": route_prototype,
    "prototype_official_leaky": route_prototype_official,
}


def score_with_codebook(book: ScopeCodebook, task: str, labels,
                        logits: np.ndarray, targets: np.ndarray, *,
                        router: str, states: np.ndarray | None = None,
                        objective=balanced_accuracy) -> dict:
    """用码本 + 指定路由给一个任务打分。

    `oracle` 路由直接查该任务真实所属的簇 —— **只作上界参照，不是方法**。
    """

    if router == "oracle":
        index = book.assignment.get(task, 0)
        chosen = np.full(logits.shape[0], index, dtype=int)
    else:
        chosen = ROUTERS[router](book, logits, labels, states)

    # 逐样本套用各自路由到的偏移。偏移是逐样本的，所以不能直接调
    # objective(logits, targets, offset)（它假设全体共用一个偏移）。
    shifted = np.empty_like(logits, dtype=np.float64)
    for j in range(book.centres.shape[0]):
        mask = chosen == j
        if mask.any():
            shifted[mask] = logits[mask] + book.offset_for(labels, j)[None, :]
    accuracy = objective(shifted, targets, None)
    return {
        "accuracy": float(accuracy),
        "router": router,
        "n_centres": int(book.centres.shape[0]),
        "radius_q_m": float(book.radius),
        "route_histogram": [int((chosen == j).sum())
                            for j in range(book.centres.shape[0])],
        "route_matches_oracle": (
            float((chosen == book.assignment[task]).mean())
            if task in book.assignment else None
        ),
    }


def fit_task_optimum(task: str, labels, pieces, logits: np.ndarray,
                     targets: np.ndarray, *, objective=balanced_accuracy) -> TaskOptima:
    """在该任务的 risk 划分上拟合它的 per-task 最优偏移。"""
    optimum = fit_offset(logits, targets, objective=objective)
    return TaskOptima(task=task, labels=tuple(labels), pieces=tuple(pieces),
                      optimum=gauge_fix(optimum))
