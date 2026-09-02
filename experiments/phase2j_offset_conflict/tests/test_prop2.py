"""数值验证 Prop 2a 的断言（见 notes/theory_prop2_proof.md）。

写这些测试不是为了「证明」定理，而是防止我在纸上写下一个数值上就不成立的
断言 —— 之前 Prop 3 的谱尾形式就是这样被推翻的。
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from experiments.phase2j_offset_conflict.offsets import (
    _worst_case_quantization_radius,
    gauge_fix,
)


def chebyshev_radius_bruteforce(points: np.ndarray, *, grid: int = 401,
                                span: float = 6.0) -> float:
    """在 zero-sum 子空间上网格搜索 Chebyshev 半径（仅用于低维核对）。"""

    K = points.shape[1]
    assert K == 2, "网格核对只在 K=2（一维 gauge 空间）上做"
    best = np.inf
    for s in np.linspace(-span, span, grid):
        v = gauge_fix(np.array([-s / 2.0, s / 2.0]))
        best = min(best, float(np.max(np.linalg.norm(points - v, axis=1))))
    return best


def test_q1_equals_chebyshev_radius_binary():
    """§3 说 q_1 就是 Chebyshev 半径；在 K=2 上与网格搜索核对。"""

    rng = np.random.default_rng(0)
    for _ in range(5):
        scalars = rng.normal(scale=1.5, size=4)
        points = np.stack([gauge_fix(np.array([-s / 2, s / 2])) for s in scalars])
        q1 = _worst_case_quantization_radius(points, 1)
        assert q1 == pytest.approx(chebyshev_radius_bruteforce(points), abs=0.02)


def test_radius_at_least_half_diameter():
    """Prop 2a 的 r_S ≥ diam/2。"""

    rng = np.random.default_rng(1)
    for _ in range(20):
        K = int(rng.integers(2, 6))
        points = np.stack([gauge_fix(rng.normal(size=K)) for _ in range(5)])
        diameter = max(
            float(np.linalg.norm(a - b))
            for a, b in itertools.combinations(points, 2)
        )
        q1 = _worst_case_quantization_radius(points, 1)
        assert q1 >= diameter / 2 - 1e-9


def test_radius_zero_iff_optima_coincide():
    point = gauge_fix(np.array([1.0, -0.4, -0.6]))
    coincident = np.stack([point] * 4)
    assert _worst_case_quantization_radius(coincident, 1) == pytest.approx(0.0)

    spread = np.stack([point, gauge_fix(np.array([-1.0, 0.5, 0.5]))])
    assert _worst_case_quantization_radius(spread, 1) > 0.0


def test_radius_monotone_nonincreasing_in_m():
    """预算越大半径不增 —— 预算界该有的形状。"""

    rng = np.random.default_rng(2)
    points = np.stack([gauge_fix(rng.normal(size=4)) for _ in range(6)])
    radii = [_worst_case_quantization_radius(points, m) for m in range(1, 7)]
    for earlier, later in zip(radii, radii[1:]):
        assert later <= earlier + 1e-9
    assert radii[-1] == pytest.approx(0.0)


def test_disjoint_verbalizers_give_zero_radius_per_group():
    """Prop 1 推论：不共享时每组内只有一个任务，半径为 0。"""

    rng = np.random.default_rng(3)
    single = np.stack([gauge_fix(rng.normal(size=3))])
    assert _worst_case_quantization_radius(single, 1) == pytest.approx(0.0)


def test_kappa_assumption_is_necessary():
    """反例：距离大而准确率损失与距离不成比例（证明 §4 的必要性论证）。

    两个任务共享二元 verbalizer，裕度极大：把 b 移动一个有界量不翻转任何预测，
    所以「损失 ≥ 常数 × 距离」在无假设下不可能成立。
    """

    n = 64
    # 任务 1：真类 logit 高出 10，无需偏移
    task1 = np.zeros((n, 2))
    task1[:, 1] = 10.0
    y1 = np.ones(n, dtype=int)
    # 任务 2：真类 logit 低 10，需要 s > 10 的偏移
    task2 = np.zeros((n, 2))
    task2[:, 1] = -10.0
    y2 = np.ones(n, dtype=int)

    from experiments.phase2j_offset_conflict.offsets import (
        balanced_accuracy,
        fit_offset,
    )

    o1 = fit_offset(task1, y1)
    o2 = fit_offset(task2, y2, span=32.0)
    distance = float(np.linalg.norm(gauge_fix(o1) - gauge_fix(o2)))

    # b = 0 时任务 1 完美、任务 2 全错：最差损失为 1，与 distance 无关
    worst = max(
        balanced_accuracy(task1, y1, fit_offset(task1, y1)) - balanced_accuracy(task1, y1, None),
        balanced_accuracy(task2, y2, o2) - balanced_accuracy(task2, y2, None),
    )
    assert worst == pytest.approx(1.0)
    assert distance > 1.0
    # 把裕度缩小 1000 倍：距离随之缩小，损失仍是 1 —— 不成比例
    small1, small2 = task1 / 1000.0, task2 / 1000.0
    o2_small = fit_offset(small2, y2, span=1.0)
    small_distance = float(np.linalg.norm(
        gauge_fix(fit_offset(small1, y1)) - gauge_fix(o2_small)))
    worst_small = balanced_accuracy(small2, y2, o2_small) - balanced_accuracy(small2, y2, None)
    assert worst_small == pytest.approx(1.0)
    assert small_distance < distance
