"""数值验证 Prop 3a/3b/3c 的断言（见 notes/theory_prop3_m2_proof.md）。

同 test_prop2.py 的用意：不是「证明」定理，而是防止纸上写下数值上就不成立的
断言 —— Prop 3 的谱尾形式当初正是这样被推翻的。

这一组里有三个测试是**故意用来钉住反面结论**的：
* `test_enumeration_is_not_exact_in_dim_two` —— 候选集枚举在 d≥2 高估 1.5×；
* `test_pigeonhole_is_strict_in_higher_dimensions` —— 鸽巢界在 d≥2 不取等；
* `test_per_example_routing_can_beat_the_single_offset_oracle` —— Cor 3c 对
  逐样本路由不成立。
它们在于把「已知不成立的范围」固定下来，避免以后被误当成一般结论引用。
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from experiments.phase2j_offset_conflict.offsets import (
    _worst_case_quantization_radius as q_m,
)


def closed_form_q2(points: np.ndarray) -> float:
    """Prop 3a 的闭式：排序后枚举连续切分，每块取中程半宽。"""
    xs = np.sort(np.ravel(points))
    n = len(xs)
    if n <= 2:
        return 0.0
    return min(
        max((xs[j] - xs[0]) / 2.0, (xs[-1] - xs[j + 1]) / 2.0)
        for j in range(n - 1)
    )


def pigeonhole_bound(points: np.ndarray, m: int) -> float:
    """Prop 3b：(1/2) · max over (m+1)-subsets of min pairwise distance."""
    n = len(points)
    if n < m + 1:
        return 0.0
    best = 0.0
    for subset in itertools.combinations(range(n), m + 1):
        best = max(best, min(
            float(np.linalg.norm(points[a] - points[b]))
            for a, b in itertools.combinations(subset, 2)
        ))
    return best / 2.0


# ---------------------------------------------------------------- Prop 3a

def test_closed_form_q2_matches_the_exact_enumerator_in_dim_one():
    """Prop 3a 的闭式必须与实际用来报数的枚举器逐位一致。"""
    rng = np.random.default_rng(3)
    for _ in range(300):
        n = int(rng.integers(2, 8))
        points = rng.normal(size=(n, 1)) * float(rng.uniform(0.1, 10.0))
        assert q_m(points, 2) == pytest.approx(closed_form_q2(points), abs=1e-9)


def test_q2_is_zero_for_two_tasks_not_merely_equal_to_q1():
    """T=2 时 q_2=0 严格小于 q_1 —— 这正是 {Bad,Good} 域实测 q_2=0 的原因。

    笔记 §2.2 记录了我最初把等号条件误写成 T<=2；这个测试钉住正确的版本。
    """
    points = np.array([[-1.5], [2.5]])
    assert q_m(points, 1) == pytest.approx(2.0, abs=1e-12)
    assert q_m(points, 2) == pytest.approx(0.0, abs=1e-12)
    single = np.array([[0.7]])
    assert q_m(single, 1) == pytest.approx(0.0, abs=1e-12)
    assert q_m(single, 2) == pytest.approx(0.0, abs=1e-12)


def test_block_midrange_is_optimal_within_a_block():
    """Prop 3a Step 2：一维块内最优中心是中程点，半径是半宽。"""
    rng = np.random.default_rng(5)
    for _ in range(200):
        block = np.sort(rng.normal(size=int(rng.integers(2, 9))))
        mid = (block[0] + block[-1]) / 2.0
        radius = (block[-1] - block[0]) / 2.0
        assert float(np.abs(block - mid).max()) == pytest.approx(radius, abs=1e-12)
        for candidate in rng.normal(size=20):
            assert float(np.abs(block - candidate).max()) >= radius - 1e-12


def test_four_member_scope_matches_hand_computation():
    """{False,True} 有 4 个成员，是 m=1,2,4 三档都有意义的那个域。"""
    points = np.array([[0.0], [1.0], [4.0], [5.0]])
    assert q_m(points, 1) == pytest.approx(2.5, abs=1e-12)
    # 最优切分是 {0,1} | {4,5}，两块半宽都是 0.5
    assert q_m(points, 2) == pytest.approx(0.5, abs=1e-12)
    assert closed_form_q2(points) == pytest.approx(0.5, abs=1e-12)
    assert q_m(points, 4) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- Prop 3b

def test_pigeonhole_is_a_valid_lower_bound_in_every_dimension():
    rng = np.random.default_rng(7)
    for _ in range(300):
        dim = int(rng.integers(1, 5))
        n = int(rng.integers(2, 7))
        m = int(rng.integers(1, 4))
        points = rng.normal(size=(n, dim))
        assert q_m(points, m) >= pigeonhole_bound(points, m) - 1e-9


def test_pigeonhole_is_tight_in_dim_one():
    """d=1 时鸽巢界恰好**等于** q_m —— 这是把界写进论文的依据。"""
    rng = np.random.default_rng(11)
    for _ in range(400):
        n = int(rng.integers(2, 8))
        m = int(rng.integers(1, 5))
        points = rng.normal(size=(n, 1)) * float(rng.uniform(0.1, 10.0))
        assert q_m(points, m) == pytest.approx(
            pigeonhole_bound(points, m), abs=1e-9)


def test_pigeonhole_is_strict_in_higher_dimensions():
    """反面钉子：d>=2 时鸽巢界普遍不取等，紧性主张只能限定在 d=1。"""
    rng = np.random.default_rng(13)
    strict = 0
    for _ in range(200):
        dim = int(rng.integers(2, 5))
        n = int(rng.integers(3, 7))
        points = rng.normal(size=(n, dim))
        if q_m(points, 2) > pigeonhole_bound(points, 2) + 1e-9:
            strict += 1
    assert strict > 20, "d>=2 应当经常严格大于；若不再如此，紧性范围的说法要重查"


# -------------------------------------------------- 枚举器的精确性范围

def test_enumeration_is_not_exact_in_dim_two():
    """反面钉子：候选集（点 + 成对中点）在 d>=2 会高估。

    正三角形边长 1：真 Chebyshev 半径是外心半径 1/sqrt(3)=0.5774，
    但外心既不是任何一点也不是任何成对中点，枚举返回 0.8660，高估 1.5 倍。
    所以「枚举精确等于 q_m」只能限定在 d=1 或 n<=2。
    """
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2.0]])
    assert q_m(triangle, 1) == pytest.approx(0.8660254, abs=1e-6)
    true_radius = 1.0 / np.sqrt(3.0)
    assert true_radius == pytest.approx(0.5773503, abs=1e-6)
    assert q_m(triangle, 1) > true_radius + 0.2


def test_enumeration_is_exact_for_two_points_in_any_dimension():
    """n<=2 时中点最优，故枚举精确 —— {Bad,Good} 与 MNLI+CB 都是 n=2。"""
    rng = np.random.default_rng(17)
    for _ in range(200):
        dim = int(rng.integers(1, 6))
        points = rng.normal(size=(2, dim))
        expected = float(np.linalg.norm(points[0] - points[1]) / 2.0)
        assert q_m(points, 1) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------- Cor 3c

def test_per_example_routing_can_beat_the_single_offset_oracle():
    """反面钉子：Cor 3c 对**逐样本**路由不成立，必须写明「按任务路由」。

    一个二元任务，两簇样本方向相反：最佳单偏移只有 0.5，而两个中心配一个读
    sign(s) 的逐样本路由能到 1.0 —— 超过单偏移 oracle，于是任何
    `R_orc - R_Pm >= 正数` 形式的界都不可能成立。
    """
    contrast = np.concatenate([np.full(50, -3.0), np.full(50, 3.0)])
    labels = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])

    def accuracy(offset):
        return float(((contrast + offset > 0).astype(int) == labels).mean())

    grid = np.linspace(-10.0, 10.0, 4001)
    best_single = max(accuracy(np.full_like(contrast, b)) for b in grid)
    assert best_single == pytest.approx(0.5, abs=1e-9)

    centres = np.array([4.0, -4.0])
    per_example = accuracy(np.where(contrast < 0, centres[0], centres[1]))
    assert per_example == pytest.approx(1.0, abs=1e-9)
    assert per_example > best_single


def test_per_task_routing_cannot_beat_the_single_offset_oracle_in_that_example():
    """同一构造下，**按任务**路由（全任务共用一个中心）无法超过单偏移。"""
    contrast = np.concatenate([np.full(50, -3.0), np.full(50, 3.0)])
    labels = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])

    def accuracy(offset):
        return float(((contrast + offset > 0).astype(int) == labels).mean())

    for centre in np.linspace(-10.0, 10.0, 801):
        assert accuracy(np.full_like(contrast, centre)) <= 0.5 + 1e-9


def test_q_m_monotone_nonincreasing_including_m_two():
    rng = np.random.default_rng(19)
    for _ in range(200):
        dim = int(rng.integers(1, 4))
        n = int(rng.integers(2, 7))
        points = rng.normal(size=(n, dim))
        radii = [q_m(points, m) for m in (1, 2, 3, 4)]
        for earlier, later in zip(radii, radii[1:]):
            assert later <= earlier + 1e-12


# ------------------------------------------- d=1 的主张要对**代码里的数组**成立

def test_binary_gauge_fixed_offsets_are_genuinely_one_dimensional():
    """码本解算器跑在 R^K 向量上，不是标量上。

    所以「d=1」有可能只是关于商空间的说法而实现悄悄违反它。这里直接验证不会：
    K=2 时 gauge_fix 输出形如 (-t, t)，去掉平移后秩为 1，且
    ||g_a - g_b|| = |s_a - s_b| / sqrt(2)。常数因子不改变哪个码本最优。
    """
    from experiments.phase2j_offset_conflict.offsets import gauge_fix

    rng = np.random.default_rng(29)
    raw = rng.normal(size=(6, 2))
    fixed = np.stack([gauge_fix(row) for row in raw])
    assert np.allclose(fixed[:, 0], -fixed[:, 1])
    assert np.linalg.matrix_rank(fixed - fixed[0], tol=1e-9) == 1

    contrast = fixed[:, 1] - fixed[:, 0]
    for a, b in itertools.combinations(range(len(fixed)), 2):
        assert float(np.linalg.norm(fixed[a] - fixed[b])) == pytest.approx(
            abs(contrast[a] - contrast[b]) / np.sqrt(2.0), abs=1e-9)


def test_enumeration_equals_pigeonhole_on_binary_gauge_fixed_offsets():
    """真实域（K=2）上，枚举器与鸽巢闭式必须逐位一致。"""
    from experiments.phase2j_offset_conflict.offsets import gauge_fix

    rng = np.random.default_rng(31)
    for _ in range(200):
        n = int(rng.integers(2, 7))
        m = int(rng.integers(1, 4))
        fixed = np.stack([gauge_fix(row) for row in rng.normal(size=(n, 2))])
        assert q_m(fixed, m) == pytest.approx(pigeonhole_bound(fixed, m), abs=1e-9)
