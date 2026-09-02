"""附录证明里两个**新**断言的数值检验。

写这两个测试的理由:附录里 Prop pigeonhole-tight(d=1、任意 m 取等)和
Lemma kappa-order(Ψ ≤ Φ)是这次新证的,不是笔记里已经验过的。笔记只对
tightness 做过 600 例随机验证而没有证明;现在有了贪心构造的证明,就该用
独立实现的暴力枚举再对一次,免得证明和实现同时错。

这些测试不碰 GPU,不读 run 记录,纯数学。
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest


# --------------------------------------------------------------- exact q_m in 1D

def q_m_exact_1d(xs, m):
    """穷举连续分块:d=1 下最优 m-codebook 必然把排序后的点切成 m 个连续块
    (附录 Prop q2-closed 的 Step 1 推广),每块取中程。

    这是独立于贪心构造的实现 —— 两者一致才算验过。
    """
    xs = np.sort(np.asarray(xs, dtype=float))
    n = len(xs)
    if m >= n:
        return 0.0
    best = np.inf
    # 在 n-1 个间隙里选 m-1 个切点
    for cuts in itertools.combinations(range(1, n), m - 1):
        bounds = (0,) + cuts + (n,)
        radius = 0.0
        for a, b in zip(bounds[:-1], bounds[1:]):
            block = xs[a:b]
            if len(block):
                radius = max(radius, (block[-1] - block[0]) / 2.0)
        best = min(best, radius)
    return float(best)


def sep_m_plus_1(xs, m):
    """(m+1)-点分离度的一半 —— 附录 eq:app-sep。暴力枚举所有 (m+1)-子集。"""
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    if n <= m:
        return 0.0
    best = 0.0
    for sub in itertools.combinations(range(n), m + 1):
        pts = xs[list(sub)]
        gap = min(abs(a - b) for a, b in itertools.combinations(pts, 2))
        best = max(best, gap)
    return float(best) / 2.0


def greedy_1d(xs, r):
    """附录 Prop pigeonhole-tight 证明里的贪心:返回覆盖半径 r 所需中心数。"""
    xs = np.sort(np.asarray(xs, dtype=float))
    centres = 0
    i = 0
    n = len(xs)
    while i < n:
        anchor = xs[i]
        centres += 1
        while i < n and xs[i] <= anchor + 2 * r + 1e-12:
            i += 1
    return centres


# ----------------------------------------------------- Prop pigeonhole-tight

def test_pigeonhole_is_tight_in_one_dimension():
    """q_m == sep_{m+1}/2 for every m, d=1 —— 附录新证的等式。"""
    rng = np.random.default_rng(20260831)
    for _ in range(600):
        n = int(rng.integers(2, 8))
        m = int(rng.integers(1, 5))
        scale = float(10 ** rng.uniform(-2, 2))
        xs = rng.normal(0.0, scale, size=n)
        assert q_m_exact_1d(xs, m) == pytest.approx(sep_m_plus_1(xs, m), abs=1e-9)


def test_greedy_construction_meets_the_bound():
    """证明的构造性一半:半径 sep/2 时贪心用的中心数 <= m。"""
    rng = np.random.default_rng(4242)
    for _ in range(400):
        n = int(rng.integers(2, 9))
        m = int(rng.integers(1, 5))
        xs = rng.normal(0.0, float(10 ** rng.uniform(-1, 1)), size=n)
        r = sep_m_plus_1(xs, m)
        assert greedy_1d(xs, r) <= m


def test_pigeonhole_can_be_strict_in_two_dimensions():
    """等边三角形、m=1:钉住 tightness 只在 d=1 成立。

    边长 1 的等边三角形:sep_2/2 = 1/2,真半径是外接圆半径 1/sqrt(3) = 0.5774。
    """
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    circumradius = 1.0 / np.sqrt(3.0)
    pairwise = min(np.linalg.norm(a - b)
                   for a, b in itertools.combinations(tri, 2))
    assert pairwise / 2.0 == pytest.approx(0.5)
    assert circumradius == pytest.approx(0.5773502692, abs=1e-9)
    assert pairwise / 2.0 < circumradius        # 界是严格的,不取等


def test_midpoint_enumerator_overestimates_in_two_dimensions():
    """附录 app:narrowed 第 1 条:点∪两两中点的枚举在 d>=2 高估半径 1.5x。"""
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    cands = [p for p in tri]
    cands += [(a + b) / 2.0 for a, b in itertools.combinations(tri, 2)]
    enumerated = min(max(np.linalg.norm(c - p) for p in tri) for c in cands)
    assert enumerated == pytest.approx(np.sqrt(3) / 2, abs=1e-9)   # 0.8660
    assert enumerated / (1.0 / np.sqrt(3.0)) == pytest.approx(1.5, abs=1e-9)


# ----------------------------------------------------------- Lemma kappa-order

def _restricted_argmax(logits, offset):
    return np.argmax(logits + offset[None, :], axis=1)


def test_accuracy_decay_never_exceeds_disagreement():
    """Ψ_g(t) <= Φ_g(t) —— 附录 Lemma kappa-order。

    随机二分类 logits,枚举 V0 上的两个单位方向(d=1 时只有 ±1),对每个 t
    比较准确率降幅与预测变动比例。
    """
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(20, 80))
        logits = rng.normal(size=(n, 2))
        labels = rng.integers(0, 2, size=n)

        # z_g:一维 gauge-fixed 最优偏移,网格搜出来
        grid = np.linspace(-6, 6, 481)
        accs = [np.mean(_restricted_argmax(logits, np.array([-t, t])) == labels)
                for t in grid]
        z = grid[int(np.argmax(accs))]
        base_off = np.array([-z, z])
        base_pred = _restricted_argmax(logits, base_off)
        base_acc = float(np.mean(base_pred == labels))

        for t in (0.1, 0.5, 1.0, 2.0):
            psi, phi = np.inf, np.inf
            for sign in (+1.0, -1.0):
                # V0 上的单位向量是 (-1,1)/sqrt(2) 的 ±,位移 t 即 z -> z + t*sign/sqrt(2)
                shift = t * sign / np.sqrt(2.0)
                off = np.array([-(z + shift), z + shift])
                pred = _restricted_argmax(logits, off)
                psi = min(psi, base_acc - float(np.mean(pred == labels)))
                phi = min(phi, float(np.mean(pred != base_pred)))
            assert psi <= phi + 1e-12


def test_disagreement_bound_is_not_vacuous():
    """Ψ <= Φ 不能是因为两边恒等 —— 构造严格小于的例子。

    二元 scope 里一个偏移方向只能把预测**单向**推(第一版测试写成「一个翻对
    一个翻错、方向相反」,那在 d=1 不可能,我自己的构造错了)。正确的构造是
    两条样本同向翻转、真标签相反:一条变错、一条变对,净损失 0 而变动 1.0。
    """
    logits = np.array([[0.0, 0.4],      # 预测 1,真标签 1 -> 翻转会变错
                       [0.0, 0.3]])     # 预测 1,真标签 0 -> 翻转会变对
    labels = np.array([1, 0])
    base_pred = _restricted_argmax(logits, np.zeros(2))
    base_acc = float(np.mean(base_pred == labels))
    assert base_acc == 0.5
    assert list(base_pred) == [1, 1]

    off = np.array([0.5, -0.5])
    pred = _restricted_argmax(logits, off)
    assert list(pred) == [0, 0]

    phi = float(np.mean(pred != base_pred))
    psi = base_acc - float(np.mean(pred == labels))
    assert phi == 1.0                    # 全部预测都变了
    assert psi == 0.0                    # 准确率一点没掉
    assert psi < phi                     # 所以两个常数确实不同
