"""协议 `notes/phase2p_streaming_scope_offset_protocol.md` §6 的反伪影检查。"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.phase2j_offset_conflict.offsets import balanced_accuracy, gauge_fix
from experiments.phase2p_sso.sso import StreamingScopeOffsets, chebyshev_centre


def _binary_logits(margin: float, n: int = 64, seed: int = 0):
    """一个二元任务：真最优偏移的标量对比约等于 −margin。"""
    rng = np.random.default_rng(seed)
    labels = np.array([0, 1] * (n // 2))
    logits = rng.normal(0.0, 0.3, size=(n, 2))
    logits[labels == 1, 1] += margin
    logits[labels == 0, 0] += margin
    logits[:, 1] += margin  # 系统性偏向类 1，需要负偏移纠正
    return logits, labels


# ---------------------------------------------------------------- Chebyshev

def test_chebyshev_centre_matches_the_closed_form_on_a_binary_scope():
    """§6：二元域上中心必须等于标量对比的 (min+max)/2。

    gauge-fix 后二元偏移是 (−c/2, +c/2)，c 是标量对比。所以中心的对比应是
    contrasts 的中点。这是 Prop 2 几何半的闭式，d=1 时 `build_codebook` 精确。
    """
    contrasts = np.array([-3.0, 0.5, 2.0, 1.25])
    X = np.stack([gauge_fix(np.array([-c / 2.0, c / 2.0])) for c in contrasts])
    centre = chebyshev_centre(X)
    got = centre[1] - centre[0]
    want = (contrasts.min() + contrasts.max()) / 2.0
    assert got == pytest.approx(want, abs=1e-9)


def test_chebyshev_centre_of_one_point_is_that_point():
    """P5 依赖这条：单任务域的中心就是该任务自己的（陈旧）最优偏移。"""
    x = gauge_fix(np.array([-1.5, 1.5]))
    assert chebyshev_centre(x[None, :]) == pytest.approx(x, abs=1e-12)


def test_chebyshev_centre_is_gauge_fixed():
    X = np.stack([gauge_fix(np.array([-c / 2, c / 2])) for c in (-2.0, 4.0)])
    assert chebyshev_centre(X).mean() == pytest.approx(0.0, abs=1e-12)


def test_chebyshev_centre_ignores_interior_points():
    """minimax 只由极值点决定：加一个内点不能移动中心。"""
    base = np.array([-4.0, 4.0])
    a = chebyshev_centre(np.stack([gauge_fix(np.array([-c / 2, c / 2])) for c in base]))
    with_inner = np.append(base, 0.7)
    b = chebyshev_centre(np.stack([gauge_fix(np.array([-c / 2, c / 2]))
                                   for c in with_inner]))
    assert a == pytest.approx(b, abs=1e-9)


# ---------------------------------------------------------------- 表的行为

def test_record_stores_one_entry_per_task_grouped_by_scope():
    sso = StreamingScopeOffsets()
    lg, lb = _binary_logits(1.0)
    sso.record("WiC", ("False", "True"), (3, 4), lg, lb)
    sso.record("QQP", ("False", "True"), (3, 4), lg, lb, )
    sso.record("IMDB", ("Bad", "Good"), (5, 6), lg, lb)
    assert set(sso.table) == {("False", "True"), ("Bad", "Good")}
    assert [e.task for e in sso.table[("False", "True")]] == ["WiC", "QQP"]
    assert sso.n_entries() == 3


def test_stored_floats_is_the_protocol_formula():
    """P6：`Σ_S |table[S]|·(K_S−1)`。"""
    sso = StreamingScopeOffsets()
    lg2, lb2 = _binary_logits(1.0)
    for name in ("WiC", "QQP", "BoolQA"):
        sso.record(name, ("False", "True"), (3, 4), lg2, lb2)
    rng = np.random.default_rng(1)
    lg3 = rng.normal(size=(30, 3))
    lb3 = np.array([0, 1, 2] * 10)
    sso.record("MNLI", ("contradiction", "entailment", "neutral"), (1, 2, 3), lg3, lb3)
    # 3 个任务 × (2−1) + 1 个任务 × (3−1) = 5
    assert sso.stored_floats() == 5


def test_pooling_two_tasks_into_one_record_is_refused():
    """SSO 一次只拟合一个任务；把多任务 logits 拼起来传进来是唯一的偷看路径。"""
    sso = StreamingScopeOffsets()
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(40, 4))          # 4 列，但标签只有 2 个
    targets = np.array([0, 1] * 20)
    with pytest.raises(ValueError, match="rehearsal"):
        sso.record("WiC", ("False", "True"), (3, 4), logits, targets)


def test_record_does_not_retain_logits_or_targets():
    """P6 的另一半：表里不能留下样本级张量。"""
    sso = StreamingScopeOffsets()
    lg, lb = _binary_logits(1.0)
    sso.record("WiC", ("False", "True"), (3, 4), lg, lb)
    entry = sso.table[("False", "True")][0]
    for value in vars(entry).values():
        if isinstance(value, np.ndarray):
            assert value.size <= len(entry.labels), (
                "table holds an array of size %d; only the (K-1)-dof optimum "
                "may be stored" % value.size)


def test_recorded_optimum_actually_helps_that_task():
    """健全性：记下来的偏移在它自己的数据上必须不劣于零偏移。"""
    sso = StreamingScopeOffsets()
    lg, lb = _binary_logits(1.5, seed=3)
    entry = sso.record("WiC", ("False", "True"), (3, 4), lg, lb)
    assert balanced_accuracy(lg, lb, entry.optimum) >= balanced_accuracy(lg, lb, None)


def test_recorded_optimum_is_gauge_fixed():
    sso = StreamingScopeOffsets()
    lg, lb = _binary_logits(1.0)
    entry = sso.record("WiC", ("False", "True"), (3, 4), lg, lb)
    assert entry.optimum.mean() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- 码本与路由

def _table_of(contrasts, scope=("False", "True")):
    """直接构表：绕开 fit_offset，让几何断言不受拟合噪声干扰。"""
    from experiments.phase2k_qoc.qoc import TaskOptima
    sso = StreamingScopeOffsets()
    for i, c in enumerate(contrasts):
        opt = gauge_fix(np.array([-c / 2.0, c / 2.0]))
        sso.table.setdefault(scope, []).append(
            TaskOptima(task="t%d" % i, labels=scope, pieces=(3, 4), optimum=opt))
    return sso


def test_m1_codebook_centre_is_the_chebyshev_centre():
    sso = _table_of([-3.0, 0.5, 2.0])
    book = sso.codebooks(1)[("False", "True")]
    assert book.centres.shape[0] == 1
    got = book.centres[0][1] - book.centres[0][0]
    assert got == pytest.approx((-3.0 + 2.0) / 2.0, abs=1e-9)


def test_m1_needs_no_router_state():
    """m=1 时不挂原型：中心只有一行，任何路由器都只能选它。"""
    sso = _table_of([-3.0, 0.5, 2.0])
    sso.prototype_state[("False", "True")] = {
        "t0": (np.ones(8), 10), "t1": (np.zeros(8), 10), "t2": (np.ones(8) * 2, 10)}
    book = sso.codebooks(1)[("False", "True")]
    assert book.prototypes is None


def test_m4_radius_is_zero_when_m_exceeds_the_table():
    """点数 ≤ m 时每点一个中心，q_m = 0（build_codebook 的退化分支）。"""
    sso = _table_of([-3.0, 2.0])
    book = sso.codebooks(4)[("False", "True")]
    assert book.centres.shape[0] == 2
    assert book.radius == pytest.approx(0.0, abs=1e-12)


def test_m2_radius_is_the_quantization_radius():
    """4 个点、m=2 的 minimax 半径：{−3,−2.5} 与 {2,2.5} 分两簇，半径 0.25。"""
    sso = _table_of([-3.0, -2.5, 2.0, 2.5])
    book = sso.codebooks(2)[("False", "True")]
    # gauge-fix 后点间欧氏距离 = |Δc|/√2，所以半径 = 0.25/√2
    assert book.radius == pytest.approx(0.25 / np.sqrt(2.0), abs=1e-9)


def test_prototypes_are_sample_count_weighted_and_match_pooling():
    """流式原型必须等于 2K 的「拼接后取均值」。"""
    sso = _table_of([-3.0, -2.5, 2.0, 2.5])
    a, b = np.array([1.0, 0.0]), np.array([0.0, 4.0])
    sso.prototype_state[("False", "True")] = {
        "t0": (a, 30), "t1": (b, 10), "t2": (a, 5), "t3": (b, 5)}
    book = sso.codebooks(2)[("False", "True")]
    for j in range(book.centres.shape[0]):
        members = [n for n, idx in book.assignment.items() if idx == j]
        num = sum(sso.prototype_state[("False", "True")][n][0]
                  * sso.prototype_state[("False", "True")][n][1] for n in members)
        den = sum(sso.prototype_state[("False", "True")][n][1] for n in members)
        assert book.prototypes[j] == pytest.approx(num / den, abs=1e-12)
        assert book.prototype_counts[j] == den


def test_router_floats_is_reported_separately_from_stored_floats():
    """P6 的公式只覆盖偏移表；路由器状态另计，两个数不合并。"""
    sso = _table_of([-3.0, 2.0])
    assert sso.router_floats() == 0
    sso.prototype_state[("False", "True")] = {
        "t0": (np.zeros(1024), 10), "t1": (np.zeros(1024), 10)}
    assert sso.stored_floats() == 2
    assert sso.router_floats() == 2048


def test_offset_for_maps_centres_back_through_a_reordered_label_tuple():
    """MNLI/CB 的标签顺序不同；域的规范序是 scope_key 的排序结果。"""
    sso = _table_of([-3.0, 2.0], scope=("contradiction", "entailment"))
    book = sso.codebooks(1)[("contradiction", "entailment")]
    fwd = book.offset_for(("contradiction", "entailment"), 0)
    rev = book.offset_for(("entailment", "contradiction"), 0)
    assert fwd == pytest.approx(rev[::-1], abs=1e-12)
