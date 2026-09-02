"""QOC 单元测试。

重点验证方法与理论对齐的那几处，而不是「代码能跑」：

* 码本覆盖半径**按构造等于** offsets.py 里报的 q_m（同一候选集）；
* 半径对 m 单调不增；
* oracle 路由的准确率 ≥ 单偏移（m=1）参照，即码本确实是容量增益；
* gauge-fix 不变性：给某个任务所有标签 logit 加常数不改变任何结论；
* 域划分按字符串集，首片碰撞由坐标承担 —— 两者不混。
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.phase2j_offset_conflict.offsets import (
    _worst_case_quantization_radius,
    balanced_accuracy,
    gauge_fix,
)
from experiments.phase2k_qoc.qoc import (
    ScopeCodebook,
    TaskOptima,
    build_codebook,
    fit_scope_codebooks,
    fit_task_optimum,
    score_with_codebook,
    scope_key,
)


def _optima(name, labels, values):
    return TaskOptima(task=name, labels=tuple(labels),
                      pieces=tuple(range(len(labels))),
                      optimum=gauge_fix(np.array(values, dtype=float)))


def test_codebook_radius_equals_probe_quantization_radius():
    """码本半径必须与 Phase-2J 报的 q_m 完全一致，否则方法和理论不是一个量。

    注意本测试断言的是**两者一致**，不是两者都等于真正的 q_m：二者共用
    「点∪成对中点」候选集，该集合只在 d = 1 或 n ≤ 2 时含最优中心
    （`notes/theory_prop3_m2_proof.md` §6）。这里 X 是 (4, 3)，即 d ≥ 2，
    所以两边可能同为上界——一致性才是这个测试要钉的东西。
    """
    rng = np.random.default_rng(0)
    for _ in range(20):
        X = rng.normal(size=(4, 3))
        X = np.stack([gauge_fix(row) for row in X])
        for m in (1, 2, 3):
            _, radius, _ = build_codebook(X, m)
            expected = _worst_case_quantization_radius(X, m)
            assert radius == pytest.approx(expected, abs=1e-9)


def test_radius_monotone_in_m():
    rng = np.random.default_rng(1)
    X = np.stack([gauge_fix(row) for row in rng.normal(size=(4, 4))])
    radii = [build_codebook(X, m)[1] for m in (1, 2, 3, 4)]
    for a, b in zip(radii, radii[1:]):
        assert b <= a + 1e-12
    assert radii[-1] == pytest.approx(0.0)


def test_m_at_least_n_gives_zero_radius_and_identity_assignment():
    X = np.stack([gauge_fix(np.array(v, dtype=float))
                  for v in ([1.0, -1.0], [-2.0, 2.0])])
    centres, radius, assign = build_codebook(X, 4)
    assert radius == pytest.approx(0.0)
    assert sorted(assign) == [0, 1]
    assert centres.shape[0] == 2


def test_centres_are_gauge_fixed():
    """中点的均值不一定为 0，必须显式 gauge-fix，否则偏移里混进无效自由度。"""
    X = np.stack([gauge_fix(np.array(v, dtype=float))
                  for v in ([3.0, 0.0, -3.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 0.0])])
    centres, _, _ = build_codebook(X, 2)
    for centre in centres:
        assert centre.mean() == pytest.approx(0.0, abs=1e-12)


def test_scope_key_is_order_insensitive():
    assert scope_key(("entailment", "contradiction", "neutral")) == \
        scope_key(("neutral", "entailment", "contradiction"))


def test_scope_groups_split_by_label_set_not_by_task():
    optima = [
        _optima("MNLI", ("neutral", "entailment", "contradiction"), [1.0, 0.0, -1.0]),
        _optima("CB", ("entailment", "contradiction", "neutral"), [0.0, -1.0, 1.0]),
        _optima("RTE", ("contradiction", "entailment"), [0.5, -0.5]),
        _optima("WiC", ("True", "False"), [0.2, -0.2]),
    ]
    books = fit_scope_codebooks(optima, m=2)
    scopes = {scope: sorted(book.members) for scope, book in books.items()}
    assert scopes[("contradiction", "entailment", "neutral")] == ["CB", "MNLI"]
    assert scopes[("contradiction", "entailment")] == ["RTE"]
    assert scopes[("False", "True")] == ["WiC"]


def test_aligned_reorders_to_canonical_scope_order():
    """MNLI 与 CB 标签顺序不同但域相同，对齐后同一坐标必须指同一标签。"""
    mnli = _optima("MNLI", ("neutral", "entailment", "contradiction"), [2.0, 0.0, -2.0])
    cb = _optima("CB", ("entailment", "contradiction", "neutral"), [0.0, -2.0, 2.0])
    assert mnli.scope == cb.scope
    np.testing.assert_allclose(mnli.aligned(), cb.aligned(), atol=1e-12)


def test_singleton_scope_has_zero_radius():
    """Prop 1 的可证伪预测：域内只有一个任务时不存在冲突。"""
    books = fit_scope_codebooks(
        [_optima("DBpedia", ("a", "b", "c"), [1.0, 0.0, -1.0])], m=4
    )
    book = books[("a", "b", "c")]
    assert book.radius == pytest.approx(0.0)


def _two_task_conflict():
    """构造一个真实冲突：两个任务在同一二元域上要求相反方向的偏移。"""
    rng = np.random.default_rng(7)
    n = 200
    # 任务 A：logit 差偏负，需要正偏移才能平衡；任务 B 反之。
    la = np.stack([rng.normal(0.0, 1.0, n), rng.normal(2.0, 1.0, n)], axis=1)
    ya = (rng.random(n) < 0.5).astype(int)
    la[ya == 0, 0] += 1.0
    lb = np.stack([rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n)], axis=1)
    yb = (rng.random(n) < 0.5).astype(int)
    lb[yb == 1, 1] += 1.0
    return (la, ya), (lb, yb)


def test_oracle_codebook_beats_single_offset_under_real_conflict():
    (la, ya), (lb, yb) = _two_task_conflict()
    labels = ("False", "True")
    oa = fit_task_optimum("A", labels, (0, 1), la, ya)
    ob = fit_task_optimum("B", labels, (0, 1), lb, yb)

    single = fit_scope_codebooks([oa, ob], m=1)[scope_key(labels)]
    双 = fit_scope_codebooks([oa, ob], m=2)[scope_key(labels)]
    assert 双.radius <= single.radius + 1e-12

    for task, logits, targets in (("A", la, ya), ("B", lb, yb)):
        one = score_with_codebook(single, task, labels, logits, targets,
                                  router="oracle")["accuracy"]
        two = score_with_codebook(双, task, labels, logits, targets,
                                  router="oracle")["accuracy"]
        assert two >= one - 1e-12


def test_score_is_invariant_to_adding_a_constant_to_all_logits():
    (la, ya), (lb, yb) = _two_task_conflict()
    labels = ("False", "True")
    book = fit_scope_codebooks([
        fit_task_optimum("A", labels, (0, 1), la, ya),
        fit_task_optimum("B", labels, (0, 1), lb, yb),
    ], m=2)[scope_key(labels)]
    base = score_with_codebook(book, "A", labels, la, ya, router="confidence")
    shifted = score_with_codebook(book, "A", labels, la + 3.7, ya, router="confidence")
    assert base["accuracy"] == pytest.approx(shifted["accuracy"])
    assert base["route_histogram"] == shifted["route_histogram"]


def test_confidence_router_uses_no_task_information():
    """置信度路由必须只依赖 logits：states=None 也要给出相同结果。"""
    (la, ya), (lb, yb) = _two_task_conflict()
    labels = ("False", "True")
    book = fit_scope_codebooks([
        fit_task_optimum("A", labels, (0, 1), la, ya),
        fit_task_optimum("B", labels, (0, 1), lb, yb),
    ], m=2)[scope_key(labels)]
    a = score_with_codebook(book, "A", labels, la, ya, router="confidence")
    b = score_with_codebook(book, "A", labels, la, ya, router="confidence",
                            states=np.zeros((la.shape[0], 8)))
    assert a["route_histogram"] == b["route_histogram"]


def test_prototype_router_falls_back_when_no_prototypes():
    labels = ("False", "True")
    (la, ya), _ = _two_task_conflict()
    book = ScopeCodebook(scope=scope_key(labels), members=["A"],
                         centres=np.stack([gauge_fix(np.array([0.5, -0.5])),
                                           gauge_fix(np.array([-0.5, 0.5]))]),
                         radius=1.0, assignment={"A": 0})
    out = score_with_codebook(book, "A", labels, la, ya, router="prototype")
    assert out["route_histogram"][0] == la.shape[0]


def test_router_histogram_sums_to_n():
    (la, ya), (lb, yb) = _two_task_conflict()
    labels = ("False", "True")
    book = fit_scope_codebooks([
        fit_task_optimum("A", labels, (0, 1), la, ya),
        fit_task_optimum("B", labels, (0, 1), lb, yb),
    ], m=2)[scope_key(labels)]
    for router in ("single", "confidence", "oracle"):
        out = score_with_codebook(book, "B", labels, lb, yb, router=router)
        assert sum(out["route_histogram"]) == lb.shape[0]


def test_offset_for_maps_back_to_task_label_order():
    scope = ("contradiction", "entailment", "neutral")
    centres = np.stack([gauge_fix(np.array([1.0, 0.0, -1.0]))])
    book = ScopeCodebook(scope=scope, members=["MNLI"], centres=centres,
                         radius=0.0, assignment={"MNLI": 0})
    # MNLI 的实际标签顺序不同，取回的偏移必须跟着重排。
    got = book.offset_for(("neutral", "entailment", "contradiction"), 0)
    np.testing.assert_allclose(got, gauge_fix(np.array([-1.0, 0.0, 1.0])), atol=1e-12)


def test_balanced_accuracy_used_consistently():
    """score 的目标函数必须是平衡准确率，类不平衡时才不会被多数类带偏。"""
    logits = np.array([[1.0, 0.0]] * 90 + [[1.0, 0.0]] * 10)
    targets = np.array([0] * 90 + [1] * 10)
    assert balanced_accuracy(logits, targets) == pytest.approx(0.5)


def test_batch_margin_router_is_constant_over_the_batch():
    (la, ya), (lb, yb) = _two_task_conflict()
    labels = ("False", "True")
    book = fit_scope_codebooks([
        fit_task_optimum("A", labels, (0, 1), la, ya),
        fit_task_optimum("B", labels, (0, 1), lb, yb),
    ], m=2)[scope_key(labels)]
    out = score_with_codebook(book, "A", labels, la, ya, router="batch_margin")
    # 整批同一个中心：直方图只有一个非零槽。
    assert sum(1 for v in out["route_histogram"] if v) == 1


def test_confidence_router_degeneracy_is_reproducible():
    """记录退化：极端偏移会赢得逐样本最大间隔。

    构造一个中心，它把所有样本粗暴推向类 1；它的逐样本间隔必然更大，
    所以 `confidence` 会选它 —— 即使它的准确率明显更差。
    """
    logits = np.array([[0.2, 0.0]] * 50 + [[0.0, 0.2]] * 50)
    targets = np.array([0] * 50 + [1] * 50)
    labels = ("False", "True")
    # 中心 0 温和（近似不动），中心 1 极端（大幅偏向类 1）。
    centres = np.stack([gauge_fix(np.array([0.0, 0.0])),
                        gauge_fix(np.array([-8.0, 8.0]))])
    book = ScopeCodebook(scope=scope_key(labels), members=["A"], centres=centres,
                         radius=8.0, assignment={"A": 0})
    out = score_with_codebook(book, "A", labels, logits, targets, router="confidence")
    assert out["route_histogram"][1] == 100      # 全部选了极端中心
    assert out["accuracy"] == pytest.approx(0.5)  # 而它把一半样本判错
    batch = score_with_codebook(book, "A", labels, logits, targets,
                                router="batch_margin")
    assert batch["route_histogram"][1] == 100    # 本例中平均间隔也偏向它
