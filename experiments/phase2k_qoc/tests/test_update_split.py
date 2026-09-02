"""`grow_update_split` 的不变量测试。

核心断言只有一条，但它承载整个收敛版重跑的可比性：**放大训练量不能动
risk/audit**。如果动了，收敛版和 cap=200 那版就是在不同评测集上打分，
`notes/method_qoc_v1.md` §M3 的逐任务对照表全部作废。

同时钉住：这个逻辑必须留在 Phase-2K，不能塞回 `order4_data.py` —— 后者的
SHA-256 被 Phase-2I 的冻结依赖清单校验，改它会让已封存结果的审计链断掉。
"""

from __future__ import annotations

import pytest

from experiments.phase2i_anchored_cvar import order4_data as od
from experiments.phase2k_qoc.run_qoc import grow_update_split


def _example(index: int, label: str) -> od.OfficialExample:
    return od.OfficialExample(
        task_name="MNLI",
        category="NLI",
        dataset="MNLI",
        subset="train",
        source_index=index,
        example_id="MNLI-train-%04d" % index,
        sentence="sentence %d" % index,
        label=label,
        prompt="prompt %d" % index,
    )


LABELS = ("neutral", "entailment", "contradiction")


def _pool(per_class: int = 30):
    return [
        _example(i * len(LABELS) + j, label)
        for i in range(per_class)
        for j, label in enumerate(LABELS)
    ]


def _baseline(pool, *, cap=5, risk=1, audit=1, seed=42):
    return od.make_train_partitions(
        pool, cap_per_class=cap, risk_per_class=risk, audit_per_class=audit, seed=seed
    )


def test_risk_and_audit_are_untouched():
    pool = _pool()
    base = _baseline(pool)
    grown = grow_update_split(base, pool, update_cap_per_class=12, seed=42)

    assert grown.risk == base.risk
    assert grown.audit == base.audit
    grown.assert_disjoint()


def test_update_grows_to_the_requested_size_per_class():
    pool = _pool()
    base = _baseline(pool)
    grown = grow_update_split(base, pool, update_cap_per_class=12, seed=42)

    counts: dict[str, int] = {}
    for example in grown.update:
        counts[example.label] = counts.get(example.label, 0) + 1
    assert counts == {label: 12 for label in LABELS}
    # cap=5、risk=1、audit=1 时原来每类只有 3 条可训练；这正是「3 个梯度步」的来源。
    assert len(base.update) == 3 * len(LABELS)
    assert len(grown.update) > len(base.update)


def test_update_never_reuses_a_risk_or_audit_example():
    pool = _pool()
    base = _baseline(pool)
    grown = grow_update_split(base, pool, update_cap_per_class=25, seed=42)

    reserved = {e.example_id for e in base.risk} | {e.example_id for e in base.audit}
    assert not reserved & {e.example_id for e in grown.update}


def test_is_deterministic_and_order_independent():
    pool = _pool()
    base = _baseline(pool)
    first = grow_update_split(base, pool, update_cap_per_class=12, seed=42)
    second = grow_update_split(base, list(reversed(pool)), update_cap_per_class=12, seed=42)
    assert first == second


def test_different_seeds_select_different_update_examples():
    """否则 seed 之间的训练数据完全相同，三 seed 就不是三次独立抽样。"""

    pool = _pool()
    base = _baseline(pool)
    a = grow_update_split(base, pool, update_cap_per_class=12, seed=1)
    b = grow_update_split(base, pool, update_cap_per_class=12, seed=2)
    assert {e.example_id for e in a.update} != {e.example_id for e in b.update}


def test_capping_larger_than_the_pool_is_clamped_not_an_error():
    pool = _pool(per_class=10)
    base = _baseline(pool)
    grown = grow_update_split(base, pool, update_cap_per_class=10_000, seed=42)
    # 每类 10 条，1 进 risk、1 进 audit，故最多 8 条可训练。
    counts: dict[str, int] = {}
    for example in grown.update:
        counts[example.label] = counts.get(example.label, 0) + 1
    assert counts == {label: 8 for label in LABELS}


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "12"])
def test_rejects_non_positive_or_non_integer(bad):
    pool = _pool()
    base = _baseline(pool)
    with pytest.raises(ValueError):
        grow_update_split(base, pool, update_cap_per_class=bad, seed=42)


def test_order4_data_is_not_modified_by_this_feature():
    """冻结依赖保护：`make_train_partitions` 不得出现 update_cap 参数。"""

    import inspect

    signature = inspect.signature(od.make_train_partitions)
    assert "update_cap_per_class" not in signature.parameters
    signature = inspect.signature(od.prepare_task_partitions)
    assert "update_cap_per_class" not in signature.parameters
