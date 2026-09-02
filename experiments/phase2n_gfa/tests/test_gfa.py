"""GFA 的测试。协议 §6 的每条防伪检查都必须有一条断言对应。

最重要的是**规范不变性**那条：它是 GFA 与 VLA 的全部区别，也是 Phase-2N 存在的
理由。如果它不成立，这个方法就是换了个损失函数的 VLA。
"""

from __future__ import annotations

import torch
import pytest

from experiments.phase2n_gfa.gfa import GaugeFixedAnchor


class _Tiny(torch.nn.Module):
    """最小 seq2seq 替身：logits 只由一个可训练向量决定，与输入无关。"""

    def __init__(self, vocab=8):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(vocab))
        self.vocab = vocab

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                **kw):
        n = input_ids.shape[0]
        logits = self.bias.unsqueeze(0).unsqueeze(0).expand(n, 1, self.vocab)
        return type("O", (), {"logits": logits})()


class _Tok:
    def __call__(self, texts, **kw):
        n = len(texts)
        return _Enc({"input_ids": torch.zeros((n, 3), dtype=torch.long),
                     "attention_mask": torch.ones((n, 3), dtype=torch.long)})


class _Enc(dict):
    def to(self, device):
        return self


class _Ex:
    """替身：字段名必须与 order4_data.OfficialExample 一致（`prompt`）。
    第一版写成 `source`，冒烟跑立刻 AttributeError——替身字段名和真实类型
    不一致时，单测全绿也挡不住接线错误。"""

    def __init__(self, s="x"):
        self.prompt = s


def _enc(n):
    return {"input_ids": torch.zeros((n, 3), dtype=torch.long),
            "attention_mask": torch.ones((n, 3), dtype=torch.long)}


# ---- 规范不变性：Phase-2N 的核心 ----

def test_gauge_fix_is_invariant_to_adding_a_constant():
    z = torch.tensor([1.0, -2.0, 3.5])
    for c in (0.0, 5.0, -100.0, 1e3):
        assert torch.allclose(GaugeFixedAnchor.gauge_fix(z),
                              GaugeFixedAnchor.gauge_fix(z + c), atol=1e-6)


def test_gauge_fix_output_is_zero_mean():
    z = torch.tensor([1.0, -2.0, 3.5, 7.0])
    assert GaugeFixedAnchor.gauge_fix(z).mean().abs().item() < 1e-6


def test_penalty_is_invariant_to_a_constant_shift_of_the_target():
    """协议 §6 冻结的那条检查：给**目标**的一个 scope 的所有列加常数，
    `L_gfa` 变化必须 < 1e-6。

    这正是 VLA 会付而 GFA 不付的那笔代价。目标向量已经是 float64（`gauge_fix`
    的输出），所以这里测到的是投影本身的不变性，不掺杂 logits 的存储精度。"""
    a = GaugeFixedAnchor(lam=1.0)
    a.observe("s", [0, 1, 2])
    model = _Tiny()
    with torch.no_grad():
        model.bias[:] = torch.tensor([0.3, -0.7, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    a.precompute(model, _Tok(), [_Ex()] * 4, "cpu")
    with torch.no_grad():
        model.bias[0] += 0.4  # 让惩罚非零，才测得出"不变"
    base = float(a.penalty(model, [0, 1, 2, 3], _enc(4), "cpu"))

    for c in (5.0, -100.0, 42.0, 1e3):
        for per_scope in a._targets.values():
            per_scope["s"] = GaugeFixedAnchor.gauge_fix(per_scope["s"] + c)
        shifted = float(a.penalty(model, [0, 1, 2, 3], _enc(4), "cpu"))
        assert abs(shifted - base) < 1e-6


def test_model_side_invariance_is_limited_only_by_float32_logit_storage():
    """模型侧加常数：不变性在数学上同样精确，但 logits 是 float32 存的，
    所以 `z + c` 本身就有 `c · 2^-24` 的表示误差，与投影无关。

    记录成一条按量级缩放的断言，而不是假装能过 1e-6——那会是拿阈值去迁就实现。
    """
    a = GaugeFixedAnchor(lam=1.0)
    a.observe("s", [0, 1, 2])
    model = _Tiny()
    with torch.no_grad():
        model.bias[:] = torch.tensor([0.3, -0.7, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    a.precompute(model, _Tok(), [_Ex()] * 4, "cpu")
    with torch.no_grad():
        model.bias[0] += 0.4
    base = float(a.penalty(model, [0, 1, 2, 3], _enc(4), "cpu"))

    c = 42.0
    with torch.no_grad():
        model.bias[0:3] += c
    shifted = float(a.penalty(model, [0, 1, 2, 3], _enc(4), "cpu"))
    # float32 eps ≈ 1.2e-7；平方误差里放大一档，留 10x 余量
    assert abs(shifted - base) < 10 * c * 1.2e-7


def test_penalty_does_change_when_the_within_scope_contrast_changes():
    """反面：动对比（不是动常数）必须被惩罚看到，否则损失是死的。"""
    a = GaugeFixedAnchor(lam=1.0)
    a.observe("s", [0, 1, 2])
    model = _Tiny()
    a.precompute(model, _Tok(), [_Ex()] * 2, "cpu")
    with torch.no_grad():
        model.bias[0] += 1.0  # 只动一列 -> 对比改变
    assert float(a.penalty(model, [0, 1], _enc(2), "cpu")) > 1e-3


# ---- 单任务 scope 排除 ----

def test_singleton_scope_contributes_exactly_zero():
    a = GaugeFixedAnchor(lam=1.0)
    a.observe("solo", [3])
    assert a.active_scopes() == []
    assert a.is_active() is False
    model = _Tiny()
    assert a.precompute(model, _Tok(), [_Ex()] * 2, "cpu") == 0
    with torch.no_grad():
        model.bias[3] += 10.0
    assert float(a.penalty(model, [0, 1], _enc(2), "cpu")) == 0.0


def test_singletons_do_not_dilute_the_mean_over_scopes():
    """加一个单任务 scope 不能改变惩罚值。"""
    model = _Tiny()
    with torch.no_grad():
        model.bias[:] = torch.arange(8, dtype=torch.float32) * 0.1

    a = GaugeFixedAnchor(lam=1.0)
    a.observe("pair", [0, 1, 2])
    a.precompute(model, _Tok(), [_Ex()] * 3, "cpu")
    with torch.no_grad():
        model.bias[1] += 0.5
    without = float(a.penalty(model, [0, 1, 2], _enc(3), "cpu"))

    with torch.no_grad():
        model.bias[1] -= 0.5
    b = GaugeFixedAnchor(lam=1.0)
    b.observe("pair", [0, 1, 2])
    b.observe("solo", [7])
    b.precompute(model, _Tok(), [_Ex()] * 3, "cpu")
    with torch.no_grad():
        model.bias[1] += 0.5
    with_solo = float(b.penalty(model, [0, 1, 2], _enc(3), "cpu"))
    # 容差是 float32 logits 的量级（约 1.2e-7），不是投影的精度：两次跑
    # 走的是同一条 float64 投影，差异只来自 logits 的存储。
    assert with_solo == pytest.approx(without, abs=1e-7)


# ---- 任务 1 / λ=0 惰性 ----

def test_anchor_is_zero_on_first_task_because_no_scope_is_registered():
    a = GaugeFixedAnchor(lam=8.0)
    assert a.active_scopes() == []
    assert a.is_active() is False
    model = _Tiny()
    assert a.precompute(model, _Tok(), [_Ex()] * 2, "cpu") == 0
    assert float(a.penalty(model, [0, 1], _enc(2), "cpu")) == 0.0


def test_lambda_zero_is_numerically_inert_even_with_history():
    a = GaugeFixedAnchor(lam=0.0)
    a.observe("s", [0, 1, 2])
    model = _Tiny()
    assert a.precompute(model, _Tok(), [_Ex()] * 3, "cpu") == 0
    with torch.no_grad():
        model.bias[0] += 5.0
    assert float(a.penalty(model, [0, 1, 2], _enc(3), "cpu")) == 0.0


def test_penalty_is_zero_when_model_matches_the_snapshot():
    a = GaugeFixedAnchor(lam=4.0)
    a.observe("s", [0, 1, 2])
    model = _Tiny()
    with torch.no_grad():
        model.bias[:] = torch.tensor([0.5, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    a.precompute(model, _Tok(), [_Ex()] * 4, "cpu")
    got = float(a.penalty(model, [0, 1, 2, 3], _enc(4), "cpu").detach())
    assert got == pytest.approx(0.0, abs=1e-10)


# ---- 数值正确性 ----

def test_penalty_matches_hand_computed_squared_error():
    a = GaugeFixedAnchor(lam=2.0)
    a.observe("s", [0, 1, 2])
    model = _Tiny()
    with torch.no_grad():
        model.bias[0:3] = torch.tensor([1.0, 0.0, -1.0])
    a.precompute(model, _Tok(), [_Ex()] * 1, "cpu")
    # 目标 g = [1,0,-1] - 0 = [1,0,-1]
    with torch.no_grad():
        model.bias[0:3] = torch.tensor([2.0, 0.0, -1.0])
    # 现在 z=[2,0,-1], mean=1/3, g=[5/3,-1/3,-4/3]
    # diff = [2/3,-1/3,-1/3], ‖diff‖² = 4/9+1/9+1/9 = 6/9
    expect = 2.0 * (6.0 / 9.0) / 1.0
    got = float(a.penalty(model, [0], _enc(1), "cpu"))
    assert got == pytest.approx(expect, abs=1e-6)
    assert got == pytest.approx(1.3333, abs=1e-4)


def test_gradient_flows_to_the_model():
    a = GaugeFixedAnchor(lam=1.0)
    a.observe("s", [0, 1, 2])
    model = _Tiny()
    a.precompute(model, _Tok(), [_Ex()] * 2, "cpu")
    with torch.no_grad():
        model.bias[0] += 1.0
    loss = a.penalty(model, [0, 1], _enc(2), "cpu")
    loss.backward()
    assert model.bias.grad is not None
    assert model.bias.grad.abs().sum().item() > 0


def test_targets_carry_no_gradient():
    a = GaugeFixedAnchor(lam=1.0)
    a.observe("s", [0, 1, 2])
    model = _Tiny()
    a.precompute(model, _Tok(), [_Ex()] * 2, "cpu")
    for per_scope in a._targets.values():
        for t in per_scope.values():
            assert t.requires_grad is False


# ---- 记账 / 契约 ----

def test_observe_dedups_and_preserves_order():
    a = GaugeFixedAnchor(lam=1.0)
    a.observe("s", [5, 2, 5, 9, 2])
    assert a.scopes["s"] == [5, 2, 9]


def test_reregistering_a_scope_with_different_columns_raises():
    a = GaugeFixedAnchor(lam=1.0)
    a.observe("Bad|Good", [10, 11])
    a.observe("Bad|Good", [10, 11])  # 共享同一 verbalizer，合法
    with pytest.raises(RuntimeError, match="one verbalizer"):
        a.observe("Bad|Good", [10, 12])


def test_same_column_set_in_a_different_order_is_accepted():
    """MNLI 与 CB 共享 verbalizer 但 label 顺序不同（真实数据里就是这样）。
    集合相同即同一个 verbalizer；比序列会在任务 2 就炸。"""
    a = GaugeFixedAnchor(lam=1.0)
    a.observe("c|e|n", [7163, 3, 27252])
    a.observe("c|e|n", [3, 27252, 7163])
    # 保留首次登记的顺序：目标缓存是按它算的，换序会让目标逐元素错位。
    assert a.scopes["c|e|n"] == [7163, 3, 27252]


def test_a_genuinely_different_column_set_is_still_refused():
    a = GaugeFixedAnchor(lam=1.0)
    a.observe("c|e|n", [7163, 3, 27252])
    with pytest.raises(RuntimeError, match="one verbalizer"):
        a.observe("c|e|n", [7163, 3, 999])


def test_rows_missing_from_the_cache_are_skipped_not_crashed():
    a = GaugeFixedAnchor(lam=1.0)
    a.observe("s", [0, 1, 2])
    model = _Tiny()
    a.precompute(model, _Tok(), [_Ex()] * 2, "cpu")
    with torch.no_grad():
        model.bias[0] += 1.0
    # 行号 99 不在缓存里
    assert float(a.penalty(model, [0, 99], _enc(2), "cpu")) > 0.0


def test_no_earlier_task_examples_are_read_during_precompute():
    """协议 §6：precompute 只吃当前任务的样本。"""
    import inspect
    src = inspect.getsource(GaugeFixedAnchor.precompute)
    for forbidden in ("risk", "audit", "test.json", "replay", "buffer"):
        assert forbidden not in src


def test_multiple_scopes_are_averaged_not_summed():
    model = _Tiny()
    a = GaugeFixedAnchor(lam=1.0)
    a.observe("s1", [0, 1])
    a.precompute(model, _Tok(), [_Ex()] * 1, "cpu")
    with torch.no_grad():
        model.bias[0] += 1.0
    one = float(a.penalty(model, [0], _enc(1), "cpu"))

    with torch.no_grad():
        model.bias[0] -= 1.0
    b = GaugeFixedAnchor(lam=1.0)
    b.observe("s1", [0, 1])
    b.observe("s2", [2, 3])
    b.precompute(model, _Tok(), [_Ex()] * 1, "cpu")
    with torch.no_grad():
        model.bias[0] += 1.0
        model.bias[2] += 1.0
    two = float(b.penalty(model, [0], _enc(1), "cpu"))
    # 两个 scope 各自贡献相同的量，取均值后应与单 scope 相同
    assert two == pytest.approx(one, abs=1e-6)
