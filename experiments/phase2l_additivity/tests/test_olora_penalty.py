"""正交惩罚的行为钉子，含协议 §6 的反伪影检查。"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.phase2l_additivity.olora_penalty import (
    OrthogonalHistory,
    lora_a_matrices,
)


class FakePeftModel(nn.Module):
    """最小的假 peft 模型：两层，各有 lora_A / lora_B，外加一个冻结的基座权重。

    命名刻意模仿 peft 的 `...lora_A.default.weight`，这样 `lora_a_matrices`
    的字符串匹配是在真实形状的名字上被测试的。
    """

    def __init__(self, *, r: int = 4, d: int = 8):
        super().__init__()
        self.base_layer_weight = nn.Parameter(torch.randn(d, d), requires_grad=False)
        self.q_lora_A_default_weight = nn.Parameter(torch.randn(r, d))
        self.q_lora_B_default_weight = nn.Parameter(torch.randn(d, r))
        self.v_lora_A_default_weight = nn.Parameter(torch.randn(r, d))
        self.v_lora_B_default_weight = nn.Parameter(torch.randn(d, r))

    def named_parameters(self, *a, **k):
        # 把下划线名字翻译成 peft 的点号名字，测真实匹配路径。
        mapping = {
            "base_layer_weight": "base_model.encoder.q.base_layer.weight",
            "q_lora_A_default_weight": "base_model.encoder.q.lora_A.default.weight",
            "q_lora_B_default_weight": "base_model.encoder.q.lora_B.default.weight",
            "v_lora_A_default_weight": "base_model.encoder.v.lora_A.default.weight",
            "v_lora_B_default_weight": "base_model.encoder.v.lora_B.default.weight",
        }
        for attr, name in mapping.items():
            yield name, getattr(self, attr)


def test_only_trainable_lora_a_is_collected():
    model = FakePeftModel()
    got = lora_a_matrices(model)
    assert set(got) == {
        "base_model.encoder.q.lora_A.default.weight",
        "base_model.encoder.v.lora_A.default.weight",
    }


def test_penalty_is_zero_on_first_task_by_construction():
    """协议 §6：任务 1 没有历史，惩罚必须恰好 0。"""
    hist = OrthogonalHistory(lam=1.0)
    assert hist.n_history() == 0
    assert float(hist.penalty(FakePeftModel())) == 0.0


def test_lambda_zero_is_numerically_inert_even_with_history():
    """协议 §6 第二条：λ=0 且代码路径激活，必须与 cl-method none 数值相同。"""
    model = FakePeftModel()
    hist = OrthogonalHistory(lam=0.0)
    hist.snapshot(model)
    hist.snapshot(model)
    assert hist.n_history() == 2
    assert float(hist.penalty(model)) == 0.0


def test_penalty_vanishes_exactly_for_orthogonal_rows():
    """A 与历史行空间正交时惩罚为 0——这是惩罚要奖励的构型。"""
    model = FakePeftModel(r=2, d=4)
    with torch.no_grad():
        # 历史占据前两个坐标，当前占据后两个 => A A_tᵀ = 0
        model.q_lora_A_default_weight.copy_(torch.tensor([[1., 0., 0., 0.],
                                                          [0., 1., 0., 0.]]))
        model.v_lora_A_default_weight.copy_(torch.tensor([[1., 0., 0., 0.],
                                                          [0., 1., 0., 0.]]))
    hist = OrthogonalHistory(lam=1.0)
    hist.snapshot(model)
    with torch.no_grad():
        for p in (model.q_lora_A_default_weight, model.v_lora_A_default_weight):
            p.copy_(torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 1.]]))
    assert float(hist.penalty(model)) == pytest.approx(0.0, abs=1e-12)


def test_penalty_matches_hand_computed_frobenius_value():
    """把惩罚钉在一个手算值上，防止公式被悄悄改形。"""
    model = FakePeftModel(r=1, d=2)
    with torch.no_grad():
        model.q_lora_A_default_weight.copy_(torch.tensor([[1., 0.]]))
        model.v_lora_A_default_weight.copy_(torch.tensor([[1., 0.]]))
    hist = OrthogonalHistory(lam=1.0)
    hist.snapshot(model)                      # A_t = [[1, 0]]
    with torch.no_grad():
        for p in (model.q_lora_A_default_weight, model.v_lora_A_default_weight):
            p.copy_(torch.tensor([[3., 4.]]))  # A A_tᵀ = [[3]] => 9
    assert float(hist.penalty(model)) == pytest.approx(18.0, abs=1e-9)  # 两层


def test_penalty_scales_linearly_in_lambda():
    model = FakePeftModel()
    a = OrthogonalHistory(lam=1.0); a.snapshot(model)
    b = OrthogonalHistory(lam=0.5); b.snapshot(model)
    assert float(b.penalty(model)) == pytest.approx(0.5 * float(a.penalty(model)),
                                                    rel=1e-6)


def test_penalty_accumulates_over_tasks():
    """两个历史快照的惩罚 = 各自之和，确认是 Σ_t 而不是只看最近一个。"""
    model = FakePeftModel(r=1, d=2)
    with torch.no_grad():
        model.q_lora_A_default_weight.copy_(torch.tensor([[1., 0.]]))
        model.v_lora_A_default_weight.copy_(torch.tensor([[1., 0.]]))
    hist = OrthogonalHistory(lam=1.0)
    hist.snapshot(model)
    with torch.no_grad():
        for p in (model.q_lora_A_default_weight, model.v_lora_A_default_weight):
            p.copy_(torch.tensor([[0., 1.]]))
    hist.snapshot(model)                       # 第二个历史与第一个正交
    with torch.no_grad():
        for p in (model.q_lora_A_default_weight, model.v_lora_A_default_weight):
            p.copy_(torch.tensor([[3., 4.]]))
    # 对 [[1,0]] 贡献 9，对 [[0,1]] 贡献 16，两层 => 50
    assert float(hist.penalty(model)) == pytest.approx(50.0, abs=1e-9)


def test_history_is_detached_and_does_not_receive_gradient():
    model = FakePeftModel()
    hist = OrthogonalHistory(lam=1.0)
    hist.snapshot(model)
    for frozen in hist.history:
        for tensor in frozen.values():
            assert not tensor.requires_grad
    loss = hist.penalty(model)
    loss.backward()
    # 当前 A 拿到梯度，历史快照不在图里
    assert model.q_lora_A_default_weight.grad is not None
    assert model.q_lora_B_default_weight.grad is None


def test_snapshot_does_not_change_trainable_parameter_count():
    """协议 §6：惩罚不引入参数。"""
    model = FakePeftModel()
    before = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    hist = OrthogonalHistory(lam=1.0)
    hist.snapshot(model); hist.snapshot(model)
    after = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    assert before == after


def test_negative_lambda_is_rejected():
    with pytest.raises(ValueError):
        OrthogonalHistory(lam=-0.1)


def test_missing_layer_in_history_raises_rather_than_skipping():
    """层集合变了是接线错误，必须炸而不是静默少算一项。"""
    model = FakePeftModel()
    hist = OrthogonalHistory(lam=1.0)
    hist.snapshot(model)
    hist.history[0].pop("base_model.encoder.q.lora_A.default.weight")
    with pytest.raises(RuntimeError, match="layer set changed"):
        hist.penalty(model)


def test_empty_model_raises():
    class Empty(nn.Module):
        def named_parameters(self, *a, **k):
            return iter(())
    with pytest.raises(RuntimeError, match="no trainable lora_A"):
        OrthogonalHistory(lam=1.0).penalty(Empty())
