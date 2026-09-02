"""VLA 的行为钉子，含协议 §6 的反伪影清单。"""

from __future__ import annotations

import argparse
import inspect
import types

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from experiments.phase2m_vla.train_with_anchor import train_one_task_with_anchor
from experiments.phase2m_vla.vla import VerbalizerAnchor


class FakeConfig:
    decoder_start_token_id = 0


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        n = len(texts)
        return {"input_ids": torch.ones((n, 3), dtype=torch.long),
                "attention_mask": torch.ones((n, 3), dtype=torch.long)}


class FakeModel(nn.Module):
    """输出 = 一个可训练的偏置向量（与输入无关），便于精确验算 KL。"""

    def __init__(self, vocab=6):
        super().__init__()
        self.config = FakeConfig()
        self.bias = nn.Parameter(torch.zeros(vocab))
        self.vocab = vocab

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                decoder_input_ids=None):
        n = input_ids.shape[0]
        logits = self.bias.unsqueeze(0).unsqueeze(0).expand(n, 1, self.vocab)
        out = types.SimpleNamespace(logits=logits)
        if labels is not None:
            out.loss = (self.bias ** 2).sum()
        return out


def _args(**over):
    base = dict(seed=1, epochs=1, batch_size=2, grad_accum=2,
                max_source=8, max_target=2)
    base.update(over)
    return argparse.Namespace(**base)


def _examples(n=8):
    return [types.SimpleNamespace(prompt="p%d" % i, label="l%d" % i)
            for i in range(n)]


def _enc(n):
    return {"input_ids": torch.ones(n, 3, dtype=torch.long)}


# ---------------------------------------------------------------- 协议 §6

def test_anchor_is_zero_on_first_task_because_V0_is_empty():
    a = VerbalizerAnchor(lam=1.0)
    assert a.active_columns() == []
    assert a.is_active() is False
    model = FakeModel()
    rows = a.precompute(model, FakeTokenizer(), _examples(), "cpu",
                        max_source=8, batch_size=2)
    assert rows == 0
    assert float(a.penalty(model, [0, 1], _enc(2), "cpu")) == 0.0


def test_lambda_zero_is_numerically_inert_even_with_history():
    a = VerbalizerAnchor(lam=0.0)
    a.observe([3, 4])
    assert a.is_active() is False
    model = FakeModel()
    assert a.precompute(model, FakeTokenizer(), _examples(), "cpu",
                        max_source=8, batch_size=2) == 0


def test_only_previous_verbalizers_are_anchored_not_the_current_one():
    """当前任务自己的 verbalizer 必须不在锚定列里，否则直接对抗 L_task。"""
    a = VerbalizerAnchor(lam=1.0)
    a.observe([3, 4])
    before = a.active_columns()
    a.observe([4, 5])
    assert before == [3, 4]
    assert a.active_columns() == [3, 4, 5]


def test_observe_deduplicates_shared_verbalizer_pieces():
    a = VerbalizerAnchor(lam=1.0)
    for pieces in ([3, 4], [3, 4], [4, 3]):
        a.observe(pieces)
    assert a.active_columns() == [3, 4]


def test_targets_come_from_a_frozen_snapshot_no_gradient_flows():
    a = VerbalizerAnchor(lam=1.0)
    a.observe([1, 2])
    model = FakeModel()
    a.precompute(model, FakeTokenizer(), _examples(4), "cpu",
                 max_source=8, batch_size=2)
    assert a.cached_rows() == 4
    assert not a._targets.requires_grad
    assert model.bias.grad is None


def test_penalty_is_zero_when_model_matches_the_snapshot():
    """锚点是自身快照时 KL=0 —— 没有漂移就没有惩罚。"""
    a = VerbalizerAnchor(lam=1.0)
    a.observe([1, 2, 3])
    model = FakeModel()
    with torch.no_grad():
        model.bias.copy_(torch.tensor([0.1, 0.7, -0.3, 0.5, 0.0, 0.2]))
    a.precompute(model, FakeTokenizer(), _examples(4), "cpu",
                 max_source=8, batch_size=2)
    got = float(a.penalty(model, [0, 1, 2, 3], _enc(4), "cpu"))
    assert got == pytest.approx(0.0, abs=1e-9)


def test_penalty_grows_as_the_model_drifts_from_the_snapshot():
    a = VerbalizerAnchor(lam=1.0)
    a.observe([1, 2, 3])
    model = FakeModel()
    a.precompute(model, FakeTokenizer(), _examples(4), "cpu",
                 max_source=8, batch_size=2)
    seen = []
    for drift in (0.0, 0.5, 2.0):
        with torch.no_grad():
            model.bias.zero_()
            model.bias[1] = drift
        seen.append(float(a.penalty(model, [0, 1, 2, 3], _enc(4), "cpu")))
    assert seen[0] == pytest.approx(0.0, abs=1e-9)
    assert seen[1] < seen[2]


def test_penalty_matches_hand_computed_kl():
    """把 KL 钉在手算值上，防止公式（含 tau^2 因子）被悄悄改形。"""
    a = VerbalizerAnchor(lam=1.0, temperature=2.0)
    a.observe([0, 1])
    model = FakeModel(vocab=2)
    a.precompute(model, FakeTokenizer(), _examples(1), "cpu",
                 max_source=8, batch_size=1)
    with torch.no_grad():
        model.bias.copy_(torch.tensor([1.0, -1.0]))
    got = float(a.penalty(model, [0], _enc(1), "cpu"))
    # 参考值用 KL 的定义直接算，不再借 F.kl_div —— 后者的 batchmean 是按
    # **第 0 维**求平均，喂 1-D 张量时会把词表维当成 batch 维，参考值本身就错
    # 一个因子。这里 teacher=[.5,.5]，student=softmax([.5,-.5])。
    teacher = torch.full((2,), 0.5)
    student = F.softmax(torch.tensor([1.0, -1.0]) / 2.0, dim=-1)
    want = float((teacher * (teacher.log() - student.log())).sum() * 4.0)
    assert got == pytest.approx(want, rel=1e-6)
    assert want == pytest.approx(0.4805, abs=1e-4)


def test_negative_lambda_and_bad_temperature_are_rejected():
    with pytest.raises(ValueError):
        VerbalizerAnchor(lam=-0.1)
    with pytest.raises(ValueError):
        VerbalizerAnchor(lam=1.0, temperature=0.0)


def test_anchor_adds_no_parameters():
    model = FakeModel()
    before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    a = VerbalizerAnchor(lam=1.0)
    a.observe([1, 2])
    a.precompute(model, FakeTokenizer(), _examples(4), "cpu",
                 max_source=8, batch_size=2)
    after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert before == after


def test_row_alignment_is_preserved_by_precompute():
    """取错行就是锚错样本。用逐样本不同的 teacher 验证行确实不同。"""
    a = VerbalizerAnchor(lam=1.0)
    a.observe([0, 1])

    class Varying(FakeModel):
        def forward(self, input_ids=None, **kw):
            n = input_ids.shape[0]
            base = torch.arange(n, dtype=torch.float32).unsqueeze(1)
            logits = (base + torch.tensor([[0.0, 1.0]])).unsqueeze(1)
            out = types.SimpleNamespace(logits=logits)
            out.loss = torch.zeros(())
            return out

    model = Varying(vocab=2)
    a.precompute(model, FakeTokenizer(), _examples(2), "cpu",
                 max_source=8, batch_size=2)
    assert a.cached_rows() == 2
    assert not torch.allclose(a._targets[0], a._targets[1])


# ------------------------------------------------------- 训练循环等价性

def _run(anchor, *, args=None, n=8):
    torch.manual_seed(0)
    model = FakeModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    out = train_one_task_with_anchor(model, FakeTokenizer(), _examples(n), "cpu",
                                    args=args or _args(), optimizer=opt,
                                    anchor=anchor)
    return out, model.bias.detach().clone()


def test_anchor_none_and_lambda_zero_give_identical_parameters():
    """协议 §6 第一条：λ_a=0 且 VLA 路径激活，必须与 none 逐位相同。"""
    zero = VerbalizerAnchor(lam=0.0)
    zero.observe([1, 2])
    a, pa = _run(None)
    b, pb = _run(zero)
    assert torch.equal(pa, pb)
    assert a["task_losses"] == b["task_losses"]
    assert all(x == 0.0 for x in b["anchor_losses"])


def test_active_anchor_changes_parameters():
    """锚定必须真的进梯度，否则整个 Phase-2M 是空转。"""
    torch.manual_seed(0)
    model = FakeModel()
    anchor = VerbalizerAnchor(lam=5.0)
    anchor.observe([1, 2, 3])
    with torch.no_grad():
        model.bias.copy_(torch.tensor([0.5, -0.5, 0.5, -0.5, 0.5, -0.5]))
    anchor.precompute(model, FakeTokenizer(), _examples(8), "cpu",
                      max_source=8, batch_size=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    out = train_one_task_with_anchor(model, FakeTokenizer(), _examples(8), "cpu",
                                    args=_args(), optimizer=opt, anchor=anchor)
    _, plain = _run(None)
    assert not torch.equal(model.bias.detach(), plain)
    assert len(out["anchor_losses"]) == len(out["task_losses"])


def test_step_count_matches_effective_batch():
    out, _ = _run(None, args=_args(batch_size=2, grad_accum=2), n=8)
    assert len(out["task_losses"]) == 2
    out, _ = _run(None, args=_args(batch_size=2, grad_accum=2, epochs=3), n=8)
    assert len(out["task_losses"]) == 6


def test_no_earlier_task_examples_are_read_during_training():
    """数据无关：签名里没有旧数据入口，precompute 里也不碰留出划分。"""
    sig = inspect.signature(train_one_task_with_anchor)
    assert set(sig.parameters) == {"model", "tokenizer", "examples", "device",
                                   "args", "optimizer", "anchor"}
    body = inspect.getsource(VerbalizerAnchor.precompute)
    for forbidden in ("risk", "audit", "test.json", "replay", "buffer"):
        assert forbidden not in body, forbidden


# ------------------------------------------------------------ 接线检查

def test_run_qoc_none_arm_untouched_by_vla():
    from experiments.phase2k_qoc import run_qoc
    source = inspect.getsource(run_qoc.main)
    idx = source.index('if args.cl_method == "none"')
    none_branch = source[idx: source.index("elif", idx)]
    assert "train_one_task(" in none_branch
    assert "train_with_anchor" not in none_branch
    assert "anchor" not in none_branch


def test_run_qoc_observes_verbalizer_after_training():
    """训完才登记，于是任务 1 的 V_0 为空、锚定恒 0。"""
    from experiments.phase2k_qoc import run_qoc
    source = inspect.getsource(run_qoc.main)
    assert source.index("anchor.observe(") > source.index("train_with_anchor(")


def test_run_qoc_precomputes_before_training():
    """锚点目标必须在开训前算：那一刻活着的模型就是 θ̄。"""
    from experiments.phase2k_qoc import run_qoc
    source = inspect.getsource(run_qoc.main)
    assert source.index("anchor.precompute(") < source.index("train_with_anchor(")
