"""Phase-2L 接线检查：两臂必须只差一个正则子。

这些测试对应协议 §6 的反伪影清单，用假模型/假分词器，不碰 GPU。
"""

from __future__ import annotations

import argparse
import types

import numpy as np
import pytest
import torch
from torch import nn

from experiments.phase2l_additivity.olora_penalty import OrthogonalHistory
from experiments.phase2l_additivity.train_with_penalty import (
    train_one_task_with_penalty,
)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        n = len(texts)
        return {"input_ids": torch.ones((n, 3), dtype=torch.long),
                "attention_mask": torch.ones((n, 3), dtype=torch.long)}


class FakeModel(nn.Module):
    """损失 = 参数平方和 × 常数，这样梯度确定、可复现，且惩罚可叠加。"""

    def __init__(self, *, r: int = 2, d: int = 4):
        super().__init__()
        self.q_A = nn.Parameter(torch.ones(r, d) * 0.5)
        self.q_B = nn.Parameter(torch.ones(d, r) * 0.5)
        self.seen_batches: list[int] = []

    def named_parameters(self, *a, **k):
        yield "base_model.encoder.q.lora_A.default.weight", self.q_A
        yield "base_model.encoder.q.lora_B.default.weight", self.q_B

    def parameters(self, *a, **k):
        for _, p in self.named_parameters():
            yield p

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        self.seen_batches.append(int(input_ids.shape[0]))
        loss = (self.q_A ** 2).sum() + (self.q_B ** 2).sum()
        return types.SimpleNamespace(loss=loss)


def _args(**over):
    base = dict(seed=1, epochs=1, batch_size=2, grad_accum=2,
                max_source=8, max_target=2)
    base.update(over)
    return argparse.Namespace(**base)


def _examples(n=8):
    return [types.SimpleNamespace(prompt="p%d" % i, label="l%d" % i) for i in range(n)]


def _run(history, *, args=None, n=8):
    torch.manual_seed(0)
    model = FakeModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    out = train_one_task_with_penalty(model, FakeTokenizer(), _examples(n), "cpu",
                                     args=args or _args(), optimizer=opt,
                                     history=history)
    final = torch.cat([model.q_A.flatten(), model.q_B.flatten()]).detach().clone()
    return out, final


def test_history_none_and_lambda_zero_give_identical_parameters():
    """协议 §6 第二条：λ=0 且代码路径激活，必须与不加惩罚逐位相同。"""
    zero = OrthogonalHistory(lam=0.0)
    zero.snapshot(FakeModel())
    a, pa = _run(None)
    b, pb = _run(zero)
    assert torch.equal(pa, pb)
    assert a["task_losses"] == b["task_losses"]
    assert all(p == 0.0 for p in b["penalties"])


def test_nonzero_lambda_changes_parameters():
    """惩罚必须真的进入梯度，否则整个 Phase-2L 是空转。"""
    hist = OrthogonalHistory(lam=1.0)
    hist.snapshot(FakeModel())
    _, pa = _run(None)
    _, pb = _run(hist)
    assert not torch.equal(pa, pb)


def test_penalty_is_added_once_per_step_not_per_micro_batch():
    """按 micro 加会让 λ 随 grad_accum 漂移。用两种 micro 拆分对比惩罚值。

    有效 batch 固定 4：(micro=2, accum=2) 与 (micro=1, accum=4) 的
    step 数相同，惩罚序列必须相同。
    """
    def penalties(micro, accum):
        hist = OrthogonalHistory(lam=1.0)
        hist.snapshot(FakeModel())
        out, _ = _run(hist, args=_args(batch_size=micro, grad_accum=accum))
        return out["penalties"]
    a = penalties(2, 2)
    b = penalties(1, 4)
    assert len(a) == len(b)
    assert a == pytest.approx(b, rel=1e-6)


def test_step_count_matches_effective_batch():
    out, _ = _run(None, args=_args(batch_size=2, grad_accum=2), n=8)
    assert len(out["task_losses"]) == 2       # 8 / (2*2)
    out, _ = _run(None, args=_args(batch_size=2, grad_accum=2, epochs=3), n=8)
    assert len(out["task_losses"]) == 6       # 3 轮


def test_first_task_penalty_is_zero_because_history_is_empty():
    hist = OrthogonalHistory(lam=1.0)          # 没 snapshot 过
    out, _ = _run(hist)
    assert all(p == 0.0 for p in out["penalties"])
    assert hist.n_history() == 0


def test_reshuffle_is_deterministic_given_seed_and_epoch():
    """两臂共享数据顺序，靠的是这个派生种子；换臂不得改变顺序。"""
    def order(seed, epochs):
        got = []
        for epoch in range(epochs):
            got.append(np.random.default_rng([seed, epoch]).permutation(8).tolist())
        return got
    assert order(1, 3) == order(1, 3)
    assert order(1, 3) != order(2, 3)


def test_run_qoc_none_arm_does_not_construct_penalty_history():
    """`--cl-method none` 必须走 Phase-2K 原路径，不碰 Phase-2L 的对象。"""
    import inspect

    from experiments.phase2k_qoc import run_qoc

    source = inspect.getsource(run_qoc.main)
    assert 'if args.cl_method == "olora"' in source
    assert 'if args.cl_method == "none"' in source
    # none 臂调用的是 Phase-2K 的 train_one_task
    idx = source.index('if args.cl_method == "none"')
    none_branch = source[idx: source.index("else:", idx)]
    assert "train_one_task(" in none_branch
    assert "train_with_penalty" not in none_branch


def test_run_qoc_records_the_arm_in_output():
    """臂必须落进产出文件，否则事后分不清哪个 json 是哪臂。"""
    import inspect

    from experiments.phase2k_qoc import run_qoc

    source = inspect.getsource(run_qoc.main)
    assert '"args": vars(args)' in source
    assert '"--cl-method"' in source and '"--olora-lambda"' in source


def test_snapshot_happens_after_training_each_task():
    """快照必须在训练后：否则任务 1 的训练里就有了自己的历史，惩罚不为 0。"""
    import inspect

    from experiments.phase2k_qoc import run_qoc

    source = inspect.getsource(run_qoc.main)
    train_idx = source.index("train_with_penalty(")
    snap_idx = source.index("history.snapshot(model)")
    assert snap_idx > train_idx


def test_order4_data_is_untouched_by_phase2l():
    """协议 §1：order4_data 的 SHA 被 Phase-2I 钉住，Phase-2L 不得改它。"""
    import inspect

    from experiments.phase2i_anchored_cvar import order4_data as od

    source = inspect.getsource(od)
    for token in ("olora", "cl_method", "penalty", "orthogonal"):
        assert token not in source.lower(), token
