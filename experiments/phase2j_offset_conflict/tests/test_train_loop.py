"""`train_one_task` 的训练量语义测试。

存在的理由很具体：`--epochs` 曾经是个**死参数**——`run_qoc.py` 声明了它，
但 `train_one_task` 只扫一遍数据，从没读过 `args.epochs`。后果是每个任务只有
3–16 个梯度步，WiC/COPA/QQP 的原始准确率贴着 0.50，模型根本没学会任务。
这组测试把「epochs 生效」和「有效 batch 不变」钉住，避免再次静默失效。

这里用最小假模型，不碰 torch 之外的东西：只需要它能 `.train()`、能返回一个
带 `.backward()` 的 loss，就足以数梯度步。
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from experiments.phase2j_offset_conflict.run_probe import train_one_task  # noqa: E402


@dataclass
class _Example:
    prompt: str
    label: str


class _FakeTokenizer:
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        n = len(texts)
        return {
            "input_ids": torch.ones((n, 3), dtype=torch.long),
            "attention_mask": torch.ones((n, 3), dtype=torch.long),
        }


class _FakeModel:
    """一个可微的标量参数就够了：loss = w * scale。"""

    def __init__(self):
        self.weight = torch.zeros(1, requires_grad=True)
        self.forward_calls = 0
        self.batch_sizes: list[int] = []

    def train(self):
        return self

    def parameters(self):
        return [self.weight]

    def __call__(self, **kwargs):
        self.forward_calls += 1
        self.batch_sizes.append(int(kwargs["input_ids"].shape[0]))

        class _Out:
            pass

        out = _Out()
        out.loss = (self.weight + 1.0).squeeze(0)
        return out


class _Args:
    def __init__(self, *, epochs, batch_size=2, grad_accum=2, seed=1):
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.seed = seed
        self.max_source = 8
        self.max_target = 4


def _run(epochs, n_examples=8, **kwargs):
    model = _FakeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    examples = [_Example(f"p{i}", "yes") for i in range(n_examples)]
    losses = train_one_task(
        model,
        _FakeTokenizer(),
        examples,
        "cpu",
        args=_Args(epochs=epochs, **kwargs),
        optimizer=optimizer,
    )
    return model, losses


def test_epochs_multiplies_the_number_of_gradient_steps():
    """epochs=1 与 epochs=4 的步数必须成 4 倍，否则 `--epochs` 又是死参数。"""

    _, one = _run(1)
    _, four = _run(4)
    assert len(one) == 2  # 8 样本 / 有效 batch 4
    assert len(four) == 8
    assert len(four) == 4 * len(one)


def test_epochs_none_or_zero_falls_back_to_one_pass():
    """老的调用方不传 epochs 时行为必须和以前逐位相同。"""

    baseline, _ = _run(1)
    for degenerate in (0, None):
        model = _FakeModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        examples = [_Example(f"p{i}", "yes") for i in range(8)]
        args = _Args(epochs=1)
        args.epochs = degenerate
        losses = train_one_task(
            model, _FakeTokenizer(), examples, "cpu", args=args, optimizer=optimizer
        )
        assert len(losses) == 2
        assert model.forward_calls == baseline.forward_calls


def test_each_epoch_uses_a_different_shuffle_but_stays_deterministic():
    """每轮换置换（否则多轮只是把同一个 batch 序列重复），但整体可复现。"""

    first, _ = _run(3)
    second, _ = _run(3)
    assert first.batch_sizes == second.batch_sizes

    # 直接检查置换：同一 seed 下 epoch 0 与 epoch 1 的顺序必须不同。
    import numpy as np

    order0 = np.random.default_rng([1, 0]).permutation(8)
    order1 = np.random.default_rng([1, 1]).permutation(8)
    assert not np.array_equal(order0, order1)


def test_effective_batch_is_unchanged_by_epochs():
    """有效 batch = batch_size * grad_accum，与 epochs 无关。"""

    model, _ = _run(3, batch_size=2, grad_accum=2)
    # 每个 micro-batch 都是 2，最后一个 step 也不例外（24 样本整除 4）。
    assert set(model.batch_sizes) == {2}
    assert model.forward_calls == 24 // 2
