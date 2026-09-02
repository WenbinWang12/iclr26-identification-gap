"""GFA 在 `run_qoc.py` 里的接线检查。协议 §6。

这些测试读源码而不是跑训练：整流程要 140 分钟，而"有没有接错"是可以静态判定的。
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path("experiments/phase2k_qoc/run_qoc.py").read_text(encoding="utf-8")


def test_none_branch_still_calls_phase2k_train_one_task_verbatim():
    """`--cl-method none` 必须与收敛跑逐字相同，否则基线就不是基线了。"""
    m = re.search(r'if args\.cl_method == "none":(.*?)elif', SRC, re.S)
    assert m is not None
    body = m.group(1)
    assert "train_one_task(" in body
    for foreign in ("gfa", "anchor", "history", "penalty"):
        assert foreign not in body


def test_gfa_registers_scope_after_training_not_before():
    """训完才 observe：任务 1 训练时没有任何 scope，锚定恒 0。"""
    i_train = SRC.index('elif args.cl_method == "gfa":')
    i_observe = SRC.index("gfa.observe(")
    assert i_train < i_observe


def test_gfa_precomputes_before_training():
    seg = SRC[SRC.index('elif args.cl_method == "gfa":'):]
    seg = seg[:seg.index("losses, penalties")]
    assert seg.index("gfa.precompute(") < seg.index("train_with_anchor(")


def test_gfa_scope_key_matches_the_codebook_scope_key():
    """GFA 的 scope 名必须和判据/码本用的域定义同源，否则两边说的不是同一个域。"""
    assert 'gfa.observe("|".join(scopes[task.name])' in SRC


def test_gfa_lambda_flag_exists_and_defaults_to_zero():
    assert '"--gfa-lambda", type=float, default=0.0' in SRC


def test_gfa_is_a_choice_of_cl_method():
    assert '"none", "olora", "vla", "gfa"' in SRC


def test_gfa_lambda_is_registered_in_the_gate():
    from experiments.phase2m_vla.lambda_gate_vla import LAMBDA_KEY
    assert LAMBDA_KEY["gfa"] == "gfa_lambda"


def test_gfa_reads_the_same_example_field_as_vla():
    """`prompt`，与 order4_data.OfficialExample 一致。冒烟跑曾因 `source` 而炸。"""
    gsrc = Path("experiments/phase2n_gfa/gfa.py").read_text(encoding="utf-8")
    vsrc = Path("experiments/phase2m_vla/vla.py").read_text(encoding="utf-8")
    assert "e.prompt for e in examples" in gsrc
    assert ".prompt" in vsrc
    assert ".source for" not in gsrc


def test_official_example_has_prompt_and_not_source():
    from experiments.phase2i_anchored_cvar.order4_data import OfficialExample
    names = set(OfficialExample.__dataclass_fields__)
    assert "prompt" in names
    assert "source" not in names
