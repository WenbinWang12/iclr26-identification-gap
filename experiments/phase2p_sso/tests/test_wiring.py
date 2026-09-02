"""接线检查：静态读 `run_qoc.py` 源码，钉住协议 §6 里「靠看代码保证」的那几条。

这些断言看起来像形式主义，但 Phase-2N 的两次 smoke 失败（`e.source` 不存在、
scope 键顺序不同）都是接线错误而不是方法错误，且都能被静态检查在跑 140 分钟
之前挡下来。这里再加一条 2N 没有的：SSO 的登记调用**只能**看当前任务的 risk。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parents[2] / "phase2k_qoc" / "run_qoc.py"
SRC = SRC_PATH.read_text(encoding="utf-8")


def test_flag_defaults_to_off():
    """默认关闭：不传 `--record-sso` 时 2K/2L/2M/2N 的产出键集必须逐字不变。"""
    assert '"--record-sso", action="store_true"' in SRC


def test_sso_is_none_unless_the_flag_is_given():
    assert "sso = None" in SRC
    assert "if args.record_sso:" in SRC


def test_sso_does_not_touch_the_cl_method_choices():
    """Phase-2P 不是第四个正则（2N §5 结局 4）：训练方法集合不许变。"""
    assert '"none", "olora", "vla", "gfa"' in SRC
    assert '"sso"' not in SRC.split("choices=[")[1].split("]")[0]


def _record_block() -> str:
    """登记块的源码：从 `if sso is not None and task.name in scorable:` 起，
    到它自己那行 log 为止。用 log 行收尾而不是用下一个 `if`，因为后者会把整个
    rehearsal 评估循环也吞进来——第一版就是那样，两条断言因此假失败。"""
    block = SRC.split("if sso is not None and task.name in scorable:")[1]
    end = block.index('log("  sso: entries=%d')
    body = block[:end]
    # 去掉注释行：这些断言查的是**代码**碰了什么，而解释「为什么不用
    # risk_logits」的注释里必然出现 `risk_logits` 这个词。
    return "\n".join(l for l in body.splitlines()
                     if not l.lstrip().startswith("#"))


def test_record_reads_only_the_current_task_risk_split():
    """协议 §6 第一条。登记块里出现的 partitions 下标只能是当前任务。"""
    block = _record_block()
    refs = set(re.findall(r"partitions\[([^\]]+)\]", block))
    assert refs == {"task.name"}, refs
    assert ".risk" in block
    assert ".update" not in block and ".audit" not in block


def test_record_does_not_iterate_over_seen_tasks():
    """登记块不许出现 `for old in seen` 这类遍历——那是 rehearsal 结构。"""
    block = _record_block()
    assert "seen" not in block
    assert "risk_logits" not in block, (
        "SSO must do its own forward pass; reading the rehearsal loop's "
        "risk_logits would give the same numbers but destroy verifiability")


def test_record_happens_after_training_at_the_task_boundary():
    """偏移必须是「该任务刚训完」那一刻的量，否则它不是那个任务的最优偏移。"""
    train = SRC.index("outcome = train_with_penalty")
    record = SRC.index("if sso is not None and task.name in scorable:")
    assert train < record


def test_scoring_uses_the_same_audit_logits_as_the_rehearsal_ceiling():
    """协议 §6：R_sso1 与 R_shr_scoped 必须在同一批 audit logits 上打分。"""
    block = SRC.split("if sso is not None and old.name in scorable:")[1]
    block = block.split("if old.name in scorable:")[0]
    assert "entry.logits" in block and "entry.labels" in block
    assert "collect_logits" not in block


def test_m1_is_scored_without_a_router():
    """m=1 不许拿状态：中心只有一行，给它状态只会制造一个假的路由决策。"""
    block = SRC.split("if sso is not None and old.name in scorable:")[1]
    m1 = block.split("if m == 1:")[1].split("else:")[0]
    assert "states" not in m1
    assert "offset_for" in m1


def test_m4_uses_the_honest_prototype_router_not_the_leaky_one():
    """`prototype_official` 读得到 `Task:` 表头，等于 oracle，不构成方法证据。"""
    block = SRC.split("if sso is not None and old.name in scorable:")[1]
    block = block.split("if old.name in scorable:")[0]
    assert 'router="prototype"' in block
    assert "official" not in block
    assert "strip_header=True" in SRC.split("state_mean=")[0][-2000:]


def test_sso_books_are_built_from_the_streaming_table_not_the_refit_optima():
    """两套码本的输入必须不同，否则 SSO 就只是 rehearsal 的别名。"""
    assert "sso.codebooks(m)" in SRC
    block = SRC.split("sso_books = ")[1][:200]
    assert "scorable_optima" not in block


def test_states_for_is_defined_before_the_sso_scoring_block():
    assert SRC.index("def states_for") < SRC.index("R_sso%d")


def test_protocol_sha_is_cited_in_the_flag_comment():
    """冻结 SHA 写在代码里，事后改协议就会和代码对不上。"""
    assert "ae59bcd5" in SRC


@pytest.mark.parametrize("token", ["R_sso1", "R_sso4"])
def test_reported_keys_are_the_protocol_names(token):
    assert '"R_sso%d"' in SRC or token in SRC
