"""接线检查：静态读 `run_qoc.py`，钉住协议 §6 里靠看代码保证的那几条。

Phase-2P 的 smoke 失败（`cap-per-class` 与 risk+audit 不相容）和 Phase-2N 的两次
（`e.source` 不存在、scope 键顺序）都能被静态检查在烧 85 分钟之前挡下来。这里再加
一条那两个阶段没有的：BPO 的偏移计算调用点**不许出现标签**。
"""

from __future__ import annotations

import re
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[2] / "phase2k_qoc" / "run_qoc.py"
SRC = SRC_PATH.read_text(encoding="utf-8")


def _bpo_block() -> str:
    """BPO 块的源码，到它自己的 `bpo_info["imbalanced"]` 赋值为止。

    用这一行收尾而不是用下一个 `if`，是因为后者会把 Phase-2L 的 λ 闸门整段吞进来
    ——Phase-2P 的接线测试第一版就是那样假失败的。注释行剥掉：这些断言查的是
    **代码**碰了什么，而解释「为什么不读标签」的注释里必然出现「标签」。
    """
    block = SRC.split("if args.score_bpo:")[1]
    end = block.index('bpo_info["imbalanced"] = imbal')
    body = block[:end]
    return "\n".join(l for l in body.splitlines()
                     if not l.lstrip().startswith("#"))


def test_flag_defaults_to_off():
    """默认关闭：不传 `--score-bpo` 时 2K-2P 的产出键集必须逐字不变。"""
    assert '"--score-bpo", action="store_true"' in SRC
    assert "bpo_info = None" in SRC
    assert "if bpo_info is not None:" in SRC


def test_bpo_does_not_touch_the_cl_method_choices():
    """Phase-2Q 不是第四个正则（2N §5 结局 4）：训练方法集合不许变。"""
    assert '"none", "olora", "vla", "gfa"' in SRC
    assert '"bpo"' not in SRC.split("choices=[")[1].split("]")[0]


def test_the_offset_call_never_receives_labels():
    """协议 §6 第一条。`fit_batch_prior_offset` 的每个调用点只能传 logits。"""
    calls = re.findall(r"fit_batch_prior_offset\(([^)]*)\)", SRC)
    assert calls, "no BPO call found — did the wiring get removed?"
    for arg in calls:
        assert "label" not in arg, arg
        assert "targets" not in arg, arg
        assert "entry.labels" not in arg, arg


def test_bpo_scored_on_the_same_audit_logits_as_every_comparator():
    """协议 §6：BPO 与对照必须在同一批 audit logits 上打分。"""
    block = _bpo_block()
    assert "entry.logits" in block
    assert "collect_logits" not in block, (
        "BPO must reuse the stage's audit logits; a fresh forward pass would "
        "make the comparison against R_raw/R_shr_global unverifiable")


def test_bpo_reads_no_old_task_partitions():
    """偏移是这一批查询的函数。块里不许出现 partitions 或 risk。"""
    block = _bpo_block()
    assert "partitions[" not in block
    assert ".risk" not in block
    assert "risk_logits" not in block


def test_q4_and_q5_are_computed_in_the_same_run_as_q1():
    """Q4 是唯一能证伪「Q1 是平衡划分假象」的工具，不许留到以后补。"""
    block = _bpo_block()
    assert "subsample_imbalanced" in block
    assert "R_bpo_by_batch" in block
    for ratio in ("0.5", "0.7", "0.9"):
        assert ratio in block


def test_q4_scores_on_the_imbalanced_subset_itself():
    """不平衡混合必须在**该子集**上评：查询混合就是它。"""
    block = _bpo_block()
    assert "sub_logits, sub_labels, b_sub" in block
    assert "class0_share" in block


def test_q4_reports_its_own_baselines():
    """Q4 的 R_bpo 必须和同一子集上的 R_raw / R_shr_global 并列，否则无从比较。"""
    block = _bpo_block().split("imbal[str(ratio)] = {")[1]
    for key in ('"R_bpo"', '"R_raw"', '"R_shr_global"'):
        assert key in block, key


def test_protocol_sha_is_cited_in_the_wiring():
    """冻结 SHA 写在代码里，事后改协议就会和代码对不上。"""
    assert "52b85c5c" in SRC


def test_histogram_recorded_both_with_and_without_the_offset():
    """要能事后判断 BPO 到底把预测分布搬动了多少，两个直方图都得留。"""
    block = _bpo_block()
    assert '"hist"' in block and '"hist_no_offset"' in block