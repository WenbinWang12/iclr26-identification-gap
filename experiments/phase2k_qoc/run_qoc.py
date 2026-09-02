"""Phase-2K：在真实 Order-4 任务流上跑 QOC，按 `notes/method_qoc_v1.md` §5 分解。

与 Phase-2J 探针共用数据路径、划分、首片索引和打分口径，唯一新增的是方法侧：
词表域划分（免费）+ 域内偏移码本（花预算）+ 任务无关路由。

一切以**首步受限 argmax 平衡准确率**报告（协议 A4.5）；自由生成 EM 只作为
与 O-LoRA 已发表数字的可比锚点单独记录，不与前者混用。

`test.json` 完全不读；偏移与码本在 train 派生的 risk 划分上拟合，在互不相交的
audit 划分上打分。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_HOME", "/mnt/data/wenbin/iclr26/models/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.phase2i_anchored_cvar import order4_data as od  # noqa: E402
from experiments.phase2j_offset_conflict.offsets import (  # noqa: E402
    TaskLogits,
    balanced_accuracy,
    fit_offset,
    fit_shared_offset,
    gauge_fix,
)
from experiments.phase2j_offset_conflict.run_probe import (  # noqa: E402
    collect_logits,
    first_piece_ids,
    log,
    precondition_ok,
    train_one_task,
    with_oom_retry,
)
from experiments.phase2s_sio.sio import bc_offset, sio_offset  # noqa: E402
from experiments.phase2q_bpo.bpo import (  # noqa: E402
    fit_batch_prior_offset, predicted_histogram, subsample_imbalanced)
from experiments.phase2k_qoc.qoc import (  # noqa: E402
    TaskOptima,
    fit_scope_codebooks,
    score_with_codebook,
    scope_key,
)
from experiments.phase2l_additivity.train_with_penalty import (  # noqa: E402
    train_one_task_with_penalty as train_with_penalty,
)
from experiments.phase2m_vla.train_with_anchor import (  # noqa: E402
    train_one_task_with_anchor as train_with_anchor,
)


def grow_update_split(parts, all_examples, *, update_cap_per_class: int, seed: int):
    """只放大 update 划分，risk/audit 逐条不变。

    为什么不直接调大 `--cap-per-class`：risk/audit 是在**已封顶的**组内排序取
    前若干条的，调大 cap 会换掉进入 risk/audit 的样本，于是收敛版和 cap=200
    那版就是在不同评测集上打分，逐任务对照全部作废。这里从完整 train 池补
    update，但严格排除已在 risk/audit 的 id，所以唯一变化的量是训练量。

    这个函数刻意放在 Phase-2K 而不是 `order4_data.py`：后者的 SHA-256 被
    Phase-2I 的冻结依赖清单钉住（`run_pcsm_lora_transfer.py` 会校验），改它会
    让那批已封存结果的审计链断掉。方法侧的新需求就该由方法侧承担。
    """

    if (
        not isinstance(update_cap_per_class, int)
        or isinstance(update_cap_per_class, bool)
        or update_cap_per_class <= 0
    ):
        raise ValueError("update_cap_per_class must be a positive integer")

    import hashlib
    from dataclasses import replace as _dc_replace  # noqa: F401  (仅为显式依赖)

    reserved = {example.example_id for example in parts.risk}
    reserved |= {example.example_id for example in parts.audit}
    groups: dict[tuple[str, str], list] = {}
    for example in all_examples:
        if example.example_id in reserved:
            continue
        groups.setdefault((example.task_name, example.label), []).append(example)

    def rank(example):
        # 与 order4_data._rank_key 同构的稳定散列，但用本阶段自己的 purpose，
        # 避免与它的 per-class-cap / train-partition 排序相撞。
        payload = json.dumps(
            {"seed": int(seed), "purpose": "phase2k-update-cap",
             "example_id": example.example_id},
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    grown = []
    for group in groups.values():
        ordered = sorted(group, key=lambda item: (rank(item), item.example_id))
        grown.extend(ordered[:update_cap_per_class])
    grown.sort(key=lambda item: (item.task_name, item.source_index))

    result = od.TrainPartitions(update=tuple(grown), risk=parts.risk, audit=parts.audit)
    result.assert_disjoint()
    return result


def strip_task_header(prompt: str) -> str:
    """去掉官方提示词开头的 `Task:...\\nDataset:...\\n` 两行。

    **这不是可选的清洁工作，是结果能否成立的前提。**官方 O-LoRA 提示词字面
    包含 `Task:WiC\\nDataset:WiC`，也就是说任务身份**以文本形式写在输入里**。
    任何读输入的路由器都能直接把它读出来，那样的 prototype 路由只是换了个
    形式的 oracle，不是任务无关方法。所以路由器必须同时在两种视图下评估：

    * `official`  —— 保留数据集名。这个数字**不构成证据**，只用来量化泄漏幅度。
    * `stripped`  —— 去掉那两行。这才是诚实的任务无关路由。

    分类本身始终用官方提示词（保持与已发表数字可比），只有路由器的视图被剥离。
    """

    lines = prompt.split("\n")
    keep = [line for line in lines[:2]
            if not (line.startswith("Task:") or line.startswith("Dataset:"))]
    return "\n".join(keep + lines[2:])


@torch.no_grad()
def collect_states(model, tokenizer, examples, device, *, max_source: int,
                   batch_size: int, strip_header: bool = False) -> np.ndarray:
    """编码器输出的掩码均值池化，供 prototype 路由使用。

    用 attention mask 做加权平均，否则 padding 会把不同长度的样本推向不同
    方向，路由就会退化成「按长度分簇」。

    `strip_header=True` 时先剥掉 `Task:`/`Dataset:` 两行，见
    `strip_task_header` 的说明 —— 这是路由结果是否算数的分界线。
    """

    model.eval()
    chunks = []
    encoder = model.get_encoder()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        texts = [example.prompt for example in batch]
        if strip_header:
            texts = [strip_task_header(text) for text in texts]
        enc = tokenizer(
            texts, max_length=max_source, truncation=True, padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = encoder(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        chunks.append(pooled.cpu())
    return torch.cat(chunks, dim=0).numpy()


def scoped_shared_offsets(risk_logits, scopes, scorable, per_task_offsets) -> dict:
    """每个词表域各拟合一个共享偏移（方法 §2 第一步，m=1）。

    这是「免费」的那一级：域是从提示词的选项列表读出来的，不需要 task ID。

    单任务域**直接复用该任务自己的最优偏移**，而不是再调一次
    `fit_shared_offset`。理由：域内只有一个任务时，两者在数学上是同一个量，
    但它们是两套不同的搜索过程（`fit_offset` 直接爬准确率，
    `fit_shared_offset` 爬「最差任务损失」），在 risk 上打平、在 audit 上
    可能分叉。smoke 跑里 COPA 就因此出现 Δ_id_scoped = −0.0625，即共享偏移
    反而胜过 per-task oracle —— 那是实现噪声，不是现象。按构造对齐消除它。
    """
    by_scope: dict[tuple[str, ...], list] = {}
    for name, entry in risk_logits.items():
        if name in scorable:
            by_scope.setdefault(scopes[name], []).append(entry)

    out = {}
    for scope, pool in by_scope.items():
        if len(pool) == 1:
            only = pool[0]
            offset = gauge_fix(per_task_offsets[only.task])
            out[scope] = {token: float(offset[i])
                          for i, token in enumerate(only.verbalizer)}
        else:
            out[scope] = fit_shared_offset(pool)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/mnt/data/wenbin/iclr26/data/order4")
    parser.add_argument("--model", default="google-t5/t5-large")
    parser.add_argument("--out", default="/mnt/data/wenbin/iclr26/runs/phase2k_qoc")
    parser.add_argument("--n-tasks", type=int, default=15)
    parser.add_argument("--cap-per-class", type=int, default=200)
    parser.add_argument("--risk-per-class", type=int, default=64)
    parser.add_argument("--audit-per-class", type=int, default=64)
    parser.add_argument("--min-rare-class", type=int, default=40)
    parser.add_argument("--m-values", default="1,2,4")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    # 只放大 update 划分，risk/audit 逐条不变（见 order4_data 的说明），
    # 这样收敛版跑出来的数可以和 cap=200 那版逐任务对照。
    parser.add_argument("--update-cap-per-class", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-source", type=int, default=512)
    parser.add_argument("--max-target", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--seed", type=int, default=1)
    # Phase-2L 的可加性测试（notes/phase2l_additivity_protocol.md，冻结 SHA
    # 8156a706…）需要在**同一条度量路径**上换掉训练时的正则子。复制本文件会让
    # 两臂的口径可能悄悄分叉，所以改成一个开关，默认 "none" 时行为与 Phase-2K
    # 收敛跑逐字相同：`train_one_task` 原样调用，不 import 任何 Phase-2L 代码。
    parser.add_argument("--cl-method", default="none",
                        choices=["none", "olora", "vla", "gfa"])
    parser.add_argument("--olora-lambda", type=float, default=0.0)
    # Phase-2M（notes/phase2m_vla_protocol.md，冻结 SHA b3225028…）：锚住
    # 旧任务 verbalizer 上的 logit 分布，即实测承载 78% 遗忘的那个量。
    parser.add_argument("--vla-lambda", type=float, default=0.0)
    parser.add_argument("--vla-temperature", type=float, default=2.0)
    # Phase-2N（notes/phase2n_gauge_fixed_anchor_protocol.md，冻结 SHA 420383b4…）：
    # 锚住 **gauge-fixed** 的 scope 内 logit 对比（零均值投影），即 Prop 2a 的对象。
    # VLA 的全部代价都落在规范方向上（2M §A2.3，移除比例 1.059），GFA 付不了那笔钱。
    parser.add_argument("--gfa-lambda", type=float, default=0.0)
    parser.add_argument("--stop-after-tasks", type=int, default=None,
                        help="只跑前 N 个任务；供 Phase-2L §4 的 λ 选择闸门使用。")
    parser.add_argument("--gate-update-eval", action="store_true",
                        help="在最后一个 stage 额外评 update 划分，供 Phase-2L §4 的 "
                             "λ 闸门在已消耗标签上选 λ。默认关闭，键集不变。")
    # Phase-2P（notes/phase2p_streaming_scope_offset_protocol.md，冻结 SHA ae59bcd5…）：
    # 流式域偏移表。**不改训练**——`--cl-method` 仍是 none——只在任务边界记下该任务
    # 自己的最优偏移，推理时按域套用。默认关闭，于是不传这个开关时 2K/2L/2M/2N 的
    # 产出键集逐字不变，旧判据的复现检查才成立。
    parser.add_argument("--record-sso", action="store_true",
                        help="记录流式域偏移表并在每个 stage 额外报告 R_sso1/R_sso4。")
    parser.add_argument("--score-sio", action="store_true",
                        help="Phase-2S: score SIO and a re-implemented Batch "
                             "Calibration on the same audit logits (K_S=2 only)")
    parser.add_argument("--dump-logits", default=None,
                        help="directory to write per-stage audit logits as .npz. "
                             "Phase-2S §A1.1: the Alibaba run stored none, so the "
                             "phase could not be re-scored without retraining")
    parser.add_argument("--score-bpo", action="store_true",
                        help="每个 stage 额外报告 Phase-2Q 的 BPO：偏移由查询批自己算出，不读标签、不存状态。含 Q4 不平衡混合与 Q5 批大小敏感性。")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    m_values = [int(v) for v in args.m_values.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(args.model, legacy=False)
    weight_dtype = getattr(torch, args.dtype)
    base = T5ForConditionalGeneration.from_pretrained(args.model, dtype=weight_dtype)
    model = get_peft_model(
        base,
        LoraConfig(r=args.lora_r, lora_alpha=32, lora_dropout=0.05, bias="none",
                   task_type="SEQ_2_SEQ_LM", target_modules=["q", "v"]),
    )
    model.to(device)
    if weight_dtype is not torch.float32:
        for _, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    # Phase-2L 的正交历史。`--cl-method none` 时为 None，且上面的分支根本不会
    # 触到 Phase-2L 的代码，所以 Phase-2K 的复现路径不受本次改动影响。
    anchor = None
    if args.cl_method == "vla":
        from experiments.phase2m_vla.vla import VerbalizerAnchor
        anchor = VerbalizerAnchor(lam=args.vla_lambda,
                                  temperature=args.vla_temperature)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log("cl_method=vla lambda=%g tau=%g trainable_params=%d（锚定不引入参数）"
            % (args.vla_lambda, args.vla_temperature, n_trainable))
    gfa = None
    if args.cl_method == "gfa":
        from experiments.phase2n_gfa.gfa import GaugeFixedAnchor
        gfa = GaugeFixedAnchor(lam=args.gfa_lambda)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log("cl_method=gfa lambda=%g trainable_params=%d（锚定不引入参数）"
            % (args.gfa_lambda, n_trainable))

    sso = None
    if args.record_sso:
        from experiments.phase2p_sso.sso import StreamingScopeOffsets
        sso = StreamingScopeOffsets()
        log("record_sso=1（流式域偏移表；训练路径不变，cl_method=%s）"
            % args.cl_method)

    history = None
    if args.cl_method == "olora":
        from experiments.phase2l_additivity.olora_penalty import OrthogonalHistory
        history = OrthogonalHistory(lam=args.olora_lambda)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log("cl_method=olora lambda=%g trainable_params=%d（惩罚不引入参数）"
            % (args.olora_lambda, n_trainable))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    stream = list(od.ORDER4_TASKS[: args.n_tasks])
    scopes = {task.name: scope_key(task.labels) for task in stream}
    scorable, excluded = [], []
    for task in stream:
        (scorable if precondition_ok(tokenizer, task.labels) else excluded).append(task.name)

    partitions, budgets, undersized = {}, {}, []
    for task in stream:
        examples = od.load_official_examples(args.data_root, task.name, "train",
                                             verify=False)
        counts: dict[str, int] = {}
        for example in examples:
            counts[example.label] = counts.get(example.label, 0) + 1
        rarest = min(counts.values())
        allowance = max(1, min(args.risk_per_class, (rarest - 1) // 2))
        audit_allowance = max(1, min(args.audit_per_class, (rarest - 1) - allowance))
        if rarest < args.min_rare_class:
            undersized.append({"task": task.name, "rarest_class_count": rarest})
        budgets[task.name] = {"rarest_class_count": rarest,
                              "risk_per_class": allowance,
                              "audit_per_class": audit_allowance}
        parts = od.prepare_task_partitions(
            args.data_root, task.name, cap_per_class=args.cap_per_class,
            risk_per_class=allowance, audit_per_class=audit_allowance, seed=args.seed,
        )
        parts.assert_disjoint()
        if args.update_cap_per_class is not None:
            parts = grow_update_split(
                parts, examples,
                update_cap_per_class=args.update_cap_per_class, seed=args.seed,
            )
        budgets[task.name]["n_update"] = len(parts.update)
        partitions[task.name] = parts
    for entry in undersized:
        if entry["task"] in scorable:
            scorable.remove(entry["task"])
            excluded.append(entry["task"])
        log("task %s excluded: rarest class has only %d (< %d)"
            % (entry["task"], entry["rarest_class_count"], args.min_rare_class))

    # 域的成员统计（方法 §2 的表），以及哪些域是单任务的 —— Prop 1 预测那里
    # Δ_id = 0，这是可证伪预测而不是拟合。
    scope_members: dict[tuple[str, ...], list[str]] = {}
    for task in stream:
        scope_members.setdefault(scopes[task.name], []).append(task.name)

    record = {
        "args": vars(args),
        "official_commit": od.OFFICIAL_COMMIT,
        "scorable_tasks": scorable,
        "excluded_tasks": excluded,
        "partition_budgets": budgets,
        "scope_members": {"|".join(k): v for k, v in scope_members.items()},
        "singleton_scopes": ["|".join(k) for k, v in scope_members.items()
                             if len(v) == 1],
        "stages": [],
    }
    log("scopes=%d singleton=%d scorable=%d"
        % (len(scope_members), len(record["singleton_scopes"]), len(scorable)))
    # 闸门只在最后一个实际会跑到的 stage 触发；stop-after-tasks 会截断流。
    last_position = len(stream)
    if args.stop_after_tasks is not None:
        last_position = min(last_position, args.stop_after_tasks)
    started = time.time()

    for position, task in enumerate(stream, start=1):
        if args.stop_after_tasks is not None and position > args.stop_after_tasks:
            log("stop-after-tasks=%d reached; ending stream early"
                % args.stop_after_tasks)
            break
        if args.cl_method == "none":
            # Phase-2K 的原始路径，逐字不变。
            losses = train_one_task(model, tokenizer,
                                    list(partitions[task.name].update),
                                    device, args=args, optimizer=optimizer)
            penalties = []
        elif args.cl_method == "vla":
            update_examples = list(partitions[task.name].update)
            # 锚点目标在开训**前**算：此刻活着的模型就是 θ̄（上一个边界的模型），
            # 所以不需要任何副本，显存零开销、存储天然 O(1)。
            rows = anchor.precompute(
                model, tokenizer, update_examples, device,
                max_source=args.max_source, batch_size=args.eval_batch_size)
            if rows:
                log("  vla: anchored %d rows on %d previous verbalizer pieces"
                    % (rows, len(anchor.active_columns())))
            outcome = train_with_anchor(model, tokenizer, update_examples, device,
                                        args=args, optimizer=optimizer,
                                        anchor=anchor)
            losses, penalties = outcome["task_losses"], outcome["anchor_losses"]
        elif args.cl_method == "gfa":
            update_examples = list(partitions[task.name].update)
            # 与 VLA 同一时序：目标在开训前算，此刻活着的模型就是 θ̄。
            rows = gfa.precompute(
                model, tokenizer, update_examples, device,
                max_source=args.max_source, batch_size=args.eval_batch_size)
            if rows:
                log("  gfa: anchored %d rows on %d previous multi-task scopes"
                    % (rows, len(gfa.active_scopes())))
            outcome = train_with_anchor(model, tokenizer, update_examples, device,
                                        args=args, optimizer=optimizer,
                                        anchor=gfa)
            losses, penalties = outcome["task_losses"], outcome["anchor_losses"]
        else:
            outcome = train_with_penalty(model, tokenizer,
                                         list(partitions[task.name].update),
                                         device, args=args, optimizer=optimizer,
                                         history=history)
            losses, penalties = outcome["task_losses"], outcome["penalties"]
        log("task %d/%d %s steps=%d loss %.4f -> %.4f%s"
            % (position, len(stream), task.name, len(losses),
               losses[0] if losses else float("nan"),
               losses[-1] if losses else float("nan"),
               ("  pen %.4g -> %.4g (hist=%d)"
                % (penalties[0], penalties[-1], history.n_history()))
               if penalties and history is not None else
               ("  anchor %.4g -> %.4g (|V|=%d)"
                % (penalties[0], penalties[-1], len(anchor.active_columns())))
               if penalties and anchor is not None else
               ("  gfa %.4g -> %.4g (S=%d)"
                % (penalties[0], penalties[-1], len(gfa.active_scopes())))
               if penalties and gfa is not None else ""))
        if anchor is not None:
            # 训完才登记：于是任务 1 训练时 V_0 为空、锚定恒 0（协议 §6）。
            anchor.observe(first_piece_ids(tokenizer, task.labels))
        if gfa is not None:
            # 同样训完才登记。scope 键与 `scopes[name]` 一致（label 元组 join），
            # 于是 GFA 的 scope 划分和判据/码本用的是同一套域定义。
            gfa.observe("|".join(scopes[task.name]),
                        first_piece_ids(tokenizer, task.labels))
        if history is not None:
            # 任务边界处冻结当前 A 作为历史。放在训练**之后**、评估之前，
            # 于是任务 1 的训练里历史为空、惩罚恒 0（协议 §6）。
            history.snapshot(model)
        if sso is not None and task.name in scorable:
            # 只登记可打分任务，和 `books` 用 `scorable_optima` 建表对齐——否则
            # SSO 的域里会多出 rehearsal 上限里没有的成员，两者就不可比了。
            # `scorable` 在任务循环之前就已定稿（预条件 + 稀有类计数），所以这个
            # 判断不用到任何未来信息。
            # 协议 §1：任务边界处、**只用 `partitions[task.name].risk`** 拟合并登记。
            # 故意自己做一次前向，而不是复用下面评估循环里的 `risk_logits`：那个
            # 循环遍历所有已见任务，是 rehearsal 结构。若从它里面取值，代码上就无法
            # 一眼确认 SSO 没碰旧数据——数值一样，可验证性不一样。
            sso_examples = list(partitions[task.name].risk)
            sso_index = {label: i for i, label in enumerate(task.labels)}
            sso_logits, _ = with_oom_retry(
                lambda bs, spec=task, ex=sso_examples: collect_logits(
                    model, tokenizer, ex, spec, device,
                    max_source=args.max_source, batch_size=bs),
                batch_size=args.eval_batch_size,
            )
            sso_targets = np.array([sso_index[e.label] for e in sso_examples])
            sso_states = with_oom_retry(
                lambda bs: collect_states(
                    model, tokenizer, sso_examples, device,
                    max_source=args.max_source, batch_size=bs, strip_header=True),
                batch_size=args.eval_batch_size,
            )[0]
            sso.record(task.name, task.labels,
                       first_piece_ids(tokenizer, task.labels),
                       sso_logits, sso_targets,
                       state_mean=sso_states.mean(axis=0),
                       state_count=int(sso_states.shape[0]))
            log("  sso: entries=%d stored_floats=%d router_floats=%d"
                % (sso.n_entries(), sso.stored_floats(), sso.router_floats()))

        seen = stream[:position]
        risk_logits, audit_logits, audit_states = {}, {}, {}
        per_task_offsets, optima = {}, []
        for old in seen:
            index = {label: i for i, label in enumerate(old.labels)}
            for split, store in (("risk", risk_logits), ("audit", audit_logits)):
                examples = list(getattr(partitions[old.name], split))
                logits, _ = with_oom_retry(
                    lambda bs, spec=old, ex=examples: collect_logits(
                        model, tokenizer, ex, spec, device,
                        max_source=args.max_source, batch_size=bs),
                    batch_size=args.eval_batch_size,
                )
                labels = np.array([index[example.label] for example in examples])
                pieces = tuple(str(p) for p in first_piece_ids(tokenizer, old.labels))
                store[old.name] = TaskLogits(old.name, pieces, logits, labels)
            per_task_offsets[old.name] = fit_offset(
                risk_logits[old.name].logits, risk_logits[old.name].labels)
            optima.append(TaskOptima(
                task=old.name, labels=tuple(old.labels),
                pieces=first_piece_ids(tokenizer, old.labels),
                optimum=gauge_fix(per_task_offsets[old.name]),
            ))

        # 全局单一共享偏移 = Phase-2J 的基线，作为分解的起点。
        fit_pool = [risk_logits[old.name] for old in seen if old.name in scorable]
        global_shared = fit_shared_offset(fit_pool) if fit_pool else {}
        # 域内共享偏移（免费的那一级）。
        scoped = scoped_shared_offsets(risk_logits, scopes, scorable,
                                       per_task_offsets)

        # 码本按域建，只用可打分任务的最优偏移。
        scorable_optima = [o for o in optima if o.task in scorable]
        books = {m: fit_scope_codebooks(scorable_optima, m) for m in m_values}

        # 流式表的码本。和上面 `books` 的区别只在**输入**：`books` 用的是本
        # stage 重新拟合的 optima（rehearsal 上限），这里用的是各任务当时记下的
        # 陈旧偏移。协议 §1/§2。m=1 是主判据，m=4 报码本那一级。
        sso_m_values = (1, 4) if sso is not None else ()
        sso_books = ({m: sso.codebooks(m) for m in sso_m_values}
                     if sso is not None else {})

        # 编码器状态缓存：(任务, 划分, 是否剥离表头) -> 池化状态。
        # 剥离视图与官方视图都要，因为路由必须在两种视图下分别评估。
        state_cache: dict[tuple[str, str, bool], np.ndarray] = {}

        def states_for(name: str, split: str, stripped: bool) -> np.ndarray:
            key = (name, split, stripped)
            if key not in state_cache:
                state_cache[key] = with_oom_retry(
                    lambda bs: collect_states(
                        model, tokenizer, list(getattr(partitions[name], split)),
                        device, max_source=args.max_source, batch_size=bs,
                        strip_header=stripped),
                    batch_size=args.eval_batch_size,
                )[0]
            return state_cache[key]

        # prototype 路由的质心：**只用 risk 划分**，且按码本分配聚合。
        # `prototypes` 是剥离视图（诚实版），`prototypes_official` 是保留
        # 数据集名的版本，只用于量化泄漏幅度。
        for m, book_set in books.items():
            for scope, book in book_set.items():
                vectors, official, counts = [], [], []
                for j in range(book.centres.shape[0]):
                    members = [name for name, idx in book.assignment.items() if idx == j]
                    pooled = [states_for(name, "risk", True) for name in members]
                    pooled_off = [states_for(name, "risk", False) for name in members]
                    if pooled:
                        stacked = np.concatenate(pooled, axis=0)
                        vectors.append(stacked.mean(axis=0))
                        official.append(np.concatenate(pooled_off, axis=0).mean(axis=0))
                        counts.append(int(stacked.shape[0]))
                    else:
                        vectors.append(np.zeros(model.config.d_model))
                        official.append(np.zeros(model.config.d_model))
                        counts.append(0)
                book.prototypes = np.stack(vectors)
                book.prototypes_official = np.stack(official)
                book.prototype_counts = counts

        stage = {"position": position, "trained_task": task.name,
                 "n_steps": len(losses), "tasks": {}}
        for old in seen:
            entry = audit_logits[old.name]
            r_raw = balanced_accuracy(entry.logits, entry.labels, None)
            r_orc = balanced_accuracy(entry.logits, entry.labels,
                                      gauge_fix(per_task_offsets[old.name]))
            g_sub = np.array([global_shared.get(p, 0.0) for p in entry.verbalizer])
            r_glob = balanced_accuracy(entry.logits, entry.labels, gauge_fix(g_sub))
            scope = scopes[old.name]
            s_map = scoped.get(scope, {})
            s_sub = np.array([s_map.get(p, 0.0) for p in entry.verbalizer])
            r_scoped = balanced_accuracy(entry.logits, entry.labels, gauge_fix(s_sub))

            # Phase-2Q BPO：偏移**由这一批查询自己算出**，不读标签、不存状态、
            # 不看任务身份。评的是与上面每个对照**同一批** audit logits，所以
            # 差异只在偏移怎么来的。协议
            # `notes/phase2q_batch_prior_offset_protocol.md`（SHA 52b85c5c…）。
            bpo_info = None
            if args.score_bpo:
                b_bpo = fit_batch_prior_offset(entry.logits)
                bpo_info = {
                    "R_bpo": float(balanced_accuracy(
                        entry.logits, entry.labels, b_bpo)),
                    "offset": [float(v) for v in b_bpo],
                    "hist": [float(v) for v in
                             predicted_histogram(entry.logits, b_bpo)],
                    "hist_no_offset": [float(v) for v in
                                       predicted_histogram(entry.logits, None)],
                }
                # Q5：批大小敏感性。整批之外再取几个前缀，看这方法是不是
                # 偷偷需要「一次拿到整个测试集」。报告，不设阈值。
                by_batch = {}
                for bs in (16, 32, 64):
                    if entry.logits.shape[0] >= bs:
                        accs = []
                        for start in range(0, entry.logits.shape[0] - bs + 1, bs):
                            chunk = entry.logits[start:start + bs]
                            accs.append(balanced_accuracy(
                                entry.logits, entry.labels,
                                fit_batch_prior_offset(chunk)))
                        by_batch[str(bs)] = float(np.mean(accs))
                bpo_info["R_bpo_by_batch"] = by_batch
                # Q4：故意做不平衡混合。同一批样例、同一个模型，只改**组成**。
                # 这是协议里唯一能证伪「Q1 只是我们 64/class 平衡划分的假象」的
                # 工具，所以它和 Q1 在同一次运行里算，不留到以后补。
                imbal = {}
                for ratio in (0.5, 0.7, 0.9):
                    try:
                        idx = subsample_imbalanced(entry.labels, ratio,
                                                   seed=args.seed)
                    except ValueError:
                        continue
                    sub_logits, sub_labels = entry.logits[idx], entry.labels[idx]
                    b_sub = fit_batch_prior_offset(sub_logits)
                    imbal[str(ratio)] = {
                        # 在**这个不平衡子集**上评，因为查询混合就是它。
                        "R_bpo": float(balanced_accuracy(
                            sub_logits, sub_labels, b_sub)),
                        "R_raw": float(balanced_accuracy(
                            sub_logits, sub_labels, None)),
                        "R_shr_global": float(balanced_accuracy(
                            sub_logits, sub_labels, gauge_fix(g_sub))),
                        "n": int(len(idx)),
                        "class0_share": float((sub_labels == 0).mean()),
                    }
                bpo_info["imbalanced"] = imbal

            # Phase-2S SIO：与 BPO 挂在**同一处**、读**同一批** audit logits、走
            # 同一个 balanced_accuracy 调用，所以两者之差只在偏移怎么来的。
            # 协议 notes/phase2s_sio_protocol.md（SHA 21e8963e…，含 §A1/§A2）。
            # BC 也在这里重算一遍，绝不引用它论文里的数字。
            # 只对 K_S=2 定义（协议 §4 S6），其余 scope 按构造弃权并记下覆盖率。
            sio_info = None
            if args.score_sio:
                K_S = entry.logits.shape[1]
                if K_S != 2:
                    sio_info = {"applicable": False, "K_S": int(K_S)}
                else:
                    res = sio_offset(entry.logits, seed=args.seed)
                    b_bc = bc_offset(entry.logits)
                    sio_info = {
                        "applicable": True, "K_S": 2,
                        "R_sio": float(balanced_accuracy(
                            entry.logits, entry.labels, res.offset)),
                        "R_bc": float(balanced_accuracy(
                            entry.logits, entry.labels, b_bc)),
                        "applied": bool(res.applied),
                        "reason": res.reason,
                        "tau": float(res.tau) if np.isfinite(res.tau) else None,
                        "snr": float(res.snr) if np.isfinite(res.snr) else None,
                        "rho": float(res.rho) if np.isfinite(res.rho) else None,
                        "rho_crit": (float(res.rho_crit)
                                     if np.isfinite(res.rho_crit) else None),
                        "ci": ([float(res.ci[0]), float(res.ci[1])]
                               if np.isfinite(res.ci[0]) else None),
                    }
                    # S3/S3b：偏斜混合。同一批样例、同一个模型，只改**组成**。
                    # S3b（协议 §A2.3）要求把 apply 率报在旁边，因为弃权也能让
                    # SIO−BC 变正，那是拒答而不是修好。
                    sio_imbal = {}
                    for ratio in (0.5, 0.7, 0.9):
                        try:
                            idx = subsample_imbalanced(entry.labels, ratio,
                                                       seed=args.seed)
                        except ValueError:
                            continue
                        sl, sy = entry.logits[idx], entry.labels[idx]
                        rs = sio_offset(sl, seed=args.seed)
                        sio_imbal[str(ratio)] = {
                            "R_sio": float(balanced_accuracy(sl, sy, rs.offset)),
                            "R_bc": float(balanced_accuracy(
                                sl, sy, bc_offset(sl))),
                            "R_raw": float(balanced_accuracy(sl, sy, None)),
                            "applied": bool(rs.applied),
                            "reason": rs.reason,
                            "tau": (float(rs.tau)
                                    if np.isfinite(rs.tau) else None),
                            "n": int(len(idx)),
                            "class0_share": float((sy == 0).mean()),
                        }
                    sio_info["imbalanced"] = sio_imbal
                    # S5b（协议 §A2.4）：机制判据要的是「校正量随批次类比例的
                    # 斜率」，不是 §4 S5 那个 sd 比值——sd 把采样噪声和系统偏差
                    # 混为一谈。两个校正量都存下来，斜率在 decide 阶段算。
                    # 每个 ratio 只重采样一次。之前写成调用两次 subsample_imbalanced
                    # 取 [1] 和 [0]，靠 seed 固定才恰好一致——能用但脆。
                    # bc 存的是 **offset 侧**（协议 §A2.6 的符号约定）。
                    slope_inputs = []
                    for r, v in sio_imbal.items():
                        try:
                            idx = subsample_imbalanced(entry.labels, float(r),
                                                       seed=args.seed)
                        except ValueError:
                            continue
                        b = bc_offset(entry.logits[idx])
                        slope_inputs.append({
                            "ratio": float(v["class0_share"]),
                            "tau": v["tau"],
                            "bc": float(b[1] - b[0]),
                        })
                    sio_info["slope_inputs"] = slope_inputs

            # Phase-2L §4 的 λ 闸门必须在 **update** 划分上选 λ（那部分标签已被
            # 训练消耗），否则选 λ 会吃掉留出信息。默认关闭：不传这个开关时
            # Phase-2K 的产出键集不变，收敛跑的复现检查才成立。
            # 修正后的闸门（协议 §A2.1）需要两个时点：任务**自己刚训完**那一刻
            # （peak_update），以及最后一个 stage（final_update）。差分掉各自的
            # peak，可塑性就精确抵消，剩下的才是保持。
            # 第一版只在最后一个 stage 取一次，于是量到的是"可塑性 + 保持"的混合，
            # λ 选在了可塑性上——这正是 A2 记录的缺陷。
            if (args.gate_update_eval and old.name in scorable
                    and (position == last_position or old.name == task.name)):
                update_examples = list(partitions[old.name].update)
                up_index = {label: i for i, label in enumerate(old.labels)}
                up_logits, _ = with_oom_retry(
                    lambda bs, spec=old, ex=update_examples: collect_logits(
                        model, tokenizer, ex, spec, device,
                        max_source=args.max_source, batch_size=bs),
                    batch_size=args.eval_batch_size,
                )
                up_labels = np.array([up_index[e.label] for e in update_examples])
                r_update = balanced_accuracy(up_logits, up_labels, None)
            else:
                r_update = None

            info = {
                "scope": "|".join(scope),
                "scope_size": len(scope_members[scope]),
                "scorable": old.name in scorable,
                "R_raw": float(r_raw),
                "R_orc": float(r_orc),
                "R_shr_global": float(r_glob),
                "R_shr_scoped": float(r_scoped),
                "Delta_id_global": float(r_orc - r_glob),
                "scope_recoverable": float(r_scoped - r_glob),
                "qoc": {},
            }
            if r_update is not None:
                info["R_update_raw"] = float(r_update)
            if bpo_info is not None:
                info["bpo"] = bpo_info
            if sio_info is not None:
                info["sio"] = sio_info
            if sso is not None and old.name in scorable:
                # 协议 §2：SSO 与 rehearsal 上限在**同一批** audit logits 上打分，
                # 于是 R_sso1/R_shr_scoped 之差是纯粹的「表何时拟合」之差。
                for m in sso_m_values:
                    book = sso_books[m].get(scope)
                    if book is None:
                        continue
                    if m == 1:
                        # m=1 只有一个中心，不需要路由器，也不该给它状态。
                        sub = book.offset_for(old.labels, 0)
                        info["R_sso%d" % m] = float(
                            balanced_accuracy(entry.logits, entry.labels,
                                              gauge_fix(sub)))
                        info["sso%d_radius" % m] = float(book.radius)
                    else:
                        st = states_for(old.name, "audit", True)
                        scored = score_with_codebook(
                            book, old.name, old.labels, entry.logits, entry.labels,
                            router="prototype", states=st)
                        info["R_sso%d" % m] = float(scored["accuracy"])
                        info["sso%d_radius" % m] = float(book.radius)
                        info["sso%d_route_histogram" % m] = scored["route_histogram"]
            if old.name in scorable:
                states = states_official = None
                for m in m_values:
                    book = books[m].get(scope)
                    if book is None:
                        continue
                    if states is None:
                        states = states_for(old.name, "audit", True)
                        states_official = states_for(old.name, "audit", False)
                    info["qoc"][str(m)] = {
                        router: score_with_codebook(
                            book, old.name, old.labels, entry.logits, entry.labels,
                            router=router,
                            states=(states_official
                                    if router == "prototype_official_leaky"
                                    else states))
                        for router in ("single", "confidence", "batch_margin",
                                       "prototype", "prototype_official_leaky",
                                       "oracle")
                    }
            stage["tasks"][old.name] = info

            # Phase-2S §A1.1：上一次跑（阿里云）没存 logits，导致这一相只能重训
            # 才能重打分。存下来，任何新的推理规则以后都是零 GPU 可复算。
            if args.dump_logits:
                dump_dir = Path(args.dump_logits)
                dump_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    dump_dir / f"seed{args.seed}_stage{position}_{old.name}.npz",
                    logits=entry.logits.astype(np.float32),
                    labels=entry.labels.astype(np.int16),
                    verbalizer=np.array(entry.verbalizer, dtype=object),
                    scope=np.array("|".join(scope)),
                    trained_task=np.array(task.name),
                )

        stage["scope_radii"] = {
            str(m): {"|".join(scope): {"q_m": book.radius,
                                       "members": book.members,
                                       "n_centres": int(book.centres.shape[0])}
                     for scope, book in books[m].items()}
            for m in m_values
        }
        record["stages"].append(stage)
        record["elapsed_minutes"] = round((time.time() - started) / 60.0, 2)
        (out_dir / ("qoc_seed%d.json" % args.seed)).write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

        scored = [e for e in stage["tasks"].values() if e["scorable"] and e["qoc"]]
        if scored:
            # 报最大的那个 m，而不是硬编码 "4"：smoke 跑用 --m-values 1,2 时
            # 硬编码会取空切片，日志里出现 nan 并伴随 numpy 的空切片警告。
            top_m = str(max(m_values))
            book_gains = [e["qoc"][top_m]["prototype"]["accuracy"] - e["R_shr_scoped"]
                          for e in scored if top_m in e["qoc"]]
            # 训练闸门的即时读数：刚训练完这个任务的原始准确率。欠训练时它会
            # 一直贴在 0.50 附近，早看到比跑完再看到好。
            trained = stage["tasks"].get(task.name, {})
            log("  stage %d: median scope_recoverable=%.4f  qoc%s/proto-scoped=%s"
                "  trained %s R_raw=%s"
                % (position,
                   float(np.median([e["scope_recoverable"] for e in scored])),
                   top_m,
                   ("%.4f" % float(np.median(book_gains))) if book_gains else "n/a",
                   task.name,
                   ("%.4f" % trained["R_raw"]) if trained.get("R_raw") is not None
                   else "n/a"))

    log("QOC_DONE seed=%d elapsed=%.1f min out=%s"
        % (args.seed, (time.time() - started) / 60.0, out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
