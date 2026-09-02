"""Phase-2J 偏移冲突探针：在真实 Order-4 任务流上测量识别间隙 Δ_id。

按 `notes/phase2j_offset_conflict_probe_protocol.md`（§1-§6 冻结，附修正 A1/A2）执行：

* T5-large + LoRA r=8 on q/v，官方提示词，200/class 上限，官方全局 batch 64。
* 训练完整的 15 任务流；每个阶段结束后重评所有已见任务。
* 每个任务用三种方式打分：
  - `R_raw`  无偏移
  - `R_orc`  只为该任务拟合的偏移（需要任务 ID，仅作上界参考）
  - `R_shr`  所有已见任务共享的单一偏移（不需要任务 ID）
  Δ_id = R_orc − R_shr。
* 共享偏移按**首片 id**索引，不按标签字符串（修正 A1.3）。
* Yelp/Amazon 照常训练，但不计入 Δ_id（修正 A1.1：它们的
  `very negative` / `very positive` 首片相同，任何首步偏移都无法区分）。
* 偏移在 train 派生的 risk 划分上拟合，在互不相交的 audit 划分上报告。
  `test.json` 完全不读。

同时记录自由生成的官方 EM，因为协议 §2 把 `R_raw` 定义为自由生成的 EM，
而偏移只作用于第一步 argmax；两者在此设定下应当一致，但这必须验证而非假定。
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
# 邻居作业的显存占用会波动，碎片整理让我们在 3–4 GiB 的余量里更能站住。
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.phase2i_anchored_cvar import order4_data as od  # noqa: E402
from experiments.phase2j_offset_conflict.offsets import (  # noqa: E402
    TaskLogits,
    balanced_accuracy,
    conflict_statistics,
    fit_offset,
    fit_shared_offset,
    gauge_fix,
    natural_accuracy,
)


def log(message: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


def with_oom_retry(function, *, batch_size: int, floor: int = 1):
    """OOM 时对半降 batch 重试，而不是让整跑作废。

    共享机器上邻居作业的显存占用会波动，一次 16 MiB 的分配失败不该毁掉
    已经跑了几十分钟的任务流。返回 (结果, 实际生效的 batch)。
    """

    current = batch_size
    while True:
        try:
            return function(current), current
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current <= floor:
                raise
            current = max(floor, current // 2)
            log("OOM -> 重试，batch 降为 %d" % current)


def first_piece_ids(tokenizer, labels) -> tuple[int, ...]:
    """每个标签第一个解码片的 id —— 偏移真正作用的坐标。"""

    return tuple(
        int(tokenizer(label, add_special_tokens=False)["input_ids"][0])
        for label in labels
    )


def precondition_ok(tokenizer, labels) -> bool:
    """任务内各标签首片必须互不相同，否则首步 argmax 无法区分它们。"""

    pieces = first_piece_ids(tokenizer, labels)
    return len(set(pieces)) == len(pieces)


@torch.no_grad()
def collect_logits(model, tokenizer, examples, task, device, *,
                   max_source: int, batch_size: int) -> np.ndarray:
    """返回 (n, K)：每个样本在该任务各标签首片上的第一步 logits。"""

    columns = list(first_piece_ids(tokenizer, task.labels))
    model.eval()
    chunks = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        enc = tokenizer(
            [example.prompt for example in batch],
            max_length=max_source,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        start_ids = torch.full(
            (len(batch), 1),
            model.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        logits = model(**enc, decoder_input_ids=start_ids).logits[:, 0, :]
        chunks.append(logits[:, columns].float().cpu())
    return torch.cat(chunks, dim=0).numpy()


@torch.no_grad()
def free_generation_report(model, tokenizer, examples, task, device, *,
                           max_source: int, max_new_tokens: int,
                           batch_size: int) -> dict:
    """官方口径的自由生成 normalized EM，外加首步一致性的分解。

    协议 §2 把 `R_raw` 定义为自由生成 EM，而偏移只作用于第一步 argmax。两者
    是否一致不能假定。这里同时记录：

    * `argmax_gen_agreement` —— 受限首步 argmax 选的标签，与解码串是否相同；
    * `gen_first_piece_matches_argmax_column` —— 生成真正走出的第一个片，是否
      正好是受限 argmax 选中的那一列；
    * `gen_offlabel_fraction` —— 生成串根本不是任何合法标签的比例；
    * `offlabel_with_label_prefix` —— 脱靶的那些里，**首步选的标签仍是解码串的
      前缀**的比例。这一项区分两种完全不同的失效：
        (i) 前缀成立 —— 首步决定是对的，只是没有及时停下（EOS 失效），
            这与偏移无关，属于解码终止问题；
        (ii) 前缀不成立 —— 解码真的走向了别的标签，那「首步决定标签」
            在这个任务上就是假的，形式化必须改。
    * `gen_emitted_eos` —— 是否在 max_new_tokens 内产生了 EOS。
    """

    columns = list(first_piece_ids(tokenizer, task.labels))
    labels_norm = {od.normalize_answer(label) for label in task.labels}
    model.eval()
    predictions, references = [], []
    agree = same_piece = offlabel = 0
    # 按「首步 argmax 选了哪个标签」分组统计脱靶率：这能区分两种失效，
    # 一是首步选错（我们的形式化该负责），二是首步选对但后续片走偏
    # （分词导致的解码失效，与偏移无关）。
    by_label = {label: {"n": 0, "offlabel": 0, "agree": 0, "prefix": 0}
                for label in task.labels}
    prefix_hits = no_eos = 0
    eos_id = model.config.eos_token_id
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        enc = tokenizer(
            [example.prompt for example in batch],
            max_length=max_source,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        start_ids = torch.full(
            (len(batch), 1), model.config.decoder_start_token_id,
            dtype=torch.long, device=device,
        )
        logits = model(**enc, decoder_input_ids=start_ids).logits[:, 0, :]
        chosen = logits[:, columns].float().argmax(dim=-1).cpu().tolist()
        generated = model.generate(
            **enc, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False
        )
        # generated[:, 0] 是 decoder_start，真正的第一个生成片在列 1。
        first_pieces = generated[:, 1].cpu().tolist()
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        rows = generated.cpu().tolist()
        for i, example in enumerate(batch):
            predictions.append(decoded[i])
            references.append(example.label)
            argmax_label = task.labels[chosen[i]]
            hit = int(od.normalized_exact_match(argmax_label, decoded[i]))
            off = int(od.normalize_answer(decoded[i]) not in labels_norm)
            agree += hit
            same_piece += int(columns[chosen[i]] == first_pieces[i])
            offlabel += off
            # 前缀判定在 normalize 之后做，和 EM 用同一套口径。
            is_prefix = od.normalize_answer(decoded[i]).startswith(
                od.normalize_answer(argmax_label)
            )
            if off and is_prefix:
                prefix_hits += 1
            if eos_id is not None and eos_id not in rows[i]:
                no_eos += 1
            bucket = by_label[argmax_label]
            bucket["n"] += 1
            bucket["agree"] += hit
            bucket["offlabel"] += off
            bucket["prefix"] += int(off and is_prefix)
    n = len(predictions)
    return {
        "free_gen_em": od.normalized_em(predictions, references),
        "argmax_gen_agreement": round(agree / n, 4),
        "gen_first_piece_matches_argmax_column": round(same_piece / n, 4),
        "gen_offlabel_fraction": round(offlabel / n, 4),
        # 脱靶样本中「首步标签仍是前缀」的份额：接近 1 表示 EOS 失效，
        # 接近 0 表示解码真的换了标签。
        "offlabel_with_label_prefix": (round(prefix_hits / offlabel, 4)
                                       if offlabel else None),
        "gen_no_eos_fraction": round(no_eos / n, 4),
        "gen_offlabel_by_argmax_label": {
            label: {
                "n": stats["n"],
                "offlabel_fraction": (round(stats["offlabel"] / stats["n"], 4)
                                      if stats["n"] else None),
                "offlabel_prefix_fraction": (round(stats["prefix"] / stats["offlabel"], 4)
                                             if stats["offlabel"] else None),
                "npieces": len(tokenizer(label, add_special_tokens=False)["input_ids"]),
            }
            for label, stats in by_label.items()
        },
        "gen_examples": predictions[:6],
    }


def train_one_task(model, tokenizer, examples, device, *, args, optimizer) -> list[float]:
    """在一个任务上训练 `args.epochs` 轮，返回每个全局 step 的损失。

    **`epochs` 之前是个死参数。**`run_qoc.py` 声明了 `--epochs` 却从没读过，
    本函数只扫一遍数据。配上 `cap_per_class=200`、有效 batch 64，结果是每个
    任务只有 3–16 个梯度步，WiC/COPA/QQP 的原始准确率一直贴着 0.50 ——
    模型没学会任务，"遗忘"就没有语义，判据 2 也就没有可恢复的东西。
    这里让 `epochs` 真正生效；每轮重新打乱（种子按轮次派生，保持确定性）。
    """

    model.train()
    epochs = max(1, int(getattr(args, "epochs", 1) or 1))
    ordered: list = []
    for epoch in range(epochs):
        # 每轮不同的置换，但由 (seed, epoch) 决定，故整体仍可复现。
        order = np.random.default_rng([args.seed, epoch]).permutation(len(examples))
        ordered.extend(examples[i] for i in order)
    micro = args.batch_size
    per_step = micro * args.grad_accum
    losses = []
    for start in range(0, len(ordered), per_step):
        window = ordered[start : start + per_step]
        if not window:
            break
        # OOM 时把 micro-batch 拆得更细，但**有效 batch 不变**：损失始终按
        # 本 step 的总样本数归一，所以梯度与未降 batch 时一致。降的是显存
        # 峰值，不是优化语义。
        while True:
            optimizer.zero_grad(set_to_none=True)
            pieces = [window[i : i + micro] for i in range(0, len(window), micro)]
            try:
                total = 0.0
                for batch in pieces:
                    enc = tokenizer(
                        [example.prompt for example in batch],
                        max_length=args.max_source,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    lab = tokenizer(
                        [example.label for example in batch],
                        max_length=args.max_target,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    label_ids = lab["input_ids"].clone()
                    label_ids[label_ids == tokenizer.pad_token_id] = -100
                    enc = {k: v.to(device) for k, v in enc.items()}
                    # 按样本数加权，保证与 micro 无关
                    scale = len(batch) / len(window)
                    loss = model(**enc, labels=label_ids.to(device)).loss * scale
                    loss.backward()
                    total += loss.detach().item()
                break
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if micro <= 1:
                    raise
                micro = max(1, micro // 2)
                log("训练 OOM -> micro-batch 降为 %d（有效 batch 不变）" % micro)
        optimizer.step()
        losses.append(total)
    return losses


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/mnt/data/wenbin/iclr26/data/order4")
    parser.add_argument("--model", default="google-t5/t5-large")
    parser.add_argument("--out", default="/mnt/data/wenbin/iclr26/runs/phase2j_probe")
    parser.add_argument("--n-tasks", type=int, default=15)
    parser.add_argument("--cap-per-class", type=int, default=200)
    parser.add_argument("--risk-per-class", type=int, default=64)
    parser.add_argument("--audit-per-class", type=int, default=64)
    # 最稀有类少于该数的任务不参与 Δ_id 打分：偏移拟合与打分都会被那一类的
    # 极小样本量支配，测出来的是估计噪声而非冲突。CB 的 neutral 只有 16 条。
    parser.add_argument("--min-rare-class", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    # 共享机器上另一个作业会占到 ~75 GiB，只剩 3–4 GiB。评测 batch 是主要的
    # 峰值来源（512 token 的注意力矩阵），保守取 4。
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-source", type=int, default=512)
    parser.add_argument("--max-target", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--seed", type=int, default=1)
    # 自由生成很慢且只用于验证「首步 argmax == 生成结果」，默认只在最后一个
    # 阶段做一次。设为 0 可完全跳过。
    parser.add_argument("--free-gen-stages", default="last",
                        choices=["last", "all", "none"])
    return parser.parse_args()


def main() -> int:
    args = build_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    if args.cap_per_class <= args.risk_per_class + args.audit_per_class:
        raise SystemExit(
            "cap-per-class (%d) must exceed risk (%d) + audit (%d), otherwise no "
            "update examples remain" % (args.cap_per_class, args.risk_per_class,
                                        args.audit_per_class)
        )

    stream = list(od.ORDER4_TASKS[: args.n_tasks])
    scorable, excluded = [], []
    for task in stream:
        (scorable if precondition_ok(tokenizer, task.labels) else excluded).append(task.name)
    log("stream=%d scorable=%d excluded_by_precondition=%s"
        % (len(stream), len(scorable), excluded))

    # 稀有类是硬约束，不是可调项。CB 的 `neutral` 在官方 train 里只有 16 条，
    # 所以固定的 risk=audit=64 对 CB 不可行。做法：按每个任务最稀有的类
    # 自适应缩小该任务的 risk/audit，并把实际用到的数额记录下来；不足以
    # 同时留出 risk、audit 和 update 的任务被明确标为不可打分，而不是让它
    # 悄悄崩掉或者悄悄用一个不同的预算。
    partitions, budgets, undersized = {}, {}, []
    for task in stream:
        examples = od.load_official_examples(
            args.data_root, task.name, "train", verify=False
        )
        counts: dict[str, int] = {}
        for example in examples:
            counts[example.label] = counts.get(example.label, 0) + 1
        rarest = min(counts.values())
        # 至少给 update 留一份：risk + audit ≤ rarest − 1
        allowance = max(1, min(args.risk_per_class, (rarest - 1) // 2))
        audit_allowance = max(1, min(args.audit_per_class, (rarest - 1) - allowance))
        if rarest < args.min_rare_class:
            undersized.append({"task": task.name, "rarest_class_count": rarest})
        budgets[task.name] = {
            "rarest_class_count": rarest,
            "risk_per_class": allowance,
            "audit_per_class": audit_allowance,
        }
        parts = od.prepare_task_partitions(
            args.data_root, task.name,
            cap_per_class=args.cap_per_class,
            risk_per_class=allowance,
            audit_per_class=audit_allowance,
            seed=args.seed,
        )
        parts.assert_disjoint()
        partitions[task.name] = parts
    for entry in undersized:
        name = entry["task"]
        if name in scorable:
            scorable.remove(name)
            excluded.append(name)
        log("task %s excluded: rarest class has only %d train examples (< %d)"
            % (name, entry["rarest_class_count"], args.min_rare_class))

    record = {
        "args": vars(args),
        "scorable_tasks": scorable,
        "excluded_tasks": excluded,
        "partition_budgets": budgets,
        "undersized_tasks": undersized,
        "official_commit": od.OFFICIAL_COMMIT,
        "stages": [],
    }
    log("final scorable=%d excluded=%s" % (len(scorable), excluded))
    post_raw: dict[str, float] = {}
    started = time.time()

    for position, task in enumerate(stream, start=1):
        losses = train_one_task(
            model, tokenizer, list(partitions[task.name].update), device,
            args=args, optimizer=optimizer,
        )
        log("task %d/%d %s trained steps=%d loss %.4f -> %.4f"
            % (position, len(stream), task.name, len(losses),
               losses[0] if losses else float("nan"),
               losses[-1] if losses else float("nan")))

        seen = stream[:position]
        # 每个已见任务：在 risk 划分上拟合偏移，在 audit 划分上打分。
        risk_logits, audit_logits, per_task_offsets = {}, {}, {}
        for old in seen:
            index = {label: i for i, label in enumerate(old.labels)}
            for split, store in (("risk", risk_logits), ("audit", audit_logits)):
                examples = list(getattr(partitions[old.name], split))
                logits, _ = with_oom_retry(
                    lambda bs: collect_logits(
                        model, tokenizer, examples, old, device,
                        max_source=args.max_source, batch_size=bs,
                    ),
                    batch_size=args.eval_batch_size,
                )
                labels = np.array([index[example.label] for example in examples])
                # 关键：verbalizer 用首片 id 而不是标签字符串，这样跨任务共享
                # 才在正确的坐标上耦合（修正 A1.3）。
                pieces = tuple(str(p) for p in first_piece_ids(tokenizer, old.labels))
                store[old.name] = TaskLogits(old.name, pieces, logits, labels)
            per_task_offsets[old.name] = fit_offset(
                risk_logits[old.name].logits, risk_logits[old.name].labels
            )

        # 单一共享偏移，只在可打分任务上拟合：让不满足前提的任务参与拟合会
        # 用一个无法满足的约束污染共享向量。
        fit_pool = [risk_logits[old.name] for old in seen if old.name in scorable]
        shared = fit_shared_offset(fit_pool) if fit_pool else {}

        # 判据 2 需要一个「没有泛化误差」的共享偏移作为对照：直接在 audit 上
        # 拟合。它是被污染的（在打分集上拟合），只用于回答「Δ_id 是冲突还是
        # 拟合误差」这一个问题，绝不作为结果上报。
        audit_pool = [audit_logits[old.name] for old in seen if old.name in scorable]
        shared_oracle = fit_shared_offset(audit_pool) if audit_pool else {}

        stage = {"position": position, "trained_task": task.name,
                 "n_steps": len(losses), "tasks": {}}
        for old in seen:
            audit = audit_logits[old.name]
            offset = per_task_offsets[old.name]
            sub = np.array([shared.get(token, 0.0) for token in audit.verbalizer])
            # 注意：R_orc 的偏移在 risk 上拟合、在 audit 上打分，所以它**可以
            # 低于** R_raw。这不是 bug，而是理论笔记 §5 要求的诚实口径：
            # 「oracle」指的是「只为该任务拟合」，不是「在打分集上作弊」。
            # 真正无泛化误差的上界另记为 R_orc_audit_fit，仅作参考。
            r_orc_audit = balanced_accuracy(
                audit.logits, audit.labels,
                gauge_fix(fit_offset(audit.logits, audit.labels)),
            )
            entry = {
                "scorable": old.name in scorable,
                "R_orc_audit_fit": r_orc_audit,
                "R_raw_balanced": balanced_accuracy(audit.logits, audit.labels, None),
                "R_orc_balanced": balanced_accuracy(audit.logits, audit.labels,
                                                    gauge_fix(offset)),
                "R_shr_balanced": balanced_accuracy(audit.logits, audit.labels,
                                                    gauge_fix(sub)),
                "R_raw_natural": natural_accuracy(audit.logits, audit.labels, None),
                "R_orc_natural": natural_accuracy(audit.logits, audit.labels,
                                                  gauge_fix(offset)),
                "R_shr_natural": natural_accuracy(audit.logits, audit.labels,
                                                  gauge_fix(sub)),
                "fitted_offset": [float(v) for v in gauge_fix(offset)],
                "first_pieces": [int(p) for p in first_piece_ids(tokenizer, old.labels)],
            }
            entry["Delta_id_balanced"] = entry["R_orc_balanced"] - entry["R_shr_balanced"]
            entry["shallow_gap_balanced"] = entry["R_orc_balanced"] - entry["R_raw_balanced"]
            # 对照量（在 audit 上拟合共享偏移，故无泛化误差）：判据 2 专用
            sub_oracle = np.array(
                [shared_oracle.get(token, 0.0) for token in audit.verbalizer]
            )
            r_shr_oracle = balanced_accuracy(
                audit.logits, audit.labels, gauge_fix(sub_oracle)
            )
            entry["R_shr_oracle_fit"] = r_shr_oracle
            entry["Delta_id_oracle_shared"] = entry["R_orc_balanced"] - r_shr_oracle
            # κ（理论笔记 §5、Prop 2b）是「偏移移动 t 时翻转样本比例」的下界
            # 斜率，需要完整的裕度分布才能估计斜率，两个分位点不够。这里落盘
            # 完整分位数网格；同时保留 mean/p10 以兼容已跑的 seed。
            top2 = np.sort(audit.logits, axis=1)[:, ::-1]
            margins = top2[:, 0] - top2[:, 1]
            entry["margin_mean"] = float(np.mean(margins))
            entry["margin_p10"] = float(np.percentile(margins, 10))
            grid = list(range(1, 21)) + [25, 30, 40, 50, 75, 100]
            entry["margin_quantiles"] = {
                str(q): float(np.percentile(margins, q)) for q in grid
            }
            # 真正需要的量：在偏移沿最坏方向移动 t 后会翻转的样本比例。对二元
            # 任务这恰好是裕度 ≤ t 的比例；多类时是一个下界（只有 top-2 之间
            # 的翻转被计入），故显式标为下界。
            entry["flip_fraction_lower_bound"] = {
                ("%.2f" % t): float(np.mean(margins <= t))
                for t in (0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
            }
            stage["tasks"][old.name] = entry
            if old.name == task.name:
                post_raw[old.name] = entry["R_raw_balanced"]
            entry["R_post_raw"] = post_raw.get(old.name)

        stage["shared_offset"] = {str(k): float(v) for k, v in shared.items()}
        stage["shared_offset_oracle_fit"] = {
            str(k): float(v) for k, v in shared_oracle.items()
        }
        # 冲突统计覆盖**所有**已见任务，包括不可打分的那些：一个任务的偏移
        # 需求是否与别人冲突，与它自己的分数是否可信无关。CB 的 neutral 只有
        # 16 条训练样本（修正 A3），所以 CB 的 Δ_id 不上报，但它确实在争夺
        # 共享偏移，把它从 ω̂ 里抹掉会低估冲突。同时单独给出仅可打分任务的
        # 版本，供判据 3 使用，避免用噪声大的任务撑起相关性。
        all_verbalizers = {
            old.name: tuple(str(p) for p in first_piece_ids(tokenizer, old.labels))
            for old in seen
        }
        stage["conflict"] = conflict_statistics(per_task_offsets, all_verbalizers)
        stage["conflict_scorable_only"] = conflict_statistics(
            {name: offset for name, offset in per_task_offsets.items()
             if name in scorable},
            {name: pieces for name, pieces in all_verbalizers.items()
             if name in scorable},
        )

        gaps = [e["Delta_id_balanced"] for name, e in stage["tasks"].items()
                if e["scorable"] and name != task.name]
        stage["median_Delta_id_old_tasks"] = float(np.median(gaps)) if gaps else None
        log("  stage %d: median Δ_id over %d old scorable tasks = %s"
            % (position, len(gaps),
               "n/a" if not gaps else "%.4f" % stage["median_Delta_id_old_tasks"]))

        if args.free_gen_stages == "all" or (
            args.free_gen_stages == "last" and position == len(stream)
        ):
            for old in seen:
                report, _ = with_oom_retry(
                    lambda bs, spec=old: free_generation_report(
                        model, tokenizer, list(partitions[spec.name].audit), spec,
                        device, max_source=args.max_source,
                        max_new_tokens=args.max_new_tokens, batch_size=bs,
                    ),
                    batch_size=args.eval_batch_size,
                )
                stage["tasks"][old.name].update(report)

        record["stages"].append(stage)
        record["elapsed_minutes"] = round((time.time() - started) / 60.0, 2)
        (out_dir / ("probe_seed%d.json" % args.seed)).write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )

    log("PROBE_DONE seed=%d elapsed=%.1f min out=%s"
        % (args.seed, (time.time() - started) / 60.0, out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())