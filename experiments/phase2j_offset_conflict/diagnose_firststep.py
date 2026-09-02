"""诊断：首步 argmax 与自由生成为什么在 MNLI/CB/RTE 上不一致。

背景：Phase-2J 探针在 15 个任务里有 13 个满足 `R_raw_balanced * 100 == free_gen_em`
（逐位相等），但 MNLI/CB/RTE/COPA 明显偏离。我们的整个偏移形式化建立在
「第一步 argmax 决定标签」之上，所以这个偏离必须查清是机制问题还是训练不足。

假设：某些标签的**首片不具判别性**。T5 词表里没有 `▁entailment`，所以
`entailment` 被切成 `['▁','en','tail','ment']`，首片是 id 3（裸 `▁`）——
这个片是任何词表外整词的通用前缀，不是 `entailment` 的特征。于是
「在 {neutral, ▁, contradiction} 三列上取 argmax」和「贪心解码出的字符串」
是两件不同的事。

本脚本只读 train 派生的 audit 划分，不读 test.json。默认在**未训练**的
基座模型上跑：如果未训练模型已经复现同样的偏离模式，机制就是分词/解码，
与训练步数无关。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_HOME", "/mnt/data/wenbin/iclr26/models/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from experiments.phase2i_anchored_cvar import order4_data as od  # noqa: E402


def first_piece_ids(tokenizer, labels) -> tuple[int, ...]:
    return tuple(
        int(tokenizer(label, add_special_tokens=False)["input_ids"][0])
        for label in labels
    )


@torch.no_grad()
def probe_task(model, tokenizer, task, examples, device, *, max_source, max_new_tokens,
               batch_size):
    """返回逐样本的 (首步argmax标签, 自由生成串, 真标签)。"""

    columns = list(first_piece_ids(tokenizer, task.labels))
    model.eval()
    rows = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        enc = tokenizer(
            [example.prompt for example in batch],
            max_length=max_source, truncation=True, padding=True, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        start_ids = torch.full(
            (len(batch), 1), model.config.decoder_start_token_id,
            dtype=torch.long, device=device,
        )
        logits = model(**enc, decoder_input_ids=start_ids).logits[:, 0, :]
        # 两种 argmax：受限在标签列上，以及全词表上。
        restricted = logits[:, columns].float().argmax(dim=-1).cpu().tolist()
        unrestricted = logits.float().argmax(dim=-1).cpu().tolist()
        generated = model.generate(
            **enc, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        first_gen_piece = generated[:, 1].cpu().tolist()  # 0 是 decoder_start
        for i, example in enumerate(batch):
            rows.append({
                "argmax_label": task.labels[restricted[i]],
                "argmax_vocab_id": unrestricted[i],
                "argmax_vocab_piece": tokenizer.convert_ids_to_tokens([unrestricted[i]])[0],
                "gen": decoded[i],
                "gen_first_piece_id": first_gen_piece[i],
                "gold": example.label,
            })
    return rows


def summarise(task, rows):
    labels_norm = {od.normalize_answer(label): label for label in task.labels}
    n = len(rows)
    argmax_correct = sum(od.normalized_exact_match(r["argmax_label"], r["gold"]) for r in rows)
    gen_correct = sum(od.normalized_exact_match(r["gen"], r["gold"]) for r in rows)
    agree = sum(od.normalized_exact_match(r["argmax_label"], r["gen"]) for r in rows)
    offlabel = sum(od.normalize_answer(r["gen"]) not in labels_norm for r in rows)
    # 首步受限 argmax 选中的列的 id，与生成真正走出的第一个片是否相同？
    same_first_piece = 0
    columns = first_piece_ids_cache[task.name]
    for r in rows:
        chosen = columns[task.labels.index(r["argmax_label"])]
        same_first_piece += int(chosen == r["gen_first_piece_id"])
    top_gen = collections.Counter(od.normalize_answer(r["gen"]) for r in rows).most_common(6)
    return {
        "task": task.name,
        "n": n,
        "labels": list(task.labels),
        "label_first_pieces": list(columns),
        "argmax_accuracy": round(argmax_correct / n, 4),
        "free_gen_em": round(100.0 * gen_correct / n, 4),
        "argmax_gen_agreement": round(agree / n, 4),
        "gen_offlabel_fraction": round(offlabel / n, 4),
        "gen_first_piece_matches_argmax_column": round(same_first_piece / n, 4),
        "top_generations": top_gen,
    }


first_piece_ids_cache: dict[str, tuple[int, ...]] = {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/mnt/data/wenbin/iclr26/data/order4")
    parser.add_argument("--model", default="google-t5/t5-large")
    parser.add_argument("--out", default="/mnt/data/wenbin/iclr26/runs/phase2j_firststep_diag.json")
    parser.add_argument("--tasks", default="MNLI,CB,RTE,COPA,AGNews,DBpedia,WiC,SST-2")
    parser.add_argument("--cap-per-class", type=int, default=200)
    parser.add_argument("--risk-per-class", type=int, default=16)
    parser.add_argument("--audit-per-class", type=int, default=32)
    parser.add_argument("--max-source", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    from transformers import AutoTokenizer, T5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(args.model, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    wanted = [name.strip() for name in args.tasks.split(",") if name.strip()]
    by_name = {task.name: task for task in od.ORDER4_TASKS}
    report = {"model": args.model, "dtype": args.dtype, "trained": False, "tasks": []}

    # 先把跨任务的首片碰撞列出来 —— 这是共享偏移真正作用的坐标。
    collisions: dict[int, list[str]] = {}
    for task in od.ORDER4_TASKS:
        pieces = first_piece_ids(tokenizer, task.labels)
        first_piece_ids_cache[task.name] = pieces
        for label, piece in zip(task.labels, pieces):
            collisions.setdefault(piece, []).append("%s/%s" % (task.name, label))
    report["first_piece_collisions"] = {
        str(piece): {
            "piece": tokenizer.convert_ids_to_tokens([piece])[0],
            "claimants": owners,
        }
        for piece, owners in sorted(collisions.items())
        if len(owners) > 1
    }

    for name in wanted:
        task = by_name[name]
        # 稀有类自适应（同 run_probe 的修正 A3）：CB 的 `neutral` 只有 16 条。
        train = od.load_official_examples(args.data_root, name, "train", verify=False)
        counts: dict[str, int] = {}
        for example in train:
            counts[example.label] = counts.get(example.label, 0) + 1
        rarest = min(counts.values())
        risk = max(1, min(args.risk_per_class, (rarest - 1) // 3))
        audit = max(1, min(args.audit_per_class, (rarest - 1) - risk))
        parts = od.prepare_task_partitions(
            args.data_root, name,
            cap_per_class=args.cap_per_class,
            risk_per_class=risk,
            audit_per_class=audit,
            seed=args.seed,
        )
        rows = probe_task(
            model, tokenizer, task, list(parts.audit), device,
            max_source=args.max_source, max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        entry = summarise(task, rows)
        entry["examples"] = rows[:8]
        entry["budget"] = {"rarest": rarest, "risk": risk, "audit": audit}
        entry["sample_prompt"] = list(parts.audit)[0].prompt[:600]
        report["tasks"].append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "examples"},
                         ensure_ascii=False), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print("DIAG_DONE -> %s" % args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
