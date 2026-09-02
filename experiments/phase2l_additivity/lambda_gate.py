"""协议 §4 的 λ 选择闸门。

在 `update` 划分上打分——那部分标签已被训练消耗，因此选 λ 不吃任何
留出信息。**不读** risk、不读 audit、不读官方 test.json。

判据（协议 §A2.1，修正版）：seed 1，只跑前 6 个任务，取 update 划分上
**保持**最好的 λ——每个早期任务用"最后 stage 的 update 准确率减去它
自己刚训完时的 update 准确率"，可塑性因此精确抵消。并列取最小 λ。
§4 的原始判据把可塑性混进了保持，已在协议 §A2 记录。
若没有 λ 优于 `--cl-method none`，记录该事实并仍以最好的 λ 跑确认种子，
把判据 C 报为预期失败。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def retention_score(record: dict, *, n_probe: int = 5) -> dict:
    """update 划分上的**保持**：每个早期任务相对**自己的 peak** 作差。

    协议 §A2.1。第一版取的是"最后一个 stage 上前 5 个任务的 update 准确率
    均值"，那个量把**可塑性**和保持混在一起：一个让每个任务学得更好的 λ
    在它上面得分更高，即使什么都没保住。第一次确认跑就是这么把 λ 选到了
    可塑性上（peak +2.99pp、later −0.65pp、遗忘反而 +3.64pp）。

    这里每个任务减掉它自己刚训完时的 update 准确率，可塑性精确抵消，
    留下的是 update 划分上的遗忘。**越大（越不负）越好。**
    """

    stages = record["stages"]
    if not stages:
        raise ValueError("record has no stages")
    last = stages[-1]
    order = [s["trained_task"] for s in stages]
    early = order[:n_probe]

    def read(entry, name, when):
        value = entry.get("R_update_raw")
        if value is None:
            raise KeyError(
                "R_update_raw missing for %s at %s: the gate needs run_qoc run "
                "with --gate-update-eval, otherwise λ would be selected on a "
                "held-out split, which protocol §4 forbids" % (name, when))
        return float(value)

    peaks, finals = {}, {}
    for stage in stages:
        # 任务刚训完的那一刻 = 它作为 trained_task 出现的那个 stage。
        name = stage["trained_task"]
        if name in early:
            entry = stage["tasks"].get(name)
            if entry is not None and entry.get("scorable"):
                peaks[name] = read(entry, name, "its own peak")
    for name in early:
        entry = last["tasks"].get(name)
        if entry is not None and entry.get("scorable"):
            finals[name] = read(entry, name, "the final gate stage")

    scores = {n: finals[n] - peaks[n] for n in early if n in peaks and n in finals}
    if not scores:
        raise ValueError("no scorable early tasks with both peak and final "
                         "update-split measurements")
    return {"per_task": scores,
            "per_task_peak": {n: peaks[n] for n in scores},
            "per_task_final": {n: finals[n] for n in scores},
            "mean": sum(scores.values()) / len(scores),
            "n_tasks": len(scores)}


def select(candidates: dict[str, dict]) -> dict:
    """`candidates` 形如 {label: record}。返回选择结果与完整表格。"""

    # 先验证候选集构成，再打分：否则一个没有 olora 臂的调用会先在
    # `retention_score` 里因缺键炸掉，报出的错误指向数据而不是真正的问题
    # （调用方压根没给 olora 臂）。
    if not any(r["args"].get("cl_method") == "olora" for r in candidates.values()):
        raise ValueError("no olora arms among candidates")

    table = {}
    for label, record in candidates.items():
        table[label] = retention_score(record)
        table[label]["lambda"] = float(record["args"].get("olora_lambda", 0.0))
        table[label]["cl_method"] = record["args"].get("cl_method", "none")

    olora = {k: v for k, v in table.items() if v["cl_method"] == "olora"}
    best = max(olora.items(), key=lambda kv: (kv[1]["mean"], -kv[1]["lambda"]))
    baseline = next((v for v in table.values() if v["cl_method"] == "none"), None)

    beats = baseline is None or best[1]["mean"] > baseline["mean"]
    return {
        "table": table,
        "selected_label": best[0],
        "selected_lambda": best[1]["lambda"],
        "selected_mean": best[1]["mean"],
        "baseline_mean": None if baseline is None else baseline["mean"],
        "beats_baseline": bool(beats),
        # 协议 §4：不优于 none 时不改判据、不扩网格，只是把判据 C 标为预期失败。
        "note": ("selected λ does not beat cl-method=none on update-split "
                 "retention; per protocol §4 the confirmatory run proceeds at "
                 "this λ and criterion C is reported as expected-to-fail")
                if not beats else "selected λ beats cl-method=none",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="label=path/to/qoc_seed1.json 形式的多个候选")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    candidates = {}
    for item in args.runs:
        label, _, path = item.partition("=")
        candidates[label] = json.loads(Path(path).read_text(encoding="utf-8"))
    result = select(candidates)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
