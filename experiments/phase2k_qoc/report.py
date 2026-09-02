"""把 QOC 结果整理成论文口径的表，包含所有必须并列报告的项。

刻意做成一次性把「好数字」和「削弱它的数字」放在同一张表里，这样写正文时
不可能只挑一半。三张表：

* 表 1 识别间隙的分解（§5）：全局共享 → 域划分 → 码本 → per-task oracle。
* 表 2 q_m 与最差任务损失按域、按 m（判据 3 的证据）。
* 表 3 路由：剥离视图 / 泄漏视图 / 退化的 confidence / batch_margin / oracle 上界。

一律用首步受限 argmax 平衡准确率（协议 A4.5）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.phase2j_offset_conflict.decide import cluster_bootstrap_ci

ROUTERS = ("single", "confidence", "batch_margin", "prototype",
           "prototype_official_leaky", "oracle")


def load(runs: Path, seeds):
    out = []
    for seed in seeds:
        path = runs / ("qoc_seed%d.json" % seed)
        if path.exists():
            out.append(json.loads(path.read_text(encoding="utf-8")))
    if not out:
        raise SystemExit("no records under %s" % runs)
    return out


def final_rows(records, m: str):
    """每个 seed 最后一个阶段的所有老任务（排除刚训完的那个）。"""
    rows = []
    for record in records:
        stage = record["stages"][-1]
        for name, entry in stage["tasks"].items():
            if name == stage["trained_task"]:
                continue
            rows.append((record["args"]["seed"], name, entry, stage))
    return rows


def table1(records, m: str) -> str:
    rows = final_rows(records, m)
    by_task: dict[str, dict[str, list[float]]] = {}
    for _, name, entry, _ in rows:
        slot = by_task.setdefault(name, {k: [] for k in
                                        ("raw", "glob", "scoped", "qoc", "orc")})
        slot["raw"].append(entry["R_raw"])
        slot["glob"].append(entry["R_shr_global"])
        slot["scoped"].append(entry["R_shr_scoped"])
        slot["orc"].append(entry["R_orc"])
        block = entry.get("qoc", {}).get(m)
        slot["qoc"].append(block["prototype"]["accuracy"] if block else float("nan"))

    lines = ["表 1  识别间隙的分解（首步受限 argmax 平衡准确率，%d seeds，"
             "m=%s，prototype 路由为剥离视图）" % (len(records), m),
             "%-10s %7s %7s %7s %7s %7s   %8s %8s" %
             ("task", "raw", "global", "scoped", "QOC", "oracle",
              "scope+", "book+")]
    for name in sorted(by_task):
        s = by_task[name]
        mean = {k: float(np.nanmean(v)) for k, v in s.items()}
        lines.append("%-10s %7.3f %7.3f %7.3f %7.3f %7.3f   %+8.3f %+8.3f" % (
            name, mean["raw"], mean["glob"], mean["scoped"], mean["qoc"],
            mean["orc"], mean["scoped"] - mean["glob"],
            mean["qoc"] - mean["scoped"]))

    scope_gain = {name: [] for name in by_task}
    book_gain = {name: [] for name in by_task}
    for _, name, entry, _ in rows:
        block = entry.get("qoc", {}).get(m)
        if not block:
            continue
        scope_gain[name].append(entry["R_shr_scoped"] - entry["R_shr_global"])
        book_gain[name].append(block["prototype"]["accuracy"] - entry["R_shr_scoped"])
    scope_gain = {k: v for k, v in scope_gain.items() if v}
    book_gain = {k: v for k, v in book_gain.items() if v}
    if scope_gain:
        p, lo, hi = cluster_bootstrap_ci(scope_gain)
        lines.append("  域划分增益   中位 %+.4f  95%%CI [%+.4f, %+.4f]" % (p, lo, hi))
    if book_gain:
        p, lo, hi = cluster_bootstrap_ci(book_gain)
        lines.append("  码本增益     中位 %+.4f  95%%CI [%+.4f, %+.4f]  "
                     "（CI 不含 0 才算买到东西）" % (p, lo, hi))
    return "\n".join(lines)


def table2(records) -> str:
    lines = ["", "表 2  q_m 与最差任务损失（按域、按 m；判据 3 的证据）",
             "%-34s %3s %7s %10s %8s" %
             ("scope", "m", "q_m", "worstLoss", "centres")]
    agg: dict[tuple[str, str], dict[str, list[float]]] = {}
    for record in records:
        stage = record["stages"][-1]
        for m, scopes in stage["scope_radii"].items():
            for scope, info in scopes.items():
                losses = [
                    entry["R_orc"] - entry["qoc"][m]["oracle"]["accuracy"]
                    for name, entry in stage["tasks"].items()
                    if name != stage["trained_task"] and entry["scope"] == scope
                    and m in entry.get("qoc", {})
                ]
                if not losses:
                    continue
                slot = agg.setdefault((scope, m), {"q": [], "loss": [], "c": []})
                slot["q"].append(info["q_m"])
                slot["loss"].append(max(losses))
                slot["c"].append(info["n_centres"])
    for (scope, m) in sorted(agg, key=lambda k: (k[0], int(k[1]))):
        slot = agg[(scope, m)]
        lines.append("%-34s %3s %7.4f %10.4f %8.1f" % (
            scope[:34], m, float(np.mean(slot["q"])), float(np.mean(slot["loss"])),
            float(np.mean(slot["c"]))))
    return "\n".join(lines)


def table3(records, m: str) -> str:
    lines = ["", "表 3  路由（m=%s）。`*_leaky` 读得到提示词里的 `Dataset:` 名，"
             "**不构成证据**；`oracle` 只作上界；`batch_margin` 带直推假设。" % m,
             "%-10s %s" % ("task", "  ".join("%-11s" % r[:11] for r in ROUTERS)),
             "%-10s %s" % ("", "  ".join("%-11s" % "acc/match" for _ in ROUTERS))]
    rows = final_rows(records, m)
    by_task: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for _, name, entry, _ in rows:
        block = entry.get("qoc", {}).get(m)
        if not block or entry["scope_size"] < 2:
            continue
        slot = by_task.setdefault(name, {r: [] for r in ROUTERS})
        for r in ROUTERS:
            match = block[r]["route_matches_oracle"]
            slot[r].append((block[r]["accuracy"],
                            float("nan") if match is None else match))
    for name in sorted(by_task):
        cells = []
        for r in ROUTERS:
            vals = by_task[name][r]
            acc = float(np.mean([v[0] for v in vals]))
            mat = float(np.nanmean([v[1] for v in vals]))
            cells.append("%-11s" % ("%.3f/%.2f" % (acc, mat)))
        lines.append("%-10s %s" % (name, "  ".join(cells)))
    # 每个路由的整体中位增益，方便正文引用。
    lines.append("")
    for r in ROUTERS:
        gains = {}
        for _, name, entry, _ in rows:
            block = entry.get("qoc", {}).get(m)
            if not block:
                continue
            gains.setdefault(name, []).append(
                block[r]["accuracy"] - entry["R_shr_scoped"])
        if gains:
            p, lo, hi = cluster_bootstrap_ci(gains)
            lines.append("  %-26s 中位增益 %+.4f  95%%CI [%+.4f, %+.4f]"
                         % (r, p, lo, hi))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="/mnt/data/wenbin/iclr26/runs/phase2k_qoc_full")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--m", default="4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    records = load(Path(args.runs), [int(s) for s in args.seeds.split(",")])
    text = "\n".join([
        "QOC 结果  seeds=%s  official_commit=%s"
        % ([r["args"]["seed"] for r in records], records[0]["official_commit"]),
        "excluded (前提不满足或稀有类过小): %s" % records[0]["excluded_tasks"],
        "单任务域 (Prop 1 预测 Δ_id=0): %s" % records[0]["singleton_scopes"],
        "",
        table1(records, args.m), table2(records), table3(records, args.m),
    ])
    print(text)
    target = Path(args.out) if args.out else Path(args.runs) / "report.txt"
    target.write_text(text, encoding="utf-8")
    print("\n-> %s" % target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
