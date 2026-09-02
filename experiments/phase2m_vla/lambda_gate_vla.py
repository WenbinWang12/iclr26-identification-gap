"""Phase-2M 的 λ_a 闸门：协议 §4。

**这个文件不重新实现打分函数。** 它 `from ... import retention_score`，
用的就是 Phase-2L §A2.1 那个修正后的函数——"每个早期任务的 final update 准确率
减去它自己的 peak"，可塑性在差里抵消。复用而不是复制，是因为 Phase-2L 的结论
（正交惩罚只买到可塑性）完全依赖这个函数的语义；如果 Phase-2M 用一份抄过去的
副本，两期之间任何一处漂移都会让"同一把尺子"的说法失效。

为什么不直接改 `phase2l_additivity/lambda_gate.py`：那个文件是 Phase-2L §A3 里
λ=0.5 那次选择的**记录**。改它的校验逻辑会让"当时是用什么代码选的"不再可查。
所以这里只泛化两处 method-specific 的东西：

* 校验哪个 `cl_method` 必须出现（Phase-2L 写死 `olora`）；
* 从 `args` 的哪个键读 λ（Phase-2L 写死 `olora_lambda`）。

打分、baseline 对比、endpoint 判定全部不变。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.phase2l_additivity.lambda_gate import retention_score

#: `cl_method` -> `args` 里存 λ 的键名。新方法在这里登记即可。
LAMBDA_KEY = {"olora": "olora_lambda", "vla": "vla_lambda",
              "gfa": "gfa_lambda"}


def select(candidates: dict[str, dict], *, method: str = "vla") -> dict:
    """`candidates` 形如 {label: record}；`method` 是被调参的那个 arm 的 cl_method。

    先校验再打分——顺序是故意的。少传一个 arm 是调用方的错，报错就该指向那个错，
    而不是让 `retention_score` 先抛一个关于 stage 结构的 KeyError（Phase-2L 的
    第一版就是这个毛病）。
    """

    if method not in LAMBDA_KEY:
        raise ValueError("unknown method %r; register its lambda key in "
                         "LAMBDA_KEY first" % (method,))
    if not any(r["args"].get("cl_method") == method for r in candidates.values()):
        raise ValueError("no %s arms among candidates" % method)

    lam_key = LAMBDA_KEY[method]
    table = {}
    for label, record in candidates.items():
        got = record["args"].get("cl_method", "none")
        if got not in ("none", method):
            raise ValueError(
                "arm %r has cl_method=%r, but this gate is selecting λ for %r; "
                "mixing methods in one grid would compare two different "
                "interventions" % (label, got, method))
        table[label] = retention_score(record)
        table[label]["lambda"] = float(record["args"].get(lam_key, 0.0))
        table[label]["cl_method"] = got

    tuned = {k: v for k, v in table.items() if v["cl_method"] == method}
    # 并列时取较小的 λ：更小的正则强度是更弱的干预，平手就该偏向它。
    best = max(tuned.items(), key=lambda kv: (kv[1]["mean"], -kv[1]["lambda"]))
    baseline = next((v for v in table.values() if v["cl_method"] == "none"), None)

    # 端点判定要把 baseline 当成 **λ = 0 这个格点**，而不是栅格外的另一件事：
    # `--cl-method vla --vla-lambda 0` 与 `--cl-method none` 逐值相同（协议 §A1
    # 记录的等价性检查，24 个值全等）。所以 λ=0 是我们真的测过的格点，栅格下界
    # 是**封闭**的——只有上界无界。Phase-2L §A1 里 λ=1.0 落在上边界才是真的边界
    # 隐忧；下端最小的正 λ 被选中不是。
    ladder = sorted(v["lambda"] for v in tuned.values())
    if baseline is not None:
        ladder = sorted(set(ladder) | {0.0})
    sel_lam = best[1]["lambda"]
    by_lambda = {v["lambda"]: v["mean"] for v in tuned.values()}
    if baseline is not None:
        by_lambda.setdefault(0.0, baseline["mean"])

    # 只有上端算未封闭的端点。
    at_endpoint = len(ladder) > 1 and sel_lam == ladder[-1]
    # "还在往外走"= 选中的端点比它唯一的内侧邻居更好。协议 §4/§A1 要求把这种情况
    # 记成 limitation，且**不准**为此扩栅格：看到选择落在边上再加点，正是制造
    # tuning-on-the-signal 假象的标准做法。
    still_improving = False
    if at_endpoint:
        still_improving = best[1]["mean"] > by_lambda[ladder[-2]]

    beats = baseline is None or best[1]["mean"] > baseline["mean"]
    return {
        "method": method,
        "table": table,
        "selected_label": best[0],
        "selected_lambda": sel_lam,
        "selected_mean": best[1]["mean"],
        "baseline_mean": None if baseline is None else baseline["mean"],
        "beats_baseline": bool(beats),
        "selection_at_grid_endpoint": bool(at_endpoint),
        "endpoint_still_improving": bool(still_improving),
        "note": ("selected λ does not beat cl-method=none on update-split "
                 "retention; per protocol §4 the confirmatory run still proceeds "
                 "at this λ and the retention criterion is reported as "
                 "expected-to-fail")
                if not beats else "selected λ beats cl-method=none",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="label=path/to/qoc_seed1.json")
    parser.add_argument("--method", default="vla", choices=sorted(LAMBDA_KEY))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    candidates = {}
    for item in args.runs:
        label, _, path = item.partition("=")
        candidates[label] = json.loads(Path(path).read_text(encoding="utf-8"))
    result = select(candidates, method=args.method)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
