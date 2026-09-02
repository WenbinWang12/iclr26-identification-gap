"""Phase-2V DEVELOPMENT (seeds 1-3, already burned). Zero GPU.

Question: which label-free location statistic of s = z_1 - z_0 best serves as a
*drift tracker* for the balanced-accuracy-optimal threshold t*, and is it robust to
the query batch's class marginal?

Batch calibration uses the MEAN, which is the least marginal-robust location
statistic there is -- that is exactly what Phase-2Q's Q4 control punished. This
script measures stability, tracking, and marginal robustness for a small
pre-specified family, so that Phase-2V can be frozen on the winner.

Nothing here is a result. Development only.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

A_LEVELS = (0.05, 0.10, 0.25)


def bacc(s, t, y):
    p = (s > t).astype(int)
    return float(np.mean([(p[y == c] == c).mean() for c in (0, 1) if (y == c).sum() > 0]))


def t_oracle(s, y):
    best, bt = None, 0.0
    for t in np.unique(np.concatenate([s, [s.min() - 1.0, s.max() + 1.0]])):
        a = bacc(s, float(t), y)
        if best is None or a > best:
            best, bt = a, float(t)
    return bt


def trackers(s):
    out = {"mean": float(s.mean()), "median": float(np.median(s))}
    for a in A_LEVELS:
        out["mid%02d" % int(a * 100)] = float((np.quantile(s, a) + np.quantile(s, 1 - a)) / 2.0)
    out["trim25"] = float(s[(s >= np.quantile(s, 0.25)) & (s <= np.quantile(s, 0.75))].mean())
    return out


def load(seed, root="data/phase2t_logits"):
    D = []
    for f in sorted(Path("%s/phase2s_full_s%d/logits" % (root, seed)).glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        Z = z["logits"]
        if Z.ndim != 2 or Z.shape[1] != 2:
            continue
        parts = f.name[:-4].split("_")
        D.append(
            dict(
                task="_".join(parts[2:]),
                stage=int(parts[1].replace("stage", "")),
                scope=str(z["scope"]),
                s=(Z[:, 1] - Z[:, 0]).astype(np.float64),
                y=z["labels"].astype(int),
            )
        )
    return D


def remix(d, pi, rng):
    """Subsample to class-1 fraction pi, keeping the majority class whole."""
    i0, i1 = np.where(d["y"] == 0)[0], np.where(d["y"] == 1)[0]
    if len(i0) == 0 or len(i1) == 0:
        return None
    n_maj = min(len(i0), len(i1))
    n_min = max(1, int(round(n_maj * (1 - pi) / pi)))
    if n_min > n_maj:
        return None
    keep = np.concatenate([rng.choice(i1, n_maj, replace=False), rng.choice(i0, min(n_min, len(i0)), replace=False)])
    return d["s"][keep], d["y"][keep]


def main():
    names = list(trackers(np.array([0.0, 1.0])).keys())
    print("Phase-2V development: drift trackers on seeds 1-3 (burned dev seeds)\n")

    stab = defaultdict(list)
    for seed in (1, 2, 3):
        D = load(seed)
        by_task = defaultdict(list)
        for d in D:
            by_task[d["task"]].append(d)
        for task, ds in by_task.items():
            if len(ds) < 3:
                continue
            for nm in names:
                cs = [t_oracle(d["s"], d["y"]) - trackers(d["s"])[nm] for d in ds]
                stab[nm].append(float(np.std(cs)))

    print("(1) STABILITY  sd of c = t* - stat across stages, within task (lower better)")
    for nm in sorted(stab, key=lambda k: np.median(stab[k])):
        print("    %-8s median sd %.3f   mean sd %.3f   n_tasks %d"
              % (nm, np.median(stab[nm]), np.mean(stab[nm]), len(stab[nm])))

    print("\n(2) MARGINAL ROBUSTNESS  |stat(pi=0.7) - stat(pi=0.5)| (lower better)")
    shift = defaultdict(list)
    for seed in (1, 2, 3):
        for d in load(seed):
            rng = np.random.default_rng(abs(hash((seed, d["task"], d["stage"]))) % (2**32))
            a, b = remix(d, 0.5, rng), remix(d, 0.7, rng)
            if a is None or b is None:
                continue
            ta, tb = trackers(a[0]), trackers(b[0])
            for nm in names:
                shift[nm].append(abs(tb[nm] - ta[nm]))
    for nm in sorted(shift, key=lambda k: np.median(shift[k])):
        print("    %-8s median shift %.3f   p90 %.3f" % (nm, np.median(shift[nm]), np.quantile(shift[nm], 0.9)))

    print("\n(3) END-TO-END balanced accuracy, stale batches only (stage > own stage)")
    print("    c=0 is zero stored state; c=own is the admissible per-scope estimate")
    for seed in (1, 2, 3):
        D = load(seed)
        own = {}
        for d in D:
            if d["task"] not in own or d["stage"] < own[d["task"]]["stage"]:
                own[d["task"]] = d
        c_own = {nm: {t: t_oracle(d["s"], d["y"]) - trackers(d["s"])[nm] for t, d in own.items()} for nm in names}
        sc = {nm: defaultdict(list) for nm in names}
        for nm in names:
            for t, d in own.items():
                sc[nm][d["scope"]].append(c_own[nm][t])
        c_sc = {nm: {k: (max(v) + min(v)) / 2 for k, v in sc[nm].items()} for nm in names}
        stale = [d for d in D if d["stage"] > own[d["task"]]["stage"]]
        row = {"raw": np.mean([bacc(d["s"], 0.0, d["y"]) for d in stale]),
               "oracle": np.mean([bacc(d["s"], t_oracle(d["s"], d["y"]), d["y"]) for d in stale])}
        for nm in names:
            row[nm + "|c=0"] = np.mean([bacc(d["s"], trackers(d["s"])[nm], d["y"]) for d in stale])
            row[nm + "|c=own"] = np.mean(
                [bacc(d["s"], trackers(d["s"])[nm] + c_sc[nm][d["scope"]], d["y"]) for d in stale])
        base = row["mean|c=0"]
        print("    seed %d (n=%d)  raw %.4f  BC(mean|c=0) %.4f  oracle %.4f" % (seed, len(stale), row["raw"], base, row["oracle"]))
        for nm in names:
            print("        %-8s c=0 %+6.2f pp   c=own %+6.2f pp  (vs BC)"
                  % (nm, 100 * (row[nm + "|c=0"] - base), 100 * (row[nm + "|c=own"] - base)))


if __name__ == "__main__":
    sys.exit(main())
