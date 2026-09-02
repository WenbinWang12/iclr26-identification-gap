"""Batch Prior Offset (BPO) — `notes/phase2q_batch_prior_offset_protocol.md`.

Phase-2P closed the streaming route: an offset fitted at `θ_g` and applied at
`θ_15` is 5.08 pp *worse than no offset at all* (§A1.2), and singleton scopes
isolate the cause as staleness with no aggregation, routing, or gauge confound
possible (§A1.3).  Since refitting against the current model demonstrably works
(`R_orc` 0.7229) and no old task's examples may be re-read, the only remaining
source of current-model information is **the query batch itself**.

BPO is that method, and it is deliberately the smallest thing that could work:

* it stores **zero** floats — no table, no per-task vector, no router;
* it reads **no labels** — the fitting function below has no label parameter,
  which the protocol requires be asserted by test rather than by inspection;
* it reads **no task identity** — only the scope, i.e. the prompt's option list;
* it has **no hyperparameter** — there is nothing to select, hence no gate.

What it optimises is a *surrogate*: make the predicted label histogram uniform.
Under balanced accuracy each class carries equal weight, so a drifted bias that
collapses a class is exactly what costs us, and uniformity is the label-free
proxy for "do not collapse a class".  The protocol's §1 states the confound this
creates — our audit splits are 64/class, so on a balanced batch the surrogate is
partly right for reasons that are an artefact of our own split construction —
and criterion Q4 measures it on deliberately imbalanced mixes.  Read that
criterion before quoting any number from here.

Not to be confused with Phase-2K's `batch_margin` *router*, which used batch
statistics to **select among stored centres** and was actively harmful (median
gain −0.0000, CI [−0.1172, 0]).  BPO stores no centres and selects nothing.
"""

from __future__ import annotations

import numpy as np

from experiments.phase2j_offset_conflict.offsets import gauge_fix


def predicted_histogram(logits: np.ndarray,
                        offset: np.ndarray | None = None) -> np.ndarray:
    """Fraction of the batch predicted into each of the ``K`` classes.

    Depends on ``logits`` only.  No labels: that is the point of this module.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if offset is not None:
        logits = logits + np.asarray(offset, dtype=np.float64)[None, :]
    pred = logits.argmax(axis=1)
    counts = np.bincount(pred, minlength=logits.shape[1])
    return counts / float(len(pred)) if len(pred) else counts.astype(np.float64)


def uniformity_objective(logits: np.ndarray,
                         offset: np.ndarray | None = None) -> float:
    """Negative total-variation distance from the predicted histogram to uniform.

    Higher is better, so this plugs into the same maximising coordinate search
    that `fit_offset` uses.  The maximum attainable value is 0.
    """
    hist = predicted_histogram(logits, offset)
    K = len(hist)
    if K == 0:
        return float("nan")
    tv = 0.5 * float(np.abs(hist - 1.0 / K).sum())
    return -tv


def fit_batch_prior_offset(logits: np.ndarray, *,
                           grid: int = 81, span: float = 8.0,
                           refine_rounds: int = 3) -> np.ndarray:
    """The BPO offset for one batch of same-scope queries.

    Signature note, load-bearing for the protocol: there is **no** ``labels``
    parameter, and adding one would break `tests/test_bpo.py::
    test_signature_has_no_labels`.  That test exists so "BPO uses no labels" is
    a checked property of the code rather than a claim in a paper.

    Mirrors `fit_offset`'s coordinate search so that BPO and every comparator
    differ in their *objective* only, never in their search.  For ``K == 2``
    this is an exact scan over the single contrast; for ``K > 2`` it is a
    coordinate-wise refinement of a piecewise-constant objective, so the result
    is a *fitted* offset and never a certified optimum.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("logits must be (n, K), got shape %r" % (logits.shape,))
    K = logits.shape[1]
    if logits.shape[0] == 0:
        return np.zeros(K)

    best = np.zeros(K)
    best_val = uniformity_objective(logits, best)

    lo, hi = -span, span
    for _ in range(refine_rounds):
        for k in range(K):
            for v in np.linspace(lo, hi, grid):
                trial = best.copy()
                trial[k] = v
                trial = gauge_fix(trial)
                val = uniformity_objective(logits, trial)
                if val > best_val + 1e-12:
                    best_val, best = val, trial
        width = (hi - lo) / 4.0
        lo, hi = -width, width
    return gauge_fix(best)


def stored_floats() -> int:
    """Zero, by construction.  Asserted by test, per protocol §6."""
    return 0


def subsample_imbalanced(labels: np.ndarray, ratio: float, *,
                         seed: int) -> np.ndarray:
    """Indices of a deliberately class-imbalanced subsample (criterion Q4).

    ``ratio`` is the share taken by class 0; the remainder is split evenly over
    the other classes.  Sampling is without replacement from the same audit
    examples, so Q4 changes the *mix* and nothing else — same model, same
    logits, same split.  Returns indices into ``labels``.
    """
    labels = np.asarray(labels)
    K = int(labels.max()) + 1 if labels.size else 0
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must lie in (0, 1), got %r" % (ratio,))
    if K < 2:
        # A "class mix" over fewer than two classes is not a thing, and silently
        # returning every index would make Q4 report a balanced-batch number
        # under an imbalanced label — the exact confusion Q4 exists to prevent.
        raise ValueError("need at least 2 classes to build a mix, got K=%d" % K)
    rng = np.random.default_rng(seed)
    per_class = [np.flatnonzero(labels == k) for k in range(K)]
    if any(len(idx) == 0 for idx in per_class):
        raise ValueError("every class must be present to build an imbalanced mix")

    # Size the draw by whichever class binds, so no class is oversampled.
    others = (1.0 - ratio) / max(K - 1, 1)
    caps = [len(per_class[0]) / ratio] + [len(idx) / others for idx in per_class[1:]]
    total = int(min(caps))
    take = [max(1, int(round(total * ratio)))]
    take += [max(1, int(round(total * others))) for _ in range(K - 1)]

    picked = []
    for k, n in enumerate(take):
        n = min(n, len(per_class[k]))
        picked.append(rng.choice(per_class[k], size=n, replace=False))
    out = np.concatenate(picked)
    rng.shuffle(out)
    return out
