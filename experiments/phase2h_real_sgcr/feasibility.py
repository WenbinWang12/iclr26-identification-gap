"""Phase-2H feasibility gates F0-F3 (notes/phase2h_real_sgcr_protocol.md).

STATUS: development scaffold.  Each gate is a pure function of already-computed
inputs (frozen signatures, evaluator-only source labels, arm metrics) and returns
a structured verdict with the exact protocol thresholds.  SGCR may not advance to
a fresh confirmation unless ALL gates pass; run_development calls these and stops
on the first failure.

The source labels used here are EVALUATOR-ONLY diagnostics (F1 probe / matching,
F2 oracle).  Their weights or labels are never available to SGCR.

Gates:
  F0 integrity/non-contradiction  -- one polarity map, split disjointness,
     dedup/conflict removal counted, NO source field in learner state,
     chronology + train/audit disjointness.
  F1 source-free observability    -- probe balanced acc >= 0.75; k-means AMI
     >= 0.35 and purity >= 0.65; rare-source best cell prec & rec >= 0.50.
  F2 oracle recoverability        -- oracle beats ER-FLOP worst-source BA by
     >= 3pp on average and loses <= 1pp macro BA (3 dev seeds).
  F3 nonlinear LoRA viability     -- warmup + offline-joint beat random; offline
     worst-source BA >= 0.65; disabling LoRA changes logits + lowers perf; every
     LoRA module finite nonzero grads + rank <= 4; no frozen base/head change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Thresholds (protocol, frozen).
F1_PROBE_BALACC = 0.75
F1_KMEANS_AMI = 0.35
F1_KMEANS_PURITY = 0.65
F1_RARE_PREC = 0.50
F1_RARE_REC = 0.50
F2_ORACLE_WORST_GAIN = 0.03        # >= 3pp over ER-FLOP
F2_ORACLE_MACRO_LOSS = 0.01        # <= 1pp macro loss
F3_OFFLINE_WORST_BA = 0.65


@dataclass
class Verdict:
    gate: str
    passed: bool
    details: Dict = field(default_factory=dict)

    def __bool__(self):
        return self.passed


# --------------------------------------------------------------------------- #
# Metric primitives (evaluator-only)
# --------------------------------------------------------------------------- #
def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                      n_classes: Optional[int] = None) -> float:
    """Mean per-class recall (balanced accuracy)."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    classes = range(n_classes) if n_classes else np.unique(y_true)
    recalls = []
    for c in classes:
        idx = y_true == c
        if idx.sum() == 0:
            continue
        recalls.append(float((y_pred[idx] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


# NOTE (dependency discipline): the protocol says the first implementation
# should require only `transformers` beyond the packages already present.  We
# therefore do NOT add scikit-learn; the source probe, stratified folds, and
# adjusted mutual information are implemented directly on numpy/scipy below.
def _softmax_probe_fit(X: np.ndarray, y: np.ndarray, n_classes: int,
                       l2: float = 1e-2) -> np.ndarray:
    """Multinomial logistic regression (softmax) via scipy L-BFGS.  Returns a
    weight matrix W of shape (n_classes, d+1) (bias-augmented)."""
    from scipy.optimize import minimize
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])          # bias term
    Y = np.zeros((n, n_classes)); Y[np.arange(n), y] = 1.0

    def loss_grad(w):
        W = w.reshape(n_classes, d + 1)
        Z = Xb @ W.T                              # (n, k)
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
        # cross-entropy + L2 (bias not regularized)
        ce = -np.sum(Y * np.log(P + 1e-12)) / n
        reg = 0.5 * l2 * np.sum(W[:, :-1] ** 2)
        grad = ((P - Y).T @ Xb) / n               # (k, d+1)
        grad[:, :-1] += l2 * W[:, :-1]
        return ce + reg, grad.ravel()

    w0 = np.zeros(n_classes * (d + 1))
    res = minimize(loss_grad, w0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 500})
    return res.x.reshape(n_classes, d + 1)


def _softmax_probe_predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return np.argmax(Xb @ W.T, axis=1)


def _stratified_folds(y: np.ndarray, folds: int, seed: int):
    """Yield (train_idx, test_idx) stratified by class label, deterministic."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    fold_of = np.empty(len(y), dtype=np.int64)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % folds
    for f in range(folds):
        te = np.where(fold_of == f)[0]
        tr = np.where(fold_of != f)[0]
        if len(te) and len(tr):
            yield tr, te


def linear_probe_balacc(signatures: np.ndarray, sources: np.ndarray,
                        seed: int = 0, folds: int = 5) -> float:
    """Post-hoc multinomial-logistic source probe on frozen signatures,
    evaluated by cross-validated balanced accuracy.  Evaluator-only: its weights
    are never seen by SGCR.  Pure numpy/scipy (no scikit-learn)."""
    X = np.asarray(signatures, dtype=np.float64)
    y = np.asarray(sources).astype(np.int64)
    n_classes = int(y.max()) + 1
    accs = []
    for tr, te in _stratified_folds(y, folds, seed):
        W = _softmax_probe_fit(X[tr], y[tr], n_classes)
        accs.append(balanced_accuracy(y[te], _softmax_probe_predict(W, X[te]),
                                      n_classes=n_classes))
    return float(np.mean(accs)) if accs else 0.0


def _expected_mutual_information(contingency: np.ndarray) -> float:
    """Expected MI under the hypergeometric null (Vinh et al. 2010), used to
    adjust MI.  contingency is the (R,C) count table."""
    from scipy.special import gammaln
    n = contingency.sum()
    a = contingency.sum(axis=1)                   # row sums
    b = contingency.sum(axis=0)                   # col sums
    emi = 0.0
    log_n = np.log(n)
    for i in range(len(a)):
        for j in range(len(b)):
            ai, bj = a[i], b[j]
            nij_low = max(1, int(ai + bj - n))
            nij_high = int(min(ai, bj))
            for nij in range(nij_low, nij_high + 1):
                term = (nij / n) * (np.log(nij) + log_n - np.log(ai) - np.log(bj))
                # log of the hypergeometric probability weight
                log_w = (gammaln(ai + 1) + gammaln(bj + 1) + gammaln(n - ai + 1)
                         + gammaln(n - bj + 1) - gammaln(n + 1) - gammaln(nij + 1)
                         - gammaln(ai - nij + 1) - gammaln(bj - nij + 1)
                         - gammaln(n - ai - bj + nij + 1))
                emi += term * np.exp(log_w)
    return float(emi)


def _entropy(counts: np.ndarray) -> float:
    n = counts.sum()
    p = counts[counts > 0] / n
    return float(-(p * np.log(p)).sum())


def kmeans_ami_purity(labels: np.ndarray, sources: np.ndarray) -> Tuple[float, float]:
    """Adjusted mutual information + many-to-one purity between learner cell
    labels and true sources.  Pure numpy/scipy (Vinh AMI, no scikit-learn)."""
    labels = np.asarray(labels); sources = np.asarray(sources)
    src_u = {v: i for i, v in enumerate(np.unique(sources))}
    lab_u = {v: i for i, v in enumerate(np.unique(labels))}
    cont = np.zeros((len(src_u), len(lab_u)), dtype=np.int64)
    for s, l in zip(sources, labels):
        cont[src_u[int(s)], lab_u[int(l)]] += 1

    n = cont.sum()
    # mutual information
    a = cont.sum(axis=1); b = cont.sum(axis=0)
    mi = 0.0
    for i in range(cont.shape[0]):
        for j in range(cont.shape[1]):
            nij = cont[i, j]
            if nij > 0:
                mi += (nij / n) * (np.log(nij) + np.log(n) - np.log(a[i]) - np.log(b[j]))
    h_true = _entropy(a); h_pred = _entropy(b)
    emi = _expected_mutual_information(cont)
    denom = max(h_true, h_pred) - emi
    ami = 0.0 if abs(denom) < 1e-12 else (mi - emi) / denom

    # many-to-one purity: each cell votes its majority true source.
    purity_hits = sum(int(cont[:, j].max()) for j in range(cont.shape[1]))
    purity = purity_hits / len(labels)
    return float(ami), float(purity)


def rare_source_cell_prec_rec(labels: np.ndarray, sources: np.ndarray,
                              rare_source: int) -> Tuple[float, float]:
    """Precision & recall of the BEST-matching cell for the globally rare
    source (protocol F1): pick the cell maximizing recall*precision proxy."""
    labels = np.asarray(labels); sources = np.asarray(sources)
    is_rare = sources == rare_source
    best = (0.0, 0.0, -1.0)
    for c in np.unique(labels):
        in_cell = labels == c
        tp = int((in_cell & is_rare).sum())
        if tp == 0:
            continue
        prec = tp / int(in_cell.sum())
        rec = tp / int(is_rare.sum())
        if prec * rec > best[2]:
            best = (prec, rec, prec * rec)
    return best[0], best[1]


# --------------------------------------------------------------------------- #
# F0 integrity
# --------------------------------------------------------------------------- #
def gate_F0(integrity: Dict) -> Verdict:
    """integrity dict is assembled by run_development from the harness checks:
      single_polarity_map: bool
      splits_disjoint: bool
      dedup_counts: dict (removed cross-split/cross-source/conflict counts)
      no_source_in_learner: bool  (schema tests passed on records/batches/traces)
      chronology_ok: bool
      role_disjoint: bool
    """
    checks = {
        "single_polarity_map": bool(integrity.get("single_polarity_map", False)),
        "splits_disjoint": bool(integrity.get("splits_disjoint", False)),
        "no_source_in_learner": bool(integrity.get("no_source_in_learner", False)),
        "chronology_ok": bool(integrity.get("chronology_ok", False)),
        "role_disjoint": bool(integrity.get("role_disjoint", False)),
    }
    passed = all(checks.values())
    return Verdict("F0", passed, {**checks, "dedup_counts": integrity.get("dedup_counts", {})})


# --------------------------------------------------------------------------- #
# F1 observability
# --------------------------------------------------------------------------- #
def gate_F1(signatures: np.ndarray, sources: np.ndarray, cell_labels: np.ndarray,
            rare_source: int, seed: int = 0) -> Verdict:
    probe = linear_probe_balacc(signatures, sources, seed=seed)
    ami, purity = kmeans_ami_purity(cell_labels, sources)
    prec, rec = rare_source_cell_prec_rec(cell_labels, sources, rare_source)
    checks = {
        "probe_balacc": probe, "probe_ok": probe >= F1_PROBE_BALACC,
        "ami": ami, "ami_ok": ami >= F1_KMEANS_AMI,
        "purity": purity, "purity_ok": purity >= F1_KMEANS_PURITY,
        "rare_prec": prec, "rare_rec": rec,
        "rare_ok": prec >= F1_RARE_PREC and rec >= F1_RARE_REC,
    }
    passed = checks["probe_ok"] and checks["ami_ok"] and checks["purity_ok"] and checks["rare_ok"]
    # Diagnostic hint (protocol): high probe + low kmeans -> impl bug; low probe
    # -> unobservable representation.
    if not passed:
        if checks["probe_ok"] and not (checks["ami_ok"] and checks["purity_ok"]):
            checks["diagnosis"] = "signature/clustering implementation (probe ok, kmeans weak)"
        elif not checks["probe_ok"]:
            checks["diagnosis"] = "unobservable frozen representation (probe weak)"
    return Verdict("F1", passed, checks)


# --------------------------------------------------------------------------- #
# F2 oracle recoverability
# --------------------------------------------------------------------------- #
def gate_F2(oracle_worst: Sequence[float], erflop_worst: Sequence[float],
            oracle_macro: Sequence[float], erflop_macro: Sequence[float]) -> Verdict:
    """Across the dev seeds, oracle worst-source BA must beat ER-FLOP by >= 3pp
    on average, and lose <= 1pp macro BA on average."""
    ow = np.asarray(oracle_worst, float); ew = np.asarray(erflop_worst, float)
    om = np.asarray(oracle_macro, float); em = np.asarray(erflop_macro, float)
    worst_gain = float((ow - ew).mean())
    macro_loss = float((em - om).mean())     # positive == oracle lost macro
    checks = {
        "worst_gain": worst_gain, "worst_ok": worst_gain >= F2_ORACLE_WORST_GAIN,
        "macro_loss": macro_loss, "macro_ok": macro_loss <= F2_ORACLE_MACRO_LOSS,
        "n_seeds": int(len(ow)),
    }
    passed = checks["worst_ok"] and checks["macro_ok"]
    return Verdict("F2", passed, checks)


# --------------------------------------------------------------------------- #
# F3 nonlinear LoRA viability
# --------------------------------------------------------------------------- #
def gate_F3(warmup_balacc: float, offline_macro_balacc: float,
            offline_worst_balacc: float, random_balacc: float,
            lora_on_off_logit_diff: float, lora_disable_perf_drop: float,
            lora_grads_finite_nonzero: bool, max_numerical_rank: int,
            frozen_params_unchanged: bool) -> Verdict:
    checks = {
        "warmup_beats_random": warmup_balacc > random_balacc,
        "offline_beats_random": offline_macro_balacc > random_balacc,
        "offline_worst_ba": offline_worst_balacc,
        "offline_worst_ok": offline_worst_balacc >= F3_OFFLINE_WORST_BA,
        "lora_changes_logits": lora_on_off_logit_diff > 0.0,
        "lora_disable_drops_perf": lora_disable_perf_drop > 0.0,
        "lora_grads_finite_nonzero": bool(lora_grads_finite_nonzero),
        "rank_ok": max_numerical_rank <= 4,
        "frozen_unchanged": bool(frozen_params_unchanged),
    }
    passed = all(v if isinstance(v, bool) else True for k, v in checks.items()
                 if k.endswith("_ok") or k in (
                     "warmup_beats_random", "offline_beats_random",
                     "lora_changes_logits", "lora_disable_drops_perf",
                     "lora_grads_finite_nonzero", "frozen_unchanged"))
    return Verdict("F3", passed, checks)


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def all_gates_pass(verdicts: Sequence[Verdict]) -> bool:
    return all(bool(v) for v in verdicts)
