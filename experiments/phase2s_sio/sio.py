"""Phase-2S — Separation-gated Invariant Offset (SIO).

Frozen protocol: notes/phase2s_sio_protocol.md
  sha256 21e8963e3ca774f93cc13bf51b438027ec9f9cf7b30f0d23869f2a1cbe52e604 (§0-§7)
  amendments §A1 (79c14b35...), §A2 (458ca2d2...)

The rule, in one line: fit a tied-variance two-component 1-D Gaussian mixture to
`s_i = Z[i,1] - Z[i,0]`, and shift the decision boundary to the midpoint of the two
component *locations*, discarding the mixture *weights*.

Why that is the whole contribution. Batch Calibration (ICLR 2024, arXiv:2309.17249)
subtracts the per-class batch mean, which is a weight-weighted mixture of both class
conditionals, so its correction moves with the batch class marginal. Under balanced
accuracy the optimal offset does not depend on that marginal at all. Component
locations are class-conditional; weights are the marginal. SIO reads the former and
throws away the latter.

Every gate term below exists because a specific measured failure demanded it; see
the protocol's §2 table. The gate abstains by returning a zero offset, which must
reproduce the raw prediction bit-for-bit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Frozen thresholds (protocol §2). Fixed before any real-data run.
TAU_SNR = 1.5     # component separation in units of the shared sd
N_MIN = 48        # too few points to locate two components
W_MIN = 0.05      # one component essentially absent
B_CI = 200        # bootstrap draws for tau's CI
B_NULL = 120      # parametric-bootstrap draws for rho's null distribution
CI_ALPHA = 5.0    # two-sided, so [2.5, 97.5]
RHO_NULL_Q = 95.0  # rho's critical value = this percentile of its null


def _em2(s: np.ndarray, *, tied: bool, iters: int = 400, tol: float = 1e-11):
    """Two-component 1-D Gaussian EM. Returns (mu, sd, w) sorted by mu.

    The tied M-step is::

        sigma^2 = sum_k sum_i R_ik (s_i - mu_k)^2 / n

    Note the `/ n`, and note that `R * (s - mu)^2` is summed over *both* axes.
    Writing it as `mean_k(var_k) / n` -- i.e. normalising by `n_k` first and then
    dividing by `n` again -- shrank sigma to 0.06 against a truth of 1.13, which
    inflated snr to 42 and silently decoupled tau from the oracle offset. Two gate
    designs were built on top of that bug before it was caught. Pinned by
    tests/test_sio.py::test_tied_em_recovers_known_sigma.
    """
    s = np.asarray(s, dtype=float).ravel()
    n = s.size
    lo, hi = np.percentile(s, [15, 85])
    if hi - lo < 1e-6:
        lo, hi = float(s.min()), float(s.max())
    mu = np.array([lo, hi], dtype=float)
    sd = np.array([max(float(s.std()), 1e-3)] * 2)
    w = np.array([0.5, 0.5])

    for _ in range(iters):
        d = np.stack([-(s - mu[k]) ** 2 / (2 * sd[k] ** 2) - np.log(sd[k])
                      for k in range(2)], axis=1)
        d += np.log(np.clip(w, 1e-12, 1.0))
        d -= d.max(axis=1, keepdims=True)
        R = np.exp(d)
        R /= R.sum(axis=1, keepdims=True)

        nk = R.sum(axis=0) + 1e-12
        new_mu = (R * s[:, None]).sum(axis=0) / nk
        sq = R * (s[:, None] - new_mu) ** 2
        if tied:
            new_sd = np.full(2, np.sqrt(max(sq.sum() / n, 1e-12)))
        else:
            new_sd = np.sqrt(np.maximum(sq.sum(axis=0) / nk, 1e-12))
        w = nk / nk.sum()

        done = (np.abs(new_mu - mu).max() < tol and np.abs(new_sd - sd).max() < tol)
        mu, sd = new_mu, new_sd
        if done:
            break

    order = np.argsort(mu)
    return mu[order], sd[order], w[order]


def _tau(s: np.ndarray) -> float:
    """The threshold: midpoint of the two component locations.

    Under a tied variance this is exactly the equal-posterior point at w = 1/2,
    i.e. the Bayes threshold for balanced accuracy. `w` is fitted but discarded --
    that discard is what makes the rule marginal-invariant.
    """
    mu, _, _ = _em2(s, tied=True)
    return 0.5 * float(mu[0] + mu[1])


def _rho(s: np.ndarray) -> float:
    """Tied-model misspecification statistic, from a *free*-variance fit.

    The free fit is used only to veto. It cannot supply the offset: under skew it
    degenerates, estimating mu_1 = 0.35 against a truth of 1.88 while inflating that
    component's sd to absorb the majority class's tail. BIC does not choose between
    the two models -- it selected `free` in every cell tested, including cells whose
    truth is tied.
    """
    _, fsd, _ = _em2(s, tied=False, iters=60)
    return float(max(fsd) / max(min(fsd), 1e-9))


@dataclass(frozen=True)
class SIOResult:
    offset: np.ndarray      # shape (2,), gauge-fixed as (0, -tau); zeros if abstaining
    applied: bool
    tau: float
    snr: float
    w_min: float
    rho: float
    rho_crit: float         # nan when the cheap terms already rejected
    ci: tuple               # (lo, hi) for tau; (nan, nan) when not computed
    reason: str             # "applied" or the first gate term that rejected


def sio_offset(Z: np.ndarray, *, seed: int = 0) -> SIOResult:
    """Compute SIO's offset for one batch of 2-class verbalizer logits.

    `Z` is (n, 2). The returned offset is added to `Z` before argmax. Abstention
    returns exactly zeros, so the prediction is bit-for-bit the raw one.

    Gauge invariance: only `s = Z[:,1] - Z[:,0]` is read, so adding any constant to
    every logit leaves the result unchanged.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[1] != 2:
        raise ValueError(f"SIO is defined for K_S=2 only; got shape {Z.shape}")
    n = Z.shape[0]
    zero = np.zeros(2)
    s = Z[:, 1] - Z[:, 0]

    if n < N_MIN:
        return SIOResult(zero, False, np.nan, np.nan, np.nan, np.nan, np.nan,
                         (np.nan, np.nan), "n_min")

    mu, sd, w = _em2(s, tied=True)
    tau = 0.5 * float(mu[0] + mu[1])
    snr = float((mu[1] - mu[0]) / max(sd[0], 1e-9))
    w_min = float(min(w))

    if snr < TAU_SNR:
        return SIOResult(zero, False, tau, snr, w_min, np.nan, np.nan,
                         (np.nan, np.nan), "snr")
    if w_min < W_MIN:
        return SIOResult(zero, False, tau, snr, w_min, np.nan, np.nan,
                         (np.nan, np.nan), "w_min")

    rng = np.random.default_rng(seed)
    rho = _rho(s)

    # rho's critical value is calibrated per batch, not a constant. A constant 1.6
    # rejected the 80:20 and 90:10 arms (rho 1.74/1.69) while genuinely
    # unequal-variance batches sat at 3.09/4.04 -- the two ranges overlap, because
    # rho is upward-biased under skew (the rare component has fewer points). The
    # null is generated at the *fitted* w, so skew is absorbed.
    nulls = np.empty(B_NULL)
    for b in range(B_NULL):
        lab = rng.random(n) < w[0]
        sim = np.where(lab, mu[0], mu[1]) + rng.normal(0.0, sd[0], n)
        nulls[b] = _rho(sim)
    rho_crit = float(np.percentile(nulls, RHO_NULL_Q))
    if rho > rho_crit:
        return SIOResult(zero, False, tau, snr, w_min, rho, rho_crit,
                         (np.nan, np.nan), "rho_misspecified")

    # tau's CI must exclude 0: there must be a bias worth correcting. A hand-picked
    # |tau|/sigma >= 0.15 let a no-bias 90:10 batch through and lost 2.78 pp.
    draws = np.empty(B_CI)
    for b in range(B_CI):
        draws[b] = _tau(s[rng.integers(0, n, n)])
    lo, hi = (float(x) for x in np.percentile(draws, [CI_ALPHA / 2, 100 - CI_ALPHA / 2]))
    if lo <= 0.0 <= hi:
        return SIOResult(zero, False, tau, snr, w_min, rho, rho_crit,
                         (lo, hi), "tau_ci_contains_zero")

    return SIOResult(np.array([0.0, -tau]), True, tau, snr, w_min, rho, rho_crit,
                     (lo, hi), "applied")


def bc_offset(Z: np.ndarray) -> np.ndarray:
    """Batch Calibration (arXiv:2309.17249), the method SIO is a fix for.

    `p_hat(y_j|C) = (1/M) sum_i p(y_j|x_i,C)`, subtracted from each row's
    probabilities. Reproduced here so both arms are scored from the same logits in
    the same process, never from separately reported numbers.
    """
    Z = np.asarray(Z, dtype=float)
    e = np.exp(Z - Z.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return -p.mean(axis=0)


def marginal_slope(corrections: np.ndarray, ratios: np.ndarray) -> float:
    """OLS slope of a correction against the batch class ratio (protocol §A2.4, S5b).

    This is the mechanism test. sd across mixes -- criterion S5 as frozen -- conflates
    sampling noise with systematic drift and fails even when the mechanism holds:
    tau scatters around the true bias with no trend while BC's correction marches
    monotonically across zero, yet the two have comparable sd. The slope separates
    them (BC -0.858 +- 0.016, negative in 20/20 seeds; SIO -0.188 +- 0.599, no sign
    consistency).
    """
    x = np.asarray(ratios, dtype=float) - np.mean(ratios)
    y = np.asarray(corrections, dtype=float)
    denom = float(np.dot(x, x))
    if denom <= 0:
        return float("nan")
    return float(np.dot(x, y - y.mean()) / denom)
