"""Correctness harness for the PSR-LoRA line (phase2f).

Every quantitative failure in the retracted phase2e v2/v3 code traces to a broken
SHARED PRIMITIVE: a recency-biased "reservoir", a gauge-dependent atom score, a
"CVaR" that never computed CVaR, a byte ledger that omitted the replay buffer, and
a "validation" that never rolled back.  This module implements those primitives
ONCE, correctly, and pins each with an executable unit test.  Nothing downstream
(benchmark_v4, baselines, psr_lora) may re-implement them.

Design rule for these tests: a green test proves nothing unless it can go red.  So
for the two primitives whose bugs were silent (reservoir uniformity, factor gauge)
we ALSO keep the known-broken implementation and assert the test REJECTS it.  A
self-test that cannot fail is not a test.

Run:  python harness.py           # runs all tests, exits nonzero on any failure
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

# ---------------------------------------------------------------------------
# 1. Reservoir sampling
# ---------------------------------------------------------------------------
# Vitter (1985) Algorithm R.  After the buffer of capacity B is full, item i
# (0-indexed, seen = i+1 items so far) replaces a uniformly random slot with
# probability B/(i+1).  At the end of a stream of length N every item is retained
# with probability exactly B/N, independent of position.
#
# The retracted code used `j = rng.integers(0, len(buf) + 1)` with a FULL buffer,
# i.e. len(buf)==B always, so the acceptance probability was a constant B/(B+1)
# for every window regardless of how many have been seen -- a strong recency bias.


class Reservoir:
    """Correct Vitter Algorithm-R reservoir over arbitrary python objects."""

    def __init__(self, capacity, rng):
        self.capacity = int(capacity)
        self.rng = rng
        self.items = []
        self.seen = 0

    def offer(self, item):
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        # replace slot j with probability capacity/seen
        j = int(self.rng.integers(0, self.seen))   # note: bound is `seen`, not capacity+1
        if j < self.capacity:
            self.items[j] = item

    def __len__(self):
        return len(self.items)


class _BuggyRecencyReservoir:
    """The RETRACTED implementation, kept ONLY so the uniformity test can prove it
    has the power to reject a recency-biased buffer.  Do not use downstream."""

    def __init__(self, capacity, rng):
        self.capacity = int(capacity)
        self.rng = rng
        self.items = []

    def offer(self, item):
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            j = int(self.rng.integers(0, len(self.items) + 1))   # the bug
            if j < self.capacity:
                self.items[j] = item


def _retention_by_position(reservoir_cls, N, B, trials, seed):
    """Empirical probability that the item at each stream position survives to the
    end, averaged over `trials` independent streams."""
    rng = np.random.default_rng(seed)
    survive = np.zeros(N)
    for _ in range(trials):
        r = reservoir_cls(B, rng)
        for i in range(N):
            r.offer(i)
        for kept in r.items:
            survive[kept] += 1
    return survive / trials


# ---------------------------------------------------------------------------
# 2. LoRA factor gauge
# ---------------------------------------------------------------------------
# A LoRA operator M = B A has a continuous gauge freedom: for any invertible g,
# (B g^{-1})(g A) = B A.  Any score computed from A (or B) alone -- e.g. the
# retracted inverse-curvature score a^T (Ain+lam)^{-1} a -- is therefore NOT a
# function of the operator and can be changed arbitrarily without changing M.
# The fix is to rebalance to the balanced SVD gauge B=U S^{1/2}, A=S^{1/2} V^T
# after every update, and to make every protection/score decision in operator
# space.


def rebalance_factors(B, A):
    """Return (B', A') with B'A' == B A (to fp error) in the balanced gauge
    B'=U S^{1/2}, A'=S^{1/2} V^T.  Columns of B' / rows of A' are the singular
    directions, so factor norms are gauge-fixed and comparable."""
    M = B @ A
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = A.shape[0]
    U, S, Vt = U[:, :r], S[:r], Vt[:r, :]
    root = np.sqrt(S)
    return U * root[None, :], root[:, None] * Vt


def gauge_transform(B, A, g):
    """Apply an arbitrary invertible gauge g to (B,A): (B g^{-1}, g A)."""
    return B @ np.linalg.inv(g), g @ A


# ---------------------------------------------------------------------------
# 3. Empirical CVaR (the actual tail objective)
# ---------------------------------------------------------------------------
# CVaR_tau(losses) = mean of the worst tau-fraction (the upper tau tail).  The
# retracted "cvar_replay" only *sampled* one worst-half buffer element to train on;
# it never formed this objective and never used excess-over-anchor, so difficult or
# noisy samples dominated the tail.  This is the objective; excess() below is what
# feeds it.


def cvar(losses, tau):
    """Upper-tail mean: average of the ceil(tau*n) largest values."""
    x = np.sort(np.asarray(losses, float))[::-1]
    n = x.size
    if n == 0:
        return 0.0
    k = max(1, int(np.ceil(tau * n)))
    return float(np.mean(x[:k]))


def excess(loss, anchor, gain, noise_margin=0.0, e_max=np.inf):
    """Excess forgetting over the post-learning anchor, normalized by the gain the
    sample once yielded and clipped: relu(loss - anchor - noise_margin)/(gain+eps),
    clipped to [0, e_max].  Keeps naturally-hard / high-noise samples from owning
    the tail -- only *regression below what was already achieved* counts."""
    num = np.maximum(loss - anchor - noise_margin, 0.0)
    return np.minimum(num / (gain + 1e-12), e_max)


# ---------------------------------------------------------------------------
# 4. Byte ledger
# ---------------------------------------------------------------------------
# Count EVERY float an arm keeps resident, the same way for every arm.  The
# retracted ledger counted FCRA's 8,960-byte curvature state but omitted the
# 262,144-byte replay buffer it also held.  Here a buffer of `n_windows` windows,
# each `batch` examples of an in/out pair of width d, is counted in full.


def ledger_bytes(*, d, R, buffer_windows=0, batch=0, curv_dxd=0,
                 protected_cols=0, extra_floats=0, dtype_bytes=8):
    """Resident float count -> bytes.  adapter B(dxR)+A(Rxd); replay buffer
    holds buffer_windows*(X:batch x d, Y:batch x d); curvature curv_dxd*(d x d);
    protected basis protected_cols*(d)."""
    floats = 2 * d * R
    floats += buffer_windows * batch * 2 * d
    floats += curv_dxd * d * d
    floats += protected_cols * d
    floats += extra_floats
    return int(floats * dtype_bytes)


# ---------------------------------------------------------------------------
# 4b. Closed-form weighted reduced-rank regression (shared solver)
# ---------------------------------------------------------------------------
# The offline reference (benchmark_v4) and the PSR-LoRA method both need the best
# rank-R operator under a weighted quadratic loss.  It lives here, defined and
# tested ONCE, so the two callers cannot drift apart and the method is not reaching
# into the benchmark.  It is pure linear algebra -- a generic estimator, not any
# oracle knowledge: whiten by the weighted second moment, truncate to rank R
# (Eckart-Young), unwhiten.


def second_moment(X):
    """Input second moment Sigma = X^T X / n."""
    return (X.T @ X) / X.shape[0]


def weighted_rrr(Tstar_est, Sigmas, weights, R):
    """Closed-form argmin over rank<=R M of sum_c w_c tr((M-T*)Sigma_c(M-T*)^T).
    Whiten by Sigma_w^{1/2}, truncate N=T* Sigma_w^{1/2} to rank R, unwhiten."""
    d = Tstar_est.shape[0]
    Sig = sum(w * S for w, S in zip(weights, Sigmas)) + 1e-9 * np.eye(d)
    ev, Q = np.linalg.eigh(Sig)
    ev = np.clip(ev, 1e-12, None)
    half = Q @ np.diag(np.sqrt(ev)) @ Q.T
    ihalf = Q @ np.diag(1.0 / np.sqrt(ev)) @ Q.T
    N = Tstar_est @ half
    Un, Sn, Vn = np.linalg.svd(N, full_matrices=False)
    Nr = (Un[:, :R] * Sn[:R][None, :]) @ Vn[:R, :]
    return Nr @ ihalf


# ---------------------------------------------------------------------------
# 5. Label-blindness / trajectory hash
# ---------------------------------------------------------------------------


def trajectory_hash(snaps):
    """Stable hash of a list of adapter snapshots (order-sensitive)."""
    h = hashlib.sha256()
    for M in snaps:
        h.update(np.ascontiguousarray(M, dtype=np.float64).tobytes())
    return h.hexdigest()


# ===========================================================================
# self-tests
# ===========================================================================

def _t(cond, msg, results):
    results.append((bool(cond), msg))
    print(("  PASS " if cond else "  FAIL ") + msg)


def run_tests():
    results = []
    print("reservoir:")
    N, B, trials = 240, 8, 6000
    # correct reservoir: uniform B/N retention, flat across position
    surv = _retention_by_position(Reservoir, N, B, trials, seed=101)
    target = B / N
    # early third vs late third mean survival should both be ~ target
    early = surv[: N // 3].mean()
    late = surv[-N // 3 :].mean()
    _t(abs(surv.mean() - target) < 0.004, f"mean retention {surv.mean():.4f} ~= B/N {target:.4f}", results)
    _t(abs(early - late) < 0.010, f"position-flat: early {early:.4f} vs late {late:.4f}", results)
    # meta-test: the buggy reservoir must be REJECTED by the same flatness check
    bsurv = _retention_by_position(_BuggyRecencyReservoir, N, B, trials, seed=101)
    bearly = bsurv[: N // 3].mean()
    blate = bsurv[-N // 3 :].mean()
    _t(blate > 5 * bearly, f"meta: buggy reservoir IS recency-biased (early {bearly:.5f} << late {blate:.4f})", results)
    _t(bsurv[0] < 1e-3, f"meta: buggy reservoir nearly drops window 0 (p={bsurv[0]:.6f})", results)

    print("factor gauge:")
    rng = np.random.default_rng(7)
    d, R = 16, 4
    Bm = rng.standard_normal((d, R)); Am = rng.standard_normal((R, d))
    g = rng.standard_normal((R, R))
    Bg, Ag = gauge_transform(Bm, Am, g)
    _t(np.allclose(Bg @ Ag, Bm @ Am, atol=1e-8), "gauge transform preserves the operator", results)
    # a raw factor-norm score is gauge-DEPENDENT (this is the bug being prevented)
    raw_before = float(np.linalg.norm(Am, axis=1).sum())
    raw_after = float(np.linalg.norm(Ag, axis=1).sum())
    _t(abs(raw_before - raw_after) > 0.1, f"meta: raw A-score IS gauge-dependent ({raw_before:.3f} vs {raw_after:.3f})", results)
    # rebalanced factors are gauge-INVARIANT: same operator -> same balanced factors
    Br1, Ar1 = rebalance_factors(Bm, Am)
    Br2, Ar2 = rebalance_factors(Bg, Ag)
    _t(np.allclose(Br1 @ Ar1, Br2 @ Ar2, atol=1e-8), "rebalanced factors reconstruct same operator", results)
    _t(np.allclose(np.sort(np.linalg.norm(Ar1, axis=1)), np.sort(np.linalg.norm(Ar2, axis=1)), atol=1e-6),
       "rebalanced factor norms are gauge-invariant", results)

    print("cvar / excess:")
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    _t(abs(cvar(x, 0.5) - 7.0) < 1e-9, f"cvar(0.5) of 0..9 = {cvar(x,0.5)} (mean of 5..9)", results)
    _t(abs(cvar(x, 0.1) - 9.0) < 1e-9, f"cvar(0.1) = {cvar(x,0.1)} (the single worst)", results)
    _t(abs(cvar(x, 1.0) - 4.5) < 1e-9, f"cvar(1.0) = {cvar(x,1.0)} (the mean)", results)
    # excess: a naturally-hard sample (high loss but also high anchor) is NOT tail
    ex_hard = excess(loss=5.0, anchor=5.0, gain=1.0)     # no regression -> 0
    ex_forgot = excess(loss=5.0, anchor=1.0, gain=1.0)   # regressed by 4
    _t(ex_hard < 1e-9, f"excess: no regression below anchor -> 0 (got {ex_hard:.3f})", results)
    _t(ex_forgot > 3.9, f"excess: real regression counted (got {ex_forgot:.3f})", results)

    print("ledger:")
    # FCRA-like arm MUST count its replay buffer, not just curvature state
    with_buf = ledger_bytes(d=32, R=4, buffer_windows=8, batch=64, curv_dxd=1, protected_cols=3)
    without_buf = ledger_bytes(d=32, R=4, buffer_windows=0, batch=64, curv_dxd=1, protected_cols=3)
    _t(with_buf - without_buf == 8 * 64 * 2 * 32 * 8, f"replay buffer counted: +{with_buf-without_buf} bytes", results)
    # 262144 (buffer) + 2048 (adapter) + 8192 (curv d^2) + 768 (protected) = 273152
    _t(with_buf == 273152, f"full FCRA ledger = {with_buf} bytes (buffer dominates)", results)

    print("weighted RRR (shared solver):")
    rng = np.random.default_rng(3)
    dd, KK, RR = 12, 4, 2
    # K=4 sources on distinct orthonormal directions, teacher gains DECAYING, and only
    # R=2 slots -> a genuine capacity floor: the uniform rank-2 fit must DROP the two
    # weakest-gain sources.  (R>=K would serve everyone and the test could not go red.)
    Ut, _ = np.linalg.qr(rng.standard_normal((dd, dd)))
    Vt2, _ = np.linalg.qr(rng.standard_normal((dd, dd)))
    gains = np.array([2.0, 1.6, 1.2, 0.8])                      # source 3 is weakest
    Tst = sum(gains[k] * np.outer(Ut[:, k], Vt2[:, k]) for k in range(KK))
    Sigs = [np.outer(Vt2[:, k], Vt2[:, k]) + 1e-3 * np.eye(dd) for k in range(KK)]
    Mu = weighted_rrr(Tst, Sigs, [1.0] * KK, RR)
    lu = [float(np.trace((Mu - Tst) @ S @ (Mu - Tst).T)) for S in Sigs]
    # the capacity floor shows as a genuinely worst (dropped) source
    worst = int(np.argmax(lu))
    _t(max(lu) > min(lu) + 1e-3, f"uniform RRR leaves a worst source (losses {[round(x,3) for x in lu]})", results)
    # upweight the worst source -> its loss must drop (reallocation actually works)
    w = [1.0] * KK; w[worst] = 30.0
    Mw = weighted_rrr(Tst, Sigs, w, RR)
    lw = [float(np.trace((Mw - Tst) @ S @ (Mw - Tst).T)) for S in Sigs]
    _t(lw[worst] < lu[worst] - 1e-3,
       f"upweighting the worst source lowers its loss ({lu[worst]:.3f} -> {lw[worst]:.3f})", results)
    _t(np.linalg.matrix_rank(Mw, tol=1e-6) <= RR, "weighted RRR stays rank <= R", results)

    print("trajectory hash:")
    s1 = [np.eye(4), 2 * np.eye(4)]
    s2 = [np.eye(4), 2 * np.eye(4)]
    s3 = [np.eye(4), 3 * np.eye(4)]
    _t(trajectory_hash(s1) == trajectory_hash(s2), "identical trajectories hash equal", results)
    _t(trajectory_hash(s1) != trajectory_hash(s3), "different trajectories hash differently", results)

    n_fail = sum(1 for ok, _ in results if not ok)
    print(f"\n{len(results)-n_fail}/{len(results)} passed")
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if run_tests() else 0)
