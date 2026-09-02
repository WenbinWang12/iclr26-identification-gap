"""Pre-registered method comparison + kill criterion for PSR-LoRA.

--------------------------------------------------------------------------------
SUPERSEDED -- historical record, kept for provenance.  Read this first.
--------------------------------------------------------------------------------
This is the ORIGINAL protocol, run on the PRE-COVERAGE method: PSR-LoRA as a
minimax reweighted rank-R RRR over a SINGLE uniform reservoir of budget 16
(coverage=False).  Its recorded output is `outputs_compare_v2.txt`, in which
criteria (b) and (c) FAIL on the cluster-robust CI (tail 0.678; (b) CI
[-0.0531,+0.2196], (c) CI [-0.0017,+0.1444], both straddle 0).  That failure is
the DIAGNOSIS that motivated the coverage remedy: a single uniform reservoir
under-samples the rare/worst source, so the reweighting has nothing to reallocate
toward.

The FROZEN, VALIDATED method is the coverage-aware buffer (buf_u=4 uniform +
buf_g=12 input-direction-diverse, total 16), decided by `preregister_heldout.py`
(recorded in `outputs_heldout.txt`: tail 0.792, all three criteria PASS on the
same cluster-robust CI).  For any paper/README number use THAT file, not this one.

This file is pinned to `coverage=False` (see `_cfg`) so it faithfully reproduces
its historical `outputs_compare_v2.txt` record.  It was previously drifting: with
`coverage` left to inherit the psr_lora DEFAULT (now True) at buf_u=8/buf_g=8 it
silently ran an 8/8 coverage HYBRID that matches NEITHER recorded file (psr tail
0.801, not 0.827) -- an unvalidated third config.  The pin removes that drift.
--------------------------------------------------------------------------------

This file freezes the decision BEFORE the numbers are seen.  It is the payoff of the
whole phase2f rebuild: for the first time PSR-LoRA is judged on a trustworthy
foundation (true reservoir, real CVaR/excess, well-posed v4 benchmark, closed-form
oracle reference, and correct strong baselines), against a criterion fixed in
advance.  The honesty rule: run it once on HELD-OUT teacher geometries and seeds,
report whatever comes out, do NOT tune to rescue the number.

--------------------------------------------------------------------------------
PRE-REGISTERED PROTOCOL (frozen 2026-08-26, before running the held-out set)
--------------------------------------------------------------------------------
Arena.
  * Regime: capacity_limited (the only regime with a genuine rank-R capacity floor
    and thus a real mean-tail tradeoff; compressible has no floor, conflicting is an
    impossibility control -- neither can test a reallocation method).
  * Development used teacher seed 20260701 and stream seed 6060.  The kill test uses
    NEW teacher geometries (seeds 20270001..20270005) x NEW stream seeds
    (7001..7004): 20 (teacher, stream) worlds never seen during design.
  * Every arm runs on the SAME (X,Y) windows of each world, sees labels never (all
    are oracle-free), and holds the SAME bounded buffer budget (buf_u+buf_g = 16).

Metrics per world (on independent test holdouts):
  * mean  = frequency-weighted source retention.
  * tail  = worst-group source retention (the object the theory says a fixed rank
            floor bounds; this is what a reallocation method must lift).
  * The offline closed-form weighted-RRR frontier gives the reachable (mean, tail)
    envelope; we report tail as a fraction of the offline best-tail too.

Arms.
  * PSR-LoRA          : bounded-buffer minimax reweighted rank-R RRR (the method).
  * buffer_rrr_uniform: SAME buffer + SAME closed-form solver, uniform weights
                        (the mechanism ablation -- isolates the reweighting).
  * excess_cvar       : strongest risk-aware replay baseline (dev-set tail leader).
  * uniform_replay, gss_replay, mir_replay, agem, ewc_lora, sequential : baselines.
  * psr_grad_defensive: the original gradient-projection design (negative control).

KILL CRITERION (all three must hold to KEEP the method line; else stop it):
  (a) NON-INFERIOR MEAN: PSR-LoRA mean >= (best-baseline mean) - 0.01, paired over
      the 20 worlds (mean of paired differences within the 0.01 margin).
  (b) STRICTLY BETTER TAIL: PSR-LoRA tail > excess_cvar tail (the strongest replay
      baseline's tail) with a paired 95% CI on the per-world difference that
      excludes 0.  [Matched-mean is automatic here: (a) forces means to coincide, so
      a tail gain is not bought with mean.]
  (c) MECHANISM CONTRIBUTES: PSR-LoRA tail > buffer_rrr_uniform tail, paired 95% CI
      excluding 0.  If (c) fails, the minimax reweighting adds nothing over "buffer
      + closed-form solve" and the method line stops regardless of (a)/(b).

Report also, for context, the grad-defensive control (expected to sit near the
uniform capacity floor) so the contrast that motivated the redesign is visible.

Run:  python compare.py            # prints the frozen verdict
"""

from __future__ import annotations

import numpy as np

import benchmark_v4 as BM
import baselines as BL
import psr_lora as P
import harness as H


# ---- frozen held-out worlds (NOT the dev seeds 20260701 / 6060) ----
TEACHER_SEEDS = [20270001, 20270002, 20270003, 20270004, 20270005]
STREAM_SEEDS = [7001, 7002, 7003, 7004]
REGIME = "capacity_limited"
RUN_SEED = 7                      # the per-arm RNG seed (buffer draws); fixed for all
MEAN_MARGIN = 0.01               # criterion (a) non-inferiority margin


def _cfg():
    """Config for the SUPERSEDED uniform-16 record (outputs_compare_v2.txt).

    `coverage=False` is PINNED here so both method arms (psr_lora and
    buffer_rrr_uniform) run the single-uniform-reservoir design that produced
    v2 -- not the psr_lora DEFAULT (now coverage=True), which would silently make
    this an 8/8 coverage hybrid matching no recorded file.  The 8+8=16 split is
    the pre-coverage total budget; with coverage off the split collapses to one
    16-slot reservoir, so buf_u/buf_g values only need to sum to 16.  The frozen
    coverage method's 4/12 split lives in preregister_heldout.py, not here."""
    cfg = dict(BM.DEFAULT_CFG)
    cfg.update({"gd_steps": 20, "lr": 0.15, "buf_u": 8, "buf_g": 8,
                "coverage": False, "tau": 0.4, "xi": 0.02})
    return cfg


def _mean_tail(M, srcs, freqs, test):
    ret = {c: BM.source_retention(M, *test[c]) for c in srcs}
    mean = sum(freqs[c] * ret[c] for c in srcs) / sum(freqs[c] for c in srcs)
    return float(mean), float(min(ret.values()))


ARMS = {
    "psr_lora": lambda cfg, b, s: P.run_psr_lora(cfg, b, s)[0][-1],
    "buffer_rrr_uniform": lambda cfg, b, s: P.run_buffer_rrr_uniform(cfg, b, s)[0][-1],
    "psr_grad_defensive": lambda cfg, b, s: P.run_psr_grad_defensive(cfg, b, s, eps_avg=0.3, eps_tail=0.3)[0][-1],
    "excess_cvar": lambda cfg, b, s: BL.run_excess_cvar(cfg, b, s)[0][-1],
    "uniform_replay": lambda cfg, b, s: BL.run_uniform_replay(cfg, b, s)[0][-1],
    "gss_replay": lambda cfg, b, s: BL.run_gss_replay(cfg, b, s)[0][-1],
    "mir_replay": lambda cfg, b, s: BL.run_mir_replay(cfg, b, s)[0][-1],
    "agem": lambda cfg, b, s: BL.run_agem(cfg, b, s)[0][-1],
    "ewc_lora": lambda cfg, b, s: BL.run_ewc_lora(cfg, b, s)[0][-1],
    "sequential": lambda cfg, b, s: BL.run_sequential(cfg, b, s)[0][-1],
}


def run_worlds(verbose=True):
    cfg = _cfg()
    per_arm_mean = {a: [] for a in ARMS}
    per_arm_tail = {a: [] for a in ARMS}
    off_tail = []
    worlds = 0
    for ts in TEACHER_SEEDS:
        teacher = BM.build_teacher(REGIME, cfg, ts)
        for ss in STREAM_SEEDS:
            windows, freqs = BM.make_stream(cfg, ss)
            batches, srcs, otr, test = BM.materialize(teacher, cfg, windows, ss)
            front, pts = BM.offline_frontier(teacher, cfg, otr, test, srcs, freqs)
            off_tail.append(max(p["worst_group"] for p in pts))
            for a, fn in ARMS.items():
                try:
                    M = fn(cfg, batches, RUN_SEED)
                    m, t = _mean_tail(M, srcs, freqs, test)
                except np.linalg.LinAlgError:
                    m, t = np.nan, np.nan       # arm diverged on this world (recorded, not hidden)
                per_arm_mean[a].append(m)
                per_arm_tail[a].append(t)
            worlds += 1
            if verbose:
                print(f"  world t{ts}/s{ss}: psr tail {per_arm_tail['psr_lora'][-1]:.3f}  "
                      f"cvar tail {per_arm_tail['excess_cvar'][-1]:.3f}  "
                      f"unif-rrr tail {per_arm_tail['buffer_rrr_uniform'][-1]:.3f}  "
                      f"offline {off_tail[-1]:.3f}")
    return per_arm_mean, per_arm_tail, np.array(off_tail), worlds


def _paired_ci(diffs, alpha=0.05):
    """Paired mean difference and a normal-approx (1-alpha) CI on it, treating the
    worlds as IID.  Reported for TRANSPARENCY only -- the decision uses the
    cluster-robust CI below, which is never narrower."""
    d = np.asarray(diffs, float)
    n = d.size
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    z = 1.959963985                                  # 95% normal quantile
    return m, m - z * se, m + z * se


# two-sided 95% Student-t quantiles (t_{0.975, df}); fall back to z for large df.
_T975 = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
         6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139,
         11: 2.200985, 12: 2.178813, 13: 2.160369, 14: 2.144787, 15: 2.131450,
         16: 2.119905, 17: 2.109816, 18: 2.100922, 19: 2.093024, 20: 2.085963}


def _t975(df):
    if df >= 30:
        return 1.959963985
    if df in _T975:
        return _T975[df]
    lo = [k for k in _T975 if k <= df]               # round df DOWN -> larger t (conservative)
    return _T975[max(lo)] if lo else _T975[1]


def _paired_ci_crossed(diffs, n_t, n_s, alpha=0.05):
    """Paired mean difference with a CLUSTER-ROBUST CI for the 5x4 (teacher x stream)
    CROSSED design, replacing the IID normal-approx.

    The 20 worlds are NOT independent: worlds sharing a teacher share its geometry,
    worlds sharing a stream seed share its window sequence.  We model the per-world
    paired difference as d_ij = mu + a_i + b_j + e_ij (teacher, stream, residual,
    all independent) and estimate the variance of the grand mean by a balanced
    two-way random-effects ANOVA:

        Var(mean) = sigma_a^2 / n_t + sigma_b^2 / n_s + sigma_e^2 / (n_t n_s),

    with sigma_a^2 = max(0,(MS_A-MS_E)/n_s), sigma_b^2 = max(0,(MS_B-MS_E)/n_t),
    sigma_e^2 = MS_E.  Because n_t,n_s>=1 this variance is ALWAYS >= the IID
    variance s^2/N, so the CI can only widen -- the conservative, honest direction.
    The CI uses a small-sample t with df = min(n_t,n_s)-1 (=3 here), conservative
    relative to a Satterthwaite df.  Returns (mean, lo, hi, se, (sa2,sb2,se2), df)."""
    D = np.asarray(diffs, float).reshape(n_t, n_s)   # C-order: row=teacher, col=stream
    grand = float(D.mean())
    row = D.mean(axis=1); col = D.mean(axis=0)
    ms_a = n_s * float(np.sum((row - grand) ** 2)) / (n_t - 1) if n_t > 1 else 0.0
    ms_b = n_t * float(np.sum((col - grand) ** 2)) / (n_s - 1) if n_s > 1 else 0.0
    resid = D - row[:, None] - col[None, :] + grand
    ms_e = float(np.sum(resid ** 2)) / ((n_t - 1) * (n_s - 1)) if n_t > 1 and n_s > 1 else 0.0
    sa2 = max((ms_a - ms_e) / n_s, 0.0)              # teacher variance component
    sb2 = max((ms_b - ms_e) / n_t, 0.0)              # stream variance component
    se2 = ms_e                                       # residual variance component
    var_mean = sa2 / n_t + sb2 / n_s + se2 / (n_t * n_s)
    se = float(np.sqrt(max(var_mean, 0.0)))
    df = min(n_t, n_s) - 1
    t = _t975(df)
    return grand, grand - t * se, grand + t * se, se, (sa2, sb2, se2), df


def robustness_offfloor(verbose=True):
    """Pre-registered sanity: on regimes WITHOUT a rank-R capacity floor, the
    reallocation must NOT needlessly hurt mean (compressible), and on an impossible
    (conflicting) geometry the method must degrade gracefully -- never worse on mean
    than the uniform buffer-RRR solve, and it must not pretend to rescue the tail.
    These bound the method's behavior outside the arena where it is meant to help."""
    cfg = _cfg()
    out = {}
    for regime in ("compressible", "conflicting"):
        agg = {"psr_lora": [[], []], "excess_cvar": [[], []], "buffer_rrr_uniform": [[], []]}
        for ts in TEACHER_SEEDS[:3]:
            teacher = BM.build_teacher(regime, cfg, ts)
            for ss in STREAM_SEEDS[:2]:
                windows, freqs = BM.make_stream(cfg, ss)
                batches, srcs, otr, test = BM.materialize(teacher, cfg, windows, ss)
                for a in agg:
                    M = ARMS[a](cfg, batches, RUN_SEED)
                    m, t = _mean_tail(M, srcs, freqs, test)
                    agg[a][0].append(m); agg[a][1].append(t)
        out[regime] = {a: (float(np.mean(v[0])), float(np.mean(v[1]))) for a, v in agg.items()}
        if verbose:
            print(f"[off-floor] {regime}:")
            for a, (m, t) in out[regime].items():
                print(f"    {a:20s} mean {m:.3f} tail {t:.3f}")
    # the two properties we assert
    comp = out["compressible"]
    conf = out["conflicting"]
    no_harm = comp["psr_lora"][0] >= comp["buffer_rrr_uniform"][0] - 0.005          # mean not hurt
    graceful = conf["psr_lora"][0] >= conf["buffer_rrr_uniform"][0] - 0.005          # no worse than uniform solve
    print(f"[off-floor] compressible mean not hurt: {'PASS' if no_harm else 'FAIL'}; "
          f"conflicting graceful (>= uniform solve): {'PASS' if graceful else 'FAIL'}")
    return no_harm and graceful


def resource_ledger(verbose=True):
    """Honest per-arm resource ledger (concern: PSR-LoRA is NOT compute-matched).

    Every arm holds the SAME bounded buffer (B=16 windows) -- reported in bytes via
    the shared harness ledger so the memory budget is provably matched.  Compute is
    NOT matched and we say so: PSR-LoRA re-solves the closed-form weighted rank-R RRR
    `rounds+1` times per window (one SVD each), whereas buffer_rrr_uniform solves it
    once and the replay/gradient arms run `gd_steps` factored GD steps with no SVD.
    We report both the analytic solve/step count and measured wall-clock on one world
    so the tail gain is never mistaken for a free lunch."""
    import time
    cfg = _cfg()
    d = cfg["d"]; batch = cfg["batch"]; R = cfg["R"]
    cap = cfg["buf_u"] + cfg["buf_g"]
    buf_bytes = H.ledger_bytes(d=d, R=R, buffer_windows=cap, batch=batch)
    n_win = cfg["n_windows"]
    rounds = P.DEFAULT["rounds"]; gd = cfg["gd_steps"]
    # analytic per-window primitive counts (SVD = weighted_rrr call; GDstep = factor step)
    counts = {
        "psr_lora":           f"{rounds+1} weighted-RRR SVD/window",
        "buffer_rrr_uniform": "1 weighted-RRR SVD/window",
        "excess_cvar":        f"{gd} GD steps/window (+ up to ceil(tau*B) replay windows)",
        "uniform_replay":     f"{gd} GD steps/window (+1 replay window)",
        "agem":               f"{gd} GD steps/window (+ buffer-grad projection)",
        "sequential":         f"{gd} GD steps/window (no replay)",
    }
    # measured wall-clock on ONE held-out world (teacher/stream held fixed for all arms)
    teacher = BM.build_teacher(REGIME, cfg, TEACHER_SEEDS[0])
    windows, freqs = BM.make_stream(cfg, STREAM_SEEDS[0])
    batches, srcs, otr, test = BM.materialize(teacher, cfg, windows, STREAM_SEEDS[0])
    secs = {}
    for a in counts:
        t0 = time.perf_counter()
        ARMS[a](cfg, batches, RUN_SEED)
        secs[a] = time.perf_counter() - t0
    if verbose:
        print("\n================ RESOURCE LEDGER (memory matched, compute NOT) ================")
        print(f"  buffer budget (ALL arms): {cap} windows = {buf_bytes} bytes "
              f"(d={d}, batch={batch}); adapter 2*d*R={2*d*R} floats")
        print(f"  stream length: {n_win} windows/world")
        for a in counts:
            print(f"  {a:20s} {secs[a]*1e3/n_win:7.2f} ms/window   [{counts[a]}]")
        slow = max(secs, key=secs.get); fast = min(secs, key=secs.get)
        print(f"  -> PSR-LoRA trades compute for the tail: {secs['psr_lora']/secs[fast]:.1f}x "
              f"the wall-clock of the cheapest arm ({fast}); this is disclosed, not matched.")
    return {"buf_bytes": buf_bytes, "counts": counts, "secs": secs}


def verdict():
    mean, tail, off_tail, W = run_worlds()
    A_mean = {a: np.array(mean[a]) for a in ARMS}
    A_tail = {a: np.array(tail[a]) for a in ARMS}

    def nm(x):
        return float(np.nanmean(x)) if np.any(~np.isnan(x)) else float("nan")

    print("\n================ per-arm summary (mean +/- sd over %d worlds) ================" % W)
    order = sorted(ARMS, key=lambda a: -(nm(A_tail[a]) if not np.isnan(nm(A_tail[a])) else -9))
    for a in order:
        ndiv = int(np.isnan(A_tail[a]).sum())
        div = f"  [diverged {ndiv}/{W}]" if ndiv else ""
        print(f"  {a:20s} mean {nm(A_mean[a]):.3f}+-{np.nanstd(A_mean[a]):.3f}   "
              f"tail {nm(A_tail[a]):.3f}+-{np.nanstd(A_tail[a]):.3f}   "
              f"(tail/offline {nm(A_tail[a])/off_tail.mean():.2f}){div}")
    print(f"  {'OFFLINE oracle':20s} tail {off_tail.mean():.3f} (reachable best)")

    # strongest baseline by mean (for (a)) and the replay tail bar excess_cvar (for (b)).
    # A baseline that diverged in any world is disqualified from being the "strongest"
    # (we cannot pair against NaN); this only makes (a) harder for PSR, never easier.
    baselines_only = [a for a in ARMS if a not in ("psr_lora", "buffer_rrr_uniform", "psr_grad_defensive")
                      and not np.any(np.isnan(A_mean[a]))]
    best_base_mean_arm = max(baselines_only, key=lambda a: A_mean[a].mean())
    psr = "psr_lora"

    n_t, n_s = len(TEACHER_SEEDS), len(STREAM_SEEDS)

    def _report(tag, diffs, strict):
        """Print IID (transparency) and cluster-robust (decision) CIs for one
        criterion.  strict=True requires the cluster-robust lower bound > 0;
        strict=False (criterion a) requires cluster-robust mean >= -MEAN_MARGIN.
        Also prints the per-world win split so a paired-average effect is never
        read as uniform dominance."""
        m_iid, lo_iid, hi_iid = _paired_ci(diffs)
        m, lo, hi, se, comps, df = _paired_ci_crossed(diffs, n_t, n_s)
        wins = int(np.sum(np.asarray(diffs) > 0)); N = len(diffs)
        sa2, sb2, se2 = comps
        passed = (lo > 0.0) if strict else (m >= -MEAN_MARGIN)
        print(f"{tag} -> {'PASS' if passed else 'FAIL'}")
        print(f"      paired mean {m:+.4f};  wins {wins}/{N}")
        print(f"      IID 95%CI      [{lo_iid:+.4f},{hi_iid:+.4f}]   (n={N}, NOT used for decision)")
        print(f"      cluster 95%CI  [{lo:+.4f},{hi:+.4f}]   (t_df={df}, se={se:.4f}; "
              f"var comps teacher/stream/resid {sa2:.2e}/{sb2:.2e}/{se2:.2e})")
        return passed

    print("\n================ PRE-REGISTERED KILL CRITERION ================")
    print("(decision uses the CLUSTER-ROBUST CI for the 5x4 crossed design; the IID")
    print(" CI is shown only to expose how much clustering widens the interval)")
    # (a) non-inferior mean vs strongest-mean baseline, paired
    da = A_mean[psr] - A_mean[best_base_mean_arm]
    a_pass = _report(f"(a) NON-INFERIOR MEAN vs {best_base_mean_arm} (margin -{MEAN_MARGIN})",
                     da, strict=False)
    # (b) strictly better tail vs excess_cvar, cluster-robust 95% CI excludes 0
    db = A_tail[psr] - A_tail["excess_cvar"]
    b_pass = _report("(b) STRICTLY BETTER TAIL vs excess_cvar", db, strict=True)
    # (c) mechanism contributes vs uniform buffer-RRR, cluster-robust 95% CI excludes 0
    dc = A_tail[psr] - A_tail["buffer_rrr_uniform"]
    c_pass = _report("(c) MECHANISM CONTRIBUTES vs buffer_rrr_uniform", dc, strict=True)

    keep = a_pass and b_pass and c_pass
    print("\nVERDICT:", "KEEP the method line (all three pass)" if keep
          else "STOP / revise -- a pre-registered criterion failed")

    print("\n================ OFF-FLOOR ROBUSTNESS (no-harm / graceful) ================")
    rob = robustness_offfloor()

    resource_ledger()

    return keep, {"a": a_pass, "b": b_pass, "c": c_pass, "robust": rob}


if __name__ == "__main__":
    import sys
    keep, _ = verdict()
    sys.exit(0 if keep else 1)
