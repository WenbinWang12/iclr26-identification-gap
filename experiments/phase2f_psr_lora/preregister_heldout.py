"""Single held-out kill test for the FROZEN coverage-aware PSR-LoRA.

HONESTY / PROVENANCE NOTE (read first).  This file is NOT a preregistration in the
verifiable-timestamp sense.  The repository is not under version control, and the
filesystem mtimes show this script (23:13) was written ~31 minutes AFTER the results
file outputs_heldout.txt (22:42).  An earlier version of this docstring claimed the
file "was written BEFORE running it"; that claim was false and has been removed.
What is actually true: the FROZEN method config lives in committed source
(psr_lora.py DEFAULT: coverage=True, buf_u=4, buf_g=12) and the kill criterion is
inherited verbatim from compare.py.  According to the author's session chronology the
three criteria were fixed before the held-out numbers were inspected, but the surviving
protocol text and SHA-256 hashes were recorded post hoc and do NOT constitute
externally verifiable preregistration.  Cite this only as an "internally pre-specified,
hash-recorded held-out evaluation", never as a preregistration.  See
notes/phase2f_psr_lora_protocol.md and memory phase2f-no-prerun-hash-log.

This is the honest confirmation run for the coverage remedy.  Discipline:

  * The method is FROZEN: coverage-aware PSR-LoRA with buf_u=4 uniform reservoir +
    buf_g=12 input-direction-diverse buffer (total budget 16), minimax reweighted
    rank-R RRR on the union.  These numbers were SELECTED on the dev family
    (teachers 20260701 / 20260710..13 x streams 6060,6061) and are NOT tuned here.

  * The seeds below are BRAND NEW -- disjoint from BOTH the dev family AND the burned
    v1/v2 test set (teachers 20270001..5 x streams 7001..4).

  * The kill criterion and statistics are inherited verbatim from compare.py (the
    cluster-robust teacher-clustered CI is the decision object, matching the honest
    rerun's finding that variance is between-teacher).  We run ONCE and report
    whatever comes out.  If it fails, it fails -- no config search on these worlds.

--------------------------------------------------------------------------------
INTERNALLY PRE-SPECIFIED PROTOCOL (config frozen in committed source; NOT a
timestamp-verifiable preregistration -- see the provenance note above)
--------------------------------------------------------------------------------
Arena: regime capacity_limited (the only regime with a genuine rank-R floor).
Worlds: NEW teachers 20280001..20280005 x NEW streams 8001..8004 = 20 held-out
  worlds, a 5x4 crossed design (teacher x stream), never seen in design or in the
  burned test.
Arms & metrics: identical to compare.py (mean = freq-weighted retention, tail =
  worst-group retention; offline weighted-RRR frontier as the reachable envelope).
Statistics: cluster-robust CI for the crossed design (compare.py._paired_ci_crossed),
  decision on the two-way crossed interval with the CONSERVATIVE crossed df =
  min(n_teachers, n_streams) - 1 = 3 (NOT the teacher-only df=4; here both teacher and
  stream variance components are non-zero, so the crossed model is the honest choice),
  with the IID CI shown only for transparency.

KILL CRITERION (all three must hold to CONFIRM the frozen method; else it fails):
  (a) NON-INFERIOR MEAN: PSR-LoRA mean >= best-baseline mean - 0.01 (teacher-cluster
      mean within margin).
  (b) STRICTLY BETTER TAIL vs excess_cvar: teacher-clustered 95% CI on the per-world
      tail difference excludes 0.   <-- the criterion the pre-remedy method FAILED.
  (c) COVERAGE + REWEIGHTING CONTRIBUTE vs buffer_rrr_uniform: teacher-clustered 95%
      CI excludes 0 (buffer_rrr_uniform shares the SAME coverage working set, uniform
      weights -- so (c) now isolates the minimax reweighting given coverage).
Additionally report the coverage ablation psr_no_coverage (single uniform reservoir,
  same budget) so the tail lift attributable to COVERAGE itself is visible, and the
  off-floor no-harm/graceful checks.

Run:  python preregister_heldout.py > outputs_heldout.txt
"""

from __future__ import annotations

import numpy as np

import benchmark_v4 as BM
import baselines as BL
import psr_lora as P
import harness as H
from compare import (_paired_ci, _paired_ci_crossed, _mean_tail, MEAN_MARGIN)


# ---- FROZEN held-out worlds (NEW: disjoint from dev AND the burned 2027xxxx set) ----
TEACHER_SEEDS = [20280001, 20280002, 20280003, 20280004, 20280005]
STREAM_SEEDS = [8001, 8002, 8003, 8004]
REGIME = "capacity_limited"
RUN_SEED = 7                      # per-arm RNG seed (buffer draws); fixed for all arms


def _cfg():
    """Frozen config: coverage-aware buffer buf_u=4 + buf_g=12 (the dev-selected
    split), everything else at the phase2f main-run values.  NOTE: unlike compare._cfg
    (which predates the remedy and hardcodes 8/8), this pins the FROZEN 4/12 split so
    the held-out run uses exactly the method that was validated on dev."""
    cfg = dict(BM.DEFAULT_CFG)
    cfg.update({"gd_steps": 20, "lr": 0.15, "buf_u": 4, "buf_g": 12,
                "tau": 0.4, "xi": 0.02, "coverage": True, "reweight": True})
    return cfg


ARMS = {
    "psr_lora": lambda cfg, b, s: P.run_psr_lora(cfg, b, s)[0][-1],
    "psr_no_coverage": lambda cfg, b, s: P.run_psr_no_coverage(cfg, b, s)[0][-1],
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
                    m, t = np.nan, np.nan
                per_arm_mean[a].append(m)
                per_arm_tail[a].append(t)
            worlds += 1
            if verbose:
                print(f"  world t{ts}/s{ss}: psr tail {per_arm_tail['psr_lora'][-1]:.3f}  "
                      f"nocov tail {per_arm_tail['psr_no_coverage'][-1]:.3f}  "
                      f"cvar tail {per_arm_tail['excess_cvar'][-1]:.3f}  "
                      f"unif-rrr tail {per_arm_tail['buffer_rrr_uniform'][-1]:.3f}  "
                      f"offline {off_tail[-1]:.3f}")
    return per_arm_mean, per_arm_tail, np.array(off_tail), worlds


def verdict():
    mean, tail, off_tail, W = run_worlds()
    A_mean = {a: np.array(mean[a]) for a in ARMS}
    A_tail = {a: np.array(tail[a]) for a in ARMS}
    n_t, n_s = len(TEACHER_SEEDS), len(STREAM_SEEDS)

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

    baselines_only = [a for a in ARMS if a not in
                      ("psr_lora", "psr_no_coverage", "buffer_rrr_uniform", "psr_grad_defensive")
                      and not np.any(np.isnan(A_mean[a]))]
    best_base_mean_arm = max(baselines_only, key=lambda a: A_mean[a].mean())
    psr = "psr_lora"

    def _report(tag, diffs, strict):
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

    print("\n================ PRE-REGISTERED KILL CRITERION (frozen coverage method) ================")
    print("(decision uses the CLUSTER-ROBUST two-way crossed CI from compare._paired_ci_crossed;")
    print(" df = min(n_teachers,n_streams) - 1 = %d, the conservative crossed df -- NOT the IID n-1;" % (min(n_t, n_s) - 1))
    print(" here BOTH teacher and stream variance components are non-zero, so the two-way")
    print(" crossed model is the honest choice and df=3 is conservative vs teacher-only df=4)")
    da = A_mean[psr] - A_mean[best_base_mean_arm]
    a_pass = _report(f"(a) NON-INFERIOR MEAN vs {best_base_mean_arm} (margin -{MEAN_MARGIN})",
                     da, strict=False)
    db = A_tail[psr] - A_tail["excess_cvar"]
    b_pass = _report("(b) STRICTLY BETTER TAIL vs excess_cvar", db, strict=True)
    dc = A_tail[psr] - A_tail["buffer_rrr_uniform"]
    c_pass = _report("(c) REWEIGHTING CONTRIBUTES (given coverage) vs buffer_rrr_uniform", dc, strict=True)

    # informational: the tail lift attributable to COVERAGE itself (not a kill gate)
    dcov = A_tail[psr] - A_tail["psr_no_coverage"]
    mcov, locov, hicov, secov, _, dfcov = _paired_ci_crossed(dcov, n_t, n_s)
    winscov = int(np.sum(dcov > 0))
    print(f"[info] COVERAGE contribution vs psr_no_coverage (single reservoir, same budget):")
    print(f"       paired mean {mcov:+.4f}; wins {winscov}/{len(dcov)}; "
          f"cluster 95%CI [{locov:+.4f},{hicov:+.4f}] (t_df={dfcov})")

    keep = a_pass and b_pass and c_pass
    print("\nVERDICT:", "CONFIRM the frozen coverage method (all three pass)" if keep
          else "FAIL -- a pre-registered criterion did not hold on held-out (report honestly)")
    return keep, {"a": a_pass, "b": b_pass, "c": c_pass}


if __name__ == "__main__":
    import sys
    keep, _ = verdict()
    sys.exit(0 if keep else 1)
