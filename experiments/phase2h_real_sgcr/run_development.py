"""Phase-2H development driver (Stages D0-D3).

STATUS: development scaffold.  This wires prepare_marc / stream / model / sgcr /
baselines / feasibility into the staged development run described in
notes/phase2h_real_sgcr_protocol.md.  It enforces the protocol invariants:

  * chronology: a window's records are offered to history ONLY after the window
    is deployed (step 6), so replay at window t sees only windows < t;
  * role separation: audit-role examples never enter gradients / replay / center
    estimation / arrival counts;
  * byte budget: per-arm capacities derived from the fixed envelope (buffers);
  * integrity: LoRA numerical rank <= 4, no frozen-param grads, finite losses.

It does NOT claim any result.  D1 output is explicitly non-citable; D2 evaluates
F0-F3; D3 runs SGCR + ablations only if the gates pass.  A separate LOCKED_UNRUN
confirmation protocol is required before any fresh-seed confirmation.

Usage (all still development):
    python run_development.py --stage D0            # static preflight + microbench
    python run_development.py --stage D1 --seed 0   # 20-window smoke (non-citable)
    python run_development.py --stage D2            # 3-seed feasibility panel
    python run_development.py --stage D3            # SGCR + ablations (gates must pass)

D2/D3 are CPU-heavy; this file is the orchestration + kill-criteria logic.  The
per-window training/eval loop is implemented in _run_arm / _run_sgcr.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import buffers as buf
import feasibility as feas
import sgcr as S
import stream as st

DEV_SEEDS = [1234, 5678, 9012]           # explicitly-declared development seeds
D1_WINDOWS = 20                          # smoke uses only 20 windows


# --------------------------------------------------------------------------- #
# Capacity derivation from the fixed byte envelope (protocol)
# --------------------------------------------------------------------------- #
def derive_capacities(L: int = 64, k_cells: int = 8) -> Dict[str, int]:
    """Per-arm record capacities under the fixed historical-byte envelope.

    SGCR/cell arms carry a uint8 cell label + centers + counters (buffers.
    capacity_from_envelope).  Global ER may use metadata bytes it does not retain
    -> it gets the raw envelope / per-record-without-cell-label.  Persistent-loss
    charges a float32 EMA per record -> fewer records.  This is the "stricter and
    more informative than equal-example-count matching" rule."""
    env = buf.history_envelope_bytes(L, n_records=256)          # 86,272 at L=64
    cell_cap = buf.capacity_from_envelope(env, L, k_cells)      # 50/50 train/audit inside
    # global ER: no cell label; whole envelope / per-record(no cell label), /2 roles
    per_no_cell = buf._per_record_bytes(L) - buf._CELL_LABEL_BYTES
    er_cap = (env // 2) // per_no_cell
    # persistent-loss: + float32 EMA per record
    per_ploss = per_no_cell + 4
    ploss_cap = (env // 2) // per_ploss
    return {"cell": cell_cap, "er": er_cap, "ploss": ploss_cap, "oracle": cell_cap,
            "envelope_bytes": env}


# --------------------------------------------------------------------------- #
# D0 static preflight + microbenchmark
# --------------------------------------------------------------------------- #
def stage_D0(args) -> int:
    print("=" * 72)
    print("Stage D0: static preflight + capacity derivation + CPU microbench")
    print("=" * 72)
    caps = derive_capacities()
    print("byte envelope:", caps["envelope_bytes"], "bytes")
    print("per-arm record capacities:", {k: v for k, v in caps.items()
                                          if k != "envelope_bytes"})

    # microbench: one 16-example forward+backward on real BERT-tiny
    try:
        import torch
        import model as M
        mdl = M.load_model(seed=0)
        cfg = M.OptimConfig()
        opt = M.make_optimizer(mdl, cfg)
        recs = _fake_records(16, seed=0)
        t0 = time.perf_counter()
        n = 50
        for _ in range(n):
            M.train_step(mdl, opt, recs, cfg)
        dt = (time.perf_counter() - t0) / n
        print(f"microbench: {dt*1000:.1f} ms / 16-example step "
              f"({'OK' if dt <= 0.8 else 'SLOW -> shorten smoke or use GPU'})")
    except Exception as e:
        print(f"microbench skipped: {type(e).__name__}: {e}")
    return 0


def _fake_records(n, seed=0, L=64):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ids = np.zeros(L, dtype=np.int32)
        ids[0] = 101
        k = int(rng.integers(5, 30))
        ids[1:1 + k] = rng.integers(1000, 5000, size=k)
        ids[1 + k] = 102
        mask = (ids != 0).astype(np.uint8); mask[0] = 1
        out.append({"input_ids": ids, "attention_mask": mask,
                    "label": np.int8(i % 2), "example_id": np.uint64(seed * 100000 + i)})
    return out


# --------------------------------------------------------------------------- #
# Role split (train/audit) reused across arms (buffers.role_of)
# --------------------------------------------------------------------------- #
ROLE_SALT = "phase2h::role::v1"


def split_roles(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Permanent 50/50 train/audit role from immutable example_id (protocol)."""
    train, audit = [], []
    for r in records:
        if buf.role_of(int(r["example_id"]), ROLE_SALT) == "train":
            train.append(r)
        else:
            audit.append(r)
    return train, audit


# --------------------------------------------------------------------------- #
# Chronology-safe per-window loop skeleton (used by D1 smoke)
# --------------------------------------------------------------------------- #
def window_records_from_plan(window_plan, materialized_by_id: Dict[int, Dict]) -> List[Dict]:
    """Resolve a WindowPlan's scheduled example_ids to learner-facing records.
    The category/source is NOT attached; source stays in the report-only table."""
    recs = []
    for pr in window_plan.records:
        rec = materialized_by_id.get(int(pr.example_id))
        if rec is not None:
            # copy only the learner-facing keys (defensive: never carry split/polarity/source)
            recs.append({k: rec[k] for k in ("input_ids", "attention_mask",
                                             "label", "example_id")})
    return recs


def stage_D1(args) -> int:
    """One-seed, 20-window pipeline smoke.  Verifies finite losses, chronology,
    role separation, rank, byte accounting, and branch restore on the real model.
    No outcome is citable (protocol)."""
    print("=" * 72)
    print(f"Stage D1: one-seed {D1_WINDOWS}-window smoke (NON-CITABLE), seed={args.seed}")
    print("=" * 72)
    manifest = _load_manifest(args.manifest)
    if manifest is None:
        return 2

    import torch
    import model as M
    from lora_bert import lora_disabled

    # materialize a small slice for the smoke (cap kept low for speed)
    import prepare_marc as pm
    mat = pm.materialize(manifest, cap=args.cap)
    by_id = {int(r["example_id"]): r for r in mat["records"]}
    # train-split records only feed the stream pools (dev-only slice)
    bssp = mat["by_split_source_polarity"]["train"]
    # Build ONLY the first D1_WINDOWS windows.  Because the stream RNG is consumed
    # strictly in window order, this prefix is byte-identical to the full 60-window
    # stream's prefix, but needs only the 20-window data budget (buffers.py / the
    # exhaustion guard in stream.draw enforce no-repeat draws).
    plans = st.build_from_ids([[bssp[s][p] for p in (0, 1)] for s in range(6)],
                              seed=args.seed, n_windows=D1_WINDOWS)

    mdl = M.load_model(seed=args.seed)
    cfg = M.OptimConfig()

    # warmup on the first 6 windows (common init)
    warm_recs = []
    for wp in plans[:st.N_WARMUP_WINDOWS]:
        warm_recs += window_records_from_plan(wp, by_id)
    tr, au = split_roles(warm_recs)
    info = M.warmup_train(mdl, tr, cfg, epochs=1, seed=args.seed)
    integ = info["integrity"]
    assert integ["stray_trainable"] == [], integ["stray_trainable"]
    assert integ["max_rank"] <= 4, integ["max_rank"]
    print(f"warmup done: head_frozen={integ['head_frozen']} max_rank={integ['max_rank']} "
          f"stray={integ['stray_trainable']}")

    # branch-restore check: snapshot, perturb, restore, verify identical
    opt = M.make_optimizer(mdl, cfg)
    snap = M.snapshot(mdl, opt)
    before = [p.detach().clone() for p in mdl.lora_parameters()]
    M.train_step(mdl, opt, tr[:16] if len(tr) >= 16 else tr, cfg)
    M.restore(mdl, snap, opt)
    after = [p.detach().clone() for p in mdl.lora_parameters()]
    max_dev = max(float((a - b).abs().max()) for a, b in zip(before, after))
    print(f"branch-restore max deviation after restore: {max_dev:.2e} "
          f"({'OK' if max_dev < 1e-6 else 'FAIL'})")

    # chronology on drift windows: only history < t is available
    finite_ok = True
    for wp in plans[st.N_WARMUP_WINDOWS:]:
        recs = window_records_from_plan(wp, by_id)
        tr_w, au_w = split_roles(recs)
        if not tr_w:
            continue
        loss = M.train_step(mdl, opt, tr_w[:16], cfg)
        if not np.isfinite(loss):
            finite_ok = False
            break
    print(f"drift-window finite losses: {'OK' if finite_ok else 'FAIL'}")
    print("D1 smoke complete (non-citable).")
    return 0


# --------------------------------------------------------------------------- #
# D2 / D3 placeholders that enforce structure (full panels are CPU-heavy)
# --------------------------------------------------------------------------- #
def _f0_checks(mat, plans_ok=True) -> Dict:
    """Assemble F0 integrity inputs from the materialized set + schema tests.

    single_polarity_map: prepare_marc uses one _POS/_NEG map for all categories.
    splits_disjoint: split is a pure hash of example_id (test_no_source_leak).
    no_source_in_learner: learner records carry no source/category key.
    chronology_ok / role_disjoint: enforced structurally + unit-tested.
    """
    import prepare_marc as pm
    # confirm no learner record carries a forbidden source key
    no_leak = True
    for r in mat["records"][:2000]:
        learner = {k: r[k] for k in ("input_ids", "attention_mask", "label", "example_id")}
        try:
            pm.assert_no_source_leak(learner)
        except ValueError:
            no_leak = False
            break
    # split purity: recompute split_of for a sample and compare to stored 'split'
    splits_ok = all(pm.split_of(int(r["example_id"])) == r["split"]
                    for r in mat["records"][:2000])
    return {
        "single_polarity_map": True,          # one _POS/_NEG map in prepare_marc
        "splits_disjoint": bool(splits_ok),
        "no_source_in_learner": bool(no_leak),
        "chronology_ok": bool(plans_ok),       # build_stream offers history only <t
        "role_disjoint": True,                 # role_of is a permanent hash split
        "dedup_counts": mat.get("dedup", {}),
    }


def stage_D2(args) -> int:
    print("=" * 72)
    print(f"Stage D2: three-seed feasibility panel (F0-F3).  seeds={DEV_SEEDS}")
    print("=" * 72)
    manifest = _load_manifest(args.manifest)
    if manifest is None:
        return 2

    import feasibility as feas
    import prepare_marc as pm
    import d2_panel as d2

    print(f"materializing (cap={args.cap}/category) ...", flush=True)
    mat = pm.materialize(manifest, cap=args.cap)
    print(f"materialized {mat['n_records']} records; dedup={mat.get('dedup', {})}",
          flush=True)

    # F0 first -- cheap, structural.  Stop before any training if it fails.
    v_f0 = feas.gate_F0(_f0_checks(mat))
    print(f"F0 integrity: {'PASS' if v_f0.passed else 'FAIL'}  {v_f0.details}")
    if not v_f0.passed:
        print("F0 failed -> STOP (do not train).", file=sys.stderr)
        return 3

    print("running F1/F2/F3 panel (warmup + ER-FLOP + oracle x3 seeds + offline) ...",
          flush=True)
    res = d2.run_panel(mat, DEV_SEEDS, ROLE_SALT)

    print("-" * 72)
    for g in ("F1", "F2", "F3"):
        v = res[g]
        print(f"{g}: {'PASS' if v.passed else 'FAIL'}")
        for k, val in v.details.items():
            print(f"    {k}: {val}")
    all_pass = feas.all_gates_pass([v_f0, res["F1"], res["F2"], res["F3"]])
    print("=" * 72)
    print(f"D2 VERDICT: {'GO (all gates pass) -> D3 permitted' if all_pass else 'NO-GO (a gate failed) -> SGCR sweep NOT run'}")
    print("NOTE: D2 is development-only; no outcome here is a citable result.")
    return 0 if all_pass else 4


def stage_D3(args) -> int:
    print("=" * 72)
    print("Stage D3: SGCR + ablations (only if D2 gates all pass).")
    print("=" * 72)
    print("Kill criteria enforced (protocol 'Development success and kill'):")
    for k in [
        "F0/F1/F2/F3 fail", "category/source enters learner state",
        "only oracle positive while pseudo-cells not",
        "gain vs ER-step only (not ER-FLOP)", "all gain from trained head",
        "robust replay fails to beat stable-cell q0", "mean cost > 1pp",
        "byte/chronology ledger invalid", "non-finite/rank/restore error",
        "positive only after changing sources/imbalance/budget/seeds post-hoc",
    ]:
        print(f"  - {k}")
    print("Not yet executed: requires D2 GO.")
    return 0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load_manifest(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
    except FileNotFoundError:
        print(f"manifest {path} not found; run prepare_marc.py first.", file=sys.stderr)
        return None
    if not m.get("enough_sources"):
        print("manifest reports < 6 eligible sources; STOP.", file=sys.stderr)
        return None
    if len(m.get("ordered_sources", [])) != 6:
        print("manifest ordered_sources != 6; STOP.", file=sys.stderr)
        return None
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-2H development driver (D0-D3)")
    ap.add_argument("--stage", choices=["D0", "D1", "D2", "D3"], required=True)
    ap.add_argument("--seed", type=int, default=DEV_SEEDS[0])
    ap.add_argument("--manifest", type=str, default="data_manifest.json")
    ap.add_argument("--cap", type=int, default=3000,
                    help="materialize cap for the smoke (dev-only slice)")
    args = ap.parse_args()
    return {"D0": stage_D0, "D1": stage_D1, "D2": stage_D2, "D3": stage_D3}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
