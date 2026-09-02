"""Phase-2H Stage-D2 three-seed feasibility panel engine (F0-F3).

STATUS: development scaffold, real computation (NOT a placeholder).  Implements
the protocol's Stage-D2:

    "Run frozen signatures, source probe/k-means diagnostics, offline joint,
     ER-FLOP, and true-category oracle.  Evaluate F0--F3."

Nothing here is citable as a result; D2 only decides whether SGCR (D3) may run.
All source labels used are EVALUATOR-ONLY (probe / k-means diagnostics / oracle /
per-source metrics) and never enter a learner-facing record, buffer, or model
call -- the deployable arms (ER-FLOP) receive only tokens + polarity.

Compute is bounded: per seed one common warmup, then two continual arms
(ER-FLOP, oracle) over the 54 drift windows at 16 steps/window, plus one
offline-joint fit.  On the D0 microbench (~40 ms / 16-example step) this is a few
minutes per seed on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import baselines as B
import feasibility as feas
import sgcr as S
import stream as st

# The globally rare source is index 5 (probability 0.01 in every phase vector).
RARE_SOURCE = 5
EVAL_PER_SOURCE_CAP = 600           # validation records / source used in eval (disclosed)
OFFLINE_EPOCHS = 3                  # offline-joint passes over train (dev upper bound)


# --------------------------------------------------------------------------- #
# Evaluation: per-source balanced accuracy of polarity predictions
# --------------------------------------------------------------------------- #
def _predict_polarity(model, records, batch_size=64) -> np.ndarray:
    """LoRA-ENABLED eval-mode polarity predictions for a list of records."""
    import torch
    import model as M
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            input_ids, attention_mask, _ = M.collate(chunk)
            logits = model(input_ids, attention_mask)
            preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds) if preds else np.zeros(0, dtype=np.int64)


def per_source_balacc(model, val_by_source: Dict[int, List[Dict]]) -> Dict[int, float]:
    """Balanced accuracy (over the two polarity classes) within each source's
    validation records.  Evaluator-only: grouping by source uses the report-only
    table, never a learner input."""
    out: Dict[int, float] = {}
    for s, recs in val_by_source.items():
        if not recs:
            continue
        y_true = np.array([int(r["label"]) for r in recs], dtype=np.int64)
        y_pred = _predict_polarity(model, recs)
        out[s] = feas.balanced_accuracy(y_true, y_pred, n_classes=2)
    return out


def worst_and_macro(bal: Dict[int, float]) -> Tuple[float, float]:
    vals = list(bal.values())
    if not vals:
        return 0.0, 0.0
    return float(min(vals)), float(np.mean(vals))


# --------------------------------------------------------------------------- #
# Continual arm loop (ER-FLOP, oracle) -- chronology + role separation
# --------------------------------------------------------------------------- #
def _window_records(wp, by_id) -> List[Dict]:
    recs = []
    for pr in wp.records:
        r = by_id.get(int(pr.example_id))
        if r is not None:
            recs.append(r)
    return recs


def _split_roles(records, role_salt):
    import buffers as buf
    tr, au = [], []
    for r in records:
        if buf.role_of(int(r["example_id"]), role_salt) == "train":
            tr.append(r)
        else:
            au.append(r)
    return tr, au


def run_continual_arm(model, cfg, arm, plans, by_id, source_table,
                      role_salt, seed) -> None:
    """Train ``arm`` continually over the drift windows in ``plans`` (warmup
    windows already applied to the common checkpoint).  Chronology: a window's
    records are offered to the buffer ONLY after that window's updates deploy.
    Role separation: only train-role records enter gradients / buffers.  The
    oracle arm additionally receives evaluator-only source labels for grouping."""
    import model as M
    rng = np.random.default_rng(seed ^ 0x5A17)
    opt = M.make_optimizer(model, cfg)

    for wp in plans:
        if wp.window < st.N_WARMUP_WINDOWS:
            continue  # warmup already trained into the common checkpoint
        recs = _window_records(wp, by_id)
        cur_train, _cur_audit = _split_roles(recs, role_salt)

        for _ in range(arm.n_deploy_steps):
            replay = arm.sample_replay(8, rng) if arm.buffer is not None else []
            if replay:
                take = min(8, len(cur_train))
                cur_idx = rng.choice(len(cur_train), size=take, replace=False) \
                    if take > 0 else []
                batch = [cur_train[int(j)] for j in cur_idx] + replay
            else:
                batch = cur_train  # first windows / sequential: current only
            if batch:
                M.train_step(model, opt, batch, cfg)

        # deploy happens implicitly (model state advanced); NOW offer to history.
        if arm.buffer is not None and cur_train:
            if arm.is_oracle:
                srcs = [int(source_table[int(r["example_id"])]) for r in cur_train]
                arm.buffer.offer(cur_train, srcs, rng)
            else:
                arm.buffer.offer(cur_train, rng)


# --------------------------------------------------------------------------- #
# Offline-joint rank-4 LoRA (F3 upper bound) + LoRA on/off probe
# --------------------------------------------------------------------------- #
def offline_joint(train_records, val_by_source, cfg, seed) -> Dict:
    """Fit a fresh warmup+LoRA model jointly on ALL train records (report-only
    unbounded-data capacity).  Returns macro/worst BA, a random-init baseline BA,
    the LoRA on/off logit difference and the LoRA-disabled performance drop, and
    integrity (grads finite nonzero, numerical rank, frozen params unchanged)."""
    import torch
    import model as M
    from lora_bert import lora_disabled

    rng = np.random.default_rng(seed ^ 0x0FF1)

    # Random-init baseline: fresh model, head+LoRA untrained -> chance-ish BA.
    rnd = M.load_model(seed=seed)
    rnd_bal = per_source_balacc(rnd, val_by_source)
    _, random_macro = worst_and_macro(rnd_bal)

    # Offline-joint: warmup then several joint epochs over all train records.
    mdl = M.load_model(seed=seed)
    info = M.warmup_train(mdl, train_records, cfg, epochs=1, seed=seed)
    warm_bal = per_source_balacc(mdl, val_by_source)
    _, warm_macro = worst_and_macro(warm_bal)

    opt = M.make_optimizer(mdl, cfg)
    grad_finite_nonzero = True
    for _ in range(OFFLINE_EPOCHS):
        for batch in M._iter_minibatches(train_records, cfg.batch_size, rng):
            M.train_step(mdl, opt, batch, cfg)
    # one more step to inspect gradient finiteness/non-zero-ness
    if len(train_records) >= cfg.batch_size:
        b = train_records[:cfg.batch_size]
        input_ids, attention_mask, labels = M.collate(b)
        opt.zero_grad(set_to_none=True)
        logits = mdl(input_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        gnorms = [float(p.grad.norm()) for p in mdl.lora_parameters()
                  if p.grad is not None]
        grad_finite_nonzero = bool(gnorms) and all(np.isfinite(g) for g in gnorms) \
            and any(g > 0 for g in gnorms)
        opt.zero_grad(set_to_none=True)

    off_bal = per_source_balacc(mdl, val_by_source)
    off_worst, off_macro = worst_and_macro(off_bal)

    # LoRA on/off: logit difference and performance drop with LoRA disabled.
    some = train_records[:cfg.batch_size] if len(train_records) >= cfg.batch_size \
        else train_records
    input_ids, attention_mask, _ = M.collate(some)
    mdl.eval()
    with torch.no_grad():
        logits_on = mdl(input_ids, attention_mask)
        with lora_disabled(mdl.bert):
            logits_off = mdl(input_ids, attention_mask)
    logit_diff = float((logits_on - logits_off).abs().mean())

    # perf drop: macro BA with LoRA disabled vs enabled
    class _Disabled:
        def __init__(self, m): self.m = m
        def __enter__(self):
            self.cm = lora_disabled(self.m.bert); self.cm.__enter__(); return self.m
        def __exit__(self, *a): self.cm.__exit__(*a)
    with _Disabled(mdl):
        off_bal_nolora = per_source_balacc(mdl, val_by_source)
    _, macro_nolora = worst_and_macro(off_bal_nolora)
    perf_drop = float(off_macro - macro_nolora)

    integ = mdl.integrity()
    frozen_unchanged = (integ["stray_trainable"] == []) and integ.get("head_frozen", False)
    max_rank = int(integ.get("max_rank", 99))

    return {
        "random_macro": random_macro,
        "warmup_macro": warm_macro,
        "offline_macro": off_macro,
        "offline_worst": off_worst,
        "offline_by_source": off_bal,
        "lora_logit_diff": logit_diff,
        "lora_perf_drop": perf_drop,
        "grad_finite_nonzero": grad_finite_nonzero,
        "max_rank": max_rank,
        "frozen_unchanged": frozen_unchanged,
        "warmup_integrity": info["integrity"],
    }


# --------------------------------------------------------------------------- #
# F1 signatures + probe + k-means (development validation only)
# --------------------------------------------------------------------------- #
def f1_diagnostics(model, warmup_mean, val_by_source, seed) -> Dict:
    """Compute frozen signatures on validation records, run the evaluator-only
    source probe and the learner spherical-k-means, and return F1 inputs."""
    import torch
    import model as M

    sigs, srcs = [], []
    model.eval()
    with torch.no_grad():
        for s, recs in val_by_source.items():
            for i in range(0, len(recs), 64):
                chunk = recs[i:i + 64]
                input_ids, attention_mask, _ = M.collate(chunk)
                sig = model.signature(input_ids, attention_mask, warmup_mean)
                sigs.append(sig.cpu().numpy())
                srcs += [s] * len(chunk)
    signatures = np.vstack(sigs)
    sources = np.array(srcs, dtype=np.int64)

    # learner spherical k-means (K_max=8), source-free
    rng = np.random.default_rng(seed ^ 0xC0DE)
    _centers, cell_labels = S.spherical_kmeans(signatures, S.K_MAX, rng)
    return {"signatures": signatures, "sources": sources, "cell_labels": cell_labels}


# --------------------------------------------------------------------------- #
# Top-level D2 panel
# --------------------------------------------------------------------------- #
def _val_by_source(mat, cap_per_source=EVAL_PER_SOURCE_CAP) -> Dict[int, List[Dict]]:
    """Evaluator-only validation records grouped by source index (report-only)."""
    by_id = {int(r["example_id"]): r for r in mat["records"]}
    bssp = mat["by_split_source_polarity"]["validation"]
    out: Dict[int, List[Dict]] = {}
    for s in range(st.N_SOURCES):
        recs = []
        for p in (0, 1):
            for eid in bssp[s][p]:
                r = by_id.get(int(eid))
                if r is not None:
                    recs.append(r)
        out[s] = recs[:cap_per_source]
    return out


def run_panel(mat, seeds, role_salt, cfg=None, verbose=True) -> Dict:
    """Run the full three-seed F0-F3 feasibility panel and return verdicts +
    raw metrics.  ``mat`` is prepare_marc.materialize output."""
    import model as M
    if cfg is None:
        cfg = M.OptimConfig()

    by_id = {int(r["example_id"]): r for r in mat["records"]}
    source_table = mat["source_table"]
    val_by_source = _val_by_source(mat)

    # train records (learner-facing) and train example_ids by source/polarity
    bssp_train = mat["by_split_source_polarity"]["train"]
    train_ids_sp = [[bssp_train[s][p] for p in (0, 1)] for s in range(st.N_SOURCES)]
    all_train = [by_id[int(eid)] for s in range(st.N_SOURCES)
                 for p in (0, 1) for eid in bssp_train[s][p] if int(eid) in by_id]

    # capacities from the fixed envelope (same as run_development.derive_capacities)
    import buffers as buf
    env = buf.history_envelope_bytes(64, n_records=256)
    cell_cap = buf.capacity_from_envelope(env, 64, 8)
    per_no_cell = buf._per_record_bytes(64) - buf._CELL_LABEL_BYTES
    caps = {"er": (env // 2) // per_no_cell, "cell": cell_cap,
            "ploss": (env // 2) // (per_no_cell + 4), "oracle": cell_cap}

    # ---- F2: ER-FLOP vs oracle, continual, three paired seeds ----
    f2 = {"oracle_worst": [], "erflop_worst": [], "oracle_macro": [], "erflop_macro": []}
    f1_inputs = None
    f3_metrics = None
    for si, seed in enumerate(seeds):
        if verbose:
            print(f"  [seed {seed}] building stream + warmup ...", flush=True)
        plans = st.build_from_ids(train_ids_sp, seed=seed)   # full 60-window stream

        # common warmup checkpoint (shared by both arms this seed)
        warm_recs = []
        for wp in plans[:st.N_WARMUP_WINDOWS]:
            warm_recs += _window_records(wp, by_id)
        warm_train, _ = _split_roles(warm_recs, role_salt)

        base = M.load_model(seed=seed)
        winfo = M.warmup_train(base, warm_train, cfg, epochs=1, seed=seed)
        warmup_mean = winfo["warmup_mean"]
        common = M.snapshot(base)

        # F1 diagnostics on the first seed (frozen signatures at warmup boundary)
        if f1_inputs is None:
            f1_inputs = f1_diagnostics(base, warmup_mean, val_by_source, seed)

        arms = B.make_arms(caps, st.N_SOURCES)
        for arm_name in ("er_flop", "oracle"):
            arm = arms[arm_name]
            mdl = M.load_model(seed=seed)
            M.restore(mdl, common)              # clone common checkpoint into the arm
            run_continual_arm(mdl, cfg, arm, plans, by_id, source_table,
                              role_salt, seed)
            bal = per_source_balacc(mdl, val_by_source)
            worst, macro = worst_and_macro(bal)
            if verbose:
                print(f"    {arm_name:8s} worst={worst:.3f} macro={macro:.3f} "
                      f"by_source={{ {', '.join(f'{k}:{v:.2f}' for k,v in sorted(bal.items()))} }}",
                      flush=True)
            if arm_name == "oracle":
                f2["oracle_worst"].append(worst); f2["oracle_macro"].append(macro)
            else:
                f2["erflop_worst"].append(worst); f2["erflop_macro"].append(macro)

        # ---- F3: offline-joint, computed once (first seed) ----
        if f3_metrics is None:
            if verbose:
                print(f"  [seed {seed}] offline-joint (F3) ...", flush=True)
            f3_metrics = offline_joint(all_train, val_by_source, cfg, seed)

    # ---- assemble verdicts ----
    v_f1 = feas.gate_F1(f1_inputs["signatures"], f1_inputs["sources"],
                        f1_inputs["cell_labels"], rare_source=RARE_SOURCE,
                        seed=seeds[0])
    v_f2 = feas.gate_F2(f2["oracle_worst"], f2["erflop_worst"],
                        f2["oracle_macro"], f2["erflop_macro"])
    v_f3 = feas.gate_F3(
        warmup_balacc=f3_metrics["warmup_macro"],
        offline_macro_balacc=f3_metrics["offline_macro"],
        offline_worst_balacc=f3_metrics["offline_worst"],
        random_balacc=f3_metrics["random_macro"],
        lora_on_off_logit_diff=f3_metrics["lora_logit_diff"],
        lora_disable_perf_drop=f3_metrics["lora_perf_drop"],
        lora_grads_finite_nonzero=f3_metrics["grad_finite_nonzero"],
        max_numerical_rank=f3_metrics["max_rank"],
        frozen_params_unchanged=f3_metrics["frozen_unchanged"],
    )
    return {
        "F1": v_f1, "F2": v_f2, "F3": v_f3,
        "f2_raw": f2, "f3_raw": f3_metrics, "caps": caps, "seeds": list(seeds),
    }
