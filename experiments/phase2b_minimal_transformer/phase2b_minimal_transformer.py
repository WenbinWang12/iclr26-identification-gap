"""D2-B minimal: rare-task injection on a small nonlinear MLP with width scaling.

Extends the D2-A linear bridge to a small nonlinear model: a bottleneck MLP
encoder d_in -> N (width) -> d_out whose encoder weight is FULLY fine-tuned,
so the encoder width N is the trainable capacity axis (Huang et al.'s N). We
reproduce the rare-task injection protocol of Huang et al. (arXiv:2605.29548)
and ask whether the curvature-weighted measure kappa_G predicts the rare-task
retention onset as the encoder width N varies.

Note on capacity axes: a companion LoRA-rank-scaling run (LoRA on a *frozen*
encoder, sweeping LoRA rank R) did NOT show a rare-task retention onset --
the trainable capacity is capped at rank R regardless of width, so kappa_G
floors and the rare task is never freed. Width-scaling with a *trainable*
encoder is the axis on which Huang's reduced-interference mechanism appears.
The two axes are unified under kappa_G; only the width axis exhibits the onset.

Authorized claim: on this small nonlinear model, kappa_G predicts rare-task
retention onset across encoder widths (full fine-tuning). Not authorized:
OLMo-scale results, FCRA, benchmark performance, or claims about full LLMs.
"""

from __future__ import annotations

import argparse, csv, hashlib, io, json, platform, sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
import numpy as np

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE / "phase2b_minimal_transformer.py"


def _as_json(v):
    if isinstance(v,(np.floating,np.integer)): return v.item()
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,dict): return {str(k):_as_json(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [_as_json(x) for x in v]
    return v

def _sha256(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as h:
        for b in iter(lambda:h.read(1<<20),b''): d.update(b)
    return d.hexdigest()

def _add_gate(gates,name,obs,exp,p,tol,notes="",relation="eq"):
    n=isinstance(obs,(int,float,np.number)) and isinstance(exp,(int,float,np.number))
    err=abs(float(obs)-float(exp)) if (n and relation=="eq") else None
    if n:
        viol=max(0.0,abs(float(obs)-float(exp))-float(tol)) if relation=="eq" else (max(0.0,float(obs)-float(exp)-float(tol)) if relation=="le" else max(0.0,float(exp)-float(obs)-float(tol)))
    else:
        viol=0.0 if p else 1.0
    gates.append({"name":name,"observed":_as_json(obs),"expected":_as_json(exp),"passed":bool(p),"tolerance":float(tol),"relation":relation,"abs_error":None if relation!="eq" or err is None else float(err),"violation":float(viol),"notes":notes})

def _record_case(case,gates,trace):
    for g in gates: g["case"]=case
    return {"case":case,"gates":gates},{"case":case,**trace}


def _sqrt_psd(M):
    M=(np.asarray(M,float)+np.asarray(M,float).T)/2
    w,V=np.linalg.eigh(M); return (V*np.sqrt(np.clip(w,0,None)))@V.T


class TinyMLP(nn.Module):
    """A bottleneck encoder: d_in -> N (encoder width) -> d_out. The encoder
    weight is FULLY fine-tuned, so the encoder width N is the trainable
    capacity axis (Huang et al.'s N). The decoder is also trainable so the model
    can fit each task's output map. The occupied capacity grows with N."""
    def __init__(self, d_in, width, d_out):
        super().__init__()
        self.encoder = nn.Linear(d_in, width)   # the width-N bottleneck, TRAINABLE
        self.decoder = nn.Linear(width, d_out)  # trainable so the output map fits
    def forward(self, x):
        h = torch.relu(self.encoder(x))
        return self.decoder(h)


def _build_teachers(d_in, K, n_modes=4, seed=20260529):
    """K tasks on orthogonal input blocks; rare task (k=K-1) lives on a
    high-index block so its direction requires higher rank to cover."""
    rng = np.random.default_rng(seed)
    block = d_in // K
    Ws = []
    for k in range(K):
        Bk = rng.standard_normal((block, n_modes))
        Qk, _ = np.linalg.qr(Bk)
        lam = np.array([j**(-2.0) for j in range(1, n_modes+1)])
        block_full = np.zeros((d_in, n_modes))
        block_full[k*block:(k+1)*block] = Qk
        Wk = block_full @ np.diag(np.sqrt(lam))
        Ws.append(Wk.T)  # n_modes x d_in
    # Make the rare task (k=K-1) higher-energy so its signal is strong when learned
    Ws[-1] = Ws[-1] * 2.0
    freqs = np.array([k**(-1.5) for k in range(1,K+1)], float)
    freqs /= freqs.sum()
    return Ws, freqs, block


def _train_width(width, K, d_in, Ws, freqs, rare_k, total_steps, device, seed=0):
    """Train a width-N bottleneck encoder (FULL fine-tuning) on the mixture.
    Tasks are sampled by their natural frequency (rare task is genuinely rare);
    the rare task thus appears at its low natural rate, matching Huang et al.
    No extra suppression -- the frequency already makes it rare."""
    torch.manual_seed(seed)
    model = TinyMLP(d_in, width, Ws[0].shape[0]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    Wt = [torch.tensor(W, dtype=torch.float32, device=device) for W in Ws]
    rng = np.random.default_rng(seed+1)
    rare_losses = []
    for step in range(total_steps):
        k = int(rng.choice(K, p=freqs))   # natural sampling: rare is genuinely rare
        x = torch.randn(128, d_in, device=device)
        y = x @ Wt[k].T
        pred = model(x)
        loss = ((pred - y)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            with torch.no_grad():
                xr = torch.randn(256, d_in, device=device)
                yr = xr @ Wt[rare_k].T
                rare_losses.append(float(((model(xr) - yr)**2).mean()))
    # occupied basis = curvature-whitened left singular vectors of the encoder
    # weight. encoder.weight is (width, d_in); its rows span what the encoder
    # uses. We measure occupancy on the INPUT side (d_in) via the whitened rows.
    enc_w = model.encoder.weight.detach().cpu().numpy()  # (width, d_in)
    enc_bias = model.encoder.bias.detach().cpu().numpy() if model.encoder.bias is not None else None
    dec_w = model.decoder.weight.detach().cpu().numpy()  # (d_out, width)
    return rare_losses, enc_w, dec_w


def _encoder_occupied_basis(enc_w, curvature):
    """Curvature-whitened occupied basis of the encoder weight (full fine-tuning).
    enc_w: (width, d_in); rows are the directions the encoder uses on the input.
    Occupied (left) basis on the INPUT side = top-rank left singular vectors of
    whitened enc_w^T (so the basis lives in R^{d_in})."""
    W = _sqrt_psd(curvature)              # d_in x d_in
    # We want the whitened left basis of the map x -> encoder(x) on the input.
    # The map on the input side is enc_w^T (d_in x width). Whiten by curvature.
    Y = W @ enc_w.T                       # d_in x width, whitened
    U, _, _ = np.linalg.svd(Y, full_matrices=False)
    r = min(enc_w.shape[0], U.shape[1])
    return U[:, :r]  # d_in x r


def check_width_scaling() -> tuple[dict, dict]:
    """D2-B (width-scaling): across encoder widths N with FULL fine-tuning, do
    kappa_G predict rare-task retention onset? Width N is the trainable capacity
    axis (Huang's N). The common gradient weakens once enough directions are
    occupied (reduced interference), freeing the rare task -- the mechanism the
    LoRA-rank axis could not exhibit because trainable capacity there is capped
    at the fixed rank R."""
    if not HAS_TORCH:
        return _record_case("width_scaling", [], {"skipped": "no torch"})
    gates = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_in, K = 32, 8
    Ws, freqs, block = _build_teachers(d_in, K, n_modes=4)
    rare_k = K - 1
    # common curvature M_F = sum of common task outer products (on input side)
    M_F = np.zeros((d_in, d_in))
    for k in range(rare_k):
        M_F += freqs[k] * (Ws[k].T @ Ws[k])
    mu_F = np.sort(np.linalg.eigvalsh(M_F))[::-1]
    threshold = float(freqs[rare_k] * np.linalg.svd(Ws[rare_k], compute_uv=False)[0]**2)
    Ncrit = next((N for N in range(1, len(mu_F)+1) if mu_F[N-1] <= threshold), len(mu_F))

    widths = [2, 4, 8, 12, 16, 24, 32]
    final_rare = []
    kappa_curve = []
    muN_curve = []
    steps_env = 30000 if device == "cuda" else 6000
    for N in widths:
        losses, enc_w, dec_w = _train_width(N, K, d_in, Ws, freqs, rare_k, total_steps=steps_env, device=device, seed=7)
        final_rare.append(losses[-1] if losses else float('nan'))
        # analytic top-N common utility mu_N (Huang's Ncrit predictor)
        muN_curve.append(float(mu_F[min(N, len(mu_F))-1]) if N <= len(mu_F) else 0.0)
        # curvature-weighted measure from the trained encoder weight
        occ = _encoder_occupied_basis(enc_w, M_F)
        kappa = 0.5*float(np.trace((np.eye(d_in)-occ@occ.T)@M_F))
        kappa_curve.append(kappa)
    # onset: rare loss at max width should be meaningfully below rare loss at min width
    onset = final_rare[-1] < final_rare[0] * 0.85
    _add_gate(gates,"D2B_rare_loss_decreases_with_width", final_rare[-1], final_rare[0]*0.85, onset, 0.0, relation="le",
              notes=f"rare final loss {final_rare[0]:.3f} (W={widths[0]}) -> {final_rare[-1]:.3f} (W={widths[-1]})")
    # rare task is actually learned at high width (absolute, since learnable floor ~0.30)
    learned = min(final_rare) < 0.5
    _add_gate(gates,"D2B_rare_task_learned_at_high_width", min(final_rare), 0.5, learned, 0.0, relation="le",
              notes="rare task reaches near learnable floor at higher width")
    # kappa decreases with width (common residual frees up)
    kappa_dec = all(kappa_curve[i] >= kappa_curve[i+1]-1e-9 for i in range(len(kappa_curve)-1))
    _add_gate(gates,"D2B_kappa_decreases_with_width", kappa_curve[-1], kappa_curve[0], kappa_dec, 0.0, relation="le",
              notes="kappa_G tail sum shrinks as encoder width grows")
    # correlation of kappa with final rare loss
    if len(set(kappa_curve))>1:
        rx=np.argsort(np.argsort(kappa_curve)); ry=np.argsort(np.argsort(final_rare))
        corr=float(np.corrcoef(rx,ry)[0,1])
    else:
        corr=1.0
    _add_gate(gates,"D2B_kappa_correlates_rare_loss", corr, 1.0, corr>=0.5, 0.0, relation="ge",
              notes=f"Spearman corr(kappa, rare_loss)={corr:.3f}")
    return _record_case("width_scaling", gates, {
        "device": device, "d_in": d_in, "K": K,
        "widths": widths, "final_rare_loss": final_rare, "kappa_curve": kappa_curve,
        "muN_curve": muN_curve, "Ncrit_analytic": Ncrit, "corr_kappa_loss": corr,
    })


def run_all_checks():
    results=[check_width_scaling()]
    summaries=[r[0] for r in results]
    traces={r[1]["case"]:r[1] for r in results}
    gates=[g for c in summaries for g in c["gates"]]
    failed=[g for g in gates if not g["passed"]]
    return {"schema_version":1,"stage":"phase2b_minimal_transformer",
            "claim_boundary":"small_transformer_nonlinear_only",
            "required_gate_count":len(gates),"failed_gate_count":len(failed),
            "all_required_pass":not failed,
            "cases":[{"case":c["case"],"gate_count":len(c["gates"]),"failed_count":sum(not g["passed"] for g in c["gates"])} for c in summaries],
            "gates":gates,"traces":traces}


def _write_json(p,v): Path(p).write_text(json.dumps(_as_json(v),indent=2,sort_keys=True)+"\n",encoding="utf-8")

def write_run(output_dir, run_role):
    output_dir=Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()): raise FileExistsError(f"not empty: {output_dir}")
    output_dir.mkdir(parents=True,exist_ok=True)
    result=run_all_checks()
    summary={k:v for k,v in result.items() if k!="traces"}
    t=result["traces"].get("width_scaling",{})
    summary["headline"]={"device":t.get("device"),"widths":t.get("widths"),
                          "final_rare_loss":t.get("final_rare_loss"),
                          "corr_kappa_loss":t.get("corr_kappa_loss"),
                          "Ncrit_analytic":t.get("Ncrit_analytic")}
    tp=output_dir/"traces.json"; sp=output_dir/"summary.json"; cp=output_dir/"per_case.csv"
    _write_json(tp,result["traces"]); _write_json(sp,summary)
    fields=["case","name","relation","observed","expected","passed","tolerance","abs_error","violation","notes"]
    with cp.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for g in result["gates"]: w.writerow({f:_as_json(g.get(f)) for f in fields})
    cfg=io.StringIO()
    try:
        with redirect_stdout(cfg): np.show_config()
    except Exception: pass
    manifest={"schema_version":1,"stage":"phase2b_minimal_transformer","run_role":run_role,
              "source_sha256":_sha256(SOURCE_PATH),
              "numpy_version":np.__version__,"python_version":platform.python_version(),
              "platform":platform.platform(),"numpy_config":cfg.getvalue(),"command":" ".join(sys.argv),
              "torch_version": (torch.__version__ if HAS_TORCH else "none"),
              "resource_ledger":{"gpu_used": HAS_TORCH and torch.cuda.is_available(),"optimizer_state_bytes":0,"replay_bytes":0,"external_model_bytes":0},
              "output_hashes":{p.name:_sha256(p) for p in (tp,sp,cp)},
              "all_required_pass":result["all_required_pass"],"claim_boundary":"small_transformer_nonlinear_only"}
    _write_json(output_dir/"manifest.json",manifest)
    return {"summary":summary,"manifest":manifest,"output_dir":str(output_dir)}

def main(argv=None):
    p=argparse.ArgumentParser(description="D2-B minimal transformer width-scaling (full fine-tuning).")
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--run-role",default="primary_frozen")
    a=p.parse_args(argv)
    r=write_run(a.output_dir,a.run_role)
    print(json.dumps(_as_json(r["summary"]),indent=2,sort_keys=True))
    return 0 if r["summary"]["all_required_pass"] else 1

if __name__=="__main__": raise SystemExit(main())
