"""Phase-2H Stage D0 static preflight.

STATUS: development scaffold.  This script does NOT train, download, or install
anything.  It reports the local environment against what the protocol
(notes/phase2h_real_sgcr_protocol.md) requires, and exits with a clear GO/BLOCK
verdict so a human decides whether to authorize a dependency install / download.

Per the protocol's execution boundary: "This document does not authorize a
dependency installation or a long run."  So this preflight is deliberately
read-only.  It never calls pip, never touches the network, never writes model
or data files.

Run:  python preflight.py
"""
from __future__ import annotations

import importlib
import importlib.util
import sys


REQUIRED = [
    # (import name, human name, needed_for, hard_required)
    ("torch", "PyTorch", "LoRA training + tensors", True),
    ("numpy", "NumPy", "stats + buffers", True),
    ("transformers", "HuggingFace Transformers", "frozen BERT-tiny backbone", True),
    ("datasets", "HuggingFace Datasets", "MARC corpus loading", True),
]

# peft is explicitly NOT required: the protocol mandates a manually audited LoRA
# wrapper (see lora_bert.py) so that every trainable parameter is accounted for.
OPTIONAL_ABSENT_OK = ["peft"]

MODEL_ID = "prajjwal1/bert-mini"          # L=4, hidden=256 (2026-08-27 fallback from bert-tiny)
DATASET_ID = "mteb/amazon_reviews_multi"  # English config; pin revision in manifest


def _probe(mod: str):
    try:
        spec = importlib.util.find_spec(mod)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except Exception as e:  # importable spec but broken import
        return f"BROKEN: {type(e).__name__}"


def main() -> int:
    print("=" * 72)
    print("Phase-2H Stage D0 preflight (read-only; no install, no download, no train)")
    print("=" * 72)
    print(f"python           {sys.version.split()[0]}")

    missing_hard = []
    for mod, name, why, hard in REQUIRED:
        ver = _probe(mod)
        if ver is None or (isinstance(ver, str) and ver.startswith("BROKEN")):
            state = "MISSING" if ver is None else ver
            print(f"  [{'X' if hard else '-'}] {name:28s} {state:20s} (needed for {why})")
            if hard:
                missing_hard.append(name)
        else:
            print(f"  [ok] {name:28s} {ver:20s} (needed for {why})")

    # CUDA report (CPU is fine for BERT-tiny D0/D1 per protocol)
    torch_ver = _probe("torch")
    if torch_ver and not str(torch_ver).startswith("BROKEN"):
        import torch  # safe: probed above
        print(f"  cuda available   {torch.cuda.is_available()} "
              f"(BERT-tiny D0/D1 run on CPU; GPU only if D0 microbench is slow)")

    for mod in OPTIONAL_ABSENT_OK:
        ver = _probe(mod)
        print(f"  [info] {mod:26s} {'absent (OK)' if ver is None else ver} "
              f"-- NOT required (manual LoRA wrapper is used instead)")

    print("-" * 72)
    print(f"target model:   {MODEL_ID}")
    print(f"target dataset: {DATASET_ID} (English; pin exact revision in the data manifest)")
    print("-" * 72)

    if missing_hard:
        print("VERDICT: BLOCK -- required packages missing:", ", ".join(missing_hard))
        print()
        print("The protocol does NOT authorize an install here.  A human must decide")
        print("whether to install `transformers` (and confirm a cached/allowed MARC")
        print("revision) before Stage D1.  Until then, only the dependency-free unit")
        print("tests (tests/) and the standalone LoRA-math checks can run.")
        return 2

    print("VERDICT: GO for Stage D1 dependency-present checks (still no long run).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
