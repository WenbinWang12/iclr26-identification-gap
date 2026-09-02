# Phase-1 Local/Remote Confirmation

Date: 2026-07-23

## Status

Phase-1 is a citable deterministic theory/falsification artifact, conditional
on the assumptions in `notes/phase1_theory_protocol.md`. It is not a neural,
LoRA, FCRA, language-model, or benchmark result. The earlier `local_primary_v1`
through `local_primary_v3` directories remain non-citable pre-audit artifacts.

## Frozen identity

The two primary runs used the same byte-level inputs:

| Input | SHA256 |
|---|---|
| `experiments/phase1_theory_checks/phase1_theory_checks.py` | `28a0c2a4ab40e0d73e7ca6e5a2b238d17598b31b456c5f564078bda65a019ac5` |
| `notes/phase1_theory_protocol.md` | `357f29474939fc24d02ec755b9702c7b1c59d098b7f0a8a43212b72e18ac1865` |
| `experiments/phase1_theory_checks/test_phase1.py` | `8f2b70e6f5e13680c7fc0ba57ffe3077f90654dcd7085258242b566f36e897b5` |

Both manifests record `run_role=primary_frozen_audited` and
`stage=phase1_theory_falsification` with these three hashes.

## Runs

| Run | Host | Environment | Command | Result |
|---|---|---|---|---|
| `local_primary_v4` | Windows 11 (`WUrbane`) | Python 3.12.10, NumPy 2.4.6 | `phase1_theory_checks.py --output-dir outputs\\local_primary_v4 --run-role primary_frozen_audited` | 7/7 tests; 163/163 gates |
| `remote_primary_v4` | `alibaba-10` (`wenbin@8.130.9.139`) | Linux 6.8.0-90-generic, glibc 2.39; Python 3.12.3, NumPy 2.4.6 | `phase1_theory_checks.py --output-dir outputs/remote_primary_v4 --run-role primary_frozen_audited` | 7/7 tests; 163/163 gates |

The local artifact is at
`experiments/phase1_theory_checks/outputs/local_primary_v4`; the pulled-back
remote artifact is at
`experiments/phase1_theory_checks/outputs/remote_primary_v4`.

The gate split is 28 Taylor, 27 sequential, 49 decomposition/spectral/capacity,
39 safe-subspace/useful-price, and 20 block-selection gates. Relation counts
are 96 `eq`, 42 `le`, and 25 `ge`; all 163 violations are zero and all JSON
values are finite. Equality gates report absolute error; one-sided gates report
only one-sided violation.

## Cross-host agreement

Recursive comparison of the primary traces gives a maximum numeric difference
of `1.3322676295501878e-15` (the leading singular value in the spectral trace),
with no value exceeding the frozen `2e-11` comparison tolerance. Output hashes
are self-consistent on each host; they are not expected to be byte-identical
because the two BLAS/LAPACK environments can serialize floating-point values
differently.

Headline values agree on both hosts:

- exact decomposition: total `3`, conflict `1`, capacity `0.5`, online `1.5`;
- independent-innovation expectation: `0.125`;
- rank-2 innovation-prefix tails: `0`, `0`, `0.5`, `0.53125`;
- ambient safe nullities: `(6, 5, 4, 3, 3, 3, 2)`;
- coefficient safe nullities: `(4, 3, 2, 1, 1, 1, 0)`;
- cross-block `rho=0.9`: selected utility `20/19`, alternative utility `1.81`,
  selected cost `19/40`, alternative cost `50/181`, `delta=0.9`, factor `19`,
  and alternative-support cost ratio `1.7195`.

## Resource accounting

Both manifests report `gpu_used=false`, `optimizer_state_bytes=0`,
`replay_bytes=0`, and `external_model_bytes=0`. This phase therefore makes no
GPU, optimizer-memory, replay, or external-checkpoint claim.

## Post-result cross-validation

The final post-result read-only four-provider audit is archived at
`_artifacts/crosscheck/20260723-150730-3c556e0d` (prompt SHA256
`ac474f8bad577d2d34c8aa8e1d1fb695cfa8d378af1ed8a4d9a7f677e805bd9c`). The
launcher completed 4/4 providers with verified identities and substantive
`GO` verdicts:

| Provider | Requested / actual | Reasoning |
|---|---|---|
| GPT | `gpt-5.5` / `gpt-5.5` | `xhigh` |
| Claude | `claude-opus-4-8` / `claude-opus-4.8` | `max` |
| GLM | `glm-5.2` / `glm-5.2` | `ultra` |
| DeepSeek | `deepseek-v4-pro` / `deepseek-v4-pro` | `xhigh` |

The audit confirmed the arithmetic, gate-direction schema, cross-host tolerance,
non-circular spectral competitors, and strict claim boundary. It reviewed the
frozen evidence packet read-only; it did not turn this artifact into an
empirical result.

## Claim boundary and next gate

The strongest supported statement is that the encoded conditional algebraic
identities, inequalities, and constructed counterexamples pass the required
tests and reproduce across two hosts under the stated assumptions. This does
not establish an online continual-learning effect, independent addressability,
FCRA, LoRA/LLM effectiveness, benchmark superiority, or scaling.

The next research gate is a separately frozen controlled bridge with explicit
online and bounded-history comparators. No GPU pilot is claimed until the
remote environment has a locked PyTorch/checkpoint/data setup and that bridge
contract has been independently reviewed.

## Reproduction

From `experiments/phase1_theory_checks`:

```powershell
python -m unittest discover -s . -p 'test_phase1.py' -v
python phase1_theory_checks.py --output-dir outputs\local_primary_v4 --run-role primary_frozen_audited
```

