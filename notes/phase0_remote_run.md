# Phase-0 Remote Confirmation Record

Date: 2026-07-23

Status: confirmed for the narrow Phase-0 diagnostic claim. This record is not
evidence for FCRA, rank allocation, task-free PEFT, LLM performance, or an
independent replay/addressability result.

## Frozen identity

- Source SHA256: `20ec7f0568a8b70e35b8eb87577fc16ee1c7328f9c85219db15e39b431d5f100`
- Protocol SHA256: `2cf6fa6c7346450608717fea6e601b1b9118f6c424a2014d192b5dc3e6562d7b`
- Test SHA256: `c10423ae110d73ffa3544e623ea75b5fcf1b02db23aed6dbe47c83bef9c61bd6`
- NumPy: `2.4.6`
- Local host: Windows 11, Python 3.12.10
- Remote host: `alibaba-10`, Linux 6.8.0, Python 3.12.3

The source, protocol, and test hashes match byte-for-byte on both hosts. The
generated-input and output hashes differ across Windows/Linux, as expected for
QR/BLAS floating-point provenance. They are recorded but are not cross-host
byte-equality gates. Each host's output hashes are self-consistent.

## Artifacts

| Run | Host | Role | Rows | Paired mean | 95% CI | Artifact |
|---|---|---|---:|---:|---|---|
| `local_20_v7` | WUrbane | `primary_frozen_20_seed` | 2640 | 0.2772669967 | [0.1619056961, 0.4082528334] | `experiments/phase0_diagnostic/outputs/local_20_v7` |
| `remote_primary_20_v2` | alibaba-10 | `primary_frozen_20_seed` | 2640 | 0.2772669967 | [0.1619056961, 0.4082528334] | `experiments/phase0_diagnostic/outputs/remote_primary_20_v2` |
| `local_100_v5` | WUrbane | `sensitivity_only_100_seed` | 13200 | 0.2113494039 | [0.1704251991, 0.2553952064] | `experiments/phase0_diagnostic/outputs/local_100_v5` |
| `remote_sensitivity_100_v2` | alibaba-10 | `sensitivity_only_100_seed` | 13200 | 0.2113494039 | [0.1704251991, 0.2553952064] | `experiments/phase0_diagnostic/outputs/remote_sensitivity_100_v2` |

The frozen decision is the 20-seed paired bootstrap. The 100-seed runs are
sensitivity-only and do not replace that decision.

## Checks

- Local and remote unit suites: 16/16 tests passed on each host.
- `gates.all_required_pass`: `true` on both remote runs.
- Compatible capacity median/max: `0.0` / `0.0`.
- Compatible acquisition fraction at or below 0.05: `1.0`.
- Correlated-minus-orthogonal paired lower bound: `0.1619056961` for the
  primary run and `0.1704251991` for sensitivity.
- Orthogonal control mean: `0.0`.
- Canceling control: raw total gap positive and acquisition-matched gap zero.
- Inadequate rank-(R+1) capacity control: median `1/3`.
- Compatible rank-specific null: max absolute gap `0.0`.
- Addressability: `pass=null`, status says replay equals the rank-R offline
  oracle and is not an independent test.
- Remote resource ledger: `gpu_used=false`, `optimizer_state_bytes=0`.

The local/remote row keys match exactly: 2640 primary rows and 13200
sensitivity rows, with no missing or extra keys. Across all numeric CSV fields,
the largest absolute difference is `2.8421709430404007e-13` in
`past_retention_ratio`; zero rows exceed `1e-10`. The selected gap fields have
maximum differences of `1.33e-14` (primary) and `2.05e-14` (sensitivity).
Direction-statistics differences are at most `4.44e-16`.

## Interpretation boundary

The confirmed result is only that correlated sequential updates can reintroduce
past residuals on a rank-feasible compatible matrix-sensing stream, while the
orthogonal null and full-observation controls separate order/history
interference from rank capacity. Compatible replay is deliberately not an
independent addressability experiment. Any FCRA, allocation, PEFT, LLM, or
broader continual-learning claim requires a later, separately gated experiment.

## Cross-validation archive

The final read-only four-provider cross-check is archived at
`_artifacts/crosscheck/20260723-140047-8df4c13b`. It completed 4/4 with verified
identities: GPT `gpt-5.5` (xhigh), Claude `claude-opus-4-8` (max), GLM
`glm-5.2` (ultra), and DeepSeek `deepseek-v4-pro` (xhigh). All four returned
`GO`; Claude's first 502 attempt and successful retry are retained in the
manifest. The preceding remote audit and final local snapshot cross-checks are
also retained under `_artifacts/crosscheck/`.
