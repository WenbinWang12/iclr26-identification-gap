# Phase-2 Protocol Cross-Check

The Phase-2 GPU bridge protocol is frozen for implementation at SHA256
`99893D34AB9B47057906B499843C4236A146242FCDAECBA5A019B12CCE06B55E`.
The audited file is `notes/phase2_gpu_bridge_protocol.md`; its bytes were
hash-asserted immediately before the verifier launch.

## Final four-provider audit

- Run: `20260723-182338-df595d88`
- Prompt SHA256: `F2383BCBFD908F0E9C12C6CFD684D53A330C06A719A60209E6DBBDC74EECA240`
- Archived raw artifacts: `_artifacts/crosscheck/20260723-182338-df595d88`
- Scope: read-only audit; no primary or calibration training was run by the verifiers.

| Provider | Requested model | Actual model | Reasoning | Invocation | Verdict |
|---|---|---|---|---|---|
| GPT | `gpt-5.5` | `gpt-5.5` | `xhigh` | success | GO |
| Claude | `claude-opus-4-8` | `claude-opus-4-8` | `max` | fallback success after one 504 | GO |
| GLM | `glm-5.2` | `glm-5.2` | `ultra` | success | GO |
| DeepSeek | `deepseek-v4-pro` | `deepseek-v4-pro` | `xhigh` | success | GO |

All four provider workers succeeded and all four substantive verdicts were
`GO`. The Claude first adaptive attempt returned a 504, then the declared
Claude provider completed successfully; both attempts remain in its status
artifact. The launcher parent was externally time-limited after workers had
finished, so the archived run has the raw provider files and this explicit
reconciliation record rather than a launcher-generated manifest.

The prior run `20260723-181352-9e11a4d1` is retained as a non-final audit: it
covered the previous protocol hash and included one incorrect GPT objection
about the sanity optimizer's zero initialization. Revision 7 states the
gradient-before-first-truncation semantics explicitly. The final audit
recomputed that first pre-truncation matrix and found no support mismatch over
all frozen regimes, ranks, prefixes, and 512 projections.

## Claim boundary

This audit authorizes implementation and locked calibration/confirmation runs
for the frozen planted linear-LoRA bridge only. It does not authorize claims
about LLMs, FCRA, real data, benchmark superiority, nonlinear LoRA optimality,
capacity-independent forgetting, or independent addressability.
