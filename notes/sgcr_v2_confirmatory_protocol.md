# SGCR-v2 confirmatory protocol v1

Status: **LOCKED_UNRUN**. Written before executing any seed in this protocol.

Date/time locked: 2026-08-27, Asia/Shanghai. This is a local prospective lock,
not an external preregistration.

Claim scope: controlled linear reduced-rank-regression mechanism evidence only.
It is not a Transformer, nonlinear LoRA, or real-dataset claim.

## Design and seed derivation

The fresh panel has 8 teacher seeds x 8 stream seeds x 3 learner-buffer seeds.
The three learner replicates are averaged inside each teacher x stream world;
the 64 worlds are the inference units.

Seeds are generated mechanically. For role in `{teacher,stream,learner}` and
zero-based two-digit index `i` (`00..07` for teacher/stream and `00..02` for
learner), define:

```text
domain = "SGCR-v2-confirmatory-2026-08-27-v1|{role}|{i:02d}"
u32 = first four bytes of SHA256(UTF-8(domain)), interpreted big-endian
seed = 10,000,000 + (u32 mod 80,000,000)
```

- teachers: `39817664,15328524,84645970,74522946,71897575,51205904,77534061,25975093`;
- streams: `37032991,78983795,10988186,84417538,20724470,26352979,33758660,55439889`;
- learners: `31770922,66872619,33154327`.

All 19 values are unique. Before lock, exact numeric-boundary searches over 440
relevant text files in the workspace, corresponding Claude project sessions,
Claude memory/subagent records, and global Claude history returned zero matches.
None of these seeds was used for smoke, development, or power estimation.

The 8x8 size was chosen from the burned-panel effect/variance before seeing this
panel. The approximate design sensitivity is 0.84 if the burned effect and
variance transfer; this is not a guarantee.

## Frozen benchmark

- regime `capacity_limited`;
- `d=24`, planted `K=8`, deployed rank `R=4`;
- 240 windows, nominal batch 96, holdout 512;
- 2--4 mixed sources per window, frequency exponent 1.5;
- input/output noise `0.1/0.03`, direction correlation `0.5`;
- replay budget inherited from `buf_u+buf_g=16` windows;
- ridge `1e-6`.

The planted `K` is available only to the generator/evaluator. The SGCR learner's
resolution rule is `K_max=2R` and is tested to be invariant to changing
`cfg["K"]` after `(X,Y)` batches have been materialized.

## Frozen SGCR-v2 configuration

```yaml
max_groups: null              # structural K_max = 2R
warmup_windows: 16
refresh_every: 20
minimum_per_cell: 32
rounds: 24
eta: 0.1
mean_margin: 0.01
min_audit_gain: 0.0
blend_grid: [0.10, 0.20, 0.35, 0.50, 0.70, 1.00]
coherence_ratio: 0.60
audit_fraction: 0.50
refit_on_all: false
return_debug: false
return_frequency_baseline: true   # evaluation instrumentation only
```

Arms are SGCR-v2, its exact same-pool frequency-q0 model returned from the same
run, frozen phase-2f PSR, and frozen excess-CVaR. Every arm uses the same
teacher, stream, learner seed, and independent source test sets in a world.

## Metrics and fixed inference

For each world and arm:

- mean is frequency-weighted source retention;
- tail is minimum source retention.

For each paired difference, use the existing two-way additive random-effects
crossed interval after within-world learner averaging:

```text
Var(mean difference) = sigma_teacher^2/8
                     + sigma_stream^2/8
                     + sigma_residual^2/64
df = min(8,8)-1 = 7
two-sided Student-t 95% interval
```

IID intervals, wins, and raw learner-seed SD are descriptive only. The primary
verdict is an intersection-union test and passes only if all three lower bounds
satisfy:

1. `CI_lower(SGCR mean - PSR mean) > -0.01`;
2. `CI_lower(SGCR tail - PSR tail) > 0`;
3. `CI_lower(SGCR tail - excess-CVaR tail) > 0`.

The separately preregistered secondary mechanism gate is:

4. `CI_lower(SGCR tail - same-pool q0 tail) > 0`.

The primary verdict is not changed by gate 4, but a failure of gate 4 forbids the
claim that the robust weighting path adds value beyond the coreset itself.

No world may be removed or replaced. A persistent exception, non-finite model,
rank violation, or budget violation is a failed run. Missing audit support is
kept and invokes the frozen frequency-baseline fallback; its frequency is
reported. There is no optional stopping.

## Resource and claim boundaries

Every run asserts the logical persistent numeric payload is at most 73,728
float-equivalents; the configured maximum is 73,695. This ledger includes stored
`x,y`, priority keys, ids, cell labels, centers, and counters. It excludes Python
container overhead, the common deployed adapter, and transient clustering/RRR
workspace; these must not be described as fixed total/peak bytes.

The audit coreset is reused, so the gate is empirical coreset selection rather
than a population-safety theorem. `K_max` is a resolution cap rather than a
consistent estimator of the number of latent sources. The solver is a
Hedge-inspired weighted-RRR/Pareto path, not exact nonlinear or NMSE Group DRO.

Any edit to a hashed file after any fresh seed is executed burns the complete
panel. Hashes are stored separately in
`experiments/phase2g_targeted_remedy/locked_sha256_v1.txt`.
