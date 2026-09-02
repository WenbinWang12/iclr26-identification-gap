# Phase-2 Planted Linear-LoRA GPU Bridge Protocol

Date frozen candidate written: 2026-07-23 (revision 6 after four-provider audit)

Status: frozen candidate. No primary run may start until a read-only
four-provider cross-check returns a substantive decision on this exact file.
Calibration runs, if any, use disjoint seeds and are permanently non-citable.

## Scope and claim boundary

This phase is a controlled planted linear-LoRA bridge between the exact NumPy
checks and later neural experiments. It uses one frozen linear layer plus one
rank-R factorized update. It is not an LLM, transformer, real-data, FCRA, or
benchmark experiment. A positive result may support only a matched-resource
statement about online error or retention on the frozen planted stream.

The primary compatible stream is designed so an exact rank-R adapter fits the
entire history. A separate capacity-stress stream has a known nonzero rank-R
spectral tail. Results from the two regimes are never pooled into one claim.

## Frozen model and data

Use `d_in=d_out=32`, `T=12`, float32 training, and float64 oracle and evaluation
arithmetic. The model is

`f(x) = W0 x + B A x`,

where `W0` is generated once from a frozen CPU seed and never trained,
`B in R^(d_out x R)`, and `A in R^(R x d_in)`. The deployed update is the
single matrix `Delta=B A`; no router, task adapter, or per-window checkpoint is
available to the learner.

The base draws are frozen independently of every run seed. Let
`S_base=SeedSequence([20260723,0])` and assign `S_base.spawn(5)` to children in
the fixed order `W0`, `compatible_U`, `compatible_V`, `stress_U`, `stress_V`.
Each child uses
`Generator(PCG64(child))` and draws a standard-normal float64 array in C order.
For `W0`, draw shape `(32,32)` and divide by `sqrt(32)`. For the compatible
basis, draw shape `(32,8)` once, compute `Q_comp,_=np.linalg.qr(G_comp,
mode="reduced")` once, and use `Q_comp[:,:R]` for `R in {2,4,8}`. For the stress
basis, draw shape `(32,12)` once, compute `Q_stress,_=np.linalg.qr(G_stress,
mode="reduced")` once, and use all 12 columns for every requested `R`. Apply
the same column canonicalization to each `U/V` basis so that the
entry with the largest absolute value (lowest row index breaks an exact tie) is
positive. These are the only base draws for `W0`, `U`, and `V`.

For each regime, the resulting `K` columns (`K=R` in the compatible regime and
`K=12` in the stress regime) form `U=(u_j)` and `V=(v_j)`.
Required construction gates are
`||U^T U-I||_F <= 1e-10` and `||V^T V-I||_F <= 1e-10` in float64. Mode
`j in {1,...,K}` has
amplitude `alpha_j=1+0.02(K-j)` and emits

`x = xi v_j`, `y = W0 x + alpha_j xi u_j`, `xi in {-1,+1}`. Construct these
arrays in float64; training and replay tensors are the exact float32 casts of
the indexed float64 records, while oracle and evaluation calculations retain
float64 values. The deployed module stores `W0`, `A`, and `B` in float32;
`W0` and the QR-produced `A` are cast once to float32 before training, and `B`
is initialized as float32 zeros. Every forward, loss, backward pass, and
optimizer update uses float32 tensors without autocast. At each checkpoint,
copy `W0`, `B`, and `A` to float64 on the evaluator side, form
`Delta=B@A`, and use that matrix in the exact float64 population risks.
Denote the deployed float32 cast of the base matrix by `W0_32`; this is the
`W0_32` used in the training-loss expression below.
The float32 target `y_32` is the cast of the complete float64 target `y`, so
the training objective contains a fixed base-matrix/cast-roundoff component
that is absent from the ideal float64 adapter oracle. Record that component
separately; it is not interpreted as adapter capacity or as a memory-policy
effect.

Each mode has exactly balanced signs. Precisely, for every even requested count
`n`, the sign stream first constructs `[+1]^(n/2) || [-1]^(n/2)` and applies
`Generator.permutation(n)` to reorder that sign vector; it then concatenates mode blocks in ascending mode
index before the frozen train/evaluation record permutation is applied. The
consumption order is lexicographic in `(regime, R, window, split, mode)`, with
compatible before stress, `R=2,4,8`, and `split=train,eval`. Thus every realized
mode block has exactly `n/2` signs of each value, independent of its permutation.
Within each `(regime,R,window,split)`, immutable record indices are assigned
contiguously in that mode-block order; train and evaluation use disjoint split
namespaces. The resulting permutation is the only storage order. For training,
`pi_t` is the restriction of that stored permutation to all 1024 records in
window `t` (768 current-mode plus 256 previous-mode for `t>1`, and 1024
current-mode at `t=1`), reindexed from zero; no extra random draw or mode
metadata is used.
Every window contains `n_train=1024`
and `n_eval=2048` examples, with train and evaluation examples generated from
disjoint indexed records and shuffled by frozen CPU permutations. For a two-mode
window, the training counts are exactly 768 current-mode and 256 previous-mode
examples (the evaluation counts are 1536 and 512). The evaluation set is
report-only for the training path: it never enters fitting, replay insertion,
replay selection, early stopping, hyperparameter choice, or an outcome claim.
The post-run evaluator may inspect evaluation indices and mode counts only for
the pre-registered split-integrity checks; the numerical oracle and G1-G3
estimands use the exact population risk, not evaluation records.
The first window contains only its current mode. Held-out Gaussian or noisy
probes, if added later, are report-only and cannot affect training or outcome
gates.

The model and optimizer receive only `(x,y)`. Runtime learner components
(model, optimizer, current minibatch iterator after it is constructed, replay
sampler, and early-stopping logic) receive no explicit `t`, mode ID, source ID,
mixture weight, or schedule table. The non-learning data-delivery scheduler
and the post-run evaluator may use `t` for window chronology, frozen
permutations, split-integrity checks, and exact prefix-risk bookkeeping. The
clock-balanced memory allocator is an explicit learner-side treatment: it may
use `t` only to address the pre-registered window-stratified slot ranges and
may not inspect content or derive a mode/source ID from `t` or `(x,y)`. The
reservoir allocator uses insertion boundaries only for eligibility and its
independent priorities for content selection. No clock or schedule signal is
placed in model/replay tensors or passed to gradient computation. This
scheduler-visible boundary is part of the treatment. This phase deliberately
contains no independent-addressability intervention, and no such estimand or
claim is made.

The stream seed `20260723` is common to every paired run and generates the
amplitudes and schedules in addition to the base arrays above. For a run seed
`s`, let `SeedSequence([20260723,s]).spawn(9)` create independent named child
streams, in this fixed order: `train_signs`, `eval_signs`,
`train_permutations`, `eval_permutations`, `clock_insert`,
`reservoir_priority`, `clock_replay_draws`, `reservoir_replay_draws`, and
`factor_init`. Each child uses `Generator(PCG64(child))`. The exact consumption
order is: signs use `(regime,R,window,split,mode)`; train and evaluation
permutations use `(regime,R,window,split)` after all mode blocks are built;
insertion priorities use `(regime,R,window,record_index)` for windows `1..T-1`;
replay draws use `(regime,R,window,update)`; and factor initialization uses
`(regime,R)`. Within every tuple, indices increase lexicographically in the
order stated above, and `Generator.permutation` is the only record-index
permutation operation.
The `factor_init` consumer draws one separate `(R,32)` matrix for every
`(regime,R,s)` in the order compatible `R=2,4,8` then stress `R=2,4,8`; that
matrix is shared by all three trainable arms for its cell. Each draw is a
`standard_normal(dtype=float64)` matrix in row-major C order, computes
`Q,_=np.linalg.qr(G.T, mode="reduced")`, sets `A=Q.T`, and
canonicalizes each row of `A` by the same lowest-index-largest-magnitude sign
rule before setting `B=0`. The child-stream
labels and resulting arrays are recorded in the config and data hashes; no
process-global RNG is used. Canonical hashes serialize contiguous little-endian
arrays (`<f8`, `<f4`, and integer dtypes as declared) and canonical JSON with
sorted keys; record IDs and raw generator specifications are included.

Before importing NumPy or initializing CUDA, the launcher sets
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`NUMEXPR_NUM_THREADS=1`, `MKL_DYNAMIC=FALSE`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. The runtime then sets
`torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic=True`,
`torch.backends.cudnn.benchmark=False`,
`torch.backends.cuda.matmul.allow_tf32=False`,
`torch.backends.cudnn.allow_tf32=False`, and
`torch.set_float32_matmul_precision("highest")`. No DataLoader workers or
PyTorch random draws are used; batches are formed by direct indexed gathers in
the main process.

## Streams

### Compatible primary regime

For each requested rank `R in {2,4,8}`, use exactly `K=R` planted modes. Window
`t` uses current mode `j_t=1+((t-1) mod K)` and, for `t>1`, previous mode
`j_{t-1}` with weights `(3/4,1/4)`. The single planted matrix

`Delta_star = sum_j alpha_j u_j v_j^T`

has rank `R` and fits every window exactly. The primary directional contrast is
pre-registered at `R=4`; `R=2` and `R=8` are secondary rank controls.
Because the public compatible schedule is a deterministic function of the
window index, an analyst can infer a mode from the clock. This is disclosed,
but the mapping is not available at runtime to the model, optimizer, or replay
content sampler. The only arm-specific learner-side use of the clock is the
pre-registered window-stratified slot addressing in the clock-balanced
allocator; data delivery and post-run evaluation may also use the boundary.
Thus the clock-balanced versus reservoir contrast is a scheduler-visible
memory-policy comparison, not delivery of a latent mode label to the model and
not an independent-addressability experiment.

### Capacity-stress regime

Use `K=12`, introduce one new mode per window with the same `(3/4,1/4)` mixture,
and run `R in {2,4,8}`. This regime validates the analytic capacity tail and
tests behavior when rank is insufficient. It cannot support the claim that an
observed gap is independent of capacity.

## Analytic prefix oracle

For prefix `t`, let `q_tj` be the exact fraction of prefix training examples from mode
`j`. The population/prefix risk of update `Delta` is

`L_t(Delta) = 0.5 sum_j q_tj ||Delta v_j-alpha_j u_j||_2^2`.

The unrestricted oracle `Delta_inf,t` fits every observed mode. Because the planted left and
right modes are orthonormal, the exact rank-R oracle keeps the R largest values
of `alpha_j^2 q_tj`; its loss is

`C_R,t = 0.5 sum_(omitted j) alpha_j^2 q_tj`.

Write `Delta_R,t` for the corresponding analytic weighted modal rank-R update,
`Delta_inf,t` for the unrestricted update, and `Delta_seq,t` for the deployed
`sequential_dense_rank_R` update at the prefix-t checkpoint. Training datasets fit the three
trainable arms; evaluation datasets are report-only and never enter fitting,
replay insertion, replay selection, early stopping, hyperparameter choice, or
outcome decisions. G0 may inspect their indices/counts for split integrity,
and all primary estimands below use the exact risk `L_t`, not evaluation
records.

Ties are broken by lower frozen mode index and recorded. Let `J_t` be the first
`min(R, |{j:q_tj>0}|)` indices after sorting positive
`s_tj^2=alpha_j^2 q_tj` descending and then by mode index. Direct float64 risk,
the analytic tail, and a float64 **weighted modal rank-R construction** must
agree within `1e-10`. The construction forms
`Y_t=sum_j alpha_j sqrt(q_tj) u_j v_j^T`, whose nonzero modal singular values
are `s_tj`; it selects `J_t` by the rule above and refits the retained canonical
modes as
`Delta_R,t=sum_(j in J_t) (s_tj/sqrt(q_tj)) u_j v_j^T`
`=sum_(j in J_t) alpha_j u_j v_j^T`.
Thus the singular values are used only to select directions and are divided by
`sqrt(q_tj)` when mapping back; they are never used as adapter amplitudes. When
equal positive singular values produce a degenerate SVD subspace, the
implementation projects that subspace onto the canonical `(u_j,v_j)` pairs and
applies the lower-index rule before the refit; it must not depend on an
arbitrary SVD basis. Modes with `q_tj=0` are never divided by zero and are not
added to the oracle, so its numerical rank may be below R. An unweighted SVD by
`|alpha_j|` is not the oracle. Oracle numerical rank must not exceed R. The
oracle is an analytic reference and is excluded from matched training-resource
averages.

Let `p_tj` denote the mixture weights of the most recent window only: it is
`p_t,j_t=1` at `t=1`, and `p_t,j_t=3/4`, `p_t,j_(t-1)=1/4` for `t>1`, with all
other entries zero. Define the current-window risk
`L_current,t(Delta)=0.5 sum_j p_tj ||Delta v_j-alpha_j u_j||_2^2`.
The positive base denominators are
`D_t=L_t(0)-L_t(Delta_inf,t)` and
`D_current,t=L_current,t(0)-L_current,t(Delta_inf,t)`.
Define the prefix gaps explicitly as
`gap_seq,t=L_t(Delta_seq,t)-L_t(Delta_R,t)`,
`gap_replay,t=L_t(Delta_replay,t)-L_t(Delta_R,t)`, and
`gap_reservoir,t=L_t(Delta_reservoir,t)-L_t(Delta_R,t)`, where
`Delta_replay,t` is the deployed clock-balanced replay update and
`Delta_reservoir,t` is the deployed reservoir update. The paired selection gain
reported below is positive when clock-balanced replay has the smaller prefix gap.
At the final checkpoint, a seed is a valid positive-gap seed iff
`gap_seq,T >= 1e-8` in float64. Seeds with `0 < gap_seq,T < 1e-8` are reported
as numerically indeterminate. Seeds with `gap_seq,T <= 0` are reported as
zero/non-positive-gap; a value below `-1e-10` also fails the oracle/risk
consistency gate. Both categories are excluded from the G3 median and its
qualifying-seed count. No seed is deleted from raw tables. Base-normalized
bootstrap rows retain every seed, while the separately reported `gap_fraction`
bootstrap is restricted to the pre-specified valid positive-gap subset and
records its qualifying count.
Required normalized estimands are

- `capacity_ratio_t=(L_t(Delta_R,t)-L_t(Delta_inf,t))/D_t`;
- `online_ratio_t(method)=(L_t(Delta_method,t)-L_t(Delta_R,t))/D_t`;
- `current_acquisition_ratio_t(method)=L_current,t(Delta_method,t)/D_current,t`;
- `bounded_history_reduction_t=(gap_seq,t-gap_replay,t)/D_t`.
- `selection_gain_t=(gap_reservoir,t-gap_replay,t)/D_t`.

The fraction used in G3 is a distinct quantity,
`gap_fraction_t=(gap_seq,t-gap_replay,t)/gap_seq,t`, and is evaluated only on
the pre-specified valid positive-gap seeds. Thus `bounded_history_reduction`
is base-normalized while `gap_fraction` is sequential-gap-normalized. Both
are operational planted-stream quantities; no quantity in this phase estimates
independent addressability.

Every table also reports the unnormalized losses and gaps. A denominator below
`1e-8` invalidates the corresponding normalized statistic instead of being
silently regularized.

## Compared arms

All trainable arms use the same paired data, factor initialization, optimizer,
learning rate, batch size, number of updates, total examples per update,
gradient clipping, and numerical precision.

1. `offline_rank_R_oracle`: the analytic prefix reference above; no stochastic
   training and no inclusion in resource-matched averages.
2. `sequential_dense_rank_R`: jointly train all R columns on the current window
   only, with no retained historical examples.
3. `clock_balanced_replay_rank_R`: use a fixed 50/50 current/replay composition
   per update. With `H=t-1` eligible past windows at the start of window `t`,
   define `B(H)=0` when `H=0` and `B(H)=B_max=352=32*(T-1)` when `H>=1`;
   both replay arms have logical occupancy `B_t=B(H)`. The clock-balanced
   allocator assigns each past window `r` the deterministic quota
   `c_(H,r)=floor(B_max/H)+1{r <= (B_max mod H)}`; quotas sum to `B(H)` and
   are exactly 32 each when `H=11`. Its physical range begins at
   `o_(H,r)=sum_(a=1)^(r-1)c_(H,a)` and contains slots
   `[o_(H,r), o_(H,r)+c_(H,r))`. Inserting completed window `h` builds the
   buffer consumed at the start of window `h+1`: set `H=h` and
   `B_{h+1}=B(H)`. Every record in window `h` receives one priority from the
   independent `clock_insert` stream. For every window `r<=H`, retain its
   lowest-priority `c_(H,r)` records with the global immutable record key
   breaking priority ties, discard the rest, and place retained records in the
   new physical range in ascending `(priority,key)` order. This is the complete
   deterministic pruning and slot-mapping operation; no other content-dependent
   selection is permitted. The non-learning
   allocator may use `t` only for these pre-registered slot ranges, receives no
   mode table, and never passes window strata to the model or sampler.
   Replay replaces current examples; it never adds updates or examples.
4. `reservoir_random_replay_rank_R`: identical capacity (`B_max=352`), replay
   fraction, raw bytes, updates, and batch composition, but uses deterministic
   global priority-reservoir replacement. Each historical record receives one
   iid priority from the independent `reservoir_priority` stream at insertion
   and keeps that priority for its lifetime. At the same post-insertion
   transition from completed window `h` to window `h+1`, set `H=h` and use
   `B_{h+1}=B(H)`. If the eligible historical count is below `B_{h+1}`, all
   records are retained; otherwise exactly the `B_{h+1}` smallest priorities
   are retained. Sort the retained records by `(priority,global_key)` and write
   them in that order to physical slots `[0,B_{h+1})`; `occupied_slots` is
   consequently `[0,1,...,B_{h+1}-1]` in ascending order. Slots
   `[B_{h+1},352)` are marked unoccupied with priority
   sentinel `1.0` and zero x/y sentinels. The unused reservoir
   `window_counts`, `window_offsets`, and `write_ptr` fields are fixed to zero,
   while `eligible_count` records the number of eligible historical records.
   Replay samples uniformly from this explicitly ordered retained-slot list.
   Thus both arms have the same logical occupancy at every prefix while using
   different window allocation policies. This is only a matched history-policy
   comparator; the protocol deliberately does not test independent addressability.

Insertion priorities are iid float64 `Uniform[0,1)` values generated by
`Generator.random(..., dtype=np.float64)`. Every record has a globally ordered
immutable key `(regime_code,R_code,window,split_code,local_index)` with regime
codes compatible=0/stress=1, rank codes in the order `2,4,8`, and split codes
train=0/eval=1; exact priority ties in either arm are broken by this key. Each arm draws
exactly 1024 insertion priorities after each completed window except the final
window, independently of the raw values.

At window `t`, replay may contain only windows `<t`. Current examples are
inserted after all training and evaluation for `t`; the final window is not
inserted after its last evaluation. Both replay arms allocate the same fixed
arrays: `x_slots[352,32]` and `y_slots[352,32]` in float32,
`priority_slots[352]` in float64, `occupied[352]` in uint8,
`window_counts[12]` and `window_offsets[12]` in uint64, and
`write_ptr[1]` and `eligible_count[1]` in uint64. The clock arm uses the
window-stratified ranges and the reservoir arm uses the same allocated fields
with the fixed mapping and sentinels above. Allocated replay bytes,
including metadata, are identical and all are counted in the ledger. The clock
and reservoir arrays are initialized with x/y equal to zero, priorities equal
to `1.0`, occupancy equal to zero, and every counter/offset/pointer equal to
zero before window 1. After inserting completed window `h in {1,...,11}`, the
clock arm sets `window_counts[r-1]=c_(h,r)` and
`window_offsets[r-1]=o_(h,r)` for `r=1,...,h`, sets all remaining entries of
both arrays to zero, sets `write_ptr[0]=0`, sets
`eligible_count[0]=1024*h`, and marks exactly slots `[0,B_{h+1})` occupied.
Its x/y/priority entries in each occupied window range follow the
`(priority,key)` order above; every unoccupied slot retains the initialized
zero x/y and priority-`1.0` sentinels. After the same insertion, the reservoir
arm keeps every `window_counts`, `window_offsets`, and `write_ptr` entry equal
to zero, sets `eligible_count[0]=1024*h`, and uses its frozen occupied-slot
mapping and sentinels above. The clock
arm's slot ranges are an explicit scheduler-visible window stratification; no
window or mode label is placed in model/replay tensors, and no checkpoint,
gradient, or optimizer state is stored in replay.

The LoRA initialization is frozen and shared across arms: `A` is a deterministic
row-orthonormal matrix from the initialization child stream, and `B=0`. For
each `(regime,R,s,arm)` for each of the three trainable arms (the offline
oracle has no optimizer), instantiate one float32 model and one float32 AdamW
optimizer before window 1; optimizer parameters are ordered `[B,A]`, and both
the parameters, moment state, and optimizer step counter persist through all 12
windows without reset or reinitialization. Use AdamW with `lr=3e-2`,
`weight_decay=0`, `betas=(0.9,0.999)`, `eps=1e-8`, `amsgrad=False`,
`foreach=False`, `fused=False`, `maximize=False`, `capturable=False`, and
`differentiable=False`; use no scheduler, batch size 128, 100 updates per
window, and global gradient-norm clipping at 5.0. For every 128-example batch
`(x_i,y_i)`, including both current-only and replay batches, the exact scalar
loss is
`ell_B=(1/(2*128))*sum_i ||(W0_32 @ x_i) + B @ (A @ x_i)-y_i||_2^2`, with the sum over
all 32 output coordinates and no elementwise-mean reduction. The exact update
order is `zero_grad(set_to_none=True)`, forward, `ell_B`, backward,
`clip_grad_norm_( [B,A], max_norm=5.0, norm_type=2.0,
error_if_nonfinite=True, foreach=False)`, then one `optimizer.step()`; there is
no gradient accumulation. Disable mixed precision and TF32. Do not use early
stopping or historical evaluation for decisions. Use the `pi_t` restriction
defined above. At update `u=0,...,99`, the current-only
arm consumes `pi_t[(128u+i) mod 1024]` for `i=0,...,127`; replay arms consume
the first 64 entries of that same cyclic slice plus 64 replay entries. Replay
entries are sampled uniformly from the ascending flat occupied-slot list using
`Generator.choice(occupied_slots, size=64, replace=(n_occupied<64),
shuffle=True)` and the
arm-specific `clock_replay_draws` or `reservoir_replay_draws` stream. The
sampler receives no `t`, mode strata, or schedule metadata. If `n_occupied=0`
(only at `t=1`), no replay draw is made and the child stream is not advanced;
otherwise exactly one 64-index choice is made per update. A fixed concatenation
order is `current64 || replay64` (current examples first, replay examples
second, for both x and y) and no additional shuffle or update is permitted. If the
historical buffer is empty, replay arms consume the full current slice and
still perform exactly 100 updates.

Before accepting factor results, run a report-only rank-projected full-matrix
sanity optimizer separately at every prefix. Initialize a float64 full matrix
`Delta=0`; for exactly 512 iterations, take a full-population gradient step
`Delta <- Delta - 0.5*grad L_t(Delta)` and replace it by its float64 SVD
truncation to rank R. There is no early stopping and no historical feedback.
The zero initialization is not itself a projected iterate. At iteration 1, the
gradient step is taken before the first truncation, so its exact pre-truncation
matrix is
`Delta_pre,1=0.5 sum_(j:q_tj>0) q_tj alpha_j u_j v_j^T`, which is nonzero on
every positive-weight mode. The first truncation and its resulting support are
included in all checks below.
The implementation records the smallest positive retained weight
`q_min,keep=min_(regime,t,R,j in J_t) q_tj`; the frozen schedules give
`q_min,keep=1/12`, and the squared modal residual/risk bound
`(1-0.5*q_min,keep)^(2*512) < 1.2e-19` is recorded before the run. This bounds
the squared modal error (equivalently its quadratic-risk component); the factor
of two comes from squaring the contraction after 512 gradient steps, not from
an extra optimizer step. Verify its
final prefix risk reaches the analytic rank-R oracle with
`(L_t(Delta_sanity)-L_t(Delta_R,t))/D_t <= 1e-4`; finite training examples are
not used by this sanity arm. After every projection define
`a_j=u_j^T Delta v_j`,
`E=Delta-sum_j a_j u_j v_j^T`, and
`tau=1e-12*max(1,||Delta||_F)`. The numerical modal support is
`S={j:q_tj>0 and |a_j|>tau}`. Require `||E||_F<=tau` and `S=J_t` under the
same canonical tie rule after each of the 512 projections; zero-weight modes
are ignored. A support/off-modal mismatch fails G1 rather than being hidden by
the residual bound. This support equality is a schedule-specific invariant,
not a general property of projected gradient descent: before every SVD
truncation, record the selected top-R modal indices and require that they equal
`J_t` under the frozen schedules and tie rule. If amplitudes or schedules ever
change, this invariant must be re-established and cross-checked.
Failure of this sanity gate invalidates the implementation. Failure of only the
bilinear factor arm is recorded as an optimization failure, not as a failure of
the analytic capacity identity.

## Seeds and stages

Calibration may use only seeds `{11,23,37}` and writes under a directory whose
run role is `non_citable_calibration`. Hyperparameters may be changed only from
those calibration results, after which this protocol must be amended and
cross-checked again before confirmation.

The primary confirmation uses 20 paired seeds
`{53,71,89,107,131,149,167,191,211,233,257,277,307,331,353,379,401,431,457,487}`.
All arms within a seed share the same data and initialization. Bootstrap
calculations use frozen `S_boot=SeedSequence([90920260723,0])`, distinct from
the stream seed, and assign `S_boot.spawn(2)` in order to `all_seed_indices`
and `valid_gap_indices`, each wrapped by `Generator(PCG64(child))`. In one
call, `all_seed_indices` draws
`integers(0,20,size=(10000,20),dtype=np.int64,endpoint=False)`; row `b` is the
paired seed-index resample for bootstrap replicate `b` and is reused for every
all-20-seed mean statistic. Every stated paired 95% lower bound is
`np.quantile(replicates,0.05,method="linear")`, the one-sided empirical 5th
percentile, not a two-sided 2.5th-percentile interval. For the descriptive
`gap_fraction` bootstrap, let `m` be the number of valid positive-gap seeds in
their original primary-seed order. If `m>0`, one call to `valid_gap_indices`
draws `integers(0,m,size=(10000,m),dtype=np.int64,endpoint=False)` and each
row's statistic is the median of those `m` resampled seed-level
`gap_fraction_T` values. If `m=0`, make no draw and report the statistic as
invalid. No best-seed or failed-seed deletion is allowed. G3 additionally
requires at least 16 of the 20 final seeds to be valid positive-gap seeds;
otherwise the gap-fraction gate fails and the invalid-seed categories are
reported.

## Acceptance gates

### G0: identity, leakage, and schema

- source, protocol, test, config, data, and split hashes are recorded;
- train/eval indices are disjoint and mode counts match the frozen schedule;
- orthogonality/sign, finite-value, and deterministic-order gates pass;
- model-facing batches and replay sample tensors contain no latent metadata;
- a runtime trace proves that `t` and schedule/mode metadata do not enter the
  model, optimizer, current minibatch tensors, replay sampler, or early
  stopping; the permitted data-delivery, buffer-allocator, and post-run
  evaluator uses are logged separately;
- requested and numerical product ranks are reported, with rank at most R;
- local CPU smoke tests and remote GPU tests pass before a primary run.

### G1: oracle and capacity

- analytic, direct, and weighted-modal-SVD oracle losses agree within `1e-10`
  in float64;
- the compatible regime has `capacity_ratio_t <= 1e-10` at every prefix;
- the stress-regime measured oracle tail agrees with the frozen analytic tail;
- the recorded `q_min,keep` equals `1/12` and its 512-step convergence bound is
  below `1.2e-19`;
- every sanity projection satisfies `||E||_F<=tau` and retains the canonical
  support `S=J_t` under the frozen tie rule;
- the rank-projected sanity optimizer has normalized oracle error at most
  `1e-4`.

### G2: phenomenon and acquisition

For the compatible `R=4` primary contrast, the paired 95% bootstrap lower bound
of the final sequential `online_ratio` is at least `0.10`. At least 228 of the
240 seed-window observations (95%) for each trainable arm must have
`current_acquisition_ratio <= 0.05`. If these fail, the bridge does not exhibit
the intended online phenomenon under the frozen optimizer.

### G3: bounded-history intervention

For compatible `R=4`, the paired 95% bootstrap lower bound of final
`bounded_history_reduction_T=(gap_seq,T-gap_replay,T)/D_T` for clock-balanced
replay versus sequential must be strictly positive. The median fraction
`gap_fraction_T=(gap_seq,T-gap_replay,T)/gap_seq,T` must be at least `0.50` on
valid positive-gap seeds, with at least 16 qualifying seeds. Report the number
of qualifying, indeterminate, and non-positive seeds beside this median. The
replay arm must continue to satisfy G2 acquisition. The comparison between
clock-balanced and reservoir-random replay reports the one-sided empirical 5th
percentile paired-bootstrap lower bound for
`selection_gain_T=(gap_reservoir,T-gap_replay,T)/D_T`; it supports a
scheduler-visible selection-policy statement only if its lower bound is
strictly positive. Controller CPU time and priority-comparison counts are
reported separately; no total-system efficiency claim is made. This comparison
is not required for the broader bounded-history result.

### G4: matched resources and reproducibility

- trainable arms have identical requested rank, factor shapes, initialization,
  parameter count, optimizer, updates, batch size, and examples processed;
- the two replay arms have identical allocated capacity (`B_max=352`), replay
  fraction, x/y slot bytes, priority-slot bytes, occupancy bytes, and
  pointer/counter bytes; sequential replay bytes are zero and this difference
  is explicit;
- at every prefix both replay arms have the frozen logical occupancy
  `B_t` and the same number of stored unique records (sampling duplicates are
  counted separately);
- both replay allocators process the same 1024 candidate records per inserted
  window and draw the same number of replay samples; occupied-slot counts,
  unique-record counts, replacement/duplicate counts, priority draws,
  comparison counts, and controller CPU time are ledgered separately;
- no arm receives extra forward/backward passes or historical-evaluation
  feedback;
- all gates, failures, per-seed rows, and resource ledgers are retained;
- a second remote confirmation is run in the same locked remote environment
  (host image, Python/NumPy/PyTorch/CUDA/cuDNN versions, and device class) with
  identical protocol, source, test/split, config, and data hashes; it must match
  analytic/oracle scalars
  within maximum absolute delta `2e-10`, trained per-seed scalar metrics within
  maximum absolute delta `1e-6`, and all gate outcomes/counts exactly;
  byte-identical CUDA trajectories are not required. Local CPU smoke outputs
  are provenance-only when their BLAS serialization differs;
- a post-result four-provider read-only audit is required before citation.

## Resource ledger

For every arm and seed record requested/active/deployed/numerical rank; factor
shapes; trainable and deployed parameters; parameter, optimizer-state, replay,
pointer/counter, curvature/sketch, and checkpoint bytes; current and replay
examples; optimizer updates; forward/backward passes; priority draws,
selection comparisons, occupied/unique records, replacement/duplicate counts,
controller CPU time; wall time; peak allocated
and reserved GPU memory; inference latency; device index; GPU, driver, CUDA,
cuDNN, Python, NumPy, and PyTorch identities; deterministic flags; and all
source/data/config hashes.

Logical LoRA parameters equal `R(d_in+d_out)`. Adam parameter and two-moment
storage are counted at their actual dtypes, including master weights if any.
Replay memory in each replay arm includes the allocated 352 fixed float32 x/y
slots (`90,112` bytes), 352 float64 priority slots (`2,816` bytes), 352 uint8
occupancy flags (`352` bytes), `window_counts[12]` and
`window_offsets[12]` (each `96` bytes), and `write_ptr[1]` plus
`eligible_count[1]` (each `8` bytes). The total fixed replay allocation is
`93,488` bytes per arm; logical occupancy and duplicate counts are reported
separately. `W0` and common CPU/GPU staging bytes are recorded separately and
cannot be used to hide an arm-specific resource.

## Decision and failure policy

Primary outcomes are `GO`, `REVISE`, or `NO-GO` against G0-G4. Thresholds,
seeds, schedules, and optimizer settings are not changed after the primary run.
Any failed gate remains in the artifact. Divergence, metadata leakage,
incomplete accounting, rank violation, oracle mismatch, replay chronology
violation, or unequal compute blocks a positive claim.

A positive result supports only: under this frozen planted linear stream and
the recorded rank/compute/history budgets, bounded replay changed online gap or
retention relative to current-only training, and clock-balanced replay changed
the final prefix gap relative to global reservoir allocation at equal allocated
memory and learner updates, only if the dedicated paired gate passes. This is
an operational scheduler-visible memory-policy result, not a claim of
selection efficiency or independent addressability; this phase contains no
independent-addressability intervention. It does not establish
FCRA, task-free LLM continual learning, nonlinear LoRA optimality, real-data
effectiveness, benchmark superiority, scaling, or that all forgetting is
capacity-driven.
