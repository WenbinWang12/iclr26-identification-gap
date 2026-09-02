# Phase-2I: Anchored-CVaR global rank-bank protocol

Status: **design / no result**  
Date locked for initial implementation: 2026-08-28 (Asia/Shanghai)

This protocol replaces the failed hidden-source/pseudo-group controller.  It is
the only proposed route currently authorized for a new neural experiment.  It
does not authorize a positive real-LoRA or LLM claim until its locked gates
pass.

## 1. Research question

Can a single continual LoRA with an exact, horizon-independent deployed payload
improve the lower tail of *retained learned behavior* by moving rank across
layers, without task/source labels and without materially reducing mean or
current-task performance?

The method must isolate two mechanisms:

1. an anchored example-tail objective that identifies forgetting rather than
   intrinsic difficulty; and
2. constrained global rank-atom admission/eviction based on measured marginal
   benefit to that tail, rather than spectral energy alone.

If the first mechanism works and the second does not add value, dynamic rank is
deleted from the contribution.  If a task-aware oracle does not have headroom,
the benchmark--budget pair is killed.

## 2. Setting and disclosure

- A frozen backbone `W0` receives a sequence of training episodes/windows.
- The learner deploys exactly one LoRA update after every episode.  It never
  selects an adapter or output head with a task ID.
- The main O-LoRA-style benchmark has known episode boundaries and public
  instruction prompts.  Those prompts can include task/dataset descriptors;
  this is disclosed and is not called fully task-identity-free.
- The proposed risk controller does not read task, dataset, source, or generator
  labels.  Such labels are used by an oracle arm and by the evaluator only.
- A fixed-byte historical envelope is divided into a risk-train buffer and a
  permanently disjoint audit buffer.  Anchors, example IDs, controller/dual
  variables, and all other persistent metadata count toward that envelope.
  Test data never enter either buffer.
- Deployed payload, history bytes, transient scratch parameters, optimizer
  state, update steps, tokens, FLOPs, wall time, and peak device memory are
  reported separately.

## 3. Why raw worst loss is the wrong signal

Heterogeneous language tasks have different class counts, label entropy,
irreducible error, and target lengths.  Raw cross-entropy or raw accuracy
therefore ranks difficulty, not forgetting.  The Phase-2H rare Amazon category
was often the easiest source, which is the same proxy failure in another form.

For a stored example `i`, record its per-target-token loss immediately before
and after its arrival episode:

\[
  \ell_i^{\mathrm{pre}},\qquad \ell_i^{\mathrm{post}}.
\]

At a later checkpoint define anchored retention regret

\[
  r_i(\theta)
  =\operatorname{clip}\!\left(
  \frac{[\ell_i(\theta)-\ell_i^{\mathrm{post}}]_+}
  {\max(\ell_i^{\mathrm{pre}}-\ell_i^{\mathrm{post}},\epsilon)},
  0,r_{\max}\right).
\]

This asks how much of the loss reduction that was actually acquired for the
example has subsequently been lost.  It does not prioritize an example merely
because it has always been hard.

Implementation rules:

- use per-target-token NLL so target length does not define the scale;
- use fixed checkpoint timing and an EMA of losses;
- examples whose acquisition denominator stays below the locked threshold are
  ineligible for the normalized-risk objective but remain in raw acquisition
  reports;
- require persistent high regret across two controller evaluations before an
  example receives maximum tail weight;
- clip regrets and cap every example's adversarial weight;
- never use the audit-buffer losses to update parameters or rank scores.

## 4. Group-free tail objective and its exact scope

For `N` risk-buffer examples and tail mass `alpha`, empirical upper-tail CVaR is

\[
  \widehat{\operatorname{CVaR}}_\alpha(r)
  =\max_{q\in\mathcal Q_\alpha}\sum_iq_ir_i,
  \quad
  \mathcal Q_\alpha=
  \left\{q_i\ge0:\sum_iq_i=1,\ q_i\le\frac1{\alpha N}\right\}.
\]

This has a useful, limited latent-group interpretation.  For any subset `G` of
the empirical buffer with mass at least `alpha`, its uniform-on-`G` weights are
feasible in `Q_alpha`; hence

\[
  \frac1{|G|}\sum_{i\in G}r_i
  \le \widehat{\operatorname{CVaR}}_\alpha(r).
\]

Thus no pseudo-group recovery is needed to upper-bound the average anchored
regret of every sufficiently large hidden buffer subgroup.  This is an
**empirical-buffer statement**, not a population guarantee; groups smaller than
`alpha`, unrepresented groups, and distribution shift outside the buffer are
not covered.

Strict tail minimization can collapse onto a few examples, so the actual method
is a constrained problem.  At each episode, a resource-matched reference branch
defines training-side tolerances `b_mean` and `b_current`:

\[
\begin{aligned}
 \min_{\theta,\,\{r_\ell\}}\quad
   &\widehat{\operatorname{CVaR}}_\alpha(r(\theta))\\
 \text{s.t.}\quad
   &\operatorname{mean}(r(\theta))\le b_{\mathrm{mean}},\\
   &\mathcal L_{\mathrm{current}}(\theta)\le b_{\mathrm{current}},\\
   &\sum_\ell c_\ell r_\ell\le C.
\end{aligned}
\]

The corresponding training Lagrangian is

\[
 \mathcal J(\theta,q,\eta_m,\eta_c)
 =\sum_iq_i r_i(\theta)
  +\eta_m\big(\operatorname{mean}(r(\theta))-b_{\mathrm{mean}}\big)
  +\eta_c\big(\mathcal L_{\mathrm{current}}(\theta)-b_{\mathrm{current}}\big),
\]

with `q` updated inside the capped CVaR set and non-negative dual variables
updated by projected ascent.  Uniform replay remains in every arm as an
optimization/data-coverage primitive, but the scientific object is the explicit
tail/mean/current constrained problem.  This is important both for stability and
for distinguishing the method from merely inserting DRO weights into an
energy-based rank allocator.

## 5. One cumulative global LoRA budget

For adapted layer `l`, use SVD-style LoRA atoms

\[
  \Delta W^\ell
  =\sum_{j=1}^{r_\ell}s_{\ell j}
  b_{\ell j}a_{\ell j}^{\top}.
\]

The deployed payload constraint is exact:

\[
  \sum_\ell c_\ell r_\ell\le C,
  \qquad c_\ell=d_{\mathrm{in}}^\ell+d_{\mathrm{out}}^\ell.
\]

`W0` remains frozen and historical dense deltas are not merged into it.  The
export step physically compacts inactive atoms into a standard per-layer LoRA
rank pattern; zero gates in an overcomplete training tensor do not count as a
payload reduction.

This differs from:

- a fixed rank for every task followed by accumulating task modules;
- a per-step low-rank delta merged into the backbone, whose cumulative
  displacement from `W0` can have growing rank; and
- a fixed rank in every layer, which is SLAO's closest capacity control.

For the clean initial theorem and probe, adapt equal-cost T5/LLaMA query/value
modules.  General unequal-cost modules use the exact parameter constraint, not
the sum of nominal ranks.

## 6. Tail-aware rank-atom exchange

For an existing atom, its first-order constrained-tail importance is

\[
  S^{\mathrm{keep}}_{\ell j}
  =\frac{\operatorname{EMA}\left(
  \left|s_{\ell j}\,\partial\mathcal J/
  \partial s_{\ell j}\right|\right)}{c_\ell}.
\]

For released capacity, form a tail-weighted residual gradient after projecting
out retained directions:

\[
  G_\ell^\perp
  =P^\perp_{\mathrm{retained}}
    \nabla_{W^\ell}\mathcal J,
  \qquad
  S^{\mathrm{grow}}_{\ell j}
  =\frac{\sigma_j^2(G_\ell^\perp)}{c_\ell}.
\]

At a rank-update checkpoint, use these scores only to shortlist rank-one
deletions and admissions.  Actually mask each shortlisted deletion and run a
short zero-impact admission probe on the same risk/current calibration batch;
the exchange value is the measured change in `J` per deployed byte.  Solve the
resulting equal-cost top-utility selection or unequal-cost knapsack while
satisfying the same global `C`, then recompress and physically export the one
adapter.  This shortlist--probe step is the proposed non-energy-only mechanism;
its forward/backward cost must be charged to the method.

New directions are zero-impact initialized and receive a short warmup.  A
transient candidate pool may exceed `C` in stored training parameters, but every
deployed checkpoint uses at most `C`; transient bytes and probe FLOPs are
reported.  A strict variant in which even the forward-active mask never exceeds
`C` is the primary implementation.

For a fixed risk distribution and an additive block-quadratic surrogate, global
top-utility selection is the exact best response under equal atom cost.  The
paper-level theory should prove the corresponding constrained saddle
formulation, give an approximation/regret statement for rank-one exchange, and
state the cross-atom error; it must not claim global optimality for bilinear
nonlinear LoRA training.

## 7. Reference branch and empirical audit gate

Every episode forks from the identical incoming checkpoint:

- **reference**: mean-oriented fixed/global-rank baseline plus uniform replay;
- **challenger**: constrained anchored-CVaR plus tail-aware global rank exchange.

Deploy the challenger only when the independent audit buffer simultaneously
shows:

\[
\begin{aligned}
  \operatorname{CVaR}_\alpha(r_{\mathrm{chal}})
    &<\operatorname{CVaR}_\alpha(r_{\mathrm{ref}}),\\
  \operatorname{mean}(r_{\mathrm{chal}})
    &\le\operatorname{mean}(r_{\mathrm{ref}})+0.01,\\
  L_{\mathrm{current}}^{\mathrm{chal}}
    &\le L_{\mathrm{current}}^{\mathrm{ref}}+0.01.
\end{aligned}
\]

The implementation should use paired confidence bounds rather than only point
estimates and should correct for every candidate inspected by the gate.  Until
that implementation exists, this is called an **empirical audit gate**, never a
population or by-construction safety guarantee.

The ungated challenger is a mandatory ablation.  If almost all apparent benefit
comes from rejecting it, the claimed rank mechanism is not supported.

## 8. Benchmark audit before training

The released O-LoRA-style long order contains 15 datasets:

- CL: Yelp, Amazon, DBpedia, Yahoo, AG News;
- GLUE: MNLI, QQP, RTE, SST-2;
- SuperGLUE: WiC, CB, COPA, MultiRC, BoolQ;
- IMDB.

The released Standard orders use DBpedia, Amazon, Yahoo, and AG News; they
should not silently be called five-task orders.  Reproduce one released order
exactly before adding Yelp as a separately named extension.

All tasks use the model vocabulary head and verbalized outputs.  Do not replace
this with a BERT task-specific-head protocol, which would require task/head
routing.  Log the exact prompts and add a later remove-dataset-name ablation.

Evaluator-only normalized retained gain for task `g` is

\[
  R_g=\frac{a_{g,T}-a_{g,0}}
  {\max(a_{g,g}-a_{g,0},0.05)}.
\]

Report `min_g R_g`, worst-two mean, CVaR over task retention, all raw per-task
scores, average performance, and BWT.  The example-level controller metric and
the evaluator's task-level metric are intentionally distinct.

### Diagnostic-to-method closure tests

The paper is intended to be a diagnosis-plus-method paper.  The diagnosis must
therefore be established on the same real-model runs and must predict the method
ablations, rather than relying on the earlier synthetic proxy:

1. **Difficulty--forgetting mismatch.** Compare raw current loss, persistent raw
   loss, and anchored regret as predictors of future normalized forgetting.
   Report top-tail precision/recall, Spearman correlation, and calibration by
   task.  The claim survives only if anchored regret is significantly more
   predictive across held-out orders.
2. **Energy--utility mismatch.** For a frozen checkpoint, remove each active
   rank atom in turn and measure the actual change in constrained tail objective.
   Compare spectral energy, gradient sensitivity, and the shortlist--probe score
   by rank correlation and top-`k` intervention regret.
3. **Budget-induced tail frontier.** Sweep the exact global payload and show the
   mean--tail Pareto frontier.  The task-ID oracle must demonstrate that a
   mean-matched tail improvement is reachable at the selected budget.
4. **Causal repair.** Raw-loss CVaR versus anchored CVaR tests diagnosis 1;
   energy-only recycling versus constrained marginal exchange tests diagnosis 2.
   If either proposed component does not outperform its matched diagnostic
   ablation, that component and its corresponding paper claim are removed.

Use paired bootstrap confidence intervals over tasks within each order and a
hierarchical bootstrap over orders/seeds for the final comparison.  All
diagnostic thresholds and candidate scores are frozen before the confirmation
orders are opened.

## 9. Locked feasibility sequence

### I0: protocol and implementation integrity

- exact prompt/order manifest and train/validation/test hashes;
- no test read before final reporting;
- no task/source metadata read by group-free arms;
- one vocabulary head and no task-indexed adapter router;
- exact exported payload and numerical-rank audit;
- branch restore bit-identical before reference/challenger training;
- disjoint risk-train and audit buffers;
- full byte/FLOP ledger.

Any failure blocks all method interpretation.

### I1: cheapest pipeline probe (not a paper result)

- T5-small;
- one released long order, initially 200 examples/class;
- source length 128, target length 16;
- query/value LoRA, initial rank 8, deployed average-rank-4-equivalent global
  budget;
- one development learner seed for wiring, then three for the headroom panel.

This validates objective, rank compaction, chronology, and resource accounting.
It is not called an LLM result.

### I2: five-arm oracle-headroom panel

All arms start from matched checkpoints and obey the same final deployed `C`:

1. SLAO-style single LoRA plus uniform experience replay;
2. mean-oriented adaptive global rank plus uniform replay;
3. task-ID oracle anchored-DRO plus risk-aware rank;
4. group-free anchored-CVaR plus uniform rank;
5. group-free anchored-CVaR plus global risk-aware rank.

The oracle may use training episode IDs and training-derived buffers only; it
may not inspect validation/test outcomes to select a model.  It tests whether
the fixed budget contains recoverable tail headroom, not whether an online
group-free learner has found it.

### I3: minimum credible model

Run only after I2 passes:

- Llama-3.2-3B for the lower-cost credible probe, or Llama-2-7B-chat for the
  strongest direct SLAO comparison;
- query/value LoRA and exact rank-8-equivalent global payload;
- fixed 4-MiB historical envelope, initially targeted at 512 risk-train and 256
  audit records but derived from actual serialized bytes;
- one released 15-task order for development, followed by three unseen orders
  and at least three learner seeds for locked confirmation;
- a 40/48-GB GPU is the minimum plausible device; the current local CPU
  environment is not adequate for this stage.

## 10. Pre-specified GO/KILL criteria

Proceed past the headroom panel only if all hold:

1. each primary task has at least five percentage points of acquisition gain,
   so the retention anchor is not dominated by noise;
2. task-ID oracle versus the strongest mean baseline improves worst normalized
   retention by at least five points while losing at most one raw average point;
3. group-free anchored-CVaR recovers at least 70% of the oracle's tail gain and
   improves tail by at least three points itself;
4. risk-aware rank exchange improves by at least two tail points over the same
   anchored-CVaR objective with uniform rank;
5. no seed has a tail catastrophe below the reference by more than three points;
6. the gain survives both update-step-matched and token-FLOP-matched controls.

Decision map:

- oracle fails -> kill this benchmark--budget pair;
- oracle passes, group-free arm fails -> do not return to pseudo-groups; either
  change the paper to a boundary/task-aware setting or kill the method;
- anchored-CVaR passes, rank exchange fails -> delete dynamic rank from the
  contribution;
- full method beats SeqLoRA but not SLAO+ER or the mean-adaptive reference -> no
  publishable method result;
- aggregate gain with recurrent catastrophic seeds -> keep as diagnosis, not a
  positive method.

## 11. Required baselines and ablations after GO

Resource-matched essentials:

- SeqLoRA;
- fixed-rank uniform replay;
- persistent-loss/JTT and ordinary example-CVaR replay;
- mean-oriented AdaLoRA/global-rank replay;
- SLAO and SLAO+replay;
- EWC-LoRA;
- ELLA and Meta-UCF where their released protocol can be reproduced;
- O-LoRA and OA-Adapter with their growing payload reported honestly;
- CoDyRA;
- DYRA;
- offline multitask fixed-global-rank upper bound.

Mechanism-level controls, even where the original paper is visual or has a
different deployment setting:

- an E2-LoRA-style output-energy rank recycler ported to the same global bank;
- a MergeProbe-style retention/mergeability scorer without CVaR constraints;
- K-Merge when task-to-adapter routing is allowed, reported as a different
  deployment setting rather than an apples-to-apples single-adapter baseline;
- FOREVER replay scheduling under the identical history and token budget.

Required ablations:

- raw loss versus anchored regret;
- instantaneous versus persistent tail membership;
- CVaR only versus rank only versus their combination;
- uniform layer rank versus global rank exchange;
- marginal-utility score versus spectral energy alone;
- first-order shortlist only versus measured shortlist--probe exchange;
- penalty-mixture training versus the explicit mean/current constrained dual;
- controller buffer versus audit gate;
- ungated challenger;
- with and without dataset-name prompt field;
- `alpha`, global payload, and history-byte sweeps.

## 12. Paper-level hypothesis, not yet a claim

Working positioning:

> Group-free constrained tail-retention allocation of one unconditional,
> cumulative global LoRA payload.

The intended contribution is the joint combination of anchored forgetting risk,
latent-group-free CVaR coverage, mean/current constraints, measured rank-one
admission/eviction/recompression, and an exact single-adapter payload.  Each
ingredient has close prior art.  E2-LoRA already recycles low-energy ranks,
MergeProbe already predicts worst-case merge retention, Meta-UCF already combines
continual LLM PEFT with a fairness term, and SLAO/EWC-LoRA/ELLA already provide
constant-size continual adapters.  Only strong interaction ablations and
real-model results can establish that the proposed joint object is more than an
obvious composition.
