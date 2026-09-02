# Claude proposal audit and research pivot

Date: 2026-08-28 (Asia/Shanghai)

## Executive decision

Claude's diagnosis that `frequency != worst group` is correct and is directly
supported by Phase-2H.  Its proposed remedy, however, is not yet a new method:
the current PSR implementation already performs multiplicative-weights minimax
reweighting of buffered losses followed by a weighted reduced-rank solve.  That
method has a positive result only in the controlled linear proxy; the nonlinear
probe is negative and the real Transformer D2 oracle has only about 1--1.5
percentage points of recoverable worst-source headroom.

The project should therefore **not** relabel PSR as a continual-LLM method and
rerun it on a different dataset.  The recommended pivot is:

> **Hard-global-budget continual PEFT that optimizes anchored tail retention by
> exchanging rank atoms according to risk-weighted marginal utility.**

The three terms in that sentence are all necessary:

1. **hard global budget**: one deployed adapter whose total payload/rank is
   independent of the number of tasks;
2. **anchored tail retention**: compare each task with its own learnable-gain
   anchor instead of comparing raw losses or raw accuracies across heterogeneous
   tasks;
3. **marginal-utility rank exchange**: allocate capacity only when an extra rank
   atom measurably improves the at-risk task, rather than merely upweighting the
   task with the largest loss.

No paper rewrite or positive LLM claim is authorized until the oracle-headroom
probe in the Phase-2I protocol passes.

## What is right and wrong in the Claude summary

### Correct

- Phase-2H falsifies the proxy `rare source = worst source`: the 1% Amazon
  product category is often the easiest source, while the true worst categories
  are frequent.
- A benchmark with heterogeneous task difficulty is needed before a tail-risk
  objective can have meaningful headroom.
- OA-Adapter is a mandatory close comparison.
- Worst-task/group metrics are largely missing from continual-PEFT papers, whose
  principal metrics are usually average accuracy/performance and average BWT.

### Material corrections

1. **PSR already uses worst-buffer loss.**  Replacing frequency by measured
   buffered worst loss is the current method, not a new remedy.  Its failure to
   transfer is precisely why a nonlinear rank-allocation mechanism is needed.
2. **OA-Adapter is an Adapter method, not ordinary LoRA.**  It adds a
   task-specific bottleneck module per task/layer and learns a thresholded
   dimension.  Its modules still accumulate with the task horizon; this differs
   from a hard total deployed budget.
3. **The 58.5% result is narrow.**  It is OA-Adapter's Standard-benchmark,
   initial-budget-16 comparison with O-LoRA (4.72M to 1.96M parameters, AA 75.3
   to 76.0), not a universal reduction relative to every SOTA or every
   15-task setting.
4. **CoDyRA does not prove that realized forgetting is monotone in rank.**  It
   upper-bounds a one-step reference-loss increase by update norm and then uses
   `||Delta W||_F <= sqrt(rho)||B||_2||A||_2`.  Rank tightens that upper bound
   only when the factor norms and other quantities are controlled.  The current
   spectral-tail lower bound and this result are an analogy, not an established
   primal--dual pair.
5. **The public O-LoRA/OA protocol is not equivalent to our hidden-source
   stream.**  Training task boundaries are known, and the released O-LoRA
   commands add task and dataset names to prompts.  It is fair to say that no
   numerical task ID is used for adapter routing at test time; it is not fair to
   say that the input is task-identity-free.
6. **Several items in the claimed LLM family are visual methods.**  InfLoRA,
   C-LoRA, and FM-LoRA are useful analogies but are not evidence on continual LLM
   PEFT.
7. **"Measured worst group" is undefined when the learner never sees groups.**
   Classical Group-DRO assumes group-labelled training examples and group-aware
   validation.  Reconstructing pseudo-groups repeats SGCR's identifiability
   failure.  A genuinely group-free arm must optimize an example-level tail and
   reserve true task/source labels for the evaluator and oracle only.
8. **Raw worst loss is not worst retention.**  The 15 tasks have different
   output spaces, target lengths, chance levels, irreducible difficulty, and
   noise.  A raw-loss controller can spend all capacity on a task it never
   learned.  The signal must instead normalize later degradation by each
   example/task's own acquisition gain.

## Novelty threats that must be in the 2026 comparison set

The 2026 literature makes each simple version of the proposal non-novel:

| Candidate claim | Direct threat | Consequence |
|---|---|---|
| single fixed-rank/single-LoRA continual learning | SLAO, CLoRA, EWC-LoRA, ELLA, Share, Meta-UCF | A single non-growing adapter or constant-sized generator is not by itself novel. |
| dynamic rank or layer allocation | OA-Adapter, AdaLoRA, FlexLoRA, LR-LoRA, FIM-LoRA, SpaRTA | Dynamic rank and a fixed sum of layer ranks are not by themselves novel. |
| spectrum/tail controls rank and forgetting | DYRA, EBLoRA, SLoRA, ReCoLoRA, SpaRTA, CoDyRA | A spectral-tail allocation theorem alone is no longer enough. |
| capacity recovery/compression | E2-LoRA, PCLR, Share, ReCoLoRA | Recycling low-energy rank vectors is already an active method family. |
| bounded online adapter merging | K-Merge, SLAO, MergeProbe, SAFE-Merge | Storage-aware merge/prune/reweight decisions and worst-case merge retention are no longer novel. |
| worst-task/fair optimization | Group-DRO, FairOCL, CORE, Adaptive Memory Replay, MT-GRPO, DRATS | Reweighting the worst task is established outside PEFT and partly inside CL. |
| worst-group + LoRA | On Fairness of Low-Rank Adaptation; FairNet | This pair of keywords cannot support a novelty claim. |
| continual PEFT plus an explicit group-fairness term | Meta-UCF | Even this pair is occupied; Meta-UCF uses a support-conditioned generator and a demographic-parity term. |

Primary sources to use in the paper-level survey include:

- O-LoRA: https://aclanthology.org/2023.findings-emnlp.715/
- OA-Adapter: https://arxiv.org/abs/2505.22358
- CoDyRA: https://arxiv.org/abs/2412.01004
- CLoRA: https://aclanthology.org/2025.acl-long.940/
- GORP: https://aclanthology.org/2025.acl-long.721/
- TreeLoRA: https://proceedings.mlr.press/v267/qian25b.html
- SLAO: https://arxiv.org/abs/2512.23017
- Meta-UCF: https://proceedings.iclr.cc/paper_files/paper/2026/hash/12202970782399ee67981dc5269c3b8a-Abstract-Conference.html
- EWC-LoRA: https://proceedings.iclr.cc/paper_files/paper/2026/hash/47197750b407ef48daa1438e0c0af242-Abstract-Conference.html
- PCLR: https://proceedings.iclr.cc/paper_files/paper/2026/hash/5a5acfd0876c940d81619c1dc60e7748-Abstract-Conference.html
- SLoRA: https://aclanthology.org/2026.acl-long.247/
- SpaRTA: https://aclanthology.org/2026.acl-long.334/
- ReCoLoRA: https://arxiv.org/abs/2607.07719
- Share: https://arxiv.org/abs/2602.06043
- DYRA: https://openreview.net/forum?id=559x6AYm1p
- E2-LoRA: https://arxiv.org/abs/2605.27482
- ELLA: https://aclanthology.org/2026.eacl-long.84/
- K-Merge: https://aclanthology.org/2026.acl-long.137/
- MergeProbe: https://arxiv.org/abs/2606.19549
- SAFE-Merge: https://arxiv.org/abs/2608.01184
- FairOCL: https://openreview.net/forum?id=klJuJp6iqu
- MT-GRPO: https://arxiv.org/abs/2602.05547
- Group-DRO: https://arxiv.org/abs/1911.08731
- Fairness of LoRA: https://arxiv.org/abs/2405.17512
- FairNet: https://arxiv.org/abs/2510.19421

## The contribution that may still survive review

The defensible working claim is not "dynamic rank", "single adapter",
"spectral allocation", "rank recycling", or "Group-DRO for LoRA".  It is the
following narrow joint object:

> Under separately declared horizon-independent deployed-payload and persistent-
> history budgets, maintain one unconditional LoRA and perform risk-weighted
> rank-one admission, eviction, and recompression using acquisition-normalized
> historical retention regret, thereby optimizing an individual-level CVaR tail
> while explicitly constraining average utility and current-task plasticity.

Every qualifier is material.  The controller reads no task/source/group label
and uses no pseudo-group.  The deployed adapter is not selected by a router or
generated from a support set.  Prompts may still contain ordinary task
instructions, which must be disclosed.  All persistent anchors, examples,
dual/controller variables, and metadata count against a fixed history-byte cap;
all transient candidate ranks and probe forwards are reported separately.

The intended differences from the closest methods are:

| Method | Capacity state | Allocation signal | Objective |
|---|---|---|---|
| OA-Adapter | task-specific Adapter modules accumulate | current-task loss and learned threshold | AA/BWT/FWT and parameter efficiency |
| CoDyRA | sparse task update is merged into the backbone; cumulative displacement rank is not capped | rank importance/sparsity | average forgetting and base capability |
| DYRA | task bases accumulate subject to residual spectral threshold | residual activation spectral tail | average performance/BWT at reduced average rank |
| SLAO | one rank-`r` LoRA is continually merged | orthogonal initialization and time scaling | average performance/forgetting |
| E2-LoRA | a finite output-rank pool is split among task-specific visual LoRAs | output-drift energy and new-task plasticity | average stability--plasticity |
| Meta-UCF | one constant hypernetwork generates a task-conditioned LoRA from a support set | task embeddings, orthogonality, and a bias term | average performance/forgetting plus demographic parity |
| K-Merge / MergeProbe | bounded adapter slots or merge decisions; routing remains possible | adapter similarity or early mergeability signals | aggregate or worst-case merge retention |
| ELLA | one aggregated update with constant memory/compute regularization | high-energy task-specific alignment | average continual performance |
| FairOCL | ordinary network plus replay | alpha-fair gradient utility | fair gradient improvement across tasks |
| **proposed Phase-2I** | one unconditional adapter plus a separately capped history sketch, both horizon-independent | group-free anchored regret x measured per-atom marginal tail benefit | example-CVaR retained gain with mean/current constraints |

The primary-source search through 2026-08 found no direct predecessor satisfying
all of these properties simultaneously.  This remains a *candidate*
contribution: the four edges of the conjunction are already occupied, so a
simple `E2-LoRA energy score + CVaR weights` composition is not sufficient.  The
safe wording is "we found no prior method that jointly enforces these
constraints," not "the first," until submission-time search.

## Setting decision

The project must stop conflating two settings.

- **Standard task-sequential benchmark:** episode boundaries are known at
  training.  The public prompt may contain task/dataset descriptors.  The
  group-free controller does not consume the episode/task label, and a single
  adapter is used without an external task-indexed router or support set at
  inference.
- **Hidden mixed-source stream:** group risk is not observable from raw losses.
  A method may use a fixed, explicitly counted group-labeled audit sketch, or it
  must assume and verify group observability.  Without one of these, a
  worst-group guarantee is not identifiable.

For the positive method paper, use the first setting for the main benchmark and
describe the second as the diagnostic boundary exposed by SGCR.  A later mixed
stream extension may use a tiny labeled audit sketch, but should not be called
source-free.

## Claims prohibited before Phase-2I passes

- "Measured worst-group loss fixes PSR/SGCR."  It is already the PSR mechanism
  at the window level and has no nonlinear positive result.
- "The method is safe by construction."  A validation/reference gate is an
  empirical selection rule unless backed by an independent holdout and a stated
  finite-sample bound.
- "Forgetting grows monotonically with rank."  CoDyRA does not establish this.
- "Our spectrum theorem is dual to CoDyRA."  No such duality is currently
  proved.
- "No task ID is used" without disclosing task/dataset descriptors in prompts
  and known training boundaries.
- "First fixed-budget rank recycling" or "first worst-case retention-guided
  merge."  E2-LoRA and MergeProbe directly occupy those mechanism-level claims.
- "First fairness-aware continual PEFT."  Meta-UCF already contains an explicit
  demographic-parity term, although its support-conditioned deployment differs.
- "Standard CL has five tasks" if reproducing the released four-task O-LoRA/OA
  orders.  Yelp may be added only as a separately named five-task extension.
- Any real-LoRA/LLM positive claim based on D2-C, D2-B, or Phase-2H.

## Immediate go/no-go sequence

1. Reproduce the exact released Standard order and prompt protocol with a small
   T5 model; record whether task and dataset descriptors are present.
2. Measure per-task base score, matched single-task anchor, immediate post-task
   score, final score, and rank response curves.
3. Run an offline fixed-global-budget oracle using anchored retention, alongside
   risk-only and allocation-only oracles.
4. Continue only if the oracle produces material worst-retention headroom at
   no more than a one-point mean cost, and if the joint risk x rank interaction
   beats both single mechanisms.
5. Only then implement online rank exchange and lock three unseen task orders
   and seeds for confirmation.
