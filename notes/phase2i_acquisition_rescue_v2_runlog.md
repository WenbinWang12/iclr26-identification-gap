# Phase-2I acquisition-rescue v2 run log

Status (2026-08-29, Asia/Shanghai): **development in progress; the strict
three-task acquisition prerequisite has not passed**.

## Why the first rescue cannot trigger oracle headroom

The first isolated T5-small rescue used LR `1e-3`, rank-4-equivalent LoRA,
96 update rows per class, field-aware length 512, and class-balanced
train-derived audits.  Recomputing the artifacts exposed two confounds:

| Task | Balanced gain | Train-prior gain | False recall pre/post | True recall pre/post | Strict interpretation |
|---|---:|---:|---:|---:|---|
| QQP | +17.77 pp | +12.41 pp | 90.23 / 85.55 | 41.80 / 82.03 | credible isolated pilot |
| BoolQA | +10.35 pp | -1.46 pp | 22.66 / 82.03 | 87.89 / 49.22 | class-flip failure |
| MultiRC | +9.77 pp | +11.78 pp | 29.69 / 59.38 | 92.19 / 82.03 | positive row-split pilot only |

The balanced audit is not natural-distribution EM.  BoolQA's apparent gain is
a label-prior reversal, and its label-constrained diagnostic also falls by
8.98 points.  MultiRC additionally has 38/512 confirmation rows whose
paragraph-question group appears in update or tune, so it is not a strict
semantic held-out result.

The implementation-level cause of the BoolQA flip is target-token weighting.
With EOS included, the pinned T5 tokenizer represents `True` with two tokens
and `False` with four.  Hugging Face `output.loss` averages across target
tokens, so a class-balanced update gives `False` approximately twice the loss
mass of `True`.  This is neither balanced per-example risk nor train-natural
per-example risk.

Pilot confirmation SHA-256 values:

- QQP: `6F7C45DA1FDC2F2D004C555D53A67CDBCC110EAA21BA59FA7299A41EDE6896EF`
- BoolQA: `1C5D34659564E30D761904FC4AA23A159B4EF34D438CE7EAD96905E144C6370C`
- MultiRC: `678D0A6AB5AA01A3473F7341D7A07D84860566E17C8A19DE89803F68719E9D74`

## v2 protocol

The corrected rescue uses:

- semantic groups: QQP question-graph connected components, BoolQA passage,
  and MultiRC paragraph plus question;
- a confirmation partition whose groups were absent from every recorded prior
  update/tune/confirm partition;
- complete pinned-train label counts to report train-prior EM alongside
  balanced EM and per-class recall;
- per-example sequence-mean NLL controls, eliminating verbalizer-length loss
  weighting;
- a strict gate: train-prior EM gain `> 5 pp`, valid-label rate at least 0.99,
  and no class-recall drop below -5 pp;
- one locked matched `hf_token_ce` checkpoint evaluated with the selected
  method on the same confirmation partition.

## BoolQA Stage A: objective intervention

Artifact root:
`D:\phase2i_runs\t5small_objective_rescue_v2_boolqa_stageA_seed42_20260829`

Everything except the objective and epoch checkpoint is matched: LR `1e-3`,
cap 96/class, batch 4, field-aware 512, and the same initialization/order.
The tune audit has 32 rows per class.

| Objective | Epoch | Train-prior gain | Balanced gain | False delta | True delta | v2 gate |
|---|---:|---:|---:|---:|---:|---|
| token CE | 1 | +9.71 | +18.75 | +56.25 | -18.75 | fail |
| token CE | 2 | +9.28 | +17.19 | +50.00 | -15.62 | fail |
| sequence mean, balanced | 1 | +6.68 | +7.81 | +12.50 | +3.13 | pass |
| sequence mean, balanced | 2 | **+9.05** | **+10.94** | **+18.75** | **+3.13** | **pass** |
| sequence mean, train-prior | 1 | +3.45 | +1.56 | -6.25 | +9.38 | fail |
| sequence mean, train-prior | 2 | +4.63 | +3.12 | -3.13 | +6.25 | fail |

The locked selection is `sequence_mean_balanced`, epoch 2.  Its selection
SHA-256 is
`3E4AF975C9C757F829C406D56B45B3A32D0F5D318CE8203A15D705B39A921A06`.
Confirmation metrics were not used for this selection.

## First v2 BoolQA confirmation: failed

Artifact root:
`D:\phase2i_runs\t5small_objective_rescue_v2_boolqa_confirm_seed42_20260829`

| Arm | Train-prior gain | Balanced gain | False delta | True delta | v2 gate |
|---|---:|---:|---:|---:|---|
| selected sequence mean | +0.15 | -1.17 | -6.64 | +4.30 | fail |
| matched token CE | +1.37 | +6.64 | +28.52 | -15.23 | fail |

The invocation did not access tune audit, all outputs were valid labels, both
payloads satisfied the exact 144-atom budget, row and semantic-group overlap
within the new split was zero, and confirmation overlap with recorded prior
groups was zero.  Confirmation SHA-256:
`720EFBBACE6A481C63E2F3A0E6AE6851E6BFBF3915954053E92F62A376291D43`.

This is an honest negative confirmation, not a positive method result.
Subsequent audit also found that the semantic exact-quota allocator considered
larger groups first.  The resulting BoolQA confirmation contains only
multi-row passage groups and therefore is not conditionally representative of
the pinned train distribution.  The allocator must be changed to hash-random
group priority with exact-quota feasibility before another confirmation.

## Splitter and headroom-runner correction

The semantic allocator is now `hash-priority-exact-dp.v2`: groups are placed
in seeded hash order independent of group size, and a binary-bitset subset-sum
solver preserves exact per-label quotas.  Historical semantic artifacts retain
the legacy `size-first-greedy.v1` reconstruction path so their hashes remain
auditable.  New manifests report split and corpus group-size histograms and
effective group counts.  On real pinned data, dry splits for QQP, BoolQA, and
MultiRC are exact and have zero row or semantic-group overlap.  The complete
Phase-2I test directory passes (`84 passed`).

The acquisition-conditioned headroom runner was separately reviewed after
the same audit-representativeness fix.  Its frozen SHA-256 is
`41B22356FCA41E77CDC5F73F52696F6A0A094EB5C0B376FD977BD8BB052549A1`.
The review found no remaining code blocker in the incoming-checkpoint
retention chain, sequence-mean objective, strict acquisition gate, test-set
sealing, or controller compute accounting.  This is implementation clearance,
not empirical clearance: the official headroom run remains prohibited until
the three train-only acquisition confirmations pass.

## BoolQA Stage C: diversity and LR/epoch rescue (failed)

Artifact root:
`D:\phase2i_runs\t5small_objective_rescue_v2_boolqa_diversity_stageC_seed43_20260829`

This selection run uses seed 43, 512 update rows per class, 128 tune rows per
class, one or two epochs, LR in `{3e-4, 1e-3, 3e-3}`, and compares the corrected
sequence-mean balanced loss against matched token CE.  Its future 32-row/class
confirmation is restricted to groups absent from all R0, R1, and Stage-A
manifests.  For BoolQA those remaining eligible groups are singleton passages,
so the next confirmation will not repeat the first confirmation's all-multirow
conditional bias.  Confirmation remains sealed during this search.

The first completed token-CE trajectory already fails the strict gate.  At LR
`3e-4`, epoch 1 gives natural-prior gain `+1.65 pp`, balanced gain `+6.64 pp`,
and per-class changes `(False +27.34, True -14.06) pp`.  Epoch 2 gives
natural-prior gain `-3.38 pp`, balanced gain `+8.20 pp`, and per-class changes
`(False +56.25, True -39.84) pp`.  Thus increasing diversity and optimizer
steps does not by itself repair target-token-weighted CE; its balanced score is
still a class-exchange artifact.

For later tasks, the rescue runner now supports a cheaper leakage-safe control
protocol.  Sequence-mean candidates alone enter the LR/epoch selection grid;
after a durable selection lock is written, exactly one token-CE control with
the selected LR/epoch/cap/sampler is trained from the same initialization.
That control is selection-ineligible, reads neither tune nor confirmation, and
is first evaluated beside the selected model during independent confirmation.
The revised runner passes 92 tests and has SHA-256
`F0A0B1A6858F511773FA5EB25E9B20EBA57AA9C0AAF32D14881EA253EF1DEE44`.

The complete 12-checkpoint grid finished after approximately 3.6 hours.  All
checkpoints produced valid labels, but none passed the joint gate:

| Objective | LR | Epoch | Natural gain | Balanced gain | False delta | True delta | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| token CE | 3e-4 | 1 | +1.65 | +6.64 | +27.34 | -14.06 | fail |
| token CE | 3e-4 | 2 | -3.38 | +8.20 | +56.25 | -39.84 | fail |
| sequence mean | 3e-4 | 1 | -0.19 | +0.00 | +0.78 | -0.78 | fail |
| sequence mean | 3e-4 | 2 | +2.16 | +7.81 | +31.25 | -15.63 | fail |
| token CE | 1e-3 | 1 | -2.80 | -4.69 | -12.50 | +3.13 | fail |
| token CE | 1e-3 | 2 | +1.25 | +8.59 | +39.06 | -21.88 | fail |
| sequence mean | 1e-3 | 1 | -2.02 | -3.91 | -11.72 | +3.91 | fail |
| sequence mean | 1e-3 | 2 | **+3.37** | +7.42 | +24.22 | -9.38 | fail |
| token CE | 3e-3 | 1 | +1.54 | +3.52 | +11.72 | -4.69 | fail |
| token CE | 3e-3 | 2 | -21.26 | -7.42 | +50.00 | -64.84 | fail |
| sequence mean | 3e-3 | 1 | -4.64 | -8.59 | -25.00 | +7.81 | fail |
| sequence mean | 3e-3 | 2 | -9.64 | -6.25 | +7.81 | -20.31 | fail |

The least-bad selected checkpoint is sequence mean at LR `1e-3`, epoch 2,
but `tune_rescue_pass=false`; no confirmation prediction was made.  Selection
SHA-256 is
`94AA4CCE49B121499D8882D10AEAEFC4F023F1DA9294AE56D685B54153CDA6D3`.

The failure is more specific than “the task is not learned.”  On the same
tune rows, constrained-label margin AUC rises from `0.7035` at the base model
to roughly `0.74--0.76` for several trained checkpoints.  Free generation
nevertheless exchanges large amounts of True and False recall because the
global label boundary moves.  A tune-only diagnostic threshold can expose
some latent headroom, but its gain over an equally calibrated base is only
about 3.07 points; calibration alone is therefore not a valid method claim.

## BoolQA BAP pilot: method failed, predeclared pointwise control positive

Artifact root:
`D:\phase2i_runs\t5small_bap_lora_boolqa_selection_seed43_20260829`

The formal four-source freshness protocol uses the three earlier split
manifests plus the content-addressed Stage-C access-only ledger (SHA-256
`C0A190EB440BF8309C1CF9B64342BAFC999ED636D92C68211A99DBF21BE72D92`).
The resulting update, development, and still-sealed confirmation IDs are
exactly the Stage-C sets.  Every BAP-accessed semantic group was already in the
four-source touched union, so this run introduces no new confirmation
contamination.  The remaining fresh pool is 33 False and 260 True rows and the
32-per-class confirmation has zero prior-touched semantic-group overlap.

BAP chose the first internally passing checkpoint at step 384, but failed its
post-selection development gate:

| Arm / estimand | Internal natural gain | Development natural gain | Development False / True delta | Interpretation |
|---|---:|---:|---:|---|
| BAP, equally calibrated base denominator | +6.254 pp | +2.400 pp | +10.156 / -2.344 pp | development fail |
| BAP, raw official free generation | +2.212 pp | +3.558 pp | +9.375 / 0 pp | acquisition fail |
| matched sequence-mean-prior pointwise, equally calibrated | +7.732 pp | +3.826 pp | +6.250 / +2.344 pp | calibrated diagnostic below 5 pp |
| matched sequence-mean-prior pointwise, raw official free generation | +7.464 pp | **+5.930 pp** | **+15.625 / 0 pp** | original acquisition-v2 gate pass |

The pointwise post valid-label rate is 1.0.  Its development natural-prior EM
rises from 66.690% to 72.620%, and its passage-cluster paired-bootstrap 95%
interval is approximately [+1.00, +10.91] pp.  This is a positive point
estimate under the original acquisition estimand, not a sealed confirmation.
The same-calibration result must be reported separately and cannot be replaced
by calibration gain.

Independent reconstruction found no sign, weighting, ordering, initialization,
or artifact-integrity bug.  Both arms use 384 optimizer steps and the identical
1,536 ordered row exposures (768 per class), but BAP scores both official label
sequences and therefore uses 3,072 sequence forwards versus pointwise's 1,536.
The result rejects the pairwise-only intervention: relative ranking plus a
common-shift anchor discards useful absolute, natural-prior-aligned supervision.
BAP remains a failed ablation and is not confirmation-eligible.

Integrity SHA-256 values:

- immutable BAP selection: `EBFC1D6826E545FD38C3C7E5224BA0E2B9258E7117A20AD47EF7ED4824CF583D`
- BAP checkpoint: `34CE90CA5614B197EE36CF043C1304F1131E0B4A97C239DF3CF1E2AF4D10AC4A`
- pointwise control JSON: `86E67E3AABE0E0832E84CA1A836BA795CB4A7FCE9F0B18431681064382F7F1AE`
- pointwise checkpoint: `D59259644796899EB4C54DC159D05E44C551217B499C696E097E8F746575B383`
- ordered exposure ledger: `D355624238461DA749539F9AC5175A7A86C5675A3B9035D92EC1D01BEE094DCC`

The locked `selection.json` correctly records the pointwise arm as not run at
selection-lock time, while the final manifest separately binds the subsequently
completed control.  Future schemas should name that field
`status_at_selection_lock` to avoid a misleading standalone read; the immutable
artifact must not be rewritten.

## BoolQA Stage D: formal prior-corrected sequence-mean rescue

Artifact root:
`D:\phase2i_runs\t5small_seqprior_rescue_v2_boolqa_stageD_seed43_20260829`

The successful control motivated a new selection-eligible run using the
existing `sequence_mean_prior` objective: per-example sequence-mean NLL removes
verbalizer-length weighting, while the pinned natural-prior-to-balanced-sampler
ratio aligns the update with the primary natural-prior metric.  The small
development grid was LR in `{3e-4, 1e-3}`, two epochs, 512 rows/class, batch 4,
field-aware truncation, and the original raw free-generation v2 gate.  This was
a fresh formal arm, not a post-hoc relabeling of the BAP control.

Both selection-eligible candidates failed:

| LR | Natural gain | Balanced gain | False delta | True delta | Gate |
|---:|---:|---:|---:|---:|---|
| 3e-4 | +2.725 pp | +5.078 pp | +14.844 pp | -4.688 pp | fail |
| 1e-3 | -0.080 pp | +0.391 pp | +2.344 pp | -1.563 pp | fail |

The 3e-4 passage-cluster bootstrap natural-gain interval is approximately
[-2.509, +7.952] pp.  The locked ranking correctly selects it as the less-bad
candidate, but `tune_rescue_pass=false`.  The matched token-CE control was
trained only after the selection lock and was not evaluated on tune or
confirmation, so it has no reportable accuracy result.  Confirmation remains
unaccessed and no oracle run is authorized.

Final Stage-D SHA-256 values:

- selection: `533F033692C1B07627E74C99969AD97C0E8A0084980DFCC355E01AA15BE43DE3`
- result: `3EE031A5B34DAD63D22333A096FE21018FE9D445BB6063AEABC7962DC1495D84`
- task manifest: `415ED1764BDCC345CD54F13B8E27B3A50CC3FD9F69CBDEC22AD7FFD5FF3A36F`
- root manifest: `7F4F4A5500A93DE5B13046E131037652A6A7A119F46D857393F87974CF426AFF`

## BoolQA PCSM exact-mechanism reproduction

Artifact root:
`D:\phase2i_runs\t5small_pcsm_lora_boolqa_selection_seed43_20260829`

Stage D shows that prior weighting alone does not reproduce the positive
pointwise control.  A dedicated runner therefore fixes all observed effective
mechanisms: the BAP internal fit split (384/class), deterministic 2-True/2-False
batches, dropout-disabled gradients, LR 3e-4, and exactly 384 optimizer steps.
The only eligible checkpoint is step 384.  Internal and development each
require the original raw official gate plus positive same-calibration gain,
class safety, and positive AUC delta.  Development is inaccessible until the
internal lock passes; confirmation is inaccessible until both locks pass.

The runner SHA-256 is
`EEF0666C5627254CB6A5D835ED0AB75BFBDD1202BC6BFA1B79D7CBCC22EB2E9B`;
the complete Phase-2I suite passes 112 tests.

The exact-mechanism reproduction passed both qualification partitions:

| Partition | Raw natural gain | Raw balanced gain | Raw False / True delta | Calibrated gain | AUC delta | Gate |
|---|---:|---:|---:|---:|---:|---|
| internal | +7.464 pp | +8.594 pp | +13.281 / +3.906 pp | +7.732 pp | +0.1462 | pass |
| development | **+5.930 pp** | +7.812 pp | **+15.625 / 0 pp** | +3.826 pp | +0.0863 | pass |

Both post models have raw valid-label rate 1.0.  The new checkpoint has a
different `torch.save` file hash from the earlier pointwise control, but all
108 adapter tensors (147,600 elements), static metadata, initialization, and
ordered exposures are bitwise identical.  The common canonical tensor digest
is `4DF314432906478F8C5DE503FD39E397C1F07E2D8B93E6A0D0ED1D74377AA369`.
Thus this is an exact formal reproduction, not an approximately similar arm.

After independent confirmation preflight passed, the one-shot 32-row/class
sealed confirmation was consumed exactly once:

| Estimand | Base | PCSM | Gain / class delta | Result |
|---|---:|---:|---:|---|
| raw official natural-prior EM | 62.595% | 68.845% | **+6.250 pp** | primary pass |
| raw balanced EM | 54.688% | 60.938% | +6.250 pp | positive |
| raw False recall | 21.875% | 28.125% | +6.250 pp | positive |
| raw True recall | 87.500% | 93.750% | +6.250 pp | positive |
| same-calibration natural-prior EM | -- | -- | +4.198 pp | report only |
| same-calibration False / True recall | -- | -- | -9.375 / +12.500 pp | secondary diagnostic fail |
| constrained-margin AUC | 0.6377 | 0.8262 | +0.1885 | positive |

The raw post valid-label rate is 1.0 (base 0.9844).  The locked primary claim
therefore passes exactly as specified: raw official natural-prior gain strictly
above five points, post validity at least 0.99, and no raw class harm.  The
secondary calibrated diagnostic fails its class-safety guard and must be
reported; it was explicitly report-only on confirmation and did not select a
checkpoint, threshold, or coefficient.  This confirmation is evidence for the
official generation endpoint, not evidence that every alternative calibrated
decision rule is class-safe.

The confirmation is small: 32 rows/class, all singleton passage groups.  A
paired bootstrap gives a raw natural-gain 95% interval of approximately
[-1.619, +14.872] pp (empirical P[gain > 0] about 0.926).  Thus the locked
point-estimate gate passes, but this run alone is not a conventional
statistically decisive positive result.  The calibrated natural-gain interval
is about [-5.177, +14.327] pp, whereas the constrained-margin AUC improvement
is more stable, with a bootstrap interval about [+0.075, +0.312].

Final PCSM SHA-256 values:

- selection: `0B8937065F200AC301D097D03D902BF3AEDC91CFB5A0E950CADBA7B8341D6588`
- selected checkpoint: `38FCAE09F005081AC80E48F75645B7A1EE07D75786E28A6541EEF6718F7369CC`
- internal lock: `8376592802C84B20800D103BBD23BD88B0DD050FA0279CC77356FD35C80CC512`
- selection root manifest: `2DA33A593E2BFE6F8E930E0A718DB189D5556D567286AAD471C1A92AE9C16CCE`
- confirmation: `F4D9A9A095558C5C4DD569DBFB7AE1F33E8D15BAE857F2ACA61D035594A27FC1`
- confirmation manifest: `C32D9758A501310A796CF2A6C8D08A054139A8C2023B6BD2DF2B1A58AD8D529F`

## Frozen PCSM cross-task transfer: QQP

Artifact root:
`D:\phase2i_runs\t5small_pcsm_lora_transfer_qqp_selection_seed43_20260829`

The BoolQA recipe was transferred without changing the objective, learning
rate, fit budget, sampler, checkpoint, or gate.  It stopped at the first
train-derived internal audit, before development or confirmation access:

| Quantity | Base | PCSM | Delta | Gate contribution |
|---|---:|---:|---:|---|
| raw natural-prior EM | 71.600% | 87.382% | **+15.782 pp** | pass |
| raw balanced EM | 65.625% | 89.062% | +23.438 pp | diagnostic |
| raw False recall | 90.625% | 82.031% | **-8.594 pp** | fail (`< -5 pp`) |
| raw True recall | 40.625% | 96.094% | +55.469 pp | pass |
| post valid-label rate | -- | 100% | -- | pass |
| same-calibration natural-prior EM | -- | -- | +10.454 pp | pass |
| same-calibration minimum class delta | -- | -- | +10.156 pp | pass |
| margin ROC AUC | 0.8434 | 0.9404 | +0.0970 | pass |

Thus `primary_pass=false`, `secondary_pass=true`, and the overall internal
qualification fails solely because the raw False-class recall drop exceeds
the predeclared safety allowance.  The result is informative rather than an
acquisition collapse: the representation/ranking improves strongly, while the
official raw decoding boundary shifts too far toward True.  Promoting the
calibrated endpoint after seeing this result would be post-hoc endpoint
switching, so the current QQP development and confirmation partitions remain
sealed (`development_tune_accessed=false`, `confirm_eligible=false`).

The transfer runner SHA-256 is
`30B92B26BF9DD0D4E6F2CBC86E76834C74707E4C7ECF2EA8E498524FDD2B7167`;
the complete Phase-2I suite passes 120 tests.  Artifact SHA-256 values:

- qualification failure: `F55E87EE54393871EAF5CD3F85B63036BADE8A6F5EEFA4846B575A34A187AE8B`
- protocol lock: `922EF994AC6F0760CB63BC936742EC3D835F7FC9BFCAEB191B0038F781EFF935`
- root manifest: `4DB2655CC01364D7C1C549AE99D49C8B491AF5037BE8A270C3B58FD911D9F85F`

## Frozen PCSM cross-task transfer: MultiRC infrastructure interruption

Artifact root:
`D:\phase2i_runs\t5small_pcsm_lora_transfer_multirc_selection_seed43_20260829`

The frozen-v1 process exited with code 1 after a PyTorch CPU allocation error
(`DefaultCPUAllocator: not enough memory`, requested allocation 3,710,976
bytes).  No qualification, checkpoint, resource-ledger, or selection artifact
was durably written.  This is **not** a scientific gate failure and supplies no
MultiRC accuracy result.  The manifest still has `status=running` because the
v1 runner has no infrastructure-exception finalizer; it explicitly records
`development_tune_accessed=false` and
`confirm_audit_model_scored_or_tokenized=false`.

The exact point of failure within fit/internal computation is not recoverable
from the durable artifacts.  Future access accounting therefore conservatively
treats both fit and internal partitions as touched, while development and
confirmation remain sealed.  The failed root is retained unchanged.  Its
artifact SHA-256 values are:

- split: `6E4D93F31DB6FB19925F98630E51ED7B26701A4068B93EBF350DF9D289F3CE2C`
- internal split: `4E0F3B7D8E7D0F28C438E6E821D94EA928F96F1679B03168E3A479E0B8A729C1`
- protocol lock: `F3F5029E2E4E79A59C6A4AD72679D248DAD49CA97B21EDE643CCD40A9CE7FA37`
- interrupted manifest: `683E0A34B6E13D2DECEE8256C15A5B5E03237A6EB90DD79CB05267D1B199C071`

The pre-result adaptive-endpoint v2 protocol is separately frozen at
`notes/phase2i_pcsm_adaptive_endpoint_v2_protocol.md`, SHA-256
`C0BB4F10D17FE10AFB30FDB499549492A2191D8E391ED3544E8610A34B51B040`.
Its training batch remains four; evaluation microbatching may be reduced as a
locked resource control without changing the ordered training exposures,
model, checkpoint, data, or endpoint rule.

## Decision boundary

Do not run or interpret official oracle headroom until QQP, BoolQA, and
MultiRC each pass the strict semantic-group/train-prior acquisition gate from
their incoming continual checkpoint.  BoolQA has now passed the locked primary
gate on its one-shot sealed confirmation.  The frozen PCSM transfer has now
failed QQP at the internal class-safety gate without accessing QQP development
or confirmation.  MultiRC v1 was interrupted by host-memory exhaustion before
any durable qualification result, with development and confirmation still
sealed.  Therefore the three-task prerequisite is not complete and oracle
headroom is still unauthorized.  Any later protocol must be declared as a new
method, preserve the still-sealed partitions, and exclude every group actually
touched in the artifacts above.
