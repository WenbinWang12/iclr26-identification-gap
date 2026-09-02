# Phase-2S — Separation-gated Invariant Offset (SIO): frozen protocol

Frozen 2026-08-31, **before any Phase-2S run exists** and before the method is
implemented in the harness, after Phase-2Q closed as an artefact (its Q4 collapsed)
and Phase-2R closed both sensitivities as negative.

## 0. Why this method, and why it is not a fourth regulariser

Phase-2Q failed in a specific, diagnosable way, and that diagnosis — not a new
idea — produced this method. BPO drives the *predicted* label histogram to uniform.
That gains +4.40 pp on balanced query mixes and **−0.67 pp at 70:30**, so it was
scored an artefact of our balanced `audit` splits.

Surveying outward from that failure found the same defect in a published method.
**Batch Calibration** (Zhou et al., ICLR 2024, arXiv:2309.17249) predicts
`argmax_y [ p(y|x_i,C) − p̂(y|C) ]` with `p̂(y_j|C) = (1/M) Σ_i p(y_j|x^(i),C)`,
i.e. it subtracts the raw per-class **batch mean**. Its Appendix B lists no
class-imbalance caveat and the paper contains no skewed-batch ablation. Our Q4 is
precisely that missing experiment, and it is negative: the correction inverts sign
under skew.

Three attempts to fix it by estimating the batch marginal better — EM
(Saerens et al. 2002), BCTS-calibrated EM (arXiv:1901.06852), and BBSE
(arXiv:1802.03916) — all made accuracy **worse** while making `π̂` more accurate.
That inversion is the mechanism: under **balanced** accuracy, every class carries
equal weight, so the optimal offset does not depend on the batch class marginal at
all; it should cancel only the model's own bias. Every marginal-estimating rule
therefore injects an irrelevant quantity, and injects more of it as skew grows.
Measured on synthetic logits: substituting the **true** `−log π` degrades balanced
accuracy monotonically (0.8097 → 0.5375 at 90:10) while the oracle offset stays
near-constant (−0.53 … −0.75) across every mix.

SIO is the rule that uses only marginal-invariant information. Training is
`--cl-method none`, byte-identical to baseline, exactly as in 2P and 2Q; nothing is
added to the loss. 2N §5 outcome 4 remains respected. This is an inference rule
occupying the *same* admissibility row as BPO in `sec:enumeration` (query batch +
current model) — it does not add a third row.

## 1. The method

At inference a batch of `n` queries arrives; its scope `S` is readable from the
option list. Restrict to `K_S = 2` (see §5 outcome 5 for why `K_S > 2` is out of
scope here). Let `Z ∈ R^{n×2}` be the verbalizer logits under the current model and

    s_i = Z[i,1] − Z[i,0]          (the gauge-invariant 1-D statistic)

Fit a **tied-variance** two-component 1-D Gaussian mixture to `{s_i}` by EM,
yielding locations `μ_0 < μ_1`, a shared `σ`, and weights `w`. Apply

    b_SIO = (0, −τ),   τ = (μ_0 + μ_1)/2

**Why this is marginal-invariant, which is the whole point.** The component
*locations* are class-conditional quantities; the *weights* are the batch marginal.
SIO reads the locations and **discards `w`** — `w` enters the EM fit only as
nuisance and never enters `τ`. Under a tied variance the midpoint of the two
locations is exactly the equal-posterior decision point at `w = 1/2`, i.e. the
Bayes threshold for balanced accuracy. BC instead subtracts a mean over the batch,
which is a `w`-weighted mixture of both class conditionals, so its correction moves
with `w`. That single difference is the contribution.

Stored state: **0 floats**, as with BPO. No table, no per-task vector, no router,
no examples from past tasks.

## 2. The gate, and why every term is here

SIO must abstain — return the zero offset, i.e. fall back to `R_raw` — whenever its
model does not hold. Each term below was added because a specific measured failure
demanded it, and each is recorded with the failure that produced it:

| term | rule | the failure that forced it |
| --- | --- | --- |
| `n ≥ 48` | too few points to locate two components | n=16/32 gave oracle-sized headroom but SIO could not find it |
| `snr = (μ_1−μ_0)/σ ≥ 1.5` | components not separated | single-class batches (snr 0.23) where BC loses −10.94 pp |
| `min(w) ≥ 0.05` | one component essentially absent | same |
| bootstrap CI on `τ` excludes 0 | no bias worth correcting | bias=0 at 90:10 passed a hand-picked `\|τ\|/σ ≥ 0.15` and lost −2.78 pp |
| `ρ = max(σ̂_free)/min(σ̂_free) ≤ ρ_crit` | tied model misspecified | unequal class-conditional variance passed at snr=4.76 (a **falsely high** snr) and lost −6.43 pp |

Two design points inside that table are themselves consequences of measured
failures and are stated here so they are not mistaken for taste:

**(a) `ρ_crit` is calibrated per batch, not a constant.** A constant 1.6 rejected
the 80:20 and 90:10 arms (ρ = 1.74/1.69) — the main result — while the genuinely
unequal-variance cases sat at 3.09/4.04. The two groups' ρ ranges **overlap**, so
no constant separates them: ρ is systematically upward-biased under skew because
the rare component has fewer points. `ρ_crit` is therefore the 95th percentile of
ρ's null distribution under a **parametric bootstrap** that resamples `n` points
from the fitted tied model at the fitted `w`. Skew is absorbed because the null is
generated at the same `w`.

**(b) The free-variance fit is used only to veto, never to produce the offset.**
A free-variance EM fixes the unequal-variance case but degenerates under skew: at
90:10 it estimates `μ_1 = 0.352` (true 1.88) while inflating that component's `σ`
to absorb the majority class's tail. BIC does not rescue this — it selected the
free model in **every** cell tested, including cells whose truth is tied. So the
tied fit produces `τ`; the free fit only supplies `ρ`.

## 3. What is already known, and what is therefore not evidence

All development above is on **synthetic** logits (`/tmp`, not in the repo), and one
seed per cell. Development numbers are **not** confirmatory evidence and will not be
quoted in the paper as such. They are recorded here only to fix the thresholds
before the real run, so the thresholds cannot be tuned to the Order-4 outcome.

Development-set state at freeze time (synthetic, seed 0, n=400, K=2):

| cell | BC | SIO | oracle | SIO gate |
| --- | --- | --- | --- | --- |
| 50:50 | +2.00 | +1.00 | +3.25 | apply |
| 60:40 | +1.35 | +1.46 | +3.85 | apply |
| 70:30 | +0.48 | +2.02 | +3.51 | apply |
| 80:20 | −0.63 | +2.34 | +4.22 | apply |
| 90:10 | **−2.36** | **+6.94** | +7.78 | apply |
| single class (either) | −10.94 / −7.81 | +0.00 | +25.78 / +4.69 | abstain |
| sep 0.1 / 0.3 | +1.00 / −0.50 | +0.00 | +1.75 / +1.00 | abstain |
| n = 16 / 32 | +0.00 | +0.00 | +9.09 / +2.27 | abstain |
| unequal var (both directions) | −0.54 / +0.36 | +0.00 | +3.39 / +5.95 | abstain |
| bias=0 at 50:50 / 70:30 / 90:10 | +0.00 / −0.54 / −0.42 | +0.00 | +0.75 / +0.83 / +1.25 | abstain |

Note what this table does **not** show: SIO never beats BC by abstaining *and*
capturing the prize. On four abstention rows the oracle headroom is large
(+25.78, +9.09, +5.95, +3.39) and SIO takes none of it. SIO's claim is
`≥ raw`, not `≈ oracle`.

The multi-seed generalisation check was **launched before this protocol was frozen
but had not returned**; its result is therefore also development evidence and is
reported in §A1 whatever it shows, including if it contradicts the table above.

## 4. Criteria, thresholds predeclared

Run on the real Order-4 T5-large stream, `--cl-method none`, the frozen
`risk`/`audit` splits, gauge-fixed balanced accuracy, cluster-bootstrap CIs
(`cluster_bootstrap_ci`, clusters = tasks, 10000 draws, seed 0), as in every prior
phase. Arms: `raw`, `BC`, `BPO` (2Q's rule, re-scored), `SIO`, `oracle`.

- **S1 (does it beat doing nothing).** Median over (task, stage) of
  `SIO − raw ≥ +1.0 pp` with a cluster-bootstrap 95% CI strictly above 0.
- **S2 (does it beat the published method it fixes).** Median of `SIO − BC ≥ 0`
  with CI not below 0, on **balanced** audit batches — SIO is not required to win
  where BC is already sound, only to not lose.
- **S3 (the defect it claims to fix, the decisive one).** On deliberately
  imbalanced query mixes subsampled from the same audit split (70:30 and 90:10,
  the `subsample_imbalanced` path already in `bpo.py`): median `SIO − BC ≥ +1.0 pp`
  with CI strictly above 0, **and** median `SIO − raw ≥ 0` with CI not below 0.
- **S4 (does the gate actually protect).** Over all batches where the gate
  abstains, `|SIO − raw| = 0` exactly (mechanical check, not statistical), and the
  abstention rate on balanced audit batches is `≤ 0.5` — a rule that abstains
  almost always is vacuous even if it never loses.
- **S5 (marginal-invariance, the mechanism claim).** Re-score the *same* batches
  with the class mix resampled at fixed size. `τ`'s across-mix standard deviation
  must be `≤ 1/3` of BC's correction's across-mix standard deviation. This tests
  the stated mechanism directly rather than inferring it from accuracy.
- **S6 (no free lunch from `K_S > 2`).** SIO is defined only for `K_S = 2`; on
  `K_S > 2` scopes it must abstain by construction, and the reported aggregate must
  state what fraction of (task, stage) pairs that removes.

## 5. Predeclared outcomes

1. **S1, S2, S3 all pass, S4 and S5 pass** → SIO is a working rule that fixes a
   documented defect in a published method. `sec:bpo` is rewritten as the SIO
   section; `sec:enumeration`'s second row gains the corrected rule; a related-work
   paragraph positions against BC and the label-shift line.
2. **S1 and S3 pass, S5 fails** → the gain is real but the stated mechanism is not
   what produces it. Report as a working rule with an *unexplained* mechanism and
   say the invariance argument was not confirmed. Do not claim the mechanism.
3. **S1 passes, S3 fails** → SIO is another balanced-split artefact, exactly like
   BPO. Report as a second artefact and say the defect diagnosis in BC stands while
   our fix for it does not.
4. **S1 fails, S4 passes** → the gate works, the rule has nothing to add on real
   logits. Report as negative; the BC defect diagnosis survives as the contribution
   and SIO is reported as a failed remedy.
5. **`K_S = 2` leaves too few (task, stage) pairs for any CI** → report the
   coverage number and declare the phase underpowered rather than quoting a
   point estimate. This is a live risk: `False|True` is the main `K_S = 2` scope
   and Phase-2R already showed most of its stages are singletons.

Any outcome not on this list gets recorded as a protocol gap in §A1, as in 2R,
rather than patched retroactively.

## 6. Anti-artefact checks, declared before running

- The tied-EM parameter recovery test must pass on synthetic data with known truth
  before any real logits are touched (`σ̂` within 10% of truth, `μ̂` within 0.4).
  This exists because a **dimensional error** in the tied M-step (`v.sum()/n` where
  `v` was already normalised by `n_k`) shrank `σ̂` to 0.06 against a truth of 1.13,
  which inflated `snr` to 42 and silently decoupled `τ` from the oracle offset. Two
  gate designs were built on top of that bug before it was caught.
- SIO's offset must be **identical** whether computed on `Z` or on `Z + c·1`
  (gauge invariance), asserted mechanically.
- SIO must be scored with the same `gauge_fix` and `balanced_accuracy` calls as
  every other arm, from the same logits, in the same process.
- The abstention fallback must be **exactly** the raw prediction, bit-for-bit.
- No official `test.json` is read at any point.

## 7. Cost

The rule is inference-only, so no retraining is required if per-stage logits are
already stored. Phase-2K's `run_qoc.py` stores `risk_logits`/`audit_logits` per
stage; if those cover the needed stages the phase costs **zero GPU**. If they do
not, the phase waits for a free card — `tej` currently holds all eight and the
non-preemption rule is absolute.

---

# §A1 — Amendment, appended 2026-08-31, frozen text above unchanged

## A1.1 §7's zero-GPU branch does not exist

§7 said: "Phase-2K's `run_qoc.py` stores `risk_logits`/`audit_logits` per stage; if
those cover the needed stages the phase costs **zero GPU**." That was checked after
freezing and is **false**. `audit_logits`/`risk_logits` are in-process locals inside
`run_qoc.py`; nothing is written to disk. Verified on the box:

| check | result |
| --- | --- |
| `grep -rl audit_logits experiments --include=*.json` | 0 files |
| `find experiments -name '*.json' -size +1M` | 0 files |
| `find runs -name '*.safetensors' -o -name '*.bin'` | 0 files |
| `runs/phase2q_bpo_s2/` contents | one 604 KB JSON of aggregate metrics only |

No per-stage adapter weights survive either, so logits cannot be recovered by a
cheap forward pass — Phase-2S requires re-running the 15-stage stream. §7's cost
estimate is wrong by the cost of a full training run, and the error was mine: I
wrote "if those cover the needed stages" as a conditional and then treated the
antecedent as satisfied without checking it.

This does not change any criterion in §4. S1–S6 stand as frozen.

## A1.2 Blocked on GPU, not on design

At 2026-08-31T20:5x all eight A100-80GB cards read ~77.5 GiB used of 81.9 GiB
(`tej`'s jobs). The non-preemption rule is absolute, so Phase-2S is queued, not run.
`/mnt/data` is also at 95% (564 GB free), which is enough for one stream but should
be re-checked before launch.

## A1.3 What the development evidence is worth, restated

§3 already declared the synthetic table non-confirmatory. A1 adds that it is now
the *only* evidence, and will stay that way until a card frees. Nothing in the paper
may cite Phase-2S until S1–S6 have been evaluated on Order-4. The BC defect
diagnosis (§0) is independent of this: it rests on 2Q's real-data Q4 collapse plus
the published paper's absent ablation, neither of which needs Phase-2S.

## A1.4 The multi-seed check

Launched before the freeze (§3 disclosed this). Result recorded in §A2 whatever it
shows, including if it contradicts §3's single-seed table.

---

# §A2 — Amendment, appended 2026-08-31, frozen text above unchanged

## A2.1 The multi-seed check overturns §3's headline cell

§3 promised this "whatever it shows". 20 seeds per cell, n=400, same generator:

| mix | BC | SIO | oracle | SIO applies |
| --- | --- | --- | --- | --- |
| 50:50 | +1.59 ± 1.03 | **+3.03 ± 2.04** | +3.86 | 20/20 |
| 60:40 | +0.97 ± 0.77 | **+3.31 ± 1.86** | +4.13 | 20/20 |
| 70:30 | +0.26 ± 0.44 | **+3.33 ± 2.21** | +4.24 | 20/20 |
| 80:20 | −0.96 ± 0.38 | **+3.32 ± 2.39** | +4.63 | 19/20 |
| 90:10 | −1.86 ± 0.73 | **+0.10 ± 0.45** | +5.54 | **1/20** |

§3's single-seed table reported **+6.94 pp at 90:10** and called it the headline.
Over 20 seeds that cell is **+0.10 pp with a 1/20 apply rate**: the +6.94 was one
lucky seed that passed the gate, and the gate correctly refuses the other 19. The
single-seed number must not be quoted again, here or in the paper.

What survives is still substantive and is the *middle* of the range, not the
extreme: from 50:50 to 80:20, BC decays monotonically (+1.59 → −0.96, crossing zero)
while SIO stays flat (+3.03 → +3.32) and always applies. The claim SIO can support
is "does not decay with skew across the range where it can act", not "wins hardest
where skew is worst".

## A2.2 Why 90:10 abstains, and why that is not fixable by loosening

Failure counts over 20 seeds at 90:10: `snr` 0, `ρ` 0, `min(w)` 2, **CI 17**.
The bootstrap CIs are only *marginally* negative — `[−0.36,+0.66]`, `[−0.37,+0.96]`,
`[−0.48,+0.47]` — with point estimates `τ` of +0.44/+1.04/+0.41 against a true bias
of 0.6. With 40 points in the rare component its location resamples unstably, so
the CI straddles 0. The gate is reporting genuine uncertainty, not malfunctioning.

Loosening the CI to capture this cell would be tuning a threshold to a known
outcome, which is what §3 froze the thresholds to prevent. The cell stays negative.

## A2.3 Defect in criterion S3: abstention scores as success

S3 requires, on skewed mixes, `SIO − BC ≥ +1.0 pp` **and** `SIO − raw ≥ 0`. At
90:10 both hold (+1.96 and +0.10) — but only because abstention returns raw while BC
loses 1.86 pp. S3 was written to test *does SIO fix the defect*; abstaining
**declines** the defect rather than fixing it, and S3 cannot tell the two apart.

This is a defect in my criterion, recorded rather than retro-patched. S3 stands as
frozen for the real-data judgement. The following is added, and binds only the
real-data run:

- **S3b (new, binds the Order-4 run).** On skewed mixes, S3 may be scored a pass
  only on batches where the gate **applies**. The apply rate on skewed mixes must be
  reported next to it, and if that rate is below 0.5 the conclusion is written as
  "SIO abstains on most skewed batches" regardless of the margin over BC.

## A2.4 Defect in criterion S5: sd compares noise with bias

S5 requires `sd(τ) ≤ sd(BC correction)/3` across mixes. Measured: ratios 0.55–1.35
over 6 seeds, **FAIL on all 6**. But the underlying values contradict the criterion,
not the mechanism. Seed 0's τ across the five mixes is
0.572 / 0.602 / 0.674 / 0.583 / 0.994 — scattered around the true bias 0.6 with no
trend. BC's is 0.210 / 0.126 / 0.041 / −0.038 / −0.124 — monotone, and it crosses
zero. Their standard deviations are comparable because τ's spread is **sampling
noise** while BC's is **systematic drift**; sd conflates the two.

The mechanism-appropriate statistic is the OLS slope against the mix ratio,
measured over 20 seeds:

| quantity | slope vs mix | sign consistency |
| --- | --- | --- |
| BC's correction | **−0.858 ± 0.016** | negative in **20/20** seeds |
| SIO's `τ` | −0.188 ± 0.599 | negative in 12/20 seeds (i.e. none) |

BC's dependence on the marginal is systematic; SIO's is not distinguishable from
zero, with a spread three times its own magnitude. `|slope_SIO| < |slope_BC|` in 95%
of seeds. This is what §1's locations-vs-weights argument predicts.

- **S5b (new, binds the Order-4 run).** Replace the sd-ratio test with: the OLS
  slope of the correction against the query-batch class ratio must have a
  cluster-bootstrap CI containing 0 for SIO, while BC's excludes 0. Report both
  slopes with CIs. S5 as frozen is reported as FAILED alongside, with this
  explanation, so the amendment cannot be read as moving a goalpost after seeing a
  number: S5's own data is what shows S5 to be the wrong statistic.

## A2.5 Net development-set state

| criterion | development-set verdict |
| --- | --- |
| S1 (`SIO − raw ≥ +1.0`) | passes 50:50–80:20; **fails at 90:10** (+0.10) |
| S2 (balanced `SIO − BC ≥ 0`) | passes (+1.44) |
| S3 (skewed) | letter passes; **defective criterion**, see A2.3 |
| S4 (gate protects; balanced abstain ≤ 0.5) | passes (0/20 abstain at 50:50) |
| S5 (invariance via sd) | **fails 6/6**; wrong statistic, see A2.4 |
| S6 (`K_S = 2` coverage) | not testable without real data |

None of this is confirmatory. Two of six frozen criteria turned out to be badly
formed, which is itself the most useful thing this development round produced.

## A2.6 Sign convention in §A2.4's table, corrected

§A2.4 reports BC's slope as **−0.858 ± 0.016**. That number is the slope of the
**batch mean** `p̄_1 − p̄_0` against the mix ratio. BC's *offset* is the negation of
the batch mean, so as an offset the slope is **+0.858 ± 0.016**. The magnitude,
the ± 0.016, and the 20/20 sign consistency are unaffected, and so is the
comparison with SIO — but the two rows of that table are not sign-comparable as
printed, because SIO's −0.188 is a slope of `τ`, which enters the offset as `−τ`.

Read the table as magnitudes: |BC| = 0.858 with 20/20 consistent sign,
|SIO| = 0.188 with no sign consistency and a spread three times its magnitude.
`marginal_slope` in `experiments/phase2s_sio/sio.py` takes whatever correction it is
handed, so callers must state which convention they passed; the regression test
`test_bc_correction_slope_against_the_mix_is_systematic` asserts magnitude and
monotonicity rather than sign, for this reason.

## A2.7 Implementation and test suite exist; the run does not

`experiments/phase2s_sio/sio.py` and `experiments/phase2s_sio/tests/test_sio.py`
(21 tests, all passing) were written after §A1/§A2 and contain no real-data results.
Tests that pin a *negative* finding, so it cannot be quietly undone:

- `test_tied_em_recovers_known_sigma` — protocol §6's precondition, and the
  dimensional bug that invalidated two earlier gate designs.
- `test_extreme_skew_abstains_on_most_seeds` — pins §A2.1, so the discredited
  single-seed +6.94 pp cannot reappear by loosening the CI gate.
- `test_free_em_degenerates_under_skew_which_is_why_it_only_vetoes` — pins the
  tied/free division of labour.
- `test_rho_critical_value_is_calibrated_not_constant` — pins the per-batch null.

---

# §A3 — Amendment, appended 2026-08-31, frozen text above unchanged

## A3.1 The run moved to a second cluster, and what that changes

§A1.2 said Phase-2S was queued behind `tej`'s eight cards on the Alibaba box. It
instead runs on CMU **BABEL** (`login.babel.cs.cmu.edu`, account `bapoczos`). The
constraint inverts: on Alibaba I must not preempt others, on BABEL my only QOS is
`preempt_qos`, so **my** jobs can be killed at any time and there is no resume
support in `run_qoc.py`. A preempted arm must be resubmitted from scratch.

Environment differences from the Alibaba box, verified rather than assumed:

| item | Alibaba | BABEL | action |
| --- | --- | --- | --- |
| GPU | A100-80GB | L40S 46GB / A6000 49GB | fits; bf16 LoRA on t5-large |
| transformers | 4.57.1 | 4.51.2 as found | **pinned to 4.57.1** |
| peft | 0.18.0 | absent | **installed 0.18.0** |
| torch | 2.10.0+cu128 | 2.6.0+cu126 | left as found |
| PyPI / HF hub | blocked, mirrors required | direct | `HF_ENDPOINT` overridden |

`run_probe.py:36` hardcodes `os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")`
for the Alibaba box's network. On BABEL that mirror is unreachable; the run exports
`HF_ENDPOINT=https://huggingface.co` to override the `setdefault` rather than editing
shared code, so the two clusters run byte-identical sources.

torch differs (2.6.0 vs 2.10.0) and is **not** aligned. This is a declared
difference, not an oversight: swapping torch under a working CUDA stack risks more
than the version gap costs for bf16 LoRA fine-tuning. If any Phase-2S number is
compared with a Phase-2K/2Q number, this difference must be stated.

## A3.2 Data was re-downloaded, not copied

The Order-4 files were fetched on BABEL from the pinned upstream commit
(`cmnfriend/O-LoRA` @ `07117e1f`), 45 blobs, all passing the manifest's sha1 checks.
Class counts match what prior phases recorded, including CB's rarest class at 16
(the constraint recorded in the CB rare-class memo). No file was copied between
clusters, so a corrupted intermediate cannot have propagated.

## A3.3 An epochs sensitivity arm, added and declared

The main arms run at the frozen setting, which yields **3-6 gradient steps per task,
~55 for the stream** — the same order as the Alibaba runs (74), so Phase-2S numbers
stay comparable with 2K/2Q. But at that step count, a failure of S1 cannot be
distinguished from "the model barely trained": the dead-`epochs` defect recorded
earlier was exactly this failure mode, and it is what broke a criterion in 2K.

So one extra arm runs at `--epochs 4`, seed 1 only. It is **not** part of the S1-S6
judgement, which is scored on the frozen setting alone. It exists to answer one
question if S1 fails: does SIO also fail when the stream has 4x the gradient steps?
Reported next to the verdict either way, never merged into it.

## A3.4 Judgement code exists and was committed before the run returned

`experiments/phase2s_sio/decide_2s.py` implements S1-S6 plus S3b/S5b, with 18 tests
in `tests/test_decide_2s.py`. The tests construct data where each criterion both
fires and does not fire, so a real-data verdict cannot come from a criterion that
never discriminates. Two of them pin the recorded criterion defects:

- `test_s3_is_fooled_by_abstention_which_is_why_s3b_exists` — constructs a case
  where every batch abstains, BC loses 2 pp, and S3's letter passes while SIO did
  nothing. S3b returns False on the same data.
- `test_s5_fails_even_when_the_mechanism_holds` — constructs tau scattering around a
  constant against BC marching monotonically, where S5's sd-ratio fails and S5b's
  slope form passes.

## A3.5 A control the protocol did not require, run and positive

Protocol §4 lists BC as the only rival. That is too weak to settle the mechanism
claim, because a rule could beat BC merely by landing on a better threshold. Three
stronger controls were therefore run on synthetic logits (20 seeds/cell, n=400),
none of which appear in the BC paper or in §4:

| mix | BC | BC-bal | BC-med | MID-raw | SIO | oracle |
| --- | --- | --- | --- | --- | --- | --- |
| 50:50 | +1.59 | +0.46 | +2.23 | +2.89 | +2.94 | +3.86 |
| 70:30 | +0.26 | +0.21 | −0.15 | −0.25 | +3.33 | +4.24 |
| 80:20 | −0.96 | −0.04 | −1.65 | −3.62 | +3.32 | +4.63 |
| 90:10 | −1.86 | −0.25 | −2.90 | −6.22 | +1.22 | +5.54 |
| **decay 50:50→80:20** | −2.55 | −0.50 | −3.87 | **−6.50** | **+0.38** | |

`MID-raw` (threshold = median of `s`) matches SIO on balanced batches (+2.89 vs
+2.94) and then collapses *harder than BC* under skew, because a median is a
marginal-dependent quantity. That is the discriminating result: it rules out "SIO
merely found a better threshold". `BC-bal` (estimate the marginal, then reweight to
balanced — the strongest "fix BC by estimating better" version) does arrest the
decay (−0.50) but only by flattening the gain to nothing (+0.46 → −0.04), which is
what §0's root cause predicts: the marginal should not enter at all.

All four rivals decay by 2.55 to 6.50 pp across the range where SIO acts; SIO does
not decay (+0.38). This is still **synthetic** evidence and is not a Phase-2S
criterion. It is recorded here because it was run before the real-data verdict and
would otherwise look like a post-hoc addition.

---

# §A4 — Real-data verdict, appended 2026-09-01, frozen text above unchanged

## A4.1 SIO fails on Order-4. Every criterion, as scored

Seed 1, full 15-task stream, `--cl-method none`, frozen setting. 70 of 120
(task, stage) pairs are `K_S = 2`, so coverage is **not** the limiting factor —
§5 outcome 5 does not apply.

| criterion | result | verdict |
| --- | --- | --- |
| S1 `SIO − raw ≥ +1.0` | median **0.0**, CI [0.0, 0.0], 8 clusters, n=70 | **FAIL** |
| S2 balanced `SIO − BC ≥ 0` | median **−2.34 pp**, CI [−3.52, 0.0] | **FAIL** |
| S3 @ 0.7 | vs BC −0.49, CI [−2.43, +0.78] | **FAIL** |
| S3 @ 0.9 | vs BC +0.78, CI [−0.78, +2.46] | **FAIL** |
| S3b @ 0.7 | apply rate **0.271** | **FAIL** |
| S3b @ 0.9 | apply rate **0.457** | **FAIL** |
| S4 gate protects | 55 abstentions, **55 exactly raw**, abstain rate **0.786** | **FAIL** (vacuous) |
| S5 frozen sd form | sd(τ)=1.415, sd(BC)=0.385, ratio 3.67 | FAIL (and wrong statistic) |
| S5b slope form | SIO CI **[−1.90, −0.066]** excludes 0; BC CI [+0.044, +0.407] excludes 0 | **FAIL** |
| S6 coverage | 70/120 = 0.583; K_S hist {2:70, 3:29, 5:13, 14:4, 4:3, 10:1} | pass |

S1's median is exactly 0.0 with a zero-width CI because the gate abstains on 55 of
70 batches and abstention is exactly raw — S4's mechanical check confirms all 55 are
bit-for-bit identical, so the implementation is correct and the *rule* is what fails.

No §5 outcome fires verbatim. Outcome 4 ("gate works, rule adds nothing") is the
closest but requires S4 to pass, and S4 fails its own vacuity bound (0.786 > 0.5).
`decide_2s.py` therefore returns `PROTOCOL_GAP_record_in_amendment`, which is this
section. As in 2R, the gap is recorded, not patched.

## A4.2 The mechanism claim is refuted on real logits

This is the finding that matters, and it goes against §1. S5b was added in §A2.4 as
the *correct* form of the invariance test, before any real data existed. It now
reports SIO's τ slope CI as **[−1.90, −0.066], excluding zero**. τ moves
systematically with the batch class ratio on real T5 verbalizer logits.

So the locations-vs-weights argument does not transfer. Discarding `w` is not
sufficient for marginal invariance when the component locations are themselves
estimated from a mixture that is not two Gaussians.

BC's slope CI [+0.044, +0.407] also excludes zero, so **the defect diagnosed in
Batch Calibration is confirmed on real data**. What fails is our remedy, not the
diagnosis. That distinction is the one thing Phase-2S establishes positively.

## A4.3 Why it fails, measured

Abstention reasons over the 70 scorable batches:

| gate term | rejections | share |
| --- | --- | --- |
| `snr < 1.5` | 23 | 33% |
| `min(w) < 0.05` | 19 | 27% |
| `τ` CI contains 0 | 9 | 13% |
| `ρ > ρ_crit` | 4 | 6% |
| **applied** | **15** | **21%** |

`min(w) < 0.05` on 27% of batches is the diagnostic: the fitted mixture puts almost
no weight on one component, i.e. **`s = z_1 − z_0` is not bimodal on real logits**.
Per-task apply rates make this concrete — BoolQA **0/10**, RTE **0/9**, IMDB
**0/8**, SST-2 **0/5** never pass the gate at all, while QQP passes 8/11.

τ's observed range is **−6.219 to +2.106** (median −0.338). On synthetic data it sat
within ±0.2 of the injected bias 0.6, and sd(τ) across mixes was 0.08–0.16 versus
**1.415** here — an order of magnitude larger.

## A4.4 What I got wrong, stated plainly

The synthetic development set generated `s` as a two-component **equal-variance
Gaussian mixture** and then verified that a two-component equal-variance Gaussian
mixture recovers it. That is circular: §3's table, §A2.1's multi-seed table, and
§A3.5's MID-raw control all test the estimator against its own generative
assumption. They are internally valid and jointly worthless as evidence about T5
logits.

The MID-raw control in §A3.5 is the sharpest illustration. It was designed to rule
out "SIO merely found a better threshold", and on synthetic data it did. But both
SIO and MID-raw were evaluated on data whose `s` is bimodal by construction, so the
comparison never tested the assumption that both share.

What would have caught this before spending a run: fit the mixture to **stored real
logits** and check `min(w)` and bimodality *first*. That was impossible for
Phase-2S because the Alibaba runs stored no logits (§A1.1) — but the correct
response was to dump logits in a short run and look, not to develop four gate
versions against a generator I wrote myself. The `--dump-logits` path added for this
run means the next inference rule can be tested this way at **zero GPU cost**:
70 real `K_S = 2` batches are now on disk.

## A4.5 Standing after Phase-2S

- **Survives:** BC's marginal dependence, confirmed on real Order-4 logits
  (slope CI [+0.044, +0.407]). Its Appendix B lists no imbalance caveat and the
  paper has no skewed-batch ablation. Phase-2Q's Q4 collapse and this slope are
  independent real-data evidence for the same defect.
- **Refuted:** SIO as a remedy (S1-S5b all fail), and the locations-vs-weights
  invariance argument as stated in §1.
- **Refuted:** the synthetic evidence in §3, §A2.1 and §A3.5 as support for anything
  about real logits, for the circularity in §A4.4.
- **Unchanged:** `sec:enumeration`'s two admissible rows. SIO occupied the same row
  as BPO and both now fail; the row is non-vacuous but nothing in it works yet.
