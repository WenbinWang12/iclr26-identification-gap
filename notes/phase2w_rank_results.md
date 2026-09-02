# Phase-2W results — does LoRA rank buy identification?

Scored by the frozen `experiments/phase2w_rank/score_2w.py`
(SHA `8b6073067b95b76f1e510fd16cc26dc4cab8dd8930ca6d19c5bb6192164a02a9` at freeze;
the only post-freeze edit is the seed-labelling fix recorded in §6). Protocol and its
two amendments: `notes/phase2w_rank_exchange_protocol.md`. Records:
`runs/phase2w_rank/` — 24 runs, 15 stages each, all `EXIT=0`: **18 primary**
(6 ranks × 3 seeds, `epochs=3`) plus the **6-run `W6` arm** (6 ranks × seed 1,
`epochs=1`, §7). Decision: `runs/phase2w_rank/decision_2w_full.json`.

## 1. The answer

**Rank does not buy identification.** Over a 32× range of LoRA rank, `Δ_id` — the
gain from giving the shared readout a free per-scope constant — does not trend.

| r | `Δ_id` median | CI | `Δ_id` mean | `R_raw` | id. share |
| --- | --- | --- | --- | --- | --- |
| 1 | +1.56 | [+0.00, +3.13] | +2.47 | 0.6795 | 0.65 |
| 2 | +1.56 | [+0.33, +3.13] | +3.20 | 0.6777 | 0.82 |
| 4 | +1.56 | [+0.57, +2.40] | +2.39 | 0.6758 | 0.63 |
| 8 | +0.78 | [+0.00, +2.34] | +2.20 | 0.6944 | 0.70 |
| 16 | +2.08 | [+0.78, +3.91] | +3.04 | 0.6939 | 0.73 |
| 32 | +0.78 | [+0.17, +2.34] | +2.51 | 0.6903 | 0.71 |

`W2 = H_flat`. Spearman ρ against rank is **−0.20 on medians and +0.03 on means** —
the sign flips with the choice of statistic, which is the cleanest single statement
that there is no trend to find. `W3`'s paired r=32 − r=1 contrast is
**+0.12 pp, CI [−0.70, +0.83]**, 12 task clusters, within-seed pairing.

`R_raw` is flat too: **+1.08 pp, CI [−0.17, +2.16]** from r=1 to r=32. Capacity is
not binding for raw accuracy either, so the flat `Δ_id` is not the special case of a
saturated model — nothing in this task suite is rank-limited above r=1.

**Consequence for the paper — a control, not a refutation.** The `+1.5`-ish pp
identification prize is **not an artifact of an undersized adapter**: it survives 32× of
rank, and `R_raw` is flat over the same range, so the sweep is neither
capacity-starved nor saturated. That answers the reviewer objection "your effect
disappears with a bigger adapter", which is what this arm is for.

It is **not** a refutation of standard practice. Per Amendment 3, "raise rank if you are
forgetting" is a strawman I wrote into the protocol's H-grow motivation; the field
already responds to forgetting with regularisation, replay, or isolation, and *rank
allocation* methods redistribute a fixed budget rather than enlarge it. This result is
**consistent with** the common observation that rank does not fix forgetting, not
contrary to it. What is new is the dissociation: a zero-rank constant buys ~2 pp
(24/24 runs, p=1.2e-07) where 32× of rank buys nothing (paired CI upper bound
+0.83 pp), so this damage is not in a shape rank can repair. `OUTCOME_1` as
pre-registered.

**And this sweep cannot answer whether higher rank *hurts*.** Measured: ρ(rank, `R_raw`)
= +0.54, ρ(rank, first-seen→final drop) = −0.37, both inside the 6-point permutation
null and non-monotone (worst drop at `r=4`). The disqualifying reason is structural:
`lora_alpha` is fixed at 32, so effective update scale `alpha/r` falls 32× as rank
rises — suppressing the very mechanism by which a bigger adapter would overwrite more.
See Amendment 3(b) and §5.

The `epochs=1` sensitivity arm (`W6`, §7) flips the trend's sign to **+0.35** while its
magnitude still cannot be told from zero (permutation p = 0.51). That is the third
analysis choice under which the sign moves freely, and it strengthens rather than
qualifies the null.

## 2. What this measurement cannot do — read before quoting §1

Two limits, both measured rather than assumed, and both large relative to the effect.

**(a) The statistic is quantized at 0.78125 pp.** Each task is scored on 128 examples,
so every per-row value lands on a `1/128` grid, and the median over 12 tasks lands on
it too. The entire `Δ_id` effect is ~2 grid units wide. Four of the six median values
in §1 are literally the same 1 or 2 units. The mean (over 279 rows) is far finer and
is why §1 reports both; it agrees, which is what licenses the flat reading.

**(b) Re-running the same configuration moves the answer by more than the trend.**
`r=8, epochs=3, cap=400` exists twice as three-seed runs — `runs/phase2q_bpo` and this
sweep's `r=8`:

| run | `Δ_id` median | CI |
| --- | --- | --- |
| `runs/phase2q_bpo` | **+2.08** | [+0.78, +3.65] |
| `phase2w r=8` | **+0.78** | [+0.00, +2.34] |

Same config, independent weights: point estimates differ by **1.30 pp (1.7 grid
units, a factor of 2.7)**. Each CI contains the other's point estimate and both
contain the paper's `+1.51`, so the two runs are not distinguishable — but the
between-run spread of this measurement is *comparable to the effect it measures*.

**Therefore the honest form of the headline is a bound, not a flatness claim:** this
design rules out a capacity effect **larger than ~1.3 pp** across 32× of rank. It
cannot rule out a smaller one. `W3`'s CI half-width of 0.83 pp is *narrower* than the
observed between-run spread, because with 3 seeds the between-run component is barely
represented in it; the looser number is the one to quote. "No monotone trend is
detectable at this resolution" is supported. "The curve is flat" is not.

## 3. Incidental finding — task identity does beat scope, by about a third of the prize

**Not pre-registered.** Found in the sweep data while checking a claim 2W was not
designed to test. It bears directly on a headline claim, so it is reported here rather
than dropped, and it needs its own confirmation run before it goes in the paper.

`notes/phase2p_streaming_scope_offset_protocol.md` reports
`per-task ORACLE − per-scope = +0.02 pp, CI [−1.00, +1.00]`, read as zero: scope is as
good as knowing the task. On these 21 runs (18 here + 3 `phase2q_bpo`), the median
reproduces that exactly — `+0.000 pp`, with a **degenerate CI of [0, 0]** at every
single rank. The mean does not:

| statistic | `R_orc − R_shr_scoped`, pooled | verdict |
| --- | --- | --- |
| median | +0.000 pp, CI [+0.000, +0.000] | zero, as published |
| mean | **+0.83 pp**, CI [+0.10, +1.55] | excludes zero |

The mechanism, at row level (n = 1674):

* **60.9 %** of rows: oracle and scope are **exactly equal** — this saturates the
  median at zero and is why the published median reads as no effect.
* 26.6 % favour the oracle, 12.5 % favour scope — **2.1 : 1**.
* When the oracle wins it wins by **+4.17 pp** on average.

Per-run means are positive in **21 of 21 runs** (sign test p = 4.8e-07; Wilcoxon on
run means p = 4.8e-07), range +0.10 to +1.98 pp, sd 0.49. The sign never flips, across
6 ranks and 2 independent run batches.

**Reading.** Scope reproduces the oracle *exactly* on most rows, and the published
median-based "zero" is a true statement about the median. But task identity does carry
information beyond scope — roughly **0.8 pp against a prize of 1.5–2.5 pp, about a
third** — concentrated in a minority of rows where it is worth several points. The
claim that survives is "scope recovers most of the identification prize, and is exactly
equal to the oracle on ~61 % of measurements", not "scope is as good as the oracle".

**Caveats that keep this provisional.** The mean's CI ignores between-run variance the
same way §2(b) describes, and its lower bound (+0.10 pp) sits below the ~1.3 pp
resolution floor — so the *magnitude* is not established, only the *sign*, which the
21/21 run-level sign test does support independently of any CI. Pooling across ranks
to get power also assumes ranks are exchangeable for this quantity; that is defensible
only because `Δ_id` is flat, which is itself a 2W result rather than an input.

## 4. What `W4` refused to let me claim

`Δ_id` at r=32 is numerically below r=1, which invites "high rank shrinks the prize".
`W4` blocks it: the pre-registered mechanism for a shrink claim is that the
identification *share* falls with rank, and it does not (0.65, 0.82, 0.63, 0.70, 0.73,
0.71 — no trend). `shrink_claimable = false`. Recorded because the guard did real work
here; without it the r=1-vs-r=32 numbers would have supported a story the mechanism
does not.

## 5. The confound that limits the mechanistic reading

Per Amendment 2, written before any result was visible: `lora_alpha` is fixed at 32,
so PEFT's `alpha/r` scaling moves the effective update magnitude **inversely to rank,
32× across this sweep**. Every claim here is therefore about
**rank-with-default-alpha**, the sweep a practitioner actually runs — never about
capacity in isolation. Since the outcome is a null, the confound is mild: it would have
to conspire to cancel a real capacity effect exactly. But the constant-scale
disambiguation (`alpha = 4r`, 6 runs) pre-registered in Amendment 2 **remains required
before any capacity-language claim enters the paper.**

## 6. Provenance and deviations

* **`W1` passes, marginally and weakly.** Swept `r=8` gives `Δ_id = +0.78125` against a
  baseline CI of `[+0.78, +3.65]` — inside by 0.00125 pp, i.e. at the boundary. And per
  Amendment 1(b) that baseline is a *local* recomputation on `runs/phase2q_bpo`, not
  the paper's `+1.51`, whose source records are absent from the repo; the CI is also
  wide enough to contain `+1.51`, so `W1` cannot distinguish the two references. It is
  a weak consistency check and nothing more.
* **The `+1.51` headline still cannot be recomputed from this repo.** Flagged in
  Amendment 1(b). 2W neither restates nor depends on it, but it must be closed before
  submission.
* **Amendment 1(a)** corrected the training budget from `epochs 1` to the converged
  `epochs 3, cap 400` before launch, so these runs are comparable to the paper's.
* **One post-freeze scorer edit**, made while runs were in flight and before any
  result was read: seeds are now parsed from the run directory name instead of by
  enumeration order, because a missing run would otherwise shift a rank's seed labels
  and `W2` pairs per-seed curves by that label. Test added
  (`test_seed_parsed_from_dir_name_not_enumeration`). No decision logic touched.
* **2 of 18 jobs first failed** with exit 75 — the sweep's own guard refusing
  Blackwell (`RTX_PRO_6000`) nodes, where this torch build dies in ~16 s while Slurm
  still reports `COMPLETED`. Resubmitted under a GPU-family constraint. Both completed
  normally. Without the guard those two r=32 runs would have entered the analysis as
  silent partial records.
* **`W6` (convergence sensitivity, `epochs=1`, seed 1, 6 ranks)** — jobs
  `10279450_[0,3,6,9,12,15]`, all `EXIT=0`; scored in §7. Amendment 1(a) had moved this
  arm to the *under*-trained end when `epochs=3` became the primary. Adding it changed
  nothing above `W6`: `curve`, `W1`–`W5`, `seeds`, `outcome` are byte-identical between
  `decision_2w.json` (primary only) and `decision_2w_full.json` (both arms), which is
  the check that the arm was not allowed to reach back into the headline.
* **Verdict files.** `runs/phase2w_rank/decision_2w_full.json` is the one to cite;
  `decision_2w.json` is the earlier primary-only pass, kept for that diff.
* **24 runs total** (18 primary + 6 `W6`), all `EXIT=0`, records in
  `runs/phase2w_rank/`. Scorer test suite: 15 passed.

## 7. `W6` — the sign flips at the under-trained end, and that is the point

Six runs, `epochs=1`, seed 1, all six ranks (jobs `10279450_[0,3,6,9,12,15]`, all
`EXIT=0`, 15 stages each, `lora_r` verified per record). Scored by the same frozen
scorer through its `--ep4` port. Adding the arm left `curve`, `W1`–`W5`, `seeds` and
`outcome` **byte-identical** — only `W6` changed from `ran: false`.

| r | `Δ_id` ep1 (1 seed) | `Δ_id` ep3 (3 seeds) | `R_raw` ep1 | `R_raw` ep3 |
|---|---|---|---|---|
| 1 | +2.08 | +1.56 | 0.6131 | 0.6795 |
| 2 | +2.60 | +1.56 | 0.6342 | 0.6777 |
| 4 | +0.78 | +1.56 | 0.6455 | 0.6758 |
| 8 | +2.08 | +0.78 | 0.6316 | 0.6944 |
| 16 | +3.91 | +2.08 | 0.6100 | 0.6939 |
| 32 | +2.34 | +0.78 | 0.6288 | 0.6903 |

`W6` **passes as pre-registered** (`agrees: true`), but not because the signs match —
they don't. ρ is **+0.35 at `epochs=1`** against **−0.20 at `epochs=3`**. The scorer's
rule forgives this because `|med_rho| < 0.5` means the primary made no signed claim to
contradict. Both readings are on the record:

* **Literal §3 wording** ("if its sign differs from the primary, the primary is reported
  as regime-specific in those words") — triggered. The sign does differ.
* **What the numbers support** — nothing to be regime-specific *about*. ρ = +0.35 is
  inside its own noise floor: an **exact permutation test over all 6! = 720 orderings
  gives p = 0.51**, and the null sd is 0.447. The paired `r=32 − r=1` contrast on this
  arm is `+0.000 pp, CI [−0.000, +1.56]` by median and `+0.96 pp, CI [−0.40, +2.54]` by
  mean — zero inside both, exactly as `W3` found at `epochs=3`.

So this is a **third** independent way the trend's sign flips without its magnitude ever
leaving zero: median vs mean (§1), and now `epochs=3` vs `epochs=1`. A sign that flips
under three arbitrary analysis choices is the cleanest evidence available that there is
no trend to sign. Reporting "regime-specific" would be naming structure in noise, so §1
stands unqualified and the flip is reported here instead.

**One correction to the protocol's own threshold.** `W2` declares flat when
`|ρ| < 0.5`. That cut is *too permissive*, not too strict: with 6 rank points, random
orderings reach `|ρ| ≤ 0.81` at p = 0.05, so a ρ of 0.6 would have been read as a trend
while being ordinary noise. The observed 0.20–0.35 are far inside the null either way,
so no conclusion here changes — but the `|ρ| < 0.5` rule should not be reused as
pre-registered.

**Incidental, one seed, do not quote as a result (W6 arm).** `epochs=1` is genuinely
under-trained (`R_raw` 0.627 vs 0.685 pooled) and `Δ_id` is *larger* there (pooled
median +2.21 vs +1.56; mean +4.08 vs +2.64). The direction is what the paper's framing
would predict — an unconverged readout leaves more for a per-scope constant to recover —
but the median difference (0.65 pp) is **below** the ~1.3 pp between-run noise floor of
§2 and the mean difference (1.44 pp) is merely at it, on **one seed against three**.
Directional only. Also note `share = 1.00` at `r=1` in this arm is degenerate, not a
finding.

## 8. A defect in `Δ_id` itself, found after scoring — read before §1 too

Found while asking whether the argument closes, not by any pre-registered criterion.
It does not change 2W's verdict; it changes what the *quantity* means, in both
directions.

**Singleton scopes make `Δ_id` a different quantity.** `scoped_shared_offsets`
(`run_qoc.py:170`) *copies* `per_task_offsets[only.task]` when a scope has one member —
deliberately, and documented there, to kill a search-procedure artifact. The
consequence was not carried through to `Δ_id`: on those scopes
`R_shr_scoped = R_orc` **by construction**. Verified: `mean(R_orc − R_shr_scoped)`
is exactly `+0.000` for all five of AGNews, COPA, DBpedia, RTE, Yahoo.

So for 5 of 12 scorable tasks, `Δ_id` is not "what a free per-scope constant buys". It
is the **per-task oracle** gain, which needs task identity — the very thing the paper's
framing says scope makes unnecessary. Those tasks are **522 of 1674 rows (31%)** and
carry the *larger* effect (mean `+3.83` vs `+2.09` pp), so they inflate the headline.

**Recomputed on the 7 tasks whose scopes genuinely contain ≥2 tasks:**

| r | `Δ_id` median | CI | `Δ_id` mean |
|---|---|---|---|
| 1 | +0.65 | [+0.00, +2.08] | +1.73 |
| 2 | +1.56 | [+0.00, +2.47] | +2.70 |
| 4 | +1.56 | [+0.00, +2.60] | +2.06 |
| 8 | +0.00 | [+0.00, +1.04] | +1.66 |
| 16 | +1.56 | [+0.78, +2.34] | +2.42 |
| 32 | +0.78 | [+0.00, +1.56] | +2.00 |

* **The 2W null survives and strengthens.** ρ = **+0.03 (median), −0.03 (mean)** —
  flatter than the full set, and still sign-flipping between statistics. Unchanged
  verdict, now on the honest subset.
* **The positive leg weakens.** Median CIs touch zero at 4 of 6 ranks. What survives is
  the sign, not the magnitude: per-run mean is **18/18 positive, p = 7.6e-06**, range
  `+0.32` to `+3.62` pp. Mean level drops `+2.5 → +2.1` pp.
* **`Δ_id`'s free parameters are not 18.** The protocol's title and §2/§6 say "18
  floats" throughout. Counted from `scope_members`: the three multi-task scopes touching
  scorable tasks (`Bad|Good`, `False|True`, `contradiction|entailment|neutral`) are
  **7 floats**; all scopes touching scorable tasks are **39**; all scopes are **44**. I
  have not located a construction giving 18. **The number is wrong and must be
  recounted wherever it appears** — the protocol's framing claim ("18 floats do work
  that 9.4 M parameters do not") is otherwise a claim about a quantity that does not
  exist. The *dissociation* it names is intact; only the count is wrong, and 7 floats
  makes it stronger, not weaker.

**This also corrects §3.** §3 read the oracle−scope gap as "scope is exactly equal to
the oracle on ~61% of measurements". That 60.9% tie rate was **inflated by the
forced-zero singletons**. On multi-task scopes only: **43.2% tied**, and the gap is
**`+1.21` pp, CI [+0.32, +2.03], 18/18 runs positive** (was `+0.83`). At the final stage
alone it is **`+2.54` pp** against a scope gain of `+2.02` pp.

**So scope is not near-sufficient, and §3's "about a third of the prize" understated
it.** On the tasks where scope is a real constraint, task identity is worth roughly as
much as scope itself. This cuts *against* the paper's framing and is the most important
thing in this file. It is a within-2W recomputation on 18 runs, not a pre-registered
result, and it needs its own confirmation — but the sign is 18/18 and does not depend on
the ~1.3 pp noise floor.

Note this is a different quantity from §6 of the paper, whose "per-task refit `0.7229`"
is a *sequential* refit, not `R_orc`. §6's own singleton split (`r_S = 0`,
`−4.18` pp row) shows the main text handles singletons separately. The defect is in
`Δ_id` as 2W and the protocol use it; whether it propagates into the main text's
`+1.51` is **not yet checked** and is the first thing to check next.
