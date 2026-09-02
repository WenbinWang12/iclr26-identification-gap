# Phase-2W — Does rank buy identification? The exchange rate between LoRA rank and 18 floats

Frozen 2026-08-31, **before any rank other than 8 has been trained**. Written after
Phase-2U closed the query-batch row (four pre-registered rules failed; the mechanism
in §A1.3 of that protocol survived). This phase does not propose a rule. It measures
a quantity, so unlike 2P/2Q/2S/2T/2U it cannot "fail" — every outcome is a number
this paper needs.

## 1. Why this is the right question now

The paper asks how to spend a fixed PEFT budget under continual learning. Phase-2K
measured that a shared per-scope additive offset recovers `+1.51` pp of the forgetting
that a global offset cannot (`scope_recoverable = R_shr_scoped − R_shr_global`,
cluster-bootstrap median over all (task, stage) rows, CI `[+0.43, +2.74]`). That
offset costs, for the 9 scopes actually observed on Order-4 at `K_S=2..14`:

| state | floats |
| --- | --- |
| per-scope gauge-fixed offset, all 9 scopes | **18** (2 per binary scope; ≤ `K_S−1` in general) |
| LoRA `r=8` on T5-large `q,v` (144 modules, `d=1024`) | **2 359 296** |

That is a ratio of `131 072 : 1`. Every rehearsal-free way of *obtaining* those 18
floats at inference has now been tested and closed. But the allocation question was
never asked in the other direction: **if I simply spend more rank, do I get the
identification gap for free?**

Nobody in this project has trained anything but `r=8`. The answer is unmeasured.

## 2. The prediction being tested

`Δ_id` is the part of forgetting that is *shape*-restricted: recoverable by a
zero-rank additive offset in verbalizer-logit space. Rank buys directions in weight
space. The two are not obviously exchangeable, and there are three coherent
hypotheses:

* **H-flat.** `Δ_id` is rank-invariant. Additive logit drift is not the kind of
  damage extra directions repair, so the 18 floats remain necessary at every budget.
* **H-shrink.** `Δ_id` decreases with rank. Extra capacity absorbs offset-shaped
  damage, and there is a finite rank at which the 18 floats stop mattering. That rank
  *is* the exchange rate, and it is the number a practitioner wants.
* **H-grow.** `Δ_id` increases with rank. More capacity produces more offset-shaped
  damage. This is the most consequential outcome: it would mean the standard advice
  "raise rank if you are forgetting" makes the identification component worse.

All three are publishable. H-grow and H-flat both say the 18 floats are not
substitutable by budget; H-shrink prices them.

## 3. Design

Sweep `--lora-r ∈ {1, 2, 4, 8, 16, 32}` — a factor of 32 in adapter capacity,
`0.29 M` to `9.44 M` trainable parameters — on **seeds 1, 2, 3**, Order-4, 15 tasks,
`t5-large`, everything else byte-identical to the runs already reported. `r=8` is the
already-published setting and acts as the internal consistency check: its `Δ_id` must
land within CI of `+1.51` pp or the sweep is not comparable to the paper and I say so.

**Reusing seeds 1–3 is deliberate and admissible here.** No threshold is being
fitted; the estimand is a *trend across rank* and each seed contributes a paired
(rank, `Δ_id`) curve. Seed reuse would only be a problem if a rank were selected on
these seeds and then reported as held-out, which W5 below explicitly forbids.

**Fixed training budget.** `--epochs 1`, `cap-per-class 200`, giving the 3–16
gradient steps per task that every prior phase used. This is the fixed-budget regime
the paper is about, and rank is the only knob that moves. A convergence objection is
real and pre-answered by a sensitivity arm, not by changing the primary.

**Sensitivity arm, reported beside and never merged:** the same sweep at
`--epochs 4` on seed 1 only. If the `Δ_id` trend has the same sign there, convergence
does not explain it.

Total: 18 primary runs + 6 sensitivity runs, 1 GPU each, ~40 min each.

## 4. Statistic and scoring

Primary statistic is `run_qoc.py`'s own `scope_recoverable`, aggregated by the
**already-committed** `experiments/phase2k_qoc/decide.py:criterion_1`, i.e. a
cluster bootstrap over (task, stage) rows clustered by task. Using the existing
scorer is the point: it is the same function that produced `+1.51`, so the sweep is
comparable by construction and I cannot tune the aggregation after seeing the curve.

Secondary, read from the same records, no new fitting:

* `Delta_id_global = R_orc − R_shr_global` — the total gap a *task-indexed* oracle
  offset reaches, to separate "rank reduces damage" from "rank reduces the
  identification share of damage".
* `R_raw` mean over scorable (task, stage) rows — absolute forgetting level, so a
  shrinking `Δ_id` caused merely by less total damage is visible as such.
* `Δ_id / (R_orc − R_raw)` — the identification **share** of recoverable loss. This
  is the confound-resistant version of the headline and is why H-shrink cannot be
  claimed from a bare drop in `Δ_id`.

## 5. Criteria, frozen

| id | question | statistic | decision |
| --- | --- | --- | --- |
| **W1** | consistency with the paper | `Δ_id` at `r=8` | must lie inside `[+0.43, +2.74]`; if not, report the sweep as a separate measurement and do not restate `+1.51` |
| **W2** | direction | sign of Spearman `ρ(r, Δ_id)` over the 6 ranks, per seed | H-flat if `|ρ|` median `< 0.5` **or** the `r=1` vs `r=32` paired difference has CI containing 0; H-shrink if `ρ < 0` and CI excludes 0; H-grow if `ρ > 0` and CI excludes 0 |
| **W3** | magnitude | paired `Δ_id(r=32) − Δ_id(r=1)`, cluster bootstrap | reported with CI whatever the sign |
| **W4** | share, not level | `Δ_id / (R_orc − R_raw)` at each rank | reported; H-shrink is only claimed if BOTH `Δ_id` and this share fall |
| **W5** | no post-hoc rank picking | — | the headline is the trend over all 6 ranks; no single rank may be selected as "the" answer after seeing results, and the exchange rate under H-shrink must be read off the curve by interpolation, stated with its CI |
| **W6** | convergence not the cause | `epochs=4` arm, seed 1 | reported beside; if its sign differs from the primary, the primary is reported as regime-specific in those words |

## 6. Predeclared outcomes

1. **H-flat confirmed** (W2 flat). Report: the identification gap is not purchasable
   with rank across a 32× budget range; 18 floats do work that 9.4 M parameters do
   not. This closes the allocation question in the paper's own terms and is the
   result §4 needs.
2. **H-shrink confirmed.** Report the exchange rate: the rank at which `Δ_id` and its
   share both reach zero, with CI, plus the parameter count that rank costs versus 18.
   State plainly that raising rank is then a valid substitute, which would be a
   positive prescription and a *negative* result for the paper's framing.
3. **H-grow confirmed.** Report that added capacity increases the offset-shaped share
   of forgetting, and that the common remedy of raising rank makes identification
   worse. Flag it as the strongest claim available and require the `epochs=4` arm to
   agree before it goes in the abstract.
4. **W1 fails** (`r=8` outside the published CI). Then this sweep is measuring
   something the paper's `+1.51` did not; report both, do not reconcile them silently,
   and investigate before claiming anything about rank.

## 7. What is frozen with this file

* `experiments/phase2w_rank/sweep_rank.sh` — the sbatch array, no scoring logic.
* `experiments/phase2w_rank/score_2w.py` — reads run records, calls the **existing**
  `phase2k_qoc.decide.criterion_1`, emits W1–W6. Committed and SHA-pinned before the
  first non-`r=8` run exists.
* Scoring reads only `record.json` fields listed in §4. It must not read logits, so
  it cannot be quietly turned into a rule-fitting script.

Amendments append below a `---` rule and never edit anything above it.

---

## Amendment 1 — training budget and the W1 baseline, corrected before launch

Made 2026-08-31, still **before any rank other than 8 has been trained**. Two errors
in §3 and §5 above, both found by checking the frozen scorer against real records
instead of against my memory of them.

**(a) The training budget in §3 was wrong.** §3 specifies `--epochs 1` and
`cap-per-class 200`, which I chose believing every prior phase ran at the 3–16
gradient steps those defaults produce. The runs this paper actually reports are
converged: `epochs=3`, `cap_per_class=400`, `update_cap_per_class=400`. A sweep at
`epochs 1` would be internally valid but connectable to nothing in the paper. The
primary sweep therefore runs at the **converged** configuration, byte-identical to
`runs/phase2q_bpo` except for `--lora-r`:

```
--epochs 3 --cap-per-class 400 --update-cap-per-class 400 --m-values 1,4
--risk-per-class 64 --audit-per-class 64 --min-rare-class 40
--batch-size 4 --grad-accum 16 --lr 3e-4 --dtype bfloat16
--max-source 512 --max-target 8 --cl-method none --n-tasks 15
```

The convergence sensitivity arm of §3/W6 inverts accordingly: it now runs at
`--epochs 1` (the *under*-trained end) on seed 1, and W6 still asks only whether the
sign of the trend survives.

**(b) The W1 baseline in §5 was not reproducible.** W1 compared `Δ_id(r=8)` to the
published `+1.51` pp CI `[+0.43, +2.74]`. That number comes from
`runs/phase2k_qoc_converged`, whose per-run records are **not in this repo** — only a
phase2R decision file over them is. The local `runs/phase2q_bpo` records are a
*different* run at the same design (`mean R_raw = 0.6945` vs the `0.7077` reported
for the same arm in `notes/phase2m_vla_protocol.md` §A2.3), so they cannot recompute
`+1.51`; and none of median, mean, or pooled-ACC over `R_shr_scoped − R_shr_global`
reproduces it (they give `+2.08`, `+2.97`, `+2.97` pp respectively). I do not know
which estimator produced `+1.51`, so I will not silently compare against it.

**W1 is restated.** The reference is the r=8 baseline computed by *this phase's own
frozen scorer* on the local records `runs/phase2q_bpo`, which is
`Δ_id = +2.08 pp, CI [+0.78, +3.65]` (row median, task-clustered, 10 000 draws,
279 rows, 12 tasks). W1 passes iff the swept `r=8` point estimate lies inside that
CI. This is a weaker check than comparing to the paper — it verifies the sweep
reproduces a *local* r=8 measurement, not the published one — and the writeup must
say so in exactly those terms.

**Consequence for the paper, to be handled separately from this phase.** The `+1.51`
pp headline cannot currently be recomputed from anything in the repo. Whatever 2W
finds, that provenance gap is now known and must be closed before submission, either
by recovering `runs/phase2k_qoc_converged` or by restating the headline on records
that exist. 2W does not restate it and does not depend on it.

**(c) `PUBLISHED_CI` in the scorer** is renamed to `BASELINE_CI = (0.78, 3.65)` with
the provenance above recorded beside it, and the test that pinned the old constant is
updated to pin the new one. No other decision logic changes.

---

## Amendment 2 — the `lora_alpha` confound, recorded while the sweep is in flight

Written 2026-08-31 with jobs `10277484_[0-17]` running and **no result yet visible**.
Recording it now so it cannot become a post-hoc excuse for whichever way the curve
goes.

`run_qoc.py:275` hardcodes `lora_alpha=32`. PEFT scales the LoRA update by `alpha/r`,
so holding `alpha` fixed while sweeping `r` moves the effective update magnitude
**inversely to rank, by a factor of 32 across this sweep**:

| r | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| `alpha/r` | 32 | 16 | 8 | 4 | 2 | 1 |

So `Δ_id(r)` as measured here is the response to **rank and effective step size moving
together**, not to capacity alone. Three things follow, and all three go in the
writeup:

1. **This is nonetheless the practitioner's sweep.** Raising `r` while leaving `alpha`
   at its default is what the PEFT default invites and what the paper's own `r=8`
   config does. A result stated as "raising rank the way people actually raise it does
   / does not buy identification" is the honest and useful claim, and it is the claim
   2W makes.
2. **It weakens the mechanistic reading.** If the curve slopes, I may not attribute
   the slope to capacity. The permitted phrasing is "rank-with-default-alpha", never
   "capacity". H-grow in particular could be an effective-LR effect: larger steps at
   low `r` produce more drift, which would show as `Δ_id` falling with `r`, i.e. as
   H-shrink, for a reason that has nothing to do with capacity absorbing offsets.
3. **Disambiguation is a separate, cheap follow-up, not a rescue.** Re-running two
   ranks at constant scale (`alpha = 4r`, matching `r=8`'s scale of 4) separates the
   two factors: if `Δ_id(r=1)` and `Δ_id(r=32)` keep their ordering at constant scale,
   capacity is implicated; if the ordering collapses, effective step size was doing
   the work. Six runs. This is pre-registered here as the designated follow-up so that
   running it later is not seen as fishing, and **it is required before any
   capacity-language claim enters the paper.**

No criterion changes. W2/W3/W4 are computed exactly as frozen; only the words allowed
in the conclusion are constrained.

---

## Amendment 3 — §2's H-grow motivation attacked a strawman, and this sweep cannot test the real claim

Made 2026-09-01, **after** the sweep was scored and the outcome (`H_flat`) was known.
This changes no criterion and no result: the sentence corrected here lives only in the
motivation for `H-grow`, which did not occur. It is recorded because the error is a
framing error, which is worse than a wrong number.

**(a) "Raise rank if you are forgetting" is not standard advice.** §2's H-grow bullet
calls it that. It is not. The literature's response to forgetting is regularisation
(EWC, orthogonality constraints such as O-LoRA), replay, or parameter isolation and
expansion; rank-allocation methods (AdaLoRA, and the E2-LoRA line closest to us)
*redistribute* a fixed budget across modules rather than enlarge it. LoRA is also
widely observed to saturate by `r≈4–8`, and in continual settings a *small* rank is
often preferred precisely because a larger adapter overwrites more. So the honest
statement is the opposite of what §2 implies: **practitioners already do not expect
rank to fix forgetting, and often observe it hurting.** The paper's own §1 states this
correctly ("methods allocate rank more cleverly, penalise interference..."); the error
is confined to this protocol.

**What 2W therefore is.** Not a refutation of a prescription anyone holds. It is
(i) a **control** against the reviewer objection that `Δ_id` is an artifact of an
undersized adapter — answered by rank-invariance across 32× with `R_raw` flat too, so
neither undersized nor saturated; and (ii) evidence of a **dissociation** — a constant
buys ~2 pp (24/24 runs positive, p=1.2e-07) where 32× of rank buys nothing (paired CI
upper bound +0.83 pp). The dissociation is *consistent with*, not contrary to, the
common observation that rank does not help forgetting.

**(b) This sweep cannot test whether higher rank hurts.** Measured here: ρ(rank,
`R_raw`) = +0.54, ρ(rank, first-seen→final drop) = −0.37, both inside the 6-point
permutation null (sd 0.447) and non-monotone (worst drop is at `r=4`, +11.59 pp). But
the reason this is uninformative is structural, not statistical: with `lora_alpha`
fixed at 32, the effective update scale `alpha/r` **falls 32× as rank rises**. The
mechanism by which a larger adapter would overwrite more is a larger effective update
— exactly what this design suppresses. Amendment 2 used this confound to restrict
*capacity* language; it also **disqualifies this sweep from the "does more rank hurt"
question entirely**. That question needs the constant-scale (`alpha = 4r`) arm, and
even then a forgetting-specific design.
