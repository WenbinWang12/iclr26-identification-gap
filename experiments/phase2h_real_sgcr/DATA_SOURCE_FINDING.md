# Phase-2H data-source finding (2026-08-27)

## The protocol's named dataset is DEAD

`notes/phase2h_real_sgcr_protocol.md` specifies the **Multilingual Amazon Reviews
Corpus (MARC)**, English config, using the `product_category` field to define the
six hidden sources. This is no longer loadable:

- Amazon defunded MARC in 2023; `amazon_reviews_multi` (and `mteb/`,
  `defunct-datasets/` mirrors) are **script-based** and `datasets 5.0.1` refuses
  script loading ("Dataset scripts are no longer supported").
- The only live parquet mirror, `SetFit/amazon_reviews_multi_en`, has schema
  `{id, text, label, label_text}` — **`product_category` was dropped**. Without it
  the 6-hidden-source benchmark cannot be built.

Verified empirically (probe commands run 2026-08-27): SetFit mirror loads but has
no category field; mteb/defunct mirrors fail on the script-loader error.

## Chosen replacement: McAuley-Lab/Amazon-Reviews-2023 (per-category files)

`McAuley-Lab/Amazon-Reviews-2023` exposes **34 review categories as separate JSONL
files** under `raw/review_categories/<Category>.jsonl`, each row:
`{rating(1-5 float), title, text, asin, parent_asin, user_id, timestamp,
helpful_vote, verified_purchase}`. Loadable NOW via direct `hf://` parquet/json
(no script, no `trust_remote_code`). Verified: `Magazine_Subscriptions.jsonl`
loads with rating+title+text.

### Why this is a BETTER fit than the original MARC design, not a weaker one
- The hidden source = **which category file** a review comes from. Category is
  therefore *structurally* implicit — it never exists as a column the learner
  could read, so the protocol's "never concatenate product_category / keep source
  metadata outside every learner-facing record" constraint is satisfied by
  construction, not by remembering to strip a field.
- Polarity mapping is unchanged and applies directly to `rating`:
  `rating in {1,2} -> negative`, `{4,5} -> positive`, `rating==3 -> excluded`.
- Many more categories than 6 are available, so the count-only eligibility
  preflight (≥512/polarity in train, ≥32 in val/test) has ample choice.

## What this changes in the protocol (to be applied honestly, not silently)
- Dataset ID, revision, and per-file hashes: pin `McAuley-Lab/Amazon-Reviews-2023`
  + the exact `raw/review_categories/*.jsonl` files used, in the data manifest.
- The count-only category-eligibility rule (F0/D0) now ranks **categories = files**
  by min train-polarity count; the "first six" rule is unchanged.
- Fields: use `title + " [SEP] " + text` (protocol said review_title/review_body;
  McAuley calls them title/text). `review_id` -> synthesize an immutable
  `example_id` from a stable hash of (category, asin, user_id, timestamp) or a
  per-file running index pinned in the manifest.
- No train/validation/test official split exists per category; construct a
  deterministic hash-based split from immutable example_id + protocol salt
  (the role split the protocol already mandates), and carve val/test from it.
  This is a genuine deviation from "official validation is development-only" —
  there is no official split — and must be disclosed.

## STATUS: user APPROVED the swap (2026-08-27); prepare_marc.py written + validated
User approved McAuley-2023 as the Phase-2H dataset. `prepare_marc.py` is written
and validated end-to-end:
- The dead `datasets` script loader is bypassed by reading each
  `raw/review_categories/<Cat>.jsonl` directly over `hf://` fsspec (range-streamed,
  no full download). Confirmed fields: rating(float)/title/text/asin/user_id/timestamp.
- Pure-logic self-checks pass: polarity map {1,2}->0 / {4,5}->1 / 3 excluded;
  NFKC+whitespace normalization; immutable uint64 example_id (deterministic,
  content-derived, category-sensitive); deterministic 80/10/10 hash split.
- Live 3-category count-only scan (cap=8000) works: Magazine/Health ELIGIBLE,
  Gift_Cards not (281<512 neg) -> the "STOP if <6 qualify" honesty gate fires.
- Deviations pinned in the manifest: SCAN_CAP (conservative, never invents
  eligibility), synthesized example_id, self-constructed split (no official split
  exists). Eligibility RULE and first-six ranking are unchanged from the protocol.

NOT YET DONE: full 34-category manifest scan (the actual D0 artifact; ~network-heavy,
run with a larger cap so the 6th source clears 512 neg), then materialize fixed
tensors, then feasibility.py (F0-F3). No training run yet. Do not treat any of
this as run evidence.
