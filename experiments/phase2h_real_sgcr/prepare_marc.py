"""Phase-2H Stage D0 count-only data preparation (source = which category file).

STATUS: development scaffold.  This module builds the *count-only* category
manifest required by notes/phase2h_real_sgcr_protocol.md before any model
forward pass, and (separately) can materialize fixed canonical tensors for the
selected categories.  No training happens here.

DATASET DEVIATION (disclosed, not silent) -- see DATA_SOURCE_FINDING.md:
The protocol names the Multilingual Amazon Reviews Corpus (MARC,
`amazon_reviews_multi`).  That dataset is DEAD: Amazon defunded it in 2023 and
`datasets 5.0.1` refuses its script loader.  We substitute
`McAuley-Lab/Amazon-Reviews-2023`, whose 34 review categories are separate
`raw/review_categories/<Category>.jsonl` files read directly over `hf://`
(no dataset script, no trust_remote_code).  Consequences, all disclosed:

  * The hidden source == WHICH category file a review came from.  Category is
    therefore structurally implicit: it is never a field a learner-facing record
    could read.  The protocol's "never concatenate product_category / keep
    source metadata outside every learner-facing record" is satisfied by
    construction.  A report-only evaluator table (keyed by example_id) is the
    ONLY place the category string is retained.
  * Fields: McAuley calls them `title`/`text` (protocol said
    `review_title`/`review_body`); `rating` (float 1-5) replaces `stars`.
    Polarity map is unchanged: {1,2}->negative(0), {4,5}->positive(1), 3 excluded.
  * There is NO official train/validation/test split per category.  We construct
    a deterministic hash split (train/val/test) from an immutable example_id and
    a fixed split salt.  This is a genuine deviation from "official validation is
    development-only"; there is no official split.  The 50/50 train/audit ROLE
    split (buffers.role_of) is a separate, second hash and is unchanged.
  * `example_id` is not provided; we synthesize an immutable uint64 from a stable
    SHA-256 of (category, asin, parent_asin, user_id, timestamp, normalized_text).
  * The count-only scan is CAPPED at SCAN_CAP rows per category so ranking is
    cheap and reproducible over a range-streamed prefix rather than a full
    multi-GB download.  A category that would qualify only beyond the cap is
    conservatively treated as not-yet-eligible; categories whose train-polarity
    count reaches the cap tie at the cap and are ordered by the protocol's
    category-string tie break.  This bias is conservative (never invents
    eligibility) and disclosed.

The eligibility RULE itself is unchanged from the protocol: retain categories
with >=512 examples of EACH polarity in train and >=32 of each polarity in BOTH
validation and test; rank eligible categories by their minimum train-polarity
count descending, category string as the deterministic tie break; take the
first six; write the ordered list to an immutable manifest.  Categories are
never replaced because their downstream results look unfavorable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Frozen configuration (pin these in the data manifest)
# --------------------------------------------------------------------------- #
DATASET_ID = "McAuley-Lab/Amazon-Reviews-2023"
CATEGORY_DIR = "raw/review_categories"
HF_PREFIX = f"hf://datasets/{DATASET_ID}/{CATEGORY_DIR}"

# Deterministic salts.  These are fixed protocol constants; changing them
# reshuffles the split/id assignment and MUST be recorded in the manifest.
SPLIT_SALT = "phase2h::split::v1"
EXAMPLE_ID_SALT = "phase2h::example_id::v1"

# train/val/test proportions (no official split exists for McAuley).
SPLIT_FRACS = {"train": 0.80, "validation": 0.10, "test": 0.10}

# Eligibility thresholds (protocol, unchanged).
MIN_TRAIN_PER_POLARITY = 512
MIN_EVAL_PER_POLARITY = 32          # applies to BOTH validation and test
N_SOURCES = 6

# Conservative streaming cap per category (disclosed deviation).
SCAN_CAP = 40_000

MAX_SEQ_LEN = 64
TOKENIZER_ID = "prajjwal1/bert-mini"   # 2026-08-27 predeclared fallback from bert-tiny
SEP = " [SEP] "

_POS = {4, 5}
_NEG = {1, 2}
# rating == 3 is excluded.


# --------------------------------------------------------------------------- #
# Text normalization + immutable ids (shared by count-only scan and materialize)
# --------------------------------------------------------------------------- #
def normalize_text(s: Optional[str]) -> str:
    """NFKC-normalize Unicode and collapse all whitespace runs to single spaces,
    then strip.  Used both for dedup hashing and (via join) for tokenization, so
    that "disjoint normalized text across splits" is well defined."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    return " ".join(s.split())


def joined_text(title: Optional[str], text: Optional[str]) -> str:
    """Learner-facing string: normalized title [SEP] normalized body."""
    return normalize_text(title) + SEP + normalize_text(text)


def polarity_of(rating) -> Optional[int]:
    """1 for positive, 0 for negative, None if excluded/unmappable."""
    try:
        r = int(round(float(rating)))
    except (TypeError, ValueError):
        return None
    if r in _POS:
        return 1
    if r in _NEG:
        return 0
    return None  # rating == 3 or out of range


def _digest_u64(salt: str, payload: str) -> int:
    d = hashlib.sha256(f"{salt}:{payload}".encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big", signed=False)


def example_id_of(category: str, row: Dict, norm_text: str) -> int:
    """Immutable uint64 example id.  Stable across processes and runs: derived
    from content + provenance, NOT from a per-process counter, so the split/role
    hashes are reproducible."""
    payload = "\x1f".join([
        category,
        str(row.get("asin", "")),
        str(row.get("parent_asin", "")),
        str(row.get("user_id", "")),
        str(row.get("timestamp", "")),
        norm_text,
    ])
    return _digest_u64(EXAMPLE_ID_SALT, payload)


def split_of(example_id: int) -> str:
    """Deterministic train/val/test split from example_id + SPLIT_SALT.
    Uses a fixed 10000-bucket hash so the fractions are exact and reproducible."""
    bucket = _digest_u64(SPLIT_SALT, str(int(example_id))) % 10_000
    tr = int(SPLIT_FRACS["train"] * 10_000)          # 8000
    va = int(SPLIT_FRACS["validation"] * 10_000)     # 1000
    if bucket < tr:
        return "train"
    if bucket < tr + va:
        return "validation"
    return "test"


# --------------------------------------------------------------------------- #
# Count-only scan
# --------------------------------------------------------------------------- #
@dataclass
class CatCounts:
    category: str
    scanned: int = 0
    excluded_rating: int = 0
    # counts[split][polarity] -> int
    counts: Dict[str, List[int]] = field(
        default_factory=lambda: {s: [0, 0] for s in SPLIT_FRACS}
    )
    hit_cap: bool = False

    def min_train_polarity(self) -> int:
        neg, pos = self.counts["train"]
        return min(neg, pos)

    def eligible(self) -> bool:
        tr = self.counts["train"]
        va = self.counts["validation"]
        te = self.counts["test"]
        return (
            min(tr) >= MIN_TRAIN_PER_POLARITY
            and min(va) >= MIN_EVAL_PER_POLARITY
            and min(te) >= MIN_EVAL_PER_POLARITY
        )


def list_category_files() -> List[str]:
    """The 34 category basenames (without extension), sorted deterministically."""
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(DATASET_ID, repo_type="dataset")
    cats = sorted(
        f.split("/")[-1][: -len(".jsonl")]
        for f in files
        if f.startswith(CATEGORY_DIR + "/") and f.endswith(".jsonl")
    )
    if not cats:
        raise RuntimeError("no per-category jsonl files found in the repository")
    return cats


def _open_category(category: str):
    import fsspec
    path = f"{HF_PREFIX}/{category}.jsonl"
    return fsspec.open(path, "r", encoding="utf-8")


def scan_category(category: str, cap: int = SCAN_CAP,
                  seen_text: Optional[Dict[str, Tuple[str, int]]] = None,
                  dup_log: Optional[Dict[str, int]] = None) -> CatCounts:
    """Stream up to `cap` rows of one category, applying the polarity filter and
    the deterministic split.  Counts positive/negative per split.

    Cross-category dedup + polarity-conflict detection use the optional shared
    `seen_text` map {normalized_text_hash: (category, polarity)}; duplicates and
    polarity conflicts are removed (not counted toward eligibility) and tallied
    in `dup_log`.  Within-cap only -- disclosed."""
    cc = CatCounts(category=category)
    with _open_category(category) as f:
        for line in f:
            if cc.scanned >= cap:
                cc.hit_cap = True
                break
            cc.scanned += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pol = polarity_of(row.get("rating"))
            if pol is None:
                cc.excluded_rating += 1
                continue
            nt = joined_text(row.get("title"), row.get("text"))
            if not nt.strip(SEP.strip()).strip():
                continue  # empty after normalization
            th = hashlib.sha256(nt.encode("utf-8")).hexdigest()
            if seen_text is not None:
                prev = seen_text.get(th)
                if prev is not None:
                    prev_cat, prev_pol = prev
                    if dup_log is not None:
                        if prev_pol != pol:
                            dup_log["polarity_conflict"] += 1
                        elif prev_cat != category:
                            dup_log["cross_category_dup"] += 1
                        else:
                            dup_log["within_category_dup"] += 1
                    continue  # drop the duplicate / conflict
                seen_text[th] = (category, pol)
            eid = example_id_of(category, row, nt)
            sp = split_of(eid)
            cc.counts[sp][pol] += 1
    return cc


def build_manifest(cap: int = SCAN_CAP, categories: Optional[List[str]] = None) -> Dict:
    """Count-only scan of every category -> eligibility -> ranked first-six
    manifest.  Returns a JSON-serializable dict; does NOT tokenize or train."""
    cats = categories if categories is not None else list_category_files()
    seen_text: Dict[str, Tuple[str, int]] = {}
    dup_log: Dict[str, int] = defaultdict(int)

    per_cat: List[CatCounts] = []
    for cat in cats:
        cc = scan_category(cat, cap=cap, seen_text=seen_text, dup_log=dup_log)
        per_cat.append(cc)
        print(f"  scanned {cat:32s} n={cc.scanned:6d} "
              f"train(neg,pos)={tuple(cc.counts['train'])} "
              f"val={tuple(cc.counts['validation'])} test={tuple(cc.counts['test'])} "
              f"{'[cap]' if cc.hit_cap else ''} {'ELIGIBLE' if cc.eligible() else ''}",
              file=sys.stderr)

    eligible = [cc for cc in per_cat if cc.eligible()]
    # rank by min train-polarity count DESC, category string as deterministic
    # tie break (ascending).
    eligible.sort(key=lambda cc: (-cc.min_train_polarity(), cc.category))
    chosen = eligible[:N_SOURCES]

    manifest = {
        "dataset_id": DATASET_ID,
        "category_dir": CATEGORY_DIR,
        "split_salt": SPLIT_SALT,
        "example_id_salt": EXAMPLE_ID_SALT,
        "split_fracs": SPLIT_FRACS,
        "eligibility": {
            "min_train_per_polarity": MIN_TRAIN_PER_POLARITY,
            "min_eval_per_polarity": MIN_EVAL_PER_POLARITY,
            "n_sources": N_SOURCES,
            "scan_cap": cap,
        },
        "polarity_map": {"positive": sorted(_POS), "negative": sorted(_NEG),
                         "excluded": [3]},
        "dedup": dict(dup_log),
        "n_categories_scanned": len(per_cat),
        "n_eligible": len(eligible),
        "enough_sources": len(eligible) >= N_SOURCES,
        "per_category": [
            {
                "category": cc.category,
                "scanned": cc.scanned,
                "hit_cap": cc.hit_cap,
                "train": cc.counts["train"],
                "validation": cc.counts["validation"],
                "test": cc.counts["test"],
                "min_train_polarity": cc.min_train_polarity(),
                "eligible": cc.eligible(),
            }
            for cc in per_cat
        ],
        # the ordered source list -> indices 0..5 for the stream construction
        "ordered_sources": [cc.category for cc in chosen],
        "ordered_source_min_train_polarity": [cc.min_train_polarity() for cc in chosen],
    }
    return manifest


# --------------------------------------------------------------------------- #
# Materialization: selected sources -> fixed canonical tensors + report-only
# source table.  Runs AFTER the count-only manifest fixes ordered_sources.
# --------------------------------------------------------------------------- #
# Learner-facing canonical record schema (protocol "Data and split
# construction").  There is deliberately NO category/source key here.
_LEARNER_KEYS = ("input_ids", "attention_mask", "label", "example_id")
_FORBIDDEN_IN_LEARNER = ("source", "category", "product_category", "source_index")


def _get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


def _encode(tokenizer, title, text):
    """title [SEP] text -> int32[64] input_ids, uint8[64] attention_mask.
    The join uses the literal ' [SEP] ' string (protocol); the tokenizer maps it
    to its real SEP id, which is intended."""
    enc = tokenizer(
        joined_text(title, text),
        truncation=True, max_length=MAX_SEQ_LEN,
        padding="max_length", return_tensors="np",
    )
    input_ids = enc["input_ids"][0].astype("int32")
    attention_mask = enc["attention_mask"][0].astype("uint8")
    return input_ids, attention_mask


def materialize(manifest: Dict, cap: int = SCAN_CAP, tokenizer=None) -> Dict:
    """Tokenize the six ordered sources into fixed canonical tensors.

    Returns a dict with:
      * ``records``: list of learner-facing dicts (NO category key), each
        {input_ids int32[64], attention_mask uint8[64], label int8, example_id
        uint64, split, polarity}.  ``split`` and ``polarity`` are metadata used
        by the harness to place the record; they are stripped before a record
        enters a replay store / model call (buffers.py enforces the schema).
      * ``source_table``: report-only {example_id -> source_index}, the ONLY
        place source identity is retained.  Rejected by learner-facing schema
        tests.
      * ``by_split_source_polarity``: nested lists of example_ids for stream
        construction: [split][source_index][polarity] -> [example_id...].

    Cross-source dedup + polarity-conflict removal is redone here (the count-only
    scan's dedup map is not persisted) so the materialized set is internally
    de-duplicated in the SAME deterministic manifest order."""
    ordered = manifest["ordered_sources"]
    if len(ordered) != N_SOURCES:
        raise ValueError(f"manifest has {len(ordered)} ordered sources, need {N_SOURCES}")
    if tokenizer is None:
        tokenizer = _get_tokenizer()

    records: List[Dict] = []
    source_table: Dict[int, int] = {}          # example_id -> source_index
    seen_text: Dict[str, Tuple[int, int]] = {}  # text_hash -> (source_index, polarity)
    dup_log: Dict[str, int] = defaultdict(int)
    id_collision = 0

    # by_split_source_polarity[split][s][p] -> list of example_ids
    bssp: Dict[str, List[List[List[int]]]] = {
        sp: [[[], []] for _ in range(N_SOURCES)] for sp in SPLIT_FRACS
    }

    for s, category in enumerate(ordered):
        n_kept = 0
        with _open_category(category) as f:
            for i, line in enumerate(f):
                if i >= cap:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pol = polarity_of(row.get("rating"))
                if pol is None:
                    continue
                nt = joined_text(row.get("title"), row.get("text"))
                if not nt.strip(SEP.strip()).strip():
                    continue
                th = hashlib.sha256(nt.encode("utf-8")).hexdigest()
                prev = seen_text.get(th)
                if prev is not None:
                    prev_s, prev_p = prev
                    if prev_p != pol:
                        dup_log["polarity_conflict"] += 1
                    elif prev_s != s:
                        dup_log["cross_source_dup"] += 1
                    else:
                        dup_log["within_source_dup"] += 1
                    continue
                seen_text[th] = (s, pol)

                eid = example_id_of(category, row, nt)
                if eid in source_table:
                    # astronomically unlikely 64-bit collision; skip + count.
                    id_collision += 1
                    continue
                sp = split_of(eid)
                input_ids, attention_mask = _encode(tokenizer, row.get("title"),
                                                     row.get("text"))
                records.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "label": np.int8(pol),
                    "example_id": np.uint64(eid),
                    "split": sp,
                    "polarity": pol,
                })
                source_table[eid] = s                 # report-only
                bssp[sp][s][pol].append(eid)
                n_kept += 1
        print(f"  materialized {category:32s} kept={n_kept}", file=sys.stderr)

    return {
        "records": records,
        "source_table": source_table,
        "by_split_source_polarity": bssp,
        "dedup": dict(dup_log),
        "id_collision": id_collision,
        "n_records": len(records),
    }


def assert_no_source_leak(record: Dict) -> None:
    """Raise if a learner-facing record carries any forbidden source key.  Used
    by tests and by the harness before a record enters a store / model call."""
    for k in _FORBIDDEN_IN_LEARNER:
        if k in record:
            raise ValueError(f"forbidden source key '{k}' in learner record")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-2H count-only category manifest")
    ap.add_argument("--cap", type=int, default=SCAN_CAP,
                    help="max rows streamed per category (disclosed conservative cap)")
    ap.add_argument("--out", type=str, default="data_manifest.json",
                    help="where to write the immutable count-only manifest")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated category subset (debug only; not for the manifest)")
    args = ap.parse_args()

    cats = args.only.split(",") if args.only else None
    print(f"Phase-2H count-only scan of {DATASET_ID} (cap={args.cap}/category)",
          file=sys.stderr)
    manifest = build_manifest(cap=args.cap, categories=cats)

    if not manifest["enough_sources"]:
        print(f"STOP: only {manifest['n_eligible']} categories qualify "
              f"(< {N_SOURCES}); revise the protocol before any model outcome.",
              file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.out}: ordered_sources={manifest['ordered_sources']} "
          f"(eligible={manifest['n_eligible']}/{manifest['n_categories_scanned']})",
          file=sys.stderr)
    return 0 if manifest["enough_sources"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
