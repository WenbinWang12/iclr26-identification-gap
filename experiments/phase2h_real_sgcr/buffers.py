"""Phase-2H minimum-plus-shared historical stores (source-free).

STATUS: development scaffold.  Pure numpy + stdlib; imports NO torch and NO
transformers so it runs now on the CPU-only environment with no network.

This implements the "Minimum-plus-shared historical stores" and "Fixed
historical-byte budget" sections of notes/phase2h_real_sgcr_protocol.md:

- A permanent train/audit role split from a hash of an immutable integer
  example_id plus a fixed protocol salt (50/50), shared across arms.
- ``MinPlusSharedStore``: fixed record capacity, K cells, a guaranteed minimum
  of ``min_per_cell`` priority-reservoir slots per cell, and one global shared
  priority overflow for every remaining slot.  Unused cell minimums return to
  the shared overflow rather than wasting fixed capacity.
- ``capacity_from_envelope``: derive how many records fit per store from actual
  dtype ``element_size * numel`` accounting, per the protocol deduction rules.

Learner-facing records carry ONLY the canonical schema keys.  There is
deliberately no 'source' / 'category' / 'product_category' field anywhere; the
byte accounting and tests enforce this.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List

import numpy as np

# Canonical replay-record schema (protocol "Data and split construction").
# Scalar fields cost a fixed number of bytes regardless of sequence length L.
#   label       int8    -> 1
#   priority    float64 -> 8
#   example_id  uint64  -> 8
# The array fields (input_ids int32[L], attention_mask uint8[L]) are charged
# from their actual numpy nbytes.  The SGCR cell label (cell_id uint8) adds one
# byte per retained record (protocol "Fixed historical-byte budget").
_SCALAR_BYTES = 1 + 8 + 8          # label + priority + example_id
_CELL_LABEL_BYTES = 1              # cell_id uint8, deducted per retained record

_FORBIDDEN_KEYS = ("source", "category", "product_category")

# Fixed controller state charged against the historical envelope.
# 2026-08-27: 128 -> 256 after the predeclared bert-mini fallback (hidden=256).
# The 86,272-byte envelope is unchanged; only the per-center/warmup-mean cost
# grows.  Verified cell capacity stays >= 6/cell (113 records/store at k=8).
_SIGNATURE_DIM = 256               # bert-mini hidden size -> 256-d signature
_FLOAT32 = 4


# --------------------------------------------------------------------------- #
# Permanent train/audit role split
# --------------------------------------------------------------------------- #
def role_of(example_id: int, salt) -> str:
    """Permanent 50/50 train/audit role from an immutable integer example_id and
    a fixed protocol salt.  Deterministic across processes and arms (uses
    hashlib, not Python's per-process-salted ``hash``)."""
    digest = hashlib.sha256(f"{salt}:{int(example_id)}".encode("utf-8")).digest()
    return "train" if (digest[0] & 1) == 0 else "audit"


# --------------------------------------------------------------------------- #
# Pure-numpy priority reservoir helper
# --------------------------------------------------------------------------- #
def priority_topk_mask(priorities, k: int) -> np.ndarray:
    """Boolean mask selecting the ``k`` highest-priority items.

    This is the priority-reservoir primitive: it keeps the highest-priority
    records and drops the rest.  Ties break deterministically toward the earlier
    index (stable sort), so the outcome is reproducible.
    """
    prio = np.asarray(priorities, dtype=np.float64)
    n = prio.shape[0]
    if k <= 0:
        return np.zeros(n, dtype=bool)
    if k >= n:
        return np.ones(n, dtype=bool)
    order = np.argsort(-prio, kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


# --------------------------------------------------------------------------- #
# Byte-budget deduction
# --------------------------------------------------------------------------- #
def _per_record_bytes(L: int) -> int:
    """Exact bytes of one retained SGCR cell-store record of sequence length L."""
    input_ids = L * np.dtype(np.int32).itemsize      # 4L
    attention_mask = L * np.dtype(np.uint8).itemsize  # L
    return int(input_ids + attention_mask + _SCALAR_BYTES + _CELL_LABEL_BYTES)


def history_envelope_bytes(L: int, n_records: int = 256) -> int:
    """Raw historical-storage envelope: ``n_records`` canonical *non-cell*
    records of length L.  For L=64, n_records=256 this is 86,272 bytes, matching
    the protocol's stated primary envelope."""
    schema = L * np.dtype(np.int32).itemsize + L * np.dtype(np.uint8).itemsize + _SCALAR_BYTES
    return int(n_records * schema)


def capacity_from_envelope(
    M_history_bytes: int,
    L: int,
    k_cells: int,
    extra_controller_bytes: int = 0,
) -> int:
    """Records that fit **per store** under the fixed historical envelope.

    Deductions (protocol "Fixed historical-byte budget"):
      * one uint8 cell label per retained record (charged inside per-record cost);
      * ``k_cells`` float32[_SIGNATURE_DIM] cluster centers;
      * one float32[_SIGNATURE_DIM] warmup mean;
      * train arrival counts + occupancy mask + persistent counters; and
      * ``extra_controller_bytes`` for any other persistent controller state.

    The remaining capacity is split 50/50 between the train and audit stores.
    Raises if fewer than six records per cell fit in a store.
    """
    per_record = _per_record_bytes(L)

    centers = k_cells * _SIGNATURE_DIM * _FLOAT32           # 8 * 256 * 4 = 8192 for k=8
    warmup_mean = _SIGNATURE_DIM * _FLOAT32                 # 1024
    # Persistent counters: train arrival count (int64) + occupancy mask (uint8)
    # per cell.  (per-record priority/example_id are already charged above.)
    counters = k_cells * np.dtype(np.int64).itemsize + k_cells * np.dtype(np.uint8).itemsize
    fixed = int(centers + warmup_mean + counters + int(extra_controller_bytes))

    available = int(M_history_bytes) - fixed
    if available <= 0:
        raise ValueError(
            f"controller state ({fixed} B) exceeds the {M_history_bytes} B envelope; "
            "stop rather than silently increase the budget."
        )
    per_store = available // 2
    capacity = int(per_store // per_record)

    if capacity < 6 * k_cells:
        raise ValueError(
            f"only {capacity} records/store fit but 6*{k_cells}={6 * k_cells} are "
            "required (six per cell); stop rather than silently increase the budget."
        )
    return capacity


# --------------------------------------------------------------------------- #
# Minimum-plus-shared store
# --------------------------------------------------------------------------- #
class MinPlusSharedStore:
    """Fixed-capacity priority store with K cells.

    Each cell is guaranteed at least ``min_per_cell`` priority-reservoir slots
    (its highest-priority records are protected from eviction).  Every remaining
    slot is a single global shared priority overflow contested across all cells.
    A cell that holds fewer than ``min_per_cell`` records leaves its unused
    minimum slots to the shared overflow instead of wasting fixed capacity.
    """

    _REQUIRED_KEYS = ("input_ids", "attention_mask", "label", "priority",
                      "example_id", "cell_id")

    def __init__(self, capacity: int, k_cells: int, min_per_cell: int = 6):
        if capacity < min_per_cell * k_cells:
            raise ValueError(
                f"capacity {capacity} < min_per_cell*k_cells "
                f"({min_per_cell}*{k_cells}); the per-cell minimum cannot be honored."
            )
        self.capacity = int(capacity)
        self.k_cells = int(k_cells)
        self.min_per_cell = int(min_per_cell)
        self._records: List[Dict] = []

    # -- construction / validation ----------------------------------------- #
    def _normalize(self, record: Dict) -> Dict:
        for key in record:
            if key in _FORBIDDEN_KEYS:
                raise ValueError(f"forbidden source/category key '{key}' in record")
        for key in self._REQUIRED_KEYS:
            if key not in record:
                raise KeyError(f"record missing required key '{key}'")
        cell = int(record["cell_id"])
        if not (0 <= cell < self.k_cells):
            raise ValueError(f"cell_id {cell} out of range [0,{self.k_cells})")
        # Coerce to the canonical dtypes so byte accounting is exact.
        return {
            "input_ids": np.asarray(record["input_ids"], dtype=np.int32),
            "attention_mask": np.asarray(record["attention_mask"], dtype=np.uint8),
            "label": np.int8(record["label"]),
            "priority": float(record["priority"]),
            "example_id": np.uint64(record["example_id"]),
            "cell_id": np.uint8(cell),
        }

    def _protected_mask(self) -> np.ndarray:
        """Mask of records protected by their cell's per-cell minimum."""
        n = len(self._records)
        protected = np.zeros(n, dtype=bool)
        if n == 0:
            return protected
        prio = np.array([r["priority"] for r in self._records], dtype=np.float64)
        cell = np.array([int(r["cell_id"]) for r in self._records], dtype=np.int64)
        for c in range(self.k_cells):
            idx = np.where(cell == c)[0]
            if idx.size == 0:
                continue
            m = min(self.min_per_cell, idx.size)
            keep = priority_topk_mask(prio[idx], m)
            protected[idx[keep]] = True
        return protected

    # -- public API --------------------------------------------------------- #
    def offer(self, record: Dict) -> None:
        """Offer a record.  Retained iff it beats an evictable (unprotected)
        record on priority, honoring every cell's minimum."""
        self._records.append(self._normalize(record))
        while len(self._records) > self.capacity:
            protected = self._protected_mask()
            unprotected = np.where(~protected)[0]
            # protected <= min_per_cell*k_cells <= capacity < len -> always some
            # unprotected record exists to evict.
            prio = np.array([self._records[i]["priority"] for i in unprotected],
                            dtype=np.float64)
            victim = int(unprotected[int(np.argmin(prio))])
            self._records.pop(victim)

    def sample(self, n: int, rng: np.random.Generator) -> List[Dict]:
        """Draw ``n`` records.  Without replacement when possible, else with."""
        m = len(self._records)
        if m == 0 or n <= 0:
            return []
        replace = n > m
        idx = rng.choice(m, size=n, replace=replace)
        return [self._records[int(i)] for i in idx]

    def bytes_used(self) -> int:
        """Exact retained bytes from dtype element_size*numel per the schema
        (including the per-record uint8 cell label)."""
        total = 0
        for r in self._records:
            total += int(r["input_ids"].nbytes)
            total += int(r["attention_mask"].nbytes)
            total += _SCALAR_BYTES + _CELL_LABEL_BYTES
        return total

    def __len__(self) -> int:
        return len(self._records)
