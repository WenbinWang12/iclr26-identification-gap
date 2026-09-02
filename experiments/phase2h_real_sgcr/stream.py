"""Phase-2H stream construction (deterministic, source-free at the learner).

STATUS: development scaffold.  Pure numpy + stdlib; imports NO torch and NO
transformers so it runs now and is unit-testable standalone.

This implements the "Stream construction and chronology" section of
notes/phase2h_real_sgcr_protocol.md exactly:

- 60 windows of 96 examples: 6 warmup windows + 54 drift windows;
- warmup window = 16 examples from EACH of the 6 sources, 8 per polarity;
- 54 drift windows form three consecutive 18-window phases A/B/C with fixed
  source-probability vectors;
- for drift window t, draw ``2 + (t mod 3)`` sources WITHOUT replacement under
  the phase probabilities, then distribute 96 records among the selected sources
  by a conditioned multinomial with at least two examples per selected source;
- labels are balanced within each realized source block, up to the deterministic
  one-example remainder for an odd count.

The generator uses source metadata (which category a record comes from) to build
the benchmark.  That metadata is NOT part of any learner-facing record; the
learner receives only tokens + polarity.  This module emits *index plans* --
(source_index, polarity, example_id) triples -- which prepare_marc.materialize
turns into token tensors.  The category string never travels with a record.

RNG DISCIPLINE (must be unit-tested and hashed, per protocol): a single seeded
``numpy.random.Generator`` (PCG64) is threaded through the whole construction and
consumed in a fixed documented order.  Every draw below is annotated with the
order in which it consumes the stream.  Changing the order changes the hash.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Frozen stream configuration (pin in the run manifest)
# --------------------------------------------------------------------------- #
N_SOURCES = 6
N_WARMUP_WINDOWS = 6
N_DRIFT_WINDOWS = 54
N_WINDOWS = N_WARMUP_WINDOWS + N_DRIFT_WINDOWS      # 60
WINDOW_SIZE = 96
WARMUP_PER_SOURCE = 16                              # 8 per polarity * 6 = 96
MIN_PER_SELECTED_SOURCE = 2

# Three consecutive 18-window phases.  Index 0..5 == ordered_sources manifest.
PHASE_PROBS = {
    "A": np.array([0.55, 0.22, 0.10, 0.07, 0.05, 0.01]),
    "B": np.array([0.10, 0.55, 0.22, 0.07, 0.05, 0.01]),
    "C": np.array([0.22, 0.10, 0.55, 0.07, 0.05, 0.01]),
}
PHASE_ORDER = ["A", "B", "C"]      # 18 windows each


# --------------------------------------------------------------------------- #
# Emitted plan records (index plans, NOT token tensors, NOT category strings)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlanRecord:
    """One scheduled example.  ``source_index`` is 0..5 into the ordered manifest
    and is used ONLY by the benchmark generator / report-only evaluator; it is
    never written into a learner-facing tensor record."""
    window: int
    source_index: int
    polarity: int          # 0 negative, 1 positive
    example_id: int        # immutable uint64 (from prepare_marc)


@dataclass
class WindowPlan:
    window: int
    phase: str             # "warmup" | "A" | "B" | "C"
    selected_sources: Tuple[int, ...]
    records: List[PlanRecord]

    def realized_source_counts(self) -> Dict[int, int]:
        d: Dict[int, int] = {}
        for r in self.records:
            d[r.source_index] = d.get(r.source_index, 0) + 1
        return d


# --------------------------------------------------------------------------- #
# Pools: available example_ids per (source, polarity), train-role only
# --------------------------------------------------------------------------- #
class SourcePools:
    """Draw-without-replacement pools of immutable example_ids, one per
    (source_index, polarity).  The stream consumes from these; a pool that runs
    dry raises rather than silently repeating an example.

    ``ids_by_source_polarity[s][p]`` is a list of uint64 example_ids belonging to
    source ``s`` (0..5) and polarity ``p`` (0/1), already restricted to the
    desired split+role by the caller (prepare_marc)."""

    def __init__(self, ids_by_source_polarity: Sequence[Sequence[Sequence[int]]],
                 rng: np.random.Generator):
        if len(ids_by_source_polarity) != N_SOURCES:
            raise ValueError(f"need {N_SOURCES} sources, got {len(ids_by_source_polarity)}")
        # Copy + shuffle each pool once, deterministically, then pop from the end.
        # RNG ORDER 0: shuffle pools, source-major then polarity-major.
        self._pools: List[List[List[int]]] = []
        for s in range(N_SOURCES):
            pols = []
            for p in (0, 1):
                arr = np.array(list(ids_by_source_polarity[s][p]), dtype=np.uint64)
                perm = rng.permutation(arr.shape[0])
                pols.append([int(x) for x in arr[perm]])
            self._pools.append(pols)

    def available(self, s: int, p: int) -> int:
        return len(self._pools[s][p])

    def draw(self, s: int, p: int, k: int) -> List[int]:
        pool = self._pools[s][p]
        if k > len(pool):
            raise RuntimeError(
                f"source {s} polarity {p} pool exhausted: need {k}, have {len(pool)}. "
                "Increase the materialized pool or reduce windows; do NOT repeat examples."
            )
        out = [pool.pop() for _ in range(k)]
        return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _balanced_polarity_counts(n: int, rng: np.random.Generator) -> Tuple[int, int]:
    """Split ``n`` into (neg, pos) as balanced as possible.  For odd n the single
    remainder example is assigned to a deterministic-random polarity (RNG ORDER
    per-block), per the protocol's "deterministic one-example remainder"."""
    half = n // 2
    if n % 2 == 0:
        return half, half
    # odd: one extra; choose which polarity gets it via one RNG draw.
    extra_pos = int(rng.integers(0, 2))
    return (half, half + 1) if extra_pos else (half + 1, half)


def _conditioned_multinomial(n: int, weights: np.ndarray, min_each: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Distribute ``n`` items among ``len(weights)`` bins with at least
    ``min_each`` per bin, proportional (in expectation) to ``weights``.

    Reserve ``min_each`` per bin first, then draw the remaining ``n - k*min_each``
    from a single multinomial with the renormalized weights.  RNG ORDER: one
    multinomial draw per window after source selection."""
    k = weights.shape[0]
    base = np.full(k, min_each, dtype=np.int64)
    remaining = int(n - k * min_each)
    if remaining < 0:
        raise ValueError(f"cannot place {min_each} in each of {k} bins from {n} items")
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    if remaining > 0:
        extra = rng.multinomial(remaining, w)
        base = base + extra
    return base


def _select_sources_without_replacement(probs: np.ndarray, m: int,
                                         rng: np.random.Generator) -> Tuple[int, ...]:
    """Choose ``m`` distinct source indices under ``probs`` without replacement.
    RNG ORDER: one weighted choice per window (before the multinomial split).
    Result is returned in ascending index order for a deterministic downstream
    layout (the *selection* is random; the *ordering* is canonical)."""
    idx = rng.choice(N_SOURCES, size=m, replace=False, p=probs)
    return tuple(int(i) for i in sorted(idx.tolist()))


# --------------------------------------------------------------------------- #
# Window builders
# --------------------------------------------------------------------------- #
def _build_warmup_window(w: int, pools: SourcePools,
                         rng: np.random.Generator) -> WindowPlan:
    """16 examples from EACH source, 8 per polarity (protocol).  No randomness in
    the counts; only the pool draw (already shuffled) picks which example_ids."""
    records: List[PlanRecord] = []
    for s in range(N_SOURCES):
        for p in (0, 1):
            for eid in pools.draw(s, p, WARMUP_PER_SOURCE // 2):
                records.append(PlanRecord(window=w, source_index=s, polarity=p,
                                          example_id=eid))
    return WindowPlan(window=w, phase="warmup",
                      selected_sources=tuple(range(N_SOURCES)), records=records)


def _build_drift_window(w: int, drift_index: int, pools: SourcePools,
                        rng: np.random.Generator) -> WindowPlan:
    phase = PHASE_ORDER[drift_index // 18]        # 0..17 -> A, 18..35 -> B, 36..53 -> C
    probs = PHASE_PROBS[phase]
    m = 2 + (w % 3)                               # 2..4 sources this window
    selected = _select_sources_without_replacement(probs, m, rng)

    # distribute 96 records among selected sources proportional to their phase
    # probability, >= 2 each.
    sel_w = probs[list(selected)]
    counts = _conditioned_multinomial(WINDOW_SIZE, sel_w, MIN_PER_SELECTED_SOURCE, rng)

    records: List[PlanRecord] = []
    for s, n_s in zip(selected, counts):
        neg, pos = _balanced_polarity_counts(int(n_s), rng)
        for p, k in ((0, neg), (1, pos)):
            for eid in pools.draw(s, p, k):
                records.append(PlanRecord(window=w, source_index=int(s),
                                          polarity=p, example_id=eid))
    return WindowPlan(window=w, phase=phase, selected_sources=selected,
                      records=records)


# --------------------------------------------------------------------------- #
# Top-level construction
# --------------------------------------------------------------------------- #
def build_stream(pools: SourcePools, seed: int,
                 n_windows: int = N_WINDOWS) -> List[WindowPlan]:
    """Build the first ``n_windows`` window plans from a single seeded Generator.

    The RNG is consumed strictly in window order: pool shuffle (in SourcePools
    __init__, before this call, from the SAME seed sequence -- see build_from_ids)
    then per drift window a source selection then a multinomial then per-source
    odd-remainder draws.  Warmup windows consume only pool draws (no counts RNG).

    Because the RNG is consumed strictly in window order, building only the first
    ``n_windows`` windows yields a schedule byte-identical to the corresponding
    prefix of the full 60-window stream -- the D1 smoke uses this to require only
    the 20-window data budget, not the full 60-window one.  ``n_windows`` must not
    exceed N_WINDOWS (60); the full-stream protocol object is always 60 windows.
    """
    if not (1 <= n_windows <= N_WINDOWS):
        raise ValueError(f"n_windows must be in [1, {N_WINDOWS}], got {n_windows}")
    rng = pools_rng_is_separate_guard(seed)  # documents intent; see build_from_ids
    plans: List[WindowPlan] = []
    for w in range(n_windows):
        if w < N_WARMUP_WINDOWS:
            plans.append(_build_warmup_window(w, pools, rng))
        else:
            plans.append(_build_drift_window(w, w - N_WARMUP_WINDOWS, pools, rng))
    return plans


def pools_rng_is_separate_guard(seed: int) -> np.random.Generator:
    """The stream RNG is a distinct Generator from the pool-shuffle RNG so that
    changing pool contents does not shift the window-schedule draws.  Both are
    derived deterministically from the master seed via SeedSequence.spawn."""
    ss = np.random.SeedSequence(seed)
    _pool_ss, stream_ss = ss.spawn(2)
    return np.random.Generator(np.random.PCG64(stream_ss))


def build_from_ids(ids_by_source_polarity: Sequence[Sequence[Sequence[int]]],
                   seed: int, n_windows: int = N_WINDOWS) -> List[WindowPlan]:
    """Convenience: derive the pool-shuffle RNG and the stream RNG from ONE master
    seed via SeedSequence.spawn (documented, reproducible), build pools, then the
    stream.  This is the canonical entry point used by run_development.

    ``n_windows`` builds only a leading prefix (see build_stream); the D1 smoke
    passes 20 so it needs only the 20-window data budget.  The default (60) is the
    full protocol stream."""
    ss = np.random.SeedSequence(seed)
    pool_ss, _stream_ss = ss.spawn(2)
    pool_rng = np.random.Generator(np.random.PCG64(pool_ss))
    pools = SourcePools(ids_by_source_polarity, pool_rng)
    return build_stream(pools, seed, n_windows=n_windows)


# --------------------------------------------------------------------------- #
# Hashing / reporting
# --------------------------------------------------------------------------- #
def stream_fingerprint(plans: Sequence[WindowPlan]) -> str:
    """Stable SHA-256 over the emitted schedule (window, source, polarity,
    example_id in order).  Two runs with the same seed + pools must match; this
    is the hash the protocol requires for the RNG consumption order."""
    h = hashlib.sha256()
    for wp in plans:
        h.update(f"W{wp.window}:{wp.phase}:{wp.selected_sources}".encode())
        for r in wp.records:
            h.update(f"{r.window},{r.source_index},{r.polarity},{r.example_id};".encode())
    return h.hexdigest()


def realized_source_matrix(plans: Sequence[WindowPlan]) -> np.ndarray:
    """(n_windows, n_sources) realized counts.  Report-only; do not repair a seed
    because a rare-source count is inconvenient (protocol)."""
    mat = np.zeros((len(plans), N_SOURCES), dtype=np.int64)
    for wp in plans:
        for s, c in wp.realized_source_counts().items():
            mat[wp.window, s] = c
    return mat
