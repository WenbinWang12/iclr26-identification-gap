"""Phase-2H baseline arms (buffer policy + replay sampler + step budget).

STATUS: development scaffold.  Each arm is a small object exposing a uniform
interface the run_development driver calls:

    arm.offer(records)          # after a window is deployed (chronology-safe)
    arm.sample_replay(n, rng)   # draw n historical-train replay records
    arm.n_deploy_steps          # deployed AdamW steps per window
    arm.name

The buffers themselves are byte-budgeted (buffers.py / this file) and source-free
for every deployable arm.  The true-category oracle is the ONLY arm that stores a
source label, and it is development-only; its records still never reach a model
input through a learner-facing schema (the source is used only to pick the group
weighting), and run_development routes it through a separate evaluator interface.

Required arms (notes/phase2h_real_sgcr_protocol.md "Baselines and resource
matching"):

1. Sequential LoRA          -- no history; diagnostic forgetting arm.
2. Global ER-step           -- global uniform reservoir, 12 steps.
3. Global ER-FLOP           -- global uniform reservoir, 16 steps (main comparator).
4. Stable-cell q0-step/FLOP -- SGCR cells+store, all sampling from q0; 12 / 16 steps.
5. Persistent-loss CVaR/JTT -- global buffer + per-record float32 loss EMA; 16 steps.
6. True-category oracle      -- dev-only source-labelled min storage; 16 steps.
7. Offline-joint rank-4      -- report-only capacity/optimization upper bound.

SGCR itself is not here: it uses the same cell store (buffers.MinPlusSharedStore)
plus the branch/gate logic in sgcr.py, orchestrated by run_development.

STEP BUDGET NOTE: the deployed-step counts (12 vs 16) isolate coverage from the
extra branch compute -- SGCR runs 8 common + 4 (A) or 4 (B) = up to 16 update
computations, so ER-FLOP / q0-FLOP get 16 to remove "SGCR won only because it
computed two branches."  ER-step / q0-step get 12 as the lower-budget reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from buffers import MinPlusSharedStore, priority_topk_mask

# Deployed-step budgets (protocol).
STEP_BUDGET = {"step": 12, "flop": 16}
SGCR_TOTAL_STEPS = 16


# --------------------------------------------------------------------------- #
# Global uniform reservoir (Vitter) -- global ER arms
# --------------------------------------------------------------------------- #
class UniformReservoir:
    """Classic Vitter reservoir over canonical records under a fixed capacity.

    Global ER may use metadata bytes it does not retain, so it can hold the most
    examples under the envelope; capacity is computed by the caller from the
    byte budget.  Source-free."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._records: List[Dict] = []
        self._seen = 0

    def offer_one(self, record: Dict, rng: np.random.Generator) -> None:
        self._seen += 1
        if len(self._records) < self.capacity:
            self._records.append(record)
        else:
            j = int(rng.integers(0, self._seen))
            if j < self.capacity:
                self._records[j] = record

    def offer(self, records: List[Dict], rng: np.random.Generator) -> None:
        for r in records:
            self.offer_one(r, rng)

    def sample_replay(self, n: int, rng: np.random.Generator) -> List[Dict]:
        m = len(self._records)
        if m == 0 or n <= 0:
            return []
        idx = rng.choice(m, size=n, replace=(n > m))
        return [self._records[int(i)] for i in idx]

    def __len__(self):
        return len(self._records)


# --------------------------------------------------------------------------- #
# Persistent-loss buffer (CVaR/JTT arm)
# --------------------------------------------------------------------------- #
class PersistentLossBuffer:
    """Global reservoir + one charged float32 loss EMA per retained record.
    Replay draws from a 50/50 mixture of the full reservoir and the highest-EMA
    20% of retained records (protocol).  Because it charges the EMA metadata, it
    retains fewer examples than global ER under the same envelope."""

    def __init__(self, capacity: int, ema_decay: float = 0.9):
        self.capacity = int(capacity)
        self.ema_decay = float(ema_decay)
        self._records: List[Dict] = []
        self._ema: List[float] = []
        self._seen = 0

    def offer_one(self, record: Dict, first_loss: float,
                  rng: np.random.Generator) -> None:
        self._seen += 1
        if len(self._records) < self.capacity:
            self._records.append(record)
            self._ema.append(float(first_loss))
        else:
            j = int(rng.integers(0, self._seen))
            if j < self.capacity:
                self._records[j] = record
                self._ema[j] = float(first_loss)

    def offer(self, records: List[Dict], first_losses, rng) -> None:
        for r, fl in zip(records, first_losses):
            self.offer_one(r, fl, rng)

    def update_ema(self, indices: List[int], losses: List[float]) -> None:
        """EMA update for records that were just evaluated: ema <- 0.9*ema +
        0.1*current."""
        for i, l in zip(indices, losses):
            self._ema[i] = self.ema_decay * self._ema[i] + (1 - self.ema_decay) * float(l)

    def sample_replay(self, n: int, rng: np.random.Generator) -> List[Dict]:
        m = len(self._records)
        if m == 0 or n <= 0:
            return []
        n_hard = n // 2
        n_full = n - n_hard
        out: List[Dict] = []
        # full-reservoir half
        if n_full:
            idx = rng.choice(m, size=n_full, replace=(n_full > m))
            out += [self._records[int(i)] for i in idx]
        # highest-EMA 20% half
        if n_hard:
            k = max(1, int(round(0.2 * m)))
            hard_idx = np.argsort(-np.asarray(self._ema))[:k]
            pick = rng.choice(hard_idx, size=n_hard, replace=(n_hard > k))
            out += [self._records[int(i)] for i in pick]
        return out

    def __len__(self):
        return len(self._records)


# --------------------------------------------------------------------------- #
# True-category oracle store (development-only)
# --------------------------------------------------------------------------- #
class OracleGroupStore:
    """Development-only source-labelled minimum storage + group weighting.
    Establishes whether PERFECT grouping could help (an upper bound on any
    pseudo-group method).  The source label lives ONLY here and is passed in
    explicitly by the driver's evaluator interface, never via a learner record.

    Keeps a per-source minimum reservoir; replay weights groups to equalize
    (inverse-frequency), i.e. an oracle Group-DRO-style coverage."""

    def __init__(self, capacity: int, n_sources: int, min_per_source: int = 6):
        self.capacity = int(capacity)
        self.n_sources = int(n_sources)
        self.min_per_source = int(min_per_source)
        self._by_source: Dict[int, List[Dict]] = {s: [] for s in range(n_sources)}
        self._seen: Dict[int, int] = {s: 0 for s in range(n_sources)}

    def offer(self, records: List[Dict], sources: List[int],
              rng: np.random.Generator) -> None:
        cap_per = max(self.min_per_source, self.capacity // self.n_sources)
        for r, s in zip(records, sources):
            s = int(s)
            self._seen[s] += 1
            pool = self._by_source[s]
            if len(pool) < cap_per:
                pool.append(r)
            else:
                j = int(rng.integers(0, self._seen[s]))
                if j < cap_per:
                    pool[j] = r

    def sample_replay(self, n: int, rng: np.random.Generator) -> List[Dict]:
        """Inverse-frequency group weighting: draw groups uniformly over
        non-empty sources (equalizing rare/common), then uniformly within."""
        nonempty = [s for s, p in self._by_source.items() if p]
        if not nonempty or n <= 0:
            return []
        out: List[Dict] = []
        for _ in range(n):
            s = int(rng.choice(nonempty))
            pool = self._by_source[s]
            out.append(pool[int(rng.integers(0, len(pool)))])
        return out

    def __len__(self):
        return sum(len(p) for p in self._by_source.values())


# --------------------------------------------------------------------------- #
# Arm descriptors -- unify the interface for the driver
# --------------------------------------------------------------------------- #
@dataclass
class Arm:
    """A deployable arm: name, deploy-step count, buffer object, and a sampler
    callable ``sample_replay(n, rng) -> [records]``.  ``kind`` tags how the
    driver should offer records after a window and whether it branches."""
    name: str
    n_deploy_steps: int
    buffer: object
    kind: str                                # 'sequential'|'er'|'cell_q0'|'ploss'|'oracle'|'sgcr'
    uses_cells: bool = False
    is_oracle: bool = False
    meta: Dict = field(default_factory=dict)

    def sample_replay(self, n: int, rng: np.random.Generator) -> List[Dict]:
        if self.buffer is None:
            return []
        return self.buffer.sample_replay(n, rng)


def make_arms(capacity: Dict[str, int], n_sources: int) -> Dict[str, Arm]:
    """Construct the required deployable arms given per-arm byte capacities.

    ``capacity`` maps arm-family -> record capacity, computed by the driver from
    the fixed envelope (global ER can hold more; persistent-loss holds fewer;
    cell arms use the min-plus-shared capacity).  SGCR shares the cell capacity
    and is built by the driver together with the codebook.
    """
    arms: Dict[str, Arm] = {}
    arms["sequential"] = Arm("sequential", 12, None, kind="sequential")
    arms["er_step"] = Arm("er_step", STEP_BUDGET["step"],
                          UniformReservoir(capacity["er"]), kind="er")
    arms["er_flop"] = Arm("er_flop", STEP_BUDGET["flop"],
                          UniformReservoir(capacity["er"]), kind="er")
    arms["cell_q0_step"] = Arm(
        "cell_q0_step", STEP_BUDGET["step"],
        MinPlusSharedStore(capacity["cell"], k_cells=8, min_per_cell=6),
        kind="cell_q0", uses_cells=True)
    arms["cell_q0_flop"] = Arm(
        "cell_q0_flop", STEP_BUDGET["flop"],
        MinPlusSharedStore(capacity["cell"], k_cells=8, min_per_cell=6),
        kind="cell_q0", uses_cells=True)
    arms["ploss_cvar"] = Arm("ploss_cvar", STEP_BUDGET["flop"],
                             PersistentLossBuffer(capacity["ploss"]), kind="ploss")
    arms["oracle"] = Arm("oracle", STEP_BUDGET["flop"],
                         OracleGroupStore(capacity["oracle"], n_sources),
                         kind="oracle", is_oracle=True)
    return arms
