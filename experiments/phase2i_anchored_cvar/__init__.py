"""Phase-2I anchored-retention risk utilities.

This package is an implementation scaffold.  It contains no experimental
result and must not be imported as evidence for a positive method claim.
"""

from .metrics import (
    AnchoredRegret,
    HeadroomGate,
    assess_headroom_gate,
    anchored_regret,
    empirical_cvar,
    empirical_cvar_weights,
    normalized_retention,
)
from .t5_rank_bank import (
    FixedSlotLoRALinear,
    LoRAAtom,
    PayloadAudit,
    T5GlobalRankBank,
)
from .buffers import (
    AnchorRecord,
    BufferRole,
    ByteLedger,
    ControllerRecord,
    HistoryBuffer,
    ReservoirDecision,
    assert_disjoint_history,
    canonical_json_bytes,
)
from .objectives import (
    TorchAnchoredRegret,
    WeightedRiskObjective,
    detached_cvar_weights,
    detached_worst_group_dro_weights,
    differentiable_anchored_regret,
    groupfree_anchored_cvar_objective,
    mixed_replay_weights,
    oracle_anchored_dro_objective,
    per_example_target_token_nll,
    uniform_replay_weights,
    weighted_risk,
)

__all__ = [
    "AnchoredRegret",
    "HeadroomGate",
    "assess_headroom_gate",
    "anchored_regret",
    "empirical_cvar",
    "empirical_cvar_weights",
    "normalized_retention",
    "FixedSlotLoRALinear",
    "LoRAAtom",
    "PayloadAudit",
    "T5GlobalRankBank",
    "AnchorRecord",
    "BufferRole",
    "ByteLedger",
    "ControllerRecord",
    "HistoryBuffer",
    "ReservoirDecision",
    "TorchAnchoredRegret",
    "WeightedRiskObjective",
    "assert_disjoint_history",
    "canonical_json_bytes",
    "detached_cvar_weights",
    "detached_worst_group_dro_weights",
    "differentiable_anchored_regret",
    "groupfree_anchored_cvar_objective",
    "mixed_replay_weights",
    "oracle_anchored_dro_objective",
    "per_example_target_token_nll",
    "uniform_replay_weights",
    "weighted_risk",
]
