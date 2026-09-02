"""Differentiable Phase-2I anchored-retention objectives.

The anchor losses are fixed measurements made around an example's arrival
episode.  Only ``loss_now`` is differentiated.  Adversarial CVaR/DRO weights
are deliberately computed from detached values: they are an exact best
response of the finite empirical adversary, not a second gradient path through
the sort or group selection operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TorchAnchoredRegret:
    """A differentiable anchored-regret vector and its fixed eligibility."""

    regret: torch.Tensor
    acquired_gain: torch.Tensor
    eligible: torch.Tensor

    @property
    def tail_values(self) -> torch.Tensor:
        """Eligible entries that may enter the normalized risk objective."""

        return self.regret[self.eligible]


@dataclass(frozen=True)
class WeightedRiskObjective:
    """A scalar empirical risk together with auditable per-example terms."""

    loss: torch.Tensor
    regret: TorchAnchoredRegret
    weights: torch.Tensor


def _floating_vector(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must be a non-empty vector, got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} contains a non-finite value")
    return value


def _boolean_mask(
    mask: torch.Tensor | Sequence[bool] | None,
    *,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    if mask is None:
        return torch.ones(length, dtype=torch.bool, device=device)
    out = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if out.ndim != 1 or out.numel() != length:
        raise ValueError(f"mask must have shape ({length},), got {tuple(out.shape)}")
    return out.detach()


def per_example_target_token_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return mean target-token NLL for every sequence in a batch.

    Padding/ignored labels do not enter either the numerator or denominator.
    A sequence with no scored target tokens is rejected instead of silently
    producing a zero or NaN anchor.
    """

    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("logits and labels must be torch tensors")
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, target_length, vocabulary]")
    if labels.ndim != 2 or tuple(labels.shape) != tuple(logits.shape[:2]):
        raise ValueError(
            "labels must have shape [batch, target_length] matching logits; "
            f"got {tuple(labels.shape)} and {tuple(logits.shape)}"
        )
    if not logits.is_floating_point():
        raise TypeError("logits must have a floating-point dtype")
    if labels.dtype == torch.bool or labels.is_floating_point():
        raise TypeError("labels must have an integer dtype")
    if logits.shape[-1] <= 0:
        raise ValueError("vocabulary dimension must be positive")
    if not bool(torch.isfinite(logits.detach()).all()):
        raise ValueError("logits contain a non-finite value")

    scored = labels.ne(ignore_index)
    token_counts = scored.sum(dim=1)
    if bool(token_counts.eq(0).any()):
        bad = token_counts.eq(0).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"examples {bad} contain no scored target tokens")
    scored_labels = labels[scored]
    if bool(((scored_labels < 0) | (scored_labels >= logits.shape[-1])).any()):
        raise ValueError("a non-ignored label lies outside the vocabulary")

    token_nll = F.cross_entropy(
        logits.transpose(1, 2),
        labels.long(),
        ignore_index=ignore_index,
        reduction="none",
    )
    return token_nll.sum(dim=1) / token_counts.to(dtype=token_nll.dtype)


def differentiable_anchored_regret(
    loss_now: torch.Tensor,
    loss_pre: torch.Tensor | Sequence[float],
    loss_post: torch.Tensor | Sequence[float],
    *,
    eligibility: torch.Tensor | Sequence[bool] | None = None,
    min_acquired_gain: float = 0.05,
    epsilon: float = 1e-8,
    r_max: float = 2.0,
) -> TorchAnchoredRegret:
    """Compute anchored regret while preserving gradients through ``loss_now``.

    Supplied ``eligibility`` is intersected with the acquisition threshold.
    This lets a caller enforce persistence/EMA rules fixed by the history
    controller without ever making an example eligible solely by mistake.
    Ineligible entries are represented by differentiable zeros; callers must
    use the returned mask when constructing risk weights.
    """

    now = _floating_vector("loss_now", loss_now)
    if min_acquired_gain < 0:
        raise ValueError("min_acquired_gain must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if r_max <= 0:
        raise ValueError("r_max must be positive")

    pre = torch.as_tensor(loss_pre, dtype=now.dtype, device=now.device).detach()
    post = torch.as_tensor(loss_post, dtype=now.dtype, device=now.device).detach()
    if pre.ndim != 1 or post.ndim != 1 or pre.shape != now.shape or post.shape != now.shape:
        raise ValueError(
            "loss_pre, loss_post, and loss_now must have the same vector shape"
        )
    if not bool(torch.isfinite(pre).all()) or not bool(torch.isfinite(post).all()):
        raise ValueError("anchor losses contain a non-finite value")

    acquired = pre - post
    eligible = acquired.ge(min_acquired_gain)
    eligible &= _boolean_mask(
        eligibility, length=now.numel(), device=now.device
    )
    normalized = F.relu(now - post) / acquired.clamp_min(epsilon)
    clipped = normalized.clamp(min=0.0, max=float(r_max))
    regret = torch.where(eligible, clipped, torch.zeros_like(clipped))
    return TorchAnchoredRegret(
        regret=regret,
        acquired_gain=acquired,
        eligible=eligible,
    )


def detached_cvar_weights(
    values: torch.Tensor,
    alpha: float,
    *,
    eligible: torch.Tensor | Sequence[bool] | None = None,
) -> torch.Tensor:
    """Exact capped-LP upper-tail CVaR weights, detached from autograd.

    The cap is ``1 / (alpha * N_eligible)``.  Ineligible entries receive zero
    mass.  Fractional tail sizes are handled exactly and ties are resolved by
    stable original order; tie choice does not change the CVaR objective.
    """

    x = _floating_vector("values", values)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    mask = _boolean_mask(eligible, length=x.numel(), device=x.device)
    eligible_indices = mask.nonzero(as_tuple=False).flatten()
    if eligible_indices.numel() == 0:
        raise ValueError("CVaR requires at least one eligible example")

    detached = x.detach().to(device="cpu", dtype=torch.float64).numpy()
    selected = eligible_indices.detach().cpu().numpy()
    selected_values = detached[selected]
    order = np.argsort(-selected_values, kind="stable")
    count = len(selected)
    cap = 1.0 / (alpha * count)
    q_selected = np.zeros(count, dtype=np.float64)
    remaining = 1.0
    for local_index in order:
        mass = min(cap, remaining)
        q_selected[local_index] = mass
        remaining -= mass
        if remaining <= 32 * np.finfo(np.float64).eps:
            break
    # Correct roundoff on the largest element without violating the cap.
    q_selected[order[0]] += 1.0 - float(q_selected.sum())
    if q_selected[order[0]] > cap + 1e-12:
        raise RuntimeError("internal CVaR construction exceeded the weight cap")

    full = np.zeros(x.numel(), dtype=np.float64)
    full[selected] = q_selected
    return torch.as_tensor(full, dtype=x.dtype, device=x.device).detach()


def detached_worst_group_dro_weights(
    values: torch.Tensor,
    group_ids: Sequence[Hashable] | torch.Tensor,
    *,
    eligible: torch.Tensor | Sequence[bool] | None = None,
) -> torch.Tensor:
    """Return an exact empirical worst-group best response.

    Group risk is the mean eligible anchored regret within a task.  All mass is
    placed uniformly on the group with largest detached mean; stable first
    occurrence resolves exact ties.  This function is for the task-aware
    oracle only--a group-free learner must call :func:`detached_cvar_weights`.
    """

    x = _floating_vector("values", values)
    if isinstance(group_ids, torch.Tensor):
        if group_ids.ndim != 1 or group_ids.numel() != x.numel():
            raise ValueError("group_ids must be a vector matching values")
        raw_groups = group_ids.detach().cpu().tolist()
    else:
        raw_groups = list(group_ids)
        if len(raw_groups) != x.numel():
            raise ValueError("group_ids must have one entry per value")
    for group in raw_groups:
        try:
            hash(group)
        except TypeError as exc:
            raise TypeError("every group id must be hashable") from exc

    mask = _boolean_mask(eligible, length=x.numel(), device=x.device)
    mask_cpu = mask.detach().cpu().numpy()
    values_cpu = x.detach().to(device="cpu", dtype=torch.float64).numpy()
    group_to_indices: dict[Hashable, list[int]] = {}
    for index, group in enumerate(raw_groups):
        if mask_cpu[index]:
            group_to_indices.setdefault(group, []).append(index)
    if not group_to_indices:
        raise ValueError("worst-group DRO requires at least one eligible example")

    worst_group: Hashable | None = None
    worst_mean = -np.inf
    for group, indices in group_to_indices.items():
        mean = float(values_cpu[indices].mean())
        if mean > worst_mean:
            worst_group = group
            worst_mean = mean
    assert worst_group is not None
    worst_indices = group_to_indices[worst_group]
    full = np.zeros(x.numel(), dtype=np.float64)
    full[worst_indices] = 1.0 / len(worst_indices)
    return torch.as_tensor(full, dtype=x.dtype, device=x.device).detach()


def uniform_replay_weights(
    values_or_count: torch.Tensor | int,
    *,
    eligible: torch.Tensor | Sequence[bool] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Construct detached uniform weights over all or selected examples."""

    if isinstance(values_or_count, torch.Tensor):
        count = values_or_count.numel()
        if values_or_count.ndim != 1 or count == 0:
            raise ValueError("values tensor must be a non-empty vector")
        dtype = values_or_count.dtype
        device = values_or_count.device
    else:
        count = int(values_or_count)
        if count <= 0:
            raise ValueError("example count must be positive")
    resolved_device = torch.device("cpu" if device is None else device)
    mask = _boolean_mask(eligible, length=count, device=resolved_device)
    selected = int(mask.sum().item())
    if selected == 0:
        raise ValueError("uniform replay requires at least one eligible example")
    return mask.to(dtype=dtype).div(float(selected)).detach()


def mixed_replay_weights(
    risk_weights: torch.Tensor,
    *,
    risk_fraction: float,
    uniform_eligible: torch.Tensor | Sequence[bool] | None = None,
) -> torch.Tensor:
    """Mix uniform coverage and normalized risk weights.

    ``risk_fraction=0`` is pure uniform replay and ``1`` is pure adversarial
    replay.  This operation only constructs sampling/loss weights; it does not
    authorize audit records to enter a gradient batch.
    """

    risk = _floating_vector("risk_weights", risk_weights).detach()
    if not 0.0 <= risk_fraction <= 1.0:
        raise ValueError("risk_fraction must lie in [0, 1]")
    if bool((risk < 0).any()):
        raise ValueError("risk weights must be non-negative")
    risk_sum = risk.sum()
    if not bool(torch.isfinite(risk_sum)) or float(risk_sum) <= 0.0:
        raise ValueError("risk weights must have positive finite mass")
    risk = risk / risk_sum
    uniform = uniform_replay_weights(risk, eligible=uniform_eligible)
    mixed = (1.0 - risk_fraction) * uniform + risk_fraction * risk
    return (mixed / mixed.sum()).detach()


def weighted_risk(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Validate and reduce a differentiable vector with detached weights."""

    x = _floating_vector("values", values)
    w = _floating_vector("weights", weights).to(device=x.device, dtype=x.dtype).detach()
    if w.shape != x.shape:
        raise ValueError("values and weights must have identical shapes")
    if bool((w < 0).any()) or not torch.isclose(
        w.sum(), torch.ones((), device=w.device, dtype=w.dtype), atol=1e-6, rtol=1e-6
    ):
        raise ValueError("weights must be non-negative and sum to one")
    return torch.sum(w * x)


def groupfree_anchored_cvar_objective(
    loss_now: torch.Tensor,
    loss_pre: torch.Tensor | Sequence[float],
    loss_post: torch.Tensor | Sequence[float],
    *,
    alpha: float,
    eligibility: torch.Tensor | Sequence[bool] | None = None,
    min_acquired_gain: float = 0.05,
    epsilon: float = 1e-8,
    r_max: float = 2.0,
) -> WeightedRiskObjective:
    """Build the group-free anchored-CVaR scalar objective."""

    anchored = differentiable_anchored_regret(
        loss_now,
        loss_pre,
        loss_post,
        eligibility=eligibility,
        min_acquired_gain=min_acquired_gain,
        epsilon=epsilon,
        r_max=r_max,
    )
    weights = detached_cvar_weights(
        anchored.regret, alpha, eligible=anchored.eligible
    )
    return WeightedRiskObjective(
        loss=weighted_risk(anchored.regret, weights),
        regret=anchored,
        weights=weights,
    )


def oracle_anchored_dro_objective(
    loss_now: torch.Tensor,
    loss_pre: torch.Tensor | Sequence[float],
    loss_post: torch.Tensor | Sequence[float],
    group_ids: Sequence[Hashable] | torch.Tensor,
    *,
    eligibility: torch.Tensor | Sequence[bool] | None = None,
    min_acquired_gain: float = 0.05,
    epsilon: float = 1e-8,
    r_max: float = 2.0,
) -> WeightedRiskObjective:
    """Build the task-aware oracle's anchored worst-group objective."""

    anchored = differentiable_anchored_regret(
        loss_now,
        loss_pre,
        loss_post,
        eligibility=eligibility,
        min_acquired_gain=min_acquired_gain,
        epsilon=epsilon,
        r_max=r_max,
    )
    weights = detached_worst_group_dro_weights(
        anchored.regret, group_ids, eligible=anchored.eligible
    )
    return WeightedRiskObjective(
        loss=weighted_risk(anchored.regret, weights),
        regret=anchored,
        weights=weights,
    )


__all__ = [
    "TorchAnchoredRegret",
    "WeightedRiskObjective",
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
