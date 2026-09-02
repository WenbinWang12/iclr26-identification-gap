"""Fixed-budget LoRA atom bank for the T5-small q/v projections.

The implementation intentionally does not depend on PEFT.  Every LoRA rank-one
component is represented by an independent ``LoRAAtom``.  This makes the
global budget physical rather than cosmetic: inactive capacity has
``requires_grad=False`` and therefore does not enter the trainable-scalar
count, while an optimizer may still be constructed over all atom parameters so
that later rank exchanges need not rebuild it.

The LoRA multiplier is always ``alpha / reference_rank``.  It never depends on
the current number of active atoms in a layer; reallocating rank therefore does
not silently rescale the atoms that remain active.  The default forward path
stacks active atoms into two small GEMMs.  A slower strict mode is available
when bitwise (rather than numerical) zero-impact activation is needed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_COMPACT_FORMAT = "phase2i.fixed-slot-t5-lora.v1"


@dataclass(frozen=True)
class PayloadAudit:
    """Exact adapter-payload accounting for the current global mask.

    ``active_weight_*`` is the method-comparison payload: only A/B values that
    affect the exported adapter.  ``compact_tensor_bytes`` additionally counts
    the int64 slot indices needed by this lossless compact representation.
    ``capacity_weight_*`` reports the larger in-memory atom reservoir and must
    not be mistaken for the exported cumulative payload.
    """

    adapter_modules: int
    active_atoms: int
    budget_atoms: int
    capacity_atoms: int
    active_weight_scalars: int
    active_trainable_scalars: int
    capacity_weight_scalars: int
    active_weight_bytes: int
    capacity_weight_bytes: int
    slot_index_bytes: int
    compact_tensor_bytes: int
    base_trainable_scalars: int
    exact_budget: bool

    def to_dict(self) -> dict[str, int | bool]:
        """Return a JSON-serializable audit record."""

        return asdict(self)


class LoRAAtom(nn.Module):
    """One rank-one LoRA component ``b @ a``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        factory_kwargs: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.empty(in_features, **factory_kwargs))
        self.b = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        self.reset_zero_impact()
        self.set_active(False)

    @torch.no_grad()
    def reset_zero_impact(self) -> None:
        """Reinitialize the atom so enabling it leaves the function unchanged."""

        # This is the row-wise equivalent of nn.Linear.reset_parameters for A.
        bound = self.a.numel() ** -0.5
        self.a.uniform_(-bound, bound)
        self.b.zero_()

    def set_active(self, active: bool) -> None:
        """Make the atom trainable or frozen without changing its values."""

        self.a.requires_grad_(active)
        self.b.requires_grad_(active)

    def utility(self, scale: float) -> Tensor:
        """Return the detached Frobenius norm of this rank-one update."""

        return (
            self.a.detach().float().norm()
            * self.b.detach().float().norm()
            * abs(float(scale))
        )


class FixedSlotLoRALinear(nn.Module):
    """A frozen linear projection augmented by fixed-scale rank-one atoms."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        max_rank: int,
        initial_rank: int,
        alpha: float,
        reference_rank: int,
        lora_dropout: float = 0.1,
        strict_bitwise_zero_impact: bool = False,
    ) -> None:
        super().__init__()
        if max_rank <= 0:
            raise ValueError("max_rank must be positive")
        if not 0 <= initial_rank <= max_rank:
            raise ValueError("initial_rank must lie in [0, max_rank]")
        if reference_rank <= 0:
            raise ValueError("reference_rank must be positive")
        if not torch.isfinite(torch.tensor(float(alpha))):
            raise ValueError("alpha must be finite")
        if not 0.0 <= lora_dropout < 1.0:
            raise ValueError("lora_dropout must lie in [0, 1)")

        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.max_rank = int(max_rank)
        self.alpha = float(alpha)
        self.reference_rank = int(reference_rank)
        self.scale = self.alpha / self.reference_rank
        self.lora_dropout_p = float(lora_dropout)
        self.lora_dropout = nn.Dropout(p=self.lora_dropout_p)
        self.strict_bitwise_zero_impact = bool(strict_bitwise_zero_impact)

        self.base = base
        self.base.requires_grad_(False)
        factory_kwargs = {
            "device": self.base.weight.device,
            "dtype": self.base.weight.dtype,
        }
        self.atoms = nn.ModuleList(
            LoRAAtom(
                self.in_features,
                self.out_features,
                factory_kwargs=factory_kwargs,
            )
            for _ in range(self.max_rank)
        )
        self.register_buffer(
            "active_mask", torch.zeros(self.max_rank, dtype=torch.bool)
        )
        initial_mask = torch.zeros(self.max_rank, dtype=torch.bool)
        initial_mask[:initial_rank] = True
        self.set_active_mask(initial_mask, zero_new=False)

    @property
    def active_rank(self) -> int:
        """Number of active atoms in this projection."""

        return int(self.active_mask.sum().item())

    def active_indices(self) -> list[int]:
        """Active slot indices in stable ascending order."""

        return self.active_mask.nonzero(as_tuple=False).flatten().tolist()

    def set_active_mask(
        self, mask: Tensor | Sequence[bool], *, zero_new: bool = True
    ) -> None:
        """Install an arbitrary slot mask.

        By default every newly enabled slot is reset with B=0.  Consequently,
        adding slots is exactly function preserving at the activation boundary.
        Set ``zero_new=False`` only when restoring already trained atom values.
        """

        new_mask = torch.as_tensor(mask, dtype=torch.bool).flatten()
        if new_mask.numel() != self.max_rank:
            raise ValueError(
                f"mask has {new_mask.numel()} slots; expected {self.max_rank}"
            )
        old_mask = self.active_mask.detach().cpu()
        new_cpu = new_mask.detach().cpu()
        for index, atom in enumerate(self.atoms):
            newly_active = bool(new_cpu[index] and not old_mask[index])
            if newly_active and zero_new:
                atom.reset_zero_impact()
            atom.set_active(bool(new_cpu[index]))
        self.active_mask.copy_(new_mask.to(device=self.active_mask.device))

    def stacked_a(self, *, active_only: bool = False) -> Tensor:
        """Materialize A as ``[rank, in_features]`` for inspection/export."""

        indices = self.active_indices() if active_only else range(self.max_rank)
        rows = [self.atoms[index].a for index in indices]
        if not rows:
            return self.base.weight.new_empty((0, self.in_features))
        return torch.stack(rows, dim=0)

    def stacked_b(self, *, active_only: bool = False) -> Tensor:
        """Materialize B as ``[out_features, rank]`` for inspection/export."""

        indices = self.active_indices() if active_only else range(self.max_rank)
        columns = [self.atoms[index].b for index in indices]
        if not columns:
            return self.base.weight.new_empty((self.out_features, 0))
        return torch.stack(columns, dim=1)

    def atom_utility(self, *, active_only: bool = False) -> Tensor:
        """Return per-slot ``|scale| * ||a||_2 * ||b||_2`` norm proxies."""

        indices = self.active_indices() if active_only else range(self.max_rank)
        values = [self.atoms[index].utility(self.scale) for index in indices]
        if not values:
            return torch.empty(0, dtype=torch.float32)
        return torch.stack(values)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.base(inputs)
        if self.active_rank == 0:
            return output
        # Standard LoRA semantics: draw one dropout mask for the adapter input
        # and share it across every active rank-one atom.
        adapter_inputs = self.lora_dropout(inputs)
        if not self.strict_bitwise_zero_impact:
            hidden = F.linear(adapter_inputs, self.stacked_a(active_only=True))
            delta = F.linear(hidden, self.stacked_b(active_only=True))
            return output + delta * self.scale

        # The audit path evaluates atoms independently.  This keeps the
        # arithmetic for an existing atom unchanged when another slot is
        # enabled.  A wider stacked GEMM can choose a different reduction
        # kernel and introduce a tiny round-off change even with a zero B
        # column, though the two paths remain numerically equivalent.
        for index in self.active_indices():
            atom = self.atoms[index]
            hidden = F.linear(adapter_inputs, atom.a.unsqueeze(0))
            delta = F.linear(hidden, atom.b.unsqueeze(1))
            output = output + delta * self.scale
        return output


class T5GlobalRankBank(nn.Module):
    """Controller and model wrapper for a global T5 q/v LoRA atom budget.

    For the canonical T5-small architecture there are 36 q/v projections.
    The defaults allocate rank four to each, hence exactly 144 active atoms.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        max_rank: int = 8,
        initial_rank: int = 4,
        alpha: float = 16.0,
        reference_rank: int = 4,
        lora_dropout: float = 0.1,
        budget_atoms: int | None = None,
        require_t5_small: bool = True,
        strict_bitwise_zero_impact: bool = False,
    ) -> None:
        super().__init__()
        if max_rank <= 0:
            raise ValueError("max_rank must be positive")
        if not 0 <= initial_rank <= max_rank:
            raise ValueError("initial_rank must lie in [0, max_rank]")

        self.model = model
        self.max_rank = int(max_rank)
        self.initial_rank = int(initial_rank)
        self.alpha = float(alpha)
        self.reference_rank = int(reference_rank)
        self.lora_dropout_p = float(lora_dropout)
        self.strict_bitwise_zero_impact = bool(strict_bitwise_zero_impact)

        # Freeze before injecting the only parameters that may be trained.
        self.model.requires_grad_(False)
        targets = _find_t5_qv_targets(self.model)
        if not targets:
            raise ValueError("no T5Attention q/v nn.Linear projections were found")
        if require_t5_small:
            _validate_t5_small(self.model, targets)

        self._adapter_names: tuple[str, ...] = tuple(name for name, _ in targets)
        for name, linear in targets:
            parent_name, leaf_name = name.rsplit(".", 1)
            parent = self.model.get_submodule(parent_name)
            wrapped = FixedSlotLoRALinear(
                linear,
                max_rank=self.max_rank,
                initial_rank=self.initial_rank,
                alpha=self.alpha,
                reference_rank=self.reference_rank,
                lora_dropout=self.lora_dropout_p,
                strict_bitwise_zero_impact=self.strict_bitwise_zero_impact,
            )
            setattr(parent, leaf_name, wrapped)

        default_budget = len(self._adapter_names) * self.initial_rank
        self.budget_atoms = int(
            default_budget if budget_atoms is None else budget_atoms
        )
        if self.budget_atoms < 0:
            raise ValueError("budget_atoms must be non-negative")
        if self.budget_atoms > len(self._adapter_names) * self.max_rank:
            raise ValueError("budget_atoms exceeds the atom-bank capacity")
        if default_budget != self.budget_atoms:
            raise ValueError(
                "the uniform initial mask does not match budget_atoms: "
                f"{default_budget} != {self.budget_atoms}"
            )
        self.assert_exact_budget()

    @property
    def adapter_names(self) -> tuple[str, ...]:
        """Stable fully-qualified q/v projection names."""

        return self._adapter_names

    @property
    def adapters(self) -> tuple[FixedSlotLoRALinear, ...]:
        """Wrapped projections in the same stable order as ``adapter_names``."""

        modules = tuple(self.model.get_submodule(name) for name in self._adapter_names)
        if not all(isinstance(module, FixedSlotLoRALinear) for module in modules):
            raise RuntimeError("a wrapped q/v projection was replaced externally")
        return modules  # type: ignore[return-value]

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate generation to the underlying Hugging Face model."""

        generate = getattr(self.model, "generate", None)
        if generate is None:
            raise AttributeError("the wrapped model does not provide generate()")
        return generate(*args, **kwargs)

    def layer_ranks(self) -> dict[str, int]:
        """Current rank allocation keyed by projection name."""

        return {
            name: adapter.active_rank
            for name, adapter in zip(self._adapter_names, self.adapters)
        }

    def global_mask(self) -> Tensor:
        """Return a CPU copy shaped ``[num_projections, max_rank]``."""

        return torch.stack(
            [adapter.active_mask.detach().cpu() for adapter in self.adapters], dim=0
        )

    def set_global_mask(
        self,
        mask: Tensor | Sequence[Sequence[bool]],
        *,
        zero_new: bool = True,
    ) -> None:
        """Apply an arbitrary non-uniform mask with the exact global budget."""

        new_mask = torch.as_tensor(mask, dtype=torch.bool)
        expected_shape = (len(self._adapter_names), self.max_rank)
        if tuple(new_mask.shape) != expected_shape:
            raise ValueError(
                f"global mask has shape {tuple(new_mask.shape)}; "
                f"expected {expected_shape}"
            )
        active = int(new_mask.sum().item())
        if active != self.budget_atoms:
            raise ValueError(
                f"global mask has {active} atoms; fixed budget is {self.budget_atoms}"
            )
        for adapter, row in zip(self.adapters, new_mask):
            adapter.set_active_mask(row, zero_new=zero_new)
        self.assert_exact_budget()

    def set_layer_ranks(
        self,
        ranks: Mapping[str, int] | Sequence[int],
        *,
        zero_new: bool = True,
    ) -> None:
        """Set a contiguous-prefix mask from a per-projection rank allocation."""

        if isinstance(ranks, Mapping):
            unknown = set(ranks).difference(self._adapter_names)
            missing = set(self._adapter_names).difference(ranks)
            if unknown or missing:
                raise ValueError(
                    f"rank map mismatch; missing={sorted(missing)}, "
                    f"unknown={sorted(unknown)}"
                )
            values = [int(ranks[name]) for name in self._adapter_names]
        else:
            values = [int(rank) for rank in ranks]
            if len(values) != len(self._adapter_names):
                raise ValueError(
                    f"received {len(values)} ranks; expected "
                    f"{len(self._adapter_names)}"
                )
        if any(rank < 0 or rank > self.max_rank for rank in values):
            raise ValueError(f"every layer rank must lie in [0, {self.max_rank}]")
        mask = torch.zeros(
            (len(self._adapter_names), self.max_rank), dtype=torch.bool
        )
        for row, rank in enumerate(values):
            mask[row, :rank] = True
        self.set_global_mask(mask, zero_new=zero_new)

    def atom_utilities(self, *, active_only: bool = False) -> dict[str, Tensor]:
        """Detached rank-one norm proxies keyed by projection name."""

        return {
            name: adapter.atom_utility(active_only=active_only)
            for name, adapter in zip(self._adapter_names, self.adapters)
        }

    def set_strict_bitwise_zero_impact(self, enabled: bool) -> None:
        """Select the strict audit path or the efficient stacked-GEMM path.

        Both modes implement the same LoRA update.  Strict mode preserves the
        exact floating-point operation sequence of existing atoms when a zero
        atom is added; the default stacked mode is substantially faster and is
        appropriate for training runs.
        """

        self.strict_bitwise_zero_impact = bool(enabled)
        for adapter in self.adapters:
            adapter.strict_bitwise_zero_impact = self.strict_bitwise_zero_impact

    def swap_atom_slots(
        self,
        *,
        remove: tuple[str | int, int],
        add: tuple[str | int, int],
        optimizer: torch.optim.Optimizer | None = None,
        zero_new: bool = True,
    ) -> tuple[nn.Parameter, ...]:
        """Exchange one active slot for one inactive slot under the budget.

        A slot reference is ``(projection_name_or_index, slot_index)``.  The
        update is validated before any mutation.  If an optimizer is supplied,
        state for both affected atoms is removed so a reset destination cannot
        inherit stale Adam moments and a retired source cannot retain hidden
        history.  Build that optimizer from :meth:`all_atom_parameters` so the
        destination parameters are already present in its parameter groups.
        """

        def resolve(reference: tuple[str | int, int], label: str) -> tuple[int, int]:
            if not isinstance(reference, tuple) or len(reference) != 2:
                raise TypeError(f"{label} must be a (module, slot) tuple")
            module_reference, slot_reference = reference
            if isinstance(module_reference, str):
                try:
                    module_index = self._adapter_names.index(module_reference)
                except ValueError as error:
                    raise ValueError(
                        f"{label} names unknown adapter {module_reference!r}"
                    ) from error
            elif isinstance(module_reference, int) and not isinstance(
                module_reference, bool
            ):
                module_index = module_reference
            else:
                raise TypeError(f"{label} module must be a name or integer index")
            if not 0 <= module_index < len(self._adapter_names):
                raise IndexError(f"{label} module index {module_index} is out of range")
            if not isinstance(slot_reference, int) or isinstance(slot_reference, bool):
                raise TypeError(f"{label} slot must be an integer")
            if not 0 <= slot_reference < self.max_rank:
                raise IndexError(f"{label} slot {slot_reference} is out of range")
            return module_index, slot_reference

        remove_index = resolve(remove, "remove")
        add_index = resolve(add, "add")
        if remove_index == add_index:
            raise ValueError("remove and add must identify different slots")
        mask = self.global_mask()
        if not bool(mask[remove_index]):
            raise ValueError("remove slot is not active")
        if bool(mask[add_index]):
            raise ValueError("add slot is already active")

        remove_atom = self.adapters[remove_index[0]].atoms[remove_index[1]]
        add_atom = self.adapters[add_index[0]].atoms[add_index[1]]
        affected = (remove_atom.a, remove_atom.b, add_atom.a, add_atom.b)
        if optimizer is not None:
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise TypeError("optimizer must be a torch.optim.Optimizer")
            optimizer_parameter_ids = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            missing = [
                parameter
                for parameter in affected
                if id(parameter) not in optimizer_parameter_ids
            ]
            if missing:
                raise ValueError(
                    "optimizer does not contain every affected atom parameter; "
                    "construct it from all_atom_parameters()"
                )
        mask[remove_index] = False
        mask[add_index] = True
        self.set_global_mask(mask, zero_new=zero_new)
        if optimizer is not None:
            self.clear_optimizer_state(optimizer, affected)
        return affected

    @staticmethod
    def clear_optimizer_state(
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[nn.Parameter],
    ) -> int:
        """Delete optimizer state for parameters and return entries removed."""

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        removed = 0
        for parameter in parameters:
            if parameter in optimizer.state:
                del optimizer.state[parameter]
                removed += 1
        return removed

    def all_atom_parameters(self) -> Iterable[nn.Parameter]:
        """Yield active and inactive atom parameters for optimizer creation.

        Constructing the optimizer from this iterator (without filtering on
        ``requires_grad``) lets newly activated slots start receiving gradients
        after a rank exchange.  Inactive slots have no gradient or optimizer
        state and remain outside the active parameter budget.
        """

        for adapter in self.adapters:
            for atom in adapter.atoms:
                yield atom.a
                yield atom.b

    def active_atom_parameters(self) -> Iterable[nn.Parameter]:
        """Yield only parameters active under the current mask."""

        for parameter in self.all_atom_parameters():
            if parameter.requires_grad:
                yield parameter

    def assert_exact_budget(self) -> None:
        """Fail closed if the mask or trainable parameter count drifts."""

        active_atoms = sum(adapter.active_rank for adapter in self.adapters)
        if active_atoms != self.budget_atoms:
            raise RuntimeError(
                f"active atom count {active_atoms} != budget {self.budget_atoms}"
            )
        active_ids = {id(parameter) for parameter in self.active_atom_parameters()}
        unexpected = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and id(parameter) not in active_ids
        ]
        if unexpected:
            raise RuntimeError(
                "non-adapter parameters became trainable: " + ", ".join(unexpected)
            )

    def payload_audit(self) -> PayloadAudit:
        """Compute exact scalar and tensor-byte accounting from live tensors."""

        active_atoms = 0
        active_scalars = 0
        active_bytes = 0
        capacity_scalars = 0
        capacity_bytes = 0
        for adapter in self.adapters:
            active_indices = set(adapter.active_indices())
            active_atoms += len(active_indices)
            for index, atom in enumerate(adapter.atoms):
                atom_scalars = atom.a.numel() + atom.b.numel()
                atom_bytes = (
                    atom.a.numel() * atom.a.element_size()
                    + atom.b.numel() * atom.b.element_size()
                )
                capacity_scalars += atom_scalars
                capacity_bytes += atom_bytes
                if index in active_indices:
                    active_scalars += atom_scalars
                    active_bytes += atom_bytes
        active_trainable = sum(
            parameter.numel() for parameter in self.active_atom_parameters()
        )
        active_ids = {id(parameter) for parameter in self.active_atom_parameters()}
        base_trainable = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in active_ids
        )
        index_bytes = active_atoms * torch.tensor([], dtype=torch.int64).element_size()
        return PayloadAudit(
            adapter_modules=len(self._adapter_names),
            active_atoms=active_atoms,
            budget_atoms=self.budget_atoms,
            capacity_atoms=len(self._adapter_names) * self.max_rank,
            active_weight_scalars=active_scalars,
            active_trainable_scalars=active_trainable,
            capacity_weight_scalars=capacity_scalars,
            active_weight_bytes=active_bytes,
            capacity_weight_bytes=capacity_bytes,
            slot_index_bytes=index_bytes,
            compact_tensor_bytes=active_bytes + index_bytes,
            base_trainable_scalars=base_trainable,
            exact_budget=(
                active_atoms == self.budget_atoms
                and active_scalars == active_trainable
                and base_trainable == 0
            ),
        )

    def compact_state_dict(self) -> dict[str, Any]:
        """Export only active adapter tensors plus lossless slot metadata."""

        self.assert_exact_budget()
        adapter_states: dict[str, dict[str, Tensor]] = {}
        for name, adapter in zip(self._adapter_names, self.adapters):
            indices = torch.tensor(adapter.active_indices(), dtype=torch.int64)
            adapter_states[name] = {
                "slot_indices": indices,
                "lora_A": adapter.stacked_a(active_only=True).detach().cpu().clone(),
                "lora_B": adapter.stacked_b(active_only=True).detach().cpu().clone(),
            }
        audit = self.payload_audit()
        return {
            "format": _COMPACT_FORMAT,
            "alpha": self.alpha,
            "reference_rank": self.reference_rank,
            "lora_dropout": self.lora_dropout_p,
            "max_rank": self.max_rank,
            "budget_atoms": self.budget_atoms,
            "strict_bitwise_zero_impact": self.strict_bitwise_zero_impact,
            "adapter_names": self._adapter_names,
            "adapters": adapter_states,
            "payload_audit": audit.to_dict(),
        }

    @torch.no_grad()
    def load_compact_state_dict(
        self, state: Mapping[str, Any], *, strict: bool = True
    ) -> None:
        """Restore a compact export into an identical frozen backbone."""

        if state.get("format") != _COMPACT_FORMAT:
            raise ValueError(f"unsupported compact format: {state.get('format')!r}")
        expected_metadata = {
            "alpha": self.alpha,
            "reference_rank": self.reference_rank,
            "lora_dropout": self.lora_dropout_p,
            "max_rank": self.max_rank,
            "budget_atoms": self.budget_atoms,
            "strict_bitwise_zero_impact": self.strict_bitwise_zero_impact,
        }
        for key, expected in expected_metadata.items():
            if state.get(key) != expected:
                raise ValueError(
                    f"compact state {key}={state.get(key)!r}; expected {expected!r}"
                )
        state_names = tuple(state.get("adapter_names", ()))
        if state_names != self._adapter_names:
            raise ValueError("compact adapter names/order do not match this model")
        states = state.get("adapters")
        if not isinstance(states, Mapping):
            raise ValueError("compact state is missing its adapters mapping")
        if strict and set(states) != set(self._adapter_names):
            raise ValueError("compact adapter mapping has missing or extra entries")

        global_mask = torch.zeros(
            (len(self._adapter_names), self.max_rank), dtype=torch.bool
        )
        for row, (name, adapter) in enumerate(zip(self._adapter_names, self.adapters)):
            if name not in states:
                raise ValueError(f"compact state is missing adapter {name}")
            entry = states[name]
            if not isinstance(entry, Mapping):
                raise ValueError(f"compact adapter {name} is not a mapping")
            indices = torch.as_tensor(entry.get("slot_indices"), dtype=torch.int64)
            if indices.ndim != 1:
                raise ValueError(f"slot indices for {name} must be one-dimensional")
            if indices.numel() and (
                int(indices.min()) < 0 or int(indices.max()) >= self.max_rank
            ):
                raise ValueError(f"slot index for {name} lies outside the bank")
            if indices.unique().numel() != indices.numel():
                raise ValueError(f"slot indices for {name} contain duplicates")
            a = torch.as_tensor(entry.get("lora_A"))
            b = torch.as_tensor(entry.get("lora_B"))
            rank = indices.numel()
            if tuple(a.shape) != (rank, adapter.in_features):
                raise ValueError(
                    f"lora_A for {name} has shape {tuple(a.shape)}; expected "
                    f"{(rank, adapter.in_features)}"
                )
            if tuple(b.shape) != (adapter.out_features, rank):
                raise ValueError(
                    f"lora_B for {name} has shape {tuple(b.shape)}; expected "
                    f"{(adapter.out_features, rank)}"
                )
            # B=0 keeps every inactive reservoir slot safe for future use.
            for atom in adapter.atoms:
                atom.b.zero_()
                atom.set_active(False)
            adapter.active_mask.zero_()
            for compact_index, slot_index_tensor in enumerate(indices):
                slot_index = int(slot_index_tensor)
                atom = adapter.atoms[slot_index]
                atom.a.copy_(a[compact_index].to(device=atom.a.device, dtype=atom.a.dtype))
                atom.b.copy_(b[:, compact_index].to(device=atom.b.device, dtype=atom.b.dtype))
                global_mask[row, slot_index] = True
        self.set_global_mask(global_mask, zero_new=False)
        self.assert_exact_budget()

    def compact_manifest(self) -> dict[str, Any]:
        """Return shapes and exact bytes without copying adapter values."""

        modules = []
        for name, adapter in zip(self._adapter_names, self.adapters):
            rank = adapter.active_rank
            modules.append(
                {
                    "name": name,
                    "slot_indices": adapter.active_indices(),
                    "lora_A_shape": [rank, adapter.in_features],
                    "lora_B_shape": [adapter.out_features, rank],
                }
            )
        return {
            "format": _COMPACT_FORMAT,
            "lora_dropout": self.lora_dropout_p,
            "strict_bitwise_zero_impact": self.strict_bitwise_zero_impact,
            "payload_audit": self.payload_audit().to_dict(),
            "modules": modules,
        }


def _find_t5_qv_targets(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Find direct q/v children of Hugging Face ``T5Attention`` modules."""

    targets: list[tuple[str, nn.Linear]] = []
    for attention_name, module in model.named_modules():
        if module.__class__.__name__ != "T5Attention":
            continue
        for projection_name in ("q", "v"):
            projection = getattr(module, projection_name, None)
            if not isinstance(projection, nn.Linear):
                raise TypeError(
                    f"{attention_name}.{projection_name} is not an nn.Linear; "
                    "the model may already be wrapped or use an unsupported T5 variant"
                )
            targets.append((f"{attention_name}.{projection_name}", projection))
    return targets


def _validate_t5_small(
    model: nn.Module, targets: Sequence[tuple[str, nn.Linear]]
) -> None:
    """Reject silent architecture drift from the canonical T5-small budget."""

    config = getattr(model, "config", None)
    problems: list[str] = []
    if config is None:
        problems.append("model has no Hugging Face config")
    else:
        if getattr(config, "model_type", None) != "t5":
            problems.append(f"model_type={getattr(config, 'model_type', None)!r}")
        if getattr(config, "d_model", None) != 512:
            problems.append(f"d_model={getattr(config, 'd_model', None)!r}")
        if getattr(config, "num_layers", None) != 6:
            problems.append(f"num_layers={getattr(config, 'num_layers', None)!r}")
        decoder_layers = getattr(
            config, "num_decoder_layers", getattr(config, "num_layers", None)
        )
        if decoder_layers != 6:
            problems.append(f"num_decoder_layers={decoder_layers!r}")
    if len(targets) != 36:
        problems.append(f"q/v projection count={len(targets)}")
    bad_shapes = [
        f"{name}:{linear.in_features}x{linear.out_features}"
        for name, linear in targets
        if linear.in_features != 512 or linear.out_features != 512
    ]
    if bad_shapes:
        problems.append("non-512 q/v projections=" + ", ".join(bad_shapes))
    if problems:
        raise ValueError("not canonical T5-small: " + "; ".join(problems))


__all__ = [
    "FixedSlotLoRALinear",
    "LoRAAtom",
    "PayloadAudit",
    "T5GlobalRankBank",
]
