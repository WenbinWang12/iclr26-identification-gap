"""Focused checks for the fixed-budget T5-small LoRA atom bank."""

from __future__ import annotations

import os
import sys

import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from t5_rank_bank import FixedSlotLoRALinear, T5GlobalRankBank  # noqa: E402


def _small_backbone() -> T5ForConditionalGeneration:
    """Canonical attention geometry with cheap embeddings/feed-forward blocks."""

    config = T5Config(
        vocab_size=32,
        d_model=512,
        d_kv=64,
        d_ff=16,
        num_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        dropout_rate=0.0,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    return T5ForConditionalGeneration(config)


def test_wraps_exact_t5_small_qv_budget_and_freezes_base():
    bank = T5GlobalRankBank(_small_backbone())
    audit = bank.payload_audit()

    assert len(bank.adapter_names) == 36
    assert bank.lora_dropout_p == 0.1
    assert not bank.strict_bitwise_zero_impact
    assert all(name.endswith((".q", ".v")) for name in bank.adapter_names)
    assert all(rank == 4 for rank in bank.layer_ranks().values())
    assert audit.active_atoms == audit.budget_atoms == 144
    assert audit.capacity_atoms == 288
    assert audit.active_weight_scalars == 144 * (512 + 512) == 147_456
    assert audit.active_trainable_scalars == 147_456
    assert audit.capacity_weight_scalars == 294_912
    assert audit.active_weight_bytes == 147_456 * 4
    assert audit.base_trainable_scalars == 0
    assert audit.exact_budget

    trainable = sum(p.numel() for p in bank.parameters() if p.requires_grad)
    assert trainable == 147_456
    assert list(bank.all_atom_parameters())


def test_fixed_scaling_and_new_slot_activation_are_zero_impact():
    torch.manual_seed(3)
    base = nn.Linear(7, 5, bias=False)
    adapter = FixedSlotLoRALinear(
        base,
        max_rank=4,
        initial_rank=1,
        alpha=12.0,
        reference_rank=3,
        lora_dropout=0.0,
        strict_bitwise_zero_impact=True,
    )
    assert adapter.scale == 4.0
    with torch.no_grad():
        adapter.atoms[0].b.normal_()
    inputs = torch.randn(2, 4, 7)
    before = adapter(inputs)

    # Adding an atom resets its B vector to zero and cannot change the output.
    adapter.set_active_mask([True, False, True, False])
    after = adapter(inputs)
    assert adapter.active_rank == 2
    assert adapter.scale == 4.0
    assert torch.equal(before, after)
    assert adapter.atoms[2].a.requires_grad
    assert adapter.atoms[2].b.requires_grad
    assert torch.count_nonzero(adapter.atoms[2].b) == 0
    assert not adapter.atoms[1].a.requires_grad


def test_fast_stacked_mode_is_numerically_zero_impact_and_eval_deterministic():
    torch.manual_seed(5)
    adapter = FixedSlotLoRALinear(
        nn.Linear(7, 5, bias=False),
        max_rank=4,
        initial_rank=1,
        alpha=16.0,
        reference_rank=4,
        lora_dropout=0.1,
    )
    with torch.no_grad():
        adapter.atoms[0].b.normal_()
    adapter.eval()
    inputs = torch.randn(3, 7)
    before = adapter(inputs)
    assert torch.equal(before, adapter(inputs))
    adapter.set_active_mask([True, False, True, False])
    after = adapter(inputs)
    assert torch.allclose(before, after, rtol=1e-6, atol=1e-6)


def test_arbitrary_noncontiguous_mask_preserves_exact_global_budget():
    bank = T5GlobalRankBank(_small_backbone())
    mask = torch.zeros((36, 8), dtype=torch.bool)
    mask[:, [0, 2, 4, 7]] = True
    bank.set_global_mask(mask)

    assert torch.equal(bank.global_mask(), mask)
    assert set(bank.layer_ranks().values()) == {4}
    assert bank.payload_audit().exact_budget
    assert sum(p.numel() for p in bank.parameters() if p.requires_grad) == 147_456

    invalid = mask.clone()
    invalid[0, 7] = False
    try:
        bank.set_global_mask(invalid)
    except ValueError as error:
        assert "fixed budget" in str(error)
    else:
        raise AssertionError("a 143-atom mask was accepted")
    assert torch.equal(bank.global_mask(), mask)

    optimizer = torch.optim.AdamW(bank.all_atom_parameters(), lr=1e-3)
    remove_atom = bank.adapters[0].atoms[0]
    add_atom = bank.adapters[0].atoms[1]
    with torch.no_grad():
        add_atom.b.fill_(1.0)
    expected_affected = (remove_atom.a, remove_atom.b, add_atom.a, add_atom.b)
    for parameter in expected_affected:
        optimizer.state[parameter]["stale_probe"] = torch.ones(())
    affected = bank.swap_atom_slots(
        remove=(bank.adapter_names[0], 0),
        add=(0, 1),
        optimizer=optimizer,
    )
    assert affected == expected_affected
    assert not bool(bank.global_mask()[0, 0])
    assert bool(bank.global_mask()[0, 1])
    assert torch.count_nonzero(add_atom.b) == 0
    assert all(parameter not in optimizer.state for parameter in affected)
    assert bank.payload_audit().exact_budget
    assert sum(p.numel() for p in bank.parameters() if p.requires_grad) == 147_456


def test_compact_export_import_is_lossless_and_byte_audited():
    torch.manual_seed(11)
    base_a = _small_backbone()
    base_b = _small_backbone()
    base_b.load_state_dict(base_a.state_dict())
    bank_a = T5GlobalRankBank(base_a)
    bank_b = T5GlobalRankBank(base_b)

    # Exercise highly non-uniform ranks while retaining 144 atoms globally.
    ranks = [8] * 18 + [0] * 18
    bank_a.set_layer_ranks(ranks)
    with torch.no_grad():
        for adapter in bank_a.adapters:
            for index in adapter.active_indices():
                adapter.atoms[index].a.normal_(mean=0.0, std=0.01)
                adapter.atoms[index].b.normal_(mean=0.0, std=0.01)

    state = bank_a.compact_state_dict()
    tensor_bytes = 0
    tensor_scalars = 0
    for entry in state["adapters"].values():
        for tensor in entry.values():
            tensor_bytes += tensor.numel() * tensor.element_size()
        tensor_scalars += entry["lora_A"].numel() + entry["lora_B"].numel()
    audit = bank_a.payload_audit()
    assert tensor_scalars == audit.active_weight_scalars
    assert tensor_bytes == audit.compact_tensor_bytes

    bank_b.load_compact_state_dict(state)
    assert bank_b.layer_ranks() == bank_a.layer_ranks()
    assert torch.equal(bank_b.global_mask(), bank_a.global_mask())
    assert bank_b.payload_audit() == audit

    input_ids = torch.tensor([[2, 3, 1], [5, 6, 1]])
    labels = torch.tensor([[7, 1], [8, 1]])
    bank_a.eval()
    bank_b.eval()
    with torch.no_grad():
        logits_a = bank_a(input_ids=input_ids, labels=labels).logits
        logits_b = bank_b(input_ids=input_ids, labels=labels).logits
    assert torch.equal(logits_a, logits_b)


if __name__ == "__main__":
    test_wraps_exact_t5_small_qv_budget_and_freezes_base()
    test_fixed_scaling_and_new_slot_activation_are_zero_impact()
    test_fast_stacked_mode_is_numerically_zero_impact_and_eval_deterministic()
    test_arbitrary_noncontiguous_mask_preserves_exact_global_budget()
    test_compact_export_import_is_lossless_and_byte_audited()
    print("test_t5_rank_bank OK")
