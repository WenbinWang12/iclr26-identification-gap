"""Guards for the Phase-2X prior-shift scorer.

These exist for a specific reason.  Phase-2X was created because Q4's numbers
were unreadable at n=7 minority examples — the measurement quantum exceeded the
effect.  A silent bug in *this* scorer would reproduce exactly that failure in a
new costume, so the properties that make the primary axis readable are pinned by
test rather than by inspection:

* the primary axis is always scored on the full balanced split (n unchanged);
* reference levels do not move across grid cells — if a skewed draw leaked into
  R_raw/R_orc, this fails;
* the label-free offset never beats the label-reading oracle on the same rows;
* the p=0.9-at-small-batch cell that broke Q4 is reachable here, by replacement.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.phase2x_prior.score_prior import (reference_levels,
                                                   score_entry,
                                                   skewed_indices)


def make_entry(n_per_class: int = 64, K: int = 2, sep: float = 1.0,
               seed: int = 0) -> dict:
    """A synthetic balanced entry shaped like a dumped audit split."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(K), n_per_class)
    logits = rng.normal(size=(len(labels), K))
    logits[np.arange(len(labels)), labels] += sep
    return {"seed": 1, "stage": 1, "task": "synthetic",
            "logits": logits, "labels": labels,
            "verbalizer": ["a", "b"][:K], "scope": "a|b",
            "trained_task": "other", "path": "synthetic"}


@pytest.mark.parametrize("prior", [0.5, 0.7, 0.9, 0.95])
@pytest.mark.parametrize("size", [16, 32, 128, 256])
def test_skewed_indices_returns_exact_size(prior, size):
    labels = make_entry()["labels"]
    idx = skewed_indices(labels, prior, size, seed=0)
    assert len(idx) == size


@pytest.mark.parametrize("prior", [0.5, 0.6, 0.75, 0.9])
def test_skewed_indices_realises_requested_prior(prior):
    entry = make_entry()
    idx = skewed_indices(entry["labels"], prior, 200, seed=3)
    share = float((entry["labels"][idx] == 0).mean())
    assert abs(share - prior) < 0.01


def test_skew_reaches_the_cell_that_broke_q4():
    """p=0.9 at batch 128 needs 115 minority draws from a 64-example class.

    Without replacement this is infeasible — it is the wall that capped Q4 at 7
    minority examples.  With replacement it is reachable, and the minority count
    is large enough that the quantum is 1/13 rather than 1/7.
    """
    entry = make_entry()
    idx = skewed_indices(entry["labels"], 0.9, 128, seed=0)
    minority = int((entry["labels"][idx] == 1).sum())
    assert minority == 128 - int(round(128 * 0.9))
    assert minority >= 12


def test_skewed_indices_never_collapses_to_one_class():
    entry = make_entry()
    idx = skewed_indices(entry["labels"], 0.99, 16, seed=0)
    present = set(np.unique(entry["labels"][idx]).tolist())
    assert len(present) >= 2


def test_primary_axis_is_scored_on_the_full_balanced_split():
    """offset_quality must be measured on all n rows, never on the draw."""
    entry = make_entry(n_per_class=64)
    rows = score_entry(entry, priors=(0.9,), batches=(32,), draws=2, seed=0)
    assert rows, "expected at least one grid cell"
    for r in rows:
        assert r["n_scored"] == entry["logits"].shape[0] == 128


def test_reference_levels_are_invariant_across_grid_cells():
    """If a skewed draw leaked into the baselines, these would drift."""
    entry = make_entry()
    rows = score_entry(entry, priors=(0.5, 0.7, 0.9), batches=(32, 64),
                       draws=2, seed=0)
    ref = reference_levels(entry["logits"], entry["labels"])
    assert len({round(r["R_raw"], 12) for r in rows}) == 1
    assert len({round(r["R_orc"], 12) for r in rows}) == 1
    assert rows[0]["R_raw"] == pytest.approx(ref["R_raw"])
    assert rows[0]["R_orc"] == pytest.approx(ref["R_orc"])


def test_label_free_offset_never_beats_the_oracle_on_the_same_rows():
    """The oracle reads labels and maximises this exact objective on these exact
    rows, so up to the shared coordinate-search grid it is an upper bound."""
    entry = make_entry(sep=0.4, seed=7)
    rows = score_entry(entry, priors=(0.5, 0.8), batches=(64,), draws=3, seed=1)
    for r in rows:
        assert r["offset_quality"] <= r["R_orc"] + 1e-9


def test_balanced_prior_is_not_worse_than_heavy_skew_on_average():
    """Direction check only.  A balanced draw is a representative sample of the
    scored split, so on average it should not infer a *worse* offset than a 95/5
    draw.  Tolerance is wide because on K=2 both draws can land on the same side
    of the single decision threshold, in which case they tie exactly."""
    gains = []
    for s in range(6):
        entry = make_entry(sep=0.5, seed=100 + s)
        rows = score_entry(entry, priors=(0.5, 0.95), batches=(64,), draws=4,
                           seed=s)
        by_prior = {r["prior"]: r["offset_quality"] for r in rows}
        if 0.5 in by_prior and 0.95 in by_prior:
            gains.append(by_prior[0.5] - by_prior[0.95])
    assert gains, "no comparable cells produced"
    assert float(np.mean(gains)) >= -0.02


def test_offset_quality_defined_even_when_realism_is_degenerate():
    """A draw can be too skewed to score on itself while still informing an
    offset.  The primary axis must survive that; only the secondary may be None.
    """
    entry = make_entry()
    rows = score_entry(entry, priors=(0.95,), batches=(256,), draws=2, seed=0)
    assert rows
    for r in rows:
        assert r["offset_quality"] is not None


def test_missing_class_yields_no_rows_rather_than_a_bogus_number():
    entry = make_entry()
    entry["labels"] = np.zeros_like(entry["labels"])   # class 1 absent
    assert score_entry(entry, priors=(0.7,), batches=(32,), draws=1,
                       seed=0) == []


def test_scoring_is_deterministic_under_a_fixed_seed():
    entry = make_entry()
    kw = dict(priors=(0.7,), batches=(64,), draws=3, seed=11)
    a = score_entry(entry, **kw)
    b = score_entry(entry, **kw)
    assert [r["offset_quality"] for r in a] == [r["offset_quality"] for r in b]
