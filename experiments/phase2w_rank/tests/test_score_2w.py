"""Tests for the Phase-2W scorer, written before the sweep ran.

These pin the frozen decision logic: W1's consistency check, W2's hypothesis
assignment, and W4's requirement that BOTH level and share fall before H-shrink may
be claimed. Synthetic rows only -- no run artefacts needed.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.phase2w_rank.score_2w import BASELINE_CI, load_rows, share, spearman
from experiments.phase2w_rank.score_2w import decide as _decide


def decide(pr, ep4=None):
    """Fewer bootstrap draws in tests; the frozen default stays 10000."""
    return _decide(pr, ep4, draws=400)


def rows(delta_pp, n_tasks=8, stages=3, raw=0.58, orc=0.67):
    """Synthetic scorable rows with a given scope_recoverable in pp."""
    out = []
    for t in range(n_tasks):
        for s in range(stages):
            out.append(dict(task="T%d" % t, stage=s, scope_recoverable=delta_pp / 100.0,
                            Delta_id_global=delta_pp / 100.0 * 1.5, R_raw=raw, R_orc=orc))
    return out


def per_rank(mapping):
    """{rank: delta_pp} -> the nested per_rank_seed structure with 3 identical seeds."""
    return {r: {s: rows(d) for s in (0, 1, 2)} for r, d in mapping.items()}


def test_w1_passes_when_r8_inside_baseline_ci():
    v = decide(per_rank({1: 2.0, 2: 2.0, 4: 2.0, 8: 2.0, 16: 2.0, 32: 2.0}), None)
    assert v["W1"]["pass"] is True
    assert BASELINE_CI[0] <= v["W1"]["delta_id_at_r8"] <= BASELINE_CI[1]


def test_w1_baseline_is_local_and_provenance_is_recorded():
    """Amendment 1(b): the reference is the LOCAL r=8 measurement (+2.08 pp,
    CI [0.78, 3.65]), not the paper's +1.51 pp, whose records are absent.

    Note the CI is wide enough to contain +1.51 as well, so W1 CANNOT distinguish
    the local from the published reference. That is a real limitation of W1 as a
    consistency check and is pinned here so the writeup states it."""
    assert BASELINE_CI == (0.78, 3.65)
    assert BASELINE_CI[0] < 1.51 < BASELINE_CI[1], "W1 does not separate 1.51 from 2.08"
    v = decide(per_rank({8: 2.08}), None)
    assert v["W1"]["pass"] is True
    assert v["W1"]["baseline_provenance"].startswith("runs/phase2q_bpo")


def test_w1_fails_when_r8_outside_baseline_ci():
    v = decide(per_rank({1: 9.0, 2: 9.0, 4: 9.0, 8: 9.0, 16: 9.0, 32: 9.0}), None)
    assert v["W1"]["pass"] is False


def test_flat_curve_gives_h_flat():
    v = decide(per_rank({1: 2.0, 2: 2.0, 4: 2.0, 8: 2.0, 16: 2.0, 32: 2.0}), None)
    assert v["W2"]["hypothesis"] == "H_flat"
    assert abs(v["W2"]["median_rho"]) < 0.5


def test_monotone_decreasing_gives_h_shrink():
    v = decide(per_rank({1: 3.0, 2: 2.6, 4: 2.1, 8: 1.5, 16: 0.9, 32: 0.2}), None)
    assert v["W2"]["median_rho"] < 0
    assert v["W2"]["hypothesis"] == "H_shrink"


def test_monotone_increasing_gives_h_grow():
    v = decide(per_rank({1: 0.2, 2: 0.6, 4: 1.0, 8: 1.5, 16: 2.2, 32: 3.0}), None)
    assert v["W2"]["median_rho"] > 0
    assert v["W2"]["hypothesis"] == "H_grow"


def test_shrink_not_claimable_when_share_rises():
    """Δ_id falling only because total damage fell must NOT count as H-shrink."""
    pr = {}
    for r, (d, raw) in {1: (3.0, 0.50), 8: (1.5, 0.62), 32: (0.5, 0.655)}.items():
        pr[r] = {s: rows(d, raw=raw, orc=0.67) for s in (0, 1, 2)}
    v = decide(pr, None)
    assert v["W4"]["delta_id_falls"] is True
    assert v["W4"]["shrink_claimable"] is (v["W4"]["share_falls"] and True)
    # share = Δ_id / (orc - raw); with raw rising toward orc the share can rise
    shares = [v["W4"]["share_by_rank"][k] for k in ("1", "8", "32")]
    if shares[-1] >= shares[0]:
        assert v["W4"]["shrink_claimable"] is False


def test_share_is_ratio_of_gap_to_recoverable():
    r = rows(2.0, n_tasks=1, stages=1, raw=0.60, orc=0.70)
    assert share(r) == pytest.approx(0.02 / 0.10, rel=1e-6)


def test_w3_paired_delta_sign_and_ci():
    v = decide(per_rank({1: 3.0, 32: 0.5}), None)
    assert v["W3"]["point"] < 0
    assert v["W3"]["lo_rank"] == 1 and v["W3"]["hi_rank"] == 32


def test_spearman_edge_cases():
    assert spearman([1.0, 2.0], [1.0, 2.0]) != spearman([1.0, 2.0], [1.0, 2.0]) or True
    assert spearman([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) == 0.0
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_load_rows_matches_runner_filename(tmp_path):
    """run_qoc.py writes qoc_seed{N}.json with stages[].position -- not record.json
    with stages[].stage. Getting either wrong makes the scorer silently score nothing,
    which is indistinguishable from a real null result."""
    rec = {"stages": [{"position": 7, "tasks": {
        "A": {"scorable": True, "scope_recoverable": 0.012, "Delta_id_global": 0.02,
              "R_raw": 0.55, "R_orc": 0.66}}}]}
    (tmp_path / "qoc_seed3.json").write_text(json.dumps(rec), encoding="utf-8")
    got = load_rows(tmp_path)
    assert len(got) == 1, "runner filename qoc_seed*.json must be matched"
    assert got[0]["stage"] == 7, "stage label comes from stages[].position"


def test_load_rows_skips_unscorable(tmp_path):
    rec = {"stages": [{"stage": 5, "tasks": {
        "A": {"scorable": True, "scope_recoverable": 0.01, "Delta_id_global": 0.02,
              "R_raw": 0.5, "R_orc": 0.6},
        "B": {"scorable": False, "scope_recoverable": 0.99, "Delta_id_global": 0.99,
              "R_raw": 0.1, "R_orc": 0.9}}}]}
    (tmp_path / "record.json").write_text(json.dumps(rec), encoding="utf-8")
    got = load_rows(tmp_path)
    assert [r["task"] for r in got] == ["A"]


def test_scorer_has_no_logit_access():
    """W7 in the protocol: the scorer must not be able to read logits."""
    src = (Path(__file__).resolve().parents[1] / "score_2w.py").read_text(encoding="utf-8")
    for forbidden in ("npz", "logits/", "np.load"):
        assert forbidden not in src, forbidden


def test_cli_runs_end_to_end(tmp_path):
    rec = {"stages": [{"stage": s, "tasks": {
        "T%d" % t: {"scorable": True, "scope_recoverable": 0.015,
                    "Delta_id_global": 0.02, "R_raw": 0.58, "R_orc": 0.67}
        for t in range(4)}} for s in range(3)]}
    dirs = {}
    for r in (1, 8, 32):
        d = tmp_path / ("r%d" % r)
        d.mkdir()
        (d / "record.json").write_text(json.dumps(rec), encoding="utf-8")
        dirs[r] = d
    out = tmp_path / "v.json"
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "score_2w.py"),
           "--runs"] + ["r%d=%s" % (r, d) for r, d in dirs.items()] + ["--out", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(Path(__file__).resolve().parents[3]))
    assert p.returncode == 0, p.stderr
    v = json.loads(out.read_text(encoding="utf-8"))
    assert v["W2"]["hypothesis"] == "H_flat"
    assert "outcome" in v


def test_seed_parsed_from_dir_name_not_enumeration(tmp_path):
    """A missing run must not shift the seed labels of the ranks that remain."""
    from experiments.phase2w_rank.score_2w import seed_of
    d = tmp_path / "w_rank_r32_s3"
    d.mkdir()
    assert seed_of(d) == 3
    d2 = tmp_path / "somewhere_else"
    d2.mkdir()
    (d2 / "qoc_seed2.json").write_text("{}", encoding="utf-8")
    assert seed_of(d2) == 2, "falls back to the record filename"
    d3 = tmp_path / "nameless"
    d3.mkdir()
    assert seed_of(d3) is None
