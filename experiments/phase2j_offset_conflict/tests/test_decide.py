"""判据脚本的单元测试。

decide.py 决定 GO/KILL，所以它自己必须在**已知答案**的合成输入上验证过。
这些测试构造符合/不符合各条判据的假 probe JSON，检查判定结果。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.phase2j_offset_conflict.decide import (
    cluster_bootstrap_ci,
    collect,
    spearman,
)


def test_spearman_matches_known_values():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(x, x) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_spearman_handles_ties():
    x = np.array([1.0, 1.0, 2.0, 3.0])
    y = np.array([5.0, 5.0, 6.0, 7.0])
    # 单调同序（含并列）应为 +1
    assert spearman(x, y) == pytest.approx(1.0)


def test_spearman_undefined_below_three_points():
    assert np.isnan(spearman(np.array([1.0, 2.0]), np.array([2.0, 1.0])))


def test_cluster_bootstrap_ci_excludes_zero_for_clear_positive():
    # 五个任务，每个都稳定在 +10 pp 附近
    values = {"t%d" % i: [0.10, 0.11, 0.09] for i in range(5)}
    point, lo, hi = cluster_bootstrap_ci(values, draws=2000, seed=0)
    assert point == pytest.approx(0.10, abs=0.02)
    # 这是判据 1 真正要求的：CI 下界严格大于 0。
    assert lo > 0.0
    # 各簇取值相同时中位数的自助分布会退化为常数，故只要求 hi >= lo。
    assert hi >= lo


def test_cluster_bootstrap_ci_widens_with_between_cluster_spread():
    """簇间差异越大，CI 越宽 —— 这是以任务为簇的意义所在。"""

    tight = {"t%d" % i: [0.10] for i in range(6)}
    spread = {"t%d" % i: [0.10 + 0.06 * (i - 2.5)] for i in range(6)}
    _, lo_t, hi_t = cluster_bootstrap_ci(tight, draws=3000, seed=0)
    _, lo_s, hi_s = cluster_bootstrap_ci(spread, draws=3000, seed=0)
    assert (hi_s - lo_s) > (hi_t - lo_t)


def test_cluster_bootstrap_ci_includes_zero_when_sign_varies():
    values = {"a": [0.10, 0.12], "b": [-0.11, -0.09], "c": [0.01, -0.01]}
    _, lo, hi = cluster_bootstrap_ci(values, draws=2000, seed=0)
    assert lo <= 0.0 <= hi


def test_cluster_bootstrap_needs_two_clusters():
    point, lo, hi = cluster_bootstrap_ci({"only": [0.1, 0.2]}, draws=100)
    assert point == pytest.approx(0.15)
    assert np.isnan(lo) and np.isnan(hi)


def _fake_run(tmp_path, seed, *, delta, oracle_delta, omega, retention_loss):
    """造一个最小可用的 probe JSON：两个任务，第二个是刚训练的。"""

    record = {
        "args": {"seed": seed},
        "stages": [
            {
                "trained_task": "new",
                "conflict_scorable_only": {"omega": omega},
                "tasks": {
                    "old": {
                        "scorable": True,
                        "Delta_id_balanced": delta,
                        "Delta_id_oracle_shared": oracle_delta,
                        "R_raw_balanced": 0.60,
                        "R_orc_balanced": 0.60 + delta,
                        "R_post_raw": 0.60 + retention_loss,
                    },
                    "new": {
                        "scorable": True,
                        "Delta_id_balanced": 0.0,
                        "Delta_id_oracle_shared": 0.0,
                        "R_raw_balanced": 0.9,
                        "R_orc_balanced": 0.9,
                        "R_post_raw": 0.9,
                    },
                },
            }
        ],
    }
    path = tmp_path / ("probe_seed%d.json" % seed)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_collect_excludes_the_just_trained_task(tmp_path):
    path = _fake_run(tmp_path, 1, delta=0.08, oracle_delta=0.08,
                     omega=1.0, retention_loss=0.05)
    data = collect([path])
    # 只有 "old" 被计入，"new" 是本阶段刚训练的任务
    assert list(data["delta_by_task"]) == ["old"]
    assert data["delta_by_task"]["old"] == [0.08]


def test_collect_reports_oracle_minus_fitted(tmp_path):
    path = _fake_run(tmp_path, 1, delta=0.08, oracle_delta=0.10,
                     omega=1.0, retention_loss=0.05)
    data = collect([path])
    assert data["oracle_extra"] == pytest.approx([0.02])


def test_collect_pairs_omega_with_worst_retention_loss(tmp_path):
    path = _fake_run(tmp_path, 1, delta=0.08, oracle_delta=0.08,
                     omega=1.7, retention_loss=0.05)
    data = collect([path])
    assert data["omega_points"] == [(1.7, pytest.approx(0.05))]
