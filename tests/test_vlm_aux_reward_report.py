"""Tests for the paired-seed VLM aux-reward report.

Rliable availability is detected at import time: when present we exercise
the full IQM + bootstrap CI path; when absent we assert the fallback body
still contains the paired / Welch tests and the per-seed table so a run
without the optional dep still produces a usable report.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis", "scripts"))

import vlm_aux_reward_report as rep  # noqa: E402


def _write_arm(tmp_path, name, records):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(records))
    return str(path)


class TestAlignment:
    def test_align_common_seeds(self):
        vlm = [{"seed": 0, "best_reward": 2400},
               {"seed": 1, "best_reward": 2500}]
        ctrl = [{"seed": 0, "best_reward": 2300},
                {"seed": 1, "best_reward": 2350},
                {"seed": 2, "best_reward": 1900}]
        seeds, v, c = rep._align_on_seed(vlm, ctrl, "best_reward")
        assert seeds == [0, 1]
        assert v.tolist() == [2400, 2500]
        assert c.tolist() == [2300, 2350]

    def test_align_empty_when_no_overlap(self):
        v = [{"seed": 0, "best_reward": 1.0}]
        c = [{"seed": 1, "best_reward": 2.0}]
        seeds, *_ = rep._align_on_seed(v, c, "best_reward")
        assert seeds == []

    def test_skips_records_missing_metric(self):
        vlm = [{"seed": 0, "best_reward": 2400},
               {"seed": 1}]
        ctrl = [{"seed": 0, "best_reward": 2200},
                {"seed": 1, "best_reward": 2300}]
        seeds, v, c = rep._align_on_seed(vlm, ctrl, "best_reward")
        assert seeds == [0]
        assert v.tolist() == [2400]
        assert c.tolist() == [2200]


class TestStats:
    def test_paired_t_signs(self):
        diffs = np.array([10.0, 15.0, 5.0, 20.0, 8.0])
        t, p = rep._paired_t(diffs)
        assert t > 0 and 0 <= p <= 1

    def test_paired_t_zero_diffs(self):
        diffs = np.zeros(5)
        t, p = rep._paired_t(diffs)
        assert np.isinf(t) and p == 0.0

    def test_cohens_d(self):
        diffs = np.array([10.0, 10.0, 10.0])
        # Zero std, infinite effect
        assert np.isinf(rep._cohens_d_paired(diffs))

    def test_welch_zero_variance(self):
        a = np.ones(5) * 3.0
        b = np.ones(5) * 1.0
        t, p = rep._welch_t(a, b)
        assert np.isinf(t) and p == 0.0


class TestLoadArm:
    def test_load_list_file(self, tmp_path):
        p = _write_arm(tmp_path, "vlm", [{"seed": 0, "best_reward": 1.0}])
        recs = rep._load_arm(p)
        assert recs == [{"seed": 0, "best_reward": 1.0}]

    def test_load_directory(self, tmp_path):
        d = tmp_path / "arm"
        d.mkdir()
        (d / "s0.json").write_text(json.dumps({"seed": 0, "best_reward": 1.0}))
        (d / "s1.json").write_text(json.dumps({"seed": 1, "best_reward": 2.0}))
        recs = rep._load_arm(str(d))
        seeds = sorted(r["seed"] for r in recs)
        assert seeds == [0, 1]

    def test_rejects_object_top_level(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"not": "a list"}))
        with pytest.raises(ValueError):
            rep._load_arm(str(p))


class TestRender:
    def test_report_contains_table_and_stats(self):
        seeds = [0, 1, 2, 3, 4]
        vlm = np.array([2400, 2500, 2600, 2700, 2800], dtype=float)
        ctrl = np.array([2300, 2350, 2400, 2500, 2600], dtype=float)
        rliable_stats = rep._rliable_block(vlm, ctrl)
        body = rep.render_report(
            metric="best_reward", seeds=seeds, vlm=vlm, ctrl=ctrl,
            rliable_stats=rliable_stats, extra_notes="pilot run")
        assert "| seed | VLM-aux | control | diff |" in body
        assert "**mean**" in body
        assert "paired t" in body
        assert "Cohen's d" in body
        assert "pilot run" in body
        if rliable_stats["available"]:
            assert "P(VLM-aux > control)" in body

    def test_report_handles_missing_rliable(self):
        seeds = [0, 1, 2]
        vlm = np.array([1.0, 2.0, 3.0])
        ctrl = np.array([0.5, 1.5, 2.5])
        body = rep.render_report(
            metric="x", seeds=seeds, vlm=vlm, ctrl=ctrl,
            rliable_stats={"available": False, "reason": "test harness"})
        assert "not run: test harness" in body
        # Fallback should still include paired/Welch tests
        assert "paired t" in body
