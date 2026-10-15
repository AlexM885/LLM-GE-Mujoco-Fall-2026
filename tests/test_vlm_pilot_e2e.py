"""End-to-end plumbing test for the VLM aux-reward pilot pipeline.

Validates that the three VLM modules compose correctly on the data
contracts they share, without needing a real VLM or a real TD3 training
loop (which run on PACE):

    1. :func:`vlm_aux_reward.emit_aux_reward` (mock backend) writes a
       sandbox-valid aux_reward cache JSON.
    2. :func:`train_rl_vlm._load_aux_reward_cache` consumes that JSON and
       returns a callable aux_reward.
    3. :func:`vlm_aux_reward_report.render_report` consumes a set of
       per-arm per-seed stats JSONs and produces the markdown summary.

If any pair's schema drifts, this test catches it before any PACE time
is spent.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("src", "sota/MujocoRL", "analysis/scripts"):
    full = os.path.join(ROOT, p)
    if full not in sys.path:
        sys.path.insert(0, full)

import vlm_aux_reward as emit  # noqa: E402
import train_rl_vlm as trv  # noqa: E402
import vlm_aux_reward_report as rep  # noqa: E402


FAKE_TELEMETRY = {
    "episodes": [{"index": 0, "seed": 0, "return": 2400.0, "steps": 1000,
                  "terminated": False, "distance": 11.0, "mean_torso_z": 1.2,
                  "final_torso_z": 1.21, "final_pitch": 0.03,
                  "foot_contact_0": 0.3, "foot_contact_1": 0.3,
                  "mean_control_cost": 0.004, "termination_step": 1000}],
    "frames": [],
}


@pytest.fixture
def mock_evidence(monkeypatch):
    """Fake parent evidence so emit_aux_reward doesn't need captures on disk."""
    def _evidence(gene_id, max_frames):
        return {
            "gene_id": gene_id, "telemetry": FAKE_TELEMETRY,
            "frame_files": [], "frame_indices": [],
        }
    monkeypatch.setattr(emit, "_parent_evidence", _evidence)
    monkeypatch.setattr(emit, "parent_metadata",
                        lambda gid: {"genome_sha256": "fade" * 16})


def test_emit_then_load_then_call(tmp_path, mock_evidence):
    # 1. Emit a cache record (mock VLM returns a sandbox-valid snippet).
    record = emit.emit_aux_reward(
        top_gene_ids=["gA", "gB"],
        bottom_gene_ids=["gY", "gZ"],
        generation=1, best_so_far=2400.0, archive_coverage=100,
        fitness_map={"gA": 2400, "gB": 2300, "gY": 200, "gZ": 150},
        backend="mock",
    )
    cache_path = tmp_path / "aux_cache.json"
    emit.write_cache_record(record, str(cache_path))

    # 2. The trainer-side loader must accept it.
    loaded = trv._load_aux_reward_cache(str(cache_path))
    assert loaded is not None
    aux_fn, lam, anneal, meta = loaded
    assert callable(aux_fn)
    assert 0.0 <= lam <= 1.0
    assert meta["generation"] == 1

    # 3. The returned callable must run through the sandbox.
    v = aux_fn(np.zeros(17), np.zeros(6),
               {"reward_forward": 0.0, "reward_survive": 1.0,
                "reward_ctrl": 0.0, "x_position": 0.0})
    assert isinstance(v, float)


def test_report_consumes_arm_stats_shape(tmp_path):
    """Simulate 5 seeds per arm with the stats JSONs the trainer would write."""
    vlm_dir = tmp_path / "vlm_arm"
    ctrl_dir = tmp_path / "ctrl_arm"
    vlm_dir.mkdir()
    ctrl_dir.mkdir()
    rng = np.random.default_rng(0)
    # VLM arm has a +60 reward edge, +/-30 noise; control is centered at 2400
    for seed in range(5):
        base = 2400.0 + rng.normal(0, 30)
        vlm = {"seed": seed, "best_reward": base + 60.0 + rng.normal(0, 30)}
        ctrl = {"seed": seed, "best_reward": base + rng.normal(0, 30)}
        (vlm_dir / f"seed_{seed}.json").write_text(json.dumps(vlm))
        (ctrl_dir / f"seed_{seed}.json").write_text(json.dumps(ctrl))

    vlm_recs = rep._load_arm(str(vlm_dir))
    ctrl_recs = rep._load_arm(str(ctrl_dir))
    seeds, v, c = rep._align_on_seed(vlm_recs, ctrl_recs, "best_reward")
    assert seeds == [0, 1, 2, 3, 4]

    stats = rep._rliable_block(v, c)
    body = rep.render_report(
        metric="best_reward", seeds=seeds, vlm=v, ctrl=c,
        rliable_stats=stats, extra_notes="e2e plumbing")
    # Report body should mention both arms and the paired test.
    assert "VLM-aux" in body
    assert "control" in body
    assert "paired t" in body


def test_fallback_cache_still_loads_as_zero_aux(tmp_path, mock_evidence,
                                                monkeypatch):
    """A VLM-failure fallback must still produce a loader-consumable cache."""
    monkeypatch.setattr(emit, "_call_observer",
                        lambda *a, **kw: "not json at all")
    record = emit.emit_aux_reward(
        top_gene_ids=["gA"], bottom_gene_ids=["gZ"],
        backend="mock", allow_repair_retry=False,
    )
    assert record["validation"] == "fallback"
    cache_path = tmp_path / "fallback.json"
    emit.write_cache_record(record, str(cache_path))
    loaded = trv._load_aux_reward_cache(str(cache_path))
    assert loaded is not None  # zero-aux is still a valid loadable snippet
    aux_fn, lam, _, _ = loaded
    # Zero-aux snippet returns 0 regardless of input
    v = aux_fn(np.zeros(17), np.zeros(6), {})
    assert v == 0.0
