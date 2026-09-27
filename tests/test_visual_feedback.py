"""Unit tests for sparse-feedback caching, schema, and the mock observer.

These tests do not need an LLM server, GPU, MuJoCo, or transformers. They
exercise the CPU-only plumbing that run_improved.py depends on. Run with::

    LLMGE_AUTO_START_SERVER=0 uv run pytest tests/test_visual_feedback.py -q
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT), str(ROOT / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

# Import the same module instance behavior_observer uses (``src.visual_feedback``)
# so monkeypatching config is visible to both.
import src.visual_feedback as visual_feedback
import src.behavior_observer as behavior_observer

# Use the config namespace visual_feedback resolved (real constants when the
# full environment is present, a light fallback otherwise).
C = visual_feedback.C


def _metadata(**overrides):
    base = {
        "parent_gene": "geneA",
        "genome_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "env_id": "Walker2d-v5",
        "episodes": 2,
        "max_steps": 1000,
        "seeds": [1000, 1001],
        "frames": 16,
        "camera": "track",
        "image_max_side": 336,
        "observer_model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "observer_model_revision": "main",
        "prompt_revision": "v1",
        "mode": "visual",
    }
    base.update(overrides)
    return base


def test_cache_key_deterministic_and_sensitive():
    key1 = visual_feedback.compute_cache_key(_metadata())
    key2 = visual_feedback.compute_cache_key(_metadata())
    assert key1 == key2
    assert key1 != visual_feedback.compute_cache_key(_metadata(seeds=[1, 2]))
    assert key1 != visual_feedback.compute_cache_key(_metadata(mode="telemetry"))
    assert key1 != visual_feedback.compute_cache_key(_metadata(prompt_revision="v2"))


def test_validate_observation_accepts_and_rejects():
    good = {
        "determinable": True,
        "visible_posture": "upright",
        "motion_pattern": "forward",
        "instability": "mild wobble",
        "supporting_frames": [0, 3],
        "uncertainty": "medium",
    }
    ok, reason = visual_feedback.validate_observation(good)
    assert ok, reason

    bad = dict(good)
    bad["uncertainty"] = "extreme"
    assert not visual_feedback.validate_observation(bad)[0]

    bad = dict(good)
    bad["supporting_frames"] = ["frame"]
    assert not visual_feedback.validate_observation(bad)[0]

    bad = dict(good)
    del bad["visible_posture"]
    assert not visual_feedback.validate_observation(bad)[0]


def test_format_and_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "FEEDBACK_CACHE_DIR", str(tmp_path / "cache"), raising=False)
    telemetry = {"episodes": [{
        "index": 0, "seed": 1000, "return": 123.4, "steps": 400,
        "terminated": True, "distance": 5.5, "mean_torso_z": 1.0,
        "final_torso_z": 0.7, "final_pitch": 0.9,
        "foot_contact_0": 0.5, "foot_contact_1": 0.4,
        "mean_control_cost": 0.01, "termination_step": 400,
    }]}
    record = {
        "determinable": True,
        "visible_posture": "torso pitches forward",
        "motion_pattern": "forward then collapse",
        "instability": "falls shortly before termination",
        "supporting_frames": [1],
        "uncertainty": "medium",
        "mode": "visual",
    }
    key = "k" * 64
    stored, path = visual_feedback.format_and_store(
        record, telemetry, cache_key=key, mode="visual")
    assert Path(path).exists()
    assert "Numerical telemetry" in stored["feedback_text"]
    assert "Visual observation" in stored["feedback_text"]
    assert "torso pitches forward" in stored["feedback_text"]
    assert "123.40" in stored["feedback_text"]

    loaded = visual_feedback.load_cached_observation(key)
    assert loaded is not None
    assert loaded["feedback_text"] == stored["feedback_text"]


def test_mock_observer_end_to_end(tmp_path, monkeypatch):
    gene = "geneZZZ"
    variant_dir = tmp_path / "models"
    trained_dir = tmp_path / "trained"
    capture_root = tmp_path / "captures"
    cache_root = tmp_path / "cache"
    variant_dir.mkdir()
    trained_dir.mkdir()
    (variant_dir / f"network_{gene}.py").write_text("# genome\n")

    capture_dir = capture_root / gene
    (capture_dir / "frames").mkdir(parents=True)
    (capture_dir / "frames" / "frame_0000.png").write_bytes(b"not-a-real-png")
    telemetry = {
        "episodes": [{
            "index": 0, "seed": 1000, "return": 40.0, "steps": 120,
            "terminated": True, "distance": 0.2, "mean_torso_z": 0.85,
            "final_torso_z": 0.6, "final_pitch": 1.2,
            "foot_contact_0": 0.2, "foot_contact_1": 0.1,
            "mean_control_cost": 0.02, "termination_step": 120,
        }],
        "frames": [{"index": 0, "episode": 0, "step": 0, "file": "frames/frame_0000.png"}],
    }
    manifest = {"parent_gene": gene, "episodes": 2, "max_steps": 1000,
                "camera": "track", "image_max_side": 336}
    (capture_dir / "telemetry.json").write_text(json.dumps(telemetry))
    (capture_dir / "manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(C, "VARIANT_DIR", str(variant_dir), raising=False)
    monkeypatch.setattr(C, "MUJOCO_TRAINED_MODEL_DIR", str(trained_dir), raising=False)
    monkeypatch.setattr(C, "FEEDBACK_CAPTURE_DIR", str(capture_root), raising=False)
    monkeypatch.setattr(C, "FEEDBACK_CACHE_DIR", str(cache_root), raising=False)

    record = behavior_observer.observe_parent(
        gene, backend="mock", mode="visual", capture_root=str(capture_root))
    assert record is not None
    assert record["determinable"] is True
    assert record["stats"]["backend"] == "mock"
    assert "feedback_text" in record

    resolved = visual_feedback.resolve_feedback(gene, mode="visual",
                                                cache_directory=str(cache_root))
    assert resolved is not None
    text, key, cached = resolved
    assert gene in cached["parent_gene"] or cached["parent_gene"] == gene
    assert "Numerical telemetry" in text


def test_parse_observation_json_handles_fences():
    text = 'Here you go:\n```json\n{"determinable": false, "notes": "x"}\n```'
    parsed = behavior_observer.parse_observation_json(text)
    assert parsed["determinable"] is False
    with pytest.raises(ValueError):
        behavior_observer.parse_observation_json("no json here")
