"""Tests for the per-generation VLM aux_reward emitter.

These tests do not require a GPU or a real VLM: they use the ``mock``
backend and monkeypatch :func:`_parent_evidence` so a parent's rollout
evidence is provided inline without needing the full
``capture_rollout``->``behavior_observer`` pipeline on disk.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("sota/MujocoRL", "src"):
    full = os.path.join(ROOT, p)
    if full not in sys.path:
        sys.path.insert(0, full)

import vlm_aux_reward as m  # noqa: E402


FAKE_TELEMETRY = {
    "episodes": [
        {"index": 0, "seed": 1000, "return": 2400.0, "steps": 1000,
         "terminated": False, "distance": 11.0, "mean_torso_z": 1.2,
         "final_torso_z": 1.21, "final_pitch": 0.03, "foot_contact_0": 0.3,
         "foot_contact_1": 0.3, "mean_control_cost": 0.004,
         "termination_step": 1000},
    ],
    "frames": [],
}


@pytest.fixture
def evidence_patch(monkeypatch, tmp_path):
    """Provide fake parent evidence so load_capture isn't called."""
    def fake(gene_id: str, max_frames: int, capture_root=None):
        return {
            "gene_id": gene_id,
            "telemetry": FAKE_TELEMETRY,
            "frame_files": [],
            "frame_indices": [],
        }
    monkeypatch.setattr(m, "_parent_evidence", fake)
    # Also stub parent_metadata to avoid filesystem lookups.
    monkeypatch.setattr(m, "parent_metadata",
                        lambda gid: {"genome_sha256": "cafe" * 16})
    return tmp_path


# --------------------------------------------------------------------------- #
# Prompt + response plumbing
# --------------------------------------------------------------------------- #
class TestPrompt:
    def test_template_loads(self):
        tpl = m.load_prompt_template("emit")
        assert "<<ROLE_PREAMBLE>>" in tpl
        assert "<<TOP_PARENTS>>" in tpl
        assert "<<BOTTOM_PARENTS>>" in tpl

    def test_build_user_text_substitutes_all_tokens(self):
        tpl = m.load_prompt_template("emit")
        text = m.build_user_text(
            tpl,
            top_parents=[{"gene_id": "gA", "telemetry": FAKE_TELEMETRY,
                          "fitness": 2400.0}],
            bottom_parents=[{"gene_id": "gZ", "telemetry": FAKE_TELEMETRY,
                             "fitness": 200.0}],
            generation=3, best_so_far=2400.0, archive_coverage=120,
            reflection=None,
        )
        for token in ("<<ROLE_PREAMBLE>>", "<<GEN_CONTEXT>>", "<<TOP_PARENTS>>",
                      "<<BOTTOM_PARENTS>>", "<<REFLECTION_BLOCK>>"):
            assert token not in text, f"unsubstituted token: {token}"
        assert "gA (eval fitness 2400.0)" in text
        assert "gZ (eval fitness 200.0)" in text
        assert "generation: 3" in text

    def test_reflection_included_when_provided(self):
        tpl = m.load_prompt_template("emit")
        text = m.build_user_text(
            tpl,
            top_parents=[{"gene_id": "gA", "telemetry": FAKE_TELEMETRY}],
            bottom_parents=[],
            generation=4, best_so_far=None, archive_coverage=None,
            reflection={
                "aux_reward_code": "def aux_reward(obs, action, info):\n    return 0.0\n",
                "lam": 0.1, "rationale": "prior rationale",
                "impact": {"mean_child_reward": 1500.0, "delta_vs_control": 25.0},
            },
        )
        assert "PRIOR AUX REWARD" in text
        assert "prior rationale" in text
        assert "delta vs control baseline 25.0" in text


class TestParsing:
    def test_parses_clean_json(self):
        resp = json.dumps({
            "aux_reward_code": "def aux_reward(obs, action, info):\n    return 0.0\n",
            "lam": 0.2, "anneal_steps": 10000, "rationale": "x",
        })
        parsed = m.parse_vlm_aux_response(resp)
        assert parsed["lam"] == pytest.approx(0.2)
        assert parsed["anneal_steps"] == 10000

    def test_strips_markdown_fences(self):
        resp = ("```json\n" + json.dumps({
            "aux_reward_code": "```python\ndef aux_reward(obs, action, info):\n    return 1.0\n```\n",
            "lam": 0.1, "rationale": "y",
        }) + "\n```")
        parsed = m.parse_vlm_aux_response(resp)
        # Code fence inside the string was stripped by _normalize_code
        assert parsed["aux_reward_code"].startswith("def aux_reward")
        assert "```" not in parsed["aux_reward_code"]

    def test_rejects_missing_fields(self):
        with pytest.raises(ValueError):
            m.parse_vlm_aux_response(json.dumps({"lam": 0.1}))

    def test_rejects_non_string_code(self):
        with pytest.raises(ValueError):
            m.parse_vlm_aux_response(json.dumps({
                "aux_reward_code": 42, "lam": 0.1, "rationale": "x",
            }))

    def test_lam_clamped_to_unit_range(self):
        resp = json.dumps({
            "aux_reward_code": "def aux_reward(obs, action, info):\n    return 0.0\n",
            "lam": 99.0, "rationale": "x",
        })
        parsed = m.parse_vlm_aux_response(resp)
        assert parsed["lam"] == 1.0

    def test_bad_lam_rejected(self):
        with pytest.raises(ValueError):
            m.parse_vlm_aux_response(json.dumps({
                "aux_reward_code": "def aux_reward(obs, action, info):\n    return 0.0\n",
                "lam": "not-a-float", "rationale": "x",
            }))


# --------------------------------------------------------------------------- #
# End-to-end emission (mock backend)
# --------------------------------------------------------------------------- #
class TestEmit:
    def test_happy_path_mock(self, evidence_patch):
        rec = m.emit_aux_reward(
            top_gene_ids=["gA", "gB"],
            bottom_gene_ids=["gZ", "gY"],
            generation=5, best_so_far=2400.0, archive_coverage=120,
            fitness_map={"gA": 2400, "gB": 2300, "gZ": 300, "gY": 150},
            backend="mock",
        )
        assert rec["schema"] == m.AUX_REWARD_SCHEMA
        assert rec["validation"] == "ok"
        assert rec["fallback_reason"] is None
        assert "def aux_reward" in rec["aux_reward_code"]
        # Sandbox already validated by emit_aux_reward
        assert len(rec["parent_hashes"]) == 4
        assert rec["generation"] == 5

    def test_record_written_to_disk(self, evidence_patch, tmp_path):
        rec = m.emit_aux_reward(
            top_gene_ids=["gA"], bottom_gene_ids=["gZ"], backend="mock",
        )
        out = tmp_path / "cache.json"
        m.write_cache_record(rec, str(out))
        on_disk = json.loads(out.read_text())
        assert on_disk["aux_reward_code"] == rec["aux_reward_code"]

    def test_empty_parent_set_falls_back(self, monkeypatch):
        # No evidence from any gene -> no_parent_captures fallback.
        monkeypatch.setattr(m, "_parent_evidence",
                            lambda gid, mf, capture_root=None: None)
        rec = m.emit_aux_reward(
            top_gene_ids=["gA"], bottom_gene_ids=["gZ"], backend="mock",
        )
        assert rec["validation"] == "fallback"
        assert "no parent captures" in rec["fallback_reason"]
        assert rec["aux_reward_code"] == m.ZERO_AUX_CODE
        assert rec["lam"] == 0.0

    def test_vlm_returns_garbage_falls_back(self, evidence_patch, monkeypatch):
        monkeypatch.setattr(m, "_call_observer",
                            lambda *a, **kw: "this is not json at all")
        rec = m.emit_aux_reward(
            top_gene_ids=["gA"], bottom_gene_ids=["gZ"], backend="mock",
            allow_repair_retry=False,
        )
        assert rec["validation"] == "fallback"
        assert rec["aux_reward_code"] == m.ZERO_AUX_CODE

    def test_vlm_returns_hostile_code_fails_sandbox_then_fallback(
            self, evidence_patch, monkeypatch):
        bad_response = json.dumps({
            "aux_reward_code": "import os\n",
            "lam": 0.1, "rationale": "evil",
        })
        monkeypatch.setattr(m, "_call_observer", lambda *a, **kw: bad_response)
        rec = m.emit_aux_reward(
            top_gene_ids=["gA"], bottom_gene_ids=["gZ"], backend="mock",
            allow_repair_retry=False,
        )
        assert rec["validation"] == "fallback"
        assert "sandbox rejected" in rec["fallback_reason"]

    def test_repair_retry_succeeds(self, evidence_patch, monkeypatch):
        bad = json.dumps({
            "aux_reward_code": "import os\n",
            "lam": 0.1, "rationale": "evil",
        })
        good = json.dumps({
            "aux_reward_code": (
                "def aux_reward(obs, action, info):\n"
                "    return float(np.sum(np.square(action)))\n"
            ),
            "lam": 0.2, "rationale": "repaired",
        })
        call_sequence = iter([bad, good])
        monkeypatch.setattr(m, "_call_observer",
                            lambda *a, **kw: next(call_sequence))
        rec = m.emit_aux_reward(
            top_gene_ids=["gA"], bottom_gene_ids=["gZ"], backend="mock",
        )
        assert rec["validation"] == "repaired"
        assert rec["attempts"] == 2
        assert "repaired" in rec["rationale"]

    def test_repair_retry_still_bad_falls_back(self, evidence_patch, monkeypatch):
        bad = json.dumps({
            "aux_reward_code": "import os\n",
            "lam": 0.1, "rationale": "evil",
        })
        monkeypatch.setattr(m, "_call_observer", lambda *a, **kw: bad)
        rec = m.emit_aux_reward(
            top_gene_ids=["gA"], bottom_gene_ids=["gZ"], backend="mock",
        )
        assert rec["validation"] == "fallback"
        assert rec["attempts"] == 2


class TestCacheWriter:
    def test_atomic_write(self, tmp_path):
        out = tmp_path / "sub" / "cache.json"
        m.write_cache_record({"schema": m.AUX_REWARD_SCHEMA, "value": 1}, str(out))
        assert out.exists()
        assert json.loads(out.read_text())["value"] == 1
        # Rewriting leaves a single file (no .tmp residue)
        m.write_cache_record({"schema": m.AUX_REWARD_SCHEMA, "value": 2}, str(out))
        assert json.loads(out.read_text())["value"] == 2
        assert not (out.parent / "cache.json.tmp").exists()
