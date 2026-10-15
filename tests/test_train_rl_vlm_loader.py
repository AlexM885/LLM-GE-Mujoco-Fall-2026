"""Tests for the VLM aux_reward cache loader in :mod:`train_rl_vlm`.

The loader is the dirty interface between the VLM observer process and
the training process: it reads whatever JSON lands in the cache path and
must fall through to vanilla training on every kind of garbage without
raising. These tests exhaustively exercise that failure contract so the
no-feedback control arm is never derailed by a bad VLM output.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sota", "MujocoRL"))

import train_rl_vlm as trv  # noqa: E402


@pytest.fixture
def valid_cache(tmp_path):
    path = tmp_path / "aux.json"
    code = textwrap.dedent("""
        def aux_reward(obs, action, info):
            return float(np.sum(np.square(action)))
    """).strip()
    path.write_text(json.dumps({
        "aux_reward_code": code,
        "lam": 0.3,
        "anneal_steps": 20000,
        "generation": 7,
        "parent_hashes": ["a" * 8],
        "model_id": "google/gemma-3-27b-it",
        "created_at": "2026-10-14T00:00:00Z",
    }))
    return str(path)


class TestHappyPath:
    def test_returns_fn_lam_anneal_meta(self, valid_cache):
        result = trv._load_aux_reward_cache(valid_cache)
        assert result is not None
        assert callable(result["fn"])
        assert result["lam"] == pytest.approx(0.3)
        assert result["anneal_steps"] == 20000
        assert result["center"] is False
        assert result["pbrs"] is False
        meta = result["meta"]
        assert meta["generation"] == 7
        assert meta["model_id"] == "google/gemma-3-27b-it"
        assert meta["lam_source"] == "cache"
        assert meta["center"] is False
        assert meta["pbrs"] is False

    def test_fn_actually_runs_through_sandbox(self, valid_cache):
        import numpy as np
        fn = trv._load_aux_reward_cache(valid_cache)["fn"]
        # The returned fn is sandbox-loaded; call it to confirm.
        assert fn(np.zeros(10), np.array([1.0, 2.0]),
                  {"reward_forward": 0.0, "reward_survive": 1.0,
                   "reward_ctrl": 0.0, "x_position": 0.0}) == pytest.approx(5.0)


class TestFallbackPaths:
    def test_none_path_returns_none(self):
        assert trv._load_aux_reward_cache(None) is None
        assert trv._load_aux_reward_cache("") is None

    def test_missing_file_returns_none(self):
        assert trv._load_aux_reward_cache("/does/not/exist") is None

    def test_garbage_json_returns_none(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not valid json")
        assert trv._load_aux_reward_cache(str(p)) is None

    def test_missing_code_field_returns_none(self, tmp_path):
        p = tmp_path / "no_code.json"
        p.write_text(json.dumps({"lam": 0.1}))
        assert trv._load_aux_reward_cache(str(p)) is None

    def test_empty_code_returns_none(self, tmp_path):
        p = tmp_path / "empty_code.json"
        p.write_text(json.dumps({"aux_reward_code": "   "}))
        assert trv._load_aux_reward_cache(str(p)) is None

    def test_hostile_code_returns_none(self, tmp_path):
        p = tmp_path / "bad_code.json"
        p.write_text(json.dumps({"aux_reward_code": "import os\n"}))
        assert trv._load_aux_reward_cache(str(p)) is None


class TestEnvOverrides:
    def test_lam_env_overrides_cache(self, valid_cache, monkeypatch):
        monkeypatch.setenv("LLMGE_AUX_REWARD_LAM", "0.9")
        result = trv._load_aux_reward_cache(valid_cache)
        assert result is not None
        assert result["lam"] == pytest.approx(0.9)
        assert result["meta"]["lam_source"] == "env"

    def test_center_env_toggles_on(self, valid_cache, monkeypatch):
        monkeypatch.setenv("LLMGE_AUX_REWARD_CENTER", "1")
        r = trv._load_aux_reward_cache(valid_cache)
        assert r["center"] is True
        assert r["meta"]["center"] is True

    def test_center_env_off_by_default(self, valid_cache):
        assert trv._load_aux_reward_cache(valid_cache)["center"] is False

    def test_pbrs_env_toggles_on(self, valid_cache, monkeypatch):
        monkeypatch.setenv("LLMGE_AUX_REWARD_PBRS", "1")
        r = trv._load_aux_reward_cache(valid_cache)
        assert r["pbrs"] is True
        assert r["meta"]["pbrs"] is True
        assert r["pbrs_gamma"] == pytest.approx(0.99)

    def test_pbrs_gamma_override(self, valid_cache, monkeypatch):
        monkeypatch.setenv("LLMGE_AUX_REWARD_PBRS", "1")
        monkeypatch.setenv("LLMGE_AUX_REWARD_PBRS_GAMMA", "0.95")
        r = trv._load_aux_reward_cache(valid_cache)
        assert r["pbrs_gamma"] == pytest.approx(0.95)

    def test_lam_env_bad_value_uses_cache(self, valid_cache, monkeypatch):
        monkeypatch.setenv("LLMGE_AUX_REWARD_LAM", "not-a-number")
        assert trv._load_aux_reward_cache(valid_cache)["lam"] == pytest.approx(0.3)

    def test_anneal_env_override(self, valid_cache, monkeypatch):
        monkeypatch.setenv("LLMGE_AUX_REWARD_ANNEAL", "9999")
        assert trv._load_aux_reward_cache(valid_cache)["anneal_steps"] == 9999

    def test_anneal_env_zero_passed_through(self, valid_cache, monkeypatch):
        monkeypatch.setenv("LLMGE_AUX_REWARD_ANNEAL", "0")
        assert trv._load_aux_reward_cache(valid_cache)["anneal_steps"] == 0

    def test_default_lam_when_cache_missing_lam(self, tmp_path):
        p = tmp_path / "nolam.json"
        p.write_text(json.dumps({
            "aux_reward_code": "def aux_reward(obs, action, info):\n    return 0.0\n",
        }))
        r = trv._load_aux_reward_cache(str(p))
        assert r["lam"] == pytest.approx(trv.DEFAULT_LAM)
        assert r["meta"]["lam_source"] == "default"
