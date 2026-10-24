"""Checks on the job-side scripts that need the MuJoCo environment."""

import importlib
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOTA = os.path.join(ROOT, "sota", "MujocoRL")

pytest.importorskip("stable_baselines3")
if SOTA not in sys.path:
    sys.path.insert(0, SOTA)


def test_exception_classification():
    train_rl = importlib.import_module("train_rl")
    assert train_rl.classify_exception(ValueError("Expected parameter loc to be finite, got nan")) == "nonfinite_training"
    assert train_rl.classify_exception(RuntimeError("CUDA out of memory")) == "resource_error"
    assert train_rl.classify_exception(OSError("disk full")) == "resource_error"
    assert train_rl.classify_exception(KeyError("information")) == "unclassified_runtime_error"
    assert train_rl.classify_exception(TypeError("bad kwarg")) == "unclassified_runtime_error"


def test_failure_results_keep_csv_schema_and_add_kind(tmp_path, monkeypatch):
    train_rl = importlib.import_module("train_rl")
    train_rl.write_failure_results("gX", 0.0, "boom", stats_dir=str(tmp_path), seed=3,
                                   failure_kind="fitness_spec_mismatch")
    stats = json.loads((tmp_path / "gX_seed3_stats.json").read_text())
    assert stats["failure_class"] == "infra_failure" and stats["mean_reward"] == -999999.0
    csv_path = os.path.join(SOTA, "results", "gX_seed3_results.csv")
    with open(csv_path) as f:
        header, row = f.read().splitlines()
    os.remove(csv_path)
    assert header.startswith("mean_reward,std_reward,train_time,param_count")
    assert row.startswith("-999999.0,0.0,")


def test_evaluate_model_explicit_seeds_and_records():
    import numpy as np
    from eval import evaluate_model, make_env

    class ZeroPolicy:
        def predict(self, obs, deterministic=True):
            return np.zeros(6), None

    env = make_env("Walker2d-v5")
    _, _, rewards, m = evaluate_model(ZeroPolicy(), env, max_steps=30, seeds=[11, 12])
    _, _, rewards_b, m_b = evaluate_model(ZeroPolicy(), env, num_episodes=2, max_steps=30, seed=11)
    env.close()
    assert [e["seed"] for e in m["episodes"]] == [11, 12] == m["eval_seeds"]
    assert rewards == rewards_b                       # seeds=[s, s+1] == seed=s
    assert m["eval_env_steps"] == sum(m["episode_lengths"])
    ep = m["episodes"][0]
    assert set(ep["descriptor"]) == {"foot_contact_0", "foot_contact_1"} and ep["nonfinite"] is False
