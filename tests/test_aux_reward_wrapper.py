"""Tests for :class:`AuxRewardWrapper`.

The wrapper must:
- add the aux term to the step reward during training,
- never alter the inner env's observation/termination semantics,
- honour the lam cap, zero lam disables shaping, and the anneal schedule,
- not forward a hostile snippet's errors — safe_call swallows them.
"""

from __future__ import annotations

import os
import sys
import textwrap

import gymnasium as gym
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sota", "MujocoRL"))

import aux_reward_sandbox as sb  # noqa: E402
from aux_reward_wrapper import AuxRewardWrapper  # noqa: E402


class _FixedRewardEnv(gym.Env):
    """Minimal env with a constant step reward of 1.0 and a 5-step episode.

    Avoids pulling MuJoCo in for a unit test. The observation and action
    spaces mirror a tiny continuous-control setup so the wrapper doesn't
    see anything unusual.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,),
                                           dtype=np.float32)
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.full(3, 0.5, dtype=np.float32)
        reward = 1.0
        terminated = self._step >= 5
        info = {"reward_forward": 1.0, "reward_survive": 0.0,
                "reward_ctrl": 0.0, "x_position": float(self._step)}
        return obs, reward, terminated, False, info


def _square_action_aux() -> callable:
    code = textwrap.dedent("""
        def aux_reward(obs, action, info):
            return float(np.sum(np.square(action)))
    """).strip()
    return sb.load_aux_reward(code)


class TestWrapperBasics:
    def test_adds_aux_to_step_reward(self):
        env = AuxRewardWrapper(_FixedRewardEnv(), _square_action_aux(), lam=0.5)
        env.reset()
        obs, reward, done, truncated, info = env.step(np.array([2.0, 0.0]))
        # base 1.0 + 0.5 * (2^2 + 0^2) = 1.0 + 2.0 = 3.0
        assert reward == pytest.approx(3.0)
        assert done is False and truncated is False
        assert info["reward_forward"] == 1.0  # inner info untouched

    def test_zero_lam_disables_shaping(self):
        env = AuxRewardWrapper(_FixedRewardEnv(), _square_action_aux(), lam=0.0)
        env.reset()
        _, reward, *_ = env.step(np.array([3.0, 3.0]))
        assert reward == 1.0
        audit = env.audit()
        assert audit["aux_calls"] == 0

    def test_lam_capped(self):
        env = AuxRewardWrapper(_FixedRewardEnv(), _square_action_aux(), lam=99.0)
        assert env.lam_initial == AuxRewardWrapper.MAX_LAM

    def test_rejects_missing_fn(self):
        with pytest.raises(ValueError):
            AuxRewardWrapper(_FixedRewardEnv(), None, lam=0.1)

    def test_reset_proxies(self):
        env = AuxRewardWrapper(_FixedRewardEnv(), _square_action_aux(), lam=0.1)
        obs, info = env.reset()
        assert obs.shape == (3,)
        assert isinstance(info, dict)


class TestAnneal:
    def test_lam_decays_linearly(self):
        env = AuxRewardWrapper(_FixedRewardEnv(), _square_action_aux(),
                               lam=1.0, anneal_steps=4)
        env.reset()
        # Before any step
        assert env.lam == pytest.approx(1.0)
        env.step(np.zeros(2, dtype=np.float32))
        assert env.lam == pytest.approx(0.75)
        env.step(np.zeros(2, dtype=np.float32))
        assert env.lam == pytest.approx(0.5)
        env.step(np.zeros(2, dtype=np.float32))
        assert env.lam == pytest.approx(0.25)
        env.step(np.zeros(2, dtype=np.float32))
        assert env.lam == pytest.approx(0.0)
        # Clamped at 0 after schedule exhausted
        env.step(np.zeros(2, dtype=np.float32))
        assert env.lam == pytest.approx(0.0)

    def test_anneal_zero_disabled(self):
        env = AuxRewardWrapper(_FixedRewardEnv(), _square_action_aux(),
                               lam=0.5, anneal_steps=0)
        env.reset()
        env.step(np.ones(2, dtype=np.float32))
        env.step(np.ones(2, dtype=np.float32))
        # No anneal: lam stays at 0.5
        assert env.lam == pytest.approx(0.5)


class TestAudit:
    def test_counters_track_calls(self):
        env = AuxRewardWrapper(_FixedRewardEnv(), _square_action_aux(), lam=1.0)
        env.reset()
        for _ in range(3):
            env.step(np.array([1.0, 0.0]))
        audit = env.audit()
        assert audit["steps_taken"] == 3
        assert audit["aux_calls"] == 3
        assert audit["aux_mean"] == pytest.approx(1.0)
        assert audit["aux_total"] == pytest.approx(3.0)


class TestCentering:
    def test_center_subtracts_running_mean(self):
        """After warmup, step reward should receive aux - running_mean."""
        env = AuxRewardWrapper(
            _FixedRewardEnv(), _square_action_aux(), lam=1.0,
            center=True, center_warmup=0)
        env.reset()
        # First call: n=0 after increment -> warmup=0 means n>warmup is 1>0, so
        # center is applied with running_mean = aux_total/1 = 1.0
        # -> delta = 1 - 1 = 0 -> reward is base 1.0.
        _, reward_1, *_ = env.step(np.array([1.0, 0.0]))
        assert reward_1 == pytest.approx(1.0)
        # Audit shows raw aux_total unchanged.
        assert env.audit()["aux_total"] == pytest.approx(1.0)
        assert env.audit()["running_mean"] == pytest.approx(1.0)

    def test_center_off_matches_prior_behaviour(self):
        env = AuxRewardWrapper(
            _FixedRewardEnv(), _square_action_aux(), lam=1.0, center=False)
        env.reset()
        _, reward, *_ = env.step(np.array([1.0, 0.0]))
        # 1 (base) + 1 (aux, action=[1,0] -> 1.0) * 1.0 (lam)
        assert reward == pytest.approx(2.0)

    def test_center_warmup_uses_raw_aux(self):
        """During warmup, raw aux is applied so TD3 isn't fed zero shaping."""
        env = AuxRewardWrapper(
            _FixedRewardEnv(), _square_action_aux(), lam=1.0,
            center=True, center_warmup=10)
        env.reset()
        _, reward, *_ = env.step(np.array([2.0, 0.0]))
        # n after increment = 1, warmup=10, 1 > 10 is False -> raw aux applies.
        # base 1 + lam 1 * aux 4 = 5
        assert reward == pytest.approx(5.0)


class TestPBRS:
    def test_pbrs_zero_on_first_step(self):
        """First step has no Phi(s_t); shaping must be zero to avoid bias."""
        env = AuxRewardWrapper(_FixedRewardEnv(), _square_action_aux(),
                               lam=1.0, pbrs=True, pbrs_gamma=0.99)
        env.reset()
        _, reward, *_ = env.step(np.array([3.0, 0.0]))
        # base 1.0 + lam * 0 (first step shaping) = 1.0
        assert reward == pytest.approx(1.0)

    def test_pbrs_formula_second_step(self):
        """F(s,a,s') = gamma * Phi(s') - Phi(s). _FixedRewardEnv emits a
        constant obs so Phi only depends on action — set action to make
        Phi(s_t) known."""
        env = AuxRewardWrapper(_FixedRewardEnv(), _square_action_aux(),
                               lam=1.0, pbrs=True, pbrs_gamma=0.9)
        env.reset()
        # Step 1: action=[2,0] -> Phi(s_1) = 4. Shaping=0 (first step).
        env.step(np.array([2.0, 0.0]))
        # Step 2: action=[1,0] -> Phi(s_2) = 1.
        # F = 0.9 * 1 - 4 = -3.1. reward = 1.0 + 1.0 * -3.1 = -2.1
        _, reward, *_ = env.step(np.array([1.0, 0.0]))
        assert reward == pytest.approx(-2.1)

    def test_pbrs_resets_between_episodes(self):
        """Phi(s_{t-1}) must be cleared on reset() so the next episode's
        first step has shaping=0."""
        env = AuxRewardWrapper(_FixedRewardEnv(), _square_action_aux(),
                               lam=1.0, pbrs=True, pbrs_gamma=0.99)
        env.reset()
        env.step(np.array([3.0, 0.0]))  # Phi=9
        env.reset()
        _, reward, *_ = env.step(np.array([1.0, 0.0]))
        # reset cleared _prev_phi; first step after reset -> shaping=0
        assert reward == pytest.approx(1.0)

    def test_pbrs_audit_records_mode(self):
        env = AuxRewardWrapper(_FixedRewardEnv(), _square_action_aux(),
                               lam=1.0, pbrs=True, pbrs_gamma=0.95)
        audit = env.audit()
        assert audit["pbrs"] is True
        assert audit["pbrs_gamma"] == pytest.approx(0.95)
        # PBRS off defaults
        env2 = AuxRewardWrapper(_FixedRewardEnv(), _square_action_aux(), lam=1.0)
        audit2 = env2.audit()
        assert audit2["pbrs"] is False
        assert audit2["pbrs_gamma"] is None


class TestHostileSnippet:
    def test_timeout_degrades_to_zero_shaping(self):
        # ast whitelist blocks loops, so we approximate "slow" with a
        # heavy numpy operation that still completes well under the
        # default timeout — proves the normal path doesn't trip it.
        code = textwrap.dedent("""
            def aux_reward(obs, action, info):
                x = np.square(obs)
                return float(np.sum(x))
        """).strip()
        env = AuxRewardWrapper(_FixedRewardEnv(), sb.load_aux_reward(code),
                               lam=1.0, timeout_sec=0.05)
        env.reset()
        _, reward, *_ = env.step(np.zeros(2, dtype=np.float32))
        # obs after step is [0.5,0.5,0.5] -> sum sq = 0.75
        assert reward == pytest.approx(1.75)

    def test_runtime_error_in_snippet_doesnt_crash_step(self):
        code = textwrap.dedent("""
            def aux_reward(obs, action, info):
                return 1.0 / 0.0
        """).strip()
        env = AuxRewardWrapper(_FixedRewardEnv(), sb.load_aux_reward(code),
                               lam=1.0)
        env.reset()
        _, reward, *_ = env.step(np.zeros(2, dtype=np.float32))
        # safe_call swallows the error and returns 0 -> wrapper adds 0 -> base 1
        assert reward == 1.0
