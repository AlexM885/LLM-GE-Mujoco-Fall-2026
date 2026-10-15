"""Gymnasium wrapper that adds a VLM-authored shaping term to the step reward.

Only the training env should be wrapped. The evaluation env in
:mod:`train_rl_vlm` is a separate, unwrapped Walker2d instance so that
reported fitness is identical to Alex's me-base baseline fitness.

The wrapper is intentionally defensive about the sandboxed snippet:

- A per-call return of ``NaN`` / ``inf`` is clipped to zero (see
  :func:`aux_reward_sandbox.safe_call`), so a pathological snippet
  degrades to "no shaping" on that step rather than crashing TD3.
- An optional linear anneal lets ``lam`` decay to zero over the training
  schedule, following the STRIDE-style curriculum trick: shaping guides
  early learning and vanishes before the final policy is evaluated.
- An invocation counter tracks how often the snippet actually fired so a
  training run can be audited for whether shaping was applied.
"""

from __future__ import annotations

import gymnasium as gym

try:
    from aux_reward_sandbox import DEFAULT_CLIP, DEFAULT_TIMEOUT_SEC, safe_call
except ImportError:  # when imported as a package module from the repo root
    from sota.MujocoRL.aux_reward_sandbox import (  # type: ignore
        DEFAULT_CLIP, DEFAULT_TIMEOUT_SEC, safe_call)


class AuxRewardWrapper(gym.Wrapper):
    """Add ``lam * aux_reward(obs, action, info)`` to the step reward.

    Parameters
    ----------
    env
        Inner gymnasium env. Not modified beyond adding this wrapper.
    aux_fn
        Sandbox-loaded callable with signature ``(obs, action, info) -> float``.
    lam
        Scaling weight. Clamped into ``[-max_lam, max_lam]`` on construction;
        STRIDE/Eureka-style shaping typically uses 0.01–1.0.
    anneal_steps
        If not ``None``, ``lam`` decays linearly to zero over this many
        wrapper step() calls. Zero disables the term entirely.
    timeout_sec, clip
        Forwarded to :func:`aux_reward_sandbox.safe_call`.
    """

    #: Hard safety cap on the magnitude of ``lam``. A VLM that proposes a
    #: huge weight cannot swamp the base walker reward.
    MAX_LAM = 5.0

    def __init__(
        self,
        env: gym.Env,
        aux_fn,
        lam: float,
        *,
        anneal_steps: int | None = None,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        clip: float = DEFAULT_CLIP,
        center: bool = False,
        center_warmup: int = 1000,
        pbrs: bool = False,
        pbrs_gamma: float = 0.99,
    ):
        super().__init__(env)
        if aux_fn is None:
            raise ValueError("aux_fn must be a callable from load_aux_reward()")
        self.aux_fn = aux_fn
        self.lam_initial = float(max(-self.MAX_LAM, min(self.MAX_LAM, lam)))
        self.anneal_steps = int(anneal_steps) if anneal_steps and anneal_steps > 0 else None
        self.timeout_sec = float(timeout_sec)
        self.clip = float(clip)
        # Running-mean centering: subtract E[aux] before scaling by lam so a
        # DC offset in the raw aux_reward does not bias the shaping. The
        # mean is estimated online and frozen for the first `center_warmup`
        # steps to avoid an unstable early transient.
        self.center = bool(center)
        self.center_warmup = int(max(0, center_warmup))
        self._running_sum = 0.0
        self._running_n = 0
        # Potential-based reward shaping (Ng, Harada, Russell 1999): replace
        # ``lam * Phi(s)`` with ``lam * (gamma * Phi(s') - Phi(s))``. This
        # F term is a potential difference and provably leaves the optimal
        # policy of the base MDP unchanged (Theorem 1 of Ng 1999). The
        # standstill and survival traps we observed with plain shaping are
        # non-optimal under the base reward, so PBRS wrapping eliminates
        # them while retaining the shaping gradient near the optimum.
        self.pbrs = bool(pbrs)
        self.pbrs_gamma = float(pbrs_gamma)
        # Phi(s_t) for the previous observation. Set during reset(); None
        # outside an episode so the first step produces a zero shaping
        # contribution rather than a spurious one against stale state.
        self._prev_phi: float | None = None
        # Counters for audit; survives reset() because training spans many
        # episodes within a single wrapper lifetime.
        self.steps_taken = 0
        self.aux_total = 0.0
        self.aux_calls = 0

    @property
    def lam(self) -> float:
        """Current shaping weight after any configured anneal."""
        if self.anneal_steps is None:
            return self.lam_initial
        progress = min(self.steps_taken / self.anneal_steps, 1.0)
        return self.lam_initial * (1.0 - progress)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lam = self.lam
        if lam != 0.0:
            phi = safe_call(
                self.aux_fn, obs, action, info,
                timeout_sec=self.timeout_sec, clip=self.clip)
            # Audit uses the raw Phi(s') so aux_mean is a diagnostic for prompt
            # tuning; the shaping term applied to the reward is PBRS-wrapped
            # and/or centered when enabled.
            self.aux_total += phi
            self.aux_calls += 1
            if self.center:
                self._running_sum += phi
                self._running_n += 1
                # During the warmup, apply the raw aux (same as center=False)
                # so TD3 isn't fed near-zero shaping while the mean estimate
                # is still noisy.
                if self._running_n > self.center_warmup:
                    mean = self._running_sum / self._running_n
                    phi = phi - mean
            if self.pbrs:
                # F(s,a,s') = gamma * Phi(s') - Phi(s). First step of an
                # episode has no Phi(s_t) to subtract, so emit zero shaping
                # to avoid a one-step bias.
                prev = self._prev_phi
                if prev is None:
                    shaping = 0.0
                else:
                    shaping = self.pbrs_gamma * phi - prev
                self._prev_phi = phi
            else:
                shaping = phi
            reward = float(reward) + lam * shaping
        self.steps_taken += 1
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._prev_phi = None
        return self.env.reset(**kwargs)

    def audit(self) -> dict:
        """Summary of the shaping term's activity over this wrapper's life."""
        mean_aux = self.aux_total / self.aux_calls if self.aux_calls else 0.0
        running_mean = (self._running_sum / self._running_n
                        if self._running_n else 0.0)
        return {
            "steps_taken": self.steps_taken,
            "aux_calls": self.aux_calls,
            "aux_mean": mean_aux,
            "aux_total": self.aux_total,
            "lam_initial": self.lam_initial,
            "lam_final": self.lam,
            "anneal_steps": self.anneal_steps,
            "center": self.center,
            "running_mean": running_mean,
            "pbrs": self.pbrs,
            "pbrs_gamma": self.pbrs_gamma if self.pbrs else None,
        }
