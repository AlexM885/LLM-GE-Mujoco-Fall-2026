"""Walker2d-v5 on MJX, matching Gymnasium's Walker2d-v5 (gymnasium 1.0.0):

  model        gymnasium's own walker2d_v5.xml, frame_skip 4 (dt = 0.008 s)
  observation  qpos[1:] (8) + clip(qvel, -10, 10) (9) = 17 floats
  reward       forward velocity + 1.0 while healthy - 1e-3 * |action|^2
  termination  unhealthy: z outside (0.8, 2.0) or torso angle outside (-1, 1)
  truncation   1000 policy steps
  reset        qpos0 and zero qvel, each plus U(-5e-3, 5e-3) noise

Everything is a pure JAX function of (state, action), so it vmaps over envs
and runs inside jit / lax.scan with the learner.
"""

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

FRAME_SKIP = 4
MAX_EPISODE_STEPS = 1000
OBS_DIM = 17
ACT_DIM = 6
RESET_NOISE = 5e-3
HEALTHY_Z = (0.8, 2.0)
HEALTHY_ANGLE = (-1.0, 1.0)
CTRL_COST = 1e-3


def model_path():
    import gymnasium.envs.mujoco as gym_mujoco
    return os.path.join(os.path.dirname(gym_mujoco.__file__), "assets", "walker2d_v5.xml")


def load_model(solver="default"):
    """The MjModel. solver="fast" caps the constraint solver at 4 iterations and
    8 line-search steps (MJX's usual speed setting; Gymnasium uses the XML's)."""
    m = mujoco.MjModel.from_xml_path(model_path())
    if solver == "fast":
        m.opt.iterations = 4
        m.opt.ls_iterations = 8
    elif solver != "default":
        raise ValueError(f"unknown solver {solver!r}")
    return m


class EnvState(NamedTuple):
    data: mjx.Data
    obs: jnp.ndarray
    t: jnp.ndarray  # policy steps since reset


class Walker2d:
    def __init__(self, mj_model):
        self.mj_model = mj_model
        self.mx = mjx.put_model(mj_model)
        self.data0 = mjx.make_data(mj_model)
        self.qpos0 = jnp.asarray(mj_model.qpos0, jnp.float32)
        self.dt = float(mj_model.opt.timestep) * FRAME_SKIP

    @staticmethod
    def _obs(data):
        return jnp.concatenate([data.qpos[1:], jnp.clip(data.qvel, -10.0, 10.0)]).astype(jnp.float32)

    def reset(self, key):
        k_pos, k_vel = jax.random.split(key)
        nq, nv = self.qpos0.shape[0], self.data0.qvel.shape[0]
        qpos = self.qpos0 + jax.random.uniform(k_pos, (nq,), minval=-RESET_NOISE, maxval=RESET_NOISE)
        qvel = jax.random.uniform(k_vel, (nv,), minval=-RESET_NOISE, maxval=RESET_NOISE)
        data = mjx.forward(self.mx, self.data0.replace(qpos=qpos, qvel=qvel))
        return EnvState(data, self._obs(data), jnp.int32(0))

    def step(self, state, action):
        """One policy step. Returns (state, reward, terminated, truncated)."""
        x_before = state.data.qpos[0]
        data = state.data.replace(ctrl=action)
        data = jax.lax.fori_loop(0, FRAME_SKIP, lambda _, d: mjx.step(self.mx, d), data)
        z, angle = data.qpos[1], data.qpos[2]
        healthy = ((z > HEALTHY_Z[0]) & (z < HEALTHY_Z[1])
                   & (angle > HEALTHY_ANGLE[0]) & (angle < HEALTHY_ANGLE[1]))
        x_velocity = (data.qpos[0] - x_before) / self.dt
        reward = x_velocity + healthy.astype(jnp.float32) - CTRL_COST * jnp.sum(jnp.square(action))
        t = state.t + 1
        terminated = ~healthy
        truncated = (t >= MAX_EPISODE_STEPS) & healthy
        return EnvState(data, self._obs(data), t), reward, terminated, truncated


def where_done(done, reset_state, state):
    """Per env: reset_state where done, else state. done has the envs' batch shape."""
    def pick(a, b):
        d = done.reshape(done.shape + (1,) * (a.ndim - done.ndim))
        return jnp.where(d, a, b)
    return jax.tree_util.tree_map(pick, reset_state, state)
