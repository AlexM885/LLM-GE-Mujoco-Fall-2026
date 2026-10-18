"""Go1 Push Recovery base environment.

The Go1 quadruped stands at the origin. Random horizontal force impulses are
applied to the torso at regular intervals. The robot must stay upright and
return to its original position after each push.

Success: Survived full episode AND average position error < 10cm from origin.
"""

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import go1_constants as consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=500,
        Kp=35.0,
        Kd=0.5,
        action_repeat=1,
        action_scale=0.5,
        push_interval=75,
        push_duration=5,
        push_force_min=150.0,
        push_force_max=400.0,
        reward_config=config_dict.create(
            scales=config_dict.create(
                position_recovery=8.0,
                heading_recovery=3.0,
                upright=4.0,
                velocity_settle=3.0,
                termination=-5.0,
                torques=-0.0001,
                action_rate=-0.01,
            ),
        ),
        impl="jax",
        nconmax=4 * 8192,
        njmax=40,
    )


class Go1PushRecovery(go1_base.Go1Env):
    """Stand and recover from random force perturbations."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides=None,
    ):
        super().__init__(
            xml_path=consts.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in consts.FEET_SITES]
        )
        self._standing_height = float(self._init_q[2])
        self._origin_xy = jp.array(self._init_q[0:2])
        self._origin_heading = self._quat_to_yaw(self._init_q[3:7])

    @staticmethod
    def _quat_to_yaw(quat):
        """Extract yaw angle from quaternion [w, x, y, z]."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return jp.arctan2(siny_cosp, cosy_cosp)

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=qpos[7:],
            impl=self.mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        info = {
            "rng": rng,
            "last_act": jp.zeros(self.mjx_model.nu),
            "step_count": jp.array(0),
            "push_force": jp.zeros(3),
            "cum_pos_error": jp.array(0.0),
            "push_count": jp.array(0.0),
        }

        metrics = {
            f"reward/{k}": jp.zeros(())
            for k in self._config.reward_config.scales.keys()
        }
        metrics["pos_error"] = jp.zeros(())
        metrics["heading_error"] = jp.zeros(())
        metrics["push_count"] = jp.zeros(())
        metrics["success"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        step_count = state.info["step_count"] + 1

        interval = self._config.push_interval
        duration = self._config.push_duration
        phase_in_cycle = step_count % interval
        push_active = phase_in_cycle < duration
        at_push_start = phase_in_cycle == 0

        push_rng = jax.random.fold_in(state.info["rng"], step_count)
        rng1, rng2 = jax.random.split(push_rng)

        angle = jax.random.uniform(rng1, (), minval=0.0, maxval=2.0 * jp.pi)
        force_mag = jax.random.uniform(
            rng2, (),
            minval=self._config.push_force_min,
            maxval=self._config.push_force_max,
        )
        new_force = jp.array([jp.cos(angle), jp.sin(angle), 0.1]) * force_mag

        push_force = jp.where(at_push_start, new_force, state.info["push_force"])
        applied_force = jp.where(push_active, push_force, jp.zeros(3))

        xfrc = jp.zeros((self.mjx_model.nbody, 6))
        xfrc = xfrc.at[self._torso_body_id, :3].set(applied_force)
        data_with_force = state.data.replace(xfrc_applied=xfrc)

        motor_targets = self._default_pose + action * self._config.action_scale
        data = mjx_env.step(
            self.mjx_model, data_with_force, motor_targets, self.n_substeps
        )

        push_count = state.info["push_count"] + at_push_start.astype(float)

        pos_xy = data.qpos[0:2]
        pos_error = jp.linalg.norm(pos_xy - self._origin_xy)
        cum_pos_error = state.info["cum_pos_error"] + pos_error

        info = {
            **state.info,
            "step_count": step_count,
            "push_force": push_force,
            "push_count": push_count,
            "cum_pos_error": cum_pos_error,
            "last_act": action,
        }

        obs = self._get_obs(data, info)
        done = self._get_termination(data)

        rewards = self._get_reward(data, action, info, done)
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        avg_pos_error = cum_pos_error / jp.maximum(step_count.astype(float), 1.0)
        heading = self._quat_to_yaw(data.qpos[3:7])
        heading_error = jp.abs(heading - self._origin_heading)
        heading_error = jp.minimum(heading_error, 2.0 * jp.pi - heading_error)

        success = (~done.astype(bool)) & (avg_pos_error < 0.10)

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["pos_error"] = pos_error
        state.metrics["heading_error"] = heading_error
        state.metrics["push_count"] = push_count
        state.metrics["success"] = success.astype(float)

        done = done.astype(reward.dtype)
        return state.replace(data=data, obs=obs, reward=reward, done=done, info=info)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        up = self.get_upvector(data)
        return up[-1] < 0.3

    def _get_obs(self, data: mjx.Data, info: dict) -> dict:
        gyro = self.get_gyro(data)
        gravity = self.get_gravity(data)
        linvel = self.get_local_linvel(data)
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]

        pos_xy = data.qpos[0:2] - self._origin_xy

        heading = self._quat_to_yaw(data.qpos[3:7])
        heading_error = heading - self._origin_heading

        state = jp.hstack([
            linvel,                              # 3: local linear velocity
            gyro,                                # 3: angular velocity
            gravity,                             # 3: gravity in body frame
            joint_angles - self._default_pose,   # 12: joint angle offsets
            joint_vel,                           # 12: joint velocities
            info["last_act"],                    # 12: previous action
            pos_xy,                              # 2: XY offset from origin
            jp.array([data.qpos[2]]),            # 1: COM height
            jp.array([heading_error]),           # 1: heading error
            info["push_force"] / 100.0,          # 3: push force (normalized)
        ])

        return {"state": state}

    def _get_reward(self, data, action, info, done) -> dict:
        up = self.get_upvector(data)
        linvel = self.get_local_linvel(data)

        pos_xy = data.qpos[0:2]
        pos_error = jp.linalg.norm(pos_xy - self._origin_xy)
        position_recovery = 1.0 - jp.tanh(5.0 * pos_error)

        heading = self._quat_to_yaw(data.qpos[3:7])
        heading_error = jp.abs(heading - self._origin_heading)
        heading_error = jp.minimum(heading_error, 2.0 * jp.pi - heading_error)
        heading_recovery = 1.0 - jp.tanh(3.0 * heading_error)

        upright = jp.clip(up[-1], 0.0, 1.0)

        vel_magnitude = jp.linalg.norm(linvel)
        velocity_settle = 1.0 - jp.tanh(2.0 * vel_magnitude)

        termination = done.astype(float)
        torques = jp.sum(jp.square(data.actuator_force))
        action_rate = jp.sum(jp.square(action - info["last_act"]))

        return {
            "position_recovery": position_recovery,
            "heading_recovery": heading_recovery,
            "upright": upright,
            "velocity_settle": velocity_settle,
            "termination": termination,
            "torques": torques,
            "action_rate": action_rate,
        }

    @property
    def observation_size(self):
        return {"state": 52}

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu
