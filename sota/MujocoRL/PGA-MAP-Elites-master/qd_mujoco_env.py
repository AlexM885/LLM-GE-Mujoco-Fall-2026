'''
QD wrapper around the Gymnasium MuJoCo environments, replacing QDgym (PyBullet).

The environment, fitness and behaviour descriptor are taken from ../eval.py so
that PGA-MAP-Elites is evaluated on exactly the same task as train_rl.py:
    Fitness: undiscounted episode return of the env built by eval.make_env
    Behaviour: proportion of steps each foot spends in contact with the ground
'''

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import make_env, validate_reward_spec, get_foot_geom_ids, _feet_in_contact
from eval import DEFAULT_ENV_ID, WALKER_MAX_EPISODE_STEPS


class QDMujocoEnv(object):
    def __init__(self, env_id=DEFAULT_ENV_ID, max_episode_steps=WALKER_MAX_EPISODE_STEPS):
        self.env = make_env(env_id)
        problems = validate_reward_spec(self.env)
        if problems:
            raise RuntimeError("fitness spec mismatch: " + "; ".join(problems))
        self.foot_geom_ids, self.foot_geom_names = get_foot_geom_ids(self.env)
        if not self.foot_geom_ids:
            raise RuntimeError(f"{env_id} has no foot geoms; no behaviour descriptor available")

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._max_episode_steps = max_episode_steps

        self.T = 0
        self.tot_reward = 0.0
        self.alive = True
        self.contact_steps = np.zeros(len(self.foot_geom_ids))
        self.desc = np.zeros(len(self.foot_geom_ids))


    @property
    def np_random(self):
        return self.env.unwrapped.np_random


    def reset(self, seed=None):
        state, _ = self.env.reset(seed=seed)
        self.T = 0
        self.tot_reward = 0.0
        self.alive = True
        self.contact_steps = np.zeros(len(self.foot_geom_ids))
        self.desc = np.zeros(len(self.foot_geom_ids))
        return state


    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        self.T += 1
        self.tot_reward += float(reward)
        self.contact_steps += np.array(_feet_in_contact(self.env, self.foot_geom_ids), dtype=float)
        self.desc = self.contact_steps / self.T
        # The robot is dead if it left the healthy range (which terminates the episode)
        if terminated:
            self.alive = False
        truncated = truncated or self.T >= self._max_episode_steps
        return state, reward, terminated, truncated, info


    def close(self):
        self.env.close()
