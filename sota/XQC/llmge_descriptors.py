"""MAP-Elites behaviour descriptors for a trained XQC agent (LLM-GE).

Matches sota/MujocoRL/eval.py so XQC, PPO and TD3 genes land in the same archive:
  mean_distance      x-position at episode end minus at reset
  mean_control_cost  sum over steps of -info["reward_ctrl"], divided by episode length
Each seed's actor is rolled out deterministically (temperature 0) for
`num_episodes` episodes, all seeds stepped together through one batched call.
"""

import numpy as np

from xqc.envs.envs import make_env


def _x_position(env, info=None):
    if isinstance(info, dict) and info.get("x_position") is not None:
        return float(info["x_position"])
    return float(env.unwrapped.data.qpos[0])


def rollout_descriptors(agent, env_name, num_seeds, num_episodes=10, seed=0, action_repeat=1):
    envs = [make_env(env_name, seed=seed, action_repeat=action_repeat) for _ in range(num_seeds)]
    returns = np.zeros((num_seeds, num_episodes))
    distances = np.zeros((num_seeds, num_episodes))
    ctrl_costs = np.zeros((num_seeds, num_episodes))

    for ep in range(num_episodes):
        obs, start_x = [], []
        for s, env in enumerate(envs):
            o, info = env.reset(seed=seed + 1000 * s + ep)
            obs.append(o)
            start_x.append(_x_position(env, info))
        obs = np.stack(obs).astype(np.float32)
        end_x = list(start_x)
        done = np.zeros(num_seeds, dtype=bool)
        steps = np.zeros(num_seeds)
        ep_ctrl = np.zeros(num_seeds)
        while not done.all():
            actions = np.asarray(agent.sample_actions(obs, temperature=0.0))
            for s, env in enumerate(envs):
                if done[s]:
                    continue
                o, r, term, trunc, info = env.step(actions[s])
                obs[s] = o
                returns[s, ep] += r
                steps[s] += 1
                end_x[s] = _x_position(env, info)
                if info.get("reward_ctrl") is not None:
                    ep_ctrl[s] += float(-info["reward_ctrl"])
                done[s] = term or trunc
        distances[:, ep] = np.array(end_x) - np.array(start_x)
        ctrl_costs[:, ep] = ep_ctrl / np.maximum(steps, 1)

    for env in envs:
        env.close()
    return {
        "returns": returns.tolist(),
        "distances": distances.tolist(),
        "control_costs": ctrl_costs.tolist(),
        "mean_return_per_seed": returns.mean(axis=1).tolist(),
        "mean_distance": float(distances.mean()),
        "mean_control_cost": float(ctrl_costs.mean()),
    }
