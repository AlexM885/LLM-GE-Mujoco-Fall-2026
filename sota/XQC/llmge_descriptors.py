"""MAP-Elites behaviour descriptors for a trained XQC agent (LLM-GE).

Matches sota/MujocoRL/eval.py so XQC, PPO and TD3 genes land in the same archive:
  mean_distance      x-position at episode end minus at reset
  mean_control_cost  sum over steps of -info["reward_ctrl"], divided by episode length
Per-episode sums of the two reward terms (column names as in Alex's eval.py on
Mujoco-ME-Base, so the multi-objective archive and his baselines read the same thing):
  mean_return_forward  sum of the forward-speed reward (v4 "reward_run", v5 "reward_forward")
  mean_return_ctrl     sum of info["reward_ctrl"] (<= 0; effort = -mean_return_ctrl)
Foot contact (Alex's eval.py: get_foot_geom_ids / _feet_in_contact): the share of
steps in which each foot geom touches anything. Feet are the geoms with "foot" in
their name, in geom-id order; on HalfCheetah foot_contact_0 = bfoot (back),
foot_contact_1 = ffoot (front). With action_repeat > 1 contact is read once per
policy step, after the repeated actions.
Each seed's actor is rolled out deterministically (temperature 0) for
`num_episodes` episodes, all seeds stepped together through one batched call.
"""

import numpy as np

from xqc.envs.envs import make_env


def _x_position(env, info=None):
    if isinstance(info, dict) and info.get("x_position") is not None:
        return float(info["x_position"])
    return float(env.unwrapped.data.qpos[0])


def get_foot_geom_ids(env):
    """Ids and names of the geoms with "foot" in their name, in geom-id order."""
    import mujoco
    model = env.unwrapped.model
    ids, names = [], []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name and "foot" in name.lower():
            ids.append(geom_id)
            names.append(name)
    return ids, names


def feet_in_contact(env, foot_geom_ids):
    """Bool per foot: is that foot geom touching anything this step?"""
    data = env.unwrapped.data
    touching = [False] * len(foot_geom_ids)
    for c in range(data.ncon):
        g1, g2 = data.contact[c].geom1, data.contact[c].geom2
        for i, geom_id in enumerate(foot_geom_ids):
            if g1 == geom_id or g2 == geom_id:
                touching[i] = True
    return touching


def _forward_reward(info):
    for key in ("reward_forward", "reward_run"):
        if info.get(key) is not None:
            return float(info[key])
    return 0.0


def rollout_descriptors(agent, env_name, num_seeds, num_episodes=10, seed=0, action_repeat=1):
    # Plain reward, never the VLM aux bonus: this rollout is the gene's fitness.
    envs = [make_env(env_name, seed=seed, action_repeat=action_repeat, aux_reward=False)
            for _ in range(num_seeds)]
    returns = np.zeros((num_seeds, num_episodes))
    distances = np.zeros((num_seeds, num_episodes))
    ctrl_costs = np.zeros((num_seeds, num_episodes))
    ret_forward = np.zeros((num_seeds, num_episodes))
    ret_ctrl = np.zeros((num_seeds, num_episodes))
    foot_ids, foot_names = get_foot_geom_ids(envs[0])
    contact = np.zeros((num_seeds, num_episodes, len(foot_ids)))

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
                    ret_ctrl[s, ep] += float(info["reward_ctrl"])
                ret_forward[s, ep] += _forward_reward(info)
                contact[s, ep] += feet_in_contact(env, foot_ids)
                done[s] = term or trunc
        distances[:, ep] = np.array(end_x) - np.array(start_x)
        ctrl_costs[:, ep] = ep_ctrl / np.maximum(steps, 1)
        contact[:, ep] /= np.maximum(steps, 1)[:, None]

    for env in envs:
        env.close()
    return {
        "returns": returns.tolist(),
        "distances": distances.tolist(),
        "control_costs": ctrl_costs.tolist(),
        "mean_return_per_seed": returns.mean(axis=1).tolist(),
        "mean_distance": float(distances.mean()),
        "mean_control_cost": float(ctrl_costs.mean()),
        "returns_forward": ret_forward.tolist(),
        "returns_ctrl": ret_ctrl.tolist(),
        "mean_return_forward": float(ret_forward.mean()),
        "mean_return_ctrl": float(ret_ctrl.mean()),
        "foot_geom_names": foot_names,
        "foot_contact": contact.tolist(),
        **{f"foot_contact_{i}": float(contact[:, :, i].mean()) for i in range(len(foot_ids))},
    }


def record_videos(agent, env_name, num_seeds, out_dir, steps=300, seed=0, action_repeat=1, fps=20, scale=2):
    """Roll out each seed's actor deterministically for `steps` steps and save
    out_dir/seed<s>.gif (frames downsampled by `scale`) plus returns.json with
    each seed's return over the recorded steps."""
    import os
    import json
    import imageio

    os.makedirs(out_dir, exist_ok=True)
    # Plain reward, never the VLM aux bonus: this rollout is the gene's fitness.
    envs = [make_env(env_name, seed=seed, action_repeat=action_repeat, aux_reward=False)
            for _ in range(num_seeds)]
    obs = np.stack([env.reset(seed=seed + 1000 * s)[0] for s, env in enumerate(envs)]).astype(np.float32)
    frames = [[] for _ in range(num_seeds)]
    returns = np.zeros(num_seeds)
    done = np.zeros(num_seeds, dtype=bool)
    for _ in range(steps):
        actions = np.asarray(agent.sample_actions(obs, temperature=0.0))
        for s, env in enumerate(envs):
            if done[s]:
                continue
            frames[s].append(env.render()[::scale, ::scale])
            o, r, term, trunc, _ = env.step(actions[s])
            obs[s] = o
            returns[s] += r
            done[s] = term or trunc
    for s, env in enumerate(envs):
        imageio.mimsave(os.path.join(out_dir, f"seed{s}.gif"), frames[s], duration=1000 / fps, loop=0)
        env.close()
    with open(os.path.join(out_dir, "returns.json"), "w") as f:
        json.dump({"steps": steps, "return_per_seed": returns.tolist()}, f, indent=2)
    return returns
