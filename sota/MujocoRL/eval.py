"""
Evaluate a trained PPO locomotion model (default: Walker2d-v5).

Can be used standalone:
    python eval.py --model path/to/model.zip

Or imported by train_rl.py for post-training evaluation:
    from eval import evaluate_model

Behaviour descriptors
---------------------
Following Nilsson & Cully, "Policy Gradient Assisted MAP-Elites" (GECCO '21),
the MAP-Elites behavioural descriptor is the proportion of simulation steps
that each foot spends in contact with the ground. Walker2d has two feet, so
the descriptor is 2-dimensional and each component lies in [0, 1].

Unlike the PyBullet environments used in that paper, the Gymnasium MuJoCo
environments do not expose foot contact in the observation or in ``info``, so
contacts are read directly out of the MuJoCo contact buffer each step.
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

DEFAULT_ENV_ID = os.getenv("MUJOCO_ENV_ID", "Walker2d-v5")

# Fitness specification for Walker2d. The episode fitness is the undiscounted
# return over a rollout of at most 1000 steps:
#
#     F = sum_t ( r_forward(t) + r_healthy(t) - c_ctrl(t) )
#
#     r_forward = forward_reward_weight * v_x        (v_x = dx/dt)
#     r_healthy = healthy_reward if healthy else 0
#     c_ctrl    = ctrl_cost_weight * ||a_t||^2       (sum over the 6 torques)
#
# "Healthy" means torso z in healthy_z_range and torso pitch in
# healthy_angle_range; leaving either range terminates the episode.
#
# These are the Gymnasium defaults, passed explicitly so the fitness function
# is pinned in this repo rather than inherited from whatever gymnasium version
# happens to be installed. Passing them is behaviourally a no-op today.
#
# NOTE: Walker2d-v4 does NOT satisfy this spec. It grants the healthy bonus on
# the terminating unhealthy step (measured: +1.000000 where healthy is False),
# so its returns are inflated by one healthy_reward. v5 fixes this.
WALKER_REWARD_SPEC = {
    "forward_reward_weight": 1.0,
    "ctrl_cost_weight": 0.001,
    "healthy_reward": 1.0,
    "healthy_z_range": (0.8, 2.0),
    "healthy_angle_range": (-1.0, 1.0),
    "terminate_when_unhealthy": True,
}
#: Max rollout length T. The Gymnasium time limit is also 1000.
WALKER_MAX_EPISODE_STEPS = 1000

#: Indicative return bands for Walker2d, for sanity-checking a run.
WALKER_SCORE_BANDS = (
    (0, 50, "falls immediately"),
    (50, 300, "barely survives"),
    (300, 800, "shuffling / minimal movement"),
    (800, 3000, "partial locomotion"),
    (3000, float("inf"), "sustained locomotion (SOTA)"),
)


def describe_score(total_reward):
    """Human-readable band for a return, for logs and sanity checks."""
    for lo, hi, label in WALKER_SCORE_BANDS:
        if lo <= total_reward < hi:
            return label
    return "negative / degenerate"


def make_env(env_id=None, **overrides):
    """Build the evaluation environment with the fitness spec pinned.

    The spec kwargs only apply to Walker2d; other robots do not accept the
    healthy_* parameters, so they fall back to that env's own defaults.
    """
    env_id = env_id or DEFAULT_ENV_ID
    kwargs = {}
    if env_id.startswith("Walker2d"):
        kwargs.update(WALKER_REWARD_SPEC)
    kwargs.update(overrides)
    try:
        return gym.make(env_id, **kwargs)
    except TypeError as exc:
        # An env that rejects the spec kwargs still runs, just on its defaults.
        print(f"  WARNING: {env_id} rejected reward-spec kwargs ({exc}); using env defaults")
        return gym.make(env_id)


def validate_reward_spec(env):
    """Return a list of human-readable mismatches against WALKER_REWARD_SPEC.

    Empty list means the live environment computes the documented fitness.
    """
    env_id = getattr(getattr(env, "spec", None), "id", "") or ""
    if not env_id.startswith("Walker2d"):
        return []

    problems = []
    if env_id.startswith("Walker2d-v4"):
        problems.append(
            "Walker2d-v4 grants the healthy bonus on the terminating unhealthy "
            "step, so F is inflated by one healthy_reward; use Walker2d-v5")

    u = env.unwrapped
    for attr, expected in (
        ("_forward_reward_weight", WALKER_REWARD_SPEC["forward_reward_weight"]),
        ("_ctrl_cost_weight", WALKER_REWARD_SPEC["ctrl_cost_weight"]),
        ("_healthy_reward", WALKER_REWARD_SPEC["healthy_reward"]),
        ("_healthy_z_range", WALKER_REWARD_SPEC["healthy_z_range"]),
        ("_healthy_angle_range", WALKER_REWARD_SPEC["healthy_angle_range"]),
        ("_terminate_when_unhealthy", WALKER_REWARD_SPEC["terminate_when_unhealthy"]),
    ):
        actual = getattr(u, attr, "<absent>")
        if isinstance(expected, tuple):
            matches = tuple(actual) == expected if actual != "<absent>" else False
        else:
            matches = actual == expected
        if not matches:
            problems.append(f"{attr}: expected {expected}, got {actual}")

    limit = getattr(getattr(env, "spec", None), "max_episode_steps", None)
    if limit is not None and limit != WALKER_MAX_EPISODE_STEPS:
        problems.append(
            f"max_episode_steps: expected {WALKER_MAX_EPISODE_STEPS}, got {limit}")
    return problems



def _get_x_position(env, info=None):
    if isinstance(info, dict) and info.get("x_position") is not None:
        return float(info["x_position"])
    try:
        return float(env.unwrapped.data.qpos[0])
    except (AttributeError, IndexError, TypeError):
        return None


def get_foot_geom_ids(env):
    """Geom ids of the robot's feet, ordered by geom id.

    Feet are identified by ``foot`` appearing in the geom name, which covers
    Walker2d (foot_geom, foot_left_geom), HalfCheetah (bfoot, ffoot), Hopper
    and Ant. Returns (ids, names); both empty if the model has no named feet,
    in which case the contact descriptor is simply not reported.
    """
    try:
        import mujoco
        model = env.unwrapped.model
    except (ImportError, AttributeError):
        return [], []

    ids, names = [], []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name and "foot" in name.lower():
            ids.append(geom_id)
            names.append(name)
    return ids, names


def _feet_in_contact(env, foot_geom_ids):
    """Bool per foot: is that foot geom touching anything this step?"""
    touching = [False] * len(foot_geom_ids)
    try:
        data = env.unwrapped.data
    except AttributeError:
        return touching
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        g1, g2 = contact.geom1, contact.geom2
        for i, geom_id in enumerate(foot_geom_ids):
            if g1 == geom_id or g2 == geom_id:
                touching[i] = True
    return touching


def evaluate_model(model, env, num_episodes=10, max_steps=1000):
    """
    Run evaluation episodes on a trained model.

    Parameters
    ----------
    model : stable_baselines3.PPO
        The trained PPO model.
    env : gymnasium.Env
        The environment to evaluate in.
    num_episodes : int
        Number of evaluation episodes to run.
    max_steps : int
        Maximum steps per episode.

    Returns
    -------
    mean_reward : float
        Mean total reward across episodes.
    std_reward : float
        Standard deviation of total rewards.
    rewards : list[float]
        Per-episode total rewards.
    metrics : dict
        Behaviour metrics. ``foot_contact_<i>`` are the MAP-Elites descriptors
        (proportion of steps foot i touched the ground, averaged over
        episodes); ``mean_distance`` and ``mean_control_cost`` are reported for
        analysis but are not descriptors.
    """
    foot_geom_ids, foot_geom_names = get_foot_geom_ids(env)
    if foot_geom_names:
        print(f"  Tracking foot contact for: {', '.join(foot_geom_names)}")
    else:
        print("  WARNING: no foot geoms found; foot-contact descriptors unavailable")

    rewards = []
    distances = []
    control_costs = []
    # Per-episode sums of the three fitness terms.
    returns_forward = []
    returns_healthy = []
    returns_ctrl = []
    decomposition_error = 0.0
    # Per-episode contact proportion, one list per foot.
    contact_fractions = [[] for _ in foot_geom_ids]

    for ep in range(num_episodes):
        obs, reset_info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        start_x = _get_x_position(env, reset_info)
        end_x = start_x
        ep_ctrl_cost = 0.0
        ep_contact_steps = np.zeros(len(foot_geom_ids))
        # Running sums of the three fitness terms, so F can be audited.
        ep_terms = {"forward": 0.0, "healthy": 0.0, "ctrl": 0.0}
        ep_term_error = 0.0

        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            current_x = _get_x_position(env, info)
            if current_x is not None:
                end_x = current_x
            # Squared-torque energy use. Computed from the action rather than
            # info["reward_ctrl"] so the number means the same thing on every
            # env: v5 reports reward_ctrl, v4 does not, and each env scales it
            # by its own ctrl_cost_weight (0.001 on Walker2d, 0.1 on HalfCheetah).
            ep_ctrl_cost += float(np.sum(np.square(np.asarray(action, dtype=float))))

            # Decompose the step reward into the three documented terms and
            # check they actually add up to it. reward_* keys exist on v5.
            r_fwd = info.get("reward_forward")
            r_healthy = info.get("reward_survive")
            r_ctrl = info.get("reward_ctrl")
            if r_fwd is not None and r_healthy is not None and r_ctrl is not None:
                ep_terms["forward"] += float(r_fwd)
                ep_terms["healthy"] += float(r_healthy)
                ep_terms["ctrl"] += float(r_ctrl)
                ep_term_error = max(
                    ep_term_error,
                    abs((float(r_fwd) + float(r_healthy) + float(r_ctrl)) - float(reward)))
            else:
                ep_terms = None

            if foot_geom_ids:
                ep_contact_steps += np.array(
                    _feet_in_contact(env, foot_geom_ids), dtype=float)

        distance = float(end_x - start_x) if (start_x is not None and end_x is not None) else 0.0
        mean_ctrl_cost = ep_ctrl_cost / max(step_count, 1)
        ep_contact_fraction = ep_contact_steps / max(step_count, 1)

        rewards.append(total_reward)
        distances.append(distance)
        control_costs.append(mean_ctrl_cost)
        if ep_terms is not None:
            returns_forward.append(ep_terms["forward"])
            returns_healthy.append(ep_terms["healthy"])
            returns_ctrl.append(ep_terms["ctrl"])
            decomposition_error = max(decomposition_error, ep_term_error)
        for i, fraction in enumerate(ep_contact_fraction):
            contact_fractions[i].append(float(fraction))

        contact_str = "  ".join(
            f"foot{i} = {f:5.3f}" for i, f in enumerate(ep_contact_fraction))
        terms_str = ""
        if ep_terms is not None:
            terms_str = (f"  [fwd {ep_terms['forward']:7.2f} + "
                         f"alive {ep_terms['healthy']:6.1f} "
                         f"- ctrl {-ep_terms['ctrl']:6.3f}]")
        print(
            f"  Episode {ep+1:2d}: reward = {total_reward:8.2f}  "
            f"distance = {distance:8.2f}  ctrl_cost = {mean_ctrl_cost:8.4f}  "
            f"steps = {step_count:4d}  {contact_str}{terms_str}"
        )

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    metrics = {
        "mean_distance": float(np.mean(distances)) if distances else 0.0,
        "mean_control_cost": float(np.mean(control_costs)) if control_costs else 0.0,
        "distances": distances,
        "control_costs": control_costs,
        "foot_geom_names": foot_geom_names,
        # Fitness decomposition: these three average to mean_reward.
        "mean_return_forward": float(np.mean(returns_forward)) if returns_forward else 0.0,
        "mean_return_healthy": float(np.mean(returns_healthy)) if returns_healthy else 0.0,
        "mean_return_ctrl": float(np.mean(returns_ctrl)) if returns_ctrl else 0.0,
        "reward_decomposition_error": float(decomposition_error),
        "score_band": describe_score(mean_reward),
    }
    if returns_forward:
        recombined = (metrics["mean_return_forward"]
                      + metrics["mean_return_healthy"]
                      + metrics["mean_return_ctrl"])
        if abs(recombined - mean_reward) > 1e-6:
            print(f"  WARNING: fitness decomposition mismatch: "
                  f"{recombined:.6f} vs mean_reward {mean_reward:.6f}")
    # MAP-Elites descriptors: mean proportion of steps each foot was grounded.
    for i, per_episode in enumerate(contact_fractions):
        metrics[f"foot_contact_{i}"] = float(np.mean(per_episode)) if per_episode else 0.0
        metrics[f"foot_contacts_{i}"] = per_episode
    return mean_reward, std_reward, rewards, metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO locomotion model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained model .zip file")
    parser.add_argument("--env", type=str, default=DEFAULT_ENV_ID,
                        help="Gymnasium environment id (e.g. Walker2d-v5)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum steps per episode")
    args = parser.parse_args()

    env_id = args.env

    print(f"Loading model from: {args.model}")
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        return

    env = make_env(env_id)
    for problem in validate_reward_spec(env):
        print(f"  WARNING: fitness spec mismatch -> {problem}")
    model = PPO.load(args.model, env=env)

    print(f"\n{'='*50}")
    print(f"  Evaluating {args.episodes} episodes on {env_id}")
    print(f"{'='*50}")
    mean_reward, std_reward, rewards, metrics = evaluate_model(
        model, env, num_episodes=args.episodes, max_steps=args.max_steps
    )

    print(f"\n{'='*50}")
    print(f"  RESULTS")
    print(f"{'='*50}")
    print(f"  Mean reward:    {mean_reward:8.2f}")
    print(f"  Std reward:     {std_reward:8.2f}")
    print(f"  Mean distance:  {metrics['mean_distance']:8.2f}")
    print(f"  Mean ctrl cost: {metrics['mean_control_cost']:8.4f}")
    print(f"  F = forward {metrics['mean_return_forward']:.2f} "
          f"+ alive {metrics['mean_return_healthy']:.2f} "
          f"+ ctrl {metrics['mean_return_ctrl']:.2f}")
    print(f"  Score band:     {metrics['score_band']}")
    for i, name in enumerate(metrics.get("foot_geom_names", [])):
        print(f"  Foot contact {i} ({name}): {metrics[f'foot_contact_{i}']:8.4f}")
    print(f"  Min reward:     {min(rewards):8.2f}")
    print(f"  Max reward:     {max(rewards):8.2f}")
    print(f"{'='*50}\n")

    env.close()
    print("Job Done")


if __name__ == "__main__":
    main()
