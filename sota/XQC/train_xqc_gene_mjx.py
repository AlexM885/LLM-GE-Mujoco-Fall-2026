"""Train and evaluate one LLM-GE XQC gene on MJX; same command line and results
CSV as train_xqc_gene.py, so it can replace it in EVAL_RUNLINE or pack tests:

    python train_xqc_gene_mjx.py -network models.network_<gene_id> -env Walker2d-v5 \\
        -timesteps 100000 -num_seeds 3

Same algorithm and settings as train_xqc_gene.py -> train_xqc.py: XQC with the
gene's networks and hyperparameters, one environment per seed, the gene's
updates_per_step gradient steps per environment step after
start_training = min(5000, timesteps // 2) random steps, reward normalization,
1-step returns. What differs is where it runs:

- Walker2d-v5 is simulated by MJX (xqc/envs/walker2d_mjx.py) in float32,
  instead of CPU MuJoCo in float64.
- The whole loop (action, physics, replay insert, sampling, updates) is one
  compiled lax.scan on the GPU, CHUNK steps per call: no Python per step.
- The replay buffer is on the GPU (like replay_buffer_on_device).
- Fitness: each seed's actor plays eval_episodes episodes deterministically,
  all in parallel on MJX, with llmge_descriptors.py's descriptors. Reset noise
  comes from jax.random, so the episodes are not the CPU trainer's episodes.

Only Walker2d-v5 for now. Writes llmge/results/<gene_id>_results.csv and
llmge/stats/<gene_id>_stats.json; no run dir, actor.pkl or descriptors.json.
LLMGE_MJX_SOLVER=fast caps MJX's constraint solver (default: the XML's
settings, as Gymnasium). LLMGE_XQC_JAX_CACHE as in train_xqc_gene.py.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train_xqc_gene import FAILED, LLMGE_DIR, gene_overrides, write_results  # noqa: E402

CHUNK = 1000
ENVS = ("Walker2d-v5",)


def make_agent(config, num_seeds, timesteps, seed, obs_dim, act_dim):
    """XQCLearner as train_xqc.py builds it: conf/agent/xqc.yaml over
    base_agent.yaml, then the gene's values. Returns (agent, batch_size, utd)."""
    import numpy as np
    from omegaconf import OmegaConf
    from xqc.agents import XQCLearner

    base = OmegaConf.load(os.path.join(HERE, "conf", "agent", "base_agent.yaml"))
    xqc = OmegaConf.load(os.path.join(HERE, "conf", "agent", "xqc.yaml"))
    xqc.pop("defaults", None)
    agent_cfg = OmegaConf.to_container(OmegaConf.merge(base, xqc), resolve=True)
    config = dict(config)
    batch_size = config.pop("batch_size", 256)
    utd = config.pop("updates_per_step", 2)
    agent_cfg.update(config)
    agent = XQCLearner(
        seed=seed, num_seeds=num_seeds, updates_per_step=utd, num_interactions=timesteps,
        observations=np.zeros((1, obs_dim), np.float32), actions=np.zeros((1, act_dim), np.float32),
        **agent_cfg)
    return agent, batch_size, utd


def make_programs(agent, env, num_seeds, batch_size, utd, capacity, gamma):
    """Compiled pieces of the training loop. Carry: (i, learner, env_state, buffer,
    normalizer, key); i is the 1-based step of train_xqc.py's loop. The random-action
    phase leaves the learner out, so its program (and compile cache entry) is the
    same for every gene."""
    import jax
    import jax.numpy as jnp
    from xqc.agents.xqc.xqc_learner import _do_multiple_updates
    from xqc.envs.walker2d_mjx import ACT_DIM, OBS_DIM, where_done
    from xqc.networks import policies
    from xqc.replay_buffer import Batch

    actor_def = agent.actor.apply_fn
    vstep, vreset = jax.vmap(env.step), jax.vmap(env.reset)
    S = num_seeds

    def init_carry(seed):
        key = jax.random.PRNGKey(seed)
        key, k_reset = jax.random.split(key)
        learner = (agent.rng, agent.actor, agent.critic, agent.target_critic, agent.temperature,
                   jnp.int32(agent.step))
        buffer = {
            "obs": jnp.zeros((S, capacity, OBS_DIM), jnp.float32),
            "next_obs": jnp.zeros((S, capacity, OBS_DIM), jnp.float32),
            "actions": jnp.zeros((S, capacity, ACT_DIM), jnp.float32),
            "rewards": jnp.zeros((S, capacity), jnp.float32),
            "masks": jnp.zeros((S, capacity), jnp.float32),
        }
        # xqc.normalization.RewardNormalizer: running variance of the discounted return G.
        zeros = jnp.zeros((S,), jnp.float32)
        normalizer = {"G": zeros, "mean": zeros, "mean_sq": zeros, "var": jnp.ones((S,), jnp.float32),
                      "count": zeros}
        carry = (jnp.int32(1), learner, vreset(jax.random.split(k_reset, S)), buffer, normalizer, key)
        # Fresh buffers: the target critic starts out sharing the critic's arrays,
        # and a donated carry must not hold one buffer twice.
        return jax.tree_util.tree_map(jnp.copy, carry)

    def update_normalizer(n, rewards, done):
        G = gamma * (1.0 - done) * n["G"] + rewards
        count = n["count"] + 1.0
        delta = G - n["mean"]
        return {"G": G, "mean": n["mean"] + delta / count,
                "mean_sq": n["mean_sq"] + (G * G - n["mean_sq"]) / count,
                "var": (n["var"] * n["count"] + delta * delta * n["count"] / count) / count,
                "count": count}

    def sample(buffer, key, size, normalizer):
        # Same indices for every seed, as DeviceReplayBuffer samples them.
        idx = jax.random.randint(key, (utd, batch_size), 0, size)
        rewards = buffer["rewards"][:, idx] / (jnp.sqrt(normalizer["var"])[:, None, None] + 1e-8)
        return Batch(observations=buffer["obs"][:, idx], actions=buffer["actions"][:, idx],
                     rewards=rewards, masks=buffer["masks"][:, idx],
                     next_observations=buffer["next_obs"][:, idx],
                     discount=jnp.full((S, utd, batch_size, 1), gamma, jnp.float32))

    def step_fn(random_actions, train):
        def one_step(carry, _):
            if random_actions:
                (i, env_state, buffer, normalizer, key), learner = carry, None
            else:
                i, learner, env_state, buffer, normalizer, key = carry
                rng, actor, critic, target_critic, temperature, step = learner
            key, k_act, k_reset, k_sample = jax.random.split(key, 4)
            if random_actions:
                actions = jax.random.uniform(k_act, (S, ACT_DIM), minval=-1.0, maxval=1.0)
            else:
                rng, actions, _ = policies.sample_actions_with_log_probs(
                    rng, actor_def, actor.params, actor.batch_stats, env_state.obs, 1.0)
                actions = jnp.clip(actions, -1.0, 1.0)
            next_state, rewards, terms, truncs = vstep(env_state, actions)
            terms_f, truncs_f = terms.astype(jnp.float32), truncs.astype(jnp.float32)
            normalizer = update_normalizer(normalizer, rewards, jnp.maximum(terms_f, truncs_f))
            values = {"obs": env_state.obs, "next_obs": next_state.obs, "actions": actions,
                      "rewards": rewards, "masks": 1.0 - terms_f * (1.0 - truncs_f)}
            buffer = {k: buffer[k].at[:, i - 1].set(values[k]) for k in buffer}
            env_state = where_done(terms | truncs, vreset(jax.random.split(k_reset, S)), next_state)
            if train:
                batches = sample(buffer, k_sample, i, normalizer)
                step, rng, actor, critic, target_critic, temperature, _ = _do_multiple_updates(
                    rng, actor, critic, target_critic, temperature, batches,
                    agent.tau, agent.target_net_update_freq, agent.target_entropy, step, utd,
                    agent.use_weight_norm)
            if random_actions:
                return (i + 1, env_state, buffer, normalizer, key), None
            learner = (rng, actor, critic, target_critic, temperature, step)
            return (i + 1, learner, env_state, buffer, normalizer, key), None
        return one_step

    cache = {}

    def run(carry, length, random_actions, train):
        k = (length, random_actions, train)
        if k not in cache:
            body = step_fn(random_actions, train)
            cache[k] = jax.jit(lambda c: jax.lax.scan(body, c, None, length=length)[0], donate_argnums=0)
        if random_actions:
            i, learner, *rest = carry
            i, *rest = cache[k]((i, *rest))
            return (i, learner, *rest)
        return cache[k](carry)

    return init_carry, run


def evaluate(agent, env, mj_model, num_seeds, num_episodes, seed):
    """Deterministic episodes, all seeds x episodes at once; llmge_descriptors.py's
    descriptors (control cost per step, foot contact share of steps)."""
    import jax
    import jax.numpy as jnp
    import mujoco
    import numpy as np
    from xqc.envs.walker2d_mjx import CTRL_COST, MAX_EPISODE_STEPS
    from xqc.networks import policies

    foot_ids, foot_names = [], []
    for g in range(mj_model.ngeom):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if name and "foot" in name.lower():
            foot_ids.append(g)
            foot_names.append(name)
    foot_ids_arr = jnp.asarray(foot_ids)
    S, E = num_seeds, num_episodes
    vstep = jax.vmap(jax.vmap(env.step))
    actor_def = agent.actor.apply_fn

    def feet_in_contact(contact):
        active = contact.dist < 0  # (..., ncon)
        hit = (contact.geom1[..., None] == foot_ids_arr) | (contact.geom2[..., None] == foot_ids_arr)
        return jnp.any(hit & active[..., None], axis=-2).astype(jnp.float32)  # (..., n_feet)

    @jax.jit
    def rollout(actor, rng):
        keys = jax.random.split(jax.random.PRNGKey(seed + 42), S * E).reshape(S, E, 2)
        state = jax.vmap(jax.vmap(env.reset))(keys)
        start_x = state.data.qpos[..., 0]
        z = jnp.zeros((S, E), jnp.float32)
        acc = {"ret": z, "fwd": z, "ctrl": z, "steps": z, "end_x": start_x,
               "contact": jnp.zeros((S, E, len(foot_ids)), jnp.float32)}

        def body(c, _):
            state, alive, acc, rng = c
            rng, actions, _ = policies.sample_actions_with_log_probs(
                rng, actor_def, actor.params, actor.batch_stats, state.obs, 0.0)
            actions = jnp.clip(actions, -1.0, 1.0)
            x_before = state.data.qpos[..., 0]
            new, reward, term, trunc = vstep(state, actions)
            x_after = new.data.qpos[..., 0]
            upd = {"ret": acc["ret"] + reward, "fwd": acc["fwd"] + (x_after - x_before) / env.dt,
                   "ctrl": acc["ctrl"] - CTRL_COST * jnp.sum(actions ** 2, axis=-1),
                   "steps": acc["steps"] + 1.0, "end_x": x_after,
                   "contact": acc["contact"] + feet_in_contact(new.data.contact)}
            acc = {k: jnp.where(alive.reshape(alive.shape + (1,) * (v.ndim - 2)), v, acc[k])
                   for k, v in upd.items()}
            return (new, alive & ~(term | trunc), acc, rng), None

        (_, _, acc, _), _ = jax.lax.scan(body, (state, jnp.ones((S, E), bool), acc, rng), None,
                                         length=MAX_EPISODE_STEPS)
        return acc, start_x

    acc, start_x = jax.device_get(rollout(agent.actor, agent.rng))
    steps = np.maximum(acc["steps"], 1)
    returns, contact = acc["ret"], acc["contact"] / steps[..., None]
    return {
        "mean_return_per_seed": returns.mean(axis=1).tolist(),
        "mean_distance": float((acc["end_x"] - start_x).mean()),
        "mean_control_cost": float((-acc["ctrl"] / steps).mean()),
        "mean_return_forward": float(acc["fwd"].mean()),
        "mean_return_ctrl": float(acc["ctrl"].mean()),
        "foot_geom_names": foot_names,
        **{f"foot_contact_{i}": float(contact[:, :, i].mean()) for i in range(len(foot_ids))},
    }


def main():
    parser = argparse.ArgumentParser(description="Train an LLM-GE XQC gene on MJX")
    parser.add_argument("-network", required=True, help='Module path like "models.network_XXXX"')
    parser.add_argument("-timesteps", type=int, default=100_000)
    parser.add_argument("-num_seeds", type=int, default=3)
    parser.add_argument("-eval_episodes", type=int, default=10)
    parser.add_argument("-eval_max_steps", type=int, default=1000, help="Unused; episodes are 1000 steps")
    parser.add_argument("-env", default="Walker2d-v5")
    parser.add_argument("-seed", type=int, default=0)
    args = parser.parse_args()

    gene_id = args.network.replace("models.network_", "")
    gene_path = os.path.join(LLMGE_DIR, "models", f"network_{gene_id}.py")
    start = time.time()
    solver = os.environ.get("LLMGE_MJX_SOLVER", "default")
    stats = {"gene_id": gene_id, "algo": "xqc", "simulator": "mjx", "mjx_solver": solver, "env": args.env,
             "timesteps": args.timesteps, "num_seeds": args.num_seeds}

    def fail(message):
        train_time = time.time() - start
        stats.update(status="failed", error=message, train_time_sec=train_time, mean_reward=FAILED)
        write_results(gene_id, [FAILED, 0.0, train_time, 999999999, 0.0, 0.0, args.timesteps, args.num_seeds,
                                0.0, 0.0, 0.0, 0.0], stats)
        print(f"ERROR evaluating generated model: {message}")
        print(f"Mean reward: {FAILED}, Std: 0.0, Time: {train_time:.1f}s")
        print("Job Done")

    if args.env not in ENVS:
        return fail(f"env {args.env!r} has no MJX version yet (have: {', '.join(ENVS)})")
    try:
        _, config = gene_overrides(gene_path)
    except Exception as e:
        return fail(f"invalid gene: {e!r}")
    stats["config"] = config

    try:
        import jax
        import numpy as np
        cache = os.environ.get("LLMGE_XQC_JAX_CACHE") or os.path.join(
            os.environ.get("XQC_RUNS", os.path.join(LLMGE_DIR, "runs")), "jax_cache")
        if cache != "off":
            os.makedirs(cache, exist_ok=True)
            jax.config.update("jax_compilation_cache_dir", cache)
        import xqc.utils
        from xqc.envs.walker2d_mjx import ACT_DIM, MAX_EPISODE_STEPS, OBS_DIM, Walker2d, load_model

        np.random.seed(args.seed)
        mj_model = load_model(solver)
        env = Walker2d(mj_model)
        agent, batch_size, utd = make_agent(config, args.num_seeds, args.timesteps, args.seed, OBS_DIM, ACT_DIM)
        gamma = float(xqc.utils.get_discount(MAX_EPISODE_STEPS, 1))
        init_carry, run = make_programs(agent, env, args.num_seeds, batch_size, utd, args.timesteps, gamma)

        # train_xqc.py's loop: i < start_training random actions; updates once i > start_training.
        start_training = min(5000, args.timesteps // 2)
        phases = [(start_training - 1, True, False), (1, False, False),
                  (args.timesteps - start_training, False, True)]
        carry = init_carry(args.seed)
        timings = []
        for length, random_actions, train in phases:
            done = 0
            while done < length:
                n = min(CHUNK, length - done)
                t0 = time.time()
                carry = jax.block_until_ready(run(carry, n, random_actions, train))
                timings.append((train, n, time.time() - t0))
                done += n
        assert int(carry[0]) == args.timesteps + 1, int(carry[0])
        rng, actor, critic, target_critic, temperature, step = carry[1]
        agent.rng, agent.actor, agent.critic, agent.target_critic, agent.temperature = (
            rng, actor, critic, target_critic, temperature)
        loop_time = time.time() - start
        # The first chunk of each kind includes its compile; the rest are steady-state.
        steady = [(n, t) for k, (tr, n, t) in enumerate(timings) if tr and k > 0 and timings[k - 1][0]]
        steps_per_s = sum(n for n, _ in steady) / sum(t for _, t in steady) if steady else float("nan")
        print(f"[timing] training loop {loop_time:.1f}s, steady {steps_per_s:.0f} env steps/s "
              f"({args.num_seeds} seeds, {utd} updates/step)", flush=True)

        desc = evaluate(agent, env, mj_model, args.num_seeds, args.eval_episodes, args.seed)
        actor_params = sum(x.size for x in jax.tree_util.tree_leaves(agent.actor.params)) // args.num_seeds
    except Exception as e:
        import traceback
        print(traceback.format_exc()[-1500:].replace("Traceback", "Trace-back"))
        return fail(f"MJX training failed: {e!r}")

    per_seed = desc["mean_return_per_seed"]
    mean_reward = sum(per_seed) / len(per_seed)
    std_reward = (sum((r - mean_reward) ** 2 for r in per_seed) / len(per_seed)) ** 0.5
    train_time = time.time() - start
    row = [mean_reward, std_reward, train_time, actor_params, desc["mean_distance"],
           desc["mean_control_cost"], args.timesteps, args.num_seeds, desc["mean_return_forward"],
           desc["mean_return_ctrl"], desc["foot_contact_0"], desc["foot_contact_1"]]
    stats.update(status="ok", train_time_sec=train_time, loop_time_sec=loop_time,
                 steady_env_steps_per_sec=steps_per_s, mean_reward=mean_reward, std_reward=std_reward,
                 per_seed_return=per_seed, param_count=int(actor_params),
                 mean_distance=desc["mean_distance"], mean_control_cost=desc["mean_control_cost"],
                 mean_return_forward=desc["mean_return_forward"], mean_return_ctrl=desc["mean_return_ctrl"],
                 foot_geom_names=desc["foot_geom_names"],
                 foot_contact=[desc["foot_contact_0"], desc["foot_contact_1"]])
    write_results(gene_id, row, stats)
    print(f"Mean reward: {mean_reward}, Std: {std_reward}, Params: {actor_params}, "
          f"Distance: {desc['mean_distance']}, CtrlCost: {desc['mean_control_cost']}, Time: {train_time:.1f}s")
    print("Job Done")


if __name__ == "__main__":
    main()
