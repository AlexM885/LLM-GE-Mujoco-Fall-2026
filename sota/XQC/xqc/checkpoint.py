"""Checkpoint and resume for the XQC training loop (train_xqc.py).

A checkpoint captures everything the loop needs to continue as if it had never
stopped:
  - agent: actor, critic, target critic, temperature (params, batch stats,
    optimizer states, per-model step), JAX PRNG keys and the learner step
  - replay buffer contents, size and insert index
  - reward normalizer running statistics
  - Python `random` and NumPy global RNG states (the env reset seeds and the
    replay buffer sampling both draw from the NumPy global RNG)
  - the action space RNG (warm-up random actions)
  - per-env MuJoCo integration state, TimeLimit counter and env RNG, plus the
    current observations, so a resume continues mid-episode
  - loop counters and accumulated wall time

Layout under <run_dir>/checkpoints:
  step_000050000/{agent.pkl, buffer.npz, state.pkl}
  latest            text file holding the name of the newest complete checkpoint
Each checkpoint is written to a temp dir and renamed into place, then `latest`
is swapped atomically, so a job killed mid-save leaves the previous checkpoint
intact.
"""

import os
import pickle
import random
import shutil
import time

import jax
import mujoco
import numpy as np
from gymnasium.wrappers import TimeLimit

FORMAT_VERSION = 1
LATEST = "latest"
MJ_STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

BUFFER_FIELDS = ("observations", "next_observations", "actions", "rewards", "masks", "timeouts")


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

def _agent_tree(agent):
    return (agent.actor, agent.critic, agent.target_critic, agent.temperature, agent.rng, agent.step)


def agent_state(agent):
    leaves = jax.tree_util.tree_leaves(_agent_tree(agent))
    return [np.asarray(jax.device_get(x)) for x in leaves]


def load_agent_state(agent, leaves):
    template_leaves, treedef = jax.tree_util.tree_flatten(_agent_tree(agent))
    if len(template_leaves) != len(leaves):
        raise ValueError(
            f"Checkpoint has {len(leaves)} agent arrays, current agent has {len(template_leaves)}. "
            "The agent config changed since the checkpoint was written."
        )
    restored = []
    for i, (t, x) in enumerate(zip(template_leaves, leaves)):
        t_shape = np.shape(t)
        if t_shape != np.shape(x):
            raise ValueError(f"Agent array {i} shape mismatch: checkpoint {np.shape(x)} vs agent {t_shape}")
        if isinstance(t, jax.Array):
            restored.append(jax.numpy.asarray(x, dtype=t.dtype))
        else:
            restored.append(type(t)(x) if np.ndim(x) == 0 else x)
    actor, critic, target_critic, temperature, rng, step = jax.tree_util.tree_unflatten(treedef, restored)
    agent.actor, agent.critic, agent.target_critic = actor, critic, target_critic
    agent.temperature, agent.rng, agent.step = temperature, rng, step


# --------------------------------------------------------------------------- #
# Replay buffer and reward normalizer
# --------------------------------------------------------------------------- #

def save_buffer(path, rb):
    n = rb.size
    arrays = {f: getattr(rb, f)[:, :n] for f in BUFFER_FIELDS}
    np.savez(path, size=rb.size, insert_index=rb.insert_index, capacity=rb.capacity, **arrays)


def load_buffer(path, rb):
    with np.load(path) as d:
        if int(d["capacity"]) != rb.capacity:
            raise ValueError(f"Replay buffer capacity changed: {int(d['capacity'])} -> {rb.capacity}")
        n = int(d["size"])
        for f in BUFFER_FIELDS:
            getattr(rb, f)[:, :n] = d[f]
        rb.size = n
        rb.insert_index = int(d["insert_index"])


def normalizer_state(rn):
    if rn is None:
        return None
    return {"G": rn.G.copy(), "G_rms": dict(vars(rn.G_rms))}


def load_normalizer_state(rn, s):
    if rn is None or s is None:
        if (rn is None) != (s is None):
            raise ValueError("reward_normalization setting changed since the checkpoint was written")
        return
    rn.G = s["G"]
    for k, v in s["G_rms"].items():
        setattr(rn.G_rms, k, v)


# --------------------------------------------------------------------------- #
# Envs
# --------------------------------------------------------------------------- #

def _time_limit(env):
    e = env
    while True:
        if isinstance(e, TimeLimit):
            return e
        if not hasattr(e, "env"):
            return None
        e = e.env


def env_states(parallel_env):
    states = []
    for env in parallel_env.envs:
        u = env.unwrapped
        s = {"np_random": u.np_random.bit_generator.state}
        if hasattr(u, "model") and hasattr(u, "data"):
            buf = np.empty(mujoco.mj_stateSize(u.model, MJ_STATE_SPEC), dtype=np.float64)
            mujoco.mj_getState(u.model, u.data, buf, MJ_STATE_SPEC)
            s["mj_state"] = buf
        else:
            raise NotImplementedError(
                f"Mid-episode checkpointing is implemented for MuJoCo envs only, got {type(u).__name__}"
            )
        tl = _time_limit(env)
        s["elapsed_steps"] = tl._elapsed_steps if tl is not None else None
        states.append(s)
    return states


def load_env_states(parallel_env, states):
    """Call after parallel_env.reset(), so wrapper bookkeeping is initialised."""
    if len(states) != len(parallel_env.envs):
        raise ValueError("num_seeds changed since the checkpoint was written")
    for env, s in zip(parallel_env.envs, states):
        u = env.unwrapped
        u.np_random.bit_generator.state = s["np_random"]
        mujoco.mj_setState(u.model, u.data, s["mj_state"], MJ_STATE_SPEC)
        mujoco.mj_forward(u.model, u.data)
        tl = _time_limit(env)
        if tl is not None:
            tl._elapsed_steps = s["elapsed_steps"]


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #

def _ckpt_root(run_dir):
    return os.path.join(run_dir, "checkpoints")


def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _write_atomic_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save(run_dir, env_step, *, agent, replay_buffer, reward_normalizer, env, observations,
         loop_state, config, keep=2, save_replay_buffer=True):
    """Write a checkpoint for `env_step` and prune old ones. Returns its path."""
    t0 = time.time()
    root = _ckpt_root(run_dir)
    os.makedirs(root, exist_ok=True)
    name = f"step_{env_step:09d}"
    final = os.path.join(root, name)
    tmp = os.path.join(root, f".tmp_{name}_{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    with open(os.path.join(tmp, "agent.pkl"), "wb") as f:
        pickle.dump(agent_state(agent), f, protocol=pickle.HIGHEST_PROTOCOL)

    if save_replay_buffer:
        save_buffer(os.path.join(tmp, "buffer.npz"), replay_buffer)

    state = {
        "format_version": FORMAT_VERSION,
        "env_step": env_step,
        "loop": loop_state,
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "reward_normalizer": normalizer_state(reward_normalizer),
        "envs": env_states(env),
        "action_space_random": env.action_space.np_random.bit_generator.state,
        "observations": np.asarray(observations).copy(),
        "has_replay_buffer": save_replay_buffer,
        "config": config,
        "saved_at": time.time(),
    }
    with open(os.path.join(tmp, "state.pkl"), "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    for fn in os.listdir(tmp):
        with open(os.path.join(tmp, fn), "rb") as f:
            os.fsync(f.fileno())
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    _fsync_dir(root)
    _write_atomic_text(os.path.join(root, LATEST), name + "\n")

    # Prune: keep the newest `keep` complete checkpoints.
    done = sorted(d for d in os.listdir(root) if d.startswith("step_"))
    for d in done[:-keep] if keep > 0 else []:
        shutil.rmtree(os.path.join(root, d), ignore_errors=True)
    for d in os.listdir(root):
        if d.startswith(".tmp_") and d != os.path.basename(tmp):
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)

    print(f"[ckpt] saved {final} in {time.time() - t0:.1f}s", flush=True)
    return final


def list_checkpoints(run_dir):
    root = _ckpt_root(run_dir)
    if not os.path.isdir(root):
        return []
    return sorted((d for d in os.listdir(root) if d.startswith("step_")), reverse=True)


def find_latest(run_dir):
    """Newest complete checkpoint dir, preferring the `latest` pointer."""
    root = _ckpt_root(run_dir)
    candidates = list_checkpoints(run_dir)
    pointer = os.path.join(root, LATEST)
    if os.path.exists(pointer):
        name = open(pointer).read().strip()
        if name in candidates:
            candidates.remove(name)
            candidates.insert(0, name)
    for name in candidates:
        path = os.path.join(root, name)
        if os.path.exists(os.path.join(path, "state.pkl")) and os.path.exists(os.path.join(path, "agent.pkl")):
            return path
    return None


def read_state(ckpt_dir):
    with open(os.path.join(ckpt_dir, "state.pkl"), "rb") as f:
        state = pickle.load(f)
    if state.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format {state.get('format_version')} in {ckpt_dir}")
    return state


def restore(ckpt_dir, state, *, agent, replay_buffer, reward_normalizer, env):
    """Restore everything except the loop counters, which the caller reads from `state`.

    Must be called after env.reset() so the RNG states restored here are the
    ones the loop continues with.
    """
    with open(os.path.join(ckpt_dir, "agent.pkl"), "rb") as f:
        load_agent_state(agent, pickle.load(f))
    if state["has_replay_buffer"]:
        load_buffer(os.path.join(ckpt_dir, "buffer.npz"), replay_buffer)
    load_normalizer_state(reward_normalizer, state["reward_normalizer"])
    load_env_states(env, state["envs"])
    env.action_space.np_random.bit_generator.state = state["action_space_random"]
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    return state["observations"].copy()
