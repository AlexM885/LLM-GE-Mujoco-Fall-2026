"""Environment check: imports, JAX device, MuJoCo env step, headless render,
and one XQC agent update. Prints versions and exits non-zero on failure.

  python pace/check_env.py [--no-gpu-required] [--env HalfCheetah-v4]
"""

import argparse
import os
import platform
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-gpu-required", action="store_true")
    p.add_argument("--env", default="HalfCheetah-v4")
    args = p.parse_args()

    print(f"host={socket.gethostname()} python={platform.python_version()} exe={sys.executable}")
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')} TMPDIR={os.environ.get('TMPDIR')}")

    import flax
    import gymnasium as gym
    import hydra
    import jax
    import jax.numpy as jnp
    import mujoco
    import numpy as np
    import optax
    import tensorflow_probability

    print(f"jax={jax.__version__} flax={flax.__version__} optax={optax.__version__} "
          f"tfp={tensorflow_probability.__version__} mujoco={mujoco.__version__} "
          f"gymnasium={gym.__version__} hydra={hydra.__version__} numpy={np.__version__}")

    devices = jax.devices()
    print(f"jax backend={jax.default_backend()} devices={devices}")
    if not args.no_gpu_required and jax.default_backend() != "gpu":
        print("FAIL: GPU required but JAX has no GPU backend")
        return 1
    x = jnp.ones((2048, 2048))
    t = time.time()
    (x @ x).block_until_ready()
    print(f"matmul ok ({time.time() - t:.2f}s incl. compile)")

    import conf.register_envs  # noqa: F401
    from xqc.envs import ParallelEnv
    env = ParallelEnv([args.env], seed=0)
    obs = env.reset()
    for _ in range(100):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
    print(f"{args.env}: obs shape {obs.shape}, 100 steps ok")

    try:
        frame = env.render(num_envs=1)
        print(f"render ok ({os.environ.get('MUJOCO_GL')}): frame {frame.shape}, mean pixel {frame.mean():.1f}")
    except Exception as e:  # rendering is optional for training
        print(f"render FAILED ({os.environ.get('MUJOCO_GL')}): {type(e).__name__}: {e}")

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from xqc.agents import XQCLearner
    from xqc.replay_buffer import ParallelReplayBuffer
    conf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conf")
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose("config", overrides=[f"env={args.env}"])
    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True)
    agent = XQCLearner(seed=0, num_seeds=1, updates_per_step=2, num_interactions=1000,
                       observations=env.observation_space.sample()[0, None],
                       actions=env.action_space.sample()[0, None], **agent_cfg)
    rb = ParallelReplayBuffer(env.observation_space, env.action_space, capacity=1000, num_seeds=1, n_steps=1, gamma=0.99)
    obs = env.reset()
    for _ in range(300):
        a = env.action_space.sample()
        nobs, r, d, tr, _ = env.step(a)
        rb.insert(obs, a, r, env.generate_masks(d, tr), tr, nobs)
        obs = nobs
    t = time.time()
    agent.update(rb.sample_parallel_multibatch(256, 2), num_updates=2)
    t_compile = time.time() - t
    t = time.time()
    for _ in range(50):
        info = agent.update(rb.sample_parallel_multibatch(256, 2), num_updates=2)
    jax.block_until_ready(info)
    print(f"XQC update ok: first call {t_compile:.1f}s (compile), then {(time.time() - t) / 50 * 1000:.1f} ms per env step (2 updates)")
    print("CHECK_ENV: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
