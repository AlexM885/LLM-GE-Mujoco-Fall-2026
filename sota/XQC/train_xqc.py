"""Resumable XQC training entry point.

Same training loop as upstream train_parallel.py, plus:
  - auto-resume from the newest checkpoint in `run_dir`
  - periodic checkpoints every `checkpoint_interval` env steps
  - checkpoint and clean exit on SIGUSR1 / SIGTERM (Slurm pre-timeout signal,
    scancel, preemption) or when `max_runtime_s` is reached
  - per-run results written to `run_dir` in a fixed format:
      eval.csv        env, seed, env_step, return, wall_time_s, slurm_job_id
      train_metrics.jsonl
      summary.json    written/updated at every eval
      DONE            created once max_steps is reached

Exit codes: 0 = finished or checkpointed on request, anything else = failure.

Usage (Hydra overrides):
  python train_xqc.py env=HalfCheetah-v4 seed=0 num_seeds=1 max_steps=1000000 \
      run_dir=/path/to/run checkpoint_interval=50000
"""

import csv
import json
import os
import random
import signal
import sys
import time
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf

import conf.register_envs  # noqa: F401, registers the env config group

# Changing these between jobs of one run is safe. Everything else must match
# the checkpoint, otherwise the resume would not be faithful.
RESUME_MUTABLE_KEYS = {
    "run_dir", "checkpoint_interval", "keep_checkpoints", "max_runtime_s", "resume",
    "eval_interval", "eval_episodes", "log_interval", "log_interval_condition_number",
    "eval_video", "eval_video_fps", "eval_video_frameskip", "eval_video_num_envs", "wandb",
}

EVAL_FIELDS = ["env", "seed", "env_step", "return", "wall_time_s", "slurm_job_id"]


class StopFlag:
    def __init__(self):
        self.signum = None

    def __call__(self, signum, frame):
        if self.signum is None:
            print(f"[signal] received {signal.Signals(signum).name}, will checkpoint and exit", flush=True)
        self.signum = signum


def _identity_config(cfg_dict):
    return {k: v for k, v in cfg_dict.items() if k not in RESUME_MUTABLE_KEYS}


def _truncate_logs(run_dir, env_step):
    """Drop log rows written after the checkpoint we resume from."""
    path = os.path.join(run_dir, "eval.csv")
    if os.path.exists(path):
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if int(r["env_step"]) <= env_step]
        with open(path + ".tmp", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
            w.writeheader()
            w.writerows(rows)
        os.replace(path + ".tmp", path)
    path = os.path.join(run_dir, "train_metrics.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            lines = [l for l in f if l.strip() and json.loads(l)["env_step"] <= env_step]
        with open(path + ".tmp", "w") as f:
            f.writelines(lines)
        os.replace(path + ".tmp", path)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    import wandb
    import xqc.utils

    xqc.utils.check_hydra_config()

    run_dir = os.path.abspath(cfg.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, "DONE")):
        print(f"[done] {run_dir}/DONE exists, nothing to do", flush=True)
        return 0
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    with wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project_name,
        name=f"{cfg.agent.name}_{cfg.env.name}_seed{cfg.seed}",
        config=cfg_dict,
        dir=run_dir,
        settings=wandb.Settings(start_method="thread"),
        mode=cfg.wandb.mode,
    ) as wandb_run:
        # Delayed imports to allow Hydra's env_set to take effect
        import numpy as np
        import tqdm

        from xqc import checkpoint
        from xqc.agents import XQCLearner
        from xqc.replay_buffer import ParallelReplayBuffer
        from xqc.envs import ParallelEnv
        from xqc.normalization import RewardNormalizer
        import xqc.logging

        stop = StopFlag()
        signal.signal(signal.SIGUSR1, stop)
        signal.signal(signal.SIGTERM, stop)
        job_start = time.time()

        ################################################################################
        # Setup (identical to train_parallel.py)
        ################################################################################

        xqc.utils.log_slurm_info(wandb_run)
        xqc.logging.print_config(cfg)

        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        make_envs = partial(
            ParallelEnv,
            env_names=[cfg.env.name] * cfg.num_seeds,
            action_repeat=cfg.env.action_repeat,
        )
        env = make_envs(seed=cfg.seed)
        eval_env = make_envs(seed=cfg.seed + 42)
        # Upstream leaves the action space unseeded, so warm-up actions come from
        # OS entropy and fixed-seed runs are not reproducible. Seed it here.
        env.action_space.seed(cfg.seed)

        agent_kwargs = OmegaConf.to_container(cfg.agent, resolve=True)
        agent_kwargs.update({
            "seed": cfg.seed,
            "num_seeds": cfg.num_seeds,
            "updates_per_step": cfg.updates_per_step,
            "num_interactions": int(cfg.max_steps / cfg.env.action_repeat),
            "observations": env.observation_space.sample()[0, None],
            "actions": env.action_space.sample()[0, None],
        })
        agent = XQCLearner(**agent_kwargs)

        discount = xqc.utils.get_discount(env.max_episode_steps, cfg.env.action_repeat)

        replay_buffer = ParallelReplayBuffer(
            env.observation_space,
            env.action_space,
            capacity=cfg.replay_buffer_size,
            num_seeds=cfg.num_seeds,
            n_steps=cfg.n_steps,
            gamma=discount,
        )

        reward_normalizer = None
        if cfg.agent.reward_normalization:
            reward_normalizer = RewardNormalizer(
                n_seeds=cfg.num_seeds,
                gamma=discount,
                max_v=cfg.agent.max_v,
            )

        observations = env.reset()
        infos = {}
        start_step = 1
        update_count = 0
        wall_time_prev = 0.0  # training wall time accumulated by earlier jobs

        ################################################################################
        # Resume
        ################################################################################

        existing = checkpoint.list_checkpoints(run_dir)
        ckpt_dir = checkpoint.find_latest(run_dir)
        if cfg.resume == "never" and existing:
            raise RuntimeError(f"resume=never but {run_dir} already has checkpoints: {existing}")
        if existing and ckpt_dir is None:
            raise RuntimeError(
                f"{run_dir} has checkpoint dirs {existing} but none is complete. "
                "Refusing to restart from scratch; inspect them by hand."
            )
        if ckpt_dir is not None:
            state = checkpoint.read_state(ckpt_dir)
            old, new = _identity_config(state["config"]), _identity_config(cfg_dict)
            if old != new:
                diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new) if old.get(k) != new.get(k)}
                raise RuntimeError(f"Config differs from checkpoint (key: (checkpoint, now)): {diff}")
            if not state["has_replay_buffer"]:
                print("[resume] WARNING: checkpoint has no replay buffer, resuming with an empty one", flush=True)
            observations = checkpoint.restore(
                ckpt_dir, state,
                agent=agent, replay_buffer=replay_buffer,
                reward_normalizer=reward_normalizer, env=env,
            )
            loop = state["loop"]
            start_step = loop["i"] + 1
            update_count = loop["update_count"]
            wall_time_prev = loop["wall_time_s"]
            _truncate_logs(run_dir, state["env_step"])
            print(f"[resume] from {ckpt_dir} at env_step={state['env_step']} "
                  f"(learner step {int(np.asarray(agent.step))}, buffer size {replay_buffer.size})", flush=True)
        else:
            _truncate_logs(run_dir, 0)
            print(f"[resume] no checkpoint in {run_dir}, starting from scratch", flush=True)

        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)

        last_i = cfg.max_steps // cfg.env.action_repeat
        seeds = list(range(cfg.seed, cfg.seed + cfg.num_seeds))

        def wall_time():
            return wall_time_prev + (time.time() - job_start)

        def save_checkpoint(i):
            checkpoint.save(
                run_dir, i * cfg.env.action_repeat,
                agent=agent, replay_buffer=replay_buffer, reward_normalizer=reward_normalizer,
                env=env, observations=observations,
                loop_state={"i": i, "update_count": update_count, "wall_time_s": wall_time()},
                config=cfg_dict, keep=cfg.keep_checkpoints,
                save_replay_buffer=cfg.save_replay_buffer,
            )

        def log_eval(env_step, eval_stats):
            path = os.path.join(run_dir, "eval.csv")
            new_file = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
                if new_file:
                    w.writeheader()
                for s, r in zip(seeds, eval_stats["return"]):
                    w.writerow({"env": cfg.env.name, "seed": s, "env_step": env_step,
                                "return": float(r), "wall_time_s": round(wall_time(), 1),
                                "slurm_job_id": job_id})
            returns = [float(r) for r in eval_stats["return"]]
            print(f"[eval] env_step={env_step} return mean={np.mean(returns):.1f} per_seed={np.round(returns, 1).tolist()}", flush=True)
            summary_path = os.path.join(run_dir, "summary.json")
            summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
            best = summary.get("best_mean_return")
            summary.update({
                "env": cfg.env.name, "agent": cfg.agent.name, "seeds": seeds,
                "max_steps": cfg.max_steps, "last_env_step": env_step,
                "last_mean_return": float(np.mean(returns)), "last_returns": returns,
                "best_mean_return": float(np.mean(returns)) if best is None else max(best, float(np.mean(returns))),
                "wall_time_s": round(wall_time(), 1), "finished": env_step >= cfg.max_steps,
            })
            with open(summary_path + ".tmp", "w") as f:
                json.dump(summary, f, indent=2)
            os.replace(summary_path + ".tmp", summary_path)

        def log_train(env_step, metrics):
            row = {"env_step": env_step}
            for k, v in metrics.items():
                v = np.asarray(v)
                if v.ndim >= 1 and v.shape[0] == cfg.num_seeds and v.size == cfg.num_seeds:
                    row[k] = [float(x) for x in v]
            with open(os.path.join(run_dir, "train_metrics.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")

        ################################################################################
        # Main XQC training loop (identical to train_parallel.py apart from the
        # checkpoint / stop / logging hooks at the end of each iteration)
        ################################################################################

        i = start_step - 1
        for i in tqdm.tqdm(
            range(start_step, last_i + 1),
            initial=start_step - 1,
            total=last_i,
            smoothing=0.1,
            disable=xqc.utils.is_slurm_job(),
        ):
            if i < cfg.start_training:
                actions = env.action_space.sample()
            else:
                actions, _ = agent.sample_actions_with_log_probs(observations)

            next_observations, rewards, dones, truncs, _ = env.step(actions)
            if cfg.agent.reward_normalization:
                reward_normalizer.update(rewards, np.logical_or(dones, truncs))

            masks = env.generate_masks(dones, truncs)
            replay_buffer.insert(observations, actions, rewards, masks, truncs, next_observations)
            observations = next_observations

            observations, terms, truncs, _ = env.reset_where_done(observations, dones, truncs)

            if i > cfg.start_training:
                batches = replay_buffer.sample_parallel_multibatch(cfg.batch_size, cfg.updates_per_step)

                if cfg.agent.reward_normalization:
                    normalized_rewards = reward_normalizer.normalize(batches.rewards)
                    batches = batches._replace(rewards=normalized_rewards)

                infos = agent.update(batches, num_updates=cfg.updates_per_step)
                update_count += cfg.updates_per_step

            env_step = i * cfg.env.action_repeat

            if i == 1 or i % cfg.eval_interval == 0 or i == last_i:
                eval_stats = eval_env.evaluate(
                    agent,
                    num_episodes=cfg.eval_episodes,
                    temperature=0.0,
                    render=cfg.eval_video,
                    render_frameskip=cfg.get("eval_video_frameskip", 1),
                    render_num_envs=cfg.get("eval_video_num_envs", 1),
                )
                xqc.logging.log_multiple_seeds_to_wandb(env_step, eval_stats, fps=cfg.get("eval_video_fps", 30))
                log_eval(env_step, eval_stats)

            if i > cfg.start_training and i % cfg.log_interval == 0:
                metrics = xqc.logging.metrics.compute_logging_metrics(agent, infos)
                xqc.logging.log_multiple_seeds_to_wandb(env_step, metrics)
                log_train(env_step, metrics)

            if cfg.log_interval_condition_number and \
                (i == cfg.start_training or i % cfg.log_interval_condition_number == 0):
                xqc.logging.log_multiple_seeds_to_wandb(
                    env_step,
                    xqc.logging.metrics.compute_hessian_eigenspectrum(agent, replay_buffer, cfg.batch_size, cfg.num_seeds)
                )

            if cfg.agent.reset_freq and update_count >= cfg.agent.reset_freq:
                agent.reset()
                update_count = 0

            # ---- checkpoint / stop hooks -------------------------------------
            if i == last_i:
                break
            out_of_time = cfg.max_runtime_s and (time.time() - job_start) >= cfg.max_runtime_s
            if stop.signum is not None or out_of_time:
                reason = signal.Signals(stop.signum).name if stop.signum else "max_runtime_s"
                print(f"[stop] {reason} at env_step={env_step}, checkpointing", flush=True)
                save_checkpoint(i)
                print(f"[stop] exiting cleanly, resume will continue from env_step={env_step}", flush=True)
                return 0
            if cfg.checkpoint_interval and env_step % cfg.checkpoint_interval == 0:
                save_checkpoint(i)

        # Reached max_steps
        save_checkpoint(last_i)
        with open(os.path.join(run_dir, "DONE"), "w") as f:
            f.write(f"env_step={last_i * cfg.env.action_repeat}\njob={job_id}\n")
        print(f"[done] reached max_steps={cfg.max_steps}, wrote {run_dir}/DONE", flush=True)
        return 0


if __name__ == "__main__":
    main()
