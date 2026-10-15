"""Train a MuJoCo locomotion TD3 policy with an optional VLM-authored shaping term.

Drop-in alternative to :mod:`train_rl` that is identical in every observable
way *except*:

- The training env is wrapped with :class:`AuxRewardWrapper` when a
  cached aux_reward snippet is available for this generation.
- The evaluation env is a separate, unwrapped Walker2d instance, so the
  reported fitness is byte-for-byte comparable to the me-base baseline.

If the aux_reward cache is missing or fails validation the script falls
through to vanilla training, so the same binary covers both the VLM-aux
arm and the no-feedback control.

Config via environment variables (no CLI changes vs train_rl.py):

- ``LLMGE_AUX_REWARD_CACHE``   path to a JSON file with
  ``{"aux_reward_code": "...", "lam": 0.1, "anneal_steps": 100000}``.
- ``LLMGE_AUX_REWARD_LAM``     override lam from the JSON (float).
- ``LLMGE_AUX_REWARD_ANNEAL``  override anneal_steps (int, 0 disables).

Keeping ``train_rl.py`` untouched means this file never conflicts with
teammates' baseline edits.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

try:
    from aux_reward_sandbox import AuxRewardSandboxError, load_aux_reward
    from aux_reward_wrapper import AuxRewardWrapper
except ImportError:  # package-style import
    from sota.MujocoRL.aux_reward_sandbox import (  # type: ignore
        AuxRewardSandboxError, load_aux_reward)
    from sota.MujocoRL.aux_reward_wrapper import AuxRewardWrapper  # type: ignore

# ``eval``, ``train_rl``, ``stable_baselines3`` are imported lazily inside
# ``main()`` so this module can be unit-tested (loader paths, fallback
# behaviour) without pulling in sb3 or mujoco.

DEFAULT_ENV_ID = "HalfCheetah-v4"


# --------------------------------------------------------------------------- #
# Aux-reward cache loading
# --------------------------------------------------------------------------- #
AUX_CACHE_ENV = "LLMGE_AUX_REWARD_CACHE"
AUX_LAM_ENV = "LLMGE_AUX_REWARD_LAM"
AUX_ANNEAL_ENV = "LLMGE_AUX_REWARD_ANNEAL"
AUX_CENTER_ENV = "LLMGE_AUX_REWARD_CENTER"
AUX_PBRS_ENV = "LLMGE_AUX_REWARD_PBRS"
AUX_PBRS_GAMMA_ENV = "LLMGE_AUX_REWARD_PBRS_GAMMA"

#: Default lam when no suggestion comes with the snippet. Chosen so the
#: aux term is a visible but not dominating influence on the Walker2d
#: step reward, whose typical per-step magnitude is ~1-5.
DEFAULT_LAM = 0.1


def _load_aux_reward_cache(path: str | None) -> tuple | None:
    """Return (aux_fn, lam, anneal_steps, cache_meta) or ``None``.

    ``None`` means run vanilla training. We swallow every failure here
    (missing path, bad JSON, sandbox rejection) because the no-aux fallback
    is the control arm and should never be derailed by a dirty cache.
    """
    if not path:
        return None
    if not os.path.isfile(path):
        print(f"VLM aux_reward cache not found at {path}, running vanilla training",
              flush=True)
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"VLM aux_reward cache {path} unreadable ({exc}); vanilla training",
              flush=True)
        return None

    code = payload.get("aux_reward_code")
    if not isinstance(code, str) or not code.strip():
        print(f"VLM aux_reward cache has no aux_reward_code; vanilla training",
              flush=True)
        return None
    try:
        fn = load_aux_reward(code)
    except AuxRewardSandboxError as exc:
        print(f"VLM aux_reward failed validation ({exc}); vanilla training",
              flush=True)
        return None

    lam = _resolve_float(AUX_LAM_ENV, payload.get("lam"), DEFAULT_LAM)
    anneal_env = os.environ.get(AUX_ANNEAL_ENV)
    anneal_steps = int(anneal_env) if anneal_env else payload.get("anneal_steps")
    try:
        anneal_steps = int(anneal_steps) if anneal_steps is not None else None
    except (TypeError, ValueError):
        anneal_steps = None

    center_env = os.environ.get(AUX_CENTER_ENV, "")
    center = center_env.lower() in ("1", "true", "yes", "on")
    pbrs_env = os.environ.get(AUX_PBRS_ENV, "")
    pbrs = pbrs_env.lower() in ("1", "true", "yes", "on")
    pbrs_gamma = _resolve_float(AUX_PBRS_GAMMA_ENV, None, 0.99)

    cache_meta = {
        "cache_path": path,
        "generation": payload.get("generation"),
        "parent_hashes": payload.get("parent_hashes"),
        "model_id": payload.get("model_id"),
        "created_at": payload.get("created_at"),
        "lam_source": "env" if AUX_LAM_ENV in os.environ else (
            "cache" if payload.get("lam") is not None else "default"),
        "center": center,
        "pbrs": pbrs,
        "pbrs_gamma": pbrs_gamma if pbrs else None,
    }
    return {
        "fn": fn,
        "lam": lam,
        "anneal_steps": anneal_steps,
        "center": center,
        "pbrs": pbrs,
        "pbrs_gamma": pbrs_gamma,
        "meta": cache_meta,
    }


def _resolve_float(env_name: str, cache_value, default_value: float) -> float:
    v = os.environ.get(env_name)
    if v is not None:
        try:
            return float(v)
        except ValueError:
            pass
    if isinstance(cache_value, (int, float)):
        return float(cache_value)
    return float(default_value)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(
    gene_id: str,
    env_id: str = DEFAULT_ENV_ID,
    timesteps: int = 500000,
    eval_episodes: int | None = None,
    eval_max_steps: int | None = None,
    model_dir: str = "sota/MujocoRL/trained_models",
    stats_dir: str = "sota/MujocoRL/stats",
) -> None:
    # Heavy imports deferred so that unit tests for the loader paths can
    # import this module without having sb3 or MuJoCo installed.
    import numpy as np
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    # Shazeb's XQC branch has a minimal eval.py (just evaluate_model); the
    # Walker2d-v5 helpers (make_env, validate_reward_spec, etc.) live in a
    # sibling module so we can share them without touching his file.
    try:
        from eval import evaluate_model, make_env, validate_reward_spec
    except ImportError:
        from eval import evaluate_model
        from vlm_eval_helpers import make_env, validate_reward_spec
    from train_rl import write_failure_results

    start_time = time.time()

    # Load gene (same contract as train_rl.py).
    try:
        module = importlib.import_module(f"models.network_{gene_id}")
        policy_kwargs = module.get_policy_kwargs()
        td3_kwargs = module.get_td3_kwargs()
    except Exception as exc:  # pylint: disable=broad-except
        write_failure_results(gene_id, start_time, repr(exc), stats_dir)
        return

    policy_class = policy_kwargs.pop("policy_class", "MlpPolicy")

    print(f"Environment: {env_id}", flush=True)

    # Two env instances: a wrapped one for training, a raw one for eval.
    # Separate instances also mean the eval roll uses untouched env state.
    train_env = make_env(env_id)
    eval_env = make_env(env_id)

    for env in (train_env, eval_env):
        problems = validate_reward_spec(env)
        if problems:
            train_env.close()
            eval_env.close()
            write_failure_results(
                gene_id, start_time,
                "fitness spec mismatch: " + "; ".join(problems), stats_dir)
            return

    # Opt-in VLM shaping. Falls through to vanilla training on any error.
    aux_cache = _load_aux_reward_cache(os.environ.get(AUX_CACHE_ENV))
    aux_meta: dict = {"enabled": False}
    if aux_cache is not None:
        try:
            train_env = AuxRewardWrapper(
                train_env, aux_cache["fn"], lam=aux_cache["lam"],
                anneal_steps=aux_cache["anneal_steps"],
                center=aux_cache["center"],
                pbrs=aux_cache["pbrs"],
                pbrs_gamma=aux_cache["pbrs_gamma"])
            aux_meta = {
                "enabled": True, "lam_initial": train_env.lam_initial,
                "anneal_steps": train_env.anneal_steps,
                "center": train_env.center,
                "pbrs": train_env.pbrs,
                "pbrs_gamma": train_env.pbrs_gamma if train_env.pbrs else None,
                **aux_cache["meta"]}
            print(
                f"VLM aux_reward active: lam={train_env.lam_initial}, "
                f"anneal_steps={train_env.anneal_steps}, "
                f"center={aux_cache['center']}, pbrs={aux_cache['pbrs']}, "
                f"gamma={aux_cache['pbrs_gamma'] if aux_cache['pbrs'] else '-'}",
                flush=True)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Could not install AuxRewardWrapper ({exc}); vanilla training",
                  flush=True)

    try:
        exploration_noise = float(td3_kwargs.pop("exploration_noise", 0.1))
        if exploration_noise > 0 and "action_noise" not in td3_kwargs:
            n_actions = train_env.action_space.shape[-1]
            td3_kwargs["action_noise"] = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=exploration_noise * np.ones(n_actions))
        if td3_kwargs.get("learning_starts", 100) >= timesteps:
            td3_kwargs["learning_starts"] = max(timesteps // 2, 1)
        model = TD3(
            policy=policy_class, env=train_env,
            policy_kwargs=policy_kwargs, **td3_kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        write_failure_results(gene_id, start_time, repr(exc), stats_dir)
        train_env.close()
        eval_env.close()
        return

    param_count = sum(p.numel() for p in model.actor.parameters())

    try:
        model.learn(total_timesteps=timesteps)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{gene_id}.zip")
        model.save(model_path)

        if eval_episodes is None:
            num_eval_episodes = 1 if timesteps <= 10000 else 10
        else:
            num_eval_episodes = eval_episodes
        if eval_max_steps is None:
            max_eval_steps = 200 if timesteps <= 10000 else 1000
        else:
            max_eval_steps = eval_max_steps

        # Evaluation runs on the UNWRAPPED env, so the reported reward is
        # pure Walker2d. This is the whole point of the split: the aux
        # term can only guide learning; it cannot inflate fitness.
        mean_reward, std_reward, rewards, metrics = evaluate_model(
            model, eval_env,
            num_episodes=num_eval_episodes, max_steps=max_eval_steps)
    except Exception as exc:  # pylint: disable=broad-except
        write_failure_results(gene_id, start_time, repr(exc), stats_dir)
        train_env.close()
        eval_env.close()
        return

    train_time = time.time() - start_time

    foot_contact_0 = float(metrics.get("foot_contact_0", 0.0))
    foot_contact_1 = float(metrics.get("foot_contact_1", 0.0))
    mean_distance = float(metrics.get("mean_distance", 0.0))
    mean_control_cost = float(metrics.get("mean_control_cost", 0.0))
    mean_return_forward = float(metrics.get("mean_return_forward", 0.0))
    mean_return_healthy = float(metrics.get("mean_return_healthy", 0.0))
    mean_return_ctrl = float(metrics.get("mean_return_ctrl", 0.0))

    # Same CSV schema as train_rl.py so run_improved.py's parser is unchanged.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{gene_id}_results.csv"), "w") as handle:
        handle.write(
            "mean_reward,std_reward,train_time,param_count,mean_distance,"
            "mean_control_cost,foot_contact_0,foot_contact_1,"
            "mean_return_forward,mean_return_healthy,mean_return_ctrl\n")
        handle.write(
            f"{mean_reward},{std_reward},{train_time},{param_count},"
            f"{mean_distance},{mean_control_cost},"
            f"{foot_contact_0},{foot_contact_1},"
            f"{mean_return_forward},{mean_return_healthy},{mean_return_ctrl}")

    stats = {
        "gene_id": gene_id,
        "env_id": env_id,
        "algorithm": "TD3",
        "timesteps": timesteps,
        "num_eval_episodes": num_eval_episodes,
        "max_eval_steps": max_eval_steps,
        "train_time_sec": train_time,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "per_episode_rewards": rewards,
        "param_count": param_count,
        "mean_distance": mean_distance,
        "mean_control_cost": mean_control_cost,
        "foot_contact_0": foot_contact_0,
        "foot_contact_1": foot_contact_1,
        "foot_geom_names": metrics.get("foot_geom_names", []),
        "mean_return_forward": mean_return_forward,
        "mean_return_healthy": mean_return_healthy,
        "mean_return_ctrl": mean_return_ctrl,
        "reward_decomposition_error": metrics.get("reward_decomposition_error", 0.0),
        "score_band": metrics.get("score_band", ""),
        "model_path": model_path,
        "distances": metrics.get("distances", []),
        "control_costs": metrics.get("control_costs", []),
        "foot_contacts_0": metrics.get("foot_contacts_0", []),
        "foot_contacts_1": metrics.get("foot_contacts_1", []),
        "aux_reward": aux_meta,
    }
    if isinstance(train_env, AuxRewardWrapper):
        stats["aux_reward_audit"] = train_env.audit()

    os.makedirs(stats_dir, exist_ok=True)
    with open(os.path.join(stats_dir, f"{gene_id}_stats.json"), "w",
              encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(
        f"Mean reward: {mean_reward}, Std: {std_reward}, "
        f"Params: {param_count}, Distance: {mean_distance}, "
        f"CtrlCost: {mean_control_cost}, "
        f"FootContact: ({foot_contact_0:.3f}, {foot_contact_1:.3f}), "
        f"Time: {train_time:.1f}s", flush=True)
    print(f"Fitness F = forward {mean_return_forward:.2f}"
          f" + alive {mean_return_healthy:.2f}"
          f" + ctrl {mean_return_ctrl:.4f}  [{metrics.get('score_band', '')}]",
          flush=True)
    if aux_meta.get("enabled"):
        audit = stats.get("aux_reward_audit", {})
        print(f"Aux audit: calls={audit.get('aux_calls')}, "
              f"mean_aux={audit.get('aux_mean'):.4f}, "
              f"lam final={audit.get('lam_final'):.4f}", flush=True)
    print(f"Saved model: {model_path}", flush=True)
    print("Job Done", flush=True)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MuJoCo TD3 with optional VLM-authored aux reward")
    parser.add_argument("-network", type=str, required=True,
                        help='Module path like "models.network_XXXX"')
    parser.add_argument("-env", type=str, default=DEFAULT_ENV_ID,
                        help="Gymnasium environment id (e.g. HalfCheetah-v4)")
    parser.add_argument("-timesteps", type=int, default=500000)
    parser.add_argument("-eval_episodes", type=int, default=None)
    parser.add_argument("-eval_max_steps", type=int, default=None)
    parser.add_argument("-model_dir", type=str,
                        default="sota/MujocoRL/trained_models")
    parser.add_argument("-stats_dir", type=str,
                        default="sota/MujocoRL/stats")
    args = parser.parse_args()
    gene_id = args.network.replace("models.network_", "")
    main(
        gene_id,
        env_id=args.env,
        timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        model_dir=args.model_dir,
        stats_dir=args.stats_dir,
    )
