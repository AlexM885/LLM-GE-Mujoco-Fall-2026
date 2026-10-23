"""Post-hoc check of one XQC gene: retrain with new seeds, then measure it.

Runs inside the XQC environment (Python 3.9, JAX; sota/XQC/pace/env.sh), one
GPU job per gene (scripts/mesb/matched/posthoc_gene.sbatch). It answers two
questions that the archive cannot be "gamed" on:

1. Re-checked reward. The gene is retrained from scratch exactly as in the run
   (Shazeb's train_xqc.py, same arguments as train_xqc_gene.py: steps, seeds
   trained together, evaluation episodes) but with a different XQC seed
   (default 1000), so both the training seeds and the evaluation episodes are
   new. The run's own score came from one selection-biased training.
2. Behaviour after training, without retraining:
   - reward under changed conditions (heavier torso, slippery floor, weak back
     leg, weak front leg), 10 episodes per trained seed each;
   - gait features MESB never optimised (share of steps each foot touches the
     ground, mean torso height, torso angle spread), one episode per seed.

How, without editing his files: his train_xqc.py imports
``llmge_descriptors.rollout_descriptors`` at the end of training. This script
puts a wrapper module under that name before running train_xqc.py in-process.
The wrapper returns his normal result (so descriptors.json is unchanged) and
additionally runs his own rollout function on perturbed environments, built by
his ``make_env`` with the MuJoCo model edited, and records the gait features.

    python scripts/mesb/matched/posthoc_eval.py --gene-file G.py --gene-id ID \\
        --out-json posthoc/ID.json --run-dir posthoc/runs/ID [--seed 1000]
"""

import argparse
import importlib.util
import json
import os
import runpy
import sys
import time
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
XQC_DIR = os.path.join(REPO_ROOT, "sota", "XQC")
ADAPTER = os.path.join(REPO_ROOT, "sota", "MujocoRL", "train_gene_xqc.py")

#: Changed conditions: (description, function editing a MuJoCo model in place).
#: HalfCheetah names: bodies torso/bthigh/..., geom "floor", actuators b*/f*.
PERTURBATIONS = {
    "heavy_torso": "torso mass and inertia x1.5",
    "slippery_floor": "sliding friction of every geom x0.5 (MuJoCo uses the larger of the two)",
    "weak_back_leg": "back-leg actuator gear (bthigh, bshin, bfoot) x0.5",
    "weak_front_leg": "front-leg actuator gear (fthigh, fshin, ffoot) x0.5",
}


def apply_perturbation(name, model):
    import mujoco
    oid = lambda kind, n: mujoco.mj_name2id(model, kind, n)  # noqa: E731
    if name == "heavy_torso":
        b = oid(mujoco.mjtObj.mjOBJ_BODY, "torso")
        model.body_mass[b] *= 1.5
        model.body_inertia[b] *= 1.5
    elif name == "slippery_floor":
        model.geom_friction[:, 0] *= 0.5
    elif name in ("weak_back_leg", "weak_front_leg"):
        prefix = "b" if name == "weak_back_leg" else "f"
        for joint in ("thigh", "shin", "foot"):
            model.actuator_gear[oid(mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + joint), 0] *= 0.5
    else:
        raise ValueError(name)


def gait_features(agent, make_env, env_name, num_seeds, seed, action_repeat):
    """One deterministic episode per trained seed; features averaged over seeds."""
    import mujoco
    import numpy as np
    envs = [make_env(env_name, seed=seed, action_repeat=action_repeat) for _ in range(num_seeds)]
    models = [e.unwrapped.model for e in envs]
    feet = [mujoco.mj_name2id(models[0], mujoco.mjtObj.mjOBJ_GEOM, g) for g in ("bfoot", "ffoot")]
    floor = mujoco.mj_name2id(models[0], mujoco.mjtObj.mjOBJ_GEOM, "floor")
    obs = np.stack([e.reset(seed=seed + 5000 + 1000 * s)[0] for s, e in enumerate(envs)]).astype(np.float32)
    done = np.zeros(num_seeds, dtype=bool)
    contact = np.zeros((num_seeds, 2))
    heights = [[] for _ in envs]
    angles = [[] for _ in envs]
    steps = np.zeros(num_seeds)
    while not done.all():
        actions = np.asarray(agent.sample_actions(obs, temperature=0.0))
        for s, env in enumerate(envs):
            if done[s]:
                continue
            o, _, term, trunc, _ = env.step(actions[s])
            obs[s] = o
            data = env.unwrapped.data
            touching = [False, False]
            for i in range(data.ncon):
                c = data.contact[i]
                for k, g in enumerate(feet):
                    if {c.geom1, c.geom2} == {g, floor}:
                        touching[k] = True
            contact[s] += touching
            heights[s].append(float(data.qpos[1]))
            angles[s].append(float(data.qpos[2]))
            steps[s] += 1
            done[s] = term or trunc
    for env in envs:
        env.close()
    per_seed = [{"back_foot_contact": float(contact[s, 0] / max(steps[s], 1)),
                 "front_foot_contact": float(contact[s, 1] / max(steps[s], 1)),
                 "torso_height_mean": float(np.mean(heights[s])),
                 "torso_angle_std": float(np.std(angles[s]))} for s in range(num_seeds)]
    keys = per_seed[0].keys()
    return {"per_seed": per_seed, "mean": {k: float(np.mean([p[k] for p in per_seed])) for k in keys}}


def install_wrapper(out_path, perturbations):
    """Put a wrapper around his llmge_descriptors module into sys.modules."""
    spec = importlib.util.spec_from_file_location("llmge_descriptors_real",
                                                  os.path.join(XQC_DIR, "llmge_descriptors.py"))
    real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real)
    wrapper = types.ModuleType("llmge_descriptors")

    def rollout_descriptors(agent, env_name, num_seeds, num_episodes=10, seed=0, action_repeat=1):
        nominal = real.rollout_descriptors(agent, env_name, num_seeds, num_episodes, seed, action_repeat)
        extra = {"perturbations": {}, "gait": None}
        original_make_env = real.make_env
        try:
            for name in perturbations:
                def perturbed_make_env(env_name, seed=0, action_repeat=1, _name=name):
                    env = original_make_env(env_name, seed=seed, action_repeat=action_repeat)
                    apply_perturbation(_name, env.unwrapped.model)
                    return env
                real.make_env = perturbed_make_env
                r = real.rollout_descriptors(agent, env_name, num_seeds, num_episodes, seed, action_repeat)
                per_seed = [float(x) for x in r["mean_return_per_seed"]]
                extra["perturbations"][name] = {
                    "description": PERTURBATIONS[name], "mean_return_per_seed": per_seed,
                    "mean_return": sum(per_seed) / len(per_seed),
                    "mean_distance": float(r["mean_distance"])}
        finally:
            real.make_env = original_make_env
        extra["gait"] = gait_features(agent, original_make_env, env_name, num_seeds, seed, action_repeat)
        with open(out_path, "w") as fh:
            json.dump(extra, fh, indent=2)
        return nominal

    wrapper.rollout_descriptors = rollout_descriptors
    wrapper.__getattr__ = lambda attr: getattr(real, attr)   # e.g. record_videos
    sys.modules["llmge_descriptors"] = wrapper


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--gene-file", required=True)
    p.add_argument("--gene-id", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arm", default="")
    p.add_argument("--env", default="HalfCheetah-v4")
    p.add_argument("--seed", type=int, default=1000, help="XQC seed for the re-check (run used 0)")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--perturbations", default=",".join(PERTURBATIONS))
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    for attr in ("gene_file", "out_json", "run_dir"):          # Hydra may change the cwd
        setattr(args, attr, os.path.abspath(getattr(args, attr)))
    start = time.time()
    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    adapter = load(ADAPTER, "train_gene_xqc")
    base = {"gene_id": args.gene_id, "arm": args.arm, "seed": args.seed, "timesteps": args.timesteps,
            "num_seeds": args.num_seeds, "gene_file": os.path.abspath(args.gene_file)}

    def finish(payload):
        payload.update(base, seconds=time.time() - start)
        adapter.atomic_write_json(args.out_json, payload)
        print(json.dumps({k: payload.get(k) for k in ("status", "mean_reward", "error")}), flush=True)
        return 0

    try:
        overrides, config = adapter.load_validator().gene_overrides(args.gene_file)
    except Exception as exc:  # noqa: BLE001
        return finish({"status": "failed", "stage": "validate", "error": repr(exc)})

    extra_path = os.path.join(args.run_dir, "posthoc_extra.json")
    perturbations = [p for p in args.perturbations.split(",") if p]
    sys.path.insert(0, XQC_DIR)            # his modules import `xqc` from here
    install_wrapper(extra_path, perturbations)
    cmd = adapter.xqc_command(argparse.Namespace(
        env=args.env, seed=args.seed, num_seeds=args.num_seeds, timesteps=args.timesteps,
        eval_episodes=args.eval_episodes, run_dir=args.run_dir), overrides, python="python")
    sys.argv = cmd[1:]                     # [train_xqc.py, hydra overrides...]
    os.chdir(XQC_DIR)
    print("Running in-process:", " ".join(cmd[1:]), flush=True)
    try:
        runpy.run_path(adapter.XQC_TRAIN, run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            return finish({"status": "failed", "stage": "train", "error": f"train_xqc.py exit {exc.code}"})
    except Exception as exc:  # noqa: BLE001
        return finish({"status": "failed", "stage": "train", "error": repr(exc)})

    desc_path = os.path.join(args.run_dir, "descriptors.json")
    if not (os.path.exists(desc_path) and os.path.exists(extra_path)):
        return finish({"status": "failed", "stage": "train", "error": "no descriptors / posthoc output"})
    with open(desc_path) as fh:
        desc = json.load(fh)
    with open(extra_path) as fh:
        extra = json.load(fh)
    per_seed = [float(r) for r in desc["mean_return_per_seed"]]
    return finish({"status": "success", "mean_reward": sum(per_seed) / len(per_seed),
                   "per_seed_mean_reward": per_seed, "mean_distance": float(desc["mean_distance"]),
                   "mean_control_cost_per_step": float(desc["mean_control_cost"]),
                   "actor_params": int(desc["actor_params"]), "xqc_config": config, **extra})


if __name__ == "__main__":
    sys.exit(main())
