"""Train and evaluate one LLM-GE XQC gene; write the standard LLM-GE results CSV.

Same command-line contract as sota/MujocoRL/train_rl.py, so run_improved.py can
launch it with EVAL_RUNLINE:

    python train_xqc_gene.py -network models.network_<gene_id> -timesteps 100000 -num_seeds 3

1. Loads llmge/models/network_<gene_id>.py, checks ALGO == "xqc", and validates every
   key of get_architecture_kwargs() | get_xqc_kwargs() against ALLOWED (type + range).
2. Runs train_xqc.py with those keys as Hydra overrides (num_seeds trained batched,
   no periodic checkpoints, final_descriptors=true). Its output goes to train.log in
   the run dir, so a crash there never prints a traceback into this job's output.
3. Writes llmge/results/<gene_id>_results.csv with the columns run_improved.py and
   MAP-Elites read:  mean_reward,std_reward,train_time,param_count,mean_distance,
   mean_control_cost  (+ algo,timesteps,num_seeds at the end).
   mean_reward = mean over seeds of each seed's 10-episode deterministic return;
   std_reward = std of those per-seed means; param_count = actor parameters.
   Any failure writes the -999999 sentinel row instead, like train_rl.py.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
LLMGE_DIR = os.path.join(HERE, "llmge")
HEADER = "mean_reward,std_reward,train_time,param_count,mean_distance,mean_control_cost,algo,timesteps,num_seeds"
FAILED = -999999.0

AGENT_KEYS = {
    # architecture
    "hidden_dims_critic": ("dims", 1, 4096),
    "hidden_dims_actor": ("dims", 1, 4096),
    "n_critics": (int, 1, 10),
    "use_batch_norm": (bool,),
    "use_weight_norm": (bool,),
    "use_layer_norm": (bool,),
    "pre_activation_bn": (bool,),
    "normalize_last_layer": (bool,),
    "skip_connections": (bool,),
    # optimisation
    "actor_lr": (float, 1e-6, 1e-2),
    "critic_lr": (float, 1e-6, 1e-2),
    "temp_lr": (float, 1e-6, 1e-2),
    "lr_end": (float, 1e-7, 1e-2),
    "tau": (float, 1e-4, 1.0),
    "policy_delay": (int, 1, 10),
    "init_temperature": (float, 1e-4, 10.0),
    "weight_decay": (float, 0.0, 1.0),
}
TOP_KEYS = {
    "batch_size": (int, 16, 2048),
    # Gradient updates per env step; capped since it multiplies GPU time.
    "updates_per_step": (int, 1, 4),
}


def validate(key, value, spec):
    kind = spec[0]
    if kind == "dims":
        if not (isinstance(value, (list, tuple)) and 1 <= len(value) <= 8
                and all(isinstance(v, int) and not isinstance(v, bool) and spec[1] <= v <= spec[2] for v in value)):
            raise ValueError(f"{key}={value!r}: expected 1-8 ints in [{spec[1]}, {spec[2]}]")
        return "[" + ",".join(str(v) for v in value) + "]"
    if kind is bool:
        # The XQC yaml writes these flags as 1/0, so accept those too.
        if not (isinstance(value, bool) or value in (0, 1)):
            raise ValueError(f"{key}={value!r}: expected True/False")
        return "true" if value else "false"
    if kind is int and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f"{key}={value!r}: expected an int")
    if kind is float and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ValueError(f"{key}={value!r}: expected a number")
    if not spec[1] <= value <= spec[2]:
        raise ValueError(f"{key}={value!r}: outside [{spec[1]}, {spec[2]}]")
    return repr(float(value)) if kind is float else str(value)


def gene_overrides(path):
    spec = importlib.util.spec_from_file_location("gene", path)
    gene = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gene)
    if getattr(gene, "ALGO", None) != "xqc":
        raise ValueError(f"gene declares ALGO={getattr(gene, 'ALGO', None)!r}, expected 'xqc'")
    arch, hyper = gene.get_architecture_kwargs(), gene.get_xqc_kwargs()
    if not isinstance(arch, dict) or not isinstance(hyper, dict):
        raise ValueError("get_architecture_kwargs() and get_xqc_kwargs() must return dicts")
    overrides, config = [], {}
    for key, value in {**arch, **hyper}.items():
        if key in AGENT_KEYS:
            overrides.append(f"agent.{key}={validate(key, value, AGENT_KEYS[key])}")
        elif key in TOP_KEYS:
            overrides.append(f"{key}={validate(key, value, TOP_KEYS[key])}")
        else:
            raise ValueError(f"unknown key {key!r}")
        config[key] = value
    return overrides, config


def write_results(gene_id, row, stats):
    results_dir = os.path.join(LLMGE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{gene_id}_results.csv"), "w") as f:
        f.write(HEADER + "\n" + ",".join(str(v) for v in row))
    stats_dir = os.path.join(LLMGE_DIR, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    with open(os.path.join(stats_dir, f"{gene_id}_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train an LLM-GE XQC gene")
    parser.add_argument("-network", required=True, help='Module path like "models.network_XXXX"')
    parser.add_argument("-timesteps", type=int, default=100_000)
    parser.add_argument("-num_seeds", type=int, default=3)
    parser.add_argument("-eval_episodes", type=int, default=10)
    parser.add_argument("-eval_max_steps", type=int, default=1000, help="Unused; MuJoCo episodes are 1000 steps")
    parser.add_argument("-env", default="HalfCheetah-v4")
    parser.add_argument("-seed", type=int, default=0)
    args = parser.parse_args()

    gene_id = args.network.replace("models.network_", "")
    gene_path = os.path.join(LLMGE_DIR, "models", f"network_{gene_id}.py")
    runs_root = os.environ.get("XQC_RUNS", os.path.join(LLMGE_DIR, "runs"))
    run_dir = os.path.join(runs_root, "llmge", gene_id)
    start = time.time()
    stats = {"gene_id": gene_id, "algo": "xqc", "env": args.env, "timesteps": args.timesteps,
             "num_seeds": args.num_seeds, "run_dir": run_dir}

    def fail(message):
        train_time = time.time() - start
        stats.update(status="failed", error=message, train_time_sec=train_time, mean_reward=FAILED)
        write_results(gene_id, [FAILED, 0.0, train_time, 999999999, 0.0, 0.0, "xqc", args.timesteps, args.num_seeds], stats)
        print(f"ERROR evaluating generated model: {message}")
        print(f"Mean reward: {FAILED}, Std: 0.0, Time: {train_time:.1f}s")
        print("Job Done")

    try:
        overrides, config = gene_overrides(gene_path)
    except Exception as e:
        return fail(f"invalid gene: {e!r}")
    stats["config"] = config

    os.makedirs(run_dir, exist_ok=True)
    eval_interval = min(25_000, args.timesteps)
    cmd = [sys.executable, os.path.join(HERE, "train_xqc.py"),
           f"env={args.env}", f"seed={args.seed}", f"num_seeds={args.num_seeds}",
           f"max_steps={args.timesteps}", f"eval_interval={eval_interval}",
           f"eval_episodes={args.eval_episodes}", f"start_training={min(5000, args.timesteps // 2)}",
           f"run_dir={run_dir}", "checkpoint_interval=0", "save_replay_buffer=false",
           "final_descriptors=true", "resume=never", *overrides]
    stats["command"] = " ".join(cmd)
    print("Running:", stats["command"], flush=True)
    log_path = os.path.join(run_dir, "train.log")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    desc_path = os.path.join(run_dir, "descriptors.json")
    if proc.returncode != 0 or not os.path.exists(desc_path):
        with open(log_path) as f:
            tail = f.read()[-1500:].replace("Traceback", "Trace-back")
        print(tail)
        return fail(f"train_xqc.py exited {proc.returncode}; see {log_path}")

    with open(desc_path) as f:
        desc = json.load(f)
    per_seed = desc["mean_return_per_seed"]
    mean_reward = sum(per_seed) / len(per_seed)
    std_reward = (sum((r - mean_reward) ** 2 for r in per_seed) / len(per_seed)) ** 0.5
    train_time = time.time() - start
    row = [mean_reward, std_reward, train_time, desc["actor_params"], desc["mean_distance"],
           desc["mean_control_cost"], "xqc", args.timesteps, args.num_seeds]
    stats.update(status="ok", train_time_sec=train_time, mean_reward=mean_reward, std_reward=std_reward,
                 per_seed_return=per_seed, param_count=desc["actor_params"],
                 mean_distance=desc["mean_distance"], mean_control_cost=desc["mean_control_cost"])
    write_results(gene_id, row, stats)
    print(f"Mean reward: {mean_reward}, Std: {std_reward}, Params: {desc['actor_params']}, "
          f"Distance: {desc['mean_distance']}, CtrlCost: {desc['mean_control_cost']}, Time: {train_time:.1f}s")
    print("Job Done")


if __name__ == "__main__":
    main()
