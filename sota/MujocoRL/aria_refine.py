"""Post-hoc ARIA refinement of frozen TD3 actors (job script).

    python aria_refine.py --input aria_input.json --out aria_output.json

Runs src/uqd/aria.run_aria on the flattened parameters of each input
checkpoint's actor (model.actor; the critics are untouched and unused at
evaluation). Every rollout is one deterministic episode with a reset seed
from the "aria" stream (src/uqd/seeds.py). Each optimised genotype is saved
as a new checkpoint with its own SHA-256, so its evaluations can never be
mixed with those of the policy it started from.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from stable_baselines3 import TD3

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repository root for src.uqd

from eval import evaluate_model, evaluation_protocol, make_env, validate_reward_spec  # noqa: E402
from src.uqd.archive import Grid  # noqa: E402
from src.uqd.aria import run_aria  # noqa: E402
from src.uqd.seeds import SeedScheme  # noqa: E402


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as f:
        job = json.load(f)
    start = time.time()
    torch.set_num_threads(1)

    protocol = job["protocol"]
    env = make_env(protocol["env_id"])
    problems = validate_reward_spec(env)
    live = evaluation_protocol(protocol["env_id"], protocol["max_steps"], "cpu")
    if json.dumps(live, sort_keys=True) != json.dumps(protocol, sort_keys=True):
        problems.append("evaluation protocol differs from the run's protocol")
    if problems:
        raise SystemExit("ARIA refused to run: " + "; ".join(problems))

    grid = Grid([tuple(d) for d in job["grid"]["descriptors"]], job["grid"]["bins"])
    names = [d[0] for d in job["grid"]["descriptors"]]
    scheme = SeedScheme(job["run_seed"], job["final_master_seed"])
    models, inputs = {}, []
    for item in job["inputs"]:
        if file_sha256(item["model_path"]) != item["checkpoint_sha256"]:
            raise SystemExit(f"checkpoint changed since it was archived: {item['model_path']}")
        model = TD3.load(item["model_path"], env=env, device="cpu")
        models[item["label"]] = model
        params = torch.nn.utils.parameters_to_vector(model.actor.parameters()).detach().cpu().numpy()
        inputs.append({"label": item["label"], "params": params.astype(np.float64)})

    counters = {"env_steps": 0, "episodes": 0}
    used_seeds = []

    def evaluate(key, params, seeds):
        model = models[key]
        vec = torch.as_tensor(np.asarray(params), dtype=torch.float32)
        torch.nn.utils.vector_to_parameters(vec, model.actor.parameters())
        _, _, _, metrics = evaluate_model(model, env, max_steps=protocol["max_steps"], seeds=seeds)
        out = []
        for ep in metrics["episodes"]:
            counters["env_steps"] += ep["length"]
            counters["episodes"] += 1
            used_seeds.append(ep["seed"])
            out.append((ep["return"], [ep["descriptor"][n] for n in names], ep["nonfinite"]))
        return out

    log_path = args.out + ".log.jsonl"
    log_file = open(log_path, "w", encoding="utf-8")

    def log(**row):
        log_file.write(json.dumps(row, default=str) + "\n")
        log_file.flush()

    rng = np.random.default_rng(scheme.search_seed() + 7)
    cells, info = run_aria(inputs, grid, job["aria_cfg"], evaluate, scheme.aria_seeds, rng,
                           post_eval_episodes=job["post_eval_episodes"], log=log)
    log_file.close()

    out_models = os.path.join(os.path.dirname(os.path.abspath(args.out)), "models")
    os.makedirs(out_models, exist_ok=True)
    results = []
    for cell, res in sorted(cells.items()):
        model = models[res["key"]]
        vec = torch.as_tensor(res["params"], dtype=torch.float32)
        torch.nn.utils.vector_to_parameters(vec, model.actor.parameters())
        tmp = os.path.join(out_models, f"aria_tmp_{'_'.join(map(str, cell))}.zip")
        model.save(tmp)
        sha = file_sha256(tmp)
        path = os.path.join(out_models, f"aria_{sha[:16]}.zip")
        os.replace(tmp, path)
        results.append({
            "cell": list(cell), "source": res["source"], "key": res["key"], "phase": res["phase"],
            "objective": res["objective"], "valid": res["valid"], "post_eval": res["post_eval"],
            "model_path": path, "checkpoint_sha256": sha,
        })
    env.close()
    payload = {"results": results, "info": info, "cost": {**counters, "wall_seconds": time.time() - start},
               "aria_seeds_used": sorted(set(used_seeds)), "log": log_path}
    with open(args.out + ".tmp", "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(args.out + ".tmp", args.out)
    print("Job Done")


if __name__ == "__main__":
    main()
