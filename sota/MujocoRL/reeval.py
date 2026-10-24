"""Roll out frozen checkpoints on explicit reset seeds (uncertain-QD jobs).

Usage:
    python reeval.py --requests requests.json --out results.json

requests.json:
    {"protocol": {...evaluation_protocol(...)...},
     "requests": [{"request_id": "...", "solution_id": "...",
                   "model_path": ".../gene_seed123.zip",
                   "checkpoint_sha256": "...", "seeds": [..]}, ...]}

For each request the checkpoint is loaded, its SHA-256 is checked against the
recorded one (statistics are tied to exact weights), and one deterministic
episode is run per seed with eval.evaluate_model. The output keeps every
episode separately; aggregation happens in the driver.

Result status per request:
    ok                  every episode ran; non-finite episodes are flagged
                        per episode and judged by the driver
    infra_failure       the checkpoint is missing, unreadable or changed, or
                        the environment does not match the protocol
The file is written atomically (temp file + rename) and ends the log with
"Job Done", so the driver can tell a finished job from a lost one.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback

from stable_baselines3 import TD3

from eval import evaluate_model, evaluation_protocol, make_env, validate_reward_spec


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_request(req, env, max_steps):
    start = time.time()
    base = {"request_id": req["request_id"], "solution_id": req.get("solution_id"),
            "seeds": req["seeds"]}
    path = req["model_path"]
    if not os.path.exists(path):
        return {**base, "status": "infra_failure", "failure_kind": "missing_artifact",
                "error": f"checkpoint not found: {path}", "wall_seconds": time.time() - start}
    expected = req.get("checkpoint_sha256")
    if expected and file_sha256(path) != expected:
        return {**base, "status": "infra_failure", "failure_kind": "missing_artifact",
                "error": "checkpoint hash changed since it was evaluated",
                "wall_seconds": time.time() - start}
    try:
        model = TD3.load(path, env=env, device="cpu")
    except Exception as exc:  # unreadable artifact, not a property of the policy
        return {**base, "status": "infra_failure", "failure_kind": "malformed_result",
                "error": f"could not load checkpoint: {exc!r}", "wall_seconds": time.time() - start}
    _, _, _, metrics = evaluate_model(model, env, max_steps=max_steps, seeds=req["seeds"])
    return {**base, "status": "ok", "episodes": metrics["episodes"],
            "env_steps": metrics["eval_env_steps"], "wall_seconds": time.time() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.requests, encoding="utf-8") as f:
        payload = json.load(f)
    protocol = payload["protocol"]
    env = make_env(protocol["env_id"])
    live_protocol = evaluation_protocol(protocol["env_id"], protocol["max_steps"], "cpu")
    problems = validate_reward_spec(env)
    # Compare through JSON so tuples (live) and lists (from the file) agree.
    if json.dumps(live_protocol, sort_keys=True) != json.dumps(protocol, sort_keys=True):
        problems.append("evaluation protocol differs from the one the samples were recorded under")

    results = []
    for req in payload["requests"]:
        if problems:
            results.append({"request_id": req["request_id"], "solution_id": req.get("solution_id"),
                            "seeds": req["seeds"], "status": "infra_failure",
                            "failure_kind": "fitness_spec_mismatch", "error": "; ".join(problems)})
            continue
        try:
            results.append(run_request(req, env, protocol["max_steps"]))
        except Exception as exc:
            results.append({"request_id": req["request_id"], "solution_id": req.get("solution_id"),
                            "seeds": req["seeds"], "status": "infra_failure",
                            "failure_kind": "resource_error" if isinstance(exc, (MemoryError, OSError))
                            else "malformed_result",
                            "error": traceback.format_exc()})
        print(f"  {req['request_id']}: {results[-1]['status']}", flush=True)
    env.close()

    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"protocol": protocol, "results": results}, f)
    os.replace(tmp, args.out)
    print("Job Done")


if __name__ == "__main__":
    sys.exit(main())
