"""Train and evaluate one XQC gene for the MESB driver; write the MESB result JSON.

The MESB counterpart of Shazeb's ``sota/XQC/train_xqc_gene.py`` (branch
shazeb/halfcheetah-xqc), which writes run_improved.py's results CSV instead.
To keep the two islands comparable this script reuses his code rather than
copying it:

* the gene is checked and turned into Hydra overrides by his
  ``gene_overrides()`` (ALGO header, allowed keys, types and ranges);
* training is his ``sota/XQC/train_xqc.py`` with the same arguments his
  trainer passes (num_seeds trained together, no periodic checkpoints,
  ``final_descriptors=true``), so a gene gets the same budget and the same
  evaluation rollouts as in his island;
* descriptors come from his ``descriptors.json`` (``llmge_descriptors.py``).

Differences, all in what is *written*, not how a gene is trained:

* output goes to the MESB run directory (``--run-dir``, ``--out-json``), not
  ``sota/XQC/llmge/results``;
* the result is MESB's named JSON (``results_io`` schema 1). ``mean_reward``
  and ``std_reward`` are his (mean / std over the per-seed 10-episode means);
  every episode's return, distance and control cost is kept as well;
* ``mean_control_cost`` follows the MESB convention, the per-episode total.
  His per-step value is kept as ``mean_control_cost_per_step``; HalfCheetah
  episodes always last ``--max-steps`` (no early termination), so
  total = per-step x max_steps. Sliding boundaries are rank based, so the
  archive is identical for either unit.

Runs inside the XQC environment (Python 3.9, JAX; ``sota/XQC/pace/env.sh``),
so it uses only the standard library and Python 3.9 syntax. It always writes
a result file and exits 0 unless the arguments are wrong; a failed gene is
``status: failed`` with the reason and the tail of ``train.log``.

    python sota/MujocoRL/train_gene_xqc.py --gene-file G.py --gene-id ID \\
        --out-json out/ID.json --run-dir runs/ID --timesteps 100000 --num-seeds 3
"""

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XQC_DIR = os.path.join(REPO_ROOT, "sota", "XQC")
GENE_VALIDATOR = os.path.join(XQC_DIR, "train_xqc_gene.py")
XQC_TRAIN = os.path.join(XQC_DIR, "train_xqc.py")
SCHEMA_VERSION = 1


def atomic_write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path), suffix=".tmp",
                               dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_validator(path=GENE_VALIDATOR):
    spec = importlib.util.spec_from_file_location("xqc_train_xqc_gene", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def xqc_command(args, overrides, python=sys.executable):
    """The train_xqc.py call, argument for argument as in train_xqc_gene.py."""
    eval_interval = min(25_000, args.timesteps)
    return [python, XQC_TRAIN,
            "env={}".format(args.env), "seed={}".format(args.seed),
            "num_seeds={}".format(args.num_seeds), "max_steps={}".format(args.timesteps),
            "eval_interval={}".format(eval_interval),
            "eval_episodes={}".format(args.eval_episodes),
            "start_training={}".format(min(5000, args.timesteps // 2)),
            "run_dir={}".format(args.run_dir), "checkpoint_interval=0",
            "save_replay_buffer=false", "final_descriptors=true", "resume=never"] + list(overrides)


def _tail(path, n=40):
    try:
        with open(path, errors="replace") as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return ""


def _mean(xs):
    return sum(xs) / len(xs)


def _pstd(xs):
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def payload_from_descriptors(desc, args, train_time, config):
    """MESB result JSON from XQC's descriptors.json."""
    per_seed = [float(r) for r in desc["mean_return_per_seed"]]
    returns = [float(r) for seed_rows in desc["returns"] for r in seed_rows]
    distances = [float(d) for seed_rows in desc["distances"] for d in seed_rows]
    ctrl_per_step = [float(c) for seed_rows in desc["control_costs"] for c in seed_rows]
    ctrl_total = [c * args.max_steps for c in ctrl_per_step]
    mean_reward = _mean(per_seed)
    values = [mean_reward, _pstd(per_seed), float(desc["mean_distance"]),
              float(desc["mean_control_cost"])]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("non-finite metric in descriptors.json: {}".format(values))
    return {
        "schema_version": SCHEMA_VERSION, "gene_id": args.gene_id, "status": "success",
        "generation": args.generation, "seed": args.seed, "eval_seed": args.seed + 42,
        "env_id": args.env, "algorithm": "XQC", "timesteps": args.timesteps,
        "num_train_seeds": args.num_seeds, "num_eval_episodes": args.eval_episodes,
        "max_eval_steps": args.max_steps, "gene_file": os.path.abspath(args.gene_file),
        "train_time_sec": train_time, "total_time_sec": train_time,
        "mean_reward": mean_reward, "std_reward": _pstd(per_seed),
        "std_reward_episodes": _pstd(returns), "per_seed_mean_reward": per_seed,
        "param_count": int(desc["actor_params"]),
        # XQC keeps its final checkpoint in the run directory (no SB3 .zip).
        "model_path": os.path.abspath(args.run_dir), "rewards": returns,
        "xqc_config": config,
        "behavior": {
            "mean_distance": float(desc["mean_distance"]),
            "mean_control_cost": float(desc["mean_control_cost"]) * args.max_steps,
            "mean_control_cost_per_step": float(desc["mean_control_cost"]),
            "distances": distances, "control_costs": ctrl_total,
            "control_costs_per_step": ctrl_per_step,
            "control_cost_source": "xqc llmge_descriptors per-step cost x max_steps",
            "initial_x_source": "xqc llmge_descriptors (info x_position or qpos[0])",
        },
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--gene-file", required=True)
    p.add_argument("--gene-id", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--run-dir", required=True, help="XQC run directory for this gene")
    p.add_argument("--generation", type=int, default=0)
    p.add_argument("--env", default="HalfCheetah-v4")
    p.add_argument("--seed", type=int, default=0,
                   help="XQC seed (his island uses 0 for every gene)")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=1000,
                   help="Episode length, used to convert per-step control cost to a total")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    start = time.time()
    base = {"generation": args.generation, "seed": args.seed, "algorithm": "XQC",
            "gene_file": os.path.abspath(args.gene_file)}

    def fail(reason, stage, log_file=None):
        payload = {"schema_version": SCHEMA_VERSION, "gene_id": args.gene_id,
                   "status": "failed", "error": reason, "stage": stage,
                   "train_time_sec": time.time() - start}
        payload.update(base)
        if log_file:
            payload.update(log_file=log_file, log_tail=_tail(log_file))
        atomic_write_json(args.out_json, payload)
        print("FAILED ({}): {}".format(stage, reason), flush=True)
        return 0

    try:
        overrides, config = load_validator().gene_overrides(args.gene_file)
    except Exception as exc:  # noqa: BLE001 - an invalid gene is a failed evaluation
        return fail("invalid gene: {!r}".format(exc), "validate")

    os.makedirs(args.run_dir, exist_ok=True)
    cmd = xqc_command(args, overrides)
    print("Running:", " ".join(cmd), flush=True)
    log_path = os.path.join(args.run_dir, "train.log")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=XQC_DIR, stdout=log, stderr=subprocess.STDOUT)
    desc_path = os.path.join(args.run_dir, "descriptors.json")
    if proc.returncode != 0 or not os.path.exists(desc_path):
        return fail("train_xqc.py exited {}".format(proc.returncode), "train", log_path)
    try:
        with open(desc_path) as fh:
            desc = json.load(fh)
        payload = payload_from_descriptors(desc, args, time.time() - start, config)
    except Exception as exc:  # noqa: BLE001
        return fail("could not read {}: {!r}".format(desc_path, exc), "descriptors", log_path)
    atomic_write_json(args.out_json, payload)
    b = payload["behavior"]
    print("Mean reward: {:.2f} (per seed {}), distance {:.1f}, control cost {:.1f} "
          "({:.4f}/step), actor params {}, {:.0f}s".format(
              payload["mean_reward"], [round(r, 1) for r in payload["per_seed_mean_reward"]],
              b["mean_distance"], b["mean_control_cost"], b["mean_control_cost_per_step"],
              payload["param_count"], payload["train_time_sec"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
