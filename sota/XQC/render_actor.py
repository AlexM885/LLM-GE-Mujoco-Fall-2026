"""Film an already-trained LLM-GE XQC gene: no retraining, a few minutes per gene.

Rebuilds the agent from the gene's run dir (config.json, written by train_xqc.py),
loads the trained actor and saves one deterministic rollout gif per seed with
llmge_descriptors.record_videos (same rollout as video_gene.py). The actor comes
from <run_dir>/actor.pkl (save_actor=true, every gene since the multi-objective
run) or, for older genes, from the final full checkpoint in <run_dir>/checkpoints.

    python render_actor.py <gene_id or run dir> [...] [-video_steps 300] [-out DIR]
Run dir: $XQC_RUNS/llmge/<gene_id>, or any run dir given as a path (e.g. a
video_gene.py run).  Output: <out>/<name>/{seed<s>.gif, returns.json}, name = the gene
id or the run dir's basename (default out: $XQC_RUNS/llmge_render).
"""

import argparse
import json
import os
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))


def build_agent(cfg):
    """The XQCLearner train_xqc.py built for this config (weights are replaced after)."""
    from xqc.agents import XQCLearner
    from xqc.envs import ParallelEnv

    env = ParallelEnv(env_names=[cfg["env"]["name"]] * cfg["num_seeds"],
                      action_repeat=cfg["env"]["action_repeat"], seed=cfg["seed"])
    agent_kwargs = dict(cfg["agent"])
    agent_kwargs.update({
        "seed": cfg["seed"],
        "num_seeds": cfg["num_seeds"],
        "updates_per_step": cfg["updates_per_step"],
        "num_interactions": int(cfg["max_steps"] / cfg["env"]["action_repeat"]),
        "observations": env.observation_space.sample()[0, None],
        "actions": env.action_space.sample()[0, None],
    })
    return XQCLearner(**agent_kwargs)


def load_actor(agent, run_dir):
    path = os.path.join(run_dir, "actor.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            actor = pickle.load(f)
        agent.actor = agent.actor.replace(params=actor["params"], batch_stats=actor["batch_stats"])
        return path
    from xqc import checkpoint
    ckpt = checkpoint.find_latest(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"{run_dir} has neither actor.pkl nor a checkpoint")
    with open(os.path.join(ckpt, "agent.pkl"), "rb") as f:
        checkpoint.load_agent_state(agent, pickle.load(f))
    return ckpt


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gene_ids", nargs="+", help="gene ids or run dir paths")
    parser.add_argument("-video_steps", type=int, default=300)
    parser.add_argument("-out", default=None)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    runs = os.environ.get("XQC_RUNS", os.path.join(HERE, "runs"))
    out_root = args.out or os.path.join(runs, "llmge_render")
    from llmge_descriptors import record_videos

    for arg in args.gene_ids:
        run_dir = arg if os.path.isdir(arg) else os.path.join(runs, "llmge", arg)
        gene_id = os.path.basename(os.path.normpath(run_dir))
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)
        agent = build_agent(cfg)
        source = load_actor(agent, run_dir)
        out_dir = os.path.join(out_root, gene_id)
        returns = record_videos(agent, cfg["env"]["name"], cfg["num_seeds"], out_dir, steps=args.video_steps,
                                seed=cfg["seed"] + 42, action_repeat=cfg["env"]["action_repeat"])
        print(f"{gene_id}: actor from {source}; gifs in {out_dir}; "
              f"{args.video_steps}-step returns {[round(float(r), 1) for r in returns]}", flush=True)


if __name__ == "__main__":
    main()
