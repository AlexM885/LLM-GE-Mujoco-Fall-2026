"""Post-hoc ARIA stage for a finished run, kept separate from its archive.

Reads <run_dir>/archive_latest.json, refines the search elites (one per
search cell: the cell's best solution by its search estimate) with
sota/MujocoRL/aria_refine.py, and writes everything to <run_dir>/aria/:

    aria_input.json / aria_output.json   job input and raw output
    archive_latest.json                  refined archive (new identities)
    run_metadata.json, cost_ledger.json  labels, choices, ARIA cost
    final_eval/final_eval.json           held-out evaluation of the refined
                                         archive, same seeds and metrics as
                                         the unrefined final evaluation

The unrefined archive and its final evaluation are never modified. Because
LLM mutations do not inherit actor weights, ARIA gains measure post-hoc
refinement of the final archive, not better evolutionary search.
"""

import json
import os

import numpy as np

from .budget import CostLedger
from .engine import git_revision
from .final_runner import run_final_evaluation, search_seeds_for_run
from .mujoco import REPO_ROOT
from .stats import estimate_from_values

ARIA_SCRIPT = "sota/MujocoRL/aria_refine.py"
IMPLEMENTATION_CHOICES = {
    "utilities": "centred ranks in [-0.5, 0.5]",
    "optimizer": "configurable; default adam, lr 0.01 (not specified in the paper)",
    "adjacency": "configurable; default von_neumann (not specified in the paper)",
    "pair_selection": "uniform random over (explored, unexplored adjacent) pairs",
    "mirrored_pair_seeds": "a mirrored pair shares its reset seed",
    "nonfinite_samples": "objective -inf, excluded from f_min",
    "parameters": "actor weights only (critics unused at evaluation)",
    "input": "search elites, one per search cell (best search estimate)",
}


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_aria_stage(run_dir, final_episodes=None, executor=None, post_eval_episodes=8, aria_cfg=None):
    from src.cfg.uqd_presets import aria_budget_label

    meta = _load(os.path.join(run_dir, "run_metadata.json"))
    snapshot = _load(os.path.join(run_dir, "archive_latest.json"))
    cfg = dict(aria_cfg or meta["resolved_config"]["aria"])
    if not cfg.get("enabled"):
        raise ValueError(f"ARIA is not enabled in preset {meta['preset']}; pass aria_cfg to run it anyway")
    label = aria_budget_label(cfg)
    aria_dir = os.path.join(run_dir, "aria")
    os.makedirs(aria_dir, exist_ok=True)

    elites = set(snapshot["elites"].values())
    inputs = [{"label": e["solution"], "model_path": e["model_path"], "checkpoint_sha256": e["checkpoint_sha256"]}
              for e in snapshot["archive"] if e["solution"] in elites and e["kind"] == "policy"]
    job = {
        "protocol": meta["evaluation_protocol"],
        "grid": snapshot["grid"],
        "aria_cfg": cfg,
        "run_seed": meta["seeds"]["run_seed"],
        "final_master_seed": meta["seeds"]["final_master_seed"],
        "post_eval_episodes": post_eval_episodes,
        "inputs": inputs,
    }
    in_path, out_path = os.path.join(aria_dir, "aria_input.json"), os.path.join(aria_dir, "aria_output.json")
    with open(in_path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=1)
    if os.path.exists(out_path):
        os.remove(out_path)
    lost = executor.run_command([ARIA_SCRIPT, "--input", in_path, "--out", out_path], out_path, "aria")
    if lost:
        raise RuntimeError(f"ARIA job did not produce {out_path} ({lost}); see {executor.work_dir}")
    output = _load(out_path)

    names = [d[0] for d in snapshot["grid"]["descriptors"]]
    archive, elite_map, invalid = [], {}, 0
    for res in output["results"]:
        if not res["valid"]:
            invalid += 1
            continue
        post = [(f, d) for f, d, bad in res["post_eval"]]
        est = estimate_from_values([f for f, _ in post], [d for _, d in post])
        sid = f"aria:{res['checkpoint_sha256'][:16]}"
        cell = res["cell"]
        archive.append({"cell": cell, "rank": 0, "solution": sid, "gene": None, "kind": "policy",
                        "status": "eligible", "model_path": res["model_path"],
                        "checkpoint_sha256": res["checkpoint_sha256"], "parent": res["source"],
                        "origin": "aria", "phase": res["phase"], "estimate": est.to_dict()})
        elite_map[",".join(map(str, cell))] = sid

    ledger = CostLedger()
    ledger.record("aria", None, 0, env_steps=output["cost"]["env_steps"], episodes=output["cost"]["episodes"],
                  wall_seconds=output["cost"]["wall_seconds"])
    search_seeds = set(search_seeds_for_run(run_dir, snapshot)) | set(output["aria_seeds_used"])
    aria_snapshot = {
        "preset": f"{meta['preset']}+aria", "strategy": "aria", "generation": snapshot.get("generation"),
        "grid": snapshot["grid"], "archive": archive, "elites": elite_map,
        "search_seeds_used": sorted(search_seeds),
        "illusory_metrics": {"filled_cells": len(archive),
                             "coverage": len(archive) / (snapshot["grid"]["bins"] ** len(names))},
    }
    with open(os.path.join(aria_dir, "archive_latest.json"), "w", encoding="utf-8") as f:
        json.dump(aria_snapshot, f, indent=1)
    aria_meta = dict(meta)
    aria_meta.update({
        "preset": f"{meta['preset']}+aria", "stage": "post_hoc_aria", "aria_budget_label": label,
        "aria_config": cfg, "aria_implementation_choices": IMPLEMENTATION_CHOICES,
        "aria_info": output["info"], "aria_invalid_results": invalid,
        "aria_code_revision": git_revision(REPO_ROOT),
        "note": "post-hoc refinement of frozen actor weights; not inherited by the LLM search",
    })
    with open(os.path.join(aria_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(aria_meta, f, indent=2, default=str)
    with open(os.path.join(aria_dir, "cost_ledger.json"), "w", encoding="utf-8") as f:
        json.dump(ledger.to_list(), f)

    final = run_final_evaluation(aria_dir, executor, episodes=final_episodes,
                                 extra_labels={"stage": "post_hoc_aria", "aria_budget_label": label})
    unrefined_path = os.path.join(run_dir, "final_eval", "final_eval.json")
    comparison = None
    if os.path.exists(unrefined_path):
        before = _load(unrefined_path)["metrics"]["reports"]["best_only"]
        after = final["metrics"]["reports"]["best_only"]
        keys = ("coverage", "qd_score_raw", "qd_score_normalized", "mean_p_in_cell", "max_fitness")
        comparison = {k: {"unrefined": before.get(k), "aria": after.get(k)} for k in keys}
    summary = {"aria_budget_label": label, "inputs": len(inputs), "refined_cells": len(archive),
               "invalid": invalid, "info": output["info"], "cost": output["cost"],
               "comparison_best_only": comparison}
    with open(os.path.join(aria_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    return {"summary": summary, "final": final}
