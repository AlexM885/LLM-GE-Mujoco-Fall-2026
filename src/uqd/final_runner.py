"""Run the held-out final evaluation for one finished run directory.

Inputs, in the run directory:
    archive_latest.json   the final archive snapshot (engine runs), or a
                          snapshot exported from a legacy checkpoint by
                          scripts/uqd_export_legacy_archive.py
    run_metadata.json     resolved config, seeds, protocol, code revision
    engine_state.pkl      (engine runs) to list every search seed used
    cost_ledger.json      search cost

Outputs in <run_dir>/final_eval/:
    final_eval.json       illusory and corrected metrics, settings, labels,
                          search and final-evaluation cost
    heldout_episodes.json every held-out episode, keyed by checkpoint
"""

import json
import os
import time

from .budget import CostLedger
from .final_eval import compute_final_metrics, final_requests
from .seeds import SeedScheme, check_disjoint
from .engine import UQDEngine, git_revision
from .mujoco import REPO_ROOT


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def search_seeds_for_run(run_dir, snapshot):
    if "search_seeds_used" in snapshot:          # legacy export
        return set(snapshot["search_seeds_used"])
    state = os.path.join(run_dir, "engine_state.pkl")
    return UQDEngine.load_state(state).search_seeds_used()


def run_final_evaluation(run_dir, executor, episodes=None, snapshot_name="archive_latest.json",
                         out_name="final_eval", max_retries=2, extra_labels=None):
    meta = _load(os.path.join(run_dir, "run_metadata.json"))
    snapshot = _load(os.path.join(run_dir, snapshot_name))
    final_cfg = dict(meta["resolved_config"]["final_eval"])
    if episodes is not None:
        final_cfg["episodes"] = int(episodes)
    protocol = meta["evaluation_protocol"]

    seeds = SeedScheme(0, final_cfg["master_seed"]).holdout_seeds(final_cfg["episodes"])
    check_disjoint(search_seeds_for_run(run_dir, snapshot), seeds)

    requests = final_requests(snapshot, seeds)
    ledger = CostLedger()
    episodes_by_key, lost = {}, []
    pending = requests
    start = time.time()
    for attempt in range(max_retries + 1):
        if not pending:
            break
        results = executor.run_rollouts(pending, protocol, label=f"final_a{attempt}")
        retry = []
        for res in results:
            if res["status"] == "ok":
                if [int(e["seed"]) for e in res["episodes"]] != seeds:
                    raise ValueError(f"{res['solution_id']}: held-out seeds were not used as requested")
                episodes_by_key[res["solution_id"]] = res["episodes"]
                ledger.record("final_eval", res["solution_id"], 0, env_steps=res.get("env_steps"),
                              episodes=len(res["episodes"]), wall_seconds=res.get("wall_seconds"))
            else:
                ledger.record("final_eval", res["solution_id"], 0, status="infra_failure",
                              note=res.get("failure_kind", ""))
                retry.append(res["request_id"])
        pending = [r for r in pending if r["request_id"] in retry]
    lost = [r["solution_id"] for r in pending]

    metrics = compute_final_metrics(snapshot, episodes_by_key, final_cfg, seeds)
    search_ledger_path = os.path.join(run_dir, "cost_ledger.json")
    search_cost = (CostLedger.from_list(_load(search_ledger_path)).summary()
                   if os.path.exists(search_ledger_path) else snapshot.get("cost"))
    out = {
        "run_dir": os.path.abspath(run_dir),
        "preset": meta.get("preset"),
        "category": meta.get("category"),
        "run_seed": meta.get("seeds", {}).get("run_seed"),
        "labels": dict(extra_labels or {}),
        "snapshot": snapshot_name,
        "final_generation": snapshot.get("generation"),
        "heldout_seed_count": len(seeds),
        "heldout_checkpoints_lost_to_infra": lost,
        "metrics": metrics,
        "search_cost": search_cost,
        "final_eval_cost": ledger.summary(),
        "final_eval_wall_seconds": time.time() - start,
        "evaluated_with_code_revision": git_revision(REPO_ROOT),
    }
    out_dir = os.path.join(run_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "final_eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    with open(os.path.join(out_dir, "heldout_episodes.json"), "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "episodes": episodes_by_key}, f)
    return out
