"""Held-out final evaluation and the corrected (common reporting) archive.

Every configuration's final archive is scored the same way:

1. Every archived checkpoint is rolled out on the same M held-out reset seeds
   (seeds.SeedScheme.holdout_seeds; disjoint from every search seed, checked
   explicitly). For retrain-sampled genes, every replicate checkpoint is
   rolled out.
2. Each search cell becomes one *entity*:
     best_only          the cell's best solution by its search estimate (UQD
                        App. D). For the archive-sampling family this is also
                        the in-cell selector.
     in_cell_selector   Deep-Grid only: draw j = 0..M-1 picks a member by the
                        Deep-Grid roulette on search estimates and uses that
                        member's j-th held-out episode (UQD Sec. III-D).
3. Each entity is placed by its held-out descriptor estimate on the same
   32 x 32 grid; when several land in one cell, the best held-out fitness
   estimate is kept (Justesen et al.; Flageat & Cully). This corrected
   archive has one layout for every configuration, including legacy runs.

Estimators default to the median over the M episodes (UQD paper, following
Doerr & Sutton); means are reported alongside.

Metrics (per corrected archive unless noted)
--------------------------------------------
coverage, filled_cells           corrected cells filled / total cells
qd_score_raw                     sum of held-out fitness estimates
qd_score_normalized              sum of clip((f - lo) / (hi - lo), 0, 1) with
                                 fixed bounds shared by every configuration
qd_score_loss                    (illusory - corrected) / illusory, both
                                 normalised; illusory = search-time estimates
max_fitness, mean_fitness
p_score / mean_p_in_cell         P(episode descriptor in the entity's corrected
                                 cell), summed / averaged (ARIA's P-Score)
search_cell_retention            mean over search elites of P(episode
                                 descriptor in the cell the search put it in)
correct_cell_fraction            fraction of search elites whose corrected
                                 cell equals their search cell ("number of
                                 correct BD", Flageat & Cully 2020)
fitness_sd_median, fitness_sem_mean   held-out spread and standard error
ndv_mean                         mean negated descriptor variance (ARIA NDV)
heldout_candidate_failures       entities with non-finite held-out episodes;
                                 they are left out of the corrected archive
                                 and counted here
The UQD Reproducibility-Score normalises variances by the maximum observed
per cell across all compared runs, so it is computed by scripts/uqd_compare.py
over a set of runs, not here.
"""

import math

import numpy as np

from .archive import DeepGridArchive, Grid
from .seeds import derive_int


def _cell_key(cell):
    return ",".join(str(int(c)) for c in cell)


def final_requests(snapshot, holdout_seeds):
    """One rollout request per archived checkpoint (all ranks, all replicates)."""
    requests = []
    seen = set()
    for entry in snapshot["archive"]:
        checkpoints = []
        if entry["kind"] == "gene":
            for rep in entry.get("replicates", []):
                checkpoints.append((f"{entry['solution']}#{rep['train_seed']}", rep["model_path"],
                                    rep["checkpoint_sha256"]))
        else:
            checkpoints.append((entry["solution"], entry["model_path"], entry["checkpoint_sha256"]))
        for key, path, sha in checkpoints:
            if key in seen:
                continue
            seen.add(key)
            requests.append({"request_id": f"final:{key}", "solution_id": key, "model_path": path,
                             "checkpoint_sha256": sha, "seeds": list(holdout_seeds), "start_index": 0})
    return requests


def episode_stats(episodes, grid, search_cell, fitness_estimator="median", descriptor_estimator="median"):
    """Held-out statistics of one entity from its episode records."""
    nonfinite = sum(1 for ep in episodes if ep.get("nonfinite"))
    good = [ep for ep in episodes if not ep.get("nonfinite")]
    names = [name for name, _, _ in grid.descriptors]
    if nonfinite or not good:
        return {"n": len(episodes), "nonfinite_episodes": nonfinite, "valid": False}
    f = np.array([ep["return"] for ep in good], dtype=float)
    d = np.array([[ep["descriptor"][n] for n in names] for ep in good], dtype=float)
    n = len(f)
    f_est = float(np.median(f) if fitness_estimator == "median" else np.mean(f))
    d_est = tuple(float(x) for x in (np.median(d, axis=0) if descriptor_estimator == "median"
                                     else np.mean(d, axis=0)))
    cell = grid.cell_of(d_est)
    cells = [grid.cell_of(row) for row in d]
    sd = float(np.std(f, ddof=1)) if n >= 2 else None
    var = np.var(d, axis=0, ddof=1) if n >= 2 else None
    return {
        "n": n,
        "nonfinite_episodes": 0,
        "valid": True,
        "fitness_estimate": f_est,
        "fitness_mean": float(np.mean(f)),
        "fitness_median": float(np.median(f)),
        "fitness_sd": sd,
        "fitness_sem": sd / math.sqrt(n) if sd is not None else None,
        "descriptor_estimate": list(d_est),
        "descriptor_mean": [float(x) for x in np.mean(d, axis=0)],
        "descriptor_var": [float(x) for x in var] if var is not None else None,
        "ndv": float(-np.sum(var)) if var is not None else None,
        "corrected_cell": list(cell),
        "p_in_corrected_cell": float(np.mean([c == cell for c in cells])),
        "search_cell": list(search_cell) if search_cell is not None else None,
        "p_in_search_cell": (float(np.mean([c == tuple(search_cell) for c in cells]))
                             if search_cell is not None else None),
        "mean_episode_length": float(np.mean([ep["length"] for ep in good])),
    }


def _norm(f, bounds):
    lo, hi = bounds
    return float(np.clip((f - lo) / (hi - lo), 0.0, 1.0))


def _entity_episodes(entry, episodes_by_key):
    """Episodes for a solution entry; genes give one list per replicate."""
    if entry["kind"] == "gene":
        return [episodes_by_key.get(f"{entry['solution']}#{rep['train_seed']}")
                for rep in entry.get("replicates", [])]
    return [episodes_by_key.get(entry["solution"])]


def _gene_entity(per_replicate, grid):
    """Gene-level entity: each replicate's held-out estimate counts once."""
    valid = [r for r in per_replicate if r["valid"]]
    if len(valid) != len(per_replicate) or not valid:
        return {"valid": False, "replicates": per_replicate,
                "nonfinite_episodes": sum(r.get("nonfinite_episodes", 0) for r in per_replicate)}
    f = np.array([r["fitness_estimate"] for r in valid])
    d = np.array([r["descriptor_estimate"] for r in valid])
    out = dict(valid[0])
    d_est = [float(x) for x in np.mean(d, axis=0)]
    out.update({
        "fitness_estimate": float(np.mean(f)),
        "descriptor_estimate": d_est,
        # Placement uses the across-replicate descriptor estimate.
        "corrected_cell": list(grid.cell_of(d_est)),
        "across_replicate_fitness_sd": float(np.std(f, ddof=1)) if len(f) >= 2 else None,
        "fitness_sd": float(np.mean([r["fitness_sd"] for r in valid if r["fitness_sd"] is not None]))
        if any(r["fitness_sd"] is not None for r in valid) else None,
        "replicates": per_replicate,
        "p_in_corrected_cell": float(np.mean([r["p_in_corrected_cell"] for r in valid])),
        "p_in_search_cell": float(np.mean([r["p_in_search_cell"] for r in valid])),
        "ndv": float(np.mean([r["ndv"] for r in valid if r["ndv"] is not None]))
        if any(r["ndv"] is not None for r in valid) else None,
    })
    return out


def compute_final_metrics(snapshot, episodes_by_key, final_cfg, holdout_seeds):
    grid = Grid(snapshot["grid"]["descriptors"], snapshot["grid"]["bins"])
    fe, de = final_cfg["fitness_estimator"], final_cfg["descriptor_estimator"]
    bounds = tuple(final_cfg["qd_score_bounds"])
    by_cell = {}
    for entry in snapshot["archive"]:
        by_cell.setdefault(tuple(entry["cell"]), []).append(entry)

    entities = {"best_only": []}
    if snapshot["strategy"] == "deep_grid":
        entities["in_cell_selector"] = []
    for cell, members in sorted(by_cell.items()):
        members = sorted(members, key=lambda e: e["rank"])
        elite_sid = snapshot["elites"][_cell_key(cell)]
        elite = next(m for m in members if m["solution"] == elite_sid)
        eps = _entity_episodes(elite, episodes_by_key)
        if any(e is None for e in eps):
            stats = {"valid": False, "missing_heldout_results": True}
        elif elite["kind"] == "gene":
            stats = _gene_entity([episode_stats(e, grid, cell, fe, de) for e in eps], grid)
        else:
            stats = episode_stats(eps[0], grid, cell, fe, de)
        entities["best_only"].append({"search_cell": list(cell), "solution": elite_sid,
                                      "search_estimate": elite["estimate"], **stats})
        if "in_cell_selector" in entities:
            fits = [m["estimate"]["fitness_mean"] for m in members]
            p = DeepGridArchive.roulette_probabilities(fits)
            rng = np.random.default_rng(derive_int(final_cfg["master_seed"], "in_cell", _cell_key(cell)))
            picks = rng.choice(len(members), size=len(holdout_seeds), p=p)
            drawn, missing = [], False
            for j, idx in enumerate(picks):
                member_eps = episodes_by_key.get(members[idx]["solution"])
                if member_eps is None:
                    missing = True
                    break
                drawn.append(member_eps[j])
            stats = ({"valid": False, "missing_heldout_results": True} if missing
                     else episode_stats(drawn, grid, cell, fe, de))
            entities["in_cell_selector"].append({
                "search_cell": list(cell), "members": [m["solution"] for m in members],
                "member_draw_counts": np.bincount(picks, minlength=len(members)).tolist(), **stats})

    # Illusory (search-time) archive from the search estimates of the elites.
    elite_entries = [next(m for m in by_cell[tuple(map(int, k.split(",")))] if m["solution"] == sid)
                     for k, sid in snapshot["elites"].items()]
    illusory_norm = sum(_norm(e["estimate"]["fitness_mean"], bounds) for e in elite_entries)
    illusory = {
        "filled_cells": len(elite_entries),
        "coverage": len(elite_entries) / grid.n_cells,
        "qd_score_raw": sum(e["estimate"]["fitness_mean"] for e in elite_entries),
        "qd_score_normalized": illusory_norm,
    }

    reports = {}
    for name, ents in entities.items():
        valid = [e for e in ents if e.get("valid")]
        corrected = {}
        for e in valid:
            c = tuple(e["corrected_cell"])
            if c not in corrected or e["fitness_estimate"] > corrected[c]["fitness_estimate"]:
                corrected[c] = e
        kept = list(corrected.values())
        qd_norm = sum(_norm(e["fitness_estimate"], bounds) for e in kept)
        sds = [e["fitness_sd"] for e in kept if e.get("fitness_sd") is not None]
        sems = [e["fitness_sem"] for e in kept if e.get("fitness_sem") is not None]
        ndvs = [e["ndv"] for e in kept if e.get("ndv") is not None]
        reports[name] = {
            "entities": len(ents),
            "heldout_candidate_failures": sum(1 for e in ents if e.get("nonfinite_episodes")),
            "missing_heldout_results": sum(1 for e in ents if e.get("missing_heldout_results")),
            "filled_cells": len(kept),
            "coverage": len(kept) / grid.n_cells,
            "qd_score_raw": float(sum(e["fitness_estimate"] for e in kept)),
            "qd_score_normalized": qd_norm,
            "qd_score_loss": ((illusory_norm - qd_norm) / illusory_norm) if illusory_norm > 0 else None,
            "max_fitness": max((e["fitness_estimate"] for e in kept), default=None),
            "mean_fitness": float(np.mean([e["fitness_estimate"] for e in kept])) if kept else None,
            "p_score": float(sum(e["p_in_corrected_cell"] for e in kept)),
            "mean_p_in_cell": float(np.mean([e["p_in_corrected_cell"] for e in kept])) if kept else None,
            "search_cell_retention": float(np.mean([e["p_in_search_cell"] for e in valid])) if valid else None,
            "correct_cell_fraction": (float(np.mean([tuple(e["corrected_cell"]) == tuple(e["search_cell"])
                                                     for e in valid])) if valid else None),
            "fitness_sd_median": float(np.median(sds)) if sds else None,
            "fitness_sem_mean": float(np.mean(sems)) if sems else None,
            "ndv_mean": float(np.mean(ndvs)) if ndvs else None,
            "corrected_archive": {_cell_key(c): {"solution": e.get("solution"),
                                                 "fitness_estimate": e["fitness_estimate"],
                                                 "descriptor_estimate": e["descriptor_estimate"],
                                                 "search_cell": e["search_cell"],
                                                 "fitness_sd": e.get("fitness_sd"),
                                                 "descriptor_var": e.get("descriptor_var"),
                                                 "p_in_corrected_cell": e.get("p_in_corrected_cell")}
                                  for c, e in sorted(corrected.items())},
        }
    return {"illusory": illusory, "reports": reports, "entities": entities,
            "settings": {"episodes": len(holdout_seeds), "fitness_estimator": fe,
                         "descriptor_estimator": de, "qd_score_bounds": list(bounds),
                         "master_seed": final_cfg["master_seed"]}}
