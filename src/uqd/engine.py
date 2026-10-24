"""Uncertain-QD engine: sample records, archive strategy, budget and logging.

The engine decides *what* to evaluate and *how* results change the archive;
it never runs a simulation itself. The driver (run_improved.py, or
scripts/uqd_smoke.py without an LLM) asks it for a plan, runs the requested
training and rollout jobs, and feeds the outputs back.

One generation, in order (UQD Alg. 1/2 for the archive-sampling family):

1. ``plan_generation``     offspring count and samples per offspring. For
                           Parallel-Adaptive-sampling N is computed from the
                           archive *before* it is re-evaluated (Alg. 2 l. 4).
2. ``reeval_requests``     one rollout request per archived policy when a
                           re-evaluation is due; results go to
                           ``ingest_rollout_results`` and then
                           ``apply_reevaluation`` rebuilds/relocates.
3. ``select_parents``      strategy-specific parent selection.
4. ``register_offspring``  training specs (seeds fixed here) for each new gene;
                           results go to ``ingest_train_stats``.
5. ``insert_offspring``    archive insertion of the eligible new solutions.
6. ``end_generation``      JSON snapshot of archive, estimates and costs.

Identities
----------
A *policy* solution is one trained checkpoint: ``<gene>@t<train_seed>``. Its
samples are the episodes of that checkpoint under one evaluation protocol;
the SHA-256 of the saved weights is recorded and checked on every
re-evaluation. A *gene* solution (sample_unit = retrain) is a gene id whose
samples are training replicates, each summarised by its own episodes.

The initial evaluation episodes produced by the training job are the first
samples of the policy's rollout stream (indices 0..N-1). They are stored one
episode per sample; the 10-episode aggregate in the results CSV is never
added as a sample, so nothing is double counted.
"""

import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .archive import DeepGridArchive, DepthArchive, Grid, MapElitesArchive
from .budget import CostLedger
from .seeds import SeedScheme
from .stats import (CANDIDATE_FAILURE, INFRA_FAILURE, OK, GeneStats, ReplicateSummary,
                    RolloutSample, RolloutStats, outcome_class, protocol_id)

ELIGIBLE = "eligible"
PENDING = "pending"
CANDIDATE_FAILED = "candidate_failed"
INFRA_UNEVALUATED = "infra_unevaluated"


class BudgetUndefined(RuntimeError):
    """The UQD sampling-size rule leaves no budget for offspring."""


@dataclass
class TrainSpec:
    gene_id: str
    solution_id: str          # the policy solution this training run creates
    replicate: int
    train_seed: int
    eval_seed: int            # first reset seed; episodes use eval_seed + k
    eval_episodes: int
    attempts: int = 0


@dataclass
class Solution:
    sid: str
    gene_id: str
    kind: str                         # "policy" | "gene"
    created_generation: int
    status: str = PENDING
    train_seed: Optional[int] = None
    model_path: Optional[str] = None
    checkpoint_id: Optional[str] = None
    param_count: Optional[int] = None
    rollouts: Optional[RolloutStats] = None
    gene_stats: Optional[GeneStats] = None
    pending_replicates: List[int] = field(default_factory=list)
    failure_kind: Optional[str] = None
    lost_reevals: int = 0
    parent: Optional[str] = None
    origin: str = "search"            # search | aria


def git_revision(repo_root):
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
                             text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=repo_root, capture_output=True, text=True, timeout=30).stdout.strip()
        return {"commit": rev or None, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def make_archive(cfg, grid):
    strategy = cfg["archive_strategy"]
    if strategy == "map_elites":
        return MapElitesArchive(grid, cfg["replacement"]["rule"], cfg["replacement"]["margin"])
    if strategy in ("archive_sampling", "parallel_adaptive_sampling"):
        return DepthArchive(grid, cfg["depth"])
    if strategy == "deep_grid":
        return DeepGridArchive(grid, cfg["depth"], cfg["deep_grid_replacement"])
    raise ValueError(f"the engine does not run archive_strategy={strategy!r}")


class UQDEngine:
    def __init__(self, cfg, descriptors, bins, run_seed, protocol, run_dir=None,
                 initial_population_size=None):
        if cfg["archive_strategy"] == "legacy_map_elites":
            raise ValueError("legacy_baseline runs the original run_improved.py code path")
        self.cfg = cfg
        self.grid = Grid(descriptors, bins)
        self.descriptor_names = [name for name, _, _ in descriptors]
        self.seeds = SeedScheme(run_seed, cfg["final_eval"]["master_seed"], search_namespace=cfg["name"])
        self.protocol = protocol
        self.protocol_id = protocol_id(protocol)
        self.archive = make_archive(cfg, self.grid)
        self.rng = np.random.default_rng(self.seeds.search_seed() + 1)
        self.ledger = CostLedger()
        self.solutions = {}
        self.retired = {}
        self.generation = 0
        self.initial_population_size = initial_population_size
        self.run_dir = run_dir
        self.plans = {}
        self.finished_reason = None
        self.started_at = time.time()
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)

    # ------------------------------------------------------------- logging
    def log(self, event, **fields):
        row = {"generation": self.generation, "event": event, "time": time.time()}
        row.update(fields)
        if self.run_dir:
            with open(os.path.join(self.run_dir, "events.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        return row

    def _log_archive_events(self, events):
        for ev in events:
            fields = dict(ev)
            kind = fields.pop("event")
            self.log(f"archive_{kind}", **fields)
            if kind == "retired":
                self.retired[ev["solution"]] = {"reason": ev.get("reason"), "generation": self.generation,
                                                "cell": ev.get("cell")}

    def write_run_metadata(self, repo_root, extra=None):
        if not self.run_dir:
            return
        meta = {
            "preset": self.cfg["name"],
            "category": self.cfg["category"],
            "resolved_config": self.cfg,
            "seeds": self.seeds.describe(),
            "evaluation_protocol": self.protocol,
            "protocol_id": self.protocol_id,
            "grid": {"descriptors": self.grid.descriptors, "bins": self.grid.bins},
            "code_revision": git_revision(repo_root),
            "python": sys.version,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if extra:
            meta.update(extra)
        with open(os.path.join(self.run_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

    # ----------------------------------------------------------- estimates
    def estimate(self, sid):
        sol = self.solutions[sid]
        if sol.kind == "gene":
            return sol.gene_stats.estimate()
        return sol.rollouts.estimate(self.cfg["sample_window"])

    def fitness_of(self, sid):
        return self.estimate(sid).fitness_mean

    def descriptor_of(self, sid):
        return self.estimate(sid).descriptor_mean

    def sample_count(self, sid):
        sol = self.solutions[sid]
        if sol.kind == "gene":
            return len(sol.gene_stats.replicate_estimates())
        return sol.rollouts.count(self.cfg["sample_window"])

    def gene_of(self, sid):
        return self.solutions[sid].gene_id

    # ------------------------------------------------------------ planning
    def reeval_due(self, generation):
        r = self.cfg["archive_reeval"]
        return (r["enabled"] and generation >= r["start_generation"]
                and (generation - r["start_generation"]) % r["every"] == 0)

    def offspring_samples(self):
        """Samples per new solution this generation (before re-evaluation)."""
        if self.cfg["offspring_sampling"] == "pas_median":
            counts = [self.sample_count(sid) for sid in self.archive.members()]
            n = int(np.median(counts)) if counts else 1      # floor, as int(nanmedian)
            n = max(n, self.cfg["pas_min_samples"])
            if self.cfg["pas_max_samples"] is not None:
                n = min(n, self.cfg["pas_max_samples"])
            return max(1, n)
        return self.cfg["initial_samples"]

    def plan_generation(self, generation, initial=False):
        self.generation = generation
        n = self.offspring_samples()
        budget = self.cfg["budget"]
        reeval_count = 0
        if not initial and self.reeval_due(generation):
            reeval_count = len(self.archive.members()) * self.cfg["archive_reeval"]["samples"]
        if budget["mode"] == "sampling_size":
            size = budget["sampling_size"]
            if initial:
                n_off = size // n
            else:
                n_off = (size - reeval_count) // n
                # UQD reference: lower N until at least one offspring fits.
                while n_off < 1 and n > 1 and self.cfg["offspring_sampling"] == "pas_median":
                    n -= 1
                    n_off = (size - reeval_count) // n
            if n_off < 1:
                raise BudgetUndefined(
                    f"sampling_size={size} cannot cover {reeval_count} archive re-evaluations plus "
                    f"one offspring with N={n}; the UQD paper leaves this case undefined")
        else:
            n_off = (self.initial_population_size if initial and self.initial_population_size
                     else budget["offspring_per_generation"])
        if budget["max_offspring_per_generation"] is not None:
            n_off = min(n_off, budget["max_offspring_per_generation"])
        plan = {"generation": generation, "offspring": int(n_off), "samples_per_offspring": int(n),
                "planned_reeval_samples": int(reeval_count), "initial": initial}
        self.plans[generation] = plan
        self.log("plan", **plan)
        return plan

    def budget_exhausted(self):
        limit = self.cfg["budget"]["total_env_steps"]
        return limit is not None and self.ledger.search_env_steps() >= limit

    # ----------------------------------------------------------- offspring
    def register_offspring(self, gene_id, generation, n_samples):
        """Create the solution for a new gene and return its training specs.

        Returns [] for a gene that is already registered (an unmutated clone
        of its parent): it is not re-trained and consumes no budget.
        """
        unit = self.cfg["sample_unit"]
        if any(s.gene_id == gene_id for s in self.solutions.values()):
            self.log("duplicate_offspring_ignored", gene=gene_id)
            return []
        if unit == "rollout":
            train_seed = self.seeds.train_seed(0)
            sid = f"{gene_id}@t{train_seed}"
            sol = Solution(sid, gene_id, "policy", generation, train_seed=train_seed,
                           rollouts=RolloutStats(checkpoint_id="", protocol_id=self.protocol_id),
                           pending_replicates=[0])
            self.solutions[sid] = sol
            seeds = self.seeds.rollout_seeds(sid, 0, n_samples)
            specs = [TrainSpec(gene_id, sid, 0, train_seed, seeds[0], n_samples)]
        else:
            sol = Solution(gene_id, gene_id, "gene", generation, gene_stats=GeneStats(gene_id),
                           pending_replicates=list(range(n_samples)))
            self.solutions[gene_id] = sol
            k = self.cfg["episodes_per_replicate"]
            specs = []
            for r in range(n_samples):
                train_seed = self.seeds.train_seed(r)
                rep_sid = f"{gene_id}@t{train_seed}"
                seeds = self.seeds.rollout_seeds(rep_sid, 0, k)
                specs.append(TrainSpec(gene_id, rep_sid, r, train_seed, seeds[0], k))
        self.log("offspring_registered", gene=gene_id, solution=sol.sid,
                 specs=[s.__dict__ for s in specs])
        return specs

    def _episode_samples(self, episodes, start_index, source, generation):
        samples = []
        for k, ep in enumerate(episodes):
            if ep.get("nonfinite"):
                samples.append(RolloutSample(seed=int(ep["seed"]), index=start_index + k,
                                             status=CANDIDATE_FAILURE, kind="nonfinite_rollout",
                                             length=ep.get("length"), source=source,
                                             generation=generation))
            else:
                samples.append(RolloutSample(
                    seed=int(ep["seed"]), index=start_index + k, status=OK,
                    fitness=float(ep["return"]),
                    descriptor=tuple(float(ep["descriptor"][n]) for n in self.descriptor_names),
                    length=int(ep["length"]), terminated=bool(ep.get("terminated")),
                    source=source, generation=generation))
        return samples

    def ingest_train_stats(self, spec, stats, generation, infra_kind=None, category=None):
        """Record one training job's outcome.

        ``stats`` is the job's stats JSON (dict) or None when the job produced
        nothing. Returns "ok", "candidate_failure", or "infra_failure"; the
        caller retries infra failures with the same spec (same seeds).
        """
        sol = self.solutions[spec.gene_id if self.cfg["sample_unit"] == "retrain" else spec.solution_id]
        if category is None:
            category = "initial" if spec.replicate == 0 else "retrain"
        train_cat = "initial_train" if category == "initial" else "retrain_train"
        eval_cat = "initial_eval" if category == "initial" else "retrain_eval"

        if stats is None or stats.get("status") == "failed":
            kind = (stats or {}).get("failure_kind") or infra_kind or "missing_artifact"
            outcome = outcome_class(kind)
            self.ledger.record(train_cat, spec.solution_id, generation, env_steps=None,
                               wall_seconds=(stats or {}).get("train_time_sec"), status=outcome,
                               note=kind)
            self.log("train_failed", solution=spec.solution_id, kind=kind, outcome=outcome,
                     attempt=spec.attempts)
            if outcome == CANDIDATE_FAILURE:
                self._disqualify(sol, kind)
            return outcome

        expected = [spec.eval_seed + k for k in range(spec.eval_episodes)]
        if [int(s) for s in stats.get("eval_seeds", [])] != expected:
            raise ValueError(f"{spec.solution_id}: job evaluated seeds {stats.get('eval_seeds')}, "
                             f"expected {expected}; the job was launched with the wrong arguments")
        self.ledger.record(train_cat, spec.solution_id, generation, env_steps=stats["timesteps"],
                           wall_seconds=stats.get("train_wall_sec"), status=OK)
        self.ledger.record(eval_cat, spec.solution_id, generation,
                           env_steps=stats.get("eval_env_steps"), episodes=len(expected),
                           wall_seconds=stats.get("eval_wall_sec"), status=OK)
        rollouts = RolloutStats(checkpoint_id=stats["checkpoint_sha256"], protocol_id=protocol_id(stats["protocol"]))
        if rollouts.protocol_id != self.protocol_id:
            raise ValueError(f"{spec.solution_id}: evaluated under protocol {stats['protocol']}, "
                             f"engine expects {self.protocol}")
        for sample in self._episode_samples(stats["episodes"], 0, "initial", generation):
            rollouts.add(sample)

        if sol.kind == "policy":
            sol.rollouts = rollouts
            sol.model_path = stats["model_path"]
            sol.checkpoint_id = stats["checkpoint_sha256"]
            sol.param_count = stats.get("param_count")
        else:
            sol.gene_stats.add(ReplicateSummary(spec.train_seed, stats["checkpoint_sha256"],
                                                stats["model_path"], rollouts, stats["timesteps"]))
            sol.param_count = stats.get("param_count")
        if spec.replicate in sol.pending_replicates:
            sol.pending_replicates.remove(spec.replicate)

        if rollouts.has_candidate_failure:
            self._disqualify(sol, "nonfinite_rollout")
            return CANDIDATE_FAILURE
        if not sol.pending_replicates and sol.status == PENDING:
            sol.status = ELIGIBLE
        self.log("train_ingested", solution=spec.solution_id, status=sol.status,
                 estimate=self.estimate(sol.sid).to_dict() if sol.status == ELIGIBLE else None)
        return OK

    def train_lost(self, spec, generation, kind):
        """A training request exhausted its retries on infrastructure failures."""
        sid = spec.gene_id if self.cfg["sample_unit"] == "retrain" else spec.solution_id
        sol = self.solutions[sid]
        if sol.status != CANDIDATE_FAILED:
            sol.status = INFRA_UNEVALUATED
            sol.failure_kind = kind
        self.log("train_lost", solution=spec.solution_id, kind=kind)

    def _disqualify(self, sol, kind):
        sol.status = CANDIDATE_FAILED
        sol.failure_kind = kind
        events = []
        if sol.sid in self.archive:
            events = self.archive.remove(sol.sid, reason=f"candidate_failure:{kind}")
        self._log_archive_events(events)
        self.log("disqualified", solution=sol.sid, kind=kind)

    def insert_offspring(self, gene_ids, generation):
        """Archive insertion for new genes, in the given order."""
        events = []
        for gene in gene_ids:
            sid = self.solution_for_gene(gene)
            if sid is None:
                continue
            sol = self.solutions[sid]
            if sol.status != ELIGIBLE or sid in self.archive or sol.created_generation != generation:
                continue
            fitness, descriptor = self.fitness_of(sid), self.descriptor_of(sid)
            if isinstance(self.archive, DeepGridArchive):
                ev = self.archive.insert(sid, fitness, descriptor, self.fitness_of, self.rng)
            else:
                ev = self.archive.insert(sid, fitness, descriptor, self.fitness_of)
            events.extend(ev)
        self._log_archive_events(events)
        return events

    def solution_for_gene(self, gene_id):
        for sid, sol in self.solutions.items():
            if sol.gene_id == gene_id and sol.origin == "search":
                return sid
        return None

    # --------------------------------------------------------- re-evaluation
    def reeval_requests(self, generation):
        if not self.reeval_due(generation):
            return []
        k = self.cfg["archive_reeval"]["samples"]
        requests = []
        for sid in self.archive.members():
            sol = self.solutions[sid]
            start = sol.rollouts.next_index
            seeds = self.seeds.rollout_seeds(sid, start, k)
            requests.append({
                "request_id": f"g{generation}:{sid}:{start}",
                "solution_id": sid,
                "model_path": sol.model_path,
                "checkpoint_sha256": sol.checkpoint_id,
                "seeds": seeds,
                "start_index": start,
            })
        self.log("reeval_requested", count=len(requests), samples_each=k)
        return requests

    def ingest_rollout_results(self, results, generation, source="reeval", category="reeval"):
        """Record rollout results; return the infra-failed results for retry."""
        failed = []
        for res in results:
            sid = res["solution_id"]
            sol = self.solutions[sid]
            if res["status"] != OK:
                kind = res.get("failure_kind", "malformed_result")
                self.ledger.record(category, sid, generation, env_steps=None,
                                   wall_seconds=res.get("wall_seconds"), status=INFRA_FAILURE, note=kind)
                failed.append(res)
                continue
            self.ledger.record(category, sid, generation, env_steps=res.get("env_steps"),
                               episodes=len(res["episodes"]), wall_seconds=res.get("wall_seconds"),
                               status=OK)
            got = [int(ep["seed"]) for ep in res["episodes"]]
            if got != [int(s) for s in res["seeds"]]:
                raise ValueError(f"{sid}: rollout seeds {got} do not match the request {res['seeds']}")
            for sample in self._episode_samples(res["episodes"], res["start_index"], source, generation):
                sol.rollouts.add(sample)
            if sol.rollouts.has_candidate_failure:
                self._disqualify(sol, "nonfinite_rollout")
        return failed

    def reeval_lost(self, result, generation):
        """A re-evaluation exhausted its retries: keep the previous estimate."""
        sol = self.solutions[result["solution_id"]]
        sol.lost_reevals += 1
        self.log("reeval_lost", solution=sol.sid, kind=result.get("failure_kind"),
                 kept_estimate=True, lost_total=sol.lost_reevals)

    def apply_reevaluation(self, generation):
        members = self.archive.members()
        if isinstance(self.archive, DepthArchive):
            estimates = {sid: (self.fitness_of(sid), self.descriptor_of(sid)) for sid in members}
            events = self.archive.rebuild(estimates, self.fitness_of)
        elif isinstance(self.archive, MapElitesArchive):
            events = []
            for sid in members:          # sorted by cell; all estimates already updated
                if sid in self.archive:
                    events.extend(self.archive.relocate(sid, self.fitness_of(sid),
                                                        self.descriptor_of(sid), self.fitness_of))
        else:
            events = []
        self._log_archive_events(events)
        return events

    # ------------------------------------------------------------ selection
    def select_parents(self, n):
        sids = self.archive.select_parents(n, self.rng, self.fitness_of)
        self.log("parents_selected", parents=sids)
        return [self.solutions[s].gene_id for s in sids]

    # ------------------------------------------------------------- snapshot
    def solution_summary(self, sid):
        sol = self.solutions[sid]
        est = self.estimate(sid) if sol.status == ELIGIBLE else None
        row = {
            "solution": sid, "gene": sol.gene_id, "kind": sol.kind, "status": sol.status,
            "created_generation": sol.created_generation, "train_seed": sol.train_seed,
            "model_path": sol.model_path, "checkpoint_sha256": sol.checkpoint_id,
            "param_count": sol.param_count, "failure_kind": sol.failure_kind,
            "lost_reevals": sol.lost_reevals, "origin": sol.origin, "parent": sol.parent,
            "estimate": est.to_dict() if est else None,
            "samples_total": (len(sol.rollouts.samples) if sol.rollouts is not None else None),
        }
        if sol.kind == "gene":
            row["replicates"] = [{
                "train_seed": r.train_seed, "checkpoint_sha256": r.checkpoint_id,
                "model_path": r.model_path,
                "estimate": r.summary().to_dict() if r.summary() else None,
            } for r in sol.gene_stats.replicates]
            row["within_run_fitness_sd"] = sol.gene_stats.within_run_fitness_sd()
        return row

    def snapshot(self):
        archive = []
        for cell in self.archive.filled_cells():
            for rank, sid in enumerate(self.archive.cells[cell]):
                archive.append({"cell": list(cell), "rank": rank, **self.solution_summary(sid)})
        status_counts = {}
        for sol in self.solutions.values():
            status_counts[sol.status] = status_counts.get(sol.status, 0) + 1
        elites = self.archive.elites(self.fitness_of)
        qd_raw = sum(self.fitness_of(sid) for sid in elites.values())
        return {
            "preset": self.cfg["name"],
            "strategy": self.cfg["archive_strategy"],
            "generation": self.generation,
            "grid": {"descriptors": self.grid.descriptors, "bins": self.grid.bins},
            "archive": archive,
            "elites": {",".join(map(str, c)): sid for c, sid in sorted(elites.items())},
            "illusory_metrics": {
                "filled_cells": len(elites),
                "coverage": len(elites) / self.grid.n_cells,
                "qd_score_raw": qd_raw,
                "archived_solutions": len(self.archive.members()),
            },
            "solution_status_counts": status_counts,
            "retired": self.retired,
            "cost": self.ledger.summary(),
            # Wall time since the engine was created, including queue waits
            # and time between resumed driver processes.
            "driver_wall_seconds_since_start": time.time() - getattr(self, "started_at", time.time()),
            "plan": self.plans.get(self.generation),
            "finished_reason": self.finished_reason,
        }

    def end_generation(self, generation):
        snap = self.snapshot()
        self.log("generation_end", illusory=snap["illusory_metrics"],
                 search_env_steps=self.ledger.search_env_steps())
        if self.run_dir:
            for name in (f"archive_gen_{generation}.json", "archive_latest.json"):
                with open(os.path.join(self.run_dir, name), "w", encoding="utf-8") as f:
                    json.dump(snap, f, indent=1, default=str)
            with open(os.path.join(self.run_dir, "solutions.json"), "w", encoding="utf-8") as f:
                json.dump([self.solution_summary(s) for s in self.solutions], f, default=str)
            with open(os.path.join(self.run_dir, "cost_ledger.json"), "w", encoding="utf-8") as f:
                json.dump(self.ledger.to_list(), f)
            self.save_state(os.path.join(self.run_dir, "engine_state.pkl"))
        return snap

    def save_state(self, path):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self, f)
        os.replace(tmp, path)

    @staticmethod
    def load_state(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def search_seeds_used(self):
        """Every reset and training seed that informed a search decision."""
        used = set()
        for sol in self.solutions.values():
            if sol.train_seed is not None:
                used.add(sol.train_seed)
            if sol.rollouts is not None:
                used.update(sol.rollouts.seeds())
            if sol.gene_stats is not None:
                for r in sol.gene_stats.replicates:
                    used.add(r.train_seed)
                    used.update(r.rollouts.seeds())
        return used
