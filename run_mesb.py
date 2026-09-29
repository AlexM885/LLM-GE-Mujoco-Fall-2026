#!/usr/bin/env python
"""LLM-Guided Evolution on MuJoCo HalfCheetah with MAP-Elites Sliding Boundaries.

Three explicit selection modes (see docs/MESB.md):

``nsga2``
    Legacy behaviour: SPEA2 elites + NSGA-II/DCD-tournament parent selection
    over (maximise mean_reward, minimise param_count), read *by name*.
``mesb-shadow``
    Identical parent selection to ``nsga2``; every successful evaluation is
    *also* inserted into a Sliding-Boundaries archive that is logged but never
    consulted. Validates the archive without changing evolution.
``mesb``
    Parents are elites sampled uniformly from occupied archive cells; the
    same LLM crossover/mutation operators produce children, which are trained,
    evaluated and inserted into the archive.

Example (tiny local smoke run, no LLM / MuJoCo)::

    python run_mesb.py --run-name smoke --selection-mode mesb \
        --llm-backend mock --evaluator mock --num-generations 3 \
        --start-population-size 16 --population-size 8 --mesb-dims 5 5 \
        --mesb-remap-frequency 10
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import json
import os
import pickle
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cfg import constants_mesb as C  # noqa: E402
from src.mesb_variation import Variation, vary  # noqa: E402
from sota.MujocoRL.mesb_archive import (FixedGridArchive, SlidingBoundariesArchive,  # noqa: E402
                                        archive_from_state_dict)
from sota.MujocoRL.mesb_evaluator import EvalJob, EvalSettings, make_evaluator  # noqa: E402
from sota.MujocoRL.mesb_logging import (EVALUATION_COLUMNS, METRIC_COLUMNS, RunPaths,  # noqa: E402
                                        append_csv, append_jsonl, boundary_record,
                                        evaluation_row, truncate_after_generation,
                                        write_snapshot)
from sota.MujocoRL.results_io import EvaluationOutcome, read_outcome  # noqa: E402

SELECTION_MODES = ("nsga2", "mesb-shadow", "mesb")
CHECKPOINT_FORMAT = "mesb-run-v1"
#: Settings that define the experiment; a resumed run must not change them.
LOCKED_KEYS = (
    "selection_mode", "archive_boundaries", "mesb_dims", "mesb_remap_frequency",
    "mesb_buffer_capacity", "mesb_remap_at_generation_end", "fixed_ranges", "qd_score_offset",
    "seed", "eval_seed", "env_id", "eval_timesteps", "eval_episodes", "eval_max_steps",
    "start_population_size", "population_size", "num_elites", "crossover_probability",
    "mutation_probability", "llm_backend", "llm_model", "llm_crossover_impl", "evaluator",
)


class CheckpointMismatchError(RuntimeError):
    """Resuming with settings that differ from the checkpointed experiment."""


# ============================================================ legacy NSGA-II
class LegacyNSGA2:
    """Parent selection exactly as run_improved.py (USE_MAP_ELITES = False):

    elites  = selSPEA2(valid, num_elites)
    parents = selTournamentDCD(selNSGA2(valid, len(valid)), k)

    with k = population_size, or len(valid) rounded down to a multiple of 4
    when fewer valid individuals exist. Objectives come from the result JSON
    by name (C.NSGA2_OBJECTIVES), not by CSV column position.
    """

    def __init__(self, num_elites: int) -> None:
        from deap import base, creator, tools
        weights = tuple(w for _, w in C.NSGA2_OBJECTIVES)
        if not hasattr(creator, "FitnessMESBLegacy"):
            creator.create("FitnessMESBLegacy", base.Fitness, weights=weights)
        if not hasattr(creator, "IndividualMESBLegacy"):
            creator.create("IndividualMESBLegacy", list, fitness=creator.FitnessMESBLegacy)
        self.creator, self.tools, self.num_elites = creator, tools, num_elites

    def individual(self, gene_id: str, fitness: tuple[float, ...] | None):
        ind = self.creator.IndividualMESBLegacy([gene_id])
        if fitness is not None:
            ind.fitness.values = fitness
        return ind

    def select(self, population: list[tuple[str, tuple | None]], population_size: int
               ) -> tuple[list[str], list[str]]:
        valid = [self.individual(g, f) for g, f in population if f is not None]
        if len(valid) < 4:
            raise RuntimeError(
                f"only {len(valid)} valid individuals; NSGA-II/DCD selection needs >= 4. "
                "Check logs/eval/*.log for why evaluations are failing.")
        elites = self.tools.selSPEA2(valid, self.num_elites)
        k = population_size if len(valid) >= population_size else len(valid) - len(valid) % 4
        ranked = self.tools.selNSGA2(valid, len(valid))
        parents = self.tools.selTournamentDCD(ranked, k)
        return [p[0] for p in parents], [e[0] for e in elites]


def nsga2_fitness(outcome: EvaluationOutcome) -> tuple[float, ...] | None:
    if not outcome.ok:
        return None
    return tuple(float(outcome.payload[name]) for name, _ in C.NSGA2_OBJECTIVES)


# ============================================================ plans
@dataclass
class GenerationPlan:
    """Everything decided before evaluation; persisted so a crash mid-generation
    resumes with the same children instead of drawing new ones."""

    generation: int
    parents: list[str]
    elites: list[str]
    slots: list[str]
    variation: list[dict]
    new_genes: list[dict]
    rng_state: dict = field(default_factory=dict)
    next_eval_index: int = 0


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state()}


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def _atomic_pickle(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)


# ============================================================ provenance
def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _llm_identity() -> dict[str, Any]:
    """Read (without importing torch) which LLM the existing operators use."""
    out: dict[str, Any] = {}
    keys = ("LLM_MODEL", "MODEL_PATH", "LOCAL_LLM", "PORT")
    try:
        path = (REPO_ROOT / "src/cfg/constants.py").resolve()
        out["constants_file"] = path.name
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if getattr(t, "id", None) in keys:
                        try:
                            out[t.id] = ast.literal_eval(node.value)
                        except ValueError:
                            out[t.id] = ast.unparse(node.value)
    except (OSError, SyntaxError):
        pass
    return out


# ============================================================ runner
class MESBRun:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.mode = cfg["selection_mode"]
        self.paths = RunPaths(Path(cfg["output_root"]) / cfg["run_name"])
        self.uses_archive = self.mode in ("mesb-shadow", "mesb")
        self.legacy = LegacyNSGA2(cfg["num_elites"]) if self.mode != "mesb" else None
        self.archive: SlidingBoundariesArchive | None = None
        self.registry: dict[str, dict[str, Any]] = {}   # gene_id -> eval metadata
        self.outcomes: dict[str, EvaluationOutcome] = {}
        self.population: list[str] = []
        self.hof: list[str] = []
        self.next_eval_index = 0
        self.remap_events_written = 0
        self.start_generation = 0

    # ---------------------------------------------------------- setup
    def _new_archive(self) -> SlidingBoundariesArchive:
        c = self.cfg
        if c["archive_boundaries"] == "fixed":
            return FixedGridArchive(dims=c["mesb_dims"], ranges=c["fixed_ranges"],
                                    buffer_capacity=c["mesb_buffer_capacity"], seed=c["seed"],
                                    qd_score_offset=c["qd_score_offset"])
        return SlidingBoundariesArchive(
            dims=c["mesb_dims"], remap_frequency=c["mesb_remap_frequency"],
            buffer_capacity=c["mesb_buffer_capacity"], seed=c["seed"],
            qd_score_offset=c["qd_score_offset"],
            remap_at_generation_end=c["mesb_remap_at_generation_end"])

    def _make_components(self) -> None:
        c = self.cfg
        self.variation = Variation(self.paths.genes, self.paths.prompts, self.paths.llm_logs,
                                   backend=c["llm_backend"], python_cmd=c["llm_python"],
                                   timeout_sec=c["llm_timeout"],
                                   llm_model=c["llm_model"],
                                   crossover_impl=c["llm_crossover_impl"])
        self.evaluator = make_evaluator(c["evaluator"], EvalSettings(
            env_id=c["env_id"], timesteps=c["eval_timesteps"], episodes=c["eval_episodes"],
            max_steps=c["eval_max_steps"], eval_seed=c["eval_seed"], device=c["device"],
            timeout_sec=c["eval_timeout"], workers=c["workers"], python_cmd=c["eval_python"]))

    def _write_config(self) -> None:
        cfg = dict(self.cfg)
        cfg.update({
            "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "objective": "mean_reward",
            "descriptors": ["mean_distance", "mean_control_cost"],
            "archive_seed": self.cfg["seed"],
            "llm": {"backend": self.cfg["llm_backend"], "llm_model": self.cfg["llm_model"],
                    **_llm_identity()},
            "seeds": {"python_random": self.cfg["seed"], "numpy_global": self.cfg["seed"],
                      "archive_rng": self.cfg["seed"],
                      "ppo_train_seed": "seed * 1_000_000 + evaluation_index (per gene)",
                      "eval_episode_seed": "eval_seed + episode_index (same for every gene)"},
        })
        self.paths.config.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    def _check_resume_config(self) -> None:
        stored = json.loads(self.paths.config.read_text(encoding="utf-8"))
        diffs = {k: (stored.get(k), self.cfg.get(k)) for k in LOCKED_KEYS
                 if json.dumps(stored.get(k)) != json.dumps(self.cfg.get(k))}
        if diffs:
            raise CheckpointMismatchError(
                "cannot resume: these settings differ from the checkpointed run "
                f"(stored, requested): {diffs}")

    # ---------------------------------------------------------- checkpoints
    def _latest_checkpoint(self) -> Path | None:
        best = None
        for p in self.paths.checkpoints.glob("checkpoint_gen_*.pkl"):
            m = re.fullmatch(r"checkpoint_gen_(\d+)\.pkl", p.name)
            if m and (best is None or int(m.group(1)) > best[0]):
                best = (int(m.group(1)), p)
        return None if best is None else best[1]

    def save_checkpoint(self, generation: int) -> None:
        state = {
            "format": CHECKPOINT_FORMAT, "generation": generation,
            "selection_mode": self.mode, "config": self.cfg,
            "population": list(self.population), "hof": list(self.hof),
            "registry": self.registry, "next_eval_index": self.next_eval_index,
            "remap_events_written": self.remap_events_written,
            "archive_state": None if self.archive is None else self.archive.state_dict(),
            "rng_state": _rng_state(),
        }
        _atomic_pickle(self.paths.checkpoint(generation), state)

    def load_checkpoint(self, path: Path) -> None:
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise CheckpointMismatchError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
        if state["selection_mode"] != self.mode:
            raise CheckpointMismatchError(
                f"checkpoint selection mode {state['selection_mode']!r} != {self.mode!r}")
        a = state["archive_state"]
        if self.uses_archive:
            if a is None:
                raise CheckpointMismatchError("checkpoint has no archive state")
            self.archive = archive_from_state_dict(a)
            if list(self.archive.dims) != list(self.cfg["mesb_dims"]):
                raise CheckpointMismatchError(
                    f"archive dims {self.archive.dims} != requested {self.cfg['mesb_dims']}")
        self.population = state["population"]
        self.hof = state["hof"]
        self.registry = state["registry"]
        self.next_eval_index = state["next_eval_index"]
        self.remap_events_written = state["remap_events_written"]
        _set_rng_state(state["rng_state"])
        self.outcomes = {g: read_outcome(self.paths.result_json(g), g) for g in self.registry}
        self.start_generation = state["generation"] + 1
        truncate_after_generation(self.paths, state["generation"])
        print(f"Resumed from {path.name}: generation {state['generation']} complete, "
              f"{len(self.registry)} evaluations, archive "
              f"{'-' if self.archive is None else self.archive.num_elites} elites", flush=True)

    # ---------------------------------------------------------- planning
    def _train_seed(self) -> int:
        seed = (self.cfg["seed"] * 1_000_000 + self.next_eval_index) % (2 ** 31 - 1)
        self.next_eval_index += 1
        return seed

    def _select_parents(self) -> tuple[list[str], list[str]]:
        n = self.cfg["population_size"]
        if self.mode == "mesb":
            # Uniform over occupied cells; with replacement only if needed.
            records = self.archive.sample_elites(n, replace=n > self.archive.num_elites)
            return [r.gene_id for r in records], []
        pop = [(g, nsga2_fitness(self.outcomes[g])) for g in self.population]
        return self.legacy.select(pop, n)

    def make_plan(self, generation: int) -> GenerationPlan:
        if generation == 0:
            results = [self.variation.create() for _ in range(self.cfg["start_population_size"])]
            parents, elites, slots = [], [], [r.gene_id for r in results]
        else:
            parents, elites = self._select_parents()
            slots, results = vary(parents, self.variation, self.cfg["crossover_probability"],
                                  self.cfg["mutation_probability"])
        by_id = {r.gene_id: r for r in results}
        new_genes, seen = [], set(self.registry)
        for gid in slots:
            if gid in seen:
                continue  # unchanged clone of an already evaluated gene
            seen.add(gid)
            r = by_id[gid]
            new_genes.append({"gene_id": gid, "operator": r.operator, "parents": r.parents,
                              "train_seed": self._train_seed()})
        plan = GenerationPlan(generation, parents, elites, slots, [asdict(r) for r in results],
                              new_genes, _rng_state(), self.next_eval_index)
        _atomic_pickle(self.paths.plan(generation), asdict(plan))
        append_jsonl(self.paths.selection_log, [{
            "generation": generation, "mode": self.mode, "parents": parents, "elites": elites,
            "slots": slots}])
        append_jsonl(self.paths.variation_log, [{"generation": generation, **asdict(r)}
                                                for r in results])
        return plan

    def load_or_make_plan(self, generation: int) -> GenerationPlan:
        path = self.paths.plan(generation)
        if path.exists():
            with open(path, "rb") as fh:
                plan = GenerationPlan(**pickle.load(fh))
            _set_rng_state(plan.rng_state)
            self.next_eval_index = plan.next_eval_index
            # Re-log the (already made) decisions that truncation removed.
            append_jsonl(self.paths.selection_log, [{
                "generation": generation, "mode": self.mode, "parents": plan.parents,
                "elites": plan.elites, "slots": plan.slots, "reused_plan": True}])
            append_jsonl(self.paths.variation_log, [{"generation": generation, **v}
                                                    for v in plan.variation])
            print(f"Reusing saved plan for generation {generation} "
                  f"({len(plan.new_genes)} children)", flush=True)
            return plan
        return self.make_plan(generation)

    # ---------------------------------------------------------- one generation
    def run_generation(self, generation: int) -> None:
        t0 = time.time()
        plan = self.load_or_make_plan(generation)
        jobs = [
            EvalJob(gene_id=g["gene_id"], gene_file=self.variation.gene_file(g["gene_id"]),
                    generation=generation, train_seed=g["train_seed"],
                    out_json=self.paths.result_json(g["gene_id"]), model_dir=self.paths.models,
                    log_file=self.paths.eval_logs / f"{g['gene_id']}.log")
            for g in plan.new_genes
        ]
        print(f"[gen {generation}] evaluating {len(jobs)} new genes "
              f"({self.cfg['workers']} workers)", flush=True)
        outcomes = self.evaluator.evaluate(jobs)
        for g in plan.new_genes:
            self.registry[g["gene_id"]] = {"generation": generation, **g}
            self.outcomes[g["gene_id"]] = outcomes[g["gene_id"]]
            if not outcomes[g["gene_id"]].ok:
                print(f"\t✗ gene {g['gene_id']} (generation {generation}) failed: "
                      f"{outcomes[g['gene_id']].reason}", flush=True)

        records = [(g, self.outcomes[g["gene_id"]].to_archive_record(generation))
                   for g in plan.new_genes if self.outcomes[g["gene_id"]].ok]
        adds: dict[str, Any] = {}
        if self.uses_archive:
            if generation == 0 or self.archive is None:
                self.archive = self._new_archive()
                # Raises loudly if zero initial individuals succeeded.
                self.archive.initialize([r for _, r in records])
                elites = self.archive.elites
                for _, r in records:
                    cell = self.archive.index_of(r.measures)
                    adds[r.gene_id] = _InitAdd(cell, elites[cell].gene_id == r.gene_id)
                if self.archive.degenerate_dimensions():
                    print(f"\t! archive dimensions {self.archive.degenerate_dimensions()} have "
                          "duplicate boundaries (fewer distinct values than cells); expected "
                          "early in a run", flush=True)
            else:
                for g, r in records:
                    adds[r.gene_id] = self.archive.add(r)
            self.archive.end_generation()
        elif generation == 0 and not records:
            raise RuntimeError("zero initial individuals evaluated successfully")

        # Population bookkeeping (legacy semantics for nsga2 / shadow).
        if self.mode == "mesb":
            self.population = [r.gene_id for r in self.archive.elites.values()]
        else:
            offspring = list(dict.fromkeys(plan.slots + plan.elites))
            self.population = offspring
        self._update_hof()
        self._log_generation(generation, plan, adds, time.time() - t0)
        self.save_checkpoint(generation)

    def _update_hof(self) -> None:
        ok = [g for g, o in self.outcomes.items() if o.ok]
        ok.sort(key=lambda g: (-self.outcomes[g].payload["mean_reward"],
                               self.outcomes[g].param_count or 0, g))
        self.hof = ok[: self.cfg["hof_size"]]

    # ---------------------------------------------------------- logging
    def _log_generation(self, generation: int, plan: GenerationPlan, adds: dict,
                        seconds: float) -> None:
        rows = []
        for g in plan.new_genes:
            o = self.outcomes[g["gene_id"]]
            rows.append(evaluation_row(o, generation, adds.get(g["gene_id"]), g["train_seed"],
                                       g["operator"], g["parents"]))
        append_csv(self.paths.evaluations_csv, EVALUATION_COLUMNS, rows)

        ok = [o for o in self.outcomes.values() if o.ok]
        best = max(ok, key=lambda o: o.payload["mean_reward"], default=None)
        pop_rewards = [self.outcomes[g].payload["mean_reward"] for g in self.population
                       if g in self.outcomes and self.outcomes[g].ok]
        metrics = {
            "generation": generation, "selection_mode": self.mode,
            "boundary_mode": None if self.archive is None else self.archive.boundary_mode,
            "total_evaluations": len(self.outcomes), "successful_evaluations": len(ok),
            "failed_evaluations": len(self.outcomes) - len(ok),
            "new_evaluations": len(plan.new_genes),
            "llm_failures": sum(not v["ok"] for v in plan.variation),
            "best_reward": None if best is None else best.payload["mean_reward"],
            "best_gene_id": None if best is None else best.gene_id,
            "population_size": len(self.population),
            "population_best_reward": max(pop_rewards, default=None),
        }
        if self.archive is not None:
            s = self.archive.summary()
            metrics.update({
                "occupied_cells": s["occupied_cells"], "total_cells": s["total_cells"],
                "coverage": s["coverage"], "raw_qd_score": s["raw_qd_score"],
                "shifted_qd_score": s["shifted_qd_score"],
                "mean_elite_reward": s["mean_elite_objective"],
                "number_of_remaps": s["num_remaps"], "archive_total_seen": s["total_seen"],
            })
            events = self.archive.remap_events[self.remap_events_written:]
            append_jsonl(self.paths.boundaries_jsonl,
                         [boundary_record(self.archive, generation, e["event"], e) for e in events]
                         + [boundary_record(self.archive, generation, "generation_end")])
            self.remap_events_written = len(self.archive.remap_events)
            write_snapshot(self.paths, self.archive, generation)
        append_csv(self.paths.metrics_csv, METRIC_COLUMNS, [metrics])

        msg = (f"[gen {generation}] done in {seconds:.0f}s: {len(plan.new_genes)} new, "
               f"{metrics['successful_evaluations']}/{metrics['total_evaluations']} ok overall, "
               f"best {metrics['best_reward']}")
        if self.archive is not None:
            msg += (f" | archive {metrics['occupied_cells']}/{metrics['total_cells']} "
                    f"({100 * metrics['coverage']:.1f}%), QD {metrics['raw_qd_score']:.1f}, "
                    f"remaps {metrics['number_of_remaps']}")
        print(msg, flush=True)

    # ---------------------------------------------------------- main loop
    def run(self, resume: bool) -> None:
        self.paths.create()
        ckpt = self._latest_checkpoint()
        if self.paths.config.exists() and not resume:
            raise SystemExit(f"{self.paths.run_dir} already exists; pass --resume to continue "
                             "it or choose a new --run-name")
        if resume and self.paths.config.exists():
            self._check_resume_config()
        else:
            self._write_config()
        self._make_components()
        random.seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])
        if resume and ckpt is not None:
            self.load_checkpoint(ckpt)
        elif resume:
            truncate_after_generation(self.paths, -1)
        for generation in range(self.start_generation, self.cfg["num_generations"] + 1):
            self.run_generation(generation)
        print(f"Run complete: {self.paths.run_dir}\n"
              f"  analyse with: python sota/MujocoRL/analyze_mesb.py --run-dir {self.paths.run_dir}",
              flush=True)


@dataclass(frozen=True)
class _InitAdd:
    """Cell assignment of an initial-population record (for evaluations.csv)."""
    cell: tuple[int, ...]
    inserted: bool
    replaced_gene_id: None = None


# ============================================================ CLI
def _capacity(v: str) -> int | None:
    return None if v.lower() in ("none", "inf", "infinity", "0") else int(v)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("run")
    g.add_argument("--run-name", default=None, help="Output subdirectory (default: timestamp)")
    g.add_argument("--output-root", default=str(C.ROOT_DIR / C.OUTPUT_ROOT))
    g.add_argument("--resume", action="store_true", help="Continue an existing run directory")
    g.add_argument("--seed", type=int, default=C.SEED)
    g.add_argument("--selection-mode", choices=SELECTION_MODES, default="mesb")

    g = p.add_argument_group("archive")
    g.add_argument("--mesb-dims", type=int, nargs=2, metavar=("X", "Y"), default=list(C.MESB_DIMS))
    g.add_argument("--mesb-remap-frequency", type=int, default=C.MESB_REMAP_FREQUENCY,
                   help="Remap every N successful evaluations (paper delta)")
    g.add_argument("--mesb-buffer-capacity", type=_capacity, default=C.MESB_BUFFER_CAPACITY,
                   help="Buffer size xi; 'none' = unlimited (paper default)")
    g.add_argument("--mesb-remap-at-generation-end", action="store_true",
                   default=C.MESB_REMAP_AT_GENERATION_END,
                   help="Project adaptation: remap once per generation instead of every N")
    g.add_argument("--qd-score-offset", type=float, default=C.MESB_QD_SCORE_OFFSET)
    g.add_argument("--archive-boundaries", choices=["sliding", "fixed"], default="sliding",
                   help="'fixed' = legacy equal-width grid, for controlled comparison only")
    g.add_argument("--fixed-ranges", type=float, nargs=4, default=None,
                   metavar=("DIST_LO", "DIST_HI", "CTRL_LO", "CTRL_HI"),
                   help="Required with --archive-boundaries fixed")

    g = p.add_argument_group("evaluation")
    g.add_argument("--env-id", default=C.ENV_ID)
    g.add_argument("--eval-timesteps", type=int, default=C.EVAL_TIMESTEPS, help="PPO timesteps")
    g.add_argument("--eval-episodes", type=int, default=C.EVAL_EPISODES)
    g.add_argument("--eval-max-steps", type=int, default=C.EVAL_MAX_STEPS)
    g.add_argument("--eval-seed", type=int, default=C.EVAL_SEED)
    g.add_argument("--eval-timeout", type=float, default=C.EVAL_TIMEOUT_SEC)
    g.add_argument("--device", default=C.EVAL_DEVICE)
    g.add_argument("--workers", type=int, default=int(os.getenv("SLURM_CPUS_PER_TASK", "1")),
                   help="Concurrent gene evaluations")
    g.add_argument("--eval-python", default=sys.executable,
                   help="Python command for evaluations (e.g. 'uv run --project "
                        "sota/MujocoRL/eval_env python')")
    g.add_argument("--evaluator", choices=["subprocess", "mock"], default="subprocess",
                   help="'mock' = synthetic results for plumbing tests only")

    g = p.add_argument_group("evolution")
    g.add_argument("--num-generations", type=int, default=C.NUM_GENERATIONS)
    g.add_argument("--start-population-size", type=int, default=C.START_POPULATION_SIZE)
    g.add_argument("--population-size", type=int, default=C.POPULATION_SIZE)
    g.add_argument("--num-elites", type=int, default=C.NUM_ELITES)
    g.add_argument("--hof-size", type=int, default=C.HOF_SIZE)
    g.add_argument("--crossover-probability", type=float, default=C.CROSSOVER_PROBABILITY)
    g.add_argument("--mutation-probability", type=float, default=C.MUTATION_PROBABILITY)
    g.add_argument("--llm-backend", choices=["llm", "mock"], default="llm",
                   help="'llm' = existing src/llm_mutation.py & llm_crossover.py; "
                        "'mock' = LLM-free edits for plumbing tests only")
    g.add_argument("--llm-python", default=sys.executable)
    g.add_argument("--llm-timeout", type=float, default=C.LLM_TIMEOUT_SEC)
    g.add_argument("--llm-model", default=C.LLM_MODEL,
                   help="Model name passed to the existing operators (run_improved.py --llm_model)")
    g.add_argument("--llm-crossover-impl", choices=["fixed", "original"], default="fixed",
                   help="'original' = src/llm_crossover.py unchanged (off-by-one chunk bug)")
    return p


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = {k: v for k, v in vars(args).items() if k not in ("resume",)}
    cfg["run_name"] = args.run_name or _dt.datetime.now().strftime("mesb_%Y%m%d_%H%M%S")
    cfg["mesb_dims"] = list(args.mesb_dims)
    if args.archive_boundaries == "fixed":
        if args.fixed_ranges is None:
            raise SystemExit("--archive-boundaries fixed needs --fixed-ranges DIST_LO DIST_HI "
                             "CTRL_LO CTRL_HI")
        r = args.fixed_ranges
        cfg["fixed_ranges"] = [[r[0], r[1]], [r[2], r[3]]]
    elif args.fixed_ranges is not None:
        raise SystemExit("--fixed-ranges only applies to --archive-boundaries fixed")
    if args.selection_mode == "nsga2" and args.archive_boundaries == "fixed":
        raise SystemExit("nsga2 mode has no archive; use mesb-shadow to log a fixed grid")
    if args.selection_mode != "mesb" and args.population_size % 4:
        raise SystemExit("NSGA-II/DCD selection needs --population-size divisible by 4")
    if args.selection_mode == "mesb" and args.population_size % 2:
        raise SystemExit("--population-size must be even (crossover pairs parents)")
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    MESBRun(cfg).run(resume=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
