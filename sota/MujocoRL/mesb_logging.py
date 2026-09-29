"""File layout and writers for a MESB run directory.

    mujoco_rl_output/<run_name>/
        config.json                 resolved configuration + provenance
        evaluations.csv             one row per evaluated gene (at evaluation time)
        evaluations/<gene>.json     canonical per-gene result
        genes/network_<gene>.py     LLM-generated networks
        models/<gene>.zip           trained PPO models
        checkpoints/                checkpoint_gen_<g>.pkl + gen_<g>_plan.pkl
        logs/                       selection.jsonl, variation.jsonl, eval/, llm/
        mesb/
            metrics.csv             one row per generation
            boundaries.jsonl        every boundary computation + per-generation snapshot
            final_archive.csv       current elites (rewritten every generation)
            archive_snapshots/      gen_<g>.json
            figures/                written by analyze_mesb.py
        videos/
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from sota.MujocoRL.mesb_archive import AddResult, SlidingBoundariesArchive
from sota.MujocoRL.results_io import DESCRIPTOR_KEYS, EvaluationOutcome

EVALUATION_COLUMNS = [
    "gene_id", "generation", "status", "mean_reward", "std_reward", "param_count",
    "mean_distance", "mean_control_cost", "mean_control_cost_per_step",
    "archive_cell_x", "archive_cell_y", "inserted", "replaced_gene_id", "model_path",
    "train_seed", "operator", "parents", "error",
]

METRIC_COLUMNS = [
    "generation", "selection_mode", "boundary_mode", "total_evaluations",
    "successful_evaluations", "failed_evaluations", "new_evaluations", "llm_failures",
    "occupied_cells", "total_cells", "coverage", "raw_qd_score", "shifted_qd_score",
    "mean_elite_reward", "best_reward", "best_gene_id", "number_of_remaps", "archive_total_seen",
    "population_size", "population_best_reward",
]

ARCHIVE_COLUMNS = [
    "cell_x", "cell_y", "gene_id", "generation", "mean_reward", "mean_distance",
    "mean_control_cost", "param_count", "model_path",
    "distance_lo", "distance_hi", "control_cost_lo", "control_cost_hi",
]


class RunPaths:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.config = self.run_dir / "config.json"
        self.evaluations_csv = self.run_dir / "evaluations.csv"
        self.evaluations_dir = self.run_dir / "evaluations"
        self.genes = self.run_dir / "genes"
        self.prompts = self.run_dir / "prompts"
        self.models = self.run_dir / "models"
        self.checkpoints = self.run_dir / "checkpoints"
        self.logs = self.run_dir / "logs"
        self.eval_logs = self.logs / "eval"
        self.llm_logs = self.logs / "llm"
        self.selection_log = self.logs / "selection.jsonl"
        self.variation_log = self.logs / "variation.jsonl"
        self.mesb = self.run_dir / "mesb"
        self.metrics_csv = self.mesb / "metrics.csv"
        self.boundaries_jsonl = self.mesb / "boundaries.jsonl"
        self.final_archive_csv = self.mesb / "final_archive.csv"
        self.snapshots = self.mesb / "archive_snapshots"
        self.figures = self.mesb / "figures"
        self.videos = self.run_dir / "videos"

    def create(self) -> None:
        for d in (self.evaluations_dir, self.genes, self.prompts, self.models, self.checkpoints,
                  self.eval_logs, self.llm_logs, self.snapshots, self.figures, self.videos):
            d.mkdir(parents=True, exist_ok=True)
        # Keep every run artefact out of git without touching the repo .gitignore.
        root_ignore = self.run_dir.parent / ".gitignore"
        if not root_ignore.exists():
            root_ignore.write_text("# MESB run outputs (created by run_mesb.py)\n*\n")

    def result_json(self, gene_id: str) -> Path:
        return self.evaluations_dir / f"{gene_id}.json"

    def checkpoint(self, generation: int) -> Path:
        return self.checkpoints / f"checkpoint_gen_{generation}.pkl"

    def plan(self, generation: int) -> Path:
        return self.checkpoints / f"gen_{generation}_plan.pkl"


def _fmt(x: Any) -> Any:
    if x is None:
        return ""
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, (list, tuple)):
        return ";".join(str(v) for v in x)
    return x


def append_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if new:
            w.writeheader()
        for row in rows:
            w.writerow({k: _fmt(row.get(k)) for k in columns})


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    append_csv(tmp, columns, rows)  # always writes the header, even with no rows
    tmp.replace(path)


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def truncate_after_generation(run: RunPaths, generation: int) -> None:
    """Drop log rows from generations after ``generation`` (used on resume).

    Guarantees that an interrupted-then-resumed run never has duplicate rows
    for the generation that was in flight.
    """
    for path in (run.evaluations_csv, run.metrics_csv):
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
            columns = list(rows[0].keys()) if rows else None
        if columns is None:
            continue
        keep = [r for r in rows if int(r["generation"]) <= generation]
        write_csv(path, columns, keep)
    for path in (run.boundaries_jsonl, run.selection_log, run.variation_log):
        if not path.exists():
            continue
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        keep = [ln for ln in lines if json.loads(ln).get("generation", -1) <= generation]
        path.write_text("".join(ln + "\n" for ln in keep), encoding="utf-8")


def evaluation_row(outcome: EvaluationOutcome, generation: int, add: AddResult | None,
                   train_seed: int | None, operator: str | None,
                   parents: list[str] | None) -> dict[str, Any]:
    p = outcome.payload
    b = p.get("behavior") or {}
    row = {
        "gene_id": outcome.gene_id, "generation": generation,
        "status": "success" if outcome.ok else "failed",
        "mean_reward": p.get("mean_reward") if outcome.ok else None,
        "std_reward": p.get("std_reward") if outcome.ok else None,
        "param_count": p.get("param_count") if outcome.ok else None,
        "mean_distance": b.get("mean_distance") if outcome.ok else None,
        "mean_control_cost": b.get("mean_control_cost") if outcome.ok else None,
        "mean_control_cost_per_step": b.get("mean_control_cost_per_step") if outcome.ok else None,
        "model_path": p.get("model_path"), "train_seed": train_seed, "operator": operator,
        "parents": parents or [], "error": None if outcome.ok else outcome.reason,
    }
    if add is not None:
        row.update({"archive_cell_x": add.cell[0], "archive_cell_y": add.cell[1],
                    "inserted": add.inserted, "replaced_gene_id": add.replaced_gene_id})
    return row


def boundary_record(archive: SlidingBoundariesArchive, generation: int, event: str,
                    source: dict | None = None) -> dict[str, Any]:
    bounds = source["boundaries"] if source else archive.boundaries
    return {
        "generation": generation, "event": event,
        "total_seen": source["total_seen"] if source else archive.total_seen,
        "num_remaps": archive.num_remaps, "boundary_mode": archive.boundary_mode,
        "dims": list(archive.dims),
        "boundaries": {k: v for k, v in zip(DESCRIPTOR_KEYS, bounds)},
    }


def archive_rows(archive: SlidingBoundariesArchive) -> list[dict[str, Any]]:
    b = archive.boundaries
    rows = []
    for cell in archive.occupied_cells():
        r = archive.elites[cell]
        rows.append({
            "cell_x": cell[0], "cell_y": cell[1], "gene_id": r.gene_id,
            "generation": r.generation, "mean_reward": r.objective,
            "mean_distance": r.measures[0], "mean_control_cost": r.measures[1],
            "param_count": r.param_count, "model_path": r.model_path,
            "distance_lo": b[0][cell[0]], "distance_hi": b[0][cell[0] + 1],
            "control_cost_lo": b[1][cell[1]], "control_cost_hi": b[1][cell[1] + 1],
        })
    return rows


def write_snapshot(run: RunPaths, archive: SlidingBoundariesArchive, generation: int) -> None:
    snap = {
        "generation": generation,
        "descriptors": list(DESCRIPTOR_KEYS),
        "objective": "mean_reward",
        "summary": archive.summary(),
        "boundaries": {k: v for k, v in zip(DESCRIPTOR_KEYS, archive.boundaries)},
        "elites": archive_rows(archive),
    }
    path = run.snapshots / f"gen_{generation:04d}.json"
    path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    write_csv(run.final_archive_csv, ARCHIVE_COLUMNS, snap["elites"])
