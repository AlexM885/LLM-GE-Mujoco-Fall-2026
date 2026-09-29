"""Evaluate genes for the MESB driver.

``SubprocessEvaluator`` runs ``train_gene.py`` once per gene in its own
process (so a broken LLM-generated network can never crash the driver),
several at a time, each with a wall-clock timeout. The canonical output is
the per-gene result JSON; if a process dies without writing one, a
``status: failed`` JSON is written on its behalf.

A gene whose result JSON already exists is *not* re-run. This is what makes
a resumed run skip evaluations that completed before an interruption.

``MockEvaluator`` produces synthetic results without MuJoCo or PPO. It is for
tests of the evolution/archive plumbing only and has no scientific meaning.
"""

from __future__ import annotations

import hashlib
import math
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from sota.MujocoRL.results_io import (SCHEMA_VERSION, EvaluationOutcome, atomic_write_json,
                                      failure_payload, read_outcome)

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_gene.py"


@dataclass(frozen=True)
class EvalJob:
    gene_id: str
    gene_file: Path
    generation: int
    train_seed: int
    out_json: Path
    model_dir: Path
    log_file: Path


@dataclass
class EvalSettings:
    env_id: str
    timesteps: int
    episodes: int
    max_steps: int
    eval_seed: int
    device: str = "cpu"
    timeout_sec: float = 6 * 3600
    workers: int = 1
    python_cmd: str = sys.executable


class _BaseEvaluator:
    def __init__(self, settings: EvalSettings) -> None:
        self.settings = settings

    def _run_one(self, job: EvalJob) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def evaluate(self, jobs: Iterable[EvalJob]) -> dict[str, EvaluationOutcome]:
        """Evaluate every job whose result file does not exist yet.

        Returns outcomes for *all* jobs (cached or new), keyed by gene id.
        """
        jobs = list(jobs)
        todo = [j for j in jobs if not j.out_json.exists()]
        if todo:
            workers = max(1, min(self.settings.workers, len(todo)))
            if workers == 1:
                for job in todo:
                    self._run_one(job)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(self._run_one, todo))
        return {j.gene_id: read_outcome(j.out_json, j.gene_id) for j in jobs}


class SubprocessEvaluator(_BaseEvaluator):
    """Run ``train_gene.py`` per gene in a separate Python process."""

    def command(self, job: EvalJob) -> list[str]:
        s = self.settings
        return shlex.split(s.python_cmd) + [
            str(TRAIN_SCRIPT),
            "--gene-file", str(job.gene_file), "--gene-id", job.gene_id,
            "--out-json", str(job.out_json), "--model-dir", str(job.model_dir),
            "--generation", str(job.generation), "--env", s.env_id,
            "--seed", str(job.train_seed), "--eval-seed", str(s.eval_seed),
            "--timesteps", str(s.timesteps), "--eval-episodes", str(s.episodes),
            "--eval-max-steps", str(s.max_steps), "--device", s.device,
        ]

    def _run_one(self, job: EvalJob) -> None:
        job.log_file.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "1")  # one core per concurrent PPO run
        env.setdefault("MKL_NUM_THREADS", "1")
        reason = None
        with open(job.log_file, "w", encoding="utf-8") as log:
            try:
                proc = subprocess.run(self.command(job), stdout=log, stderr=subprocess.STDOUT,
                                      timeout=self.settings.timeout_sec, env=env)
                if proc.returncode != 0 and not job.out_json.exists():
                    reason = f"evaluation process exited with code {proc.returncode}"
            except subprocess.TimeoutExpired:
                reason = f"timeout after {self.settings.timeout_sec:.0f}s"
                if job.out_json.exists():  # a result from the killed run is not trustworthy
                    job.out_json.unlink()
            except OSError as exc:
                reason = f"could not start evaluation process: {exc}"
        if reason is not None:
            tail = _tail(job.log_file)
            atomic_write_json(job.out_json, failure_payload(
                job.gene_id, reason, generation=job.generation, seed=job.train_seed,
                stage="subprocess", log_file=str(job.log_file), log_tail=tail))
            print(f"\t✗ evaluation of {job.gene_id} (generation {job.generation}) failed: {reason}",
                  flush=True)


def _tail(path: Path, n: int = 20) -> str:
    try:
        return "".join(Path(path).read_text(errors="replace").splitlines(True)[-n:])
    except OSError:
        return ""


class MockEvaluator(_BaseEvaluator):
    """Deterministic synthetic results from the gene file contents (tests only).

    Distances are drawn from a skewed distribution so that equal-width and
    empirical-percentile boundaries differ, like real HalfCheetah data.
    A gene file containing ``MOCK_FAIL`` (or ~1 in 9 genes) fails.
    """

    def _run_one(self, job: EvalJob) -> None:
        if not job.gene_file.exists():
            atomic_write_json(job.out_json, failure_payload(
                job.gene_id, f"gene file not found: {job.gene_file}", generation=job.generation))
            return
        code = job.gene_file.read_text(encoding="utf-8")
        digest = hashlib.sha256((code + str(self.settings.eval_seed)).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        if "MOCK_FAIL" in code or rng.random() < 1 / 9:
            atomic_write_json(job.out_json, failure_payload(
                job.gene_id, "RuntimeError: mock training diverged", generation=job.generation,
                stage="train"))
            return
        distance = float(rng.lognormal(mean=5.0, sigma=0.8) - 60.0)
        control = float(rng.gamma(shape=2.0, scale=60.0))
        reward = distance - 0.1 * control + float(rng.normal(0, 20))
        n = self.settings.episodes
        payload = {
            "schema_version": SCHEMA_VERSION, "gene_id": job.gene_id, "status": "success",
            "generation": job.generation, "seed": job.train_seed,
            "eval_seed": self.settings.eval_seed, "env_id": "MOCK", "algorithm": "MOCK",
            "timesteps": self.settings.timesteps, "num_eval_episodes": n,
            "max_eval_steps": self.settings.max_steps, "train_time_sec": 0.0,
            "mean_reward": reward, "std_reward": 0.0, "param_count": 1000 + len(code),
            "model_path": None, "rewards": [reward] * n,
            "behavior": {"mean_distance": distance, "mean_control_cost": control,
                         "mean_control_cost_per_step": control / self.settings.max_steps,
                         "distances": [distance] * n, "control_costs": [control] * n},
        }
        assert math.isfinite(reward)
        atomic_write_json(job.out_json, payload)


def make_evaluator(kind: str, settings: EvalSettings) -> _BaseEvaluator:
    if kind == "subprocess":
        return SubprocessEvaluator(settings)
    if kind == "mock":
        return MockEvaluator(settings)
    raise ValueError(f"unknown evaluator {kind!r}")
