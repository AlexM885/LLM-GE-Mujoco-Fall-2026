"""Evaluate genes for the MESB driver.

``SubprocessEvaluator`` runs ``train_gene.py`` once per gene in its own
process (so a broken LLM-generated network can never crash the driver),
several at a time, each with a wall-clock timeout. The canonical output is
the per-gene result JSON; if a process dies without writing one, a
``status: failed`` JSON is written on its behalf.

A gene whose result JSON already exists is *not* re-run. This is what makes
a resumed run skip evaluations that completed before an interruption.

``SlurmEvaluator`` submits one Slurm job per gene instead (``sbatch --wait``),
several at a time; used for XQC, which needs a GPU per gene. The job script
(``scripts/mesb/xqc_gene.sbatch``) runs the same per-gene trainer, so the
result JSON is identical to a local run.

The trainer depends on the learner (``algo``): ``train_gene.py`` (PPO, SB3)
or ``train_gene_xqc.py`` (XQC, Shazeb's trainer on shazeb/halfcheetah-xqc).

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
XQC_TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_gene_xqc.py"


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
    #: Learner: "td3" or "ppo" (train_gene.py) or "xqc" (train_gene_xqc.py).
    algo: str = "td3"
    #: XQC: seeds trained together per gene, and the XQC seed (Shazeb's island
    #: uses 0 for every gene, so every gene sees the same evaluation rollouts).
    num_seeds: int = 1
    xqc_seed: int = 0
    #: Slurm evaluator: job script and extra sbatch flags (e.g. "-A coc").
    slurm_script: str | None = None
    slurm_flags: str = ""


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


def trainer_args(settings: EvalSettings, job: EvalJob) -> list[str]:
    """Script + arguments of the per-gene trainer (without the Python command)."""
    s = settings
    if s.algo == "xqc":
        return [
            str(XQC_TRAIN_SCRIPT),
            "--gene-file", str(job.gene_file), "--gene-id", job.gene_id,
            "--out-json", str(job.out_json), "--run-dir", str(job.model_dir / job.gene_id),
            "--generation", str(job.generation), "--env", s.env_id,
            "--seed", str(s.xqc_seed), "--timesteps", str(s.timesteps),
            "--num-seeds", str(s.num_seeds), "--eval-episodes", str(s.episodes),
            "--max-steps", str(s.max_steps),
        ]
    if s.algo not in ("td3", "ppo"):
        raise ValueError(f"unknown algo {s.algo!r}")
    return [
        str(TRAIN_SCRIPT),
        "--gene-file", str(job.gene_file), "--gene-id", job.gene_id,
        "--out-json", str(job.out_json), "--model-dir", str(job.model_dir),
        "--generation", str(job.generation), "--env", s.env_id,
        "--seed", str(job.train_seed), "--eval-seed", str(s.eval_seed),
        "--timesteps", str(s.timesteps), "--eval-episodes", str(s.episodes),
        "--eval-max-steps", str(s.max_steps), "--device", s.device,
    ]


def _write_failure(job: EvalJob, reason: str, stage: str) -> None:
    atomic_write_json(job.out_json, failure_payload(
        job.gene_id, reason, generation=job.generation, seed=job.train_seed,
        stage=stage, log_file=str(job.log_file), log_tail=_tail(job.log_file)))
    print(f"\t✗ evaluation of {job.gene_id} (generation {job.generation}) failed: {reason}",
          flush=True)


class SubprocessEvaluator(_BaseEvaluator):
    """Run the per-gene trainer in a separate Python process."""

    def command(self, job: EvalJob) -> list[str]:
        return shlex.split(self.settings.python_cmd) + trainer_args(self.settings, job)

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
            _write_failure(job, reason, "subprocess")


class SlurmEvaluator(_BaseEvaluator):
    """One Slurm job per gene (``sbatch --wait``), ``workers`` jobs at a time.

    The job script receives the trainer arguments; Slurm writes the job's
    output to the gene's log file. ``timeout_sec`` covers queueing and
    running; on timeout the job is cancelled and the gene counts as failed.
    """

    def command(self, job: EvalJob) -> list[str]:
        s = self.settings
        if not s.slurm_script:
            raise ValueError(f"no Slurm job script for algo {s.algo!r}")
        args = trainer_args(s, job)[1:]  # the job script calls the trainer itself
        return (["sbatch", "--wait", "--parsable", "--job-name", f"mesb-{job.gene_id[:12]}",
                 "--output", str(job.log_file)] + shlex.split(s.slurm_flags)
                + [str(s.slurm_script)] + args)

    def _run_one(self, job: EvalJob) -> None:
        job.log_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.command(job)
        reason, job_id = None, None
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                out, err = proc.communicate(timeout=self.settings.timeout_sec)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
                job_id = _first_job_id(out)
                if job_id:
                    subprocess.run(["scancel", job_id], check=False)
                reason = f"Slurm job {job_id or '?'} timed out after {self.settings.timeout_sec:.0f}s"
                if job.out_json.exists():
                    job.out_json.unlink()
            else:
                job_id = _first_job_id(out)
                if not job.out_json.exists():
                    reason = (f"Slurm job {job_id or '?'} ended (sbatch exit {proc.returncode}) "
                              f"without a result file{': ' + err.strip()[-300:] if err.strip() else ''}")
        except OSError as exc:
            reason = f"could not run sbatch: {exc}"
        if reason is not None:
            _write_failure(job, reason, "slurm")


def _first_job_id(text: str | None) -> str | None:
    for line in (text or "").splitlines():
        token = line.strip().split(";")[0]
        if token.isdigit():
            return token
    return None


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
        contact = [float(rng.beta(2.0, 2.0)), float(rng.beta(2.0, 2.0))]
        n = self.settings.episodes
        payload = {
            "schema_version": SCHEMA_VERSION, "gene_id": job.gene_id, "status": "success",
            "generation": job.generation, "seed": job.train_seed,
            "eval_seed": self.settings.eval_seed, "env_id": "MOCK", "algorithm": "MOCK",
            "timesteps": self.settings.timesteps, "num_eval_episodes": n,
            "max_eval_steps": self.settings.max_steps, "train_time_sec": 0.0,
            "mean_reward": reward, "std_reward": 0.0, "param_count": 1000 + len(code),
            "model_path": None, "rewards": [reward] * n,
            "behavior": {"foot_contact_0": contact[0], "foot_contact_1": contact[1],
                         "mean_distance": distance, "mean_control_cost": control,
                         "mean_control_cost_per_step": control / self.settings.max_steps,
                         "distances": [distance] * n, "control_costs": [control] * n},
        }
        assert math.isfinite(reward)
        atomic_write_json(job.out_json, payload)


def make_evaluator(kind: str, settings: EvalSettings) -> _BaseEvaluator:
    if kind == "subprocess":
        return SubprocessEvaluator(settings)
    if kind == "slurm":
        return SlurmEvaluator(settings)
    if kind == "mock":
        return MockEvaluator(settings)
    raise ValueError(f"unknown evaluator {kind!r}")
