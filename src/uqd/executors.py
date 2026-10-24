"""Run training and rollout jobs for the uncertain-QD engine.

LocalExecutor runs jobs as subprocesses on the current node (smoke tests,
small final evaluations). SlurmExecutor submits the same commands with
sbatch and polls for the result file. Both return plain dicts so the engine
does not depend on how a job was run.

Failure handling: a job that exits without writing its result file, is
cancelled, or never leaves the queue before the timeout produces
``infra_failure`` results for every request it carried, with a kind naming
the cause. The retry helpers resubmit such requests with identical seeds.
"""

import json
import os
import shlex
import subprocess
import time
import uuid

from .stats import INFRA_FAILURE, OK, outcome_class

EVAL_PROJECT = "sota/MujocoRL/eval_env"
REEVAL_SCRIPT = "sota/MujocoRL/reeval.py"
TRAIN_SCRIPT = "sota/MujocoRL/train_rl.py"


def default_python_cmd():
    """uv command for the evaluation environment (matches constants.UV_PYTHON)."""
    return ["uv", "run", "--isolated", "--project", EVAL_PROJECT, "python"]


def _clean_env():
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    return env


def train_command(python_cmd, spec, timesteps, env_id, eval_max_steps, model_dir, stats_dir):
    return list(python_cmd) + [
        TRAIN_SCRIPT,
        "-network", f"models.network_{spec.gene_id}",
        "-env", env_id,
        "-timesteps", str(timesteps),
        "-eval_episodes", str(spec.eval_episodes),
        "-eval_max_steps", str(eval_max_steps),
        "-seed", str(spec.train_seed),
        "-eval_seed", str(spec.eval_seed),
        "-eval_device", "cpu",
        "-model_dir", model_dir,
        "-stats_dir", stats_dir,
    ]


def train_stats_path(stats_dir, spec):
    return os.path.join(stats_dir, f"{spec.gene_id}_seed{spec.train_seed}_stats.json")


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, "missing_artifact"
    except (json.JSONDecodeError, OSError):
        return None, "malformed_result"


class LocalExecutor:
    def __init__(self, repo_root, work_dir, python_cmd=None, timeout=None):
        self.repo_root = repo_root
        self.work_dir = work_dir
        self.python_cmd = python_cmd or default_python_cmd()
        self.timeout = timeout
        os.makedirs(work_dir, exist_ok=True)

    def _run(self, cmd, log_path):
        with open(log_path, "w", encoding="utf-8") as log:
            try:
                proc = subprocess.run(cmd, cwd=self.repo_root, stdout=log, stderr=subprocess.STDOUT,
                                      env=_clean_env(), timeout=self.timeout)
                return proc.returncode, None
            except subprocess.TimeoutExpired:
                return None, "scheduler_timeout"

    def run_rollouts(self, requests, protocol, label="rollouts"):
        if not requests:
            return []
        tag = f"{label}_{uuid.uuid4().hex[:8]}"
        req_path = os.path.join(self.work_dir, f"{tag}_requests.json")
        out_path = os.path.join(self.work_dir, f"{tag}_results.json")
        with open(req_path, "w", encoding="utf-8") as f:
            json.dump({"protocol": protocol, "requests": requests}, f)
        cmd = list(self.python_cmd) + [REEVAL_SCRIPT, "--requests", req_path, "--out", out_path]
        _, timeout_kind = self._run(cmd, os.path.join(self.work_dir, f"{tag}.log"))
        return _collect_rollout_results(requests, out_path, timeout_kind or "job_lost")

    def run_command(self, args, done_path, label):
        """Run an arbitrary job script; returns None on success or a lost kind."""
        _, timeout_kind = self._run(list(self.python_cmd) + list(args),
                                    os.path.join(self.work_dir, f"{label}.log"))
        return None if os.path.exists(done_path) else (timeout_kind or "job_lost")

    def run_train(self, spec, timesteps, env_id, eval_max_steps, model_dir, stats_dir):
        cmd = train_command(self.python_cmd, spec, timesteps, env_id, eval_max_steps, model_dir, stats_dir)
        log = os.path.join(self.work_dir, f"train_{spec.gene_id}_seed{spec.train_seed}.log")
        path = train_stats_path(stats_dir, spec)
        if os.path.exists(path):
            os.remove(path)   # never read a stale file from an earlier attempt
        _, timeout_kind = self._run(cmd, log)
        stats, read_error = read_json(path)
        if stats is None:
            return None, timeout_kind or ("job_lost" if read_error == "missing_artifact" else read_error)
        return stats, None


def _collect_rollout_results(requests, out_path, lost_kind):
    """Match a job's results to its requests; anything missing is infra."""
    payload, read_error = read_json(out_path)
    by_id = {}
    if payload is not None:
        by_id = {r["request_id"]: r for r in payload.get("results", [])}
    out = []
    for req in requests:
        res = by_id.get(req["request_id"])
        if res is None:
            res = {"request_id": req["request_id"], "solution_id": req["solution_id"],
                   "seeds": req["seeds"], "status": INFRA_FAILURE,
                   "failure_kind": lost_kind if payload is None else "malformed_result"}
        res["start_index"] = req.get("start_index", 0)
        out.append(res)
    return out


class SlurmExecutor:
    """Submit the same commands with sbatch and wait for their result files."""

    def __init__(self, repo_root, work_dir, script_template, python_cmd=None, poll_interval=30,
                 timeout=6 * 3600, chunk_size=64):
        self.repo_root = repo_root
        self.work_dir = work_dir
        self.template = script_template
        self.python_cmd = python_cmd or default_python_cmd()
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.chunk_size = chunk_size
        os.makedirs(work_dir, exist_ok=True)

    def _submit(self, command, tag):
        script = os.path.join(self.work_dir, f"{tag}.sh")
        log = os.path.join(self.work_dir, f"{tag}.out")
        body = self.template.replace("{output}", log).replace("{command}", command)
        with open(script, "w", encoding="utf-8") as f:
            f.write(body)
        proc = subprocess.run(["sbatch", "--parsable", script], cwd=self.repo_root,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return None, log
        return proc.stdout.strip().split(";")[0], log

    @staticmethod
    def _in_queue(job_id):
        proc = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T"], capture_output=True, text=True)
        return bool(proc.stdout.strip())

    def _wait(self, jobs):
        """jobs: list of (job_id, done_path, log). Returns {job_id: lost_kind or None}."""
        start = time.time()
        pending = {j for j, _, _ in jobs if j is not None}
        outcome = {}
        while pending:
            for job_id, done_path, log in jobs:
                if job_id not in pending:
                    continue
                if os.path.exists(done_path):
                    outcome[job_id] = None
                    pending.discard(job_id)
                elif not self._in_queue(job_id):
                    time.sleep(5)    # allow the filesystem to catch up
                    if not os.path.exists(done_path):
                        started = os.path.exists(log)
                        outcome[job_id] = "job_lost" if started else "submission_failed"
                        pending.discard(job_id)
            if pending and time.time() - start > self.timeout:
                logs = {j: log for j, _, log in jobs}
                for job_id in pending:
                    subprocess.run(["scancel", str(job_id)], capture_output=True)
                    # Started but never finished: the candidate ran too long
                    # (counted as a candidate failure, as the legacy pipeline
                    # did). Never started: the scheduler is to blame.
                    outcome[job_id] = ("candidate_timeout" if os.path.exists(logs[job_id])
                                       else "scheduler_timeout")
                break
            if pending:
                time.sleep(self.poll_interval)
        return outcome

    def run_command(self, args, done_path, label):
        cmd = " ".join(shlex.quote(x) for x in list(self.python_cmd) + list(args))
        job_id, log = self._submit(cmd, f"{label}_{uuid.uuid4().hex[:8]}")
        if job_id is None:
            return "submission_failed"
        return self._wait([(job_id, done_path, log)]).get(job_id)

    def run_rollouts(self, requests, protocol, label="rollouts"):
        jobs, chunks = [], []
        for i in range(0, len(requests), self.chunk_size):
            chunk = requests[i:i + self.chunk_size]
            tag = f"{label}_{uuid.uuid4().hex[:8]}"
            req_path = os.path.join(self.work_dir, f"{tag}_requests.json")
            out_path = os.path.join(self.work_dir, f"{tag}_results.json")
            with open(req_path, "w", encoding="utf-8") as f:
                json.dump({"protocol": protocol, "requests": chunk}, f)
            cmd = " ".join(shlex.quote(x) for x in list(self.python_cmd) +
                           [REEVAL_SCRIPT, "--requests", req_path, "--out", out_path])
            job_id, log = self._submit(cmd, tag)
            jobs.append((job_id, out_path, log))
            chunks.append((chunk, out_path, job_id))
        outcome = self._wait(jobs)
        results = []
        for chunk, out_path, job_id in chunks:
            lost = "submission_failed" if job_id is None else (outcome.get(job_id) or "job_lost")
            results.extend(_collect_rollout_results(chunk, out_path, lost))
        return results

    def run_trains(self, specs, timesteps, env_id, eval_max_steps, model_dir, stats_dir):
        jobs = []
        for spec in specs:
            path = train_stats_path(stats_dir, spec)
            if os.path.exists(path):
                os.remove(path)
            cmd = " ".join(shlex.quote(x) for x in train_command(
                self.python_cmd, spec, timesteps, env_id, eval_max_steps, model_dir, stats_dir))
            job_id, log = self._submit(cmd, f"train_{spec.gene_id}_seed{spec.train_seed}")
            jobs.append((job_id, path, log, spec))
        outcome = self._wait([(j, p, l) for j, p, l, _ in jobs])
        out = []
        for job_id, path, _, spec in jobs:
            stats, err = read_json(path)
            kind = None
            if stats is None:
                kind = "submission_failed" if job_id is None else (outcome.get(job_id) or err or "job_lost")
            out.append((spec, stats, kind))
        return out


def run_rollouts_with_retries(engine, executor, requests, generation, max_retries,
                              source="reeval", category="reeval", label="reeval"):
    """Run rollout requests; retry infra failures with the same seeds."""
    pending = list(requests)
    last_failed = {}
    for attempt in range(max_retries + 1):
        if not pending:
            break
        results = executor.run_rollouts(pending, engine.protocol, label=f"{label}_a{attempt}")
        failed = engine.ingest_rollout_results(results, generation, source=source, category=category)
        failed_ids = {r["request_id"] for r in failed}
        if failed:
            engine.log("rollout_retry" if attempt < max_retries else "rollout_retries_exhausted",
                       failed=len(failed), attempt=attempt,
                       kinds=sorted({r.get("failure_kind") for r in failed}))
        pending = [r for r in pending if r["request_id"] in failed_ids]
        last_failed = {r["request_id"]: r for r in failed}
    for req in pending:
        engine.reeval_lost(last_failed[req["request_id"]], generation)
    return pending


def run_train_specs_with_retries(engine, executor, specs, generation, max_retries, timesteps,
                                 env_id, eval_max_steps, model_dir, stats_dir):
    """Train and ingest specs; infra failures are retried with the same seeds."""
    pending = list(specs)
    last_kind = {}
    for attempt in range(max_retries + 1):
        if not pending:
            break
        if hasattr(executor, "run_trains"):
            outputs = executor.run_trains(pending, timesteps, env_id, eval_max_steps, model_dir, stats_dir)
        else:
            outputs = [(s, *executor.run_train(s, timesteps, env_id, eval_max_steps, model_dir, stats_dir))
                       for s in pending]
        retry = []
        for spec, stats, kind in outputs:
            spec.attempts += 1
            result = engine.ingest_train_stats(spec, stats, generation, infra_kind=kind)
            if result == INFRA_FAILURE:
                retry.append((spec, (stats or {}).get("failure_kind") or kind))
        pending = [s for s, _ in retry]
        last_kind = dict((s.solution_id, k) for s, k in retry)
    for spec in pending:
        engine.train_lost(spec, generation, last_kind[spec.solution_id])
    return pending


__all__ = ["LocalExecutor", "SlurmExecutor", "run_rollouts_with_retries",
           "run_train_specs_with_retries", "train_command", "train_stats_path", "outcome_class", "OK"]
