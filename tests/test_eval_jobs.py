"""Tests for src/eval_jobs.py (gene packing and queue-aware eval timeouts).

Standard library only:  python -m pytest --noconftest tests/test_eval_jobs.py
"""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval_jobs import (gene_output_path, job_ended, job_shared_with_running_gene,  # noqa: E402
                           packed_gene_state, packed_runline, pending_job_ids, update_eval_clock)


def check_contents(contents):
    """Same rules as run_improved.check_contents_for_error."""
    text = contents.lower()
    if "traceback" in text or "slurmstepd: error" in text:
        return False
    if "cancelled at" in text or "due to time limit" in text:
        return False
    if "job done" in text:
        return True
    if "begin slurm epilog" in text:
        return False
    return None


def test_packed_runline_runs_genes_in_parallel_with_separate_outputs(tmp_path):
    out_dir = str(tmp_path / "genes")
    script = packed_runline([("geneA", "sleep 1; echo A; echo Job Done"),
                             ("geneB", "sleep 1; echo B; echo Job Done")], out_dir)
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    a = Path(gene_output_path(out_dir, "geneA")).read_text()
    b = Path(gene_output_path(out_dir, "geneB")).read_text()
    assert "A" in a and "B" not in a
    assert "B" in b and "A" not in b
    assert "Packed genes finished" in result.stdout


def test_packed_runline_waits_for_slowest_gene(tmp_path):
    out_dir = str(tmp_path / "genes")
    script = packed_runline([("fast", "echo Job Done"), ("slow", "sleep 2; echo Job Done")], out_dir)
    subprocess.run(["bash", "-c", script], check=True, timeout=30)
    assert "Job Done" in Path(gene_output_path(out_dir, "slow")).read_text()


def test_packed_runline_keeps_shell_operators_inside_each_gene(tmp_path):
    # Real runlines chain commands with && (aux reward prefix, XQC env.sh).
    out_dir = str(tmp_path / "genes")
    script = packed_runline([("g", "export X=1 && echo value=$X && echo Job Done")], out_dir)
    subprocess.run(["bash", "-c", script], check=True, timeout=30)
    assert "value=1" in Path(gene_output_path(out_dir, "g")).read_text()


def test_packed_gene_state():
    assert packed_gene_state("training...\nJob Done\n", None, check_contents) is True
    assert packed_gene_state("Traceback (most recent call last)", None, check_contents) is False
    # Gene still running, job still running.
    assert packed_gene_state("training...", "Launching Python Evaluation", check_contents) is None
    # No gene file yet, job still running.
    assert packed_gene_state(None, "Launching Python Evaluation", check_contents) is None
    # Job hit its time limit before the gene finished.
    assert packed_gene_state("training...", "slurmstepd: error: *** JOB 1 CANCELLED AT ... DUE TO TIME LIMIT",
                             check_contents) is False
    # Job is over (epilog) but this gene never wrote a verdict.
    assert packed_gene_state(None, "Begin Slurm Epilog: Nov-08-2026", check_contents) is False
    # The gene's own verdict wins over the job's end markers.
    assert packed_gene_state("Job Done", "Begin Slurm Epilog", check_contents) is True


def test_job_ended():
    assert job_ended("Begin Slurm Epilog") and job_ended("CANCELLED AT 2026") and not job_ended("") \
        and not job_ended(None) and not job_ended("Launching Python Evaluation")


def test_update_eval_clock_holds_clock_while_pending():
    entry = {"results_job": "123", "eval_start_time": 0.0, "submit_time": 0.0}
    assert update_eval_clock(entry, {"123"}, now=500.0) is False
    assert entry["eval_start_time"] == 500.0          # clock pushed to now while queued
    assert update_eval_clock(entry, {"999"}, now=900.0) is False
    assert entry["eval_start_time"] == 500.0          # running: clock left alone from here on


def test_update_eval_clock_unknown_queue_leaves_clock_alone():
    entry = {"results_job": "123", "eval_start_time": 10.0, "submit_time": 10.0}
    assert update_eval_clock(entry, None, now=500.0) is False
    assert entry["eval_start_time"] == 10.0


def test_update_eval_clock_queue_limit():
    entry = {"results_job": 7, "eval_start_time": 0.0, "submit_time": 0.0}
    assert update_eval_clock(entry, {"7"}, now=100.0, max_queue_seconds=200) is False
    assert update_eval_clock(entry, {"7"}, now=300.0, max_queue_seconds=200) is True
    assert update_eval_clock(entry, {"7"}, now=10_000.0, max_queue_seconds=0) is False  # 0 = no limit


def test_pending_job_ids_parses_squeue():
    fake = lambda *a, **k: SimpleNamespace(returncode=0, stdout="101\n 102 \n\n")
    assert pending_job_ids(run=fake) == {"101", "102"}


def test_pending_job_ids_returns_none_on_failure():
    assert pending_job_ids(run=lambda *a, **k: SimpleNamespace(returncode=1, stdout="")) is None

    def boom(*a, **k):
        raise FileNotFoundError("squeue")
    assert pending_job_ids(run=boom) is None


def test_job_shared_with_running_gene():
    data = {"a": {"results_job": "5", "status": "running eval"},
            "b": {"results_job": "5", "status": "running eval"},
            "c": {"results_job": "6", "status": "running eval"}}
    assert job_shared_with_running_gene(data, "5", "a")      # b still needs job 5
    data["b"]["status"] = "FAILED: TIMEOUT"
    assert not job_shared_with_running_gene(data, "5", "a")  # nobody else: safe to cancel
    assert not job_shared_with_running_gene(data, "6", "c")
