"""Helpers for gene evaluation jobs that run_improved.py submits to Slurm.

Kept free of the controller's globals so they can be tested on their own.

Two speed-ups live here:

* Packing: GENES_PER_JOB genes train side by side in one Slurm job (one GPU).
  Each gene writes its own output file, because the controller decides a gene
  is finished by finding "job done" in its output, and one shared file would
  mark every gene of the job finished when the first one is.
* Queue-aware timeouts: a gene's no-progress clock starts when its job starts
  running, not when it was submitted, so a gene is never killed for waiting
  in the queue.
"""

import os
import shlex
import subprocess


def gene_output_path(output_dir, gene_id):
    """Output file of one gene inside a packed job."""
    return os.path.join(output_dir, f"gene-{gene_id}.out")


def packed_runline(gene_runlines, output_dir):
    """Shell code that runs several genes at once and waits for all of them.

    gene_runlines: list of (gene_id, runline). Each runline runs in its own
    subshell with its output in gene_output_path(output_dir, gene_id).
    """
    lines = [f"mkdir -p {shlex.quote(output_dir)}"]
    for gene_id, runline in gene_runlines:
        out = shlex.quote(gene_output_path(output_dir, gene_id))
        lines.append(f"( {runline} ) > {out} 2>&1 &")
    lines.append("wait")
    lines.append('echo "Packed genes finished"')
    return "\n".join(lines)


def job_ended(job_contents):
    """True if a Slurm job's own output shows the job is over (finished, killed or cancelled)."""
    text = (job_contents or "").lower()
    return any(marker in text for marker in
               ("begin slurm epilog", "cancelled at", "due to time limit", "slurmstepd: error"))


def packed_gene_state(gene_contents, job_contents, check_contents):
    """Verdict for one gene of a packed job: True done, False failed, None still running.

    check_contents is run_improved.check_contents_for_error. The gene's own file
    decides; if it has no verdict but the job is over, the gene died with it.
    """
    if gene_contents is not None:
        state = check_contents(gene_contents)
        if state is not None:
            return state
    if job_ended(job_contents):
        return False
    return None


def pending_job_ids(run=subprocess.run):
    """Ids of this user's jobs that are still waiting in the queue.

    Returns None if squeue fails, so callers can leave their clocks alone
    instead of treating every job as started.
    """
    try:
        result = run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-t", "PENDING", "-o", "%i"],
                     capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def update_eval_clock(entry, pending, now, max_queue_seconds=0):
    """Hold a gene's no-progress clock at zero while its job waits in the queue.

    entry: the gene's GLOBAL_DATA record ('results_job', 'eval_start_time',
    'submit_time'). The controller's timeout counts from 'eval_start_time', so
    pushing it forward while the job is pending starts the clock at job start.
    Returns True if the job has waited in the queue longer than
    max_queue_seconds (0 = no limit) and should be given up on.
    """
    if pending is None or str(entry.get("results_job")) not in pending:
        return False
    entry["eval_start_time"] = now
    submitted = entry.get("submit_time", now)
    return bool(max_queue_seconds) and now - submitted > max_queue_seconds


def job_shared_with_running_gene(global_data, job_id, gene_id):
    """True if another gene sharing job_id is still waiting for its result."""
    return any(other != gene_id and str(data.get("results_job")) == str(job_id)
               and data.get("status") == "running eval"
               for other, data in global_data.items())
