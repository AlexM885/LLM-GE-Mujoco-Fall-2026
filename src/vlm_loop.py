"""Per-generation VLM aux reward (Neel's design) for run_improved.py and run_mesb.py.

After a generation's archive update the driver calls :func:`refresh_aux_reward`
with the archive's genes ranked by fitness. It picks the best VLM_TOP_GENES and
worst VLM_BOTTOM_GENES of them that have a saved model, and runs one GPU job
(scripts/vlm/vlm_generation.sbatch) that records their rollouts and lets the
VLM write a new aux_reward(obs, action, info), seeing its previous one too.
The next generation trains with the file this returns; fitness is still
measured on the plain reward.

If anything fails (no saved models, job error, timeout, no usable reward),
the previous generation's file is kept, so the run never stops for the VLM.
"""

import json
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "cfg"))
import run_options  # noqa: E402

JOB_SCRIPT = os.path.join(REPO_ROOT, "scripts", "vlm", "vlm_generation.sbatch")


def xqc_runs_root():
    """Where XQC gene runs live: the rule of sota/XQC/pace/env.sh, which the
    evaluation jobs source (XQC_RUNS, else $XQC_BASE/runs, else ~/scratch)."""
    if os.environ.get("XQC_RUNS"):
        return os.environ["XQC_RUNS"]
    base = os.environ.get("XQC_BASE") or os.path.join(
        os.path.realpath(os.path.expanduser("~/scratch")), "xqc-baseline")
    return os.path.join(base, "runs")


def pick_genes(ranked, has_model, n_top, n_bottom):
    """Best ``n_top`` and worst ``n_bottom`` gene ids with a saved model.

    ``ranked`` is [(gene_id, fitness), ...]; the two groups never overlap.
    """
    usable = [(g, f) for g, f in sorted(ranked, key=lambda gf: -gf[1]) if has_model(g)]
    top = [g for g, _ in usable[:n_top]]
    bottom = [g for g, _ in usable[::-1] if g not in top][:n_bottom]
    return top, bottom


def refresh_aux_reward(*, generation, ranked, model_path, kind, out_dir, previous_file,
                       archive_cells=None, local=False):
    """Run one generation's VLM step; return the aux-reward file to train with next.

    model_path(gene_id) gives the saved model of a gene (TD3 .zip, or XQC run
    dir holding actor.pkl); kind is "td3" or "xqc"; out_dir is the run's VLM
    folder (captures/ and one aux_reward_gen_<g>.json per generation).
    """
    def has_model(gene_id):
        path = model_path(gene_id)
        if kind == "xqc":
            return os.path.isfile(os.path.join(path, "actor.pkl"))
        return os.path.isfile(path)

    top, bottom = pick_genes(ranked, has_model, run_options.VLM_TOP_GENES,
                             run_options.VLM_BOTTOM_GENES)
    if not top and not bottom:
        print(f"\t☠ VLM: no gene with a saved model to watch; keeping {previous_file}", flush=True)
        return previous_file

    os.makedirs(out_dir, exist_ok=True)
    captures = os.path.join(out_dir, "captures")
    cache = os.path.join(out_dir, f"aux_reward_gen_{generation}.json")
    parents_file = os.path.join(out_dir, f"parents_gen_{generation}.txt")
    fitness_file = os.path.join(out_dir, f"fitness_gen_{generation}.json")
    fitness = dict(ranked)
    with open(parents_file, "w") as f:
        for gene in top + bottom:
            f.write(f"{gene} {kind} {model_path(gene)}\n")
    with open(fitness_file, "w") as f:
        json.dump({g: fitness[g] for g in top + bottom}, f)

    env = dict(os.environ)
    env.update({
        "VLM_PARENTS_FILE": parents_file, "VLM_CAPTURE_ROOT": captures,
        "VLM_ENV_ID": run_options.ENV_ID,
        "AUX_TOP_GENES": ",".join(top), "AUX_BOTTOM_GENES": ",".join(bottom),
        "AUX_CACHE_PATH": cache, "AUX_GENERATION": str(generation),
        "AUX_BEST_SO_FAR": str(max(fitness.values())), "AUX_FITNESS_MAP": fitness_file,
        "AUX_MODEL_ID": run_options.VLM_MODEL_ID, "HF_HOME": run_options.VLM_HF_HOME,
    })
    if archive_cells is not None:
        env["AUX_ARCHIVE_COVERAGE"] = str(archive_cells)
    if previous_file and os.path.isfile(previous_file):
        env["AUX_REFLECTION_PATH"] = previous_file
    log = os.path.join(out_dir, f"vlm_gen_{generation}.log")
    # Local runs: the bash on PATH (on Windows, not System32's WSL launcher).
    cmd = ([shutil.which("bash") or "bash", JOB_SCRIPT] if local else
           ["sbatch", "--wait", "--parsable", "--output", log, JOB_SCRIPT])
    print(f"\t‣ VLM generation {generation}: watching best {top} and worst {bottom}; log {log}",
          flush=True)
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=local, text=True,
                              timeout=run_options.VLM_TIMEOUT_SECONDS)
        if local:
            with open(log, "w") as f:
                f.write(proc.stdout + proc.stderr)
        ok = proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"\t☠ VLM job failed to run: {exc}", flush=True)
        ok = False

    record = None
    if os.path.isfile(cache):
        try:
            with open(cache) as f:
                record = json.load(f)
        except (OSError, ValueError):
            record = None
    if not ok or record is None or record.get("fallback_reason"):
        reason = (record or {}).get("fallback_reason") or "job failed or wrote no file"
        print(f"\t☠ VLM gave no usable aux reward ({reason}); keeping {previous_file}", flush=True)
        return previous_file
    code = (record.get("aux_reward_code") or "").strip()
    print(f"\t☑ VLM aux reward for the next generation ({record.get('validation')}): {cache}\n"
          f"{code}", flush=True)
    return cache
