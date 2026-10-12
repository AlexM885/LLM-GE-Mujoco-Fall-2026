"""Pick genes from both arms of a matched pair and submit their post-hoc checks.

Per arm, the same number of genes (default 10):
  - the top ``--k-top`` (5) valid genes by the run's reward;
  - ``--k-diverse`` (5) more from the arm's final archive, chosen to be as far
    apart as possible in (distance, control cost), normalised over both arms
    (farthest-point selection, starting from the farthest from the top genes);
    if the archive has too few elites, the next-best valid genes fill in.
Plus the seed gene (sota/XQC/llmge/network.py) once, as the reference.

Each gene becomes one GPU job (scripts/mesb/matched/posthoc_gene.sbatch ->
posthoc_eval.py) writing mujoco_rl_output/matched/<pair>/posthoc/<arm>/<gene>.json.
The selection is saved to posthoc/selection.json. compare_arms.py reads the
results when they exist.

    uv run python scripts/mesb/matched/posthoc_submit.py --pair xqc_matched_p1
    uv run python scripts/mesb/matched/posthoc_submit.py --pair xqc_matched_p1 --dry-run
"""

import argparse
import csv
import glob
import json
import math
import os
import pickle
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ARMS = ("fixed", "mesb")
GENES_DIR = REPO_ROOT / "sota" / "XQC" / "llmge" / "models"
SEED_GENE = REPO_ROOT / "sota" / "XQC" / "llmge" / "network.py"
JOB_SCRIPT = REPO_ROOT / "scripts" / "mesb" / "matched" / "posthoc_gene.sbatch"


def ok_genes(report_dir):
    rows = []
    with open(Path(report_dir) / "genes.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "ok":
                continue
            try:
                r["mean_reward"] = float(r["mean_reward"])
                r["d"], r["c"] = float(r["mean_distance"]), float(r["mean_control_cost"])
            except ValueError:
                continue
            if all(math.isfinite(v) for v in (r["mean_reward"], r["d"], r["c"])):
                rows.append(r)
    return rows


def final_archive_ids(run_dir):
    ckpts = sorted(glob.glob(f"{run_dir}/island_*/checkpoint_gen_*.pkl"),
                   key=lambda p: int(p.rsplit("_", 1)[1][:-4]))
    if not ckpts:
        return set()
    try:
        from deap import base, creator
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti, file_id=None)
    except ImportError:
        pass
    with open(ckpts[-1], "rb") as fh:
        archive = pickle.load(fh).get("MAP_ELITES_ARCHIVE", {})
    return {ind[0] for ind in archive.values()}


def select(rows, archive_ids, k_top, k_div, scale):
    rows = sorted(rows, key=lambda r: r["mean_reward"], reverse=True)
    chosen = rows[:k_top]
    pos = lambda r: ((r["d"] - scale[0][0]) / scale[0][1], (r["c"] - scale[1][0]) / scale[1][1])  # noqa: E731
    pool = [r for r in rows if r["gene_id"] in archive_ids and r not in chosen]
    for _ in range(k_div):
        if not pool:
            break
        far = max(pool, key=lambda r: min(math.dist(pos(r), pos(c)) for c in chosen) if chosen else 0)
        chosen.append(far)
        pool.remove(far)
    for r in rows:                                    # top up if the archive was too small
        if len(chosen) >= k_top + k_div:
            break
        if r not in chosen:
            chosen.append(r)
    return chosen


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pair", required=True)
    p.add_argument("--base", default=str(REPO_ROOT / "mujoco_rl_output" / "matched"))
    p.add_argument("--k-top", type=int, default=5)
    p.add_argument("--k-diverse", type=int, default=5)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--sbatch-flags", default=os.getenv("GENE_SBATCH_FLAGS", ""))
    p.add_argument("--dry-run", action="store_true", help="write selection.json, print commands only")
    args = p.parse_args(argv)
    pair = Path(args.base) / args.pair
    reports = pair / "compare"
    if not all((reports / arm / "genes.csv").exists() for arm in ARMS):
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts/mesb/matched/compare_arms.py"),
                        "--pair", args.pair, "--base", args.base], check=True, cwd=REPO_ROOT)
    rows = {arm: ok_genes(reports / arm) for arm in ARMS}
    pooled = rows["fixed"] + rows["mesb"]
    scale = []
    for k in ("d", "c"):
        lo, hi = min(r[k] for r in pooled), max(r[k] for r in pooled)
        scale.append((lo, (hi - lo) or 1.0))
    selection = {"seed": [{"gene_id": "seed", "gene_file": str(SEED_GENE), "role": "reference"}]}
    for arm in ARMS:
        chosen = select(rows[arm], final_archive_ids(pair / arm), args.k_top, args.k_diverse, scale)
        selection[arm] = [{"gene_id": r["gene_id"],
                           "gene_file": str(GENES_DIR / f"network_{r['gene_id']}.py"),
                           "run_reward": r["mean_reward"], "distance": r["d"], "control_cost": r["c"],
                           "role": "top" if i < args.k_top else "diverse"}
                          for i, r in enumerate(chosen)]
    out = pair / "posthoc"
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(
        {"seed": args.seed, "timesteps": args.timesteps, "num_seeds": args.num_seeds,
         "k_top": args.k_top, "k_diverse": args.k_diverse, "genes": selection}, indent=2))
    n = 0
    for arm, genes in selection.items():
        for g in genes:
            log = out / arm / f"{g['gene_id']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = (["sbatch", "--parsable", "--job-name", f"posthoc-{arm}-{g['gene_id'][:10]}",
                    "--output", str(log)] + shlex.split(args.sbatch_flags) + [str(JOB_SCRIPT),
                   "--gene-file", g["gene_file"], "--gene-id", g["gene_id"], "--arm", arm,
                   "--out-json", str(out / arm / f"{g['gene_id']}.json"),
                   "--run-dir", str(out / arm / "runs" / g["gene_id"]), "--seed", str(args.seed),
                   "--timesteps", str(args.timesteps), "--num-seeds", str(args.num_seeds)])
            if args.dry_run:
                print(" ".join(shlex.quote(c) for c in cmd))
            else:
                res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
                print(f"{arm:5s} {g['gene_id']}: {res.stdout.strip() or res.stderr.strip()}")
            n += 1
    print(f"{n} post-hoc jobs {'listed' if args.dry_run else 'submitted'}; selection in {out}/selection.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
