"""Kendall tau of gene rankings: return at step k vs final, on the two midterm Walker runs.

Same measure as Alex's analysis/scripts/budget_curve_analysis.py (da9fd3d), plus
two checks that separate evaluation noise from budget:
  * test-retest: the 500k learning-curve eval vs the gene's official fitness
    (a separate 10-episode rollout of the same final policy). Nothing earlier can
    rank genes better than two evaluations of the same final policy agree.
  * smoothed final: mean of the 450k/475k/500k evals as the "final" score.
Genes: 500k steps x 1 seed, complete curve (21 points), final return > 500.

Results for the two midterm runs (Nov 14): analysis/results/kendall_walker/.

    python analysis/scripts/kendall_walker.py \
        --mesb-models   <midterm>/mujoco_rl_output/walker_xqc_vlm_mesb/models \
        --mesb-evals    <midterm>/mujoco_rl_output/walker_xqc_vlm_mesb/evaluations.csv \
        --emitters-runs $XQC_RUNS/llmge \
        --emitters-results <midterm2>/sota/XQC/llmge/results \
        --out analysis/results/kendall_walker
"""
import csv
import glob
import json
import os
import argparse

import numpy as np
from scipy.stats import kendalltau

STEPS = [25_000 * i for i in range(1, 21)]


def curves(root):
    out = {}
    for path in glob.glob(os.path.join(root, "*", "eval.csv")):
        by = {}
        for row in csv.DictReader(open(path)):
            by.setdefault(int(row["env_step"]), []).append(float(row["return"]))
        if all(s in by for s in STEPS):
            out[os.path.basename(os.path.dirname(path))] = {s: float(np.mean(by[s])) for s in STEPS}
    return out


def official_mesb(path):
    """Official fitness of each gene of a run_mesb.py run (its evaluations.csv)."""
    fit = {}
    for row in csv.DictReader(open(path)):
        if row["status"] == "success":
            fit[row["gene_id"]] = float(row["mean_reward"])
    return fit


def official_emitters(results_dir):
    """Official fitness of each gene of a run_improved.py run (<gene>_results.csv files)."""
    fit = {}
    for path in glob.glob(os.path.join(results_dir, "*_results.csv")):
        rows = list(csv.DictReader(open(path)))
        if rows and float(rows[-1]["mean_reward"]) > -1e5:
            fit[os.path.basename(path).replace("_results.csv", "")] = float(rows[-1]["mean_reward"])
    return fit


def tau(a, b):
    return float(kendalltau(a, b).statistic)


def analyse(name, root, official):
    c = curves(root)
    genes = sorted(g for g in c if c[g][500_000] > 500 and g in official)
    final = np.array([c[g][500_000] for g in genes])
    smooth = np.array([np.mean([c[g][s] for s in (450_000, 475_000, 500_000)]) for g in genes])
    fit = np.array([official[g] for g in genes])
    res = {"name": name, "n": len(genes),
           "retest": tau(final, fit),
           "retest_smooth": tau(smooth, fit),
           "vs_final": {s: tau([c[g][s] for g in genes], final) for s in STEPS},
           "vs_official": {s: tau([c[g][s] for g in genes], fit) for s in STEPS},
           "vs_smooth": {s: tau([c[g][s] for g in genes], smooth) for s in STEPS}}
    print(f"\n== {name}: {len(genes)} genes (complete curve, final > 500, official fitness found)")
    print(f"test-retest: 500k eval vs official fitness  tau = {res['retest']:.2f}")
    print(f"             smoothed final vs official     tau = {res['retest_smooth']:.2f}")
    print(f"{'steps':>8} {'vs 500k eval':>13} {'vs official':>12} {'vs smoothed':>12}")
    for s in STEPS:
        print(f"{s:>8,} {res['vs_final'][s]:>13.2f} {res['vs_official'][s]:>12.2f} {res['vs_smooth'][s]:>12.2f}")
    return res


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.2), sharey=True)
    for ax, r in zip(axes, results):
        x = [s / 1000 for s in STEPS]
        ax.plot(x, [r["vs_final"][s] for s in STEPS], marker="o", ms=3, lw=2, color="#2a6fdb",
                label="vs 500k learning-curve eval (Alex's measure)")
        ax.plot(x, [r["vs_official"][s] for s in STEPS], marker="o", ms=3, lw=2, color="#e0832a",
                label="vs official fitness (separate final eval)")
        ax.axhline(r["retest"], color="#888", ls="--", lw=1.5,
                   label=f"noise ceiling: same policy evaluated twice ({r['retest']:.2f})")
        ax.set(title=f"{r['name']} ({r['n']} genes, Walker2d-v5, 1 seed)", xlabel="training steps (k)",
               ylim=(-0.2, 1.02))
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Kendall tau of gene ranking")
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mesb-models", required=True, help="run_mesb.py run: folder with one dir per gene (eval.csv)")
    p.add_argument("--mesb-evals", required=True, help="run_mesb.py run: evaluations.csv")
    p.add_argument("--emitters-runs", required=True, help="run_improved.py run: XQC_RUNS/llmge (one dir per gene)")
    p.add_argument("--emitters-results", required=True, help="run_improved.py run: sota/XQC/llmge/results")
    p.add_argument("--out", default=".", help="folder for kendall_results.json and kendall_walker.png")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    results = [analyse("Sliding boundaries run", a.mesb_models, official_mesb(a.mesb_evals)),
               analyse("LLM emitters run", a.emitters_runs, official_emitters(a.emitters_results))]
    json.dump(results, open(os.path.join(a.out, "kendall_results.json"), "w"), indent=1, default=str)
    plot(results, os.path.join(a.out, "kendall_walker.png"))
