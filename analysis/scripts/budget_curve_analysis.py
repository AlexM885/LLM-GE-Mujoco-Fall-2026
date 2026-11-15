"""Would a shorter training budget per gene change which genes win?

Reads the learning curves XQC genes already wrote (eval.csv in each gene's run
dir: return every eval_interval steps) and, for each candidate budget k,
compares the return at k with the final return:

  median_return   median gene return at k
  pct_of_final    median of (return at k / final return), per gene
  spearman        rank correlation of returns at k vs final, across genes:
                  1.0 = the archive would rank genes the same way
  kendall         Kendall tau-b of returns at k vs final: the share of gene
                  pairs ordered the same way minus the share ordered the other
                  way (0.8 = 90% of pairs agree). Stricter than spearman.
  p_kendall       two-sided p-value for kendall != 0 (normal approximation,
                  no tie correction: fine for continuous returns)
  top25_kept      share of the final top-25% genes already top-25% at k
  top25_in_half   share of the final top-25% genes in the top half at k: what
                  stopping the bottom half of genes at step k would keep

A high kendall at an early step means the early score can rank genes, so the
bottom ones can be stopped there and only the rest trained to the full budget.

Only genes trained to --final steps with a curve point at every step are used,
so pass the run's per-gene budget. Standard library + numpy only; --plot also
needs matplotlib.

    python analysis/scripts/budget_curve_analysis.py ~/scratch/xqc-baseline/runs/llmge --final 500000 --plot budget_curve.png

Caveat: eval.csv holds returns only, not foot contact, so this cannot show
whether a gene would land in a different cell at the shorter budget.
"""

import argparse
import csv
import glob
import json
import math
import os

import numpy as np


def load_curves(root, final, num_seeds):
    curves = {}
    for path in glob.glob(os.path.join(root, "*", "eval.csv")):
        run_dir = os.path.dirname(path)
        config = os.path.join(run_dir, "config.json")
        if num_seeds and os.path.exists(config):
            with open(config) as f:
                if json.load(f).get("num_seeds") != num_seeds:
                    continue
        by_step = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                by_step.setdefault(int(row["env_step"]), []).append(float(row["return"]))
        if final in by_step:
            curves[os.path.basename(run_dir)] = {s: float(np.mean(r)) for s, r in by_step.items()}
    return curves


def ranks(x):
    order = np.argsort(x)
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r


def kendall_tau(x, y):
    """Kendall tau-b (tie-corrected) and its two-sided p-value under no association."""
    n = len(x)
    i, j = np.triu_indices(n, 1)
    dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
    pairs = n * (n - 1) / 2
    ties_x, ties_y = np.sum(dx == 0), np.sum(dy == 0)
    denom = math.sqrt((pairs - ties_x) * (pairs - ties_y))
    tau = float(np.sum(dx * dy) / denom) if denom else float("nan")
    z = 3 * tau * math.sqrt(n * (n - 1)) / math.sqrt(2 * (2 * n + 5))
    return tau, math.erfc(abs(z) / math.sqrt(2))


def plot(path, genes, curves, steps, final_step, taus):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final = np.array([curves[g][final_step] for g in genes])
    # Scatter at roughly 10%, 20%, 40% and 60% of the budget (60% of 500k = 300k).
    picks = sorted({min(steps, key=lambda s: abs(s - f * final_step)) for f in (0.1, 0.2, 0.4, 0.6)})
    fig, axes = plt.subplots(1, len(picks) + 1, figsize=(4 * (len(picks) + 1), 4))
    lim = [0, final.max() * 1.05]
    for ax, s in zip(axes, picks):
        at = np.array([curves[g][s] for g in genes])
        ax.plot(lim, lim, color="#bbbbbb", lw=1, zorder=0)
        ax.scatter(at, final, s=24, color="#2a6fdb", edgecolor="white", linewidth=1)
        ax.set(xlim=[0, max(lim[1], at.max() * 1.05)], ylim=lim,
               xlabel=f"return at {s:,} steps ({100 * s / final_step:.0f}%)",
               title=f"Kendall tau = {taus[s]:.2f}")
        ax.set_ylabel(f"return at {final_step:,} steps")
    ax = axes[-1]
    ax.plot(steps, [taus[s] for s in steps], color="#2a6fdb", lw=2, marker="o", ms=4)
    ax.set(ylim=[min(0, min(taus.values())), 1], xlabel="env steps",
           title=f"Kendall tau vs final ({len(genes)} genes)")
    ax.set_ylabel("Kendall tau")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    for a in axes:
        a.grid(alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nwrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs_root", help="folder with one run dir per gene (XQC_RUNS/llmge)")
    p.add_argument("--final", type=int, default=500_000, help="the run's steps per gene")
    p.add_argument("--num-seeds", type=int, default=1, help="only genes trained with this many seeds (0 = all)")
    p.add_argument("--plot", help="save return-at-k vs final scatters and the Kendall tau curve to this PNG")
    a = p.parse_args()

    curves = load_curves(os.path.expanduser(a.runs_root), a.final, a.num_seeds)
    steps = sorted(set.intersection(*(set(c) for c in curves.values()))) if curves else []
    steps = [s for s in steps if s > 1]
    # Failed or degenerate genes (falling at once) carry no signal about the budget.
    genes = [g for g, c in curves.items() if c[a.final] > 500]
    print(f"{len(curves)} genes reached {a.final:,} steps with a learning curve, "
          f"{len(genes)} of them with final return > 500\n")
    if len(genes) < 5:
        raise SystemExit("too few genes to compare")
    final = np.array([curves[g][a.final] for g in genes])
    top = set(np.argsort(final)[-max(1, len(genes) // 4):])
    taus = {}
    print(f"{'steps':>9} {'median_return':>14} {'pct_of_final':>13} {'spearman':>9} "
          f"{'kendall':>8} {'p_kendall':>10} {'top25_kept':>11} {'top25_in_half':>14}")
    for s in steps:
        at = np.array([curves[g][s] for g in genes])
        rho = float(np.corrcoef(ranks(at), ranks(final))[0, 1])
        taus[s], p_tau = kendall_tau(at, final)
        kept = len(top & set(np.argsort(at)[-len(top):])) / len(top)
        in_half = len(top & set(np.argsort(at)[-math.ceil(len(genes) / 2):])) / len(top)
        print(f"{s:>9,} {np.median(at):>14,.0f} {100 * np.median(at / final):>12.0f}% "
              f"{rho:>9.2f} {taus[s]:>8.2f} {p_tau:>10.1e} {100 * kept:>10.0f}% {100 * in_half:>13.0f}%")
    if a.plot:
        plot(a.plot, genes, curves, steps, a.final, taus)


if __name__ == "__main__":
    main()
