"""Why do gene rankings change late in training, and which metric should judge it?

Follow-up to budget_curve_analysis.py. Reads the same per-gene learning curves
(eval.csv: one row per seed per checkpoint, each the mean of eval_episodes
episodes) but keeps the seeds apart, then reports:

1. Agreement of the ranking at each checkpoint k with the final one, by several
   metrics (columns of the table):
     pearson     correlation of the returns themselves: uses how far apart genes
                 are, so a swap between two near-equal genes barely lowers it
     spearman    rank correlation (all pairs, squared rank differences)
     kendall     share of gene pairs in the same order minus share swapped
     wkendall    Kendall tau weighted towards the top of the ranking (scipy's
                 weightedtau): swaps among the best genes count most
     adj_kendall Kendall tau of k vs the previous checkpoint (k - interval): how
                 much the ranking moves in one step; late values are a ceiling
                 for any "k vs final" metric
     top10/top25 share of the final top 10% / 25% already in the top 10% / 25% at k
     regret      final return of the best gene at k, short of the final best
                 (0 = picking at k would have found the final winner)
     top10_loss  mean final return of the top 10% at k vs of the final top 10%
2. Case studies: of the genes in the top 25% at --compare (default: the
   checkpoint before the final), the --cases with the largest rank drop by the
   final checkpoint. For each, the per-seed returns over the last checkpoints,
   how many other genes are within 2% of it, and a likely reason:
     one seed collapsed   one seed lost >25% while the others held (< 10% loss)
     all seeds dropped    every seed lower and the mean down >10%
     near-tie             mean changed <3% but neighbours were that close
     noisy curve          its own last checkpoints swing by >10% of their mean
   (eval.csv holds per-seed means only, not single episodes, so a fall inside
   one evaluation shows up as a lower seed value, not as a single episode.)

Genes need a curve point at every checkpoint up to --final and a final return
> 500, as in budget_curve_analysis.py. numpy only; scipy (weighted Kendall,
p-values) and matplotlib (--plot) are used when installed.

    python analysis/scripts/rank_stability.py <XQC_RUNS>/llmge --final 500000 --num-seeds 3 \\
        --compare 475000 --cases 6 --plot rank_stability.png | tee rank_stability.txt
"""

import argparse
import csv
import glob
import json
import os

import numpy as np


def load_curves(root, final, num_seeds):
    """{gene: {step: per-seed returns, ordered by seed}} for genes that reached `final`."""
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
                by_step.setdefault(int(row["env_step"]), {})[int(row["seed"])] = float(row["return"])
        if final in by_step:
            curves[os.path.basename(run_dir)] = {
                s: np.array([v[k] for k in sorted(v)]) for s, v in by_step.items()}
    return curves


def ranks(x):
    """Average ranks (ties share their mean rank), 0 = lowest."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        tie = x == v
        if tie.sum() > 1:
            r[tie] = r[tie].mean()
    return r


def kendall(x, y):
    i, j = np.triu_indices(len(x), 1)
    dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
    denom = np.sqrt(np.sum(dx != 0) * np.sum(dy != 0))
    return float(np.sum(dx * dy) / denom) if denom else float("nan")


def weighted_kendall(x, y):
    try:
        from scipy.stats import weightedtau
    except ImportError:
        return float("nan")
    return float(weightedtau(x, y).statistic)


def top_overlap(at, final, frac):
    k = max(1, int(round(len(final) * frac)))
    return len(set(np.argsort(final)[-k:]) & set(np.argsort(at)[-k:])) / k


def metrics(at, final, prev):
    k = max(1, int(round(len(final) * 0.1)))
    return {
        "pearson": float(np.corrcoef(at, final)[0, 1]),
        "spearman": float(np.corrcoef(ranks(at), ranks(final))[0, 1]),
        "kendall": kendall(at, final),
        "wkendall": weighted_kendall(at, final),
        "adj_kendall": kendall(prev, at) if prev is not None else float("nan"),
        "top10": top_overlap(at, final, 0.10),
        "top25": top_overlap(at, final, 0.25),
        "regret": float(final.max() - final[np.argmax(at)]),
        "top10_loss": float(np.sort(final)[-k:].mean() - final[np.argsort(at)[-k:]].mean()),
    }


def reason(per_seed_before, per_seed_after, mean_before, mean_after, earlier_means, near):
    """earlier_means: the gene's mean return at the checkpoints before the final one."""
    change = (per_seed_after - per_seed_before) / np.maximum(np.abs(per_seed_before), 1.0)
    mean_change = (mean_after - mean_before) / max(abs(mean_before), 1.0)
    swing = (earlier_means.max() - earlier_means.min()) / max(abs(earlier_means.mean()), 1.0)
    found = []
    if len(change) > 1 and change.min() < -0.25 and np.sort(change)[1] > -0.10:
        found.append(f"one seed collapsed (seed {int(np.argmin(change))}: {100 * change.min():+.0f}%)")
    elif change.max() < -0.05 and mean_change < -0.10:
        found.append("all seeds dropped")
    if abs(mean_change) < 0.03:
        found.append(f"near-tie ({near} other genes within 2%)" if near >= 2
                     else "held steady; others overtook it")
    if swing > 0.10:
        found.append(f"noisy curve (earlier checkpoints swing {100 * swing:.0f}%)")
    return "; ".join(found) or f"moderate drop ({100 * mean_change:+.0f}%), no single cause"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs_root", help="folder with one run dir per gene (XQC_RUNS/llmge)")
    p.add_argument("--final", type=int, default=500_000, help="the run's steps per gene")
    p.add_argument("--num-seeds", type=int, default=0, help="only genes trained with this many seeds (0 = all)")
    p.add_argument("--compare", type=int, help="checkpoint for the case studies (default: the one before --final)")
    p.add_argument("--cases", type=int, default=6, help="number of case-study genes")
    p.add_argument("--history", type=int, default=5, help="checkpoints shown per case")
    p.add_argument("--plot", help="save metrics vs checkpoint and the case-study curves to this PNG")
    a = p.parse_args()

    curves = load_curves(os.path.expanduser(a.runs_root), a.final, a.num_seeds)
    if not curves:
        raise SystemExit(f"no gene reached {a.final:,} steps under {a.runs_root}")
    steps = sorted(set.intersection(*(set(c) for c in curves.values())))
    steps = [s for s in steps if 1 < s <= a.final]
    genes = sorted(g for g, c in curves.items() if c[a.final].mean() > 500 and all(s in c for s in steps))
    seeds = {len(curves[g][a.final]) for g in genes}
    print(f"{len(curves)} genes reached {a.final:,} steps; {len(genes)} with a full curve and final return > 500; "
          f"seeds per gene: {sorted(seeds)}\n")
    if len(genes) < 5:
        raise SystemExit("too few genes to compare")
    mean = {s: np.array([curves[g][s].mean() for g in genes]) for s in steps}
    final = mean[a.final]

    # -------------------------------------------------------- 1. metrics table
    rows = {}
    cols = ["pearson", "spearman", "kendall", "wkendall", "adj_kendall", "top10", "top25", "regret", "top10_loss"]
    print(f"{'steps':>9} " + " ".join(f"{c:>11}" for c in cols))
    for i, s in enumerate(steps):
        rows[s] = metrics(mean[s], final, mean[steps[i - 1]] if i else None)
        cells = [f"{100 * rows[s][c]:>10.0f}%" if c.startswith("top") and c != "top10_loss"
                 else f"{rows[s][c]:>11,.0f}" if c in ("regret", "top10_loss")
                 else f"{rows[s][c]:>11.2f}" for c in cols]
        print(f"{s:>9,} " + " ".join(cells))
    spread = np.percentile(final, [10, 25, 50, 75, 90])
    print(f"\nfinal returns: p10 {spread[0]:,.0f}  p25 {spread[1]:,.0f}  median {spread[2]:,.0f}  "
          f"p75 {spread[3]:,.0f}  p90 {spread[4]:,.0f}  best {final.max():,.0f}")
    if len(seeds) == 1 and next(iter(seeds)) > 1:
        seed_sd = np.median([curves[g][a.final].std() for g in genes])
        print(f"median seed-to-seed std of a gene's final return: {seed_sd:,.0f} "
              f"(vs interquartile spread of genes {spread[3] - spread[1]:,.0f})")

    # -------------------------------------------------------- 2. case studies
    compare = a.compare or steps[-2]
    if compare not in mean:
        raise SystemExit(f"--compare {compare} is not a checkpoint; have {steps}")
    r_cmp, r_fin = len(genes) - ranks(mean[compare]), len(genes) - ranks(final)  # 1 = best
    top = np.argsort(mean[compare])[-max(1, len(genes) // 4):]
    cases = sorted(top, key=lambda i: r_cmp[i] - r_fin[i])[:a.cases]
    hist = steps[max(0, steps.index(a.final) - a.history + 1):]
    print(f"\nCase studies: top-25% genes at {compare:,} with the largest rank drop by {a.final:,} "
          f"(rank 1 = best of {len(genes)})")
    for i in cases:
        g = genes[i]
        near = int(np.sum(np.abs(mean[compare] - mean[compare][i]) < 0.02 * abs(mean[compare][i]))) - 1
        why = reason(curves[g][compare], curves[g][a.final], mean[compare][i], final[i],
                     np.array([curves[g][s].mean() for s in hist[:-1]]), near)
        print(f"\n  {g}: rank {r_cmp[i]:.0f} -> {r_fin[i]:.0f}, return {mean[compare][i]:,.0f} -> {final[i]:,.0f}"
              f"  [{why}]")
        print("    " + f"{'seed':>6} " + " ".join(f"{s // 1000:>7}k" for s in hist))
        for k in range(len(curves[g][a.final])):
            print("    " + f"{k:>6} " + " ".join(f"{curves[g][s][k]:>8,.0f}" for s in hist))
        print("    " + f"{'mean':>6} " + " ".join(f"{curves[g][s].mean():>8,.0f}" for s in hist))

    if a.plot:
        plot(a.plot, steps, rows, genes, curves, cases, compare, a.final, mean)


def plot(path, steps, rows, genes, curves, cases, compare, final_step, mean):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"pearson": "#2a6fdb", "spearman": "#2e9e5b", "kendall": "#e0832a", "wkendall": "#8a4fd1",
              "adj_kendall": "#9aa0a6"}
    labels = {"pearson": "Pearson r (returns)", "spearman": "Spearman rho", "kendall": "Kendall tau",
              "wkendall": "Kendall tau, top-weighted", "adj_kendall": "Kendall tau vs previous checkpoint"}
    ncase = len(cases)
    fig = plt.figure(figsize=(14, 4.5 + 2.6 * ((ncase + 2) // 3)))
    grid = fig.add_gridspec(1 + (ncase + 2) // 3, 3)
    ax = fig.add_subplot(grid[0, :2])
    x = [s / 1000 for s in steps[:-1]]
    for key, color in colors.items():
        ax.plot(x, [rows[s][key] for s in steps[:-1]], marker="o", ms=3, lw=2 if key != "adj_kendall" else 1.5,
                ls="-" if key != "adj_kendall" else "--", color=color, label=labels[key])
    ax.set(xlabel="checkpoint k (thousand steps)", ylabel=f"agreement with {final_step // 1000}k",
           title="How well checkpoint k predicts the final result", ylim=(-0.3, 1.0))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    ax = fig.add_subplot(grid[0, 2])
    ax.scatter(mean[compare], mean[final_step], s=14, color="#9aa0a6")
    ax.scatter(mean[compare][cases], mean[final_step][cases], s=30, color="#d1453b", label="case studies")
    lim = [min(mean[compare].min(), mean[final_step].min()), max(mean[compare].max(), mean[final_step].max())]
    ax.plot(lim, lim, color="#5f6670", lw=1)
    ax.set(xlabel=f"return at {compare // 1000}k", ylabel=f"return at {final_step // 1000}k",
           title=f"{compare // 1000}k vs {final_step // 1000}k (n={len(genes)})")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    xs = np.array(steps) / 1000
    for n, i in enumerate(cases):
        ax = fig.add_subplot(grid[1 + n // 3, n % 3])
        g = genes[i]
        per_seed = np.array([curves[g][s] for s in steps])
        for k in range(per_seed.shape[1]):
            ax.plot(xs, per_seed[:, k], lw=1, alpha=0.7, label=f"seed {k}" if n == 0 else None)
        ax.plot(xs, per_seed.mean(axis=1), lw=2.2, color="black", label="mean" if n == 0 else None)
        ax.axvline(compare / 1000, color="#9aa0a6", ls=":", lw=1)
        ax.set_title(g[:18], fontsize=9)
        ax.grid(alpha=0.25)
        if n == 0:
            ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
