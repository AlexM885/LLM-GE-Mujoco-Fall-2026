"""Would a shorter training budget per gene change which genes win?

Reads the learning curves XQC genes already wrote (eval.csv in each gene's run
dir: return every eval_interval steps) and, for each candidate budget k,
compares the return at k with the final return:

  median_return   median gene return at k
  pct_of_final    median of (return at k / final return), per gene
  spearman        rank correlation of returns at k vs final, across genes:
                  1.0 = the archive would rank genes the same way
  top25_kept      share of the final top-25% genes already top-25% at k

Only genes trained to --final steps with a curve point at every step are used,
so pass the run's per-gene budget. Standard library + numpy only.

    python analysis/scripts/budget_curve_analysis.py ~/scratch/xqc-baseline/runs/llmge --final 500000

Caveat: eval.csv holds returns only, not foot contact, so this cannot show
whether a gene would land in a different cell at the shorter budget.
"""

import argparse
import csv
import glob
import json
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


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs_root", help="folder with one run dir per gene (XQC_RUNS/llmge)")
    p.add_argument("--final", type=int, default=500_000, help="the run's steps per gene")
    p.add_argument("--num-seeds", type=int, default=1, help="only genes trained with this many seeds (0 = all)")
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
    print(f"{'steps':>9} {'median_return':>14} {'pct_of_final':>13} {'spearman':>9} {'top25_kept':>11}")
    for s in steps:
        at = np.array([curves[g][s] for g in genes])
        rho = float(np.corrcoef(ranks(at), ranks(final))[0, 1])
        kept = len(top & set(np.argsort(at)[-len(top):])) / len(top)
        print(f"{s:>9,} {np.median(at):>14,.0f} {100 * np.median(at / final):>12.0f}% "
              f"{rho:>9.2f} {100 * kept:>10.0f}%")


if __name__ == "__main__":
    main()
