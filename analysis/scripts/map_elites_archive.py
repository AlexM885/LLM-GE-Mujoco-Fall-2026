# map_elites_archive.py
# Read-only analysis of MAP-Elites checkpoints produced by run_improved.py.
#
# Reports archive coverage, QD-score (raw and offset-adjusted) and the best
# elite per generation; optionally renders the behaviour grid as a heatmap per
# generation plus an animated GIF, and reports per-gene robustness from the
# per-episode series in the evaluation stats.
#
# Usage:
#   uv run python analysis/scripts/map_elites_archive.py <checkpoint_dir> \
#       [--global-dir DIR] [--stats-dir DIR] [--qd-offset X] \
#       [--plots OUTDIR] [--gif OUT.gif]

from pathlib import Path
import argparse
import json
import math
import pickle
import re
import sys

# DEAP needs the creator classes registered before a checkpoint unpickles.
from deap import base, creator, tools  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.cfg.constants import (  # noqa: E402
    FITNESS_WEIGHTS,
    MAP_BINS,
    MAP_ELITES_DESCRIPTORS,
    MAP_ELITES_OBJECTIVE,
)

if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=FITNESS_WEIGHTS)
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti, file_id=None)


def generation_of(filename, prefix):
    match = re.match(prefix + r"_gen_(\d+)\.pkl$", filename)
    return int(match.group(1)) if match else None


def load_checkpoints(checkpoint_dir, prefix):
    """Return (generation, unpickled dict) pairs in generation order."""
    found = []
    for path in Path(checkpoint_dir).glob(prefix + "_gen_*.pkl"):
        gen = generation_of(path.name, prefix)
        if gen is None:
            continue
        try:
            with open(path, "rb") as handle:
                found.append((gen, pickle.load(handle)))
        except Exception as exc:  # a partial write should not abort the report
            print("  ! skipping " + path.name + ": " + str(exc))
    return sorted(found, key=lambda item: item[0])


def collect_metrics(global_dir):
    """gene_id -> named metrics, merged across every global_data checkpoint."""
    metrics = {}
    if global_dir is None:
        return metrics
    for _gen, data in load_checkpoints(global_dir, "global"):
        for key in ("GLOBAL_DATA_HIST", "GLOBAL_DATA"):
            for gene_id, info in (data.get(key) or {}).items():
                if isinstance(info, dict) and info.get("metrics"):
                    metrics[gene_id] = info["metrics"]
    return metrics


def quality_of(gene_id, individual, metrics_by_gene):
    metrics = metrics_by_gene.get(gene_id)
    if metrics and MAP_ELITES_OBJECTIVE in metrics:
        return metrics[MAP_ELITES_OBJECTIVE]
    if individual.fitness.valid:
        return individual.fitness.values[0]
    return float("nan")


# --------------------------------------------------------------------------
# QD-score
# --------------------------------------------------------------------------
def qd_scores(qualities, offset):
    """Raw and offset-adjusted QD-score for one archive.

    raw     = sum of elite fitnesses (what run_improved prints live)
    offset  = sum of (fitness - offset), the pyribs / QDax / Pugh et al.
              convention, so that discovering a negative-fitness niche can
              never lower the score.
    """
    finite = [q for q in qualities if q == q]
    raw = sum(finite)
    adjusted = sum(q - offset for q in finite)
    return raw, adjusted


def summarize(checkpoint_dir, global_dir):
    metrics_by_gene = collect_metrics(global_dir)
    total_cells = MAP_BINS ** len(MAP_ELITES_DESCRIPTORS)
    rows = []

    for gen, data in load_checkpoints(checkpoint_dir, "checkpoint"):
        archive = data.get("MAP_ELITES_ARCHIVE")
        if not archive:
            continue
        qualities = {}
        for cell, ind in archive.items():
            qualities[cell] = quality_of(ind[0], ind, metrics_by_gene)
        finite = [q for q in qualities.values() if q == q]
        best_cell = max(qualities, key=lambda c: qualities[c]) if finite else None
        rows.append({
            "gen": gen,
            "archive": archive,
            "qualities": qualities,
            "filled": len(archive),
            "coverage": len(archive) / total_cells * 100 if total_cells else 0.0,
            "best_cell": best_cell,
            "best_quality": qualities[best_cell] if best_cell else float("nan"),
            "best_gene": archive[best_cell][0] if best_cell else None,
        })
    return rows, total_cells, metrics_by_gene


def resolve_offset(rows, explicit):
    """Pick the QD-score offset and say where it came from.

    The paper offsets by the lowest fitness found across every algorithm
    compared on a task, so pass --qd-offset to share one offset across runs.
    Without it, fall back to the lowest fitness seen in this run.
    """
    if explicit is not None:
        return float(explicit), "supplied via --qd-offset"
    all_q = [q for row in rows for q in row["qualities"].values() if q == q]
    if not all_q:
        return 0.0, "no finite fitness found; defaulted to 0"
    return float(min(all_q)), "auto: lowest fitness in this run"


def print_report(rows, total_cells, offset, offset_note):
    descriptor_names = ", ".join(name for name, _lo, _hi in MAP_ELITES_DESCRIPTORS)
    print("Descriptors : " + descriptor_names)
    print("Grid        : {} bins per axis ({} cells)".format(MAP_BINS, total_cells))
    print("Objective   : " + MAP_ELITES_OBJECTIVE)
    print("QD offset   : {:.4f}  ({})".format(offset, offset_note))
    print()

    if not rows:
        print("No MAP-Elites archive found in these checkpoints.")
        return

    all_q = [q for row in rows for q in row["qualities"].values() if q == q]
    if all_q and offset > min(all_q):
        print("  ! WARNING: offset {:.4f} exceeds the lowest fitness {:.4f}; "
              "some terms are negative.".format(offset, min(all_q)))
        print()

    header = "{:>5}  {:>7}  {:>9}  {:>14}  {:>14}  {:>12}  best gene"
    print(header.format("gen", "filled", "coverage", "QD raw",
                        "QD offset-adj", "best"))
    for row in rows:
        raw, adjusted = qd_scores(row["qualities"].values(), offset)
        row["qd_raw"], row["qd_offset"] = raw, adjusted
        print("{:>5}  {:>7}  {:>8.1f}%  {:>14.2f}  {:>14.2f}  {:>12.2f}  {}".format(
            row["gen"], row["filled"], row["coverage"], raw, adjusted,
            row["best_quality"], row["best_gene"]))

    last = rows[-1]
    print("\nFinal archive: {}/{} cells ({:.1f}% coverage)".format(
        last["filled"], total_cells, last["coverage"]))
    print("  QD-score (raw)           : {:.2f}".format(last["qd_raw"]))
    print("  QD-score (offset-adjusted): {:.2f}".format(last["qd_offset"]))
    print("Best elite   : {} ({}={:.2f}) in cell {}".format(
        last["best_gene"], MAP_ELITES_OBJECTIVE,
        last["best_quality"], last["best_cell"]))

    first_seen = {}
    for row in rows:
        for ind in row["archive"].values():
            first_seen.setdefault(ind[0], row["gen"])
    print("\nBest elite first entered the archive in generation {}".format(
        first_seen.get(last["best_gene"], "?")))


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------
def load_stats(stats_dir):
    """gene_id -> evaluation stats json."""
    stats = {}
    if stats_dir is None:
        return stats
    for path in Path(stats_dir).glob("*_stats.json"):
        gene_id = path.name[: -len("_stats.json")]
        try:
            with open(path, encoding="utf-8") as handle:
                stats[gene_id] = json.load(handle)
        except Exception as exc:  # noqa: BLE001
            print("  ! skipping " + path.name + ": " + str(exc))
    return stats


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def robustness_rows(archive, stats):
    """Per-elite robustness from the per-episode series."""
    out = []
    for cell, ind in archive.items():
        gene_id = ind[0]
        entry = stats.get(gene_id)
        if not entry:
            continue
        rewards = entry.get("rewards") or []
        if not rewards:
            continue
        mean, std = mean_std(rewards)
        # Paper's Fitness Difference: one evaluation vs the mean of all of them.
        single = rewards[0]
        diff_pct = abs(single - mean) / abs(mean) * 100 if mean else float("nan")
        # A less arbitrary estimator: expected deviation of any single episode.
        exp_pct = (sum(abs(r - mean) for r in rewards) / len(rewards)
                   / abs(mean) * 100) if mean else float("nan")

        out.append({
            "cell": cell,
            "gene": gene_id,
            "episodes": len(rewards),
            "mean": mean,
            "std": std,
            "min": min(rewards),
            "max": max(rewards),
            "cv": (std / abs(mean) * 100) if mean else float("nan"),
            "single": single,
            "fitness_diff_pct": diff_pct,
            "expected_diff_pct": exp_pct,
        })
    return out


def print_robustness(rows, stats, best_gene):
    if not rows:
        print("\nNo per-episode stats matched the final archive "
              "(check --stats-dir).")
        return

    print("\n" + "=" * 70)
    print("ROBUSTNESS  ({} of the final archive's elites have stats)".format(len(rows)))
    print("=" * 70)

    means = [r["mean"] for r in rows]
    cvs = [r["cv"] for r in rows if r["cv"] == r["cv"]]
    diffs = [r["fitness_diff_pct"] for r in rows if r["fitness_diff_pct"] == r["fitness_diff_pct"]]
    eps = rows[0]["episodes"]

    print("Episodes per evaluation : {}".format(eps))
    if eps < 2:
        print("  ! Only one episode per gene: no within-gene variance to report.")
        return

    m, s = mean_std(cvs)
    print("Fitness CV across elites: mean {:.2f}%  (std {:.2f}%)".format(m, s))
    m, s = mean_std(diffs)
    print("Fitness Difference      : mean {:.2f}%  (std {:.2f}%)"
          "   [1 eval vs mean of {}]".format(m, s, eps))

    # The paper reports this metric for the max-fitness solution specifically.
    best = next((r for r in rows if r["gene"] == best_gene), None)
    if best:
        print("\nBest elite ({}):".format(best_gene))
        print("  fitness        : mean {:.2f}  std {:.2f}  "
              "min {:.2f}  max {:.2f}".format(
                  best["mean"], best["std"], best["min"], best["max"]))
        print("  single eval    : {:.2f}".format(best["single"]))
        print("  Fitness Diff   : {:.2f}%   (expected over any single episode: "
              "{:.2f}%)".format(best["fitness_diff_pct"], best["expected_diff_pct"]))

    worst = sorted(rows, key=lambda r: -(r["cv"] if r["cv"] == r["cv"] else -1))[:5]
    print("\nLeast reproducible elites (by fitness CV):")
    print("  {:<30} {:>10} {:>10} {:>8}".format("gene", "mean", "std", "CV%"))
    for r in worst:
        print("  {:<30} {:>10.2f} {:>10.2f} {:>7.1f}%".format(
            r["gene"][:30], r["mean"], r["std"], r["cv"]))


def render_plots(rows, plots_dir, gif_path):
    if not rows:
        return
    if len(MAP_ELITES_DESCRIPTORS) != 2:
        print("\nHeatmaps need exactly 2 descriptors; skipping plots.")
        return
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("\nPlotting skipped (" + str(exc) + "). Install matplotlib to enable it.")
        return

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    x_name, x_lo, x_hi = MAP_ELITES_DESCRIPTORS[0]
    y_name, y_lo, y_hi = MAP_ELITES_DESCRIPTORS[1]
    finite_all = [q for row in rows for q in row["qualities"].values() if q == q]
    vmin, vmax = (min(finite_all), max(finite_all)) if finite_all else (0, 1)

    frames = []
    for row in rows:
        grid = np.full((MAP_BINS, MAP_BINS), np.nan)
        for cell, quality in row["qualities"].items():
            bx, by = cell
            if 0 <= bx < MAP_BINS and 0 <= by < MAP_BINS:
                grid[by, bx] = quality

        fig, ax = plt.subplots(figsize=(8, 6.5))
        mesh = ax.imshow(grid, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                         extent=[x_lo, x_hi, y_lo, y_hi], aspect="auto")
        fig.colorbar(mesh, ax=ax, label=MAP_ELITES_OBJECTIVE)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title("MAP-Elites archive - generation {}\n{} cells ({:.1f}%), "
                     "QD-score {:.1f}".format(row["gen"], row["filled"],
                                              row["coverage"],
                                              row.get("qd_offset", 0.0)))
        frame_path = plots_dir / "archive_gen_{:03d}.png".format(row["gen"])
        fig.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frames.append(frame_path)

    print("\nWrote {} heatmaps to {}".format(len(frames), plots_dir))

    if gif_path and frames:
        try:
            from PIL import Image
        except ImportError:
            print("GIF skipped (Pillow not installed).")
            return
        images = [Image.open(f) for f in frames]
        images[0].save(gif_path, save_all=True, append_images=images[1:],
                       optimize=False, duration=500, loop=0)
        print("Wrote animation to " + str(gif_path))


def main():
    parser = argparse.ArgumentParser(
        description="Summarize and plot MAP-Elites archives from run_improved.py checkpoints.")
    parser.add_argument("checkpoint_dir",
                        help="Island checkpoint directory holding checkpoint_gen_*.pkl")
    parser.add_argument("--global-dir", default=None,
                        help="Directory holding global_gen_*.pkl (for named metrics)")
    parser.add_argument("--stats-dir", default=None,
                        help="Directory of <gene>_stats.json, for robustness analysis")
    parser.add_argument("--qd-offset", type=float, default=None,
                        help="Fitness offset for the adjusted QD-score. Pass the "
                             "same value across every run you compare (the paper "
                             "uses the lowest fitness seen by any algorithm on the "
                             "task). Defaults to this run's lowest fitness.")
    parser.add_argument("--plots", default=None,
                        help="Directory to write per-generation heatmaps into")
    parser.add_argument("--gif", default=None,
                        help="Path to write an animated GIF of the archive filling up")
    args = parser.parse_args()

    rows, total_cells, _metrics = summarize(args.checkpoint_dir, args.global_dir)
    offset, offset_note = resolve_offset(rows, args.qd_offset)
    print_report(rows, total_cells, offset, offset_note)

    if args.stats_dir and rows:
        stats = load_stats(args.stats_dir)
        last = rows[-1]
        print_robustness(robustness_rows(last["archive"], stats), stats,
                         last["best_gene"])

    if args.plots or args.gif:
        render_plots(rows, args.plots or "map_elites_plots", args.gif)


if __name__ == "__main__":
    main()
