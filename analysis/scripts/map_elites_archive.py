# map_elites_archive.py
# Read-only analysis of MAP-Elites checkpoints produced by run_improved.py.
#
# Reports archive coverage, QD-score and the best elite per generation, and
# optionally renders the behaviour grid as a heatmap per generation plus an
# animated GIF of the archive filling up.
#
# Usage:
#   uv run python analysis/scripts/map_elites_archive.py <checkpoint_dir> \
#       [--global-dir DIR] [--plots OUTDIR] [--gif OUT.gif]

from pathlib import Path
import argparse
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
        finite = [q for q in qualities.values() if q == q]  # drops NaN
        best_cell = max(qualities, key=lambda c: qualities[c]) if finite else None
        rows.append({
            "gen": gen,
            "archive": archive,
            "qualities": qualities,
            "filled": len(archive),
            "coverage": len(archive) / total_cells * 100 if total_cells else 0.0,
            "qd_score": sum(finite),
            "best_cell": best_cell,
            "best_quality": qualities[best_cell] if best_cell else float("nan"),
            "best_gene": archive[best_cell][0] if best_cell else None,
        })
    return rows, total_cells


def print_report(rows, total_cells):
    descriptor_names = ", ".join(name for name, _lo, _hi in MAP_ELITES_DESCRIPTORS)
    print("Descriptors : " + descriptor_names)
    print("Grid        : {} bins per axis ({} cells)".format(MAP_BINS, total_cells))
    print("Objective   : " + MAP_ELITES_OBJECTIVE + "\n")

    if not rows:
        print("No MAP-Elites archive found in these checkpoints.")
        return

    header = "{:>5}  {:>7}  {:>9}  {:>12}  {:>12}  best gene"
    print(header.format("gen", "filled", "coverage", "QD-score", "best"))
    for row in rows:
        print("{:>5}  {:>7}  {:>8.1f}%  {:>12.2f}  {:>12.2f}  {}".format(
            row["gen"], row["filled"], row["coverage"],
            row["qd_score"], row["best_quality"], row["best_gene"]))

    last = rows[-1]
    print("\nFinal archive: {}/{} cells ({:.1f}% coverage), QD-score {:.2f}".format(
        last["filled"], total_cells, last["coverage"], last["qd_score"]))
    print("Best elite   : {} ({}={:.2f}) in cell {}".format(
        last["best_gene"], MAP_ELITES_OBJECTIVE,
        last["best_quality"], last["best_cell"]))

    # Generation in which each elite currently in the archive first appeared.
    first_seen = {}
    for row in rows:
        for ind in row["archive"].values():
            first_seen.setdefault(ind[0], row["gen"])
    print("\nBest elite first entered the archive in generation {}".format(
        first_seen.get(last["best_gene"], "?")))


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
        ax.set_title("MAP-Elites archive - generation {}\n{} cells ({:.1f}%), QD-score {:.1f}".format(
            row["gen"], row["filled"], row["coverage"], row["qd_score"]))
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
    parser.add_argument("--plots", default=None,
                        help="Directory to write per-generation heatmaps into")
    parser.add_argument("--gif", default=None,
                        help="Path to write an animated GIF of the archive filling up")
    args = parser.parse_args()

    rows, total_cells = summarize(args.checkpoint_dir, args.global_dir)
    print_report(rows, total_cells)
    if args.plots or args.gif:
        render_plots(rows, args.plots or "map_elites_plots", args.gif)


if __name__ == "__main__":
    main()
