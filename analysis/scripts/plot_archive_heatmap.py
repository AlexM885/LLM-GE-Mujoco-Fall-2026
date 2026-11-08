"""Heatmap of a run_improved.py MAP-Elites archive: foot-contact grid coloured by return.

Reads the newest checkpoint_gen_<g>.pkl of a run (or the one given) and each
elite's results CSV, and writes a PNG of the 32 x 32 grid (foot_contact_0 x
foot_contact_1), empty cells left blank, the best elite ringed.

    uv run python analysis/scripts/plot_archive_heatmap.py --run walker_xqc_emitters2
    uv run python analysis/scripts/plot_archive_heatmap.py --run walker_xqc_emitters2 --gen 2 --zoom

--zoom crops to the occupied region (the full grid is mostly empty early on).
"""

import argparse
import glob
import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator
from matplotlib.colors import LinearSegmentedColormap

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Sequential blue ramp, steps 100 -> 700 (light = low return, dark = high).
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load_archive(checkpoint):
    # The checkpoint pickles DEAP individuals; recreate their classes to load it.
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti, file_id=None)
    with open(checkpoint, "rb") as f:
        archive = pickle.load(f).get("MAP_ELITES_ARCHIVE", {})
    # Pareto-front-per-cell runs keep a list per cell: plot each cell's best.
    return {cell: (v if isinstance(v, list) and v and isinstance(v[0], list) else [v])
            for cell, v in archive.items()}


def read_results(gene_id, results_dirs):
    for d in results_dirs:
        path = os.path.join(d, f"{gene_id}_results.csv")
        if os.path.exists(path):
            lines = [l.strip() for l in open(path) if l.strip()]
            if len(lines) > 1:
                return dict(zip(lines[0].split(","), (float(v) for v in lines[-1].split(","))))
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="run folder, e.g. walker_xqc_emitters2")
    p.add_argument("--island", default=None, help="island folder (default: the only one in the run)")
    p.add_argument("--gen", type=int, default=None, help="checkpoint generation (default: newest)")
    p.add_argument("--bins", type=int, default=32)
    p.add_argument("--zoom", action="store_true", help="crop to the occupied cells")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    run = a.run if os.path.isabs(a.run) else os.path.join(REPO, a.run)
    islands = [a.island] if a.island else [os.path.basename(d) for d in glob.glob(os.path.join(run, "island_*"))]
    if len(islands) != 1:
        raise SystemExit(f"pick one island with --island: {islands}")
    folder = os.path.join(run, islands[0])
    ckpts = {int(re.search(r"checkpoint_gen_(\d+)\.pkl$", f).group(1)): f
             for f in glob.glob(os.path.join(folder, "checkpoint_gen_*.pkl"))}
    gen = a.gen if a.gen is not None else max(ckpts)
    archive = load_archive(ckpts[gen])
    results_dirs = [os.path.join(REPO, "sota", "XQC", "llmge", "results"),
                    os.path.join(REPO, "sota", "MujocoRL", "results")]

    grid = np.full((a.bins, a.bins), np.nan)
    best = None
    for (i, j), members in archive.items():
        rewards = [r["mean_reward"] for r in (read_results(m[0], results_dirs) for m in members) if r]
        if not rewards:
            continue
        grid[j, i] = max(rewards)   # row = foot 1, column = foot 0
        if best is None or grid[j, i] > best[0]:
            best = (grid[j, i], i, j, members[int(np.argmax(rewards))][0])
    filled = int(np.isfinite(grid).sum())
    print(f"generation {gen}: {filled} cells, QD score {np.nansum(grid):,.0f}, "
          f"best {best[0]:,.0f} in cell ({best[1]},{best[2]}) gene {best[3]}")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "text.color": TEXT,
                         "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2})
    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    cmap = LinearSegmentedColormap.from_list("ramp", RAMP)
    cmap.set_bad(SURFACE)
    edges = np.linspace(0, 1, a.bins + 1)
    mesh = ax.pcolormesh(edges, edges, np.ma.masked_invalid(grid), cmap=cmap,
                         edgecolors=SURFACE, linewidth=1.5)   # surface gap between cells
    cell = 1 / a.bins
    bx, by = (best[1] + 0.5) * cell, (best[2] + 0.5) * cell
    # Ring with a white halo, and a boxed label, so both read on dark and light cells.
    ax.scatter([bx], [by], s=300, facecolors="none", edgecolors=SURFACE, linewidths=5, zorder=3)
    ax.scatter([bx], [by], s=300, facecolors="none", edgecolors=TEXT, linewidths=2, zorder=4)
    ax.annotate(f"best {best[0]:,.0f}", (bx, by), xytext=(-18, 30), textcoords="offset points",
                ha="right", color=TEXT, fontsize=12, fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE, edgecolor=GRID),
                arrowprops=dict(arrowstyle="-", color=TEXT, linewidth=1.2))
    if a.zoom:
        occ = np.argwhere(np.isfinite(grid))
        (r0, c0), (r1, c1) = occ.min(0), occ.max(0)
        ax.set_xlim(max(c0 - 2, 0) * cell, min(c1 + 3, a.bins) * cell)
        ax.set_ylim(max(r0 - 2, 0) * cell, min(r1 + 3, a.bins) * cell)
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_xlabel("Foot 0 contact (share of steps on the ground)")
    ax.set_ylabel("Foot 1 contact (share of steps on the ground)")
    ax.set_title(f"Archive after generation {gen}: {filled} of {a.bins * a.bins} cells",
                 loc="left", fontsize=14, fontweight="bold", pad=12)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Mean return of the cell's elite")
    cbar.outline.set_visible(False)
    fig.tight_layout()
    out = a.out or os.path.join(folder, f"archive_heatmap_gen_{gen}{'_zoom' if a.zoom else ''}.png")
    fig.savefig(out, dpi=200)
    print("written", out)


if __name__ == "__main__":
    main()
