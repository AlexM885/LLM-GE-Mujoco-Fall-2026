"""
Put LIMEN and PGA-MAP-Elites on the LLM-GE playing field at a common budget.

Both are cut off at the same number of training timesteps and their solutions
are placed on the LLM-GE MAP-Elites grid (MAP_BINS x MAP_BINS over the two
foot-contact proportions), so that best fitness, coverage and QD-score mean
the same thing as the numbers run_improved.py prints for LLM-GE.

    python sota/MujocoRL/Baseline/compare_baselines.py \
        --limen baseline_runs/limen/walker2d \
        --pga baseline_runs/pga_map_elites/seed0_job6202104 \
        --timesteps 26000000
"""

import argparse
import glob
import json
import os

MAP_BINS = 32            # src/cfg/constants_Mujoco.py
STEPS_PER_TRAINING = 500000


def to_bin(foot_contact_0, foot_contact_1):
    """Same binning as get_bin() in run_improved.py for descriptors in [0, 1]."""
    return tuple(min(int(min(max(x, 0.0), 1.0) * MAP_BINS), MAP_BINS - 1)
                 for x in (foot_contact_0, foot_contact_1))


def summarise(name, budget, solutions, note):
    """solutions: list of (fitness, foot_contact_0, foot_contact_1)."""
    grid = {}
    for fitness, fc0, fc1 in solutions:
        cell = to_bin(fc0, fc1)
        if cell not in grid or fitness > grid[cell]:
            grid[cell] = fitness
    cells = MAP_BINS * MAP_BINS
    best = max(grid.values()) if grid else float("nan")
    print(f"{name:<16} {budget:>12,} {len(solutions):>10} {best:>10.1f} "
          f"{len(grid):>5}/{cells} ({100 * len(grid) / cells:4.1f}%) {sum(grid.values()):>12.1f}  {note}")


def limen(run_dir, timesteps):
    """The first trainings of a LIMEN run, in the order they finished."""
    programs = []
    for path in glob.glob(os.path.join(run_dir, "database", "programs", "*.json")):
        with open(path) as f:
            p = json.load(f)
        m = p.get("metrics") or {}
        if p.get("stage") == "full_train" and "final_return" in m:
            programs.append((p["timestamp"], m["final_return"], m["foot_contact_0"], m["foot_contact_1"]))
    programs.sort()
    kept = programs[: timesteps // STEPS_PER_TRAINING]
    summarise("LIMEN", len(kept) * STEPS_PER_TRAINING, [p[1:] for p in kept],
              f"first {len(kept)} of {len(programs)} trainings; fitness = mean of 10 episodes")


def pga(run_dir, timesteps):
    """The PGA-MAP-Elites archive snapshot closest to the budget."""
    progress = glob.glob(os.path.join(run_dir, "progress_*.dat"))[0]
    steps_at = {}
    with open(progress) as f:
        for line in f:
            cols = line.split()
            steps_at[int(cols[0])] = int(cols[-1])   # evaluations -> timesteps

    def snapshot_steps(path):
        evals = int(path.rsplit("_", 1)[1].split(".")[0])
        return steps_at.get(evals, steps_at[min(steps_at, key=lambda e: abs(e - evals))])

    snapshots = glob.glob(os.path.join(run_dir, "archive_*.dat"))
    path = min(snapshots, key=lambda p: abs(snapshot_steps(p) - timesteps))
    solutions = []
    with open(path) as f:
        for line in f:
            cols = line.split()   # fitness, centroid (2), descriptor (2), id
            solutions.append((float(cols[0]), float(cols[3]), float(cols[4])))
    summarise("PGA-MAP-Elites", snapshot_steps(path), solutions,
              f"{os.path.basename(path)}; fitness = one episode")


def pga_final(run_dir):
    """The final PGA-MAP-Elites archive, re-evaluated over 10 episodes (results_*.csv)."""
    path = glob.glob(os.path.join(run_dir, "results_*.csv"))[0]
    with open(path) as f:
        header = f.readline().strip().split(",")
        col = {name: header.index(name) for name in ("mean_reward", "foot_contact_0", "foot_contact_1")}
        solutions = []
        for line in f:
            cols = line.strip().split(",")
            solutions.append(tuple(float(cols[col[name]]) for name in col))
    progress = glob.glob(os.path.join(run_dir, "progress_*.dat"))[0]
    with open(progress) as f:
        steps = int(f.readlines()[-1].split()[-1])
    summarise("PGA-ME (final)", steps, solutions, "whole run; fitness = mean of the final evaluation episodes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limen", help="LIMEN run directory")
    parser.add_argument("--pga", help="PGA-MAP-Elites run directory")
    parser.add_argument("--timesteps", type=int, default=26000000, help="Common budget of training timesteps")
    args = parser.parse_args()

    print(f"{'method':<16} {'timesteps':>12} {'solutions':>10} {'best':>10} {'coverage':>17} {'QD-score':>12}")
    if args.limen:
        limen(args.limen, args.timesteps)
    if args.pga:
        pga(args.pga, args.timesteps)
        if glob.glob(os.path.join(args.pga, "results_*.csv")):
            pga_final(args.pga)
