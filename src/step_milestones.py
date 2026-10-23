"""Results snapshots at fixed step intervals, so runs compare at equal budgets.

Both drivers (run_improved.py, run_mesb.py) charge each successfully trained
gene the environment steps it trained (STEPS_PER_GENE in
src/cfg/run_options.py). Every time the total passes a multiple of
MILESTONE_TIMESTEPS (96M by default), and once more when the whole budget is
spent, the driver appends one row to ``milestones.csv`` in its run folder and
saves the archive as it stands then. A snapshot is taken at the end of the
generation that passes the mark, so ``steps_used`` can be slightly above it.
"""

import csv
import os

MILESTONE_COLUMNS = [
    "milestone_steps", "steps_used", "successful_genes", "generation", "cells", "total_cells",
    "coverage_pct", "qd_score", "best_reward", "mean_elite_reward", "archive_file",
]


def milestone_targets(budget_steps, interval):
    """Every multiple of ``interval`` up to the budget, then the budget itself."""
    if interval <= 0:
        return []
    targets = list(range(interval, budget_steps + 1, interval)) if budget_steps else []
    if budget_steps and (not targets or targets[-1] != budget_steps):
        targets.append(budget_steps)
    return targets


def recorded_milestones(path):
    """Milestones already in ``path`` (so a resumed run does not repeat them)."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {int(row["milestone_steps"]) for row in csv.DictReader(f)}


def due_milestones(steps_used, budget_steps, interval, path):
    """Milestones reached by ``steps_used`` that ``path`` does not record yet."""
    done = recorded_milestones(path)
    return [m for m in milestone_targets(budget_steps, interval) if m <= steps_used and m not in done]


def append_milestone(path, row):
    new_file = not os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MILESTONE_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in MILESTONE_COLUMNS})


def milestone_label(steps):
    """Short name for file names: 96000000 -> '0096M'."""
    return f"{steps // 1_000_000:04d}M"
