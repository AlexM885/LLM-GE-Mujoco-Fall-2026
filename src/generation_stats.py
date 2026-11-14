"""Per-generation health of an LLM-GE run, written to <checkpoints>/generation_stats.csv.

The leads asked every team (Nov 9) to track, each generation, how many
individuals train and evaluate successfully and how many enter the archive /
Pareto front. One row per generation, so control and experiment runs can be
compared generation by generation (e.g. a Welch's t-test per generation over
repeated runs).
"""

import csv
import os
import statistics
import time

COLUMNS = ["generation", "time", "offspring", "trained_ok", "duplicates", "failed", "timeouts",
           "archive_updates", "archive_cells", "archive_genes", "best", "median_archive", "qd_score",
           "successful_evals_total"]


def classify(entry, failed_fitness, objective, sentinel):
    """'duplicate', 'timeout', 'failed', 'trained_ok' or 'pending' for one GLOBAL_DATA entry."""
    status = str(entry.get("status", ""))
    if entry.get("duplicate_of"):
        return "duplicate"
    if "TIMEOUT" in status:
        return "timeout"
    metrics = entry.get("metrics") or {}
    if status == "completed" and metrics.get(objective, sentinel) > sentinel:
        return "trained_ok"
    if status == "completed" or status.startswith("FAILED") or entry.get("fitness") == failed_fitness:
        return "failed"
    return "pending"


def generation_row(generation, offspring_entries, archive_qualities, archive_cells, archive_updates,
                   successful_evals_total, failed_fitness, objective, sentinel, now=None):
    """One generation_stats.csv row (a dict keyed by COLUMNS)."""
    counts = {"trained_ok": 0, "duplicate": 0, "failed": 0, "timeout": 0, "pending": 0}
    for entry in offspring_entries:
        counts[classify(entry, failed_fitness, objective, sentinel)] += 1
    q = [x for x in archive_qualities if x is not None]
    return {
        "generation": generation,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
        "offspring": len(offspring_entries),
        "trained_ok": counts["trained_ok"],
        "duplicates": counts["duplicate"],
        "failed": counts["failed"] + counts["pending"],
        "timeouts": counts["timeout"],
        "archive_updates": archive_updates,
        "archive_cells": archive_cells,
        "archive_genes": len(q),
        "best": round(max(q), 2) if q else "",
        "median_archive": round(statistics.median(q), 2) if q else "",
        "qd_score": round(sum(q), 2) if q else 0,
        "successful_evals_total": successful_evals_total,
    }


def append_row(folder, row):
    """Append row to <folder>/generation_stats.csv (header on first write); returns the path."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "generation_stats.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    return path
