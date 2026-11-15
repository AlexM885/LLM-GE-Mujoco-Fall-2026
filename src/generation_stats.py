"""Per-generation health of an LLM-GE run, written to <checkpoints>/generation_stats.csv.

The leads asked every team (Nov 9) to track, each generation, how many
individuals train and evaluate successfully and how many enter the archive /
Pareto front. One row per generation, so control and experiment runs can be
compared generation by generation (e.g. a Welch's t-test per generation over
repeated runs).

The timing columns (minutes) show where each generation's time goes, as Dr. Zutty
suggested timing each part of the loop:
  llm_min           crossover + mutation LLM jobs (generation 0: creating the population)
  crossover_min, mutation_min   the two LLM phases separately (blank for generation 0)
  genes_min         first gene submitted -> last gene done (the whole training phase)
  gene_queue_*_min  per gene: submitted -> job started (waiting in the SLURM queue)
  gene_train_*_min  per gene: job started -> result read (training + final evaluation)
  vlm_min           the VLM writing the next generation's aux reward (vlm tier)
  generation_min    whole generation, start to end
Queue/train times only count genes that trained this generation (not duplicates or
timeouts) and are accurate to the controller's polling interval (~1 min).
"""

import csv
import os
import statistics
import time

COLUMNS = ["generation", "time", "offspring", "trained_ok", "duplicates", "failed", "timeouts",
           "archive_updates", "archive_cells", "archive_genes", "best", "median_archive", "qd_score",
           "successful_evals_total"]
TIMING_COLUMNS = ["llm_min", "crossover_min", "mutation_min", "genes_min",
                  "gene_queue_median_min", "gene_queue_max_min",
                  "gene_train_median_min", "gene_train_max_min", "vlm_min", "generation_min"]
COLUMNS = COLUMNS + TIMING_COLUMNS


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


def _minutes(seconds):
    return round(seconds / 60.0, 1)


def _span(phases, start, end):
    if phases.get(start) is None or phases.get(end) is None:
        return ""
    return _minutes(phases[end] - phases[start])


def timing_columns(phases, gene_entries, since):
    """Timing columns (TIMING_COLUMNS) for one generation.

    phases: time.time() marks the controller took during the generation, any of
      start, crossover_done, mutation_done (or llm_done for generation 0),
      genes_done, vlm_start, vlm_done, end.
    gene_entries: the generation's GLOBAL_DATA records. A gene counts if it was
      submitted at or after `since` (older entries are parents carried over unchanged)
      and is not a duplicate; queue time needs 'submit_time' and 'eval_start_time',
      train time also 'finish_time'.
    """
    llm_end = "mutation_done" if phases.get("mutation_done") is not None else "llm_done"
    queue, train, submits, finishes = [], [], [], []
    for e in gene_entries:
        submitted, started, finished = e.get("submit_time"), e.get("eval_start_time"), e.get("finish_time")
        if e.get("duplicate_of") or submitted is None or submitted < since:
            continue
        submits.append(submitted)
        if started is not None:
            queue.append(max(0.0, started - submitted))
            if finished is not None:
                train.append(max(0.0, finished - started))
                finishes.append(finished)
    genes_end = phases.get("genes_done")
    return {
        "llm_min": _span(phases, "start", llm_end),
        "crossover_min": _span(phases, "start", "crossover_done"),
        "mutation_min": _span(phases, "crossover_done", "mutation_done"),
        "genes_min": _minutes(genes_end - min(submits)) if submits and genes_end is not None else "",
        "gene_queue_median_min": _minutes(statistics.median(queue)) if queue else "",
        "gene_queue_max_min": _minutes(max(queue)) if queue else "",
        "gene_train_median_min": _minutes(statistics.median(train)) if train else "",
        "gene_train_max_min": _minutes(max(train)) if train else "",
        "vlm_min": _span(phases, "vlm_start", "vlm_done"),
        "generation_min": _span(phases, "start", "end"),
    }


def append_row(folder, row):
    """Append row to <folder>/generation_stats.csv (header on first write); returns the path."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "generation_stats.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        if new:
            writer.writeheader()
        writer.writerow(row)
    return path
