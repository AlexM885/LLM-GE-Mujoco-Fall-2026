"""Hooks for the matched MESB vs fixed-grid runs of Shazeb's XQC island.

Both arms run ``run_improved_matched.py``, a generated copy of his
``run_improved.py`` (see ``scripts/mesb/matched/build_matched_driver.py``) that
calls into this module at five marked places. Everything else - LLM operators,
prompts, gene validation, evaluation jobs, parent selection, checkpoints - is
his code. Options come from the environment so the generated driver keeps his
command line:

``MATCHED_ARCHIVE``        ``fixed`` (his ``get_bin``: equal bins over his fixed
                           ranges) or ``sliding`` (MAP-Elites with Sliding
                           Boundaries). Required.
``MATCHED_FIX_CROSSOVER``  ``1`` (default): crossover jobs run
                           ``src/llm_crossover_fixed.py`` (same CLI, chunk-index
                           fix); ``0``: his ``src/llm_crossover.py``.
``MATCHED_GENE_TIME``      Slurm time limit of a gene evaluation job, replacing
                           the ``-t 1:30:00`` in slurm_config.yaml (default
                           ``3:00:00``; ``keep`` leaves it unchanged).
``MATCHED_STOP_AFTER_GEN0``  ``1``: create and evaluate generation 0, save it as
                           ``checkpoint_gen_0.pkl`` / ``global_gen_0.pkl`` and
                           exit, so both arms can resume from the same genes.

Sliding archive (``MATCHED_ARCHIVE=sliding``)
    Every gene that his ``update_archive`` would accept (valid fitness, not the
    invalid / placeholder / failed sentinel, finite quality, both descriptors
    present) is added to a buffer, once per gene id. After each call - i.e.
    once per generation, plus the seeding call - each descriptor's bin edges
    are recomputed at percentiles of the buffer (Fontaine et al. 2019; same
    rule as sota/MujocoRL/mesb_archive.compute_boundaries_1d) and the archive
    is rebuilt from the buffer: every buffered gene goes to its cell, the
    highest quality wins, ties keep the earlier gene (his strict ``>``). The
    archive keeps his format (dict: cell tuple -> individual), so his
    selection, checkpoints, resume and xqc_island_report.py work unchanged.
    The paper remaps every 100 additions; with ~10 valid genes per generation
    that would never trigger in a 6-generation run, hence once per generation.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Callable

import numpy as np

from sota.MujocoRL.mesb_archive import compute_boundaries_1d

ARCHIVE_KINDS = ("fixed", "sliding")


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


ARCHIVE = os.environ.get("MATCHED_ARCHIVE", "").strip().lower()
SLIDING = ARCHIVE == "sliding"
FIX_CROSSOVER = _env_flag("MATCHED_FIX_CROSSOVER", "1")
GENE_TIME = os.environ.get("MATCHED_GENE_TIME", "3:00:00").strip()
STOP_AFTER_GEN0 = _env_flag("MATCHED_STOP_AFTER_GEN0", "0")
ORIGINAL_GENE_TIME_LINE = "#SBATCH -t 1:30:00"
CROSSOVER_SCRIPT = "src/llm_crossover.py"
FIXED_CROSSOVER_SCRIPT = "src/llm_crossover_fixed.py"


def check_options() -> None:
    """Fail fast on a missing or unknown archive kind (called at driver start)."""
    if ARCHIVE not in ARCHIVE_KINDS:
        raise SystemExit(f"MATCHED_ARCHIVE must be one of {ARCHIVE_KINDS}, got {ARCHIVE!r}")
    print(f"[matched] archive={ARCHIVE} fix_crossover={FIX_CROSSOVER} gene_time={GENE_TIME} "
          f"stop_after_gen0={STOP_AFTER_GEN0}", flush=True)


# ------------------------------------------------------------------ hooks 1-2
def adjust_slurm_config(config: dict) -> dict:
    """Hook in ``load_yaml``: raise the gene-evaluation job's time limit."""
    if GENE_TIME.lower() == "keep":
        return config
    script = config.get("python_bash_script", "")
    if ORIGINAL_GENE_TIME_LINE not in script:
        raise RuntimeError(f"slurm_config.yaml python_bash_script no longer contains "
                           f"{ORIGINAL_GENE_TIME_LINE!r}; update mesb_matched.py")
    config = dict(config)
    config["python_bash_script"] = script.replace(ORIGINAL_GENE_TIME_LINE, f"#SBATCH -t {GENE_TIME}")
    return config


def crossover_script(python_file: str) -> str:
    """Hook in ``write_bash_script``: which crossover script the LLM job runs."""
    if FIX_CROSSOVER and python_file == CROSSOVER_SCRIPT:
        return FIXED_CROSSOVER_SCRIPT
    return python_file


# ------------------------------------------------------------------ sliding archive
class SlidingState:
    """Buffer + current edges; pickled into the checkpoint as ``MESB_STATE``."""

    def __init__(self, bins: int, descriptors: list[tuple[str, float, float]]) -> None:
        self.bins = int(bins)
        self.names = [d[0] for d in descriptors]
        self.buffer: list[dict[str, Any]] = []   # gene_id, measures, quality, ind
        self.edges: list[list[float]] | None = None
        self.remaps = 0
        self.history: list[dict[str, Any]] = []

    # -- persistence
    def state_dict(self) -> dict[str, Any]:
        return {"bins": self.bins, "names": self.names, "buffer": self.buffer,
                "edges": self.edges, "remaps": self.remaps, "history": self.history}

    def load(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        self.bins, self.names = state["bins"], state["names"]
        self.buffer, self.edges = state["buffer"], state["edges"]
        self.remaps, self.history = state.get("remaps", 0), state.get("history", [])

    # -- placement
    def cell(self, measures: tuple[float, ...]) -> tuple[int, ...]:
        out = []
        for value, edges in zip(measures, self.edges):
            internal = np.asarray(edges[1:self.bins], dtype=float)
            idx = int(np.searchsorted(internal, value, side="right"))
            out.append(min(max(idx, 0), self.bins - 1))
        return tuple(out)

    def update(self, population, old_archive: dict, *, is_valid: Callable, metrics: Callable,
               quality: Callable) -> tuple[dict, int, int]:
        """Add new valid genes, recompute edges, rebuild the archive."""
        known = {e["gene_id"] for e in self.buffer}
        for ind in population:
            gene_id = ind[0]
            if gene_id in known or not is_valid(ind):
                continue
            m = metrics(gene_id) or {}
            if any(n not in m for n in self.names):
                continue
            measures = tuple(float(m[n]) for n in self.names)
            q = float(quality(ind))
            if not all(math.isfinite(v) for v in measures + (q,)):
                continue
            self.buffer.append({"gene_id": gene_id, "measures": measures, "quality": q, "ind": ind})
            known.add(gene_id)
        if not self.buffer:
            return {}, 0, 0
        self.edges = [compute_boundaries_1d([e["measures"][k] for e in self.buffer], self.bins).tolist()
                      for k in range(len(self.names))]
        self.remaps += 1
        archive: dict = {}
        best: dict = {}
        for e in self.buffer:
            c = self.cell(e["measures"])
            if c not in best or e["quality"] > best[c]["quality"]:
                best[c] = e
        for c, e in best.items():
            archive[c] = e["ind"]
        old_ids = {ind[0] for ind in old_archive.values()}
        new_ids = {ind[0] for ind in archive.values()}
        added, replaced = len(new_ids - old_ids), len(old_ids - new_ids)
        self.history.append({"remap": self.remaps, "buffer": len(self.buffer),
                             "cells": len(archive), "edges": self.edges})
        return archive, added, replaced

    def describe(self) -> str:
        parts = []
        for name, edges in zip(self.names, self.edges or []):
            inner = ", ".join(f"{v:.4g}" for v in edges[1:self.bins])
            parts.append(f"{name} [{edges[0]:.4g} | {inner} | {edges[-1]:.4g}]")
        return f"remap {self.remaps}, buffer {len(self.buffer)}: " + "; ".join(parts)


def options() -> dict[str, Any]:
    """What this process runs as; a generation-0 snapshot is arm-neutral."""
    return {"archive": "gen0" if STOP_AFTER_GEN0 else ARCHIVE,
            "fix_crossover": FIX_CROSSOVER, "gene_time": GENE_TIME}


def check_resume(stored: dict[str, Any] | None) -> None:
    """Hook after loading a checkpoint: refuse to continue an arm as the other arm."""
    if not stored:
        return
    mine = options()
    if stored.get("archive") not in ("gen0", mine["archive"]):
        raise SystemExit(f"[matched] checkpoint was written by the {stored.get('archive')!r} arm; "
                         f"this process is the {mine['archive']!r} arm")
    for key in ("fix_crossover", "gene_time"):
        if stored.get(key) != mine[key]:
            raise SystemExit(f"[matched] checkpoint has {key}={stored.get(key)!r}, "
                             f"this process has {mine[key]!r}")


def checkpoint_extra(state: SlidingState | None) -> dict[str, Any]:
    """Hook in ``save_checkpoint``: extra keys his loader ignores."""
    out: dict[str, Any] = {"MATCHED_OPTIONS": options()}
    if SLIDING and state is not None:
        out["MESB_STATE"] = state.state_dict()
    return out


def write_boundaries_log(state: SlidingState, path: str, generation: int) -> None:
    """Append the current edges to a JSONL log next to the checkpoints."""
    if state.edges is None:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"generation": generation, "remaps": state.remaps,
                             "buffer": len(state.buffer), "descriptors": state.names,
                             "edges": state.edges}) + "\n")
