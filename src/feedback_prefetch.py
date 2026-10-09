"""List parent gene ids from a MAP-Elites checkpoint, to pre-populate feedback.

The observer batch job (``observer.sh``) needs the parents that evolution may
select. This helper reads the latest checkpoint's archive and prints those gene
ids, so a run can seed the feedback cache before it starts::

    GENES=$(uv run python src/feedback_prefetch.py --checkpoints <dir>)
    sbatch observer.sh "$GENES" visual

Importing the pickled deap individuals requires the same creator classes that
``run_improved.py`` defines, so they are recreated here without importing the
heavy evolution module.
"""

import argparse
import os
import pickle
import re
import sys

_CHECKPOINT_RE = re.compile(r"^checkpoint_gen_(\d+)\.pkl$")


def _ensure_deap_classes():
    from deap import base, creator
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)


def latest_checkpoint(folder):
    candidates = [f for f in os.listdir(folder) if _CHECKPOINT_RE.match(f)]
    if not candidates:
        return None
    candidates.sort(key=lambda f: int(_CHECKPOINT_RE.match(f).group(1)))
    return os.path.join(folder, candidates[-1])


def _fitness_value(ind):
    """Best-effort first-objective fitness for checkpoint individuals."""
    try:
        values = ind.fitness.values
        if values:
            return float(values[0])
    except Exception:
        pass
    for attr in ("fitness", "mean_reward", "reward"):
        value = getattr(ind, attr, None)
        if isinstance(value, (int, float)):
            return float(value)
    return float("-inf")


def _archive_items(data):
    archive = data.get("MAP_ELITES_ARCHIVE") or {}
    if isinstance(archive, dict):
        return list(archive.items())
    return list(enumerate(archive))


def gene_ids_from_checkpoint(path, strategy="archive"):
    _ensure_deap_classes()
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    archive_items = _archive_items(data)
    if archive_items:
        if strategy == "top-fitness":
            archive_items.sort(key=lambda item: _fitness_value(item[1]), reverse=True)
        elif strategy == "low-fitness":
            archive_items.sort(key=lambda item: _fitness_value(item[1]))
        elif strategy == "diverse":
            # Stable interleaving across the archive's cell order. This keeps
            # behavior-space spread without requiring descriptor internals.
            evens = archive_items[::2]
            odds = archive_items[1::2]
            archive_items = evens + odds
        genes = [ind[0] for _, ind in archive_items]
        return genes
    # Fall back to the live population when the archive is empty (non-ME runs).
    population = data.get("population") or []
    if strategy == "top-fitness":
        population = sorted(population, key=_fitness_value, reverse=True)
    elif strategy == "low-fitness":
        population = sorted(population, key=_fitness_value)
    return [ind[0] for ind in population if hasattr(ind, "__getitem__")]


def main():
    parser = argparse.ArgumentParser(description="List archive parent gene ids")
    parser.add_argument("--checkpoints", required=True,
                        help="Checkpoint directory used by run_improved.py")
    parser.add_argument("--gene-file", default=None,
                        help="Also write one gene id per line to this file")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optionally cap the number of genes (0 = all)")
    parser.add_argument("--checkpoint-file", default=None,
                        help="Explicit checkpoint file instead of the latest")
    parser.add_argument("--strategy", default="archive",
                        choices=("archive", "top-fitness", "low-fitness", "diverse"),
                        help="Parent ordering before applying --limit")
    args = parser.parse_args()

    path = args.checkpoint_file or latest_checkpoint(args.checkpoints)
    if not path or not os.path.isfile(path):
        raise SystemExit(f"no checkpoint found in {args.checkpoints}")
    genes = gene_ids_from_checkpoint(path, strategy=args.strategy)
    if args.limit > 0:
        genes = genes[:args.limit]
    print(f"# parent genes from {path} ({len(genes)}, strategy={args.strategy})",
          file=sys.stderr, flush=True)
    for gene in genes:
        print(gene)
    if args.gene_file:
        with open(args.gene_file, "w", encoding="utf-8") as handle:
            handle.write("\n".join(genes) + ("\n" if genes else ""))
        print(f"# wrote {len(genes)} genes to {args.gene_file}", file=sys.stderr,
              flush=True)


if __name__ == "__main__":
    main()
