"""Seed protocol for uncertain-QD runs.

Every random draw that matters for a comparison comes from one of these
streams, and each stream owns a disjoint range of integer seeds, so a seed can
never mean two things at once. All values stay below 2**31 so they are valid
for Gymnasium, NumPy and PyTorch.

    stream            range                         used for
    ----------------  ----------------------------  ----------------------------------
    (legacy)          [0, 10_000_000)               MUJOCO_EVAL_SEED and seed+0..9
    train             [10_000_000, 100_000_000)     TD3 training seeds (weights, noise,
                                                    replay sampling, first env reset)
    search_rollout    [100_000_000, 1_100_000_000)  episodes whose results drive
                                                    insertion, replacement or N
    aria              [1_100_000_000, 1_500_000_000) ARIA perturbation rollouts
    final_holdout     [1_500_000_000, 1_900_000_000) held-out final evaluation only
    final_train       [1_900_000_000, 2_000_000_000) held-out retraining (final eval)

The driver's own randomness (parent selection, crossover and mutation coin
flips, LLM temperature draws, gene ids, Deep-Grid replacement) is seeded from
``derive_int(run_seed, "search", namespace)`` where the namespace is the preset
name, so concurrent runs of different presets never draw the same gene ids
(gene files share one directory). Training and rollout seeds depend on the
run seed only, so replicate r of every preset trains with the same seeds. LLM token sampling happens on the
inference server and is not reproducible, so a whole search can never be
replayed bit for bit; evaluations of a given checkpoint can.

Rollout seeds are allocated in per-solution blocks: solution ``s`` draws a
block start from the stream, and its k-th search episode uses ``start + k``.
The initial evaluation consumes k = 0..N-1 and every later re-evaluation
continues from where the last one stopped, so no episode seed is ever reused
for the same checkpoint. Blocks of different solutions may overlap; that only
means two policies happen to see the same start state, which is harmless.

Held-out seeds do not depend on the run seed: every configuration and every
replicate is scored on the same list of starts, which removes reset noise from
cross-configuration comparisons of the final archive.
"""

import hashlib

LEGACY_RANGE = (0, 10_000_000)
STREAM_RANGES = {
    "train": (10_000_000, 100_000_000),
    "search_rollout": (100_000_000, 1_100_000_000),
    "aria": (1_100_000_000, 1_500_000_000),
    "final_holdout": (1_500_000_000, 1_900_000_000),
    "final_train": (1_900_000_000, 2_000_000_000),
}
#: Seeds available to one solution in a rollout stream.
BLOCK_SIZE = 1 << 16


def _digest(*parts):
    text = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def derive_int(*parts, bound=2**31 - 1):
    """Deterministic integer in [0, bound) from any sequence of labels."""
    return _digest(*parts) % bound


def stream_of(seed):
    """Name of the stream a seed belongs to ('legacy' below the train range)."""
    if LEGACY_RANGE[0] <= seed < LEGACY_RANGE[1]:
        return "legacy"
    for name, (lo, hi) in STREAM_RANGES.items():
        if lo <= seed < hi:
            return name
    raise ValueError(f"seed {seed} is outside every stream range")


class SeedScheme:
    """Seed allocation for one run (one ``run_seed``)."""

    def __init__(self, run_seed, final_master_seed=20261012, search_namespace=""):
        self.run_seed = int(run_seed)
        self.final_master_seed = int(final_master_seed)
        self.search_namespace = str(search_namespace)

    def search_seed(self):
        return derive_int(self.run_seed, "search", self.search_namespace)

    def train_seed(self, replicate):
        """Training seed for replicate ``replicate`` of any gene in this run.

        Shared across genes (as the legacy pipeline shares MUJOCO_EVAL_SEED),
        so replicate r of two genes starts from the same RNG state.
        """
        lo, hi = STREAM_RANGES["train"]
        return lo + derive_int(self.run_seed, "train", int(replicate), bound=hi - lo)

    def _block_start(self, stream, *labels):
        lo, hi = STREAM_RANGES[stream]
        n_blocks = (hi - lo) // BLOCK_SIZE
        return lo + derive_int(*labels, bound=n_blocks) * BLOCK_SIZE

    def rollout_block(self, solution_id):
        """First seed of ``solution_id``'s search-rollout block."""
        return self._block_start("search_rollout", self.run_seed, "rollout", solution_id)

    def rollout_seeds(self, solution_id, start_index, count):
        """Seeds for search episodes start_index .. start_index+count-1."""
        if start_index + count > BLOCK_SIZE:
            raise ValueError(f"{solution_id} exhausted its {BLOCK_SIZE}-episode seed block")
        base = self.rollout_block(solution_id)
        return [base + start_index + k for k in range(count)]

    def aria_seeds(self, label, step, count):
        """Rollout seeds for one ARIA gradient step of target ``label``."""
        base = self._block_start("aria", self.run_seed, "aria", label, step)
        if count > BLOCK_SIZE:
            raise ValueError("ARIA step needs more seeds than one block holds")
        return [base + k for k in range(count)]

    def holdout_seeds(self, count):
        """Held-out final-evaluation reset seeds, common to every run."""
        base = self._block_start("final_holdout", self.final_master_seed, "holdout")
        if count > BLOCK_SIZE:
            raise ValueError("final evaluation asks for more seeds than one block holds")
        return [base + k for k in range(count)]

    def final_train_seeds(self, count):
        lo, hi = STREAM_RANGES["final_train"]
        return [lo + derive_int(self.final_master_seed, "final_train", r, bound=hi - lo)
                for r in range(count)]

    def describe(self):
        return {
            "run_seed": self.run_seed,
            "search_seed": self.search_seed(),
            "search_namespace": self.search_namespace,
            "final_master_seed": self.final_master_seed,
            "stream_ranges": {k: list(v) for k, v in STREAM_RANGES.items()},
            "legacy_range": list(LEGACY_RANGE),
            "rollout_block_size": BLOCK_SIZE,
        }


def check_disjoint(search_seeds, holdout_seeds):
    """Raise if any held-out seed was also used to make a search decision."""
    overlap = set(search_seeds) & set(holdout_seeds)
    if overlap:
        raise ValueError(f"{len(overlap)} held-out seeds were used during search, e.g. {sorted(overlap)[:5]}")
