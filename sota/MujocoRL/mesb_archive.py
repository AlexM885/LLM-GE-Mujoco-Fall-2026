"""MAP-Elites with Sliding Boundaries (MESB).

Implements the archive from

    Fontaine, Lee, Soros, de Mesentier Silva, Togelius, Hoover (2019).
    "Mapping Hearthstone Deck Spaces through MAP-Elites with Sliding
    Boundaries." GECCO '19.

Standard MAP-Elites divides every behaviour dimension into equal-width bins
between a manually chosen ``[lo, hi]``. If that range is guessed badly, most
cells are unreachable. MESB keeps the *number* of cells per dimension but
places the cell boundaries at empirical percentiles of the behaviour values
that have actually been observed, and periodically recomputes ("slides")
them as more solutions are evaluated.

    Sliding Boundaries makes numeric archive boundary positions data-driven.
    It does not eliminate the archive resolution choice.

This module is deliberately independent of Slurm, the LLM operators and
PPO: it operates purely on :class:`ArchiveRecord` values and depends only on
the Python standard library and NumPy.

Terminology (paper -> this module)
----------------------------------
* ``xi`` (buffer size)          -> ``buffer_capacity``; ``None`` means infinity.
* ``delta`` (remap frequency)   -> ``remap_frequency`` (successful additions).
* behaviour characteristic      -> ``measures``.
* fitness                       -> ``objective`` (maximised).

Boundary rule (exact indexing convention)
-----------------------------------------
For one behaviour dimension with ``B`` bins, let ``v[0] <= ... <= v[N-1]``
be that dimension's buffered values sorted ascending (``N >= 1``). The cell
edges are::

    edge[0] = v[0]                                   (observed minimum)
    edge[j] = v[ min(floor(j * N / B), N - 1) ]      for j = 1 .. B-1
    edge[B] = v[N-1]                                 (observed maximum)

Every edge is an observed value; nothing is interpolated or jittered.
Duplicate edges are allowed (e.g. ``N < B`` or many tied values); the cells
between two equal edges are simply unreachable until the data spreads out.
This is the same convention as the pyribs ``SlidingBoundariesArchive``.

Cell indexing
-------------
Cell ``j`` is the half-open interval ``[edge[j], edge[j+1])``; the last cell
also contains ``edge[B]``. Concretely::

    index = searchsorted(edge[1:B], x, side="right"), clipped to [0, B-1]

so a value exactly equal to an internal edge goes to the *upper* cell, a
value below the observed minimum goes to cell 0, and a value above the
observed maximum goes to cell ``B-1``. Out-of-range values therefore sit in
the nearest edge cell until the next remap moves the boundaries.

Elite replacement
-----------------
A candidate replaces the current elite of its cell only if its objective is
*strictly* greater. Ties keep the existing elite. No other tie-breaker
(parameter count, age, novelty, variance) is used.

Remapping
---------
Every ``remap_frequency`` successful additions (or, if
``remap_at_generation_end=True``, once per generation via
:meth:`end_generation`), the boundaries are recomputed from the buffer, all
cell assignments are cleared, and every buffered record is re-inserted in
buffer order (oldest first) under the new boundaries. Collisions keep the
highest objective (ties -> the earlier record). With a finite buffer, an
elite whose record has already been evicted from the buffer is dropped at
the next remap; this matches the paper and pyribs. With
``buffer_capacity=None`` nothing is ever dropped.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "ArchiveRecord",
    "AddResult",
    "SlidingBoundariesArchive",
    "FixedGridArchive",
    "InvalidRecordError",
    "ArchiveNotInitializedError",
    "compute_boundaries_1d",
    "archive_from_state_dict",
]

STATE_VERSION = 1

Cell = tuple[int, ...]


class InvalidRecordError(ValueError):
    """Raised when a record has a non-finite objective or measures."""


class ArchiveNotInitializedError(RuntimeError):
    """Raised when the archive is used before :meth:`initialize`."""


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ArchiveRecord:
    """One successfully evaluated controller, as seen by the archive.

    ``objective`` decides who is the elite of a cell (higher is better);
    ``measures`` decide which cell the record belongs to. ``param_count`` and
    ``model_path`` are metadata only and never influence the archive.
    """

    gene_id: str
    generation: int
    objective: float
    measures: tuple[float, ...]
    param_count: int | None = None
    model_path: str | None = None

    def __post_init__(self) -> None:
        # Normalise to plain Python floats/tuples so equality and JSON
        # round-trips are exact and independent of NumPy scalar types.
        object.__setattr__(self, "objective", float(self.objective))
        object.__setattr__(self, "measures", tuple(float(m) for m in self.measures))
        object.__setattr__(self, "generation", int(self.generation))
        if self.param_count is not None:
            object.__setattr__(self, "param_count", int(self.param_count))

    def is_finite(self) -> bool:
        """True if the objective and every measure are finite numbers."""
        return math.isfinite(self.objective) and all(math.isfinite(m) for m in self.measures)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["measures"] = list(self.measures)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArchiveRecord":
        return cls(
            gene_id=str(d["gene_id"]),
            generation=int(d["generation"]),
            objective=float(d["objective"]),
            measures=tuple(float(m) for m in d["measures"]),
            param_count=None if d.get("param_count") is None else int(d["param_count"]),
            model_path=d.get("model_path"),
        )


@dataclass(frozen=True)
class AddResult:
    """Outcome of :meth:`SlidingBoundariesArchive.add`.

    ``cell`` is the cell the record was assigned to *at insertion time*,
    i.e. under the boundaries in force before any remap this addition may
    have triggered.
    """

    gene_id: str
    cell: Cell
    inserted: bool
    replaced_gene_id: str | None
    remapped: bool


# --------------------------------------------------------------------------
# Boundary computation
# --------------------------------------------------------------------------
def compute_boundaries_1d(values: Sequence[float] | np.ndarray, bins: int) -> np.ndarray:
    """Empirical-rank cell edges for one dimension (see module docstring).

    Returns an array of length ``bins + 1``: observed min, ``bins - 1``
    internal boundaries taken from the sorted observations, observed max.
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    if n == 0:
        raise ValueError("cannot compute boundaries from zero observations")
    if not np.all(np.isfinite(v)):
        raise InvalidRecordError("non-finite descriptor value in buffer")
    edges = np.empty(bins + 1, dtype=float)
    edges[0] = v[0]
    for j in range(1, bins):
        rank = min((j * n) // bins, n - 1)
        edges[j] = v[rank]
    edges[bins] = v[n - 1]
    return edges


# --------------------------------------------------------------------------
# Archive
# --------------------------------------------------------------------------
class SlidingBoundariesArchive:
    """MAP-Elites archive whose cell boundaries follow the observed data.

    Parameters
    ----------
    dims:
        Number of cells per behaviour dimension, e.g. ``(20, 20)``.
    remap_frequency:
        Recompute boundaries after this many successful additions (paper's
        delta). ``0``/``None`` disables count-based remapping.
    buffer_capacity:
        Maximum number of most-recent records kept for boundary estimation
        (paper's xi). ``None`` keeps every record (xi = infinity).
    seed:
        Seed for the archive's private RNG, used only by
        :meth:`sample_elites`. It never touches Python's or NumPy's global
        RNG, so an archive in shadow mode cannot perturb parent selection.
    qd_score_offset:
        If given, :meth:`shifted_qd_score` reports ``sum(objective - offset)``.
        The raw QD score is always available and never modified.
    remap_at_generation_end:
        Project adaptation: suppress count-based remaps and instead remap once
        per generation when :meth:`end_generation` is called. This is *not*
        the paper's schedule.
    """

    boundary_mode = "sliding"

    def __init__(
        self,
        dims: Sequence[int],
        remap_frequency: int | None = 100,
        buffer_capacity: int | None = None,
        seed: int | None = None,
        qd_score_offset: float | None = None,
        remap_at_generation_end: bool = False,
    ) -> None:
        dims = tuple(int(d) for d in dims)
        if not dims or any(d < 1 for d in dims):
            raise ValueError(f"dims must be a non-empty sequence of positive ints, got {dims}")
        if buffer_capacity is not None and int(buffer_capacity) < 1:
            raise ValueError("buffer_capacity must be >= 1 or None (unlimited)")
        if remap_frequency is not None and int(remap_frequency) < 0:
            raise ValueError("remap_frequency must be >= 0 or None")

        self._dims: tuple[int, ...] = dims
        self.remap_frequency: int = int(remap_frequency or 0)
        self.buffer_capacity: int | None = None if buffer_capacity is None else int(buffer_capacity)
        self.seed = seed
        self.qd_score_offset = None if qd_score_offset is None else float(qd_score_offset)
        self.remap_at_generation_end = bool(remap_at_generation_end)

        self._rng = np.random.default_rng(seed)
        self._buffer: deque[ArchiveRecord] = deque(maxlen=self.buffer_capacity)
        self._elites: dict[Cell, ArchiveRecord] = {}
        self._boundaries: list[np.ndarray] = []
        self._initialized = False
        self.total_seen = 0
        self.evaluations_since_remap = 0
        self.last_remap_at = 0
        self.num_remaps = 0
        self.num_evicted = 0
        #: One entry per boundary computation (initialisation and each remap).
        self.remap_events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ props
    @property
    def dims(self) -> tuple[int, ...]:
        return self._dims

    @property
    def ndim(self) -> int:
        return len(self._dims)

    @property
    def total_cells(self) -> int:
        return int(np.prod(self._dims))

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def boundaries(self) -> list[list[float]]:
        """Current cell edges per dimension (each of length ``dims[i] + 1``)."""
        return [b.tolist() for b in self._boundaries]

    @property
    def buffer(self) -> list[ArchiveRecord]:
        """Buffered records, oldest first (a copy)."""
        return list(self._buffer)

    @property
    def elites(self) -> dict[Cell, ArchiveRecord]:
        """Mapping cell -> elite record (a copy)."""
        return dict(self._elites)

    @property
    def num_elites(self) -> int:
        return len(self._elites)

    # ------------------------------------------------------------ validation
    def _check_record(self, record: ArchiveRecord) -> None:
        if not isinstance(record, ArchiveRecord):
            raise TypeError(f"expected ArchiveRecord, got {type(record).__name__}")
        if len(record.measures) != self.ndim:
            raise InvalidRecordError(
                f"gene {record.gene_id} (generation {record.generation}): expected "
                f"{self.ndim} measures, got {len(record.measures)}")
        if not record.is_finite():
            raise InvalidRecordError(
                f"gene {record.gene_id} (generation {record.generation}): non-finite "
                f"objective/measures ({record.objective}, {record.measures}); failed "
                "evaluations must never be added to the archive")

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise ArchiveNotInitializedError(
                "archive has no boundaries yet; call initialize(records) with the "
                "successfully evaluated initial population first")

    # -------------------------------------------------------------- boundary
    def _compute_boundaries(self) -> list[np.ndarray]:
        values = np.array([r.measures for r in self._buffer], dtype=float)
        return [compute_boundaries_1d(values[:, d], self._dims[d]) for d in range(self.ndim)]

    def degenerate_dimensions(self) -> list[int]:
        """Dimensions whose current edges contain duplicates.

        This happens when there are fewer distinct observed values than
        cells; some cells in those dimensions are then unreachable until the
        next remap. It is expected early in a run and is not an error.
        """
        out = []
        for d, edges in enumerate(self._boundaries):
            if np.unique(edges).size < edges.size:
                out.append(d)
        return out

    def index_of(self, measures: Sequence[float]) -> Cell:
        """Deterministic cell index for ``measures`` under current boundaries."""
        self._require_initialized()
        m = [float(x) for x in measures]
        if len(m) != self.ndim:
            raise InvalidRecordError(f"expected {self.ndim} measures, got {len(m)}")
        if not all(math.isfinite(x) for x in m):
            raise InvalidRecordError(f"non-finite measures {tuple(m)}")
        cell = []
        for d, x in enumerate(m):
            bins = self._dims[d]
            internal = self._boundaries[d][1:bins]
            idx = int(np.searchsorted(internal, x, side="right"))
            cell.append(min(max(idx, 0), bins - 1))
        return tuple(cell)

    # ------------------------------------------------------------- insertion
    def _insert(self, record: ArchiveRecord) -> tuple[Cell, bool, str | None]:
        cell = self.index_of(record.measures)
        current = self._elites.get(cell)
        if current is None:
            self._elites[cell] = record
            return cell, True, None
        if record.objective > current.objective:
            self._elites[cell] = record
            return cell, True, current.gene_id
        return cell, False, None

    def _append_to_buffer(self, record: ArchiveRecord) -> None:
        if self.buffer_capacity is not None and len(self._buffer) == self.buffer_capacity:
            self.num_evicted += 1
        self._buffer.append(record)
        self.total_seen += 1

    def initialize(self, records: Iterable[ArchiveRecord]) -> None:
        """Build the first boundaries from the initial population.

        No behaviour range is assumed: the successful initial evaluations
        alone define the first empirical distribution.
        """
        if self._initialized:
            raise RuntimeError("archive is already initialized")
        records = list(records)
        for r in records:
            self._check_record(r)
        if not records:
            raise ValueError(
                "MESB initialisation needs at least one successfully evaluated "
                "individual; zero succeeded (check the evaluation logs)")
        for r in records:
            self._append_to_buffer(r)
        self._initialized = True
        self._rebuild(event="initialize")
        self.evaluations_since_remap = 0

    def add(self, record: ArchiveRecord) -> AddResult:
        """Add one successful evaluation; may trigger a remap."""
        self._require_initialized()
        self._check_record(record)
        self._append_to_buffer(record)
        self.evaluations_since_remap += 1
        cell, inserted, replaced = self._insert(record)
        remapped = False
        if (not self.remap_at_generation_end and self.remap_frequency > 0
                and self.evaluations_since_remap >= self.remap_frequency):
            self.remap()
            remapped = True
        return AddResult(record.gene_id, cell, inserted, replaced, remapped)

    def add_many(self, records: Iterable[ArchiveRecord]) -> list[AddResult]:
        """Add records in the given order (order matters for remap timing)."""
        return [self.add(r) for r in records]

    # --------------------------------------------------------------- remap
    def _rebuild(self, event: str) -> None:
        if not self._buffer:
            raise RuntimeError("cannot remap an empty buffer")
        old = self.boundaries
        self._boundaries = self._compute_boundaries()
        self._elites = {}
        for record in self._buffer:  # oldest first -> ties keep the earlier record
            self._insert(record)
        # Sanity check: with an unlimited buffer every successful record ever
        # added must still be present.
        if self.buffer_capacity is None and len(self._buffer) != self.total_seen:
            raise AssertionError(
                f"buffer lost records: {len(self._buffer)} buffered vs {self.total_seen} seen")
        self.last_remap_at = self.total_seen
        self.remap_events.append({
            "event": event,
            "total_seen": self.total_seen,
            "buffer_size": len(self._buffer),
            "boundaries": self.boundaries,
            "previous_boundaries": old,
            "num_elites": len(self._elites),
        })

    def remap(self) -> None:
        """Recompute boundaries from the buffer and re-insert every record."""
        self._require_initialized()
        self._rebuild(event="remap")
        self.evaluations_since_remap = 0
        self.num_remaps += 1

    def end_generation(self) -> bool:
        """Remap once if generation-end remapping is enabled and data is new.

        Returns True if a remap happened. A no-op for the paper schedule.
        """
        if self.remap_at_generation_end and self._initialized and self.evaluations_since_remap > 0:
            self.remap()
            return True
        return False

    # ------------------------------------------------------------- queries
    def occupied_cells(self) -> list[Cell]:
        """Occupied cells in sorted (deterministic) order."""
        return sorted(self._elites)

    def coverage(self) -> float:
        return len(self._elites) / self.total_cells

    def sample_elites(self, n: int, replace: bool = True) -> list[ArchiveRecord]:
        """Uniformly sample occupied cells and return their elites.

        Uses only the archive's private, seedable RNG.
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        cells = self.occupied_cells()
        if not cells:
            raise RuntimeError("cannot sample parents from an empty archive")
        if not replace and n > len(cells):
            raise ValueError(f"cannot draw {n} distinct elites from {len(cells)} occupied cells")
        idx = self._rng.choice(len(cells), size=n, replace=replace)
        return [self._elites[cells[int(i)]] for i in idx]

    def best_elite(self) -> ArchiveRecord | None:
        best = None
        for cell in self.occupied_cells():  # deterministic tie-break: first cell
            r = self._elites[cell]
            if best is None or r.objective > best.objective:
                best = r
        return best

    def qd_score(self) -> float:
        """Raw QD score: sum of elite objectives (negative values untouched)."""
        return float(sum(r.objective for r in self._elites.values()))

    def shifted_qd_score(self) -> float | None:
        """``sum(objective - qd_score_offset)`` if an offset was configured."""
        if self.qd_score_offset is None:
            return None
        return float(sum(r.objective - self.qd_score_offset for r in self._elites.values()))

    def mean_elite_objective(self) -> float | None:
        if not self._elites:
            return None
        return self.qd_score() / len(self._elites)

    def summary(self) -> dict[str, Any]:
        best = self.best_elite()
        return {
            "boundary_mode": self.boundary_mode,
            "dims": list(self._dims),
            "occupied_cells": self.num_elites,
            "total_cells": self.total_cells,
            "coverage": self.coverage(),
            "raw_qd_score": self.qd_score(),
            "shifted_qd_score": self.shifted_qd_score(),
            "mean_elite_objective": self.mean_elite_objective(),
            "best_objective": None if best is None else best.objective,
            "best_gene_id": None if best is None else best.gene_id,
            "best_measures": None if best is None else list(best.measures),
            "total_seen": self.total_seen,
            "buffer_size": len(self._buffer),
            "num_remaps": self.num_remaps,
            "num_evicted": self.num_evicted,
            "evaluations_since_remap": self.evaluations_since_remap,
            "last_remap_at": self.last_remap_at,
            "degenerate_dimensions": self.degenerate_dimensions() if self._initialized else [],
        }

    # -------------------------------------------------------- serialisation
    def _config_dict(self) -> dict[str, Any]:
        return {
            "dims": list(self._dims),
            "remap_frequency": self.remap_frequency,
            "buffer_capacity": self.buffer_capacity,
            "seed": self.seed,
            "qd_score_offset": self.qd_score_offset,
            "remap_at_generation_end": self.remap_at_generation_end,
        }

    def state_dict(self) -> dict[str, Any]:
        """Complete, JSON-serialisable archive state (for checkpoints)."""
        return {
            "version": STATE_VERSION,
            "boundary_mode": self.boundary_mode,
            "config": self._config_dict(),
            "initialized": self._initialized,
            "boundaries": self.boundaries,
            "buffer": [r.to_dict() for r in self._buffer],
            "elites": [{"cell": list(c), "record": self._elites[c].to_dict()}
                       for c in self.occupied_cells()],
            "total_seen": self.total_seen,
            "evaluations_since_remap": self.evaluations_since_remap,
            "last_remap_at": self.last_remap_at,
            "num_remaps": self.num_remaps,
            "num_evicted": self.num_evicted,
            "remap_events": copy.deepcopy(self.remap_events),
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    @classmethod
    def _construct_from_config(cls, config: dict[str, Any]) -> "SlidingBoundariesArchive":
        return cls(
            dims=config["dims"],
            remap_frequency=config["remap_frequency"],
            buffer_capacity=config["buffer_capacity"],
            seed=config["seed"],
            qd_score_offset=config["qd_score_offset"],
            remap_at_generation_end=config["remap_at_generation_end"],
        )

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SlidingBoundariesArchive":
        """Restore an archive exactly as it was when ``state_dict`` was taken."""
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"unsupported archive state version {state.get('version')}")
        if state.get("boundary_mode") != cls.boundary_mode:
            raise ValueError(
                f"state is for a '{state.get('boundary_mode')}' archive, "
                f"not '{cls.boundary_mode}'")
        archive = cls._construct_from_config(state["config"])
        archive._initialized = bool(state["initialized"])
        archive._boundaries = [np.asarray(b, dtype=float) for b in state["boundaries"]]
        archive._buffer = deque((ArchiveRecord.from_dict(r) for r in state["buffer"]),
                                maxlen=archive.buffer_capacity)
        archive._elites = {tuple(int(i) for i in e["cell"]): ArchiveRecord.from_dict(e["record"])
                           for e in state["elites"]}
        archive.total_seen = int(state["total_seen"])
        archive.evaluations_since_remap = int(state["evaluations_since_remap"])
        archive.last_remap_at = int(state["last_remap_at"])
        archive.num_remaps = int(state["num_remaps"])
        archive.num_evicted = int(state.get("num_evicted", 0))
        archive.remap_events = copy.deepcopy(state.get("remap_events", []))
        archive._rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        for d, edges in enumerate(archive._boundaries):
            if len(edges) != archive._dims[d] + 1:
                raise ValueError("restored boundaries do not match dims")
        return archive


class FixedGridArchive(SlidingBoundariesArchive):
    """Conventional fixed-grid MAP-Elites, for controlled comparison only.

    Reproduces ``get_bin`` in run_improved.py (commit 6b73ae167):
    ``B`` equal-width bins over a manually specified ``[lo, hi]`` per
    dimension, out-of-range values clipped into the edge bins::

        frac  = clip((x - lo) / (hi - lo), 0, 1)
        index = min(int(frac * B), B - 1)

    Boundaries never move and no remapping happens. Everything else
    (insertion rule, buffer, metrics, sampling) is shared with MESB so that
    the boundary approach is the only difference in a comparison.
    """

    boundary_mode = "fixed"

    def __init__(self, dims: Sequence[int], ranges: Sequence[Sequence[float]],
                 buffer_capacity: int | None = None, seed: int | None = None,
                 qd_score_offset: float | None = None, **_ignored: Any) -> None:
        super().__init__(dims, remap_frequency=0, buffer_capacity=buffer_capacity, seed=seed,
                         qd_score_offset=qd_score_offset, remap_at_generation_end=False)
        ranges = [(float(lo), float(hi)) for lo, hi in ranges]
        if len(ranges) != len(self._dims):
            raise ValueError("need one (lo, hi) range per dimension")
        if any(not hi > lo for lo, hi in ranges):
            raise ValueError(f"every range needs hi > lo, got {ranges}")
        self.ranges = ranges

    def _compute_boundaries(self) -> list[np.ndarray]:
        return [np.linspace(lo, hi, b + 1) for (lo, hi), b in zip(self.ranges, self._dims)]

    def index_of(self, measures: Sequence[float]) -> Cell:
        self._require_initialized()
        m = [float(x) for x in measures]
        if len(m) != self.ndim or not all(math.isfinite(x) for x in m):
            raise InvalidRecordError(f"invalid measures {tuple(m)}")
        cell = []
        for (lo, hi), bins, x in zip(self.ranges, self._dims, m):
            frac = min(max((x - lo) / (hi - lo), 0.0), 1.0)
            cell.append(min(int(frac * bins), bins - 1))
        return tuple(cell)

    def remap(self) -> None:  # boundaries are fixed by definition
        return None

    def end_generation(self) -> bool:
        return False

    def _config_dict(self) -> dict[str, Any]:
        d = super()._config_dict()
        d["ranges"] = [list(r) for r in self.ranges]
        return d

    @classmethod
    def _construct_from_config(cls, config: dict[str, Any]) -> "FixedGridArchive":
        return cls(dims=config["dims"], ranges=config["ranges"],
                   buffer_capacity=config["buffer_capacity"], seed=config["seed"],
                   qd_score_offset=config["qd_score_offset"])


def archive_from_state_dict(state: dict[str, Any]) -> SlidingBoundariesArchive:
    """Restore either archive type from its ``state_dict``."""
    mode = state.get("boundary_mode")
    if mode == FixedGridArchive.boundary_mode:
        return FixedGridArchive.from_state_dict(state)
    if mode == SlidingBoundariesArchive.boundary_mode:
        return SlidingBoundariesArchive.from_state_dict(state)
    raise ValueError(f"unknown archive boundary_mode {mode!r}")
