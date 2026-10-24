"""Archive containers for the uncertain-QD strategies.

Archives store solution ids only. Estimates (fitness, descriptor) are passed in
by the engine, which owns the sample records, so an archive never mixes the
statistics of two checkpoints.

Each mutating call returns a list of event dicts that the engine writes to the
run's event log, so every insertion, replacement, relocation and removal is
traceable after the run.

Grid
    Regular grid over MAP_ELITES_DESCRIPTORS with MAP_BINS bins per dimension.
    ``cell_of`` uses exactly the binning of run_improved.get_bin, so every
    strategy (and the legacy archive) shares one reporting layout.

MapElitesArchive  (MAP-Elites, MAP-Elites-sampling)
    One solution per cell. A newcomer replaces the incumbent only if its
    estimate is strictly greater (the legacy rule), or, in the experimental
    ``margin`` rule, greater by at least ``margin``. ``relocate`` handles an
    incumbent whose re-estimated descriptor has left its cell: it is removed
    and re-inserted under the same rule, and is retired if it loses; the cell
    it left stays empty (MAP-Elites has no reservoir).

DepthArchive  (Archive-sampling, Parallel-Adaptive-sampling; UQD Alg. 1/2)
    Up to D solutions per cell, ordered by fitness estimate. Insertion keeps
    the D best; ties keep the solution already in the cell, as the reference
    implementation's top_k over [cell, candidates] does. Only the best of a
    cell can be selected as a parent. ``rebuild`` empties the archive and
    re-adds every stored solution with its updated estimate (Alg. 1 lines
    4-7); solutions that no longer fit in their (possibly new) cell are
    retired with reason ``displaced``.

DeepGridArchive  (Deep-Grid, Deep-Grid-sampling; Flageat & Cully 2020)
    Up to D solutions per cell. Every evaluated offspring is added: while the
    cell has room it is appended, otherwise it replaces a member chosen
    uniformly at random (``replacement='random'``, the paper and its sferes
    implementation) or the oldest member (``replacement='oldest'``, the UQD
    JAX implementation's seniority rule). Fitness and age are otherwise
    ignored. Parent selection picks a filled cell uniformly, then a member by
    fitness-proportional roulette on fitness shifted by the cell minimum:
    w_i = f_i - min_j f_j. This handles negative and zero fitness. When all
    weights are zero (every member has the same fitness, or the cell has one
    member) the choice is uniform. The worst member of a cell with distinct
    fitnesses gets probability 0, as in both reference implementations.
    Deviation: the sferes code initialises the running minimum at 1, so for
    cells whose fitnesses are all above 1 it subtracts 1 rather than the
    minimum; we always subtract the true minimum, as the UQD code does.
"""

import itertools

import numpy as np


class Grid:
    def __init__(self, descriptors, bins):
        if not descriptors:
            raise ValueError("a grid needs at least one descriptor")
        self.descriptors = [(name, float(lo), float(hi)) for name, lo, hi in descriptors]
        for name, lo, hi in self.descriptors:
            if hi <= lo:
                raise ValueError(f"descriptor {name} has an empty range")
        self.bins = int(bins)
        self.dims = len(self.descriptors)

    @property
    def n_cells(self):
        return self.bins ** self.dims

    def cell_of(self, descriptor):
        """Cell index tuple; identical to run_improved.get_bin."""
        if len(descriptor) != self.dims:
            raise ValueError(f"descriptor has {len(descriptor)} dims, grid has {self.dims}")
        coords = []
        for value, (_, lo, hi) in zip(descriptor, self.descriptors):
            frac = np.clip((float(value) - lo) / (hi - lo), 0, 1)
            coords.append(min(int(frac * self.bins), self.bins - 1))
        return tuple(coords)

    def contains(self, cell, descriptor):
        return self.cell_of(descriptor) == tuple(cell)

    def center(self, cell):
        return tuple(lo + (c + 0.5) * (hi - lo) / self.bins
                     for c, (_, lo, hi) in zip(cell, self.descriptors))

    def neighbors(self, cell, adjacency="von_neumann"):
        """In-grid neighbouring cells (shared edge, or also shared corner)."""
        out = []
        for delta in itertools.product((-1, 0, 1), repeat=self.dims):
            if not any(delta):
                continue
            if adjacency == "von_neumann" and sum(abs(x) for x in delta) != 1:
                continue
            nb = tuple(c + d for c, d in zip(cell, delta))
            if all(0 <= x < self.bins for x in nb):
                out.append(nb)
        return out

    def all_cells(self):
        return list(itertools.product(range(self.bins), repeat=self.dims))


def _event(kind, sid, cell=None, **extra):
    row = {"event": kind, "solution": sid, "cell": list(cell) if cell is not None else None}
    row.update(extra)
    return row


class _BaseArchive:
    strategy = None

    def __init__(self, grid, depth):
        self.grid = grid
        self.depth = int(depth)
        self.cells = {}          # cell tuple -> list of solution ids
        self.location = {}       # solution id -> cell tuple

    def __contains__(self, sid):
        return sid in self.location

    def members(self):
        return [sid for cell in sorted(self.cells) for sid in self.cells[cell]]

    def filled_cells(self):
        return sorted(c for c, m in self.cells.items() if m)

    def remove(self, sid, reason):
        cell = self.location.pop(sid)
        self.cells[cell].remove(sid)
        if not self.cells[cell]:
            del self.cells[cell]
        return [_event("retired", sid, cell, reason=reason)]

    def to_dict(self):
        return {
            "strategy": self.strategy,
            "depth": self.depth,
            "cells": {",".join(map(str, c)): list(m) for c, m in sorted(self.cells.items())},
        }


class MapElitesArchive(_BaseArchive):
    strategy = "map_elites"

    def __init__(self, grid, rule="strictly_greater", margin=0.0):
        super().__init__(grid, depth=1)
        if rule not in ("strictly_greater", "margin"):
            raise ValueError(f"unknown replacement rule {rule!r}")
        self.rule = rule
        self.margin = float(margin)

    def _beats(self, challenger, incumbent):
        if self.rule == "margin":
            return challenger >= incumbent + self.margin
        return challenger > incumbent

    def insert(self, sid, fitness, descriptor, fitness_of):
        if sid in self.location:
            raise ValueError(f"{sid} is already archived")
        cell = self.grid.cell_of(descriptor)
        occupants = self.cells.get(cell, [])
        if not occupants:
            self.cells[cell] = [sid]
            self.location[sid] = cell
            return [_event("added", sid, cell, fitness=fitness)]
        incumbent = occupants[0]
        incumbent_fitness = fitness_of(incumbent)
        if self._beats(fitness, incumbent_fitness):
            self.location.pop(incumbent)
            self.cells[cell] = [sid]
            self.location[sid] = cell
            return [_event("replaced", sid, cell, fitness=fitness, displaced=incumbent,
                           displaced_fitness=incumbent_fitness),
                    _event("retired", incumbent, cell, reason="replaced", by=sid)]
        return [_event("rejected", sid, cell, fitness=fitness, incumbent=incumbent,
                       incumbent_fitness=incumbent_fitness)]

    def relocate(self, sid, fitness, descriptor, fitness_of):
        """Re-place an archived solution after its estimate changed."""
        old_cell = self.location[sid]
        new_cell = self.grid.cell_of(descriptor)
        if new_cell == old_cell:
            return [_event("kept", sid, old_cell, fitness=fitness)]
        self.location.pop(sid)
        del self.cells[old_cell]
        events = [_event("relocating", sid, old_cell, to=list(new_cell), fitness=fitness)]
        result = self.insert(sid, fitness, descriptor, fitness_of)
        events.extend(result)
        if result[0]["event"] == "rejected":
            events.append(_event("retired", sid, new_cell, reason="lost_after_relocation"))
        return events

    def select_parents(self, n, rng, fitness_of=None):
        cells = self.filled_cells()
        if not cells:
            return []
        return [self.cells[cells[i]][0] for i in rng.integers(len(cells), size=n)]

    def elites(self, fitness_of=None):
        return {cell: m[0] for cell, m in self.cells.items()}


class DepthArchive(_BaseArchive):
    strategy = "depth"

    def insert(self, sid, fitness, descriptor, fitness_of):
        if sid in self.location:
            raise ValueError(f"{sid} is already archived")
        cell = self.grid.cell_of(descriptor)
        members = self.cells.setdefault(cell, [])
        # Insert after every member with fitness >= newcomer: ties keep incumbents.
        pos = 0
        while pos < len(members) and fitness_of(members[pos]) >= fitness:
            pos += 1
        if pos >= self.depth:
            if not members:
                del self.cells[cell]
            return [_event("rejected", sid, cell, fitness=fitness, rank=pos)]
        members.insert(pos, sid)
        self.location[sid] = cell
        events = [_event("added", sid, cell, fitness=fitness, rank=pos)]
        if len(members) > self.depth:
            dropped = members.pop()
            self.location.pop(dropped)
            events.append(_event("retired", dropped, cell, reason="displaced", by=sid))
        return events

    def rebuild(self, estimates, fitness_of):
        """Empty the archive and re-add every stored solution (Alg. 1, 4-7).

        ``estimates`` maps solution id -> (fitness, descriptor) after the
        re-evaluation. Solutions are re-added in their previous order (cell,
        then rank), which only matters for exact fitness ties.
        """
        previous = self.members()
        old_location = dict(self.location)
        self.cells, self.location = {}, {}
        events = []
        for sid in previous:
            if sid not in estimates:
                continue  # removed by the engine (e.g. disqualified)
            fitness, descriptor = estimates[sid]
            result = self.insert(sid, fitness, descriptor, fitness_of)
            new_cell = self.grid.cell_of(descriptor)
            if new_cell != old_location[sid]:
                events.append(_event("relocating", sid, old_location[sid], to=list(new_cell),
                                     fitness=fitness))
            for row in result:
                if row["event"] == "rejected":
                    row = _event("retired", sid, new_cell, reason="displaced", fitness=fitness)
                events.append(row)
        return events

    def select_parents(self, n, rng, fitness_of=None):
        cells = self.filled_cells()
        if not cells:
            return []
        return [self.cells[cells[i]][0] for i in rng.integers(len(cells), size=n)]

    def elites(self, fitness_of=None):
        return {cell: m[0] for cell, m in self.cells.items()}


class DeepGridArchive(_BaseArchive):
    strategy = "deep_grid"

    def __init__(self, grid, depth, replacement="random"):
        super().__init__(grid, depth)
        if replacement not in ("random", "oldest"):
            raise ValueError(f"unknown Deep-Grid replacement {replacement!r}")
        self.replacement = replacement
        self.arrival = {}          # solution id -> insertion counter (age)
        self._counter = 0

    def insert(self, sid, fitness, descriptor, fitness_of, rng):
        if sid in self.location:
            raise ValueError(f"{sid} is already archived")
        cell = self.grid.cell_of(descriptor)
        members = self.cells.setdefault(cell, [])
        self._counter += 1
        self.arrival[sid] = self._counter
        if len(members) < self.depth:
            members.append(sid)
            self.location[sid] = cell
            return [_event("added", sid, cell, fitness=fitness)]
        if self.replacement == "random":
            idx = int(rng.integers(len(members)))
        else:
            idx = min(range(len(members)), key=lambda i: self.arrival[members[i]])
        victim = members[idx]
        members[idx] = sid
        self.location.pop(victim)
        self.arrival.pop(victim, None)
        self.location[sid] = cell
        return [_event("replaced", sid, cell, fitness=fitness, displaced=victim,
                       rule=self.replacement),
                _event("retired", victim, cell, reason="deep_grid_replaced", by=sid)]

    def remove(self, sid, reason):
        self.arrival.pop(sid, None)
        return super().remove(sid, reason)

    @staticmethod
    def roulette_probabilities(fitnesses):
        """Selection probabilities within one cell (see module docstring)."""
        f = np.asarray(fitnesses, dtype=float)
        if f.size == 0:
            raise ValueError("empty cell")
        w = f - f.min()
        total = w.sum()
        if total <= 0:
            return np.full(f.size, 1.0 / f.size)
        return w / total

    def select_parents(self, n, rng, fitness_of):
        cells = self.filled_cells()
        if not cells:
            return []
        out = []
        for i in rng.integers(len(cells), size=n):
            members = self.cells[cells[i]]
            p = self.roulette_probabilities([fitness_of(m) for m in members])
            out.append(members[int(rng.choice(len(members), p=p))])
        return out

    def elites(self, fitness_of):
        """Best member of each cell (ties: the earlier arrival)."""
        out = {}
        for cell, members in self.cells.items():
            out[cell] = max(members, key=lambda m: (fitness_of(m), -self.arrival[m]))
        return out

    def to_dict(self):
        data = super().to_dict()
        data["replacement"] = self.replacement
        data["arrival"] = dict(self.arrival)
        return data
