import numpy as np
import pytest

from src.uqd.archive import DeepGridArchive, DepthArchive, Grid, MapElitesArchive

GRID = Grid([("foot_contact_0", 0.0, 1.0), ("foot_contact_1", 0.0, 1.0)], 4)


def test_grid_matches_legacy_binning():
    assert GRID.cell_of((0.0, 0.0)) == (0, 0)
    assert GRID.cell_of((1.0, 1.0)) == (3, 3)       # top edge in the last bin
    assert GRID.cell_of((0.26, 0.74)) == (1, 2)
    assert GRID.cell_of((-5, 7)) == (0, 3)          # clipped
    assert sorted(GRID.neighbors((0, 0))) == [(0, 1), (1, 0)]
    assert len(GRID.neighbors((1, 1), "moore")) == 8


def test_map_elites_strictly_greater_and_ties():
    fit = {"a": 10.0, "b": 10.0, "c": 11.0}
    arch = MapElitesArchive(GRID)
    assert arch.insert("a", 10.0, (0.1, 0.1), fit.get)[0]["event"] == "added"
    assert arch.insert("b", 10.0, (0.1, 0.1), fit.get)[0]["event"] == "rejected"   # tie keeps incumbent
    events = arch.insert("c", 11.0, (0.1, 0.1), fit.get)
    assert [e["event"] for e in events] == ["replaced", "retired"]
    assert arch.members() == ["c"]


def test_map_elites_margin_rule():
    fit = {"a": 10.0}
    arch = MapElitesArchive(GRID, rule="margin", margin=5.0)
    arch.insert("a", 10.0, (0.1, 0.1), fit.get)
    assert arch.insert("b", 14.9, (0.1, 0.1), fit.get)[0]["event"] == "rejected"
    assert arch.insert("c", 15.0, (0.1, 0.1), fit.get)[0]["event"] == "replaced"


def test_map_elites_relocation_collision():
    fit = {"a": 10.0, "b": 50.0}
    arch = MapElitesArchive(GRID)
    arch.insert("a", 10.0, (0.1, 0.1), fit.get)
    arch.insert("b", 50.0, (0.9, 0.9), fit.get)
    # a's re-estimate moves it into b's cell and it loses: a is retired, its old cell empties
    events = arch.relocate("a", 10.0, (0.9, 0.9), fit.get)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "relocating" and "rejected" in kinds and kinds[-1] == "retired"
    assert arch.members() == ["b"] and (0, 0) not in arch.cells
    # staying in the same cell is a no-op
    assert arch.relocate("b", 40.0, (0.95, 0.95), fit.get)[0]["event"] == "kept"


def test_map_elites_relocation_wins_new_cell():
    fit = {"a": 60.0, "b": 50.0}
    arch = MapElitesArchive(GRID)
    arch.insert("a", 60.0, (0.1, 0.1), fit.get)
    arch.insert("b", 50.0, (0.9, 0.9), fit.get)
    events = arch.relocate("a", 60.0, (0.9, 0.9), fit.get)
    assert any(e["event"] == "replaced" and e["displaced"] == "b" for e in events)
    assert arch.location["a"] == (3, 3) and "b" not in arch


def test_depth_archive_keeps_best_d_and_ties_keep_incumbents():
    fit = {"a": 5.0, "b": 9.0, "c": 7.0, "d": 7.0, "e": 1.0}
    arch = DepthArchive(GRID, depth=2)
    for sid in ("a", "b"):
        arch.insert(sid, fit[sid], (0.1, 0.1), fit.get)
    assert arch.cells[(0, 0)] == ["b", "a"]
    events = arch.insert("c", 7.0, (0.1, 0.1), fit.get)
    assert arch.cells[(0, 0)] == ["b", "c"]
    assert events[-1] == {"event": "retired", "solution": "a", "cell": [0, 0], "reason": "displaced", "by": "c"}
    assert arch.insert("d", 7.0, (0.1, 0.1), fit.get)[0]["event"] == "rejected"   # tie with c
    assert arch.insert("e", 1.0, (0.1, 0.1), fit.get)[0]["event"] == "rejected"


def test_depth_archive_rebuild_relocates_and_displaces():
    fit = {"a": 9.0, "b": 8.0, "x": 20.0, "y": 19.0}
    arch = DepthArchive(GRID, depth=2)
    arch.insert("a", 9.0, (0.1, 0.1), fit.get)
    arch.insert("b", 8.0, (0.1, 0.1), fit.get)
    arch.insert("x", 20.0, (0.9, 0.9), fit.get)
    arch.insert("y", 19.0, (0.9, 0.9), fit.get)
    # After re-evaluation a drifts into the full (3, 3) cell and is the worst there
    fit["a"] = 3.0
    events = arch.rebuild({"a": (3.0, (0.9, 0.9)), "b": (8.0, (0.1, 0.1)),
                           "x": (20.0, (0.9, 0.9)), "y": (19.0, (0.9, 0.9))}, fit.get)
    assert arch.cells[(0, 0)] == ["b"]                 # b now the elite of its cell
    assert arch.cells[(3, 3)] == ["x", "y"]
    assert any(e["event"] == "relocating" and e["solution"] == "a" for e in events)
    assert any(e["event"] == "retired" and e["solution"] == "a" and e["reason"] == "displaced" for e in events)


def test_depth_archive_parents_are_cell_elites():
    fit = {"a": 1.0, "b": 2.0}
    arch = DepthArchive(GRID, depth=2)
    arch.insert("a", 1.0, (0.1, 0.1), fit.get)
    arch.insert("b", 2.0, (0.1, 0.1), fit.get)
    parents = arch.select_parents(50, np.random.default_rng(0), fit.get)
    assert set(parents) == {"b"}


def test_deep_grid_random_replacement_ignores_fitness():
    fit = {}
    arch = DeepGridArchive(GRID, depth=2, replacement="random")
    rng = np.random.default_rng(0)
    for i, f in enumerate([100.0, 90.0]):
        fit[f"s{i}"] = f
        arch.insert(f"s{i}", f, (0.1, 0.1), fit.get, rng)
    fit["low"] = -50.0
    events = arch.insert("low", -50.0, (0.1, 0.1), fit.get, rng)   # even a bad solution is added
    assert events[0]["event"] == "replaced" and "low" in arch.cells[(0, 0)]
    assert len(arch.cells[(0, 0)]) == 2


def test_deep_grid_oldest_replacement():
    fit = {"a": 1.0, "b": 2.0, "c": 3.0}
    arch = DeepGridArchive(GRID, depth=2, replacement="oldest")
    rng = np.random.default_rng(0)
    for sid in ("a", "b", "c"):
        arch.insert(sid, fit[sid], (0.1, 0.1), fit.get, rng)
    assert sorted(arch.cells[(0, 0)]) == ["b", "c"]


@pytest.mark.parametrize("fits,expected", [
    ([5.0, 5.0, 5.0], [1 / 3, 1 / 3, 1 / 3]),              # all equal -> uniform
    ([0.0, 0.0], [0.5, 0.5]),                             # all zero -> uniform
    ([-10.0, -20.0, -30.0], [2 / 3, 1 / 3, 0.0]),          # negative: shifted by the min
    ([0.0, 1.0, 3.0], [0.0, 0.25, 0.75]),
    ([42.0], [1.0]),
])
def test_deep_grid_roulette(fits, expected):
    np.testing.assert_allclose(DeepGridArchive.roulette_probabilities(fits), expected)


def test_deep_grid_selection_frequencies():
    fit = {"a": 0.0, "b": 1.0, "c": 3.0}
    arch = DeepGridArchive(GRID, depth=3)
    rng = np.random.default_rng(1)
    for sid in fit:
        arch.insert(sid, fit[sid], (0.1, 0.1), fit.get, rng)
    picks = arch.select_parents(4000, np.random.default_rng(2), fit.get)
    assert picks.count("a") == 0
    assert picks.count("c") / len(picks) == pytest.approx(0.75, abs=0.03)
    assert arch.elites(fit.get) == {(0, 0): "c"}
