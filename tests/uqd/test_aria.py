import numpy as np
import pytest

from src.cfg.uqd_presets import DEFAULTS
from src.uqd.archive import Grid
from src.uqd.aria import (Adam, FitnessFloor, aria_objective, centered_ranks, es_gradient,
                          reproducibility_improvement, run_aria)
from src.uqd.seeds import SeedScheme, stream_of

GRID = Grid([("foot_contact_0", 0.0, 1.0), ("foot_contact_1", 0.0, 1.0)], 4)


def test_objective_inside_and_outside_cell():
    assert aria_objective(42.0, (0.1, 0.1), (0, 0), GRID, f_min=-5.0) == 42.0
    centre = np.array(GRID.center((0, 0)))
    d = np.array((0.9, 0.9))
    assert aria_objective(42.0, d, (0, 0), GRID, -5.0) == pytest.approx(-5.0 - np.linalg.norm(d - centre))
    # every in-cell sample outranks every out-of-cell one (f >= f_min by construction)
    assert aria_objective(-5.0, (0.1, 0.1), (0, 0), GRID, -5.0) > aria_objective(1e9, d, (0, 0), GRID, -5.0)


def test_centered_ranks():
    np.testing.assert_allclose(centered_ranks([3.0, 1.0, 2.0]), [0.5, -0.5, 0.0])
    np.testing.assert_allclose(centered_ranks([1.0, 1.0]), [0.0, 0.0])          # ties share rank
    r = centered_ranks([-np.inf, 0.0, 5.0])
    assert r[0] == -0.5                                                          # -inf ranked last


def test_es_gradient_matches_formula():
    eps = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(es_gradient(eps, [0.5, -0.5], sigma=0.1), [5.0, -5.0])


def test_fitness_floor_ignores_nonfinite():
    floor = FitnessFloor()
    assert floor.update([np.nan, 3.0, 1.0]) == 1.0
    assert floor.update([5.0]) == 1.0


def _toy_world(nonfinite_after=None):
    """Descriptor = clip(theta[:2]); fitness = -||theta - target||; deterministic."""
    def evaluate(key, params, seeds):
        p = np.asarray(params)
        bad = nonfinite_after is not None and p[0] > nonfinite_after
        d = np.clip(p[:2], 0, 1)
        return [(float(-np.sum((p - 0.6) ** 2)), d.tolist(), bad) for _ in seeds]
    return evaluate


def _cfg(**kw):
    cfg = dict(DEFAULTS["aria"])
    cfg.update(enabled=True, n_samples=8, n_grad_steps=30, sigma=0.05, learning_rate=0.05)
    cfg.update(kw)
    return cfg


def test_reproducibility_improvement_moves_toward_target_cell():
    scheme = SeedScheme(0)
    phi, objective, used = reproducibility_improvement(
        "k", np.array([0.1, 0.1, 0.0]), (2, 2), GRID, _cfg(), _toy_world(),
        lambda step, n: scheme.aria_seeds("t", step, n), np.random.default_rng(0), FitnessFloor())
    assert GRID.cell_of(np.clip(phi[:2], 0, 1)) == (2, 2)
    assert used == 30 * 8 * 2                    # mirrored sampling doubles evaluations


def test_run_aria_phases_cap_and_labels():
    scheme = SeedScheme(1)
    seen = []

    def seeds(label, step, count):
        out = scheme.aria_seeds(label, step, count)
        seen.extend(out)
        return out
    inputs = [{"label": "a", "params": np.array([0.1, 0.1, 0.0])},
              {"label": "b", "params": np.array([0.12, 0.13, 0.0])}]     # same target cell as a
    cells, info = run_aria(inputs, GRID, _cfg(n_grad_steps=3, max_completion_cells=2), _toy_world(),
                           seeds, np.random.default_rng(0), post_eval_episodes=2)
    assert (0, 0) in cells and info["completion_cells"] == 2
    assert info["completion_stop"].startswith("max_completion_cells")
    assert len(cells) == 3
    completed = [c for c, r in cells.items() if r["phase"] == "completion"]
    assert all(any(nb in cells for nb in GRID.neighbors(c)) for c in completed)
    assert all(stream_of(s) == "aria" for s in seen)


def test_run_aria_completion_disabled():
    cells, info = run_aria([{"label": "a", "params": np.array([0.1, 0.1])}], GRID,
                           _cfg(n_grad_steps=1, completion_phase=False), _toy_world(),
                           SeedScheme(0).aria_seeds, np.random.default_rng(0), post_eval_episodes=1)
    assert len(cells) == 1 and info["completion_stop"].startswith("completion phase disabled")


def test_nonfinite_rollouts_rank_last_and_invalidate():
    cells, _ = run_aria([{"label": "a", "params": np.array([0.1, 0.1])}], GRID,
                        _cfg(n_grad_steps=2, completion_phase=False), _toy_world(nonfinite_after=-1.0),
                        SeedScheme(0).aria_seeds, np.random.default_rng(0), post_eval_episodes=1)
    # every rollout is non-finite: the initial evaluation has no usable descriptor -> skipped
    assert cells == {}


def test_adam_ascends():
    opt = Adam(1, lr=0.1)
    assert opt.ascent_step(np.array([2.0]))[0] > 0
