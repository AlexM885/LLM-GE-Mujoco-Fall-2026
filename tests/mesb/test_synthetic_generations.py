"""Synthetic multi-generation run of the archive alone (no driver, LLM or MuJoCo).

The behaviour distribution drifts over generations the way a learning
population does (controllers get faster). MESB boundaries must follow the
data, and coverage must stay high where a badly guessed fixed grid collapses.
"""

import numpy as np

from sota.MujocoRL.mesb_archive import ArchiveRecord, FixedGridArchive, SlidingBoundariesArchive


def _generation(rng, g, n):
    distance = rng.lognormal(mean=3 + 0.4 * g, sigma=0.5, size=n) - 10
    control = rng.gamma(2.0, 20 + 5 * g, size=n)
    reward = distance - 0.1 * control + rng.normal(0, 5, size=n)
    return [ArchiveRecord(f"g{g}_{i}", g, reward[i], (distance[i], control[i]))
            for i in range(n)]


def test_boundaries_track_drifting_population():
    rng = np.random.default_rng(0)
    sliding = SlidingBoundariesArchive(dims=(10, 10), remap_frequency=100, seed=0)
    # A plausible-but-wrong manual guess, like the legacy [-500, 2000] x [0, 10].
    fixed = FixedGridArchive(dims=(10, 10), ranges=[(-500, 2000), (0, 10)])
    init = _generation(rng, 0, 64)
    sliding.initialize(init)
    fixed.initialize(init)
    first = sliding.boundaries
    history = [first]
    for g in range(1, 12):
        batch = _generation(rng, g, 64)
        sliding.add_many(batch)
        fixed.add_many(batch)
        sliding.end_generation()
        history.append(sliding.boundaries)

    assert sliding.num_remaps == (64 * 11) // 100
    assert sliding.boundaries != first
    # median distance boundary moved upward as the population got faster
    assert history[-1][0][5] > history[0][0][5]
    # every successful record retained with xi = infinity
    assert len(sliding.buffer) == sliding.total_seen == 64 * 12
    # the data-driven grid is far better used than the mis-ranged fixed grid
    assert sliding.coverage() > 0.5
    assert fixed.coverage() < 0.2
    # QD metrics are consistent with the elites
    elites = sliding.elites.values()
    assert np.isclose(sliding.qd_score(), sum(r.objective for r in elites))
    assert sliding.best_elite().objective == max(r.objective for r in elites)


def test_each_remap_gives_roughly_equal_mass_marginals():
    rng = np.random.default_rng(1)
    a = SlidingBoundariesArchive(dims=(8, 8), remap_frequency=100, seed=0)
    a.initialize(_generation(rng, 0, 50))
    a.add_many(_generation(rng, 1, 350))  # remaps at 100, 200, 300
    a.remap()
    vals = np.array([r.measures for r in a.buffer])
    for d in range(2):
        counts = np.histogram(vals[:, d], bins=np.array(a.boundaries[d]))[0]
        assert counts.min() >= 0.8 * len(vals) / 8
