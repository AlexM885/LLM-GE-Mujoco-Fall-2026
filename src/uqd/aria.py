"""ARIA (Grillotti et al., "Don't Bet on Luck Alone", GECCO '23), on frozen
actor weights.

What is implemented, following Algorithms 1 and 2 of the paper:

* Reproducibility-Improvement Mechanism (Alg. 2): natural ES on a parameter
  vector phi. Each step samples eps_s ~ N(0, I), evaluates theta = phi +
  sigma * eps_s (and phi - sigma * eps_s with mirrored sampling, as in the
  paper's experiments), ranks the samples by the objective of Eq. 1, and
  ascends ``gradient = (1 / sigma) * sum_s u_s * eps_s``.
* Objective (Eq. 1) for target cell C with centre c:
      o = f                      if d is in C
      o = f_min - ||d - c||_2    otherwise
  f_min is the minimum fitness observed so far in the ARIA run (the paper's
  footnote for unbounded fitness).
* Reproducibility-Improvement phase (Alg. 1, lines 4-8): each input genotype
  is evaluated N_s times, its target is the cell of its mean descriptor, and
  it is optimised for N_grad steps.
* Archive-Completion phase (Alg. 1, lines 10-15): repeatedly pick an
  explored cell and an adjacent unexplored cell, optimise the explored cell's
  genotype toward the unexplored one, and add the result to that cell,
  whether or not its mean descriptor actually lands there (the paper's
  footnote; our metrics, like the paper's, place it by its held-out
  descriptor).

Choices the paper text leaves open, made here and recorded with every result:

* utilities      centred ranks in [-0.5, 0.5] (Salimans et al. 2017, the ES
                 the paper builds on); mirrored pairs are ranked as two samples
* optimizer      Adam (default lr 0.01); "gradient ascent step" in the paper
* adjacency      cells sharing an edge (von Neumann); Moore is configurable
* pair order     uniform random among valid (explored, unexplored) adjacent
                 pairs, matching "selected without taking into account their
                 solution's performance or reproducibility" (Sec. 6)
* duplicates     if two inputs share a target cell, the cell keeps the one with
                 the higher mean objective at its last step
* seeds          the two members of a mirrored pair share a reset seed
* post-eval      each optimised genotype is evaluated ``post_eval_episodes``
                 times to give it a search-time estimate (extra ARIA cost)
* non-finite     a perturbation whose rollout produces non-finite actions,
                 observations or rewards gets objective -inf (ranked last) and
                 is left out of f_min; an optimised genotype with a
                 non-finite post-evaluation episode is marked invalid

Budget: the paper uses N_s = 1440 (x2 mirrored), N_grad = 75, sigma = 0.01 on
Walker and runs completion over every empty cell. Anything smaller, including
a cap on completion cells, is labelled ``aria_reduced_budget``.

Representation caveat: our LLM mutations act on code and every child is
trained from scratch, so ARIA's refined weights are never inherited by the
search. ARIA results are a post-hoc refinement of the final archive and are
reported separately; they say nothing about the quality of the evolutionary
search itself.
"""

import numpy as np


def aria_objective(fitness, descriptor, cell, grid, f_min):
    if grid.contains(cell, descriptor):
        return float(fitness)
    centre = np.asarray(grid.center(cell))
    return float(f_min - np.linalg.norm(np.asarray(descriptor, dtype=float) - centre))


def centered_ranks(values):
    """Centred rank utilities in [-0.5, 0.5]; ties share the mean rank."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 1:
        return np.zeros(1)
    order = values.argsort(kind="stable")
    ranks = np.empty(n)
    ranks[order] = np.arange(n, dtype=float)
    for v in np.unique(values):          # average ranks of ties
        mask = values == v
        ranks[mask] = ranks[mask].mean()
    return ranks / (n - 1) - 0.5


def es_gradient(eps, utilities, sigma):
    """(1 / sigma) * sum_s u_s eps_s  (paper, Sec. 2.1)."""
    return (np.asarray(utilities)[:, None] * np.asarray(eps)).sum(axis=0) / sigma


class Adam:
    def __init__(self, dim, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = np.zeros(dim)
        self.v = np.zeros(dim)
        self.t = 0

    def ascent_step(self, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class SGD:
    def __init__(self, dim, lr):
        self.lr = lr

    def ascent_step(self, grad):
        return self.lr * grad


class FitnessFloor:
    """Running minimum fitness seen during the ARIA run (Eq. 1 footnote)."""

    def __init__(self):
        self.value = None

    def update(self, fitnesses):
        finite = [f for f in fitnesses if np.isfinite(f)]
        if finite:
            low = float(np.min(finite))
            self.value = low if self.value is None else min(self.value, low)
        return self.value


def reproducibility_improvement(key, phi0, cell, grid, cfg, evaluate, seed_fn, rng, floor, log=None):
    """Alg. 2. ``evaluate(key, params, seeds)`` -> list of (fitness, descriptor,
    nonfinite). Returns (phi, last mean objective over finite samples,
    episodes used)."""
    phi = np.array(phi0, dtype=float)
    opt = (Adam if cfg["optimizer"] == "adam" else SGD)(phi.size, cfg["learning_rate"])
    n, sigma = cfg["n_samples"], cfg["sigma"]
    episodes = 0
    last = None
    for step in range(cfg["n_grad_steps"]):
        eps = rng.standard_normal((n, phi.size))
        seeds = seed_fn(step, n)
        thetas = [phi + sigma * e for e in eps]
        noise = list(eps)
        pair_seeds = list(seeds)
        if cfg["mirrored"]:
            thetas += [phi - sigma * e for e in eps]
            noise += [-e for e in eps]
            pair_seeds += list(seeds)
        results = [evaluate(key, theta, [s])[0] for theta, s in zip(thetas, pair_seeds)]
        episodes += len(results)
        f_min = floor.update([f for f, _, bad in results if not bad])
        objectives = [-np.inf if bad or f_min is None else aria_objective(f, d, cell, grid, f_min)
                      for f, d, bad in results]
        grad = es_gradient(noise, centered_ranks(objectives), sigma)
        phi = phi + opt.ascent_step(grad)
        finite = [o for o in objectives if np.isfinite(o)]
        last = float(np.mean(finite)) if finite else -np.inf
        if log:
            log(step=step, mean_objective=last, nonfinite=int(sum(bad for _, _, bad in results)),
                in_cell=float(np.mean([(not bad) and grid.contains(cell, d) for _, d, bad in results])))
    return phi, last, episodes


def run_aria(inputs, grid, cfg, evaluate, seeds, rng, post_eval_episodes=8, log=None):
    """Alg. 1.

    inputs: list of dicts with "label" and "params" (flat actor weights).
    evaluate(key, params, seed_list) -> list of (fitness, descriptor, nonfinite);
    ``key`` names the input checkpoint whose architecture ``params`` fits.
    seeds: callable(label, step, count) -> list of reset seeds (ARIA stream).
    Returns (cells, cost): cells maps cell -> result dict with "params",
    "source" (input label), "phase", "objective", "post_eval" samples.
    """
    floor = FitnessFloor()
    cells = {}
    episodes = 0
    explored = set()

    def optimise(key, label, phi0, cell, phase):
        nonlocal episodes
        seed_fn = lambda step, count: seeds(f"{label}->{cell}", step, count)  # noqa: E731
        phi, objective, used = reproducibility_improvement(
            key, phi0, cell, grid, cfg, evaluate, seed_fn, rng, floor,
            log=(lambda **kw: log(phase=phase, source=label, cell=list(cell), **kw)) if log else None)
        episodes += used
        post = evaluate(key, phi, seeds(f"{label}->{cell}:post", 0, post_eval_episodes))
        episodes += post_eval_episodes
        return {"params": phi, "key": key, "source": label, "phase": phase, "objective": objective,
                "valid": not any(bad for _, _, bad in post),
                "post_eval": [(float(f), [float(x) for x in d], bool(bad)) for f, d, bad in post]}

    # Reproducibility-Improvement phase
    for item in inputs:
        init = evaluate(item["label"], item["params"], seeds(f"{item['label']}:init", 0, cfg["n_samples"]))
        episodes += len(init)
        floor.update([f for f, _, bad in init if not bad])
        good = [d for _, d, bad in init if not bad]
        if not good:
            if log:
                log(phase="improvement", source=item["label"], skipped="every initial rollout non-finite")
            continue
        mean_d = np.mean(good, axis=0)
        cell = grid.cell_of(mean_d)
        result = optimise(item["label"], item["label"], item["params"], cell, "improvement")
        if cell not in cells or result["objective"] > cells[cell]["objective"]:
            cells[cell] = result
        explored.add(cell)

    # Archive-Completion phase
    completed = 0
    stopped = "all reachable cells explored"
    if cfg.get("completion_phase", True):
        all_cells = set(grid.all_cells())
        while explored != all_cells:
            if cfg.get("max_completion_cells") is not None and completed >= cfg["max_completion_cells"]:
                stopped = "max_completion_cells reached (reduced budget)"
                break
            pairs = sorted((s, t) for s in explored for t in grid.neighbors(s, cfg["adjacency"])
                           if t not in explored)
            if not pairs:
                stopped = "no unexplored cell is adjacent to an explored cell"
                break
            source, target = pairs[int(rng.integers(len(pairs)))]
            result = optimise(cells[source]["key"], f"cell{list(source)}", cells[source]["params"], target,
                              "completion")
            cells[target] = result
            explored.add(target)
            completed += 1
    else:
        stopped = "completion phase disabled (reduced budget)"
    return cells, {"episodes": episodes, "completion_cells": completed, "completion_stop": stopped}
