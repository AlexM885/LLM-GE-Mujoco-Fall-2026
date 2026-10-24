"""Named configurations for the uncertain-QD (UQD) experiments on MuJoCo.

A preset is selected with the MUJOCO_UQD_PRESET environment variable (default
``legacy_baseline``) and may be adjusted with MUJOCO_UQD_OVERRIDES, a JSON
object merged on top of it. constants_Mujoco.py resolves and validates the
result once at import, so an invalid combination stops the run before any job
is submitted. See sota/MujocoRL/UQD.md for the method descriptions.

Preset categories
-----------------
legacy      The pre-existing pipeline, untouched: one TD3 run per gene, 10
            evaluation episodes with reset seeds seed+0..9 shared by every
            gene, one elite per cell, strict-improvement replacement.
            run_improved.py takes its original code path for this preset.
controlled  Ablations under one shared protocol: the same offspring count per
            generation (so the same training cost), the same 10-episode
            initial evaluation, per-solution rollout seed streams, and an
            optional common total interaction budget. Only the uncertainty
            mechanism differs between controlled presets.
paper       Reproductions of the published algorithms with their own sample
            definitions (one sample = one episode) and the UQD paper's
            per-generation sampling-size budget rule.
experimental
            Our own variants. Names start with ``exp_``; nothing here is
            claimed to come from a paper.

What a "sample" is
------------------
sample_unit = "rollout": one evaluation episode of one trained checkpoint
    under a fixed evaluation protocol, with a fresh reset seed. Estimates
    built from rollouts describe the reliability of that trained policy, not
    the reliability of its gene across training runs.
sample_unit = "retrain": one TD3 training run of the gene with a fresh
    training seed, summarised by the mean over ``episodes_per_replicate``
    episodes of the resulting checkpoint. Episodes from the same training run
    are never counted as independent training samples.
"""

import copy
import json

ARCHIVE_STRATEGIES = (
    "legacy_map_elites",
    "map_elites",
    "archive_sampling",
    "parallel_adaptive_sampling",
    "deep_grid",
)
SAMPLE_UNITS = ("rollout", "retrain")
CATEGORIES = ("legacy", "controlled", "paper", "experimental")
OFFSPRING_SAMPLING = ("fixed", "pas_median")
PAS_COUNT_STATISTICS = ("median",)
DEEP_GRID_REPLACEMENT = ("random", "oldest")
REPLACEMENT_RULES = ("strictly_greater", "margin")
BUDGET_MODES = ("fixed_offspring", "sampling_size")
REEVAL_FAILURE_POLICIES = ("keep_estimate",)
FINAL_ESTIMATORS = ("median", "mean")
ARIA_ADJACENCY = ("von_neumann", "moore")

DEFAULTS = {
    "name": None,
    "category": None,
    "description": "",
    "archive_strategy": "map_elites",
    "sample_unit": "rollout",
    # rollout: episodes in a new policy's initial evaluation.
    # retrain: training replicates per new gene.
    "initial_samples": 10,
    # retrain only: episodes used to summarise each training replicate.
    "episodes_per_replicate": None,
    # "pas_median": offspring get N = statistic of the archive's sample counts
    # (Parallel-Adaptive-sampling, UQD Alg. 2 line 4), clipped to
    # [pas_min_samples, pas_max_samples].
    "offspring_sampling": "fixed",
    "pas_count_statistic": "median",
    "pas_min_samples": 1,
    "pas_max_samples": None,
    # Solutions kept per cell. 1 for MAP-Elites; D for the depth archives.
    "depth": 1,
    # Deep-Grid only. "random" follows Flageat & Cully (2020) and its sferes
    # implementation; "oldest" follows the UQD JAX implementation (seniority).
    "deep_grid_replacement": "random",
    # Periodic re-evaluation of every archived solution.
    "archive_reeval": {
        "enabled": False,
        "every": 1,            # generations between re-evaluations
        "samples": 1,          # new samples per archived solution
        "start_generation": 1,
    },
    # At most this many of a solution's most recent valid samples enter its
    # estimate (UQD reference implementation: max_number_evals = 500).
    "sample_window": 500,
    "replacement": {
        "rule": "strictly_greater",
        # "margin" only: a challenger must beat the incumbent's estimate by at
        # least this many return units. An experimental heuristic, not a test.
        "margin": 0.0,
    },
    "budget": {
        # fixed_offspring: offspring_per_generation new genes each generation.
        # sampling_size: UQD paper rule, offspring = (S - archive re-evals) // N.
        "mode": "fixed_offspring",
        "offspring_per_generation": None,   # None -> population_size constant
        "sampling_size": None,              # episodes per generation
        "max_offspring_per_generation": None,
        # Stop the run once measured train + evaluation env steps reach this.
        "total_env_steps": None,
    },
    "infra": {
        # Resubmissions of an evaluation lost to the scheduler or filesystem.
        # Retries reuse the same seeds, so a lost job never consumes seeds.
        "max_retries": 2,
        "on_reeval_failure": "keep_estimate",
    },
    "final_eval": {
        "episodes": 64,                   # UQD paper uses M = 512
        "fitness_estimator": "median",    # UQD paper: median over M
        "descriptor_estimator": "median",
        # Fixed bounds for the normalised QD-score, shared by every config.
        "qd_score_bounds": [0.0, 5000.0],
        # Held-out reset seeds are drawn from this master seed, independent of
        # the run seed, so every configuration is scored on the same starts.
        "master_seed": 20261012,
        # Optional, expensive: retrain each final gene with held-out training
        # seeds to measure gene-level (training) reproducibility.
        "retrain_replicates": 0,
    },
    "aria": {
        "enabled": False,
        # Paper (Walker): n_samples 1440 (x2 mirrored), n_grad_steps 75,
        # sigma 0.01. Any smaller budget is reported as reduced.
        "n_samples": 1440,
        "mirrored": True,
        "n_grad_steps": 75,
        "sigma": 0.01,
        "optimizer": "adam",       # not specified in the paper text
        "learning_rate": 0.01,     # not specified in the paper text
        "adjacency": "von_neumann",  # not specified in the paper text
        "completion_phase": True,
        "max_completion_cells": None,  # None = every reachable empty cell
        "episodes_per_sample": 1,
    },
}

PAPER_ARIA_BUDGET = {"n_samples": 1440, "mirrored": True, "n_grad_steps": 75, "sigma": 0.01}


def _preset(**kwargs):
    cfg = copy.deepcopy(DEFAULTS)
    return _deep_merge(cfg, kwargs)


def _deep_merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# A reduced ARIA budget for post-hoc refinement on CPU MuJoCo. Every result
# produced with it is labelled "aria_reduced_budget".
_ARIA_REDUCED = {
    "enabled": True,
    "n_samples": 16,
    "mirrored": True,
    "n_grad_steps": 10,
    "sigma": 0.01,
    "max_completion_cells": 32,
}

PRESETS = {
    # ---------------------------------------------------------------- legacy
    "legacy_baseline": _preset(
        name="legacy_baseline",
        category="legacy",
        archive_strategy="legacy_map_elites",
        description=(
            "Unchanged pipeline: one TD3 run per gene with the shared "
            "MUJOCO_EVAL_SEED, 10 episodes with reset seeds seed+0..9, one "
            "elite per cell, strict-improvement replacement."),
    ),
    # ------------------------------------------------------------ controlled
    "controlled_baseline": _preset(
        name="controlled_baseline",
        category="controlled",
        archive_strategy="map_elites",
        description=(
            "MAP-Elites under the controlled protocol: 10-episode initial "
            "evaluation with per-solution rollout seeds, run-seed training "
            "seeds, no re-evaluation. The reference for controlled ablations."),
    ),
    "controlled_me_sampling_n32": _preset(
        name="controlled_me_sampling_n32",
        category="controlled",
        archive_strategy="map_elites",
        initial_samples=32,
        description="MAP-Elites-sampling with N=32 episodes per new policy.",
    ),
    "controlled_archive_sampling": _preset(
        name="controlled_archive_sampling",
        category="controlled",
        archive_strategy="archive_sampling",
        depth=2,
        archive_reeval={"enabled": True, "every": 1, "samples": 1, "start_generation": 1},
        description=(
            "Archive-sampling mechanism (re-evaluate every archived policy "
            "once per generation, keep the D=2 best per cell) with the shared "
            "10-episode initial evaluation."),
    ),
    "controlled_parallel_adaptive_sampling": _preset(
        name="controlled_parallel_adaptive_sampling",
        category="controlled",
        archive_strategy="parallel_adaptive_sampling",
        depth=2,
        offspring_sampling="pas_median",
        pas_min_samples=10,
        archive_reeval={"enabled": True, "every": 1, "samples": 1, "start_generation": 1},
        description=(
            "Parallel-Adaptive-sampling mechanism; offspring get the median "
            "archive sample count, floored at the shared 10-episode protocol."),
    ),
    "controlled_deep_grid_sampling_n10": _preset(
        name="controlled_deep_grid_sampling_n10",
        category="controlled",
        archive_strategy="deep_grid",
        depth=32,
        deep_grid_replacement="random",
        description=(
            "Deep-Grid (D=32, random replacement, fitness-proportional "
            "in-cell selection) with the shared 10-episode initial evaluation, "
            "i.e. Deep-Grid-sampling with N=10."),
    ),
    # Combinations with post-hoc ARIA (reduced budget).
    "controlled_baseline_aria": _preset(
        name="controlled_baseline_aria",
        category="controlled",
        archive_strategy="map_elites",
        aria=_ARIA_REDUCED,
        description=(
            "controlled_baseline followed by post-hoc ARIA on frozen actor "
            "weights (ARIA-ME-init analogue), reduced budget."),
    ),
    "controlled_archive_sampling_aria": _preset(
        name="controlled_archive_sampling_aria",
        category="controlled",
        archive_strategy="archive_sampling",
        depth=2,
        archive_reeval={"enabled": True, "every": 1, "samples": 1, "start_generation": 1},
        aria=_ARIA_REDUCED,
        description=(
            "controlled_archive_sampling followed by post-hoc ARIA, reduced "
            "budget: tests whether a reproducibility-aware archive is a "
            "better ARIA starting point."),
    ),
    # ----------------------------------------------------------------- paper
    "paper_map_elites": _preset(
        name="paper_map_elites",
        category="paper",
        archive_strategy="map_elites",
        initial_samples=1,
        budget={"mode": "sampling_size", "sampling_size": 320},
        description="MAP-Elites, one episode per offspring (UQD baseline).",
    ),
    "paper_me_sampling": _preset(
        name="paper_me_sampling",
        category="paper",
        archive_strategy="map_elites",
        initial_samples=32,
        budget={"mode": "sampling_size", "sampling_size": 320},
        description="MAP-Elites-sampling, N=32 (UQD App. F-A best value).",
    ),
    "paper_deep_grid": _preset(
        name="paper_deep_grid",
        category="paper",
        archive_strategy="deep_grid",
        initial_samples=1,
        depth=32,
        deep_grid_replacement="random",
        budget={"mode": "sampling_size", "sampling_size": 320},
        description="Deep-Grid, D=32 (UQD App. F-B), one episode per offspring.",
    ),
    "paper_deep_grid_sampling": _preset(
        name="paper_deep_grid_sampling",
        category="paper",
        archive_strategy="deep_grid",
        initial_samples=8,
        depth=32,
        deep_grid_replacement="random",
        budget={"mode": "sampling_size", "sampling_size": 320},
        description="Deep-Grid-sampling, D=32, N=8 (UQD App. F-C).",
    ),
    "paper_archive_sampling": _preset(
        name="paper_archive_sampling",
        category="paper",
        archive_strategy="archive_sampling",
        initial_samples=1,
        depth=2,
        archive_reeval={"enabled": True, "every": 1, "samples": 1, "start_generation": 1},
        budget={"mode": "sampling_size", "sampling_size": 320},
        description="Archive-sampling, D=2 (UQD Alg. 1, App. F-D).",
    ),
    "paper_parallel_adaptive_sampling": _preset(
        name="paper_parallel_adaptive_sampling",
        category="paper",
        archive_strategy="parallel_adaptive_sampling",
        initial_samples=1,
        depth=2,
        offspring_sampling="pas_median",
        archive_reeval={"enabled": True, "every": 1, "samples": 1, "start_generation": 1},
        budget={"mode": "sampling_size", "sampling_size": 320},
        description="Parallel-Adaptive-sampling, D=2 (UQD Alg. 2, App. F-E).",
    ),
    # ---------------------------------------------------------- experimental
    "exp_me_sampling_retrain_n3": _preset(
        name="exp_me_sampling_retrain_n3",
        category="experimental",
        archive_strategy="map_elites",
        sample_unit="retrain",
        initial_samples=3,
        episodes_per_replicate=10,
        description=(
            "MAP-Elites-sampling over training seeds: each gene is trained 3 "
            "times and scored by the mean of its per-run 10-episode means."),
    ),
    "exp_map_elites_periodic_reeval": _preset(
        name="exp_map_elites_periodic_reeval",
        category="experimental",
        archive_strategy="map_elites",
        archive_reeval={"enabled": True, "every": 5, "samples": 10, "start_generation": 5},
        description=(
            "MAP-Elites with 10 extra episodes for every elite every 5 "
            "generations; elites whose estimate moves are relocated."),
    ),
    "exp_map_elites_margin": _preset(
        name="exp_map_elites_margin",
        category="experimental",
        archive_strategy="map_elites",
        replacement={"rule": "margin", "margin": 25.0},
        description=(
            "MAP-Elites where a challenger must beat the elite by 25 return "
            "units. A heuristic noise floor, not a statistical test."),
    ),
}


class UQDConfigError(ValueError):
    """Raised when a UQD configuration is invalid or internally inconsistent."""


def resolve_uqd_config(preset_name, overrides_json="", population_size=None):
    """Return the validated configuration for ``preset_name``.

    ``overrides_json`` is a JSON object merged on top of the preset. Overriding
    a legacy, controlled or paper preset relabels it as experimental (with an
    ``exp_`` name) unless the override only touches final-evaluation or ARIA
    settings, because a modified preset no longer matches what its name claims.
    """
    if preset_name not in PRESETS:
        raise UQDConfigError(
            f"Unknown UQD preset {preset_name!r}. Choose one of: {', '.join(sorted(PRESETS))}")
    cfg = copy.deepcopy(PRESETS[preset_name])
    overrides = {}
    if overrides_json:
        try:
            overrides = json.loads(overrides_json)
        except json.JSONDecodeError as exc:
            raise UQDConfigError(f"MUJOCO_UQD_OVERRIDES is not valid JSON: {exc}") from exc
        if not isinstance(overrides, dict):
            raise UQDConfigError("MUJOCO_UQD_OVERRIDES must be a JSON object")
        _reject_unknown_keys(overrides, DEFAULTS, path="")
        for key in ("name", "category"):
            if key in overrides:
                raise UQDConfigError(f"'{key}' cannot be overridden; it is set by the preset")
        cfg = _deep_merge(cfg, overrides)
        search_keys = set(overrides) - {"final_eval", "aria", "description"}
        if search_keys and cfg["category"] != "experimental":
            cfg["category"] = "experimental"
            cfg["name"] = f"exp_{preset_name}_override"
    cfg["overrides"] = overrides
    if (cfg["archive_strategy"] != "legacy_map_elites"
            and cfg["budget"]["offspring_per_generation"] is None
            and population_size is not None):
        cfg["budget"]["offspring_per_generation"] = int(population_size)
    validate_uqd_config(cfg)
    return cfg


def _reject_unknown_keys(update, reference, path):
    for key, value in update.items():
        if key not in reference:
            raise UQDConfigError(f"Unknown UQD setting {path}{key!r}")
        if isinstance(value, dict) and isinstance(reference[key], dict):
            _reject_unknown_keys(value, reference[key], path=f"{path}{key}.")


def _require(condition, message):
    if not condition:
        raise UQDConfigError(message)


def _positive_int(value, name, allow_none=False):
    if value is None and allow_none:
        return
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 1,
             f"{name} must be a positive integer, got {value!r}")


def validate_uqd_config(cfg):
    """Fail loudly on any setting that would be ignored or contradictory."""
    _require(cfg["category"] in CATEGORIES, f"category must be one of {CATEGORIES}")
    strategy = cfg["archive_strategy"]
    _require(strategy in ARCHIVE_STRATEGIES, f"archive_strategy must be one of {ARCHIVE_STRATEGIES}")
    _require(cfg["sample_unit"] in SAMPLE_UNITS, f"sample_unit must be one of {SAMPLE_UNITS}")
    _require(cfg["offspring_sampling"] in OFFSPRING_SAMPLING,
             f"offspring_sampling must be one of {OFFSPRING_SAMPLING}")
    _require(cfg["pas_count_statistic"] in PAS_COUNT_STATISTICS,
             f"pas_count_statistic must be one of {PAS_COUNT_STATISTICS}")
    _require(cfg["deep_grid_replacement"] in DEEP_GRID_REPLACEMENT,
             f"deep_grid_replacement must be one of {DEEP_GRID_REPLACEMENT}")
    _positive_int(cfg["initial_samples"], "initial_samples")
    _positive_int(cfg["depth"], "depth")
    _positive_int(cfg["sample_window"], "sample_window", allow_none=True)
    _positive_int(cfg["pas_min_samples"], "pas_min_samples")
    _positive_int(cfg["pas_max_samples"], "pas_max_samples", allow_none=True)

    if strategy == "legacy_map_elites":
        legacy = copy.deepcopy(PRESETS["legacy_baseline"])
        for key in ("sample_unit", "initial_samples", "depth", "archive_reeval",
                    "replacement", "offspring_sampling", "budget"):
            _require(cfg[key] == legacy[key],
                     f"legacy_map_elites runs the original code path; '{key}' cannot be changed. "
                     "Use controlled_baseline for a configurable MAP-Elites.")
        _require(cfg["category"] == "legacy", "legacy_map_elites is only valid for the legacy preset")
        _require(not cfg["aria"]["enabled"],
                 "ARIA needs the engine's checkpoint registry; use controlled_baseline_aria")

    reeval = cfg["archive_reeval"]
    _positive_int(reeval["every"], "archive_reeval.every")
    _positive_int(reeval["samples"], "archive_reeval.samples")
    _positive_int(reeval["start_generation"], "archive_reeval.start_generation")

    rule = cfg["replacement"]["rule"]
    _require(rule in REPLACEMENT_RULES, f"replacement.rule must be one of {REPLACEMENT_RULES}")
    if rule == "margin":
        _require(cfg["category"] == "experimental",
                 "replacement.rule='margin' is an experimental heuristic; use an exp_ preset")
        _require(strategy == "map_elites", "replacement margin only applies to map_elites insertion")
        _require(float(cfg["replacement"]["margin"]) > 0, "replacement.margin must be > 0")
    else:
        _require(float(cfg["replacement"]["margin"]) == 0.0,
                 "replacement.margin is set but replacement.rule is not 'margin'")

    unit = cfg["sample_unit"]
    if unit == "retrain":
        _positive_int(cfg["episodes_per_replicate"], "episodes_per_replicate")
        _require(strategy == "map_elites",
                 "sample_unit='retrain' is only supported for map_elites (MAP-Elites-sampling over "
                 "training seeds); re-training every archived gene each generation is not supported")
        _require(not reeval["enabled"], "archive re-evaluation is not supported with sample_unit='retrain'")
        _require(cfg["offspring_sampling"] == "fixed", "pas_median needs sample_unit='rollout'")
        _require(not cfg["aria"]["enabled"],
                 "ARIA refines one frozen checkpoint; it is not defined for retrain-sampled genes")
    else:
        _require(cfg["episodes_per_replicate"] is None,
                 "episodes_per_replicate only applies to sample_unit='retrain'")

    if strategy == "map_elites":
        _require(cfg["depth"] == 1, "map_elites keeps one elite per cell; depth must be 1")
        _require(cfg["offspring_sampling"] == "fixed", "pas_median is only defined for parallel_adaptive_sampling")
    if strategy in ("archive_sampling", "parallel_adaptive_sampling"):
        _require(reeval["enabled"], f"{strategy} re-evaluates the archive every generation; "
                                    "archive_reeval.enabled must be true")
        if cfg["category"] in ("paper", "controlled"):
            _require(reeval["every"] == 1 and reeval["samples"] == 1,
                     f"{strategy} (UQD Alg. 1/2) re-evaluates each stored solution once per generation; "
                     "other schedules need an exp_ preset")
    if strategy == "archive_sampling":
        _require(cfg["offspring_sampling"] == "fixed",
                 "archive_sampling evaluates offspring a fixed number of times")
    if strategy == "parallel_adaptive_sampling":
        _require(cfg["offspring_sampling"] == "pas_median",
                 "parallel_adaptive_sampling requires offspring_sampling='pas_median'")
    else:
        _require(cfg["pas_max_samples"] is None and cfg["pas_min_samples"] == 1,
                 "pas_min_samples/pas_max_samples only apply to parallel_adaptive_sampling")
    if strategy == "deep_grid":
        _require(not reeval["enabled"], "Deep-Grid does not re-evaluate archived solutions")
        _require(cfg["depth"] >= 2, "deep_grid needs depth >= 2")
    else:
        _require(cfg["deep_grid_replacement"] == "random",
                 "deep_grid_replacement only applies to archive_strategy='deep_grid'")
    if strategy == "map_elites" and reeval["enabled"]:
        _require(cfg["category"] == "experimental",
                 "periodic re-evaluation of a MAP-Elites archive is our own variant; use an exp_ preset")

    budget = cfg["budget"]
    _require(budget["mode"] in BUDGET_MODES, f"budget.mode must be one of {BUDGET_MODES}")
    _positive_int(budget["max_offspring_per_generation"], "budget.max_offspring_per_generation", allow_none=True)
    _positive_int(budget["total_env_steps"], "budget.total_env_steps", allow_none=True)
    if budget["mode"] == "sampling_size":
        _positive_int(budget["sampling_size"], "budget.sampling_size")
        _require(unit == "rollout", "the sampling-size budget counts episodes; use sample_unit='rollout'")
    else:
        _require(budget["sampling_size"] is None,
                 "budget.sampling_size is set but budget.mode is 'fixed_offspring'")
        _positive_int(budget["offspring_per_generation"], "budget.offspring_per_generation", allow_none=True)
    if cfg["category"] == "paper":
        _require(budget["mode"] == "sampling_size", "paper presets use the UQD sampling-size budget")
    if cfg["category"] == "controlled":
        _require(budget["mode"] == "fixed_offspring", "controlled presets fix the offspring count")

    infra = cfg["infra"]
    _require(isinstance(infra["max_retries"], int) and infra["max_retries"] >= 0,
             "infra.max_retries must be a non-negative integer")
    _require(infra["on_reeval_failure"] in REEVAL_FAILURE_POLICIES,
             f"infra.on_reeval_failure must be one of {REEVAL_FAILURE_POLICIES}")

    final = cfg["final_eval"]
    _positive_int(final["episodes"], "final_eval.episodes")
    _require(final["fitness_estimator"] in FINAL_ESTIMATORS, f"final_eval.fitness_estimator in {FINAL_ESTIMATORS}")
    _require(final["descriptor_estimator"] in FINAL_ESTIMATORS,
             f"final_eval.descriptor_estimator in {FINAL_ESTIMATORS}")
    lo, hi = final["qd_score_bounds"]
    _require(float(hi) > float(lo), "final_eval.qd_score_bounds must be [low, high] with high > low")
    _require(isinstance(final["retrain_replicates"], int) and final["retrain_replicates"] >= 0,
             "final_eval.retrain_replicates must be a non-negative integer")

    aria = cfg["aria"]
    if aria["enabled"]:
        _positive_int(aria["n_samples"], "aria.n_samples")
        _positive_int(aria["n_grad_steps"], "aria.n_grad_steps")
        _positive_int(aria["episodes_per_sample"], "aria.episodes_per_sample")
        _positive_int(aria["max_completion_cells"], "aria.max_completion_cells", allow_none=True)
        _require(float(aria["sigma"]) > 0, "aria.sigma must be > 0")
        _require(float(aria["learning_rate"]) > 0, "aria.learning_rate must be > 0")
        _require(aria["optimizer"] in ("adam", "sgd"), "aria.optimizer must be 'adam' or 'sgd'")
        _require(aria["adjacency"] in ARIA_ADJACENCY, f"aria.adjacency must be one of {ARIA_ADJACENCY}")
        _require(unit == "rollout", "ARIA refines frozen checkpoints; it needs sample_unit='rollout'")
    return cfg


def aria_budget_label(aria_cfg):
    """'aria_paper_budget' only when every budget setting matches the paper."""
    matches = all(aria_cfg.get(key) == value for key, value in PAPER_ARIA_BUDGET.items())
    matches = matches and aria_cfg.get("max_completion_cells") is None and aria_cfg.get("completion_phase", True)
    return "aria_paper_budget" if matches else "aria_reduced_budget"
