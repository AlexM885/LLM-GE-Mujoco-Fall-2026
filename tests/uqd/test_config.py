import json

import pytest

from src.cfg.uqd_presets import (PRESETS, UQDConfigError, aria_budget_label, resolve_uqd_config,
                                 validate_uqd_config)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_resolves(name):
    cfg = resolve_uqd_config(name, population_size=32)
    assert cfg["name"] == name
    assert cfg["overrides"] == {}


def test_preset_name_prefixes_match_categories():
    for name, cfg in PRESETS.items():
        assert name.startswith({"legacy": "legacy_", "controlled": "controlled_", "paper": "paper_",
                                "experimental": "exp_"}[cfg["category"]]), name


def test_legacy_preset_is_untouched_by_population_size():
    cfg = resolve_uqd_config("legacy_baseline", population_size=32)
    assert cfg["archive_strategy"] == "legacy_map_elites"
    assert cfg["budget"]["offspring_per_generation"] is None


def test_legacy_rejects_search_changes():
    with pytest.raises(UQDConfigError):
        resolve_uqd_config("legacy_baseline", json.dumps({"initial_samples": 32}))


def test_unknown_preset_and_keys_fail():
    with pytest.raises(UQDConfigError, match="Unknown UQD preset"):
        resolve_uqd_config("nope")
    with pytest.raises(UQDConfigError, match="Unknown UQD setting"):
        resolve_uqd_config("controlled_baseline", json.dumps({"archive_reeval": {"evry": 2}}))
    with pytest.raises(UQDConfigError, match="not valid JSON"):
        resolve_uqd_config("controlled_baseline", "{bad")


def test_search_override_relabels_as_experimental():
    cfg = resolve_uqd_config("controlled_baseline", json.dumps({"initial_samples": 20}), 32)
    assert cfg["category"] == "experimental"
    assert cfg["name"] == "exp_controlled_baseline_override"
    # final-eval-only overrides keep the preset's identity
    cfg = resolve_uqd_config("controlled_baseline", json.dumps({"final_eval": {"episodes": 8}}), 32)
    assert cfg["name"] == "controlled_baseline"


@pytest.mark.parametrize("name,override,message", [
    ("controlled_baseline", {"depth": 2}, "depth must be 1"),
    ("controlled_baseline", {"replacement": {"rule": "margin", "margin": 5.0}}, None),  # relabelled -> ok
    ("controlled_deep_grid_sampling_n10", {"archive_reeval": {"enabled": True}}, "does not re-evaluate"),
    ("controlled_archive_sampling", {"archive_reeval": {"enabled": False}}, "must be true"),
    ("controlled_parallel_adaptive_sampling", {"offspring_sampling": "fixed"}, "pas_median"),
    ("controlled_baseline", {"sample_unit": "retrain"}, "episodes_per_replicate"),
    ("controlled_archive_sampling", {"sample_unit": "retrain", "episodes_per_replicate": 5}, "only supported"),
    ("controlled_baseline", {"budget": {"sampling_size": 100}}, "fixed_offspring"),
    ("controlled_baseline", {"deep_grid_replacement": "oldest"}, "only applies"),
    ("controlled_baseline", {"pas_max_samples": 4}, "only apply"),
    ("controlled_baseline", {"replacement": {"margin": 3.0}}, "rule is not 'margin'"),
    ("controlled_baseline", {"aria": {"enabled": True, "sigma": 0}}, "sigma"),
    ("exp_me_sampling_retrain_n3", {"aria": {"enabled": True}}, "not defined for retrain"),
    ("paper_archive_sampling", {"budget": {"mode": "fixed_offspring", "sampling_size": None}}, None),
])
def test_invalid_combinations_fail(name, override, message):
    if message is None:
        resolve_uqd_config(name, json.dumps(override), 32)  # becomes a valid exp_ variant
        return
    with pytest.raises(UQDConfigError, match=message):
        resolve_uqd_config(name, json.dumps(override), 32)


def test_paper_and_controlled_schedules_are_enforced():
    cfg = resolve_uqd_config("paper_archive_sampling", population_size=32)
    cfg["archive_reeval"]["samples"] = 3
    with pytest.raises(UQDConfigError, match="once per generation"):
        validate_uqd_config(cfg)
    cfg = resolve_uqd_config("controlled_baseline", population_size=32)
    cfg["replacement"] = {"rule": "margin", "margin": 5.0}
    with pytest.raises(UQDConfigError, match="experimental heuristic"):
        validate_uqd_config(cfg)


def test_aria_budget_label():
    assert aria_budget_label(PRESETS["controlled_baseline_aria"]["aria"]) == "aria_reduced_budget"
    paper = dict(PRESETS["controlled_baseline_aria"]["aria"], n_samples=1440, n_grad_steps=75,
                 sigma=0.01, mirrored=True, max_completion_cells=None)
    assert aria_budget_label(paper) == "aria_paper_budget"
