"""Training-cost cap of XQC genes (sota/XQC/train_xqc_gene.py, LLMGE_XQC_COST_CAP).

Standard library only:
    python -m pytest tests/xqc/test_cost_cap.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sota", "XQC"))
import train_xqc_gene as t  # noqa: E402

SEED = t.gene_values(t.SEED_GENE)

GENE = '''ALGO = "xqc"

def get_architecture_kwargs():
    return dict(hidden_dims_critic={critic}, hidden_dims_actor={actor}, n_critics=2)

def get_xqc_kwargs():
    return dict(actor_lr=3e-4, batch_size={batch}, updates_per_step={updates})
'''


def cost(**changes):
    return t.relative_cost({**SEED, **changes}, SEED)


def write_gene(tmp_path, critic=(512,) * 4, actor=(256,) * 4, batch=256, updates=2):
    path = tmp_path / "network_g.py"
    path.write_text(GENE.format(critic=list(critic), actor=list(actor), batch=batch, updates=updates))
    return str(path)


def test_seed_costs_one_and_omitted_keys_mean_seed_defaults():
    assert cost() == 1.0
    assert t.relative_cost({}, SEED) == 1.0


def test_cost_examples_given_to_the_llm():
    assert cost(updates_per_step=3) == pytest.approx(1.5)
    assert cost(batch_size=512) == pytest.approx(2.0)
    assert cost(hidden_dims_critic=[1024] * 4) == pytest.approx(3.5, abs=0.1)


def test_smaller_than_seed_still_costs_its_update_overhead():
    # Below the seed's size each update is overhead-bound: only updates_per_step counts.
    assert cost(batch_size=64, hidden_dims_critic=[128, 128]) == 1.0
    assert cost(batch_size=64, updates_per_step=4) == 2.0


@pytest.mark.parametrize("gene", [
    dict(batch_size=512, updates_per_step=3),                                  # 3x, 500k run median 3.56 h
    dict(hidden_dims_critic=[1024] * 4, hidden_dims_actor=[512] * 4),          # 3.9x, 1 of 13 finished
    dict(batch_size=1024, updates_per_step=4),
    dict(hidden_dims_critic=[4096] * 8, hidden_dims_actor=[4096] * 8, batch_size=2048, updates_per_step=4),
    dict(updates_per_step=4, batch_size=16, hidden_dims_critic=[16], hidden_dims_actor=[16]),
])
def test_fit_brings_any_gene_under_the_cap_and_keeps_it_valid(gene):
    for cap in (1.5, 2.0):
        values, changes = t.fit_cost_cap({**SEED, **gene}, SEED, cap)
        assert t.relative_cost(values, SEED) <= cap
        assert bool(changes) == (cost(**gene) > cap)
        for key, value in values.items():
            t.validate(key, value, t.AGENT_KEYS.get(key) or t.TOP_KEYS[key])


def test_fit_leaves_genes_within_the_cap_alone():
    gene = {**SEED, "batch_size": 384}
    assert t.fit_cost_cap(gene, SEED, 2.0) == (gene, {})


def test_gene_overrides_shrinks_only_when_the_cap_is_set(tmp_path, monkeypatch):
    path = write_gene(tmp_path, critic=[1024] * 4, batch=512, updates=3)
    monkeypatch.delenv("LLMGE_XQC_COST_CAP", raising=False)
    overrides, config = t.gene_overrides(path)
    assert "batch_size=512" in overrides and config["updates_per_step"] == 3

    monkeypatch.setenv("LLMGE_XQC_COST_CAP", "2")
    report = {}
    overrides, config = t.gene_overrides(path, report=report)
    assert t.relative_cost(config, SEED) <= 2.0
    assert report["cap"] == 2.0 and report["cost_before"] > 2.0 >= report["cost_after"]
    assert report["changed"]
    # The overrides train what the config says.
    assert f"batch_size={config['batch_size']}" in overrides
    assert f"updates_per_step={config['updates_per_step']}" in overrides


def test_gene_overrides_still_rejects_invalid_genes(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMGE_XQC_COST_CAP", "2")
    with pytest.raises(ValueError):
        t.gene_overrides(write_gene(tmp_path, batch=4096))
