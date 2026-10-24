"""One generation of an engine-driven uncertain-QD search.

Used by scripts/uqd_smoke.py (no LLM) and by the tests. run_improved.py
follows the same order but keeps its own job submission for training.
"""

from .executors import run_rollouts_with_retries, run_train_specs_with_retries


def _train(engine, executor, specs, generation, train_kwargs):
    return run_train_specs_with_retries(engine, executor, specs, generation,
                                        engine.cfg["infra"]["max_retries"], **train_kwargs)


def initial_generation(engine, executor, genes, train_kwargs):
    """Evaluate and insert the initial population (generation 0)."""
    plan = engine.plan_generation(0, initial=True)
    genes = list(genes)[:plan["offspring"]]
    specs = []
    for gene in genes:
        specs.extend(engine.register_offspring(gene, 0, plan["samples_per_offspring"]))
    _train(engine, executor, specs, 0, train_kwargs)
    engine.insert_offspring(genes, 0)
    return engine.end_generation(0)


def run_generation(engine, executor, generation, make_children, train_kwargs):
    """make_children(parent_gene_ids, generation) -> list of new gene ids."""
    plan = engine.plan_generation(generation)
    requests = engine.reeval_requests(generation)
    if requests:
        run_rollouts_with_retries(engine, executor, requests, generation,
                                  engine.cfg["infra"]["max_retries"], label=f"reeval_g{generation}")
        engine.apply_reevaluation(generation)
    parents = engine.select_parents(plan["offspring"])
    children = make_children(parents, generation)
    specs = []
    for gene in children:
        specs.extend(engine.register_offspring(gene, generation, plan["samples_per_offspring"]))
    _train(engine, executor, specs, generation, train_kwargs)
    engine.insert_offspring(children, generation)
    return engine.end_generation(generation)
