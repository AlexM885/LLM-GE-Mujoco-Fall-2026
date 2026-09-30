"""Parallel LLM calls give the same results as sequential ones; invalid
children are rejected and the parent is kept; empty replies are retried."""

import random
import sys
import types
from argparse import Namespace
from pathlib import Path

from src.mesb_variation import Variation, vary

REPO = Path(__file__).resolve().parents[2]


def _variation(tmp_path, name, workers):
    root = tmp_path / name
    return Variation(root / "genes", root / "prompts", root / "logs", backend="mock",
                     python_cmd=sys.executable, workers=workers)


def _run(tmp_path, name, workers):
    v = _variation(tmp_path, name, workers)
    random.seed(123)
    created = v.create_many(6)
    parents = [r.gene_id for r in created]
    slots, log = vary(parents, v, crossover_probability=0.6, mutation_probability=0.8)
    files = {p.name: p.read_text() for p in sorted(v.genes_dir.glob("*.py"))}
    summary = [(r.gene_id, r.operator, r.parents, r.ok, r.temperature, r.op_seed) for r in log]
    return [r.gene_id for r in created], slots, summary, files


def test_parallel_equals_sequential(tmp_path):
    seq = _run(tmp_path, "seq", workers=1)
    par = _run(tmp_path, "par", workers=6)
    assert seq == par
    assert any(op == "crossover" for _, op, *_ in seq[2])  # both phases exercised
    assert any(op == "mutation" for _, op, *_ in seq[2])


def test_seconds_recorded(tmp_path):
    v = _variation(tmp_path, "t", workers=2)
    random.seed(0)
    for r in v.create_many(2):
        assert r.ok and r.seconds is not None and r.seconds >= 0


def test_invalid_child_keeps_parent(tmp_path):
    v = _variation(tmp_path, "inv", workers=2)
    random.seed(1)
    parent = v.create_many(1)[0].gene_id
    pfile = v.gene_file(parent)
    pfile.write_text(pfile.read_text() + "\n# MOCK_INVALID_CHILD\n")
    slots, log = vary([parent, parent], v, crossover_probability=0.0, mutation_probability=1.0)
    assert slots == [parent, parent]                      # parents kept
    assert all(not r.ok and "INVALID CHILD" in r.error for r in log)
    assert all(not v.gene_file(r.gene_id).exists() for r in log)
    assert all(v.genes_dir.joinpath(f"network_{r.gene_id}.rejected.py").exists() for r in log)


def test_retry_on_empty_reply(monkeypatch):
    sys.path.insert(0, str(REPO / "src"))
    import mesb_llm_operator as op

    calls = []

    def generator(prompt, **kw):
        calls.append(prompt)
        return None if len(calls) == 1 else "```python\nx = 1\n```"

    fake_utils = types.SimpleNamespace(
        get_llm_code_generator=lambda model: (generator, None),
        clean_code_from_llm=lambda text: "original")
    monkeypatch.setitem(sys.modules, "llm_utils", fake_utils)
    waits = []
    monkeypatch.setattr(op, "wait_for_llm_server", lambda c, t: waits.append(t))
    args = Namespace(extraction="robust", llm_retries=3, server_timeout=5.0)
    op._install_guards(types.SimpleNamespace(LOCAL_LLM=True), args)

    gen, _ = fake_utils.get_llm_code_generator("llama3")
    assert gen("prompt", temperature=0.1) == "```python\nx = 1\n```"
    assert len(calls) == 2 and waits == [5.0]
    assert fake_utils.clean_code_from_llm("```python\ny = 2\n```") == "y = 2"


def test_retry_gives_up(monkeypatch):
    sys.path.insert(0, str(REPO / "src"))
    import mesb_llm_operator as op

    fake_utils = types.SimpleNamespace(get_llm_code_generator=lambda m: (lambda p, **k: None, None),
                                       clean_code_from_llm=lambda t: t)
    monkeypatch.setitem(sys.modules, "llm_utils", fake_utils)
    monkeypatch.setattr(op, "wait_for_llm_server", lambda c, t: None)
    op._install_guards(types.SimpleNamespace(LOCAL_LLM=True),
                       Namespace(extraction="original", llm_retries=2, server_timeout=1.0))
    gen, _ = fake_utils.get_llm_code_generator("x")
    assert gen("p") is None
    assert fake_utils.clean_code_from_llm("keep") == "keep"   # --extraction original
