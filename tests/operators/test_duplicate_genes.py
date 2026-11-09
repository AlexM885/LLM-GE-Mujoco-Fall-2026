"""Duplicate genes reuse an identical gene's result instead of training again.

The functions under test are taken from run_improved.py's source and run
against stand-in globals, so the test needs no LLM server, SLURM or config.
"""

import ast
import os
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FUNCTIONS = ("count_successful_evals", "gene_signature", "has_real_result", "find_original",
             "reuse_result", "submit_or_reuse", "check_duplicate")
SENTINEL = -999999.0

XQC_GENE = '''# --PROMPT LOG--
ALGO = "xqc"

def get_architecture_kwargs():
    return dict(hidden_dims_critic=[512, 512, 512], hidden_dims_actor=[256, 256, 256],
                n_critics=2, use_batch_norm=True)

def get_xqc_kwargs():
    return dict(actor_lr=3e-4, critic_lr=3e-4, batch_size=256, updates_per_step={ups})
'''
# Same settings, other formatting, comments and key order.
XQC_GENE_REWRITTEN = '''ALGO = "xqc"
# a different comment
def get_xqc_kwargs():
    return {{"updates_per_step": {ups}, "batch_size": 256, "critic_lr": 0.0003, "actor_lr": 0.0003}}

def get_architecture_kwargs():
    return {{"use_batch_norm": True, "n_critics": 2,
            "hidden_dims_actor": [256, 256, 256], "hidden_dims_critic": [512, 512, 512]}}
'''


@pytest.fixture
def ri(tmp_path):
    """run_improved's duplicate functions with stand-in globals."""
    source = (ROOT / "run_improved.py").read_text(encoding="utf-8")
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    assert {n.name for n in nodes} == set(FUNCTIONS)
    (tmp_path / "models").mkdir()
    (tmp_path / "results").mkdir()
    submitted = []
    ns = dict(os=os, time=time, GLOBAL_DATA={}, GLOBAL_DATA_HIST={}, VARIANT_DIR=str(tmp_path / "models"),
              MODEL="network", SOTA_ROOT=str(tmp_path), ROOT_DIR=str(ROOT), SEED="xqc",
              EVAL_RUNLINE="python {} -network models.{} -timesteps 500000 -num_seeds 3",
              USE_VLM=False, VLM_AUX_REWARD_FILE=None, MAP_ELITES_OBJECTIVE="mean_reward",
              FAILED_EVAL_SENTINEL=SENTINEL, INVALID_FITNESS_MAX=(float("-inf"),),
              REUSE_DUPLICATES=True, _XQC_GENE_TRAINER=None,
              submit_run=lambda g: (submitted.append(g), ns["GLOBAL_DATA"][g].update(status="running eval")))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "run_improved.py", "exec"), ns)
    ns["submitted"] = submitted

    def add(gene_id, text):
        (tmp_path / "models" / f"network_{gene_id}.py").write_text(text)
        ns["GLOBAL_DATA"][gene_id] = {"status": "subbed file", "sub_flag": True, "fitness": None,
                                      "start_time": time.time()}

    def finish(gene_id, reward):
        (tmp_path / "results" / f"{gene_id}_results.csv").write_text(f"mean_reward,std_reward\n{reward},1.0\n")
        ns["GLOBAL_DATA"][gene_id].update(status="completed", fitness=(reward, 1.0),
                                          metrics={"mean_reward": reward, "std_reward": 1.0})

    ns["add"], ns["finish"], ns["tmp"] = add, finish, tmp_path
    return ns


def test_signature_ignores_formatting_but_not_settings(ri):
    ri["add"]("a", XQC_GENE.format(ups=2))
    ri["add"]("b", XQC_GENE_REWRITTEN.format(ups=2))
    ri["add"]("c", XQC_GENE.format(ups=3))
    sig = ri["gene_signature"]
    assert sig("a") is not None
    assert sig("a") == sig("b")
    assert sig("a") != sig("c")


def test_invalid_gene_has_no_signature(ri):
    ri["add"]("bad", 'ALGO = "xqc"\ndef get_architecture_kwargs():\n    return dict(n_critics=5)\n'
                     'def get_xqc_kwargs():\n    return {}\n')
    assert ri["gene_signature"]("bad") is None
    ri["submit_or_reuse"]("bad")
    assert ri["submitted"] == ["bad"]


def test_reuses_a_finished_identical_gene(ri):
    ri["add"]("a", XQC_GENE.format(ups=2))
    ri["submit_or_reuse"]("a")
    ri["finish"]("a", 6173.6)
    ri["add"]("b", XQC_GENE_REWRITTEN.format(ups=2))
    ri["submit_or_reuse"]("b")
    b = ri["GLOBAL_DATA"]["b"]
    assert ri["submitted"] == ["a"]
    assert (b["status"], b["fitness"], b["duplicate_of"]) == ("completed", (6173.6, 1.0), "a")
    assert (ri["tmp"] / "results" / "b_results.csv").read_text() == \
           (ri["tmp"] / "results" / "a_results.csv").read_text()
    # Training budget: only the gene that actually trained counts.
    assert ri["count_successful_evals"](ri["GLOBAL_DATA"]) == 1


def test_waits_for_an_identical_gene_still_training(ri):
    ri["add"]("a", XQC_GENE.format(ups=2))
    ri["add"]("b", XQC_GENE.format(ups=2))
    ri["submit_or_reuse"]("a")
    ri["submit_or_reuse"]("b")
    assert ri["submitted"] == ["a"]
    assert ri["GLOBAL_DATA"]["b"]["status"] == "running eval"
    assert ri["check_duplicate"]("b") is False          # a still training
    ri["finish"]("a", 5914.5)
    assert ri["check_duplicate"]("b") is True
    assert ri["GLOBAL_DATA"]["b"]["fitness"] == (5914.5, 1.0)
    assert ri["submitted"] == ["a"]


def test_trains_the_copy_when_the_original_fails(ri):
    ri["add"]("a", XQC_GENE.format(ups=2))
    ri["add"]("b", XQC_GENE.format(ups=2))
    ri["submit_or_reuse"]("a")
    ri["submit_or_reuse"]("b")
    ri["finish"]("a", SENTINEL)                        # e.g. a node failure
    assert ri["check_duplicate"]("b") is True
    assert ri["submitted"] == ["a", "b"]
    assert "duplicate_of" not in ri["GLOBAL_DATA"]["b"]
    # A third copy waits on b, now training, not on the failed a.
    ri["add"]("c", XQC_GENE.format(ups=2))
    ri["submit_or_reuse"]("c")
    assert ri["GLOBAL_DATA"]["c"]["duplicate_of"] == "b"
    assert ri["submitted"] == ["a", "b"]


def test_older_generations_count(ri):
    ri["add"]("a", XQC_GENE.format(ups=2))
    ri["submit_or_reuse"]("a")
    ri["finish"]("a", 6000.0)
    ri["GLOBAL_DATA_HIST"]["a"] = ri["GLOBAL_DATA"].pop("a")   # a previous generation
    ri["add"]("b", XQC_GENE.format(ups=2))
    ri["submit_or_reuse"]("b")
    assert ri["GLOBAL_DATA"]["b"]["duplicate_of"] == "a"
    assert ri["submitted"] == ["a"]


def test_switch_off_trains_everything(ri):
    ri["REUSE_DUPLICATES"] = False
    ri["add"]("a", XQC_GENE.format(ups=2))
    ri["add"]("b", XQC_GENE.format(ups=2))
    ri["submit_or_reuse"]("a")
    ri["finish"]("a", 6000.0)
    ri["submit_or_reuse"]("b")
    assert ri["submitted"] == ["a", "b"]
