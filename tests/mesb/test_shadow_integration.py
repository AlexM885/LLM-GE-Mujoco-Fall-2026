"""End-to-end driver tests with the mock LLM and mock evaluator (no MuJoCo/LLM).

* mesb-shadow must choose exactly the same parents as nsga2,
* mesb must draw parents only from occupied-cell elites,
* a run interrupted (even mid-generation) and resumed must match an
  uninterrupted run, without re-evaluating anything.
"""

import csv
import json
import sys

import pytest

import run_mesb
from sota.MujocoRL.mesb_logging import RunPaths


def _cfg(tmp_path, name, mode, gens=3, extra=()):
    argv = ["--output-root", str(tmp_path), "--run-name", name, "--selection-mode", mode,
            "--llm-backend", "mock", "--evaluator", "mock", "--llm-python", sys.executable,
            "--num-generations", str(gens), "--start-population-size", "12",
            "--population-size", "8", "--num-elites", "4", "--mesb-dims", "4", "4",
            "--mesb-remap-frequency", "7", "--seed", "11", "--workers", "2", *extra]
    return run_mesb.config_from_args(run_mesb.build_parser().parse_args(argv))


def _run(tmp_path, name, mode, gens=3, resume=False, extra=()):
    run_mesb.MESBRun(_cfg(tmp_path, name, mode, gens, extra)).run(resume=resume)
    return RunPaths(tmp_path / name)


def _jsonl(path):
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _csv(path, drop=()):
    with open(path, newline="") as fh:
        return [{k: v for k, v in r.items() if k not in drop} for r in csv.DictReader(fh)]


def test_shadow_does_not_change_parent_selection(tmp_path):
    pytest.importorskip("deap")
    legacy = _run(tmp_path, "legacy", "nsga2")
    shadow = _run(tmp_path, "shadow", "mesb-shadow")

    sel_l = [(r["generation"], r["parents"], r["elites"], r["slots"])
             for r in _jsonl(legacy.selection_log)]
    sel_s = [(r["generation"], r["parents"], r["elites"], r["slots"])
             for r in _jsonl(shadow.selection_log)]
    assert sel_l == sel_s and len(sel_l) == 4
    assert all(parents for g, parents, _, _ in sel_l if g > 0)

    ev_l = [r["gene_id"] for r in _csv(legacy.evaluations_csv)]
    ev_s = [r["gene_id"] for r in _csv(shadow.evaluations_csv)]
    assert ev_l == ev_s

    # The shadow archive exists and received every successful evaluation.
    ok = [r for r in _csv(shadow.evaluations_csv) if r["status"] == "success"]
    assert all(r["archive_cell_x"] != "" for r in ok)
    metrics = _csv(shadow.metrics_csv)
    assert int(metrics[-1]["archive_total_seen"]) == len(ok)
    assert int(metrics[-1]["occupied_cells"]) > 0
    assert shadow.boundaries_jsonl.exists() and shadow.final_archive_csv.exists()
    assert _jsonl(shadow.boundaries_jsonl)[0]["event"] == "initialize"
    # The legacy run has no archive at all.
    assert _csv(legacy.metrics_csv)[-1]["occupied_cells"] == ""
    assert not legacy.boundaries_jsonl.exists()


def test_failed_evaluations_are_logged_but_never_archived(tmp_path):
    pytest.importorskip("deap")
    run = _run(tmp_path, "shadow", "mesb-shadow")
    rows = _csv(run.evaluations_csv)
    failed = {r["gene_id"] for r in rows if r["status"] == "failed"}
    assert failed, "mock evaluator should produce some failures"
    assert all(r["archive_cell_x"] == "" for r in rows if r["gene_id"] in failed)
    for snap in run.snapshots.glob("gen_*.json"):
        assert not failed & {e["gene_id"] for e in json.loads(snap.read_text())["elites"]}


def test_mesb_parents_are_archive_elites(tmp_path):
    run = _run(tmp_path, "mesb", "mesb", gens=4)
    selections = {r["generation"]: r for r in _jsonl(run.selection_log)}
    for g in range(1, 5):
        prev = json.loads((run.snapshots / f"gen_{g - 1:04d}.json").read_text())
        elite_ids = {e["gene_id"] for e in prev["elites"]}
        parents = selections[g]["parents"]
        assert len(parents) == 8 and set(parents) <= elite_ids
        assert selections[g]["elites"] == []  # no NSGA/SPEA elitism in mesb mode
    metrics = _csv(run.metrics_csv)
    assert int(metrics[-1]["number_of_remaps"]) >= 1  # delta=7 is crossed
    events = _jsonl(run.boundaries_jsonl)
    init = next(e for e in events if e["event"] == "initialize")
    remaps = [e for e in events if e["event"] == "remap"]
    assert remaps and remaps[-1]["boundaries"] != init["boundaries"]
    assert remaps[-1]["total_seen"] > init["total_seen"]


def test_resume_matches_uninterrupted_run(tmp_path):
    straight = _run(tmp_path, "straight", "mesb", gens=4)
    _run(tmp_path, "resumed", "mesb", gens=2)
    resumed = _run(tmp_path, "resumed", "mesb", gens=4, resume=True)
    assert _csv(straight.evaluations_csv) == _csv(resumed.evaluations_csv)
    assert _csv(straight.metrics_csv) == _csv(resumed.metrics_csv)
    assert _jsonl(straight.boundaries_jsonl) == _jsonl(resumed.boundaries_jsonl)
    assert straight.final_archive_csv.read_text() == resumed.final_archive_csv.read_text()


def test_resume_after_mid_generation_crash(tmp_path):
    straight = _run(tmp_path, "straight", "mesb", gens=3)
    _run(tmp_path, "crash", "mesb", gens=2)

    # Simulate a crash in generation 3: plan made, only some genes evaluated.
    r = run_mesb.MESBRun(_cfg(tmp_path, "crash", "mesb", gens=3))
    r._make_components()
    r.load_checkpoint(r._latest_checkpoint())
    plan = r.make_plan(3)
    partial = plan.new_genes[:2]
    from sota.MujocoRL.mesb_evaluator import EvalJob
    r.evaluator.evaluate([EvalJob(g["gene_id"], r.variation.gene_file(g["gene_id"]), 3,
                                  g["train_seed"], r.paths.result_json(g["gene_id"]),
                                  r.paths.models, r.paths.eval_logs / "x.log") for g in partial])
    mtimes = {g["gene_id"]: r.paths.result_json(g["gene_id"]).stat().st_mtime_ns
              for g in partial}

    resumed = _run(tmp_path, "crash", "mesb", gens=3, resume=True)
    # Already-finished evaluations were reused, not repeated.
    assert all(resumed.result_json(g).stat().st_mtime_ns == t for g, t in mtimes.items())
    assert _csv(straight.evaluations_csv) == _csv(resumed.evaluations_csv)
    assert _csv(straight.metrics_csv) == _csv(resumed.metrics_csv)
    assert _jsonl(straight.selection_log) == [
        {k: v for k, v in e.items() if k != "reused_plan"} for e in _jsonl(resumed.selection_log)]


def test_resume_refuses_changed_experiment(tmp_path):
    _run(tmp_path, "run", "mesb", gens=1)
    with pytest.raises(run_mesb.CheckpointMismatchError):
        _run(tmp_path, "run", "mesb", gens=2, resume=True, extra=("--mesb-remap-frequency", "50"))


def test_existing_run_requires_resume_flag(tmp_path):
    _run(tmp_path, "run", "mesb", gens=0)
    with pytest.raises(SystemExit):
        _run(tmp_path, "run", "mesb", gens=1)


def test_zero_successful_initial_individuals_fails_loudly(tmp_path, monkeypatch):
    from sota.MujocoRL import mesb_evaluator
    from sota.MujocoRL.results_io import atomic_write_json, failure_payload

    def always_fail(self, job):
        atomic_write_json(job.out_json, failure_payload(job.gene_id, "boom"))
    monkeypatch.setattr(mesb_evaluator.MockEvaluator, "_run_one", always_fail)
    with pytest.raises(ValueError, match="zero succeeded"):
        _run(tmp_path, "dead", "mesb", gens=1)


def test_generation_end_remap_mode(tmp_path):
    run = _run(tmp_path, "genend", "mesb", gens=3, extra=("--mesb-remap-at-generation-end",))
    events = _jsonl(run.boundaries_jsonl)
    remaps = [e for e in events if e["event"] == "remap"]
    # exactly one remap per evolutionary generation (each adds new evaluations)
    assert [e["generation"] for e in remaps] == [1, 2, 3]


def test_fixed_grid_shadow_for_comparison(tmp_path):
    pytest.importorskip("deap")
    run = _run(tmp_path, "fixed", "mesb-shadow",
               extra=("--archive-boundaries", "fixed", "--fixed-ranges", "-60", "1500", "0", "500"))
    events = _jsonl(run.boundaries_jsonl)
    assert all(e["boundary_mode"] == "fixed" for e in events)
    assert len({json.dumps(e["boundaries"]) for e in events}) == 1  # never moves
