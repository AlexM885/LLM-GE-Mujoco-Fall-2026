"""Result parsing: failed / malformed / non-finite evaluations never reach MESB."""

import json

import pytest

from sota.MujocoRL.mesb_archive import SlidingBoundariesArchive
from sota.MujocoRL.results_io import atomic_write_json, failure_payload, read_outcome


def _success(gene_id="g", **over):
    p = {"gene_id": gene_id, "status": "success", "generation": 0, "mean_reward": 12.5,
         "std_reward": 1.0, "param_count": 1234, "model_path": "m.zip", "rewards": [12.5],
         "behavior": {"mean_distance": 3.0, "mean_control_cost": 0.4,
                      "mean_control_cost_per_step": 0.0004, "distances": [3.0],
                      "control_costs": [0.4]}}
    p.update(over)
    return p


def test_success_becomes_archive_record(tmp_path):
    atomic_write_json(tmp_path / "g.json", _success())
    out = read_outcome(tmp_path / "g.json", "g")
    assert out.ok
    r = out.to_archive_record(generation=3)
    assert r.objective == 12.5 and r.measures == (3.0, 0.4) and r.param_count == 1234
    assert r.generation == 3


@pytest.mark.parametrize("payload, reason", [
    (failure_payload("g", "RuntimeError: boom", stage="train"), "status='failed'"),
    (_success(mean_reward=float("nan")), "non-finite mean_reward"),
    (_success(mean_reward=float("inf")), "non-finite mean_reward"),
    (_success(behavior={"mean_distance": float("nan"), "mean_control_cost": 1.0}),
     "non-finite behaviour descriptor mean_distance"),
    (_success(behavior={"mean_distance": 1.0, "mean_control_cost": float("-inf")}),
     "non-finite behaviour descriptor mean_control_cost"),
    (_success(behavior={"mean_distance": 1.0}), "missing behaviour descriptor"),
    ({k: v for k, v in _success().items() if k != "behavior"}, "missing fields"),
    (_success(gene_id="other"), "gene_id mismatch"),
    (_success(mean_reward="12"), "non-finite mean_reward"),
])
def test_invalid_payloads(tmp_path, payload, reason):
    atomic_write_json(tmp_path / "g.json", payload)
    out = read_outcome(tmp_path / "g.json", "g")
    assert not out.ok and reason in out.reason
    with pytest.raises(ValueError):
        out.to_archive_record(0)


def test_malformed_and_missing_files(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    out = read_outcome(tmp_path / "bad.json", "bad")
    assert not out.ok and "malformed" in out.reason
    out = read_outcome(tmp_path / "missing.json", "missing")
    assert not out.ok and "missing" in out.reason


def test_nan_survives_json_and_is_rejected(tmp_path):
    atomic_write_json(tmp_path / "g.json", _success(mean_reward=float("nan")))
    assert "NaN" in (tmp_path / "g.json").read_text()
    assert not read_outcome(tmp_path / "g.json", "g").ok


def test_only_successful_outcomes_reach_archive(tmp_path):
    payloads = {"a": _success("a"), "b": failure_payload("b", "boom"),
                "c": _success("c", mean_reward=float("nan")),
                "d": _success("d", mean_reward=3.0, behavior={"mean_distance": -1.0,
                                                              "mean_control_cost": 2.0})}
    for gid, p in payloads.items():
        atomic_write_json(tmp_path / f"{gid}.json", p)
    outcomes = [read_outcome(tmp_path / f"{gid}.json", gid) for gid in payloads]
    records = [o.to_archive_record(0) for o in outcomes if o.ok]
    archive = SlidingBoundariesArchive(dims=(2, 2), seed=0)
    archive.initialize(records)
    assert sorted(r.gene_id for r in archive.buffer) == ["a", "d"]
    json.dumps(archive.state_dict())
