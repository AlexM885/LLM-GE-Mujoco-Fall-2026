"""MESB with the XQC learner (--algo xqc): gene checks, operator, trainer
adapter, Slurm evaluator and a mock end-to-end run. No JAX, GPU or LLM."""

import ast
import json
import os
import stat
import sys
import textwrap
from argparse import Namespace
from pathlib import Path

import pytest

import run_mesb
from sota.MujocoRL import train_gene_xqc as adapter
from sota.MujocoRL.mesb_evaluator import EvalJob, EvalSettings, SlurmEvaluator, trainer_args
from sota.MujocoRL.mesb_logging import RunPaths
from sota.MujocoRL.results_io import read_outcome, validate_payload
from src.cfg import constants_mesb as C
from src.mesb_llm_guard import validate_child, validate_xqc_gene
from src import mesb_llm_operator as op

REPO = Path(__file__).resolve().parents[2]
XQC_SEED = C.ALGO_PROFILES["xqc"]["seed_network"]
XQC_REQUIRED = C.ALGO_PROFILES["xqc"]["required_functions"]


# ------------------------------------------------------------------ gene checks
def test_profiles_point_at_existing_files():
    for name, prof in C.ALGO_PROFILES.items():
        assert Path(prof["seed_network"]).exists(), name
        assert list(REPO.glob(prof["prompt_glob"])), name
        assert (REPO / prof["constant_rules"]).exists(), name
        assert Path(prof["train_script"]).exists(), name
    assert C.XQC_GENE_VALIDATOR.exists() and C.XQC_TRAIN_SCRIPT.exists()


def test_xqc_seed_gene_passes_both_checks():
    assert validate_child(XQC_SEED, XQC_REQUIRED)[0]
    assert validate_xqc_gene(XQC_SEED, C.XQC_GENE_VALIDATOR) == (True, "ok")


def test_ppo_gene_is_not_an_xqc_gene():
    ok, reason, _ = validate_child(C.SEED_NETWORK, XQC_REQUIRED)
    assert not ok and "get_architecture_kwargs" in reason


@pytest.mark.parametrize("old,new,needle", [
    ("n_critics=2", "n_critics=3", "n_critics"),
    ("tau=0.005", "tau=0.005, gamma=0.99", "gamma"),
    ("actor_lr=3e-4", "actor_lr=0.5", "actor_lr"),
    ('ALGO = "xqc"', 'ALGO = "ppo"', "ALGO"),
])
def test_xqc_validator_rejects_bad_genes(tmp_path, old, new, needle):
    src = XQC_SEED.read_text()
    assert old in src
    bad = tmp_path / "bad.py"
    bad.write_text(src.replace(old, new))
    ok, reason = validate_xqc_gene(bad, C.XQC_GENE_VALIDATOR)
    assert not ok and needle in reason


def test_mock_xqc_mutations_stay_valid(tmp_path):
    import random
    code = XQC_SEED.read_text()
    for seed in range(25):
        child = op.mock_mutate_xqc_text(code, random.Random(seed))
        path = tmp_path / f"c{seed}.py"
        path.write_text(child)
        assert validate_xqc_gene(path, C.XQC_GENE_VALIDATOR)[0], child


def test_operator_mock_xqc_ok_and_invalid(tmp_path, capsys):
    out = tmp_path / "child.py"
    rc = op.main(["mutate", "--backend", "mock", "--algo", "xqc", "--input", str(XQC_SEED),
                  "--output", str(out), "--seed", "3"])
    assert rc == 0 and "CHILD OK" in capsys.readouterr().out
    parent = tmp_path / "parent.py"
    parent.write_text(XQC_SEED.read_text() + "\n# MOCK_INVALID_CHILD\n")
    rc = op.main(["mutate", "--backend", "mock", "--algo", "xqc", "--input", str(parent),
                  "--output", str(tmp_path / "bad.py"), "--seed", "3"])
    assert rc == op.EXIT_INVALID_CHILD
    assert (tmp_path / "bad.rejected.py").exists()


# ------------------------------------------------------------------ trainer adapter
def test_adapter_is_python39_syntax():
    ast.parse(Path(adapter.__file__).read_text(), feature_version=(3, 9))


def test_adapter_command_matches_shazebs_trainer():
    """train_gene_xqc.py must call train_xqc.py the way train_xqc_gene.py does."""
    his = C.XQC_GENE_VALIDATOR.read_text()
    for fragment in ('f"env={args.env}"', 'f"num_seeds={args.num_seeds}"',
                     'f"max_steps={args.timesteps}"', 'f"eval_interval={eval_interval}"',
                     'f"eval_episodes={args.eval_episodes}"',
                     'f"start_training={min(5000, args.timesteps // 2)}"',
                     '"checkpoint_interval=0"', '"save_replay_buffer=false"',
                     '"final_descriptors=true"', '"resume=never"',
                     "eval_interval = min(25_000, args.timesteps)"):
        assert fragment in his, f"Shazeb's trainer changed: {fragment} not found"
    args = Namespace(env="HalfCheetah-v4", seed=0, num_seeds=3, timesteps=100_000,
                     eval_episodes=10, run_dir="/r")
    cmd = adapter.xqc_command(args, ["agent.tau=0.005"], python="py")
    assert cmd[:2] == ["py", adapter.XQC_TRAIN]
    assert cmd[2:] == ["env=HalfCheetah-v4", "seed=0", "num_seeds=3", "max_steps=100000",
                       "eval_interval=25000", "eval_episodes=10", "start_training=5000",
                       "run_dir=/r", "checkpoint_interval=0", "save_replay_buffer=false",
                       "final_descriptors=true", "resume=never", "agent.tau=0.005"]


def _descriptors(num_seeds=3, episodes=10):
    returns = [[1000.0 + 100 * s + e for e in range(episodes)] for s in range(num_seeds)]
    return {"env_step": 100000, "actor_params": 12345, "returns": returns,
            "distances": [[50.0 + s for _ in range(episodes)] for s in range(num_seeds)],
            "control_costs": [[0.3 for _ in range(episodes)] for _ in range(num_seeds)],
            "mean_return_per_seed": [sum(r) / len(r) for r in returns],
            "mean_distance": 51.0, "mean_control_cost": 0.3}


def test_payload_from_descriptors():
    args = Namespace(gene_id="g1", generation=2, seed=0, env="HalfCheetah-v4",
                     timesteps=100000, num_seeds=3, eval_episodes=10, max_steps=1000,
                     gene_file="g1.py", run_dir="runs/g1")
    p = adapter.payload_from_descriptors(_descriptors(), args, 12.0, {"tau": 0.005})
    assert validate_payload(p, "g1") == (True, None)
    assert p["mean_reward"] == pytest.approx(1104.5)
    assert p["std_reward"] == pytest.approx((2 / 3 * 100 ** 2) ** 0.5)
    assert p["behavior"]["mean_control_cost"] == pytest.approx(300.0)
    assert p["behavior"]["mean_control_cost_per_step"] == pytest.approx(0.3)
    assert len(p["rewards"]) == 30 and p["param_count"] == 12345


def _fake_xqc_train(tmp_path, fail=False):
    """Stand-in for sota/XQC/train_xqc.py: writes descriptors.json into run_dir."""
    script = tmp_path / "fake_train_xqc.py"
    desc = json.dumps(_descriptors())
    script.write_text(textwrap.dedent(f"""
        import os, sys
        kv = dict(a.split("=", 1) for a in sys.argv[1:])
        assert kv["agent.n_critics"] == "2" and kv["final_descriptors"] == "true"
        if {fail!r}:
            print("Traceback: boom"); sys.exit(1)
        open(os.path.join(kv["run_dir"], "descriptors.json"), "w").write({desc!r})
    """))
    return str(script)


def test_adapter_main_success_and_failures(tmp_path, monkeypatch):
    base = ["--gene-file", str(XQC_SEED), "--gene-id", "g1", "--timesteps", "2000",
            "--num-seeds", "3"]
    monkeypatch.setattr(adapter, "XQC_TRAIN", _fake_xqc_train(tmp_path))
    out = tmp_path / "ok.json"
    assert adapter.main(base + ["--out-json", str(out), "--run-dir", str(tmp_path / "r1")]) == 0
    o = read_outcome(out, "g1")
    assert o.ok and o.payload["algorithm"] == "XQC" and o.payload["xqc_config"]["n_critics"] == 2

    monkeypatch.setattr(adapter, "XQC_TRAIN", _fake_xqc_train(tmp_path, fail=True))
    out = tmp_path / "fail.json"
    adapter.main(base + ["--out-json", str(out), "--run-dir", str(tmp_path / "r2")])
    p = json.loads(out.read_text())
    assert p["status"] == "failed" and p["stage"] == "train" and "boom" in p["log_tail"]

    bad = tmp_path / "bad.py"
    bad.write_text(XQC_SEED.read_text().replace("n_critics=2", "n_critics=4"))
    out = tmp_path / "bad.json"
    adapter.main(["--gene-file", str(bad), "--gene-id", "g1", "--out-json", str(out),
                  "--run-dir", str(tmp_path / "r3")])
    p = json.loads(out.read_text())
    assert p["status"] == "failed" and p["stage"] == "validate"


# ------------------------------------------------------------------ Slurm evaluator
def _exe(path: Path, text: str) -> str:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


@pytest.fixture
def fake_slurm(tmp_path, monkeypatch):
    """Fake sbatch/scancel on PATH; sbatch runs the job script synchronously."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _exe(bin_dir / "sbatch", textwrap.dedent(f"""\
        #!{sys.executable}
        import subprocess, sys
        args = sys.argv[1:]
        out = args[args.index("--output") + 1]
        script_at = next(i for i, a in enumerate(args) if a.endswith(".sh"))
        print("4242", flush=True)
        with open(out, "w") as log:
            rc = subprocess.call([args[script_at]] + args[script_at + 1:], stdout=log, stderr=log)
        sys.exit(rc)
        """))
    _exe(bin_dir / "scancel", f"#!/bin/sh\necho \"$@\" >> {tmp_path}/scancelled\n")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return tmp_path


def _job_script(tmp_path, body):
    return _exe(tmp_path / "job.sh", f"#!{sys.executable}\n" + textwrap.dedent(body))


def _settings(script, timeout=60.0, workers=3):
    return EvalSettings(env_id="HalfCheetah-v4", timesteps=1000, episodes=10, max_steps=1000,
                        eval_seed=1000, timeout_sec=timeout, workers=workers, algo="xqc",
                        num_seeds=3, slurm_script=script, slurm_flags="-A coc")


def _jobs(tmp_path, n):
    return [EvalJob(gene_id=f"g{i}", gene_file=XQC_SEED, generation=0, train_seed=i,
                    out_json=tmp_path / "evaluations" / f"g{i}.json",
                    model_dir=tmp_path / "models", log_file=tmp_path / "logs" / f"g{i}.log")
            for i in range(n)]


def test_slurm_command_shape(tmp_path):
    ev = SlurmEvaluator(_settings("/x/xqc_gene.sbatch"))
    job = _jobs(tmp_path, 1)[0]
    cmd = ev.command(job)
    assert cmd[:3] == ["sbatch", "--wait", "--parsable"]
    assert "-A" in cmd and "coc" in cmd
    i = cmd.index("/x/xqc_gene.sbatch")
    assert cmd[i + 1:] == trainer_args(ev.settings, job)[1:]
    assert "--num-seeds" in cmd and cmd[cmd.index("--seed") + 1] == "0"


def test_slurm_evaluator_runs_jobs(fake_slurm):
    tmp = fake_slurm
    payload = json.dumps(adapter.payload_from_descriptors(
        _descriptors(), Namespace(gene_id="GENE", generation=0, seed=0, env="HalfCheetah-v4",
                                  timesteps=1000, num_seeds=3, eval_episodes=10,
                                  max_steps=1000, gene_file="x.py", run_dir="r"), 1.0, {}))
    script = _job_script(tmp, f"""
        import sys
        a = sys.argv[1:]
        gid, out = a[a.index("--gene-id") + 1], a[a.index("--out-json") + 1]
        import os; os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "w").write({payload!r}.replace("GENE", gid))
        print("trained", gid)
    """)
    outcomes = SlurmEvaluator(_settings(script)).evaluate(_jobs(tmp, 4))
    assert all(o.ok for o in outcomes.values()) and len(outcomes) == 4
    assert "trained g2" in (tmp / "logs" / "g2.log").read_text()


def test_slurm_evaluator_timeout_and_missing_result(fake_slurm):
    tmp = fake_slurm
    script = _job_script(tmp, "import time; time.sleep(30)\n")
    outcomes = SlurmEvaluator(_settings(script, timeout=1.0, workers=1)).evaluate(_jobs(tmp, 1))
    assert not outcomes["g0"].ok and "timed out" in outcomes["g0"].reason
    assert "4242" in (tmp / "scancelled").read_text()
    script = _job_script(tmp, "print('crashed before writing a result')\n")
    outcomes = SlurmEvaluator(_settings(script)).evaluate(_jobs(tmp / "b", 1))
    assert not outcomes["g0"].ok and "without a result file" in outcomes["g0"].reason


# ------------------------------------------------------------------ driver
def _cfg(tmp_path, name, extra=()):
    argv = ["--run-name", name, "--output-root", str(tmp_path), "--selection-mode", "mesb",
            "--llm-backend", "mock", "--evaluator", "mock", "--llm-python", sys.executable,
            "--num-generations", "2", "--start-population-size", "8", "--population-size", "6",
            "--mesb-dims", "4", "4", "--mesb-remap-frequency", "5", "--seed", "3",
            "--workers", "2", *extra]
    return run_mesb.config_from_args(run_mesb.build_parser().parse_args(argv))


def test_driver_xqc_defaults_and_mock_run(tmp_path):
    cfg = _cfg(tmp_path, "x", ["--algo", "xqc"])
    assert cfg["eval_timesteps"] == 100_000 and cfg["eval_num_seeds"] == 3
    run_mesb.MESBRun(cfg).run(resume=False)
    paths = RunPaths(tmp_path / "x")
    stored = json.loads(paths.config.read_text())
    assert stored["algo"] == "xqc"
    genes = list(paths.genes.glob("network_*.py"))
    assert genes and all('ALGO = "xqc"' in g.read_text() for g in genes)
    assert all(validate_xqc_gene(g, C.XQC_GENE_VALIDATOR)[0] for g in genes)


def test_driver_rejects_bad_combinations(tmp_path):
    with pytest.raises(SystemExit):
        _cfg(tmp_path, "a", ["--evaluator", "slurm"])           # no PPO job script
    with pytest.raises(SystemExit):
        _cfg(tmp_path, "b", ["--eval-num-seeds", "3"])           # PPO trains one seed
    assert _cfg(tmp_path, "c")["eval_timesteps"] == 500_000     # PPO default unchanged


def test_resume_of_pre_xqc_run_assumes_ppo(tmp_path):
    cfg = _cfg(tmp_path, "old")
    run_mesb.MESBRun(cfg).run(resume=False)
    paths = RunPaths(tmp_path / "old")
    stored = json.loads(paths.config.read_text())
    for key in ("algo", "eval_num_seeds"):
        stored.pop(key)
    paths.config.write_text(json.dumps(stored))
    run_mesb.MESBRun(_cfg(tmp_path, "old")).run(resume=True)       # no mismatch error
    with pytest.raises(run_mesb.CheckpointMismatchError):
        run_mesb.MESBRun(_cfg(tmp_path, "old", ["--algo", "xqc"])).run(resume=True)


def test_seed_network_override(tmp_path):
    small = tmp_path / "small_gene.py"
    small.write_text(XQC_SEED.read_text().replace("hidden_dims_actor=[256, 256, 256, 256]",
                                                  "hidden_dims_actor=[32, 32]"))
    cfg = _cfg(tmp_path, "s", ["--algo", "xqc", "--seed-network", str(small),
                               "--num-generations", "0"])
    run_mesb.MESBRun(cfg).run(resume=False)
    genes = list(RunPaths(tmp_path / "s").genes.glob("network_*.py"))
    assert genes and all("hidden_dims_actor=[" in g.read_text() for g in genes)
    assert all("256, 256, 256, 256]" not in g.read_text().split("hidden_dims_actor")[1][:30]
               for g in genes)
