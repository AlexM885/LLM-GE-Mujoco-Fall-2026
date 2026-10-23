"""Matched MESB vs fixed-grid runs on Shazeb's XQC island: driver generation,
the sliding archive hook, Slurm/crossover hooks and the fixed crossover."""

import importlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "mesb" / "matched"))
import build_matched_driver as bmd  # noqa: E402


def _matched(monkeypatch, **env):
    for k in ("MATCHED_ARCHIVE", "MATCHED_FIX_CROSSOVER", "MATCHED_GENE_TIME", "MATCHED_STOP_AFTER_GEN0"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import src.mesb_matched as mm
    return importlib.reload(mm)


# ------------------------------------------------------------------ driver generation
def test_driver_builds_from_current_run_improved(tmp_path):
    out = tmp_path / "run_improved_matched.py"
    assert bmd.main(["--out", str(out)]) == 0
    text = out.read_text()
    compile(text, str(out), "exec")
    assert text.count("# MESB-MATCHED") >= len(bmd.HOOKS)
    original = (REPO / "run_improved.py").read_text().splitlines()
    generated = text.splitlines()[1:]           # drop the GENERATED header
    # every original line is still there, in order (hooks only add or tag lines)
    it = iter(generated)
    for line in original:
        for g in it:
            if g == line or (g.endswith("# MESB-MATCHED") and "{mesb_matched." in g) or \
                    g.startswith("    return mesb_matched.adjust_slurm_config"):
                break
        else:
            pytest.fail(f"original line lost: {line!r}")


def test_driver_refuses_a_changed_source(tmp_path):
    src = (REPO / "run_improved.py").read_text().replace("    global MAP_ELITES_ARCHIVE\n", "")
    bad = tmp_path / "run_improved.py"
    bad.write_text(src)
    with pytest.raises(SystemExit, match="H3"):
        bmd.main(["--source", str(bad), "--out", str(tmp_path / "x.py")])


# ------------------------------------------------------------------ simple hooks
def test_slurm_time_and_crossover_hooks(monkeypatch):
    mm = _matched(monkeypatch, MATCHED_ARCHIVE="fixed")
    cfg = {"python_bash_script": "#!/bin/bash\n#SBATCH -t 1:30:00\n#SBATCH -G 1\n{}", "x": 1}
    out = mm.adjust_slurm_config(cfg)
    assert "#SBATCH -t 3:00:00" in out["python_bash_script"] and "1:30:00" not in out["python_bash_script"]
    assert cfg["python_bash_script"].count("1:30:00") == 1            # input not mutated
    assert mm.crossover_script("src/llm_crossover.py") == "src/llm_crossover_fixed.py"
    assert mm.crossover_script("src/llm_mutation.py") == "src/llm_mutation.py"
    with pytest.raises(RuntimeError):
        mm.adjust_slurm_config({"python_bash_script": "#SBATCH -t 2:00:00"})
    mm = _matched(monkeypatch, MATCHED_ARCHIVE="fixed", MATCHED_FIX_CROSSOVER="0",
                  MATCHED_GENE_TIME="keep")
    assert mm.crossover_script("src/llm_crossover.py") == "src/llm_crossover.py"
    assert mm.adjust_slurm_config(cfg) is cfg


def test_slurm_config_on_branch_has_the_expected_line(monkeypatch):
    import yaml
    mm = _matched(monkeypatch, MATCHED_ARCHIVE="fixed")
    cfg = yaml.safe_load((REPO / "slurm-config" / "slurm_config.yaml").read_text())
    assert "#SBATCH -t 3:00:00" in mm.adjust_slurm_config(cfg)["python_bash_script"]


def test_options_and_resume_guard(monkeypatch):
    mm = _matched(monkeypatch, MATCHED_ARCHIVE="bogus")
    with pytest.raises(SystemExit):
        mm.check_options()
    mm = _matched(monkeypatch, MATCHED_ARCHIVE="sliding")
    from src.cfg import run_options
    monkeypatch.setattr(run_options, "USE_LLM_EMITTERS", True)
    with pytest.raises(SystemExit, match="USE_LLM_EMITTERS"):
        mm.check_options()                                            # emitters need a fixed grid
    monkeypatch.setattr(run_options, "USE_LLM_EMITTERS", False)
    mm.check_options()
    gen0 ={"archive": "gen0", "fix_crossover": True, "gene_time": "3:00:00"}
    mm.check_resume(gen0)                                             # both arms may resume gen 0
    mm.check_resume(dict(gen0, archive="sliding"))
    with pytest.raises(SystemExit, match="fixed"):
        mm.check_resume(dict(gen0, archive="fixed"))                  # not the other arm
    with pytest.raises(SystemExit, match="gene_time"):
        mm.check_resume(dict(gen0, gene_time="1:30:00"))
    mm = _matched(monkeypatch, MATCHED_ARCHIVE="fixed", MATCHED_STOP_AFTER_GEN0="1")
    assert mm.options()["archive"] == "gen0"


# ------------------------------------------------------------------ sliding archive
class Ind(list):
    def __init__(self, gene_id, valid=True):
        super().__init__([gene_id])
        self.fitness = SimpleNamespace(valid=valid)


def _state(mm, bins=4):
    return mm.SlidingState(bins, [("mean_distance", -500.0, 2000.0), ("mean_control_cost", 0.0, 0.6)])


def test_sliding_archive_rebuild(monkeypatch):
    mm = _matched(monkeypatch, MATCHED_ARCHIVE="sliding")
    st = _state(mm)
    metrics = {f"g{i}": {"mean_distance": 400.0 + i, "mean_control_cost": 0.20 + 0.01 * (i % 5),
                         "mean_reward": 7000.0 + 10 * i} for i in range(12)}
    metrics["bad"] = {"mean_distance": 1.0}                          # missing a descriptor
    pop = [Ind(g) for g in metrics] + [Ind("invalid", valid=False), Ind("g3")]  # duplicate id
    archive, added, replaced = st.update(
        pop, {}, is_valid=lambda ind: ind.fitness.valid, metrics=metrics.get,
        quality=lambda ind: metrics[ind[0]]["mean_reward"])
    assert len(st.buffer) == 12 and [e["gene_id"] for e in st.buffer][:2] == ["g0", "g1"]
    # percentile edges: 12 values, 4 bins -> internal edges at sorted ranks 3, 6, 9
    assert st.edges[0][1:4] == [403.0, 406.0, 409.0] and st.edges[0][0] == 400.0
    # every cell keeps its best gene; distances 400..411 spread over all 4 distance bins
    assert {c[0] for c in archive} == {0, 1, 2, 3}
    for cell, ind in archive.items():
        same = [e for e in st.buffer if st.cell(e["measures"]) == cell]
        assert ind[0] == max(same, key=lambda e: e["quality"])["gene_id"]
    assert added == len(archive) and replaced == 0
    # a better gene in an occupied cell takes it over at the next remap
    metrics["g12"] = dict(metrics["g11"], mean_reward=99999.0)
    archive2, added2, replaced2 = st.update(
        [Ind("g12")], archive, is_valid=lambda ind: True, metrics=metrics.get,
        quality=lambda ind: metrics[ind[0]]["mean_reward"])
    assert "g12" in {i[0] for i in archive2.values()} and st.remaps == 2 and added2 >= 1
    # state survives a checkpoint round trip
    st2 = _state(mm)
    st2.load(st.state_dict())
    assert st2.edges == st.edges and len(st2.buffer) == 13 and "remap 2" in st2.describe()


def test_sliding_archive_skips_nonfinite_and_empty(monkeypatch):
    mm = _matched(monkeypatch, MATCHED_ARCHIVE="sliding")
    st = _state(mm)
    metrics = {"a": {"mean_distance": math.nan, "mean_control_cost": 0.2}}
    archive, added, replaced = st.update([Ind("a")], {}, is_valid=lambda i: True,
                                         metrics=metrics.get, quality=lambda i: 1.0)
    assert archive == {} and st.buffer == [] and (added, replaced) == (0, 0)


def test_checkpoint_extra(monkeypatch, tmp_path):
    mm = _matched(monkeypatch, MATCHED_ARCHIVE="sliding")
    st = _state(mm)
    extra = mm.checkpoint_extra(st)
    assert extra["MATCHED_OPTIONS"]["archive"] == "sliding" and "MESB_STATE" in extra
    mm = _matched(monkeypatch, MATCHED_ARCHIVE="fixed")
    assert "MESB_STATE" not in mm.checkpoint_extra(st)


# ------------------------------------------------------------------ fixed crossover
XQC_GENE = (REPO / "sota" / "XQC" / "llmge" / "network.py").read_text()


def _crossover(monkeypatch, tmp_path, module_name):
    sys.path.insert(0, str(REPO / "src"))
    mod = importlib.import_module(module_name)
    marker = "def get_xqc_kwargs():\n    return dict(tau=0.01)\n"
    monkeypatch.setattr(mod, "generate_augmented_code", lambda *a, **k: marker)
    x = tmp_path / "x.py"
    y = tmp_path / "y.py"
    x.write_text(XQC_GENE)
    y.write_text(XQC_GENE.replace("tau=0.005", "tau=0.02"))      # differ only in chunk 2
    out = tmp_path / f"{module_name}.py"
    fn = getattr(mod, "augment_network_fixed", None) or mod.augment_network
    fn(str(x), str(y), str(out), llm_model="llama3")
    return out.read_text()


def test_fixed_crossover_writes_the_shown_chunk(monkeypatch, tmp_path):
    fixed = _crossover(monkeypatch, tmp_path, "llm_crossover_fixed")
    assert 'ALGO = "xqc"' in fixed                                   # header kept
    assert "def get_architecture_kwargs" in fixed and "tau=0.01" in fixed
    original = _crossover(monkeypatch, tmp_path, "llm_crossover")
    # the bug: the answer lands one chunk early and overwrites get_architecture_kwargs
    # (or, when the first chunk differs, the ALGO header) -> the trainer rejects the gene
    assert "def get_architecture_kwargs" not in original and original.count("def get_xqc_kwargs") == 2


# ------------------------------------------------------------------ post-hoc metrics
def _compare_module():
    import compare_arms
    return importlib.reload(compare_arms)


def test_mean_best_over_budget_and_counts():
    ca = _compare_module()
    table = [{"evaluated": 4, "best_so_far": 10.0}, {"evaluated": 4, "best_so_far": 20.0},
             {"evaluated": 4, "best_so_far": 30.0}]
    ca.add_evaluation_counts(table)
    assert [r["evaluations_so_far"] for r in table] == [4, 8, 12]
    assert ca.mean_best_over_budget(table, 8) == 15.0          # equal budget cuts the 3rd gen
    assert ca.mean_best_over_budget(table, 6) == round((4 * 10 + 2 * 20) / 6, 1)


def test_generation_cap_and_high_quality_cells(tmp_path):
    ca = _compare_module()
    import posthoc_submit as ps
    csv_path = tmp_path / "genes.csv"
    csv_path.write_text(
        "gene_id,generation,status,mean_reward,mean_distance,mean_control_cost\n"
        "a,0,ok,100,1,0.1\nb,5,ok,90,2,0.2\nc,6,ok,999,3,0.3\nd,5,invalid,,,\n")
    assert [r["gene_id"] for r in ca.load_genes(csv_path, max_gen=5)] == ["a", "b", "d"]
    assert len(ca.load_genes(csv_path)) == 4
    assert [r["gene_id"] for r in ps.ok_genes(tmp_path, max_gen=5)] == ["a", "b"]
    genes = [{"mean_distance": d, "mean_control_cost": 0.0, "mean_reward": r}
             for d, r in ((0, 100), (0, 50), (1, 70), (2, 85))]
    cell = lambda d, c: (int(d), 0)                       # noqa: E731
    assert ca.high_quality_cells(genes, cell, 80) == 2      # cells 0 (100) and 2 (85)
    island = tmp_path / "run" / "island_x"
    island.mkdir(parents=True)
    import pickle
    for g, ids in ((5, ["g5"]), (6, ["g6"])):
        with open(island / f"checkpoint_gen_{g}.pkl", "wb") as fh:
            pickle.dump({"MAP_ELITES_ARCHIVE": {(0, 0): ids}}, fh)
    assert ps.final_archive_ids(tmp_path / "run", max_gen=5) == {"g5"}
    assert ps.final_archive_ids(tmp_path / "run") == {"g6"}


def test_sbatch_stops_an_arm_that_finished(tmp_path):
    text = (REPO / "scripts/mesb/matched/matched_island.sbatch").read_text()
    guard = text.index('checkpoint_gen_${LAST_GEN}.pkl')
    assert guard < text.index("build_matched_driver.py --out")   # before any work is started


def test_posthoc_selection_is_top_then_far_apart():
    import posthoc_submit as ps
    rows = [{"gene_id": f"g{i}", "mean_reward": 100.0 - i, "d": float(i), "c": 0.0} for i in range(10)]
    chosen = ps.select(rows, {f"g{i}" for i in range(10)}, k_top=2, k_div=2,
                       scale=[(0.0, 9.0), (0.0, 1.0)])
    assert [r["gene_id"] for r in chosen] == ["g0", "g1", "g9", "g5"]
    few = ps.select(rows, {"g9"}, k_top=2, k_div=3, scale=[(0.0, 9.0), (0.0, 1.0)])
    assert [r["gene_id"] for r in few] == ["g0", "g1", "g9", "g2", "g3"]   # topped up by reward


def test_posthoc_summary_and_verdict(tmp_path):
    import json as _json
    ca = _compare_module()
    pair = tmp_path / "p"
    sel = {"seed": [{"gene_id": "seed", "role": "reference"}],
           "fixed": [{"gene_id": "a", "role": "top", "run_reward": 9000.0},
                     {"gene_id": "b", "role": "diverse", "run_reward": 8000.0}],
           "mesb": [{"gene_id": "c", "role": "top", "run_reward": 9100.0},
                    {"gene_id": "d", "role": "diverse", "run_reward": 7000.0}]}
    (pair / "posthoc").mkdir(parents=True)
    (pair / "posthoc" / "selection.json").write_text(_json.dumps({"k_top": 1, "genes": sel}))

    def res(group, gid, reward, cond, gait):
        d = pair / "posthoc" / group
        d.mkdir(exist_ok=True)
        (d / f"{gid}.json").write_text(_json.dumps({
            "status": "success", "mean_reward": reward,
            "perturbations": {"heavy_torso": {"mean_return": cond}, "slippery_floor": {"mean_return": cond}},
            "gait": {"mean": dict(zip(("back_foot_contact", "front_foot_contact", "torso_height_mean",
                                       "torso_angle_std"), gait))}}))
    res("seed", "seed", 7000.0, 5000.0, (0.5, 0.5, 0.0, 0.1))
    res("fixed", "a", 8500.0, 6000.0, (0.5, 0.5, 0.0, 0.1))
    res("fixed", "b", 7500.0, 6500.0, (0.6, 0.4, 0.0, 0.1))
    res("mesb", "c", 8600.0, 7000.0, (0.2, 0.8, 0.1, 0.2))
    res("mesb", "d", 6000.0, 7200.0, (0.9, 0.1, -0.1, 0.05))
    post = ca.posthoc_summary(pair)
    f, m = post["arms"]["fixed"], post["arms"]["mesb"]
    assert f["recheck_top5_mean"] == 8500.0 and m["recheck_top5_mean"] == 8600.0
    assert f["robustness_score"] == 6500.0 and m["robustness_score"] == 7200.0
    assert m["gait_spread"] > f["gait_spread"]
    assert m["improvement_over_seed_best"] == round((8600 - 7000) / 7000, 3)
    assert ca.verdict(post) == (True, True)
    (pair / "posthoc" / "mesb" / "c.json").unlink()
    (pair / "posthoc" / "mesb" / "d.json").unlink()
    assert ca.verdict(ca.posthoc_summary(pair)) is None           # incomplete -> pending


def test_perturbations_change_the_model():
    pytest.importorskip("mujoco")
    gym = pytest.importorskip("gymnasium")
    import posthoc_eval as pe
    base = gym.make("HalfCheetah-v4").unwrapped.model
    for name in pe.PERTURBATIONS:
        m = gym.make("HalfCheetah-v4").unwrapped.model
        pe.apply_perturbation(name, m)
        changed = any((getattr(m, a) != getattr(base, a)).any()
                      for a in ("body_mass", "body_inertia", "geom_friction", "actuator_gear"))
        assert changed, name
