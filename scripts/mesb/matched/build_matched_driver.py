"""Generate run_improved_matched.py: Shazeb's run_improved.py plus marked hooks.

The matched MESB vs fixed-grid runs must use his pipeline unchanged except for
what the experiment varies, so the driver is not hand-edited: this script
copies ``run_improved.py`` and inserts a few calls into ``src/mesb_matched.py``
at exact, asserted places. Each insertion is marked ``# MESB-MATCHED``. If his
file changes so that a place is no longer found exactly once, generation stops
with an error instead of guessing.

Hooks (all inactive unless the env options of src/mesb_matched.py ask for them):
  H0  import src/mesb_matched.py; a SlidingState next to MAP_ELITES_ARCHIVE
  H1  load_yaml: gene-evaluation job time limit (MATCHED_GENE_TIME)
  H2  write_bash_script: crossover job script (MATCHED_FIX_CROSSOVER)
  H3  update_archive: sliding-boundaries archive (MATCHED_ARCHIVE=sliding)
  H4  save_checkpoint: MESB state + options in the checkpoint; boundaries log
  H5  checkpoint load: restore MESB state; refuse to resume the other arm
  H6  after generation 0 is evaluated: save it and exit (MATCHED_STOP_AFTER_GEN0)
  H7  start of main: validate the options

    python scripts/mesb/matched/build_matched_driver.py --out <run dir>/run_improved_matched.py
"""

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TAG = "  # MESB-MATCHED"

HOOKS = [
    ("H0 import",
     "from src.cfg import constants\n",
     "from src.cfg import constants\n"
     "from src import mesb_matched" + TAG + "\n"),
    ("H0 state",
     "\nMAP_ELITES_ARCHIVE = {}\n",
     "\nMAP_ELITES_ARCHIVE = {}\n"
     "MESB_STATE = mesb_matched.SlidingState(MAP_BINS, MAP_ELITES_DESCRIPTORS)" + TAG + "\n"),
    ("H1 load_yaml",
     "        config = yaml.safe_load(file)\n    return config\n",
     "        config = yaml.safe_load(file)\n"
     "    return mesb_matched.adjust_slurm_config(config)" + TAG + "\n"),
    ("H2 crossover script",
     '        temp_text = f"{python_file} {input_filename_x} {input_filename_y} {output_filename} '
     '--top_p {top_p} --temperature {temperature}"\n',
     '        temp_text = f"{mesb_matched.crossover_script(python_file)} {input_filename_x} '
     '{input_filename_y} {output_filename} --top_p {top_p} --temperature {temperature}"' + TAG + "\n"),
    ("H3 update_archive",
     "    global MAP_ELITES_ARCHIVE\n    added = replaced = 0\n",
     "    global MAP_ELITES_ARCHIVE\n"
     "    if mesb_matched.SLIDING:" + TAG + "\n"
     "        def _accepted(ind):" + TAG + "\n"
     "            # exactly the filters of the fixed-grid code below\n"
     "            if not ind.fitness.valid or tuple(ind.fitness.values) == tuple(INVALID_FITNESS_MAX):\n"
     "                return False\n"
     "            if ind.fitness.values == PLACEHOLDER_FITNESS:\n"
     "                return False\n"
     "            if get_bin(ind) is None:\n"
     "                return False\n"
     "            quality = get_quality(ind)\n"
     "            return bool(np.isfinite(quality)) and quality > FAILED_EVAL_SENTINEL\n"
     "        MAP_ELITES_ARCHIVE, added, replaced = MESB_STATE.update(" + TAG + "\n"
     "            population, MAP_ELITES_ARCHIVE, is_valid=_accepted, metrics=get_metrics,\n"
     "            quality=get_quality)\n"
     "        total_cells = MAP_BINS ** len(MAP_ELITES_DESCRIPTORS)\n"
     "        coverage = (len(MAP_ELITES_ARCHIVE) / total_cells * 100) if total_cells else 0.0\n"
     "        qd_score = sum(get_quality(i) for i in MAP_ELITES_ARCHIVE.values())\n"
     "        print(f\"\\t‣ MAP-Elites (sliding boundaries): {MESB_STATE.describe()}\")\n"
     "        print(f\"\\t‣ MAP-Elites: {added} added, {replaced} replaced, \"\n"
     "              f\"{len(MAP_ELITES_ARCHIVE)}/{total_cells} cells filled \"\n"
     "              f\"({coverage:.1f}% coverage), QD-score {qd_score:.2f}\")\n"
     "        return added, replaced\n"
     "    added = replaced = 0\n"),
    ("H4 checkpoint data",
     '            "MAP_ELITES_ARCHIVE": MAP_ELITES_ARCHIVE,\n        }\n',
     '            "MAP_ELITES_ARCHIVE": MAP_ELITES_ARCHIVE,\n'
     "            **mesb_matched.checkpoint_extra(MESB_STATE)," + TAG + "\n        }\n"),
    ("H4 boundaries log",
     '    print(f"Population data saved as {filename}")\n',
     '    print(f"Population data saved as {filename}")\n'
     "    if mesb_matched.SLIDING:" + TAG + "\n"
     "        mesb_matched.write_boundaries_log(MESB_STATE, os.path.join(folder_name, "
     "'mesb_boundaries.jsonl'), gen)\n"),
    ("H5 load",
     '        MAP_ELITES_ARCHIVE = population_data.get("MAP_ELITES_ARCHIVE", {})\n',
     '        MAP_ELITES_ARCHIVE = population_data.get("MAP_ELITES_ARCHIVE", {})\n'
     '        mesb_matched.check_resume(population_data.get("MATCHED_OPTIONS"))' + TAG + "\n"
     '        MESB_STATE.load(population_data.get("MESB_STATE"))' + TAG + "\n"),
    ("H6 stop after generation 0",
     "    check_and_update_fitness(population)\n    if USE_MAP_ELITES:\n"
     '        box_print("SEEDING MAP-ELITES ARCHIVE"',
     "    check_and_update_fitness(population)\n"
     "    if mesb_matched.STOP_AFTER_GEN0:" + TAG + "\n"
     "        if population_data:\n"
     "            raise SystemExit('[matched] MATCHED_STOP_AFTER_GEN0 but a checkpoint already exists')\n"
     "        save_checkpoint(0, folder_name=args.checkpoints, global_path=args.global_path)\n"
     "        print('[matched] generation 0 evaluated and saved; exiting', flush=True)\n"
     "        raise SystemExit(0)\n"
     "    if USE_MAP_ELITES:\n"
     '        box_print("SEEDING MAP-ELITES ARCHIVE"'),
    ("H7 options",
     "    args = parser.parse_args()\n",
     "    args = parser.parse_args()\n"
     "    mesb_matched.check_options()" + TAG + "\n"),
]


def build(source: str) -> str:
    out = source
    for name, old, new in HOOKS:
        n = out.count(old)
        if n != 1:
            raise SystemExit(f"{name}: expected exactly one match in run_improved.py, found {n}; "
                             "his file changed - update build_matched_driver.py")
        out = out.replace(old, new)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", default=str(REPO_ROOT / "run_improved.py"))
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    src = Path(args.source).read_text(encoding="utf-8")
    driver = build(src)
    compile(driver, args.out, "exec")
    header = (f"# GENERATED by scripts/mesb/matched/build_matched_driver.py from run_improved.py "
              f"(sha256 {hashlib.sha256(src.encode()).hexdigest()[:16]}). Do not edit.\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(header + driver, encoding="utf-8")
    print(f"wrote {args.out} ({len(HOOKS)} hooks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
