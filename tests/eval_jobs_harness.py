"""Exercise run_improved.py's eval-job handling with fake training jobs (no GPU, no LLM).

    cd <repo root> && .venv/bin/python tests/eval_jobs_harness.py   (needs the controller venv; ~15 s)

Fake sbatch runs each job script in the background with bash; fake genes write a
results CSV and print "Job Done" (or crash). Checks: packing 2 genes per job,
a crashing gene inside a pack, one gene per job (old path), and gen-0 checkpoint resume.
"""
import os
import sys
import time
import shutil
import tempfile

REPO = os.getcwd()
sys.path.insert(0, REPO)
import run_improved as rim  # noqa: E402

work = tempfile.mkdtemp(prefix="ctrl_harness_", dir=REPO)
os.chdir(REPO)
counter = {"n": 0}

# fake sbatch: run the script in the background, output to run_job_outputs/evaluation/slurm-<id>.out
fake_sbatch = os.path.join(work, "fake_sbatch.sh")
with open(fake_sbatch, "w") as f:
    f.write('#!/bin/bash\n'
            'while [ "${1#-}" != "$1" ]; do shift 2; done\n'   # drop "-t TIME"
            'ID=$((900000 + RANDOM))\n'
            'mkdir -p run_job_outputs/evaluation\n'
            '( bash "$1"; echo "Begin Slurm Epilog" ) > run_job_outputs/evaluation/slurm-$ID.out 2>&1 &\n'
            'echo "Submitted batch job $ID"\n')
os.chmod(fake_sbatch, 0o755)

# fake gene: "$1" = train file (unused), "$2" = models.network_<id>
fake_gene = os.path.join(work, "fake_gene.sh")
results_dir = os.path.join(work, "sota", "results")
with open(fake_gene, "w") as f:
    f.write('#!/bin/bash\n'
            'G=${2#models.network_}\n'
            'sleep 3\n'
            'if [[ "$G" == *crash* ]]; then echo "Traceback (most recent call last): boom"; exit 1; fi\n'
            f'mkdir -p {results_dir}\n'
            f'printf "mean_reward,std_reward\\n%s,1.0\\n" "${{#G}}00" > {results_dir}/${{G}}_results.csv\n'
            'echo "Job Done"\n')

rim.RUN_COMMAND = fake_sbatch
rim.LOCAL = False
rim.EVAL_RUNLINE = f"bash {fake_gene} {{}} models.{{}}"
rim.SOTA_ROOT = os.path.join(work, "sota")
rim.VARIANT_DIR = os.path.join(work, "models")
rim.OUTPUT_DIR = os.path.join(work, "out")
rim.GENE_JOB_TIME = ""
os.makedirs(rim.VARIANT_DIR, exist_ok=True)


def make_population(names):
    pop = []
    for name in names:
        open(os.path.join(rim.VARIANT_DIR, f"{rim.MODEL}_{name}.py"), "w").write("# fake gene\n")
        ind = rim.creator.Individual([name])
        ind.fitness.values = rim.PLACEHOLDER_FITNESS
        rim.GLOBAL_DATA[name] = {"sub_flag": True, "job_id": "llm", "status": "subbed file", "start_time": time.time(), "fitness": rim.PLACEHOLDER_FITNESS}
        pop.append(ind)
    return pop


def run_case(label, names, genes_per_job):
    rim.GENES_PER_JOB = genes_per_job
    pop = make_population(names)
    t0 = time.time()
    rim.check_and_update_fitness(pop, timeout=120, loop_delay=2)
    jobs = {rim.GLOBAL_DATA[n]["results_job"] for n in names}
    fits = {n: tuple(ind.fitness.values) for n, ind in zip(names, pop)}
    print(f"\n[{label}] {time.time() - t0:.0f}s jobs={len(jobs)} fitness={fits}", flush=True)
    return jobs, fits


ok = True
def expect(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    ok &= bool(cond)

# 1) packing: 4 genes, 2 per job, one crashes
jobs, fits = run_case("pack2", ["aa", "bbb", "crash1", "dddd"], 2)
expect(len(jobs) == 2, "4 genes went into 2 jobs")
expect(fits["aa"][0] == 200 and fits["bbb"][0] == 300 and fits["dddd"][0] == 400,
       "packed genes each got their own result")
expect(fits["crash1"] == tuple(rim.INVALID_FITNESS_MAX), "crashing packed gene marked invalid")
expect(all(rim.GLOBAL_DATA[n]["packed"] for n in ["aa", "bbb", "crash1", "dddd"]), "genes flagged packed")

# 2) one gene per job: old path
jobs, fits = run_case("single", ["eeeee", "crash2"], 1)
expect(len(jobs) == 2, "2 genes went into 2 jobs")
expect(fits["eeeee"][0] == 500 and fits["crash2"] == tuple(rim.INVALID_FITNESS_MAX), "single-gene results correct")
expect(not rim.GLOBAL_DATA["eeeee"]["packed"], "single genes not flagged packed")

# 3) gen-0 checkpoint resumes at generation 1
ck, gp = os.path.join(work, "ck"), os.path.join(work, "global")
rim.population, rim.hof = make_population(["ffffff"]), rim.tools.ParetoFront()
rim.save_checkpoint(0, folder_name=ck, global_path=gp)
pop_data, start_gen, _ = rim.load_checkpoint(folder_name=ck, global_path=gp)
expect(pop_data is not None and start_gen == 1, f"gen-0 checkpoint resumes at gen 1 (got {start_gen})")

shutil.rmtree(work, ignore_errors=True)
print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
