# Tiered runs: how to launch yours

We are rerunning the midterm comparison properly: a **control** run and one run per **experiment tier**
(VLM, LLM emitters, sliding boundaries), each **repeated** so we can test differences with a Welch's
t-test per generation (Lecture 6, *Statistics for EMADE*). For that to mean anything, **every run must
use exactly the same code and settings except its tier**. This page is how to make sure yours does.

> **Status (Nov 19): ready for the baseline.** First real runs: `control` × 3 (Caleb, Suhan, Aayush) over
> Thanksgiving week, all on Shazeb's shared LLM server ([Baseline runs](#baseline-runs-control--3-thanksgiving-week)).
> The other tiers follow once the baseline is in.

## The settings (fixed, same for everyone)

All of these are set inside `scripts/tiered_run.sh`. You never type them yourself.

| Setting | Value | Why |
|---|---|---|
| Robot / learner | Walker2d-v5 / XQC | midterm config |
| Steps per gene | **300k** | Kendall tau (Nov 14): 300k ranks genes about as well as 475k ([PR #2](https://github.com/AlexM885/LLM-GE-Mujoco-Fall-2026/pull/2)) |
| Training seeds per gene | **3** (averaged) | one seed is too noisy to rank genes |
| Genes per generation | 32 | midterm config |
| Final evaluation | 10 episodes | midterm config |
| Archive | 32×32 MAP-Elites (left/right foot contact) | midterm config |
| Budget | 10 generations' worth of steps (288M), compared at equal steps | fair compute for every tier |
| Learning curve | every 25k steps (same as the midterm runs) | lets us redo the Kendall tau check |
| Gene jobs | 2 genes per GPU job on H100/H200, 3 h wall time, 3.5 h no-progress timeout, 48 h max queue wait | timing + smoke tests below |
| Speed-ups | GPU replay buffer, compile cache, duplicate reuse, queue-aware timeouts, gen-0 checkpoint, 2 genes per job | change speed, not results |
| Training seeds per repeat | repeat 1: 0–2, repeat 2: 1000–1002, repeat 3: 2000–2002 | repeats differ in training luck too |

The **only** things that change between runs:

| Tier | Turns on |
|---|---|
| `control` | nothing (plain LLM-GE + MAP-Elites) |
| `vlm` | VLM-written aux reward |
| `emitters` | LLM emitters + allocator |
| `sliding` | sliding-boundaries archive (its own driver, `run_mesb.py`) |

### Why these gene-job settings (timing test, Nov 14)

PACE counts the wall time a job *asks for* against each person's 32 GPU-hours, and the LLM server
holds 16 of them. So the gene wall time decides how many genes train at once: with the old 6 h,
only about 2 at a time.

| Test (seed gene, 300k × 3 seeds, eval every 25k, H200) | Time |
|---|---|
| 1 gene alone on a GPU | ~24 min |
| 2 genes packed on one GPU | ~27 min for both (~1.75× genes per GPU-hour) |
| Evals every 25k instead of 50k | under 1 min extra |

**Smoke test (Nov 15, `control_r99`, any GPU, 3 h):** Slurm put every gene job on a V100, RTX 6000
or A40. A packed pair took 1.1–3 h there (median 137 min) and 4 of 23 jobs hit the 3 h limit, so
generation 0 took 7.8 h. Gene jobs now ask for **H100/H200 only** (every tier the same). Three nodes
that fail JAX CUDA start-up are excluded (`BAD_NODES` in the script). The `sliding` tier gets the same
gene wall time, timeout and bad-node list.

**Smoke test 2 (Nov 16, `control_r98`, H100/H200, 2 h):** generation 0 took 3.7 h (was 7.8 h); packed
pairs trained in 43–53 min (gen 1: up to 82 min), nothing hit the wall time. The wall time is now
**3 h** anyway: evolved genes grow (uncapped genes reached 5–8× the seed's training time in Alex's 500k
run), and both genes of a pair die when the job hits its limit. Without your own LLM server (shared
server below) 3 h jobs still fit ~10 at once (20 genes) in your 32 GPU-hours.

It also found a bug, fixed before the baseline: a duplicate gene waiting for its still-training original
ran its own 2.5 h timeout from the moment it started waiting, so when the original sat ~2 h in the queue
the copies "timed out" with −inf although the original then trained fine (8 of 21 duplicates in
generation 0), and `generation_stats.csv` counted them as duplicates, not timeouts.

## Hard rules

1. **Use the agreed branch and commit:** branch `Midterm-Merge`, code commit `d79719b90`. Check with
   `git log -1 --format=%h -- . ':(exclude)docs'` (the last commit that changed code; doc-only commits after
   it are fine).
2. **Don't edit any code or config**, and don't set any of the fixed settings yourself. The script refuses to
   launch with uncommitted changes.
3. **Launch only with `scripts/tiered_run.sh <tier> <repeat>`.** Never `sbatch run.sh` directly.
4. **Use the tier and repeat you were assigned** (table at the bottom). Never reuse a run name.
5. **If something breaks, stop and tell the group chat.** Don't patch it locally and carry on: a run on
   different code can't be compared.

## One-time setup (PACE-ICE)

1. GT VPN on, then `ssh <you>@login-ice.pace.gatech.edu`.
2. Work in scratch, not home (home is only 30 GB):
   ```bash
   cd ~/scratch
   git clone https://github.com/AlexM885/LLM-GE-Mujoco-Fall-2026.git llmge && cd llmge
   git checkout Midterm-Merge             # then check the code commit (rule 1)
   ```
3. Python environments: the controller uses the repo's uv project (`module load uv && uv sync`); XQC genes
   use the XQC venv. Follow `sota/XQC/README.md` (setup section) once. If you already ran Shazeb's
   HalfCheetah XQC config, you have it.
4. VLM tier only: a Hugging Face cache with `google/gemma-3-12b-it` (accept its license on Hugging Face
   first), passed as `LLMGE_VLM_HF_HOME`.

## Launch

```bash
cd ~/scratch/llmge
scripts/tiered_run.sh <tier> <repeat> --dry-run    # check: tier, repeat, commit, settings
scripts/tiered_run.sh <tier> <repeat>              # launch
# VLM tier:
LLMGE_VLM_HF_HOME=/path/to/hf_cache scripts/tiered_run.sh vlm <repeat>
```

This submits a chain of LLM-server jobs and controller jobs that keeps the run going for about a week,
and writes `<tier>_r<repeat>/run_spec.txt` (every setting plus the commit) and `chain_jobs.txt` (job IDs).

## After launching: checklist

- [ ] Within ~1 h: `squeue -u $USER` shows the server running and gene jobs submitted
  (`evaluateGene`; sliding tier: `mesb-...`).
- [ ] After generation 0 (about 1–2 h): these exist under `<tier>_r<repeat>/island_llama3_Mujoco-XQC/`:
  - `checkpoint_gen_0.pkl` (the run can resume after a timeout)
  - `generation_stats.csv` with a row for generation 0

  Sliding tier: `mujoco_rl_output/sliding_r<repeat>/checkpoints/` and `mesb/metrics.csv` instead.
- [ ] Post in the group chat: run name, commit, and the generation-0 row of `generation_stats.csv`
  (sliding: `mesb/metrics.csv`).

Useful commands:
```bash
squeue -u $USER                                            # what's running / queued
tail -f run_job_outputs/islands/slurm-<controller id>.out  # controller log
column -s, -t < <run>/island_llama3_Mujoco-XQC/generation_stats.csv
scancel $(grep -h -E '^(servers|controllers):' <run>/chain_jobs.txt | cut -d: -f2)   # stop the run
```

`generation_stats.csv` has one row per generation: how many genes trained, were duplicates, failed or
timed out; how many entered the archive; archive size, best, median and QD score. This is what we'll
compare across tiers.

It also says **where each generation's time went** (minutes), so we can see which part of the loop
is slow (Dr. Zutty's suggestion):

| Column | What it times |
|---|---|
| `llm_min` (`crossover_min`, `mutation_min`) | LLM jobs writing the offspring (generation 0: creating the population) |
| `genes_min` | first gene submitted → last gene done |
| `gene_queue_median_min`, `gene_queue_max_min` | per gene: waiting in the SLURM queue |
| `gene_train_median_min`, `gene_train_max_min` | per gene: training + final evaluation |
| `vlm_min` | VLM writing the next aux reward (`vlm` tier) |
| `generation_min` | the whole generation |

The `sliding` tier writes the same timing columns into `mujoco_rl_output/sliding_r<repeat>/mesb/metrics.csv`.

## Baseline runs: control × 3 (Thanksgiving week)

| Run | Person | Server |
|---|---|---|
| control_r1 | Caleb | Shazeb's shared server |
| control_r2 | Suhan | Shazeb's shared server |
| control_r3 | Aayush | Shazeb's shared server |

Why this way: Slurm's fairshare is per user, and Shazeb's is ~0 after the speed-up tests, so his gene jobs
waited ~2 h each in smoke test 2. Each run launches from **its own person's account**. Shazeb hosts one
LLM server for all three (`scripts/shared_server.sh`), so all 32 of your GPU-hours go to gene jobs and
nobody's server sits in the queue.

**Before the launch day (setup, ~1 h):** do [One-time setup](#one-time-setup-pace-ice), then:
```bash
cd ~/scratch/llmge && git fetch && git checkout Midterm-Merge && git pull
git log -1 --format=%h -- . ':(exclude)docs'          # must print the code commit in rule 1
scripts/tiered_run.sh control <your repeat> --dry-run  # must show commit, repeat, LLMGE_GENE_TIME=3:00:00
```

**Launch day,** once Shazeb posts "server up":
```bash
cd ~/scratch/llmge
export LLMGE_HOSTNAME_FILE=/storage/ice1/0/9/swani8/llmge-shared/hostname.log LLMGE_SHARED_PORT=8742
curl -s -m 5 http://$(cat $LLMGE_HOSTNAME_FILE):$LLMGE_SHARED_PORT/ && echo SERVER-OK   # must print SERVER-OK
scripts/tiered_run.sh control <your repeat> --dry-run  # check: servers=0, LLMGE_PORT=8742
scripts/tiered_run.sh control <your repeat>
```
If `cat` says *Permission denied* or `curl` prints nothing, stop and post it in the group chat (don't
launch with your own server instead: that run would differ from the other two).

`servers=0` means the script submits only controllers; they wait (up to 8 h) whenever the shared server
is between jobs, so a short server gap never fails genes. Then do the [checklist](#after-launching-checklist).

### Hosting the shared server (Shazeb)

One-time, so teammates can read only the hostname file (`chmod 711` lets others pass through the
directory without listing it; everything else stays private):
```bash
cd /storage/ice1/0/9/swani8
chmod go-rwx ./*            # keep existing folders private
chmod 711 .                 # others may pass through, not list
mkdir -p llmge-shared && chmod 755 llmge-shared
```
Launch (from a checkout at the code commit), the day before teammates launch:
```bash
scripts/shared_server.sh 9          # 2 chains x 9 x 8 h, second chain 4 h later: ~3 days
```
Each new server job overwrites `llmge-shared/hostname.log`, and with the two chains offset by 4 h the
file always names the server with the most time left. Stop with
`scancel $(grep -E '^[AB] ' llmge-shared/server_jobs.txt | cut -d' ' -f2)` once all three runs are done.

## Known limitations

- The `sliding` tier runs a different driver (`run_mesb.py`): one gene per GPU job (no packing), so it
  trains fewer genes at once and its generations take longer. Runs are compared at equal steps, so
  this changes speed, not results. It has no `generation_stats.csv`: use its `mesb/metrics.csv` and
  `evaluations.csv`.
- Packing 2 genes per job was tested on the seed gene; the smoke test checks it on real evolved genes.
- Many duplicates in generation 0 (21 of 32 in smoke test 2) are expected: every gen-0 gene is one
  mutation of the seed, which has 2 editable parts, with one of 4 XQC prompts at `top_p` 0.1 (nearly
  greedy), so there are only ~8 distinct prompts. Duplicates reuse a result and cost no budget.
- One shared server serves all three runs: when their LLM phases overlap, each waits a little longer
  (the server batches requests). This changes speed, not results.

## For coding agents (paste this into your agent)

```
You are launching one LLM-GE tiered run on PACE-ICE for the MuJoCo VIP team.
Follow docs/TIERED_RUNS.md exactly. Strict rules:
- Use branch Midterm-Merge at code commit d79719b90. Verify with
  `git log -1 --format=%h -- . ':(exclude)docs'` before anything else.
- Do NOT modify any file in the repo, and do NOT set or override any of these environment variables:
  LLMGE_*, XQC_*, MUJOCO_*, USE_LLM_EMITTERS, SEED, GENE_SBATCH_FLAGS
  (exceptions: LLMGE_VLM_HF_HOME for the vlm tier; LLMGE_HOSTNAME_FILE / LLMGE_SHARED_PORT only if told to share a server).
- Launch only with: scripts/tiered_run.sh <TIER> <REPEAT>   (run it with --dry-run first and show me the output).
- My assignment: tier=<TIER>, repeat=<REPEAT>.
- Baseline runs (control x 3): use Shazeb's shared server exactly as in the doc's "Baseline runs" section
  (LLMGE_HOSTNAME_FILE=/storage/ice1/0/9/swani8/llmge-shared/hostname.log LLMGE_SHARED_PORT=8742).
  If the curl check does not print SERVER-OK, stop: never fall back to starting my own server.
- If anything fails, stop and report the error. Do not fix code or change settings to work around it.
- After launch, do the checklist in docs/TIERED_RUNS.md and report the generation-0 row of generation_stats.csv
  (sliding tier: of mesb/metrics.csv), including the timing columns.
```

## Assignments

| Run | Person | Launched (date) | Status |
|---|---|---|---|
| control_r1 | Caleb |  |  |
| control_r2 | Suhan |  |  |
| control_r3 | Aayush |  |  |
| vlm_r1 |  |  |  |
| vlm_r2 |  |  |  |
| vlm_r3 |  |  |  |
| emitters_r1 |  |  |  |
| emitters_r2 |  |  |  |
| emitters_r3 |  |  |  |
| sliding_r1 |  |  |  |
| sliding_r2 |  |  |  |
| sliding_r3 |  |  |  |
