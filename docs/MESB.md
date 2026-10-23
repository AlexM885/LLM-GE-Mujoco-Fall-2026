# MAP-Elites with Sliding Boundaries (MESB) for LLM-GE MuJoCo

> **Sliding Boundaries makes numeric archive boundary positions data-driven.
> It does not eliminate the archive resolution choice.**

This branch adds a MESB quality-diversity loop for LLM-guided evolution of PPO
controllers on HalfCheetah. It is built on `origin/Mujoco-ME-Base@6b73ae167`
("Base Map elites from last semester": HalfCheetah-v4 + PPO + fixed-grid
MAP-Elites), the same starting point as `shazeb/halfcheetah-xqc`. It is
**purely additive**: no existing file was modified. The MESB driver is
`run_mesb.py`; `run_improved.py` (NSGA-II / fixed-grid MAP-Elites) is untouched.

Branch `aayush/mesb-xqc` is the same MESB code on top of
`shazeb/halfcheetah-xqc`, with one addition: `--algo xqc` trains every gene
with Shazeb's XQC island instead of PPO (section 21). It changes none of his
files.

- [1. Problem](#1-problem)
- [2. Fixed-grid limitation](#2-standard-map-elites-fixed-grid-limitation)
- [3. Sliding Boundaries](#3-fontaine-et-al-2019-sliding-boundaries)
- [4–6. ξ, δ, rank rule](#4-xi--buffer_capacity)
- [7–11. Descriptors, objective, initialisation, insertion, remapping](#7-descriptors)
- [12–13. Modes](#12-shadow-mode-mesb-shadow)
- [14. Checkpointing](#14-checkpointing-and-resume)
- [15–16. Running locally and on PACE](#15-run-commands)
- [17–18. Visualisation and video](#17-visualisation)
- [19. Fixed-grid comparison](#19-fixed-grid-comparison)
- [20. Known limitations](#20-known-limitations)
- [21. XQC learner (`--algo xqc`)](#21-xqc-learner---algo-xqc)
- [22. Matched runs on Shazeb's XQC island](#22-matched-runs-on-shazebs-xqc-island-scriptsmesbmatched)
- [Appendix: files, bugs found, tests](#appendix-a-files)

---

## 1. Problem

Earlier MuJoCo MAP-Elites work (`origin/Mujoco-ME-Base@6b73ae167`) binned
controllers into a 20 × 20 grid over **manually guessed** descriptor ranges:
`mean_distance ∈ [-500, 2000]`, `mean_control_cost ∈ [0, 10]`. When the real
behaviour distribution occupies a small part of that box, most cells can
never be reached. Coverage then measures the guess, not the search.

## 2. Standard MAP-Elites fixed-grid limitation

A fixed grid has two design choices per descriptor: the **resolution**
(number of bins) and the **numeric bin edges** (here, equal-width over a
guessed `[lo, hi]`). The second is the fragile one. Behaviours outside the
range are clipped into the edge bins. Most controllers crowd into a few
cells, and the rest stay empty.

In the smoke run below, the legacy ranges put **all 23 controllers into 1 of
16 cells**. The sliding archive spread the same 23 controllers over 15 cells.

## 3. Fontaine et al. (2019): Sliding Boundaries

Fontaine, Lee, Soros, de Mesentier Silva, Togelius & Hoover, *Mapping
Hearthstone Deck Spaces through MAP-Elites with Sliding Boundaries*, GECCO '19.

MESB keeps the number of cells per dimension, but places each dimension's
boundaries at **percentiles of the behaviour values actually observed**. It
recomputes them ("slides" them) periodically as new solutions arrive, then
re-inserts the stored solutions under the new boundaries. Every dimension is
handled independently.

Implementation: `sota/MujocoRL/mesb_archive.py` (`SlidingBoundariesArchive`).
It uses only NumPy and the standard library (no pyribs), and knows nothing
about Slurm, the LLM or PPO. It only sees `ArchiveRecord` values.

## 4. ξ / `buffer_capacity`

The buffer holds the most recent ξ successful evaluations. It is the sample
from which boundaries are estimated.

- `buffer_capacity=None` (default, `--mesb-buffer-capacity none`) means
  ξ = ∞. Every successful evaluation is kept. The archive asserts that the
  buffer size equals `total_seen` at every remap.
- With a finite ξ, the oldest records are evicted (the count is logged as
  `num_evicted`). At the next remap, only buffered records are re-inserted.
  An elite whose record was evicted therefore drops out, as in the paper and
  in pyribs. The full history is still on disk in `evaluations.csv`.
- Failed evaluations never enter the buffer. Records with NaN/∞ values
  raise `InvalidRecordError`.

## 5. δ / `remap_frequency`

`--mesb-remap-frequency 100` (default, paper-oriented) recomputes boundaries
after every 100 **successful** additions. The count starts after
initialisation. Setting it to 0 disables count-based remapping.

## 6. Exact rank rule

For one dimension with `B` bins, let `v[0] ≤ … ≤ v[N-1]` be the buffered values:

```
edge[0] = v[0]                                  (observed minimum)
edge[j] = v[ min(floor(j * N / B), N - 1) ]     for j = 1 .. B-1
edge[B] = v[N-1]                                (observed maximum)
```

Every edge is an observed value. Nothing is interpolated or jittered.
Duplicate edges are allowed (e.g. `N < B` or tied values). The cells between
equal edges are simply unreachable until the data spreads out, and the run
log prints a warning. This matches pyribs' `SlidingBoundariesArchive`.

**Cell index:** `searchsorted(edge[1:B], x, side="right")`, clipped to
`[0, B-1]`. Cell `j` is `[edge[j], edge[j+1])`. A value exactly on an
internal edge goes to the upper cell. Values beyond the observed range sit in
the nearest edge cell until the next remap.

Example (tested): values 0…9 with `B=5` give edges `[0, 2, 4, 6, 8, 9]`, two
values per cell.

## 7. Descriptors

| axis | key | definition |
|---|---|---|
| x | `mean_distance` | mean over evaluation episodes of `final_x − initial_x` |
| y | `mean_control_cost` | mean over episodes of the **episode total** of `−info["reward_ctrl"]` |

This was verified on HalfCheetah-v4 (gymnasium 1.2.3, mujoco 3.14):

- `reset()` returns an empty info dict, so `initial_x` is read from
  `data.qpos[0]` right after the seeded reset. After each step,
  `info["x_position"]` (which equals `qpos[0]`) is used. Distance is never
  inferred from reward.
- `info["reward_ctrl"] = −0.1 · Σ aᵢ²`, checked numerically. If an
  environment lacked `reward_ctrl`, the code would fall back to that
  environment's own `control_cost(action)` method. If neither exists it
  fails; it never guesses a formula.

Also recorded: `mean_control_cost_per_step`, plus per-episode `distances`,
`control_costs` and `rewards`.

HalfCheetah never terminates early, so every episode has the same length.
Total and per-step control cost are then a constant rescaling of each other,
and MESB cell assignment is identical under either. The fixed-grid ranges of
`6b73ae167` were for the per-step version, though; see §19.

## 8. Objective

`mean_reward`, the mean undiscounted evaluation return, decides which
individual is the elite of a cell. The descriptors decide *which* cell. They
are never optimised. `param_count` is metadata only and is never a
tie-breaker.

## 9. Initialisation

No descriptor range is ever assumed. The initial population is created by
LLM mutation of the seed network, exactly as in `run_improved.py`. It is
trained and evaluated, and then `archive.initialize(successful_records)`
computes the first boundaries from those descriptors alone.

- Zero successful individuals raises
  `ValueError("... zero succeeded ...")` and the run stops.
- A single success, or fewer distinct values than bins, works and produces
  duplicate edges (see §6).

## 10. Insertion

`cell = index_of(measures)`. An empty cell takes the candidate. An occupied
cell is replaced only if `candidate.mean_reward > elite.mean_reward`
(strictly). Ties keep the incumbent.

`evaluations.csv` logs `archive_cell_x/y`, `inserted` and `replaced_gene_id`
**at evaluation time**. After a remap, current assignments come from
`mesb/archive_snapshots/` and `mesb/final_archive.csv`.

## 11. Remapping

Remapping happens every δ successful additions. Optionally
(`--mesb-remap-at-generation-end`, a **project adaptation, not the paper's
schedule**), it happens instead once per generation, after all of that
generation's children are inserted.

1. Recompute the edges from the buffer (§6).
2. Clear all cell assignments.
3. Re-insert every buffered record, oldest first. Collisions keep the
   highest `mean_reward` (ties keep the earlier record).
4. Check that no buffered record was lost (ξ = ∞).
5. Update `num_remaps`, `last_remap_at` and `evaluations_since_remap`, and
   append a `remap_events` entry (logged to `boundaries.jsonl`).

Children are inserted in their deterministic slot order, so remap timing
within a generation is reproducible.

## 12. Shadow mode (`mesb-shadow`)

- Parents are chosen exactly as in `nsga2`, and the operators are unchanged.
- Every successful evaluation is *also* inserted into MESB.
- MESB uses its own `numpy.random.Generator` and never touches Python's or
  NumPy's global RNG. Shadow mode therefore cannot change parent selection.
- This is tested. `test_shadow_does_not_change_parent_selection` runs both
  modes and requires identical parents, elites and children. The real-PPO
  smoke run showed the same (§ appendix C).

## 13. Full MESB mode (`mesb`)

Each generation:

1. Take the archive's occupied cells (sorted, deterministic).
2. Sample `population_size` cells uniformly, without replacement unless
   more parents are needed than there are occupied cells.
3. Each sampled cell's elite becomes a parent.
4. Consecutive pairs go through LLM crossover with p = 0.35 (two children);
   then each slot goes through LLM mutation with p = 0.8. These are the
   legacy probabilities and the same crossover-then-mutation order.
5. New children are trained (PPO) and evaluated (standard protocol).
6. Successful children are inserted, and remapping happens on schedule.

NSGA-II / SPEA2 / tournaments are **not** used in this mode. A slot whose
parent was neither crossed nor mutated (≈13 % at the default probabilities)
yields no new gene and costs no evaluation. A failed LLM call keeps the
parent, as in `run_improved.py`.

Legacy `nsga2` mode keeps `run_improved.py`'s selection: `selSPEA2` elites
plus `selTournamentDCD(selNSGA2(...))`. It reads (maximise `mean_reward`,
minimise `param_count`) **by name** from the result JSON.

## 14. Checkpointing and resume

After every generation, `checkpoints/checkpoint_gen_<g>.pkl` stores:

- generation and selection mode
- the full run config
- the population and hall of fame
- the evaluation registry (gene → generation, seed, operator, parents)
- the complete archive `state_dict()`: dims, boundaries, full buffer,
  elites, `total_seen`, remap counters, remap events, archive RNG state
- the Python and NumPy global RNG states
- the evaluation-seed counter

A human-readable snapshot is also written to
`mesb/archive_snapshots/gen_<g>.json`.

Before evaluating generation g, `checkpoints/gen_<g>_plan.pkl` stores the
selected parents, the created children and the RNG state.

**Resume:** `python run_mesb.py --run-name <name> ... --resume`

- Loads the latest checkpoint and restores the archive exactly (it is not
  re-initialised, and boundaries and buffer are kept).
- Truncates log rows from any partially logged generation.
- If a plan exists for the next generation, reuses it (same children).
- Skips every gene whose result JSON already exists, so nothing is
  evaluated twice.
- Refuses to resume if any experiment-defining setting changed
  (`CheckpointMismatchError`). `--num-generations`, `--workers`, timeouts and
  Python commands may change.

This is tested: an interrupted run, including a crash in the middle of a
generation, resumed to the end produces byte-identical `evaluations.csv`,
`metrics.csv`, `boundaries.jsonl` and `final_archive.csv` to an
uninterrupted run.

## 15. Run commands

Environments (both already in the repo): the root uv project runs the driver,
the LLM operators and the analysis (`deap`, `transformers`, `matplotlib`, …);
`sota/MujocoRL/eval_env` runs PPO/MuJoCo (`gymnasium[mujoco]`,
`stable-baselines3`, Python 3.12).

```bash
uv sync                                   # driver + LLM operators + analysis
uv sync --project sota/MujocoRL/eval_env  # PPO / MuJoCo evaluation
EVAL_PY="env -u VIRTUAL_ENV uv run --project sota/MujocoRL/eval_env python"
```

**LLM server.** With `LOCAL_LLM = True` (the default in
`src/cfg/constants_Mujoco.py`), mutation/crossover go to the team's local LLM
server (`server.py`, Llama-3.3-70B at `MODEL_PATH`). Start it before a real run:

```bash
mkdir -p mujoco_rl_output/slurm_logs
sbatch scripts/mesb/llm_server.sbatch   # writes hostname.log; operators wait for it
```

`scripts/mesb/llm_server.sbatch` is `server.sh` without the island
controller, listening on `PORT` from the constants (see Appendix B.3). It
rewrites the tracked `hostname.log`, exactly like `server.sh`; don't commit
that change. `--llm-backend mock` needs no server.

The server job asks for 8 hours, the most the GPU QOS allows for a 2-GPU job
(a longer request stays pending with `QOSMaxGRESMinutesPerJob`). For a longer
run, queue several servers back to back:

```bash
scripts/mesb/start_llm_server.sh 3   # 3 x 8 h, each starts when the previous ends
```

**How MESB calls the LLM.** The wrapper (`src/mesb_llm_operator.py`) still
calls the team's `llm_mutation` / `llm_crossover` / `llm_utils` unchanged, but
adds three guards around them (in `src/mesb_llm_guard.py`):

1. **Code extraction.** `llm_utils.clean_code_from_llm` keeps the *first*
   fenced block. Llama-3.3 often answers with a placeholder block
   (`[modified code]`) or a usage snippet first and the real module later, so
   the child became a syntax error. The wrapper swaps in `extract_code` for its
   own process only: it collects every fenced block (including an unterminated
   last one), drops placeholders, and keeps the last block that parses and
   defines a function or class. `--extraction original` restores the team's
   behaviour.
2. **Child validation before training.** Each child must parse and define
   top-level `get_policy_kwargs` and `get_ppo_kwargs`. An invalid child is
   renamed to `*.rejected.py`, the operator exits with code 3 (logged as
   `INVALID CHILD: <reason>`), and the parent is kept, exactly like any other
   failed LLM call; no PPO time is spent on it. Duplicate definitions (the LLM
   rewrote the whole module) are logged but allowed: Python keeps the last one.
   `--no-validate` turns this off.
3. **Retry on server gaps.** An empty reply (server down, HTTP error, between
   two 8-hour server jobs) makes the wrapper wait for the server again (up to
   `LLM_SERVER_READY_TIMEOUT` seconds, 4 h in the Slurm script) and retry, up to
   `--llm-retries` times (default 3).

The three flags above belong to the operator script; `run_mesb.py` always uses
the defaults.

**Parallel LLM calls.** `--llm-workers N` (default 8, `LLM_WORKERS` in Slurm)
runs up to N operator calls at once. `server.py` batches up to 8 queued
requests (`BATCH_SIZE = 8`), so 8 matches it. Each generation runs its
crossovers in parallel, then its mutations. Parents, temperatures and gene ids
are drawn in the same order as the sequential version, so the result does not
depend on N. Each generation's log line ends with the LLM timing, e.g.
`LLM 32 calls, 1 failed (1 invalid child), 41.2s avg / 88.0s max per call`.

**Unit and integration tests** (NumPy + DEAP only):

```bash
uv run python -m pytest tests/mesb -q
```

`tests/mesb/conftest.py` defaults `LLMGE_AUTO_START_SERVER=0`, so the shared
`tests/conftest.py` does not `sbatch server.sh` just for these tests. Run the
whole suite (`pytest`) only when you want that server.

**Single-gene PPO smoke test** (about 20 s):

```bash
$EVAL_PY sota/MujocoRL/train_gene.py --gene-file sota/MujocoRL/network.py \
    --gene-id seed_smoke --out-json mujoco_rl_output/smoke/seed_smoke.json \
    --model-dir mujoco_rl_output/smoke/models --timesteps 2048 \
    --eval-episodes 2 --eval-max-steps 200
```

**Short experiments** (real PPO; 2–3 generations; a few minutes on 2 CPUs).
Add `--llm-backend mock` to test without the LLM server:

```bash
uv run python run_mesb.py --run-name short_shadow --selection-mode mesb-shadow \
    --eval-python "$EVAL_PY" --num-generations 2 --start-population-size 8 \
    --population-size 4 --num-elites 2 --mesb-dims 4 4 --mesb-remap-frequency 5 \
    --eval-timesteps 2048 --eval-episodes 2 --eval-max-steps 200 --workers 2

uv run python run_mesb.py --run-name short_mesb --selection-mode mesb \
    --eval-python "$EVAL_PY" --num-generations 3 --start-population-size 8 \
    --population-size 6 --mesb-dims 4 4 --mesb-remap-frequency 5 \
    --eval-timesteps 2048 --eval-episodes 2 --eval-max-steps 200 --workers 2
```

**Full experiment** (paper-style defaults: 20×20, δ = 100, ξ = ∞, 500k PPO
steps, 10 × 1000-step evaluation episodes, 30 generations, population 32):

```bash
uv run python run_mesb.py --run-name mesb_full_s0 --selection-mode mesb --seed 0 \
    --eval-python "$EVAL_PY" --workers 16
```

The project adaptation (remap once per generation) is the same command plus
`--mesb-remap-at-generation-end`.

The LLM model name passed to the operators is `--llm-model` (default
`llama3`, as `ISLAND_LLMS` in `constants_Mujoco.py`; with `LOCAL_LLM` every
llama/mixtral name goes to the local server).

All CLI options: `uv run python run_mesb.py --help`. Defaults live in
`src/cfg/constants_mesb.py`, the single source; the resolved values are
written to `config.json`.

## 16. PACE / Slurm

One job runs the whole evolution. PPO evaluations run in parallel on the
job's CPUs, one per CPU. No path is hard-coded: the script `cd`s to
`$SLURM_SUBMIT_DIR`, or to the git top level when run by hand.

```bash
cd <your clone>                                   # submit from the repo root
mkdir -p mujoco_rl_output/slurm_logs
sbatch scripts/mesb/llm_server.sbatch             # 2x H200 LLM server (once)
scripts/mesb/submit_mesb.sh mesb_full_s0 0 mesb   # RUN_NAME SEED SELECTION_MODE
# overrides via environment variables:
DIMS_X=20 DIMS_Y=20 REMAP_FREQUENCY=100 PPO_TIMESTEPS=500000 EVAL_EPISODES=10 \
EVAL_MAX_STEPS=1000 NUM_GENERATIONS=30 \
SBATCH_FLAGS="-A <account> -p <partition>" scripts/mesb/submit_mesb.sh mesb_full_s1 1
# shadow validation run:
scripts/mesb/submit_mesb.sh shadow_s0 0 mesb-shadow
# resubmitting the same RUN_NAME automatically resumes it
```

`submit_mesb.sh` creates `mujoco_rl_output/slurm_logs/` before calling
`sbatch`, since Slurm needs the `--output` directory to exist. Other knobs:

- `CHAIN=N`: submit N driver jobs back to back (`afterany`); each later job
  resumes the run from its last checkpoint, and one that finds the run finished
  only re-runs the analysis
- `LLM_WORKERS` (default 8)
- `LLM_SERVER_READY_TIMEOUT` (seconds an LLM call waits for a restarted
  server; default 14400)
- `BUFFER_CAPACITY`
- `WORKERS`
- `EXTRA_ARGS` (any `run_mesb.py` flag)
- `MESB_ENV_SETUP` (site setup commands; default `module load uv`)
- `DRIVER_PYTHON`, `LLM_PYTHON`, `EVAL_PYTHON`

### Full run recipe

```bash
cd <your clone>
git switch aayush/mesb-sliding-boundaries && git pull
mkdir -p mujoco_rl_output/slurm_logs

scripts/mesb/start_llm_server.sh 3        # LLM server for up to 24 h
# wait until the first server answers:
curl http://$(cat hostname.log):8169/     # {"message":"LLM API is running!"}

CHAIN=2 LLM_WORKERS=8 scripts/mesb/submit_mesb.sh mesb_full_s0 0 mesb

tail -f mujoco_rl_output/slurm_logs/mesb-mesb_full_s0-*.out
```

When the run finishes, `scancel` the server jobs that are still queued.

**Time estimate** (defaults: 30 generations, population 32, 500k PPO steps,
10 × 1000-step evaluation episodes, 16 CPUs):

- PPO: about 1.1 ms per step on one CPU core (measured with the seed
  network), so about 9 minutes per gene. 32 genes on 16 workers take 2
  rounds, about 20 minutes per generation.
- LLM: about 32 calls per generation. At about 50 s per call one after
  another that is about 27 minutes per generation (about 14 h for the run,
  beyond one 8-hour server job). With 8 calls batched together it should drop
  to a few minutes per generation, but the batched speed of the real server
  has not been measured yet; the per-generation log line reports it.
- Total: roughly 12–16 h, so `CHAIN=2` for the driver (16 h per job) and 2–3
  chained servers leave margin. Check the first two generation lines and
  adjust.

## 17. Visualisation

```bash
uv run python sota/MujocoRL/analyze_mesb.py \
    --run-dir mujoco_rl_output/<run_name>
```

This runs headless (Agg) and writes to `<run>/mesb/figures/`:

| file | content |
|---|---|
| `summary.md` | evaluation counts; final occupied/total cells, coverage, raw (and shifted) QD score, best reward, gene and descriptor; final boundaries |
| `final_archive.png` | elites drawn on their **true, unequal** cells (x = distance, y = control cost, colour = reward); boundary lines; all evaluations as dots |
| `snapshots/archive_gen_XXXX.png` | the same, per generation (shared colour scale) |
| `coverage.png`, `qd_score.png`, `best_reward.png`, `mean_elite_reward.png` | per-generation curves from `mesb/metrics.csv` |
| `boundaries_mean_distance.png`, `boundaries_mean_control_cost.png` | every boundary position at every remap |
| `descriptor_distributions.png` | histograms with the final boundaries overlaid |

Raw data:

- `mesb/metrics.csv`
- `mesb/boundaries.jsonl` (every `initialize` / `remap` event, plus a
  `generation_end` snapshot)
- `evaluations.csv`
- `evaluations/<gene>.json` (the canonical result)

## 18. Video export

Evaluate any saved controller on the standard seeds, and optionally render
an MP4:

```bash
RUN=mujoco_rl_output/<run_name>
BEST=$(uv run python sota/MujocoRL/analyze_mesb.py --run-dir $RUN --print-best-model)
MUJOCO_GL=egl $EVAL_PY sota/MujocoRL/behavior_eval.py --model "$BEST" \
    --seed 1000 --episodes 10 --max-steps 1000 --video $RUN/videos/best.mp4
```

Use the run's `--eval-seed` (default 1000), `--eval-episodes` and
`--eval-max-steps` values here.

This reproduces exactly the numbers logged during the run. Use
`MUJOCO_GL=osmesa` on machines without EGL. Rendering is never enabled during
training.

## 19. Fixed-grid comparison

The first comparison should change **only** the boundary approach. Keep the
environment, seed, PPO settings, evaluation protocol, LLM, prompts,
probabilities, descriptors and resolution identical.

**Controlled run of fixed-grid MAP-Elites.** `FixedGridArchive` reproduces
`get_bin` from `6b73ae167` (`min(int(clip((x-lo)/(hi-lo),0,1)·B), B-1)`) and
shares everything else with MESB:

```bash
uv run python run_mesb.py --run-name fixed_s0 --selection-mode mesb \
    --archive-boundaries fixed --fixed-ranges -500 2000 0 500 --seed 0 --eval-python "$EVAL_PY"
uv run python run_mesb.py --run-name mesb_s0  --selection-mode mesb --seed 0 --eval-python "$EVAL_PY"
```

`run_improved.py` at this base also runs this fixed grid by default
(`USE_MAP_ELITES = True`), with the positional-fitness bug and parents drawn
with replacement via `random.choice`. `run_mesb.py --archive-boundaries fixed`
is the like-for-like baseline instead, because it shares MESB's evaluation,
logging and parent sampling. `--selection-mode nsga2` reproduces
`run_improved.py`'s `USE_MAP_ELITES = False` path.

**Offline replay of any run's evaluations** through both archive types (it
also works for `nsga2` runs):

```bash
uv run python sota/MujocoRL/analyze_mesb.py \
    --run-dir mujoco_rl_output/<run> --compare-fixed -500 2000 0 500
```

The replay is descriptive only: parent selection in that run was not driven
by the fixed grid.

**Caveat on the legacy ranges.** In `6b73ae167`, `mean_control_cost` was a
per-step mean with a guessed range of [0, 10]. Here it is the per-episode
total (per-step × 1000 for full episodes). A fair fixed baseline must convert
the range, e.g. `0 10000`, or choose one explicitly. That this choice is
needed at all is the point of MESB.

## 20. Known limitations

- **Resolution is still a hyper-parameter** (`--mesb-dims`). MESB only
  removes the need to choose numeric edges.
- The real Llama-3.3 server was used on PACE for one short run (3
  generations, 0 failed LLM calls). The extraction, validation, retry and
  parallel-call changes were tested against a stand-in server that copies the
  real server's reply format, not against the real server. `--llm-backend
  mock` runs have no scientific meaning.
- `server.py` samples a whole batch with the first request's temperature and
  top-p. With parallel calls, requests of one generation can share a batch,
  so a child may be sampled at a sibling's temperature (all are drawn from the
  same small ranges). Use `--llm-workers 1` for strictly per-call temperatures.
- The LLM often rewrites the whole module rather than one function, leaving
  duplicate definitions (the last one wins). These are logged, not rejected.
- Islands (`islands_wrapper.py`, migration) are not integrated; MESB runs a
  single population.
- Evaluation runs inside a single Slurm allocation (parallel subprocesses).
  There is no per-gene `sbatch` fan-out.
- `nsga2` mode stops with an error if fewer than 4 valid individuals remain.
  `run_improved.py` would re-seed the population instead.
- EoT prompting is off (as in `constants_Mujoco.py`, `PROB_EOT = 0`).
- PPO is seeded (`seed * 1_000_000 + evaluation index`) and runs on CPU by
  default, where repeated runs were bit-identical in testing. GPU training
  is not guaranteed deterministic.
- With finite ξ, elites evicted from the buffer vanish at the next remap
  (paper behaviour).

## 21. XQC learner (`--algo xqc`)

`--algo xqc` swaps the learner and nothing else: archive, selection, LLM
operators, checkpoints and analysis are the same as for PPO. It uses Shazeb's
XQC island from `shazeb/halfcheetah-xqc` (`LLMGE_ALGO=xqc` in his
`run_improved.py` setup) as it is:

| part | XQC profile (`ALGO_PROFILES["xqc"]` in `src/cfg/constants_mesb.py`) |
|---|---|
| starting gene | `sota/XQC/llmge/network.py` (paper settings: critic 4×512, actor 4×256, batch + weight norm) |
| LLM prompts / rules | `templates/Mujoco/XQC/*.txt`, `templates/Mujoco/XQC_ConstantRules.txt` |
| child check | `get_architecture_kwargs` and `get_xqc_kwargs` must exist, then his `gene_overrides()` (keys, types, ranges, `n_critics = 2`) runs before any GPU time is spent |
| team operators | run with `LLMGE_ALGO=xqc`, so `constants_Mujoco.py` gives them his seed gene and `FORBIDDEN_PATTERNS` |
| training | `sota/MujocoRL/train_gene_xqc.py` calls his `sota/XQC/train_xqc.py` with exactly the arguments his `train_xqc_gene.py` uses: 100k steps × 3 seeds trained together, XQC seed 0 for every gene, `final_descriptors=true` |
| budget | 100k steps × 3 seeds per gene; one GPU job per gene (`scripts/mesb/xqc_gene.sbatch`: ice-gpu, NVIDIA, 1.5 h, his job settings) |

`mean_reward` and `std_reward` are his (mean / std of the per-seed 10-episode
means), so a gene needs to be good across 3 training seeds, not lucky on one.
Every episode's return, distance and control cost is kept in the result JSON.
Control cost is stored as the per-episode total, like the PPO runs; his
per-step value is `mean_control_cost_per_step` (total = per-step × 1000 on
HalfCheetah). Sliding boundaries are rank based, so the archive is the same in
either unit; for a fixed grid matching his `MAP_ELITES_DESCRIPTORS` use
`--fixed-ranges -500 2000 0 600`.

**One-time setup on PACE** (the XQC Python 3.9 + JAX environment, in your
scratch space, `~/scratch/xqc-baseline` by default):

```bash
cd <your clone> && git fetch && git switch aayush/mesb-xqc
XQC_REPO=$PWD bash sota/XQC/pace/install_env.sh
git checkout -- sota/XQC/requirements.lock.txt   # the installer rewrites this tracked file
```

**Smoke test** (2 generations of 8 genes, 20k steps, 1 seed; about an hour):

```bash
scripts/mesb/start_llm_server.sh 1
ALGO=xqc NUM_GENERATIONS=1 TIMESTEPS=20000 NUM_SEEDS=1 GPU_JOBS=8 \
  EXTRA_ARGS="--start-population-size 8 --population-size 8" \
  scripts/mesb/submit_mesb.sh xqc_smoke 0 mesb
```

**Full run** (defaults: 15 generations, population 32, 100k × 3 seeds):

```bash
scripts/mesb/start_llm_server.sh 4
ALGO=xqc GPU_JOBS=16 CHAIN=3 scripts/mesb/submit_mesb.sh mesb_xqc_s0 0 mesb
```

The driver job asks for 4 CPUs; each gene is its own GPU job, named
`mesb-<gene id>`, logged to `<run>/logs/eval/<gene>.log`. `GPU_JOBS` caps how
many are in flight; extra sbatch flags for them go in `GENE_SBATCH_FLAGS`
(e.g. `"-A coc"`). A gene job that queues and runs for more than
`--eval-timeout` (6 h) is cancelled and counted as failed.

Time: his NVIDIA jobs took up to 45 min per gene. With 16 in flight a
generation of 32 genes takes about 1.5–2 h plus LLM time, so 15 generations
about 25–35 h. The GPU QOS also counts the LLM server's 2 GPUs, so fewer gene
jobs may run at once; check `squeue -u $USER` in the first generation.

**Matched baseline**: the same command with
`EXTRA_ARGS="--archive-boundaries fixed --fixed-ranges -500 2000 0 600"`
(fixed 20×20 grid over his ranges, everything else identical).

Not available for XQC: `behavior_eval.py` videos (SB3 only). XQC's final
checkpoint is kept in `<run>/models/<gene>/` for a later renderer.

## 22. Matched runs on Shazeb's XQC island (`scripts/mesb/matched/`)

The question "does MESB help?" needs two runs that differ only in the archive.
These runs use **his** pipeline (`run_improved.py`, `LLMGE_ALGO=xqc`), not
`run_mesb.py`, so the only differences from his `xqc_island_run2` are the
archive and two agreed fixes, applied to both arms.

| | fixed arm | MESB arm |
|---|---|---|
| driver | `run_improved_matched.py` = his `run_improved.py` + marked hooks | same |
| archive | his `get_bin`: 20 equal bins over (−500, 2000) m and (0, 0.6) per step | 20×20 sliding boundaries, rebuilt after every generation |
| parents, operators, prompts, validation, gene jobs | his code | his code |
| crossover | `src/llm_crossover_fixed.py` (chunk-index fix) | same |
| gene job time limit | 3 h (his 1.5 h timed out 8 of 83 genes) | same |
| population / generations | 16 / 6 (generations 0–5: his `migration_gen = 5` ends an island there, as in run2) | same |
| starting population | one shared generation 0 | same |

**How the driver is made.** `build_matched_driver.py` copies his
`run_improved.py` and inserts 10 calls into `src/mesb_matched.py` at places it
asserts exist exactly once (marked `# MESB-MATCHED`). If his file changes so a
place is not found, generation stops instead of guessing. The generated
driver goes into the run directory, so each run records the exact code used.
Options come from the environment (`MATCHED_ARCHIVE`, `MATCHED_FIX_CROSSOVER`,
`MATCHED_GENE_TIME`, `MATCHED_STOP_AFTER_GEN0`).

**Sliding archive in his pipeline.** Every gene his `update_archive` would
accept goes into a buffer once. After each generation the boundaries are
recomputed at percentiles of the buffer and the archive is rebuilt (best gene
per cell, ties keep the earlier gene). The archive keeps his format, so his
selection (uniform from the elites), checkpoints, resume and
`xqc_island_report.py` work unchanged. The buffer and boundaries are saved in
the checkpoint (`MESB_STATE`) and logged to `mesb_boundaries.jsonl`. A
checkpoint written by one arm cannot be resumed as the other arm.

**Shared generation 0.** `ROLE=gen0` creates and evaluates the starting genes
once and saves `checkpoint_gen_0.pkl` / `global_gen_0.pkl`. Each arm copies
that snapshot into its own run directory the first time and resumes from it,
so both arms select their first parents from the same 16 evaluated genes.
(`run_improved.py` does not seed its random draws and LLM replies are not
reproducible, so the arms diverge from generation 1; replicate pairs are the
answer to that.)

**Run a pair on PACE** (from `aayush/mesb-xqc`, XQC environment installed):

```bash
scripts/mesb/start_llm_server.sh 3
until curl -s http://$(cat hostname.log):8169/ | grep -q running; do sleep 30; done
scripts/mesb/matched/launch_pair.sh xqc_matched_p1 2     # gen 0, then both arms (2 x 16 h each)
squeue -u $USER
tail -f mujoco_rl_output/slurm_logs/matched-xqc_matched_p1-*.out
```

Each arm is a chain of jobs (`afterany`). His driver runs every launch up to
the next multiple of `migration_gen`, so a part that starts after generation 5
would run generations 6-10. `matched_island.sbatch` therefore exits at once
when the arm already has `checkpoint_gen_5.pkl` (`MATCHED_LAST_GEN`). Pair
`xqc_matched_p1` was launched before this check: its fixed arm finished
generation 5 in part 1 (13.3 h) and part 2 was cancelled during generation 6;
`compare_arms.py` and `posthoc_submit.py` take `--max-gen 5` (default) and
ignore anything later.

Outputs: `mujoco_rl_output/matched/<pair>/{gen0,fixed,mesb}`, each with his
layout (`global_data/`, `island_llama3_Mujoco-XQC/`). Genes, results and stats
go to his `sota/XQC/llmge/{models,results,stats}` (unique gene ids).

**Post-hoc checks** when both arms finish (no new evolution; one GPU job per gene):

```bash
uv run python scripts/mesb/matched/posthoc_submit.py --pair xqc_matched_p1
squeue -u $USER          # 21 jobs: 10 genes per arm + the seed gene
```

Per arm it picks the same number of genes: the top 5 by the run's reward and 5
archive elites as far apart as possible in (distance, control cost) (topped up
with the next-best genes if the archive is small), plus the seed gene once.
`posthoc_eval.py` retrains each from scratch exactly as in the run (100k steps
x 3 seeds, his `train_xqc.py` and arguments) but with XQC seed 1000, so the
training seeds and evaluation episodes are new, then measures without further
training:

| measured | how |
|---|---|
| re-checked reward | mean of the 3 new seeds' 10-episode returns |
| reward under changed conditions | same rollouts with the MuJoCo model edited: torso mass x1.5, all sliding friction x0.5, back-leg or front-leg motor gear x0.5 |
| gait features MESB never optimised | share of steps each foot touches the floor, mean torso height, torso angle spread |

It does this without editing his files: his `train_xqc.py` imports
`llmge_descriptors` at the end of training, so the script runs it in-process
with a wrapper module under that name that returns his normal result and adds
the extra rollouts (written to `posthoc_extra.json`).

**Compare**:

```bash
uv run python scripts/mesb/matched/compare_arms.py --pair xqc_matched_p1
cat mujoco_rl_output/matched/xqc_matched_p1/compare/compare.md
```

`compare.md` reports, fixed vs MESB (generations 0 to `--max-gen`, default 5):

- **High-quality cells:** cells of the shared sliding grid whose elite reaches
  80% of the best reward of both arms (`--hq-fraction`). Coverage counts poor
  cells and QD rewards filling them; this counts only cells that are both new
  and good, which is the goal of the run.

- **Primary (post-hoc):** re-checked top-5 reward (next to the same genes'
  reward during the run, to show selection luck); re-checked best; both vs the
  seed gene; best reward under each changed condition and their mean
  ("robustness": does the archive hold a controller that copes?); share of the
  best reward kept; gait spread (mean pairwise distance of standardised gait
  features).
- **Search and diversity during the run:** mean best-so-far over an equal
  number of evaluated genes (area under the best-vs-evaluations curve), final
  best and top-10 mean, distinct architectures and distinct settings among
  valid genes, valid / invalid / timed-out counts.
- **Secondary, not used for the verdict:** coverage and QD score on shared
  grids. MESB shapes the grid these are measured on, so they partly measure
  MESB itself.

**Verdict rule** (fixed before the results): MESB helps if its best reward
under changed conditions is higher **and** its re-checked top-5 reward is not
lower than the fixed arm's by more than 2%. One pair is evidence, not proof.
`compare.png` plots best reward against genes evaluated, the top-10 mean, and
the two secondary curves.

---

## Appendix A: files

All of these are new; no existing file was changed. The seed network
(`sota/MujocoRL/network.py`), `sota/MujocoRL/eval_env/` and `templates/Mujoco/`
are used as they already are at `6b73ae167`.

| path | purpose |
|---|---|
| `sota/MujocoRL/mesb_archive.py` | `ArchiveRecord`, `SlidingBoundariesArchive`, `FixedGridArchive` |
| `sota/MujocoRL/results_io.py` | canonical per-gene JSON: atomic write, validation, `→ ArchiveRecord` |
| `sota/MujocoRL/behavior_eval.py` | deterministic evaluation (reward, distance, control cost), CLI, MP4 |
| `sota/MujocoRL/train_gene.py` | train one gene with PPO, evaluate, write JSON (always, including failures) |
| `sota/MujocoRL/mesb_evaluator.py` | parallel subprocess evaluation with timeouts; mock evaluator for tests |
| `sota/MujocoRL/mesb_logging.py` | run-directory layout, CSV/JSONL writers, snapshots |
| `sota/MujocoRL/analyze_mesb.py` | figures, `summary.md`, fixed-vs-sliding replay |
| `run_mesb.py` | driver: modes, selection, checkpoints, logging |
| `src/mesb_variation.py` | driver side of LLM variation (legacy probabilities/temperatures/gene ids) |
| `src/mesb_llm_operator.py` | subprocess wrapper around the existing LLM operators (+ mock backend, retry, child validation) |
| `src/mesb_llm_guard.py` | robust code-block extraction and child validation (no torch needed) |
| `src/cfg/constants_mesb.py` | single source of defaults |
| `scripts/mesb/run_mesb.sbatch`, `scripts/mesb/submit_mesb.sh` | Slurm launch |
| `scripts/mesb/llm_server.sbatch` | local LLM server on the constants' `PORT`, no island controller |
| `scripts/mesb/start_llm_server.sh` | N chained 8-hour LLM server jobs |
| `sota/MujocoRL/train_gene_xqc.py` | XQC gene → MESB result JSON (Shazeb's validator and trainer; Python 3.9) |
| `scripts/mesb/xqc_gene.sbatch` | one GPU job per XQC gene |
| `src/mesb_matched.py`, `src/llm_crossover_fixed.py` | hooks for the matched runs; crossover with the chunk-index fix |
| `scripts/mesb/matched/*` | driver generator, Slurm scripts (shared gen 0 + two arms), post-hoc checks (`posthoc_submit.py`, `posthoc_eval.py`, `posthoc_gene.sbatch`), arm comparison |
| `tests/mesb/*` | 139 tests |

Run outputs go under `mujoco_rl_output/`. That directory's own `.gitignore`
(`*`, created on first run) keeps them out of git without editing the root
`.gitignore`.

## Appendix B: pre-existing bugs found (not fixed in place)

1. **Positional fitness parsing** (in this branch's `run_improved.py`, and on
   MosesTheRedSea-main / Mujoco-ME-Base). `train_rl.py` writes
   `mean_reward,std_reward,train_time,param_count,…`. `check4results` keeps
   the first `len(FITNESS_WEIGHTS) == 2` columns. `constants_Mujoco.py`
   documents `(1.0, -1.0)` as *maximise reward, minimise parameter count*,
   but NSGA-II actually **minimised `std_reward`**. `run_mesb.py` reads
   objectives by name (`C.NSGA2_OBJECTIVES`). MESB uses only `mean_reward`.
2. **Crossover chunk off-by-one**, in `src/llm_crossover.py` (here, on
   MosesTheRedSea-main and on `new_main`). Candidates are enumerated over `parts_x[1:]`
   starting from 0, but the LLM output is written to `parts_x[augment_idx]`.
   That is one chunk too early, so it overwrites the import block or the
   policy class. Against a stand-in LLM server, 3/3 MuJoCo crossover
   children from the original function failed to import (`NameError: nn`);
   3/3 from the corrected version loaded. The MESB wrapper
   defaults to a corrected re-implementation built from the same helpers,
   templates and random-draw order. `--llm-crossover-impl original` calls the
   untouched function.
3. **Port mismatch** at `6b73ae167`: `server.sh` starts uvicorn on port
   8137, while `constants_Mujoco.py` (`PORT = 8169`) makes the operators call
   8169. Fixed upstream in `6f51a9361`; here `scripts/mesb/llm_server.sbatch`
   reads `PORT` from the constants instead of editing `server.sh`.
4. `tests/conftest.py` starts an LLM server (`sbatch server.sh`, which also
   submits `island_controller.sbatch`) at the start of *any* pytest session,
   unless `LLMGE_AUTO_START_SERVER=0`.
5. `hostname.log` is committed although `.gitignore` lists it; every server
   start rewrites it.

## Appendix C: verification performed

- `pytest tests/mesb`: 77 passed. This covers boundary ranks, cell indexing
  edge cases, insertion, remap/collisions, buffers, duplicates,
  serialisation, seeded sampling, failed/malformed/NaN results, the
  shadow-vs-nsga2 parent identity, MESB parents ⊆ elites, resume
  equivalence (including a mid-generation crash), config-mismatch refusal,
  generation-end remapping, the fixed-grid baseline, and a synthetic
  12-generation drift scenario.
- Real PPO (2048 steps, 2 × 200-step episodes, HalfCheetah-v4, CPU):
  - a single gene evaluates, and repeated runs are bit-identical;
  - broken, missing and NaN genes each yield `status: failed`;
  - standalone re-evaluation reproduces the logged metrics;
  - the MP4 renders headlessly with EGL.
- Real-PPO `nsga2` vs `mesb-shadow` (seed 7): identical parents, elites and
  children in every generation.
- Real-PPO `mesb`, 3 generations: parents were always previous-archive
  elites; remaps at 13, 18 and 23 successful evaluations; boundaries moved;
  resume to generation 4 added no duplicate rows.
- After rebasing onto `6b73ae167`: 77/77 tests pass (the shared conftest did
  not start a server), and a 2-generation real-PPO `mesb` run through the
  existing LLM operators and a stand-in `server.py` made 18 LLM calls
  (6 create, 6 crossover, 6 mutation), all children loaded, 14/14 trained and
  evaluated.
- LLM-side changes (extraction, validation, retry, parallel calls):
  - `pytest tests/mesb`: 106 passed. New tests cover the exact reply shape
    seen on PACE (placeholder block first, real module later), language tags,
    unterminated blocks, unusable replies, child validation, parallel equals
    sequential for the same seed, an invalid child keeping its parent, and
    retry after an empty reply.
  - On the PACE reply shape, the original extraction produced a child with a
    syntax error (rejected, exit code 3, parent kept); the new extraction
    produced the intended edit.
  - A 3-generation real-PPO `mesb` run with `--llm-workers 4` against a
    stand-in server that failed its first 3 requests: 4 requests reached the
    server at once, the 3 failures were retried, 25 of 26 operator calls
    produced a valid child and the invalid one kept its parent.
- XQC learner (`aayush/mesb-xqc`):
  - `pytest tests/mesb`: 126 passed, including the XQC profile, his gene
    validator on valid and broken genes, the operator, the trainer adapter
    (command identical to `train_xqc_gene.py`, Python 3.9 syntax), the Slurm
    evaluator (fake `sbatch`: results, timeout + `scancel`, missing result) and
    resuming pre-XQC runs.
  - Real LLM path with the XQC prompts (stand-in server): 21 operator calls,
    all valid XQC genes, real edits.
  - Real XQC training (Python 3.9, JAX on CPU, tiny gene, 1,200 steps):
    `train_gene_xqc.py` alone, a 2-generation MESB run with local processes,
    and a run through `xqc_gene.sbatch` with a fake `sbatch`.
- Matched runs (`scripts/mesb/matched/`):
  - `pytest tests/mesb`: 135 passed, including the driver generator against
    the current `run_improved.py` (every original line kept; a changed source
    is refused), the sliding archive hook (percentile edges, best per cell,
    duplicates and invalid genes skipped, checkpoint round trip), the Slurm
    time and crossover hooks, the resume guard between arms, and the fixed
    crossover (keeps the gene valid where the original drops
    `get_architecture_kwargs` or the `ALGO` header).
  - His real pipeline end to end on this machine (fake `sbatch`, stand-in LLM
    server, tiny real XQC genes): shared generation 0 saved, both arms copied
    it and ran generations 1–2 side by side, crossover jobs used the fixed
    script, gene jobs carried `-t 3:00:00`; the MESB arm was stopped and
    resumed from generation 2 with its buffer and archive intact; his report
    and `compare_arms.py` ran on both arms.
- Post-hoc checks: 139 tests pass (equal-budget efficiency, gene selection,
  post-hoc summary and verdict, each changed condition edits the MuJoCo
  model). In the scratch harness, `posthoc_submit.py` picked genes from both
  arms, 9 of 9 checks (real XQC retraining on CPU) succeeded, and
  `compare_arms.py` filled every primary metric. A first version of the
  slippery-floor condition changed only the floor and had no effect (MuJoCo
  uses the larger friction of the two touching surfaces); it now halves every
  surface's friction.
