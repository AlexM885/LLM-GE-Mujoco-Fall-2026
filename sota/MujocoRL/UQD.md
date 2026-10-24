# Uncertainty handling for MuJoCo MAP-Elites (uncertain QD)

This framework adds configurable methods for noisy fitness and behaviour
descriptors to the MuJoCo MAP-Elites pipeline. It covers repeated evaluation,
archive re-evaluation, depth archives and post-hoc ARIA. Every run is
evaluated the same way on held-out seeds, and every run keeps a full record
of its seeds and costs. The original pipeline is kept unchanged as the
`legacy_baseline` preset.

References:
- **UQD**: Flageat & Cully, *Uncertain Quality-Diversity*, [arXiv:2302.00463](https://arxiv.org/abs/2302.00463).
- **ARIA**: Grillotti et al., *Don't Bet on Luck Alone*, [arXiv:2304.03672](https://arxiv.org/abs/2304.03672).
- **Justesen**: Justesen, Risi & Mouret, *MAP-Elites for noisy domains by adaptive sampling*, GECCO '19 Companion. Read from the authors' ITU copy.
- **Deep-Grid**: Flageat & Cully, *Fast and stable MAP-Elites in noisy domains using deep grids*, [arXiv:2006.14253](https://arxiv.org/abs/2006.14253).

Reference implementations checked:
- [Uncertain_Quality_Diversity](https://github.com/adaptive-intelligent-robotics/Uncertain_Quality_Diversity) (JAX/QDax).
- [Deep-Grid_MAP-Elites](https://github.com/adaptive-intelligent-robotics/Deep-Grid_MAP-Elites) (sferes, C++).

## 1. What an archived candidate is

| | |
|---|---|
| Genotype | An LLM-written gene file `models/network_<gene>.py`, holding the architecture and the TD3 hyperparameters. LLM mutation and crossover act on this code. |
| Phenotype | One TD3 training run of the gene, producing a frozen, deterministic actor checkpoint. |
| Archive entry (legacy) | One checkpoint per gene, scored by the mean of 10 episodes. |
| Archive entry (engine) | A *policy solution* `<gene>@t<train_seed>`. That is one exact checkpoint, identified by the SHA-256 of its weights. With `sample_unit = "retrain"` the entry is a *gene solution* whose samples are training runs. |

Children are always retrained from scratch. Weights are never inherited.

### Sources of randomness

| Source | Controlled by | Notes |
|---|---|---|
| Env reset jitter (U±0.005 on qpos/qvel) | episode reset seed | the only noise in a frozen-policy rollout (the actor is deterministic) |
| Policy action sampling | none | `predict(deterministic=True)` |
| Training: weight init, exploration noise, warm-up actions, replay sampling, first reset | training seed | GPU runs are not bit-exact |
| Search: parent choice, crossover/mutation coin flips, LLM temperature draws, Deep-Grid replacement | search seed | seeded for engine presets, unseeded in legacy |
| LLM token sampling | the inference server | not reproducible; whole searches cannot be replayed bit for bit |

**Rollout-only methods estimate how reliable a particular trained policy is.
They say nothing about how reliable its gene is across training runs.** To
measure training-seed variation, use the `exp_me_sampling_retrain_n3` preset
or the optional retraining in the final evaluation. Do not assume that
training noise dominates until it has been measured.

## 2. Configuration

Select a preset with environment variables. They are read by
`src/cfg/constants_Mujoco.py` and validated at import, so an invalid
combination stops the run before any job is submitted.

| Variable | Default | Meaning |
|---|---|---|
| `MUJOCO_UQD_PRESET` | `legacy_baseline` | named preset in `src/cfg/uqd_presets.py` |
| `MUJOCO_UQD_OVERRIDES` | empty | JSON merged over the preset. Changing a search setting relabels the run `exp_<preset>_override`; final-eval and ARIA settings keep the name |
| `MUJOCO_RUN_SEED` | 0 | replicate seed for engine presets |
| `MUJOCO_UQD_RUN_DIR` | `mujoco_rl_output/uqd/<preset>_seed<R>` | where everything about a run is written |

Mutually exclusive choices are named settings: `archive_strategy`,
`sample_unit`, `offspring_sampling`, `deep_grid_replacement`,
`replacement.rule` and `budget.mode`. Independent features are flags:
`archive_reeval.enabled` and `aria.enabled`. Unknown keys, settings that
would be silently ignored, and contradictory combinations raise
`UQDConfigError`. `tests/uqd/test_config.py` lists the rules.

### Presets

| Preset | Category | What it is | Fidelity |
|---|---|---|---|
| `legacy_baseline` | legacy | The unchanged pipeline. One TD3 run per gene with `MUJOCO_EVAL_SEED`, 10 episodes with reset seeds `seed+0..9` shared by all genes, one elite per cell, strict-improvement replacement, unseeded search. `run_improved.py` takes its original code path. | exact previous behaviour |
| `controlled_baseline` | controlled | MAP-Elites under the controlled protocol (below) | reference for ablations |
| `controlled_me_sampling_n32` | controlled | MAP-Elites-sampling, N = 32 episodes | faithful mechanism |
| `controlled_archive_sampling` | controlled | Archive-sampling, D = 2, plus the shared 10-episode initial evaluation | faithful mechanism; initial N differs from the paper's N = 1 |
| `controlled_parallel_adaptive_sampling` | controlled | PAS, D = 2, offspring N = median archive count, floored at 10 | faithful mechanism plus a documented floor |
| `controlled_deep_grid_sampling_n10` | controlled | Deep-Grid, D = 32, random replacement, N = 10 | Deep-Grid-sampling with N = 10 |
| `controlled_baseline_aria` | controlled | `controlled_baseline` + post-hoc ARIA (reduced budget) | ARIA-ME-init analogue |
| `controlled_archive_sampling_aria` | controlled | `controlled_archive_sampling` + post-hoc ARIA | combination |
| `paper_map_elites` | paper | MAP-Elites, 1 episode per offspring, sampling-size budget | UQD baseline |
| `paper_me_sampling` | paper | MAP-Elites-sampling, N = 32 (UQD App. F-A) | faithful |
| `paper_deep_grid` | paper | Deep-Grid, D = 32 (App. F-B) | faithful |
| `paper_deep_grid_sampling` | paper | D = 32, N = 8 (App. F-C) | faithful |
| `paper_archive_sampling` | paper | UQD Alg. 1, D = 2 (App. F-D) | faithful |
| `paper_parallel_adaptive_sampling` | paper | UQD Alg. 2, D = 2 (App. F-E) | faithful |
| `exp_me_sampling_retrain_n3` | experimental | MAP-Elites-sampling over 3 training seeds × 10 episodes | our variant |
| `exp_map_elites_periodic_reeval` | experimental | MAP-Elites, 10 extra episodes per elite every 5 generations, with relocation | our variant |
| `exp_map_elites_margin` | experimental | MAP-Elites where a challenger must win by 25 return units | heuristic |

**Controlled protocol.** All controlled presets share these settings:
- the same offspring count per generation (`population_size`, so the same training cost);
- the same 10-episode initial evaluation;
- run-seed-derived training seeds and per-solution rollout seed streams;
- optionally, the same total interaction budget, via `budget.total_env_steps`: the run stops once measured training plus evaluation env steps reach it.

The only thing that differs between controlled presets is the uncertainty
mechanism. Because the controlled baseline differs from the legacy baseline
only in its seed protocol, the two can be compared directly.

**Paper protocol.** One sample is one episode. The per-generation budget
follows UQD's *sampling-size* rule:
- MAP-Elites-sampling makes `S // N` offspring per generation.
- Archive-sampling and PAS make `(S − archive re-evaluations) // N`. If that comes out below 1, `BudgetUndefined` is raised, which is the case the paper leaves undefined.
- PAS lowers N until at least one offspring fits, as the reference implementation does.

In our pipeline each offspring is also a full TD3 training run. Use
`budget.max_offspring_per_generation` to cap LLM and GPU load; the cap is
recorded in the plan log.

## 3. Samples and statistics (`src/uqd/stats.py`)

- **Rollout sample**: one episode of one checkpoint under one *evaluation protocol*. The protocol covers the environment, the max steps, the deterministic policy, the reward spec, the descriptor definition and the policy device. Samples from different protocols are never pooled.
- **Training sample** (`retrain`): one training run, summarised by the mean of its own `episodes_per_replicate` episodes. A gene's estimate counts each training run once. The within-run episode SD is reported separately as `within_run_fitness_sd`.
- **Initial episodes**: the episodes the training job runs are samples 0..N−1 of the policy's rollout stream, stored one episode per sample. The 10-episode aggregate in the results CSV is never added as a sample, so nothing is double counted. Re-evaluations continue the same stream (sample N, N+1, …) without reusing a seed.
- **Estimates**:
  - fitness is the mean of the valid samples in the window (the most recent 500, as in the UQD reference implementation);
  - the descriptor is the per-dimension mean of those samples;
  - `fitness_sd` is the sample SD (ddof = 1), i.e. the spread of outcomes;
  - `fitness_sem = fitness_sd / sqrt(n)` is the uncertainty of the mean, assuming independent episodes;
  - both are `None` when n < 2.

### Failures

| Kind | Class | Effect on fitness / eligibility | Retry | Cost |
|---|---|---|---|---|
| robot falls, low return | none, a normal sample | counted in the estimate | — | measured |
| `gene_build_error` (import, kwargs, TD3 constructor) | candidate | solution ineligible | no | train time if any |
| `nonfinite_training` (NaN/Inf during learning) | candidate | ineligible | no | partial, unknown steps |
| `nonfinite_rollout` (non-finite action, obs or reward in any episode) | candidate | solution disqualified and removed from the archive. Its other episodes are not used to rank it, so dropping the bad episode can never inflate fitness | no | measured |
| `candidate_timeout` (job started but exceeded the limit) | candidate | ineligible | no | unknown |
| `unclassified_runtime_error` | candidate (conservative) | ineligible; tagged for review | no | unknown |
| `submission_failed`, `job_lost`, `scheduler_timeout`, `missing_artifact`, `malformed_result` | infra | no sample recorded | yes, same seeds, up to `infra.max_retries` (default 2) | recorded as unknown |
| `fitness_spec_mismatch`, `resource_error` (OOM, OSError) | infra | no sample recorded | yes | unknown |

Once retries are exhausted:
- A new solution becomes `infra_unevaluated`. It is excluded from the archive and counted separately, not as a candidate failure.
- An archived solution keeps its previous estimate and its `lost_reevals` count increases (`infra.on_reeval_failure = keep_estimate`).
- A gene the LLM failed to produce is not a candidate at all. It is logged as `llm_generation_failed` and has no environment cost.

## 4. Archive strategies (`src/uqd/archive.py`)

All strategies use the legacy 32 × 32 grid over `foot_contact_0/1`, with the
same binning as `get_bin`.

| Strategy | Insertion / replacement | Selection | Re-evaluation |
|---|---|---|---|
| `map_elites` | one per cell; challenger must be strictly greater (ties keep the incumbent) | uniform over filled cells | optional (experimental): relocate each elite after re-estimation; it is retired if it loses its new cell, and its old cell stays empty |
| `archive_sampling` (UQD Alg. 1) | keep the D best per cell by estimate; ties keep incumbents | the best of a uniformly chosen cell | each generation: empty, re-evaluate every stored solution once, re-add with the updated mean; misfits retire as `displaced` |
| `parallel_adaptive_sampling` (UQD Alg. 2) | as Archive-sampling | as Archive-sampling | as Archive-sampling; offspring N = ⌊median of archive sample counts⌋, computed before the re-evaluation (Alg. 2 l. 4) |
| `deep_grid` (Flageat & Cully 2020) | every offspring is added; a full cell replaces a uniformly random member (`random`), or the oldest (`oldest`, the UQD JAX rule) | uniform filled cell, then roulette on `f − min_cell f` (uniform if all equal; the worst member gets 0) | none |

Descriptor relocation and collisions:
- In the depth archives, a re-estimated solution is re-added into whichever cell its new mean descriptor falls in, competing there under the insertion rule.
- Displaced and replaced solutions are retired: they stay in `solutions.json` and the event log with their reason, but they no longer select or compete.

### Paper fidelity and deliberate deviations

The following are checked against the papers and the reference code:
- MAP-Elites-sampling averages N samples.
- Archive-sampling re-evaluates every stored solution once per generation, keeps the D best, and only the best of a cell is a parent (UQD text and `archive_sampling.py`).
- PAS uses the median count with floor (`int(nanmedian)`) and the decrement-N rule (`parallel_adaptive_sampling.py`).
- Deep-Grid adds every offspring, replaces a random member, and uses fitness-proportional in-cell selection (paper text and sferes `deep.hpp`, `proportional.hpp`).

Deviations:
1. **Grid**: a regular grid (the legacy archive) instead of CVT.
2. **Variation and evaluation**: LLM code mutation plus TD3 training instead of iso-line mutation of weights.
3. **Deep-Grid roulette shift**: the sferes code starts the running minimum at 1, so a cell whose fitnesses are all above 1 is shifted by 1. We always shift by the true minimum, as the UQD JAX code does.
4. **Deep-Grid replacement**: the UQD JAX code replaces the oldest member (seniority); the paper and sferes use random. Random is the default; `oldest` is available.
5. **Moving average**: the UQD paper does not define it. We use the cumulative mean over at most 500 recent samples (the reference implementation's `nanmean` over `max_number_evals = 500`).
6. **Archive-sampling start**: re-evaluation starts at generation 1 (the first loop iteration), which is the sequential order of Alg. 1. The paper's parallel implementation starts at generation 2 only because it batches re-evaluation with offspring evaluation (App. E).
7. **Controlled presets**: the 10-episode initial evaluation and the PAS floor of 10 are our protocol, not the paper's.

### Not implemented

**Justesen et al. (2019) sequential adaptive sampling** is not implemented. It evaluates one challenger at a time and keeps re-sampling both the challenger and the elite it challenges until their sample counts match. It moves drifting elites and resumes their evaluation recursively, so the archive changes in the middle of a single evaluation.

In this pipeline every one of those single samples would be a separate scheduler job. The UQD paper itself calls the method "not parallelisable" and "extremely slow in real-time computation", and introduces PAS as its parallel version. The sferes re-implementation (`adaptive_bd.hpp`) also differs from the paper's pseudo-code: it re-evaluates whole full cells.

Implementing it would need either a persistent rollout worker inside the driver job, or a round-synchronous approximation. The approximation would have to be named as such, not as Justesen's method. No flag for it is exposed.

## 5. Replacement rules

The default is strict improvement of the mean estimate (`replacement.rule = strictly_greater`), which is the legacy rule.

`margin` (experimental presets only) requires `challenger ≥ incumbent + margin`, in return units. It is a heuristic noise floor, not a statistical test:
- it ignores sample counts and variances;
- under adaptive or repeated sampling, the comparisons are neither independent nor fixed in number.

No confidence-based replacement is implemented. A valid one would need:
- a model of the episode return distribution (non-Gaussian; falls make it bimodal);
- a correction for the many sequential comparisons made in each cell;
- a treatment of the data-dependent sample sizes that PAS and archive re-evaluation create.

## 6. ARIA (`src/uqd/aria.py`, `sota/MujocoRL/aria_refine.py`)

ARIA is implemented as an optional, post-hoc stage that operates on the
**frozen actor weights** of the final archive's elites. It follows the paper:

- **Algorithm 2 (Reproducibility-Improvement)**: an ES with N_s samples, mirrored sampling, rank utilities, and gradient = (1/σ) Σ u ε.
- **Eq. 1 objective**: f inside the target cell, and f_min − ‖d − c‖ outside it, where f_min is the running minimum fitness.
- **Algorithm 1**: the improvement phase targets the cell of each input's N_s-sample mean descriptor. The completion phase then moves outward over adjacent empty cells.

Choices the paper text leaves open are recorded in every output (`aria_implementation_choices`):
- utilities: centred ranks;
- optimizer: Adam, lr 0.01;
- adjacency: von Neumann;
- pair choice: uniform random;
- a mirrored pair shares its reset seed;
- non-finite samples get objective −∞;
- a duplicate target cell keeps the higher objective;
- `post_eval_episodes` gives each result a search estimate.

Budgets:
- The paper's Walker budget is N_s = 1440 (×2), N_grad = 75, σ = 0.01, with completion over every empty cell. That is about 216k episodes per cell.
- Presets use N_s = 16, N_grad = 10 and at most 32 completion cells. Any reduced budget is labelled `aria_reduced_budget`.

Refined weights are saved as new checkpoints with new identities (`aria:<sha>`), and their results go to `<run>/aria/`. The unrefined archive and its final evaluation are never modified. **LLM mutations do not inherit actor weights, so ARIA gains are post-hoc refinement of the final archive and are not evidence of better evolutionary search.**

## 7. Seeds (`src/uqd/seeds.py`)

| Stream | Range | Use |
|---|---|---|
| legacy | [0, 10⁷) | `MUJOCO_EVAL_SEED`, `seed+0..9` |
| train | [10⁷, 10⁸) | training seed of replicate r (shared across genes within a run) |
| search_rollout | [10⁸, 1.1·10⁹) | per-solution blocks of 65 536; sample k uses `block+k` |
| aria | [1.1·10⁹, 1.5·10⁹) | ARIA rollouts |
| final_holdout | [1.5·10⁹, 1.9·10⁹) | held-out evaluation, the **same for every run and configuration** |
| final_train | [1.9·10⁹, 2·10⁹) | held-out retraining |

The search RNG (Python `random`, which also drives gene IDs and LLM temperatures) is seeded from `run_seed` and saved at every checkpoint. The final evaluation refuses to run if any held-out seed was used during the search (`check_disjoint`).

## 8. Final evaluation (`src/uqd/final_eval.py`)

Every archived checkpoint is rolled out on M held-out seeds (default M = 64; the UQD paper uses 512). Each search cell then becomes an entity:
- `best_only`: the cell's best solution by search estimate (UQD App. D).
- `in_cell_selector` (Deep-Grid only): draw j picks a member by roulette and uses that member's j-th episode.

Each entity is placed on the common 32 × 32 grid by its **median** held-out descriptor, and when several land in one cell the best **median** fitness is kept. This corrected archive has the same layout for every configuration, including exported legacy runs.

Reported per run:
- coverage;
- raw and normalised QD-score (fixed bounds [0, 5000] for every configuration);
- QD-score loss against the search-time ("illusory") archive;
- max and mean fitness;
- P-score and mean P(d ∈ cell);
- search-cell retention, and the fraction of elites whose corrected cell equals their search cell;
- the held-out fitness SD median and mean SEM;
- mean NDV;
- held-out candidate failures and lost checkpoints;
- search cost and final-evaluation cost.

The UQD Reproducibility-Scores normalise variances per cell by the maximum across runs, so they are computed by `scripts/uqd_compare.py` over the set of runs being compared.

## 9. Costs (`src/uqd/budget.py`)

Every job is logged in `cost_ledger.json` by category: `initial_train`, `initial_eval`, `reeval`, `retrain_*`, `aria`, `final_*`. Each entry records:
- the measured env steps (training timesteps, or the sum of episode lengths);
- the measured wall-clock seconds inside the job;
- the episode count and the status.

Anything not measured, such as a lost job, is recorded as `None`, and the summaries count those entries. Search cost (training plus evaluation) and final-evaluation and ARIA cost are always kept apart. Each snapshot also records `driver_wall_seconds_since_start`, the elapsed wall time including queue waits, which is what you would compare for time to convergence.

## 10. Outputs of an engine run (`MUJOCO_UQD_RUN_DIR`)

| File | Contents |
|---|---|
| `run_metadata.json` | preset, category, resolved config, seed scheme, evaluation protocol, git commit plus dirty flag, host, start time |
| `events.jsonl` | plans, registrations, every archive add/replace/reject/relocate/retire, failures, retries |
| `archive_gen_<g>.json`, `archive_latest.json` | archive snapshot with per-solution estimates and costs |
| `solutions.json` | every solution, including retired and failed ones |
| `cost_ledger.json` | per-job costs |
| `engine_state.pkl`, `search_random_state.pkl` | resume state |
| `final_eval/`, `aria/` | held-out evaluation; post-hoc ARIA |

## 11. Running

All commands run from the repository root, on a compute node or in `sbatch` jobs, never on a login node.

**Unit tests** (seconds; needs no LLM or GPU):
```bash
uv run --isolated --project sota/MujocoRL/eval_env --with pytest python -m pytest --confcutdir=tests/uqd tests/uqd -q
```

**Smoke run of one mode** (real MuJoCo, tiny budget, no LLM; offspring come from a seeded test-only hyperparameter mutation):
```bash
uv run --isolated --project sota/MujocoRL/eval_env python scripts/uqd_smoke.py \
    --preset controlled_archive_sampling --run-dir mujoco_rl_output/uqd_smoke/as \
    --generations 3 --population 3 --timesteps 1500 --final-episodes 4
# add --aria with an ARIA preset; use --overrides '{"aria": {...}}' for a smaller budget
```

**One real run** (LLM server running; same entry point as `island_controller.sbatch`):
```bash
export MUJOCO_UQD_PRESET=controlled_archive_sampling MUJOCO_RUN_SEED=0
uv run python islands_wrapper.py mujoco_rl_output/uqd_ckpt/as_seed0 --num_islands 1 --llms llama3 --prompt_groups "Mujoco/Normal"
```
Leave `MUJOCO_UQD_PRESET` unset (or set it to `legacy_baseline`) for the unchanged pipeline.

**Multi-seed sweep** (writes jobs; submits nothing without `--submit`):
```bash
python scripts/uqd_sweep.py --sweep-dir mujoco_rl_output/uqd_sweeps/pilot --seeds 0,1,2,3,4
```
The default preset list is the legacy baseline, the controlled baseline, one controlled ablation per mechanism, and the Archive-sampling + ARIA combination: 7 presets × 5 seeds. Each run trains `num_generations × population_size` TD3 genes, so check GPU-hours before submitting. Start with 2–3 seeds.

**After a search**:
```bash
# legacy runs only: export the final checkpoint first
uv run --isolated --project sota/MujocoRL/eval_env --with deap python scripts/uqd_export_legacy_archive.py \
    --checkpoint <ckpt>/checkpoint_gen_N.pkl --global-data <ckpt>/global_data/global_gen_N.pkl --seed 0 --out-dir <run_dir>
uv run --isolated --project sota/MujocoRL/eval_env python scripts/uqd_final_eval.py --run-dir <run_dir> [--episodes 512] [--slurm]
uv run --isolated --project sota/MujocoRL/eval_env python scripts/uqd_aria.py --run-dir <run_dir> [--slurm]     # ARIA presets
python scripts/uqd_compare.py <sweep>/runs/* --out-prefix <sweep>/comparison                            # add --stage aria/final_eval for ARIA
```

## 12. Known limitations

- Engine presets are single-island: one `MUJOCO_UQD_RUN_DIR` per island, and no migration. `run_improved.py` refuses a run directory owned by another island.
- `run_improved.py`'s engine path has been import-checked, and its engine calls are the same ones the smoke driver exercises. It has not been run end to end with the LLM server.
- Training jobs run on GPU, so training is not bit-reproducible. Evaluation protocols pin the policy to CPU.
- Gene-level held-out entities average the replicates' `P(d ∈ cell)` relative to each replicate's own corrected cell.
- `retrain` sampling is only supported for MAP-Elites-sampling. Re-training archived genes every generation, as Archive-sampling would require, is rejected as too costly.
