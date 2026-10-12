# Baselines for LLM-GE on Walker2d

Two published methods, adapted to run on exactly the problem LLM-GE solves in `sota/MujocoRL`, so their results can be compared with LLM-GE's.

| Folder | Method | Paper |
|---|---|---|
| `PGA-MAP-Elites-master/` | PGA-MAP-Elites | Nilsson & Cully, "Policy Gradient Assisted MAP-Elites", GECCO 2021 |
| `LIMEN-main/` | LIMEN | Jaswal, Baghel & Chopra, "Discovering Reinforcement Learning Interfaces with Large Language Models", arXiv 2605.03408 |
| `compare_baselines.py` | Puts both on the LLM-GE grid at a common budget | |

Both are modified copies of the authors' code. Each folder's own README lists what was changed.

## The shared problem

All three methods use the same task, scoring and budget unit, taken from the files one level up.

- **Environment:** Gymnasium `Walker2d-v5`, built by `../eval.py` with the reward spec pinned.
- **Fitness:** the environment's own episode return (forward velocity + 1 per healthy step − control cost), at most 1000 steps per episode.
- **Behaviour descriptor:** the proportion of steps each of the two feet is in contact with the ground, read from the MuJoCo contact buffer in `../eval.py`.
- **Network:** the seed in `../network.py`: a 128×128 actor (19,590 parameters) and twin 256×256 critics, with the TD3 settings from the PGA-MAP-Elites paper.
- **Budget:** counted in environment timesteps used for training. One LLM-GE gene or one LIMEN candidate is one TD3 training of 500,000 steps. Candidates that fail before training are not charged.

The budget is currently set to 96,000,000 steps (192 trainings) in all three: `MUJOCO_TOTAL_TIMESTEPS` in `src/cfg/constants_Mujoco.py`, `--max_timesteps` in `PGA-MAP-Elites-master/configure_experiment/Walker2d_config.txt`, and `max_total_timesteps` in `LIMEN-main/configs/gym_walker2d.yaml`.

## What each method does

**LLM-GE** (the method being evaluated). An LLM edits the code of the learner: the policy and critic network and the TD3 hyperparameters in `network.py`. Each gene is scored by training TD3 from scratch and evaluating the policy over 10 episodes. Genes are kept in a 32×32 MAP-Elites grid over the two foot-contact proportions. New genes come from LLM mutation and LLM crossover of genes sampled from the grid, 32 per generation.

**PGA-MAP-Elites.** No LLM. It evolves the weights of a fixed network. Each candidate is a policy that is run for a single episode, with no training of its own. All episodes feed one shared replay buffer that trains one shared critic. Half of each batch of 100 new policies comes from random variation of archive members, the other half from a few policy-gradient steps against the critic. The archive has 1024 niches over the same two foot-contact proportions.

**LIMEN.** An LLM writes the task interface: a function for what the agent observes and a function for the reward it is trained on. The learner is fixed (TD3 with the seed network). Each candidate is scored by training TD3 on the evolved interface and then measuring the environment's own return, so the evolved reward cannot inflate the score. Candidates are kept in a MAP-Elites archive over observation size and reward-code complexity, on 3 islands. New candidates come from LLM mutation only; the prompt includes the parent, the best candidates so far, recent failures and feedback from the parent's training.

## How they compare with LLM-GE

| | LLM-GE | LIMEN | PGA-MAP-Elites |
|---|---|---|---|
| Uses an LLM | Yes | Yes | No |
| What is searched | The learner (network and hyperparameters) | The task interface (observation and reward) | The policy weights |
| What is fixed | Observation and reward | Network and TD3 settings | Network shape, observation and reward |
| One evaluation | A 500k-step TD3 training | A 500k-step TD3 training | One episode, at most 1000 steps |
| Evaluations in 26M steps | 52 | 52 | About 70,000 |
| Variation | LLM mutation and LLM crossover | LLM mutation | Random variation and policy gradient |
| Archive axes | Foot contact of each foot | Observation size and reward complexity | Foot contact of each foot |
| Archive shape | 32×32 grid | 5×5 grid on each of 3 islands | 1024 CVT niches |
| Fitness of an elite | Mean of 10 episodes after training | Mean of 10 episodes after training | One episode |
| Shared learning between candidates | None; each trains from scratch | None; each trains from scratch | One critic and replay buffer |

Points that follow from this:

- LLM-GE and LIMEN are the same kind of search over opposite halves of the problem. LLM-GE changes the learner and keeps the task; LIMEN changes the task and keeps the learner.
- PGA-MAP-Elites evaluates over a thousand times more candidates for the same number of steps, because its candidates are not trained individually. This is why it fills far more of the behaviour space.
- Equal timesteps is not equal compute. PGA-MAP-Elites runs on CPU only; LLM-GE and LIMEN need one GPU training per candidate, plus LLM inference.
- LIMEN does not search over foot contact, so its coverage of the foot-contact grid is a by-product, not a goal.

## Running them

All commands are run from the repository root on the cluster.

### PGA-MAP-Elites

One CPU-only job. It needs no LLM server.

```bash
sbatch sota/MujocoRL/Baseline/PGA-MAP-Elites-master/slurm/pga_run.sh [SEED]
```

A different step budget than the config:

```bash
PGA_MAX_TIMESTEPS=26000000 sbatch sota/MujocoRL/Baseline/PGA-MAP-Elites-master/slurm/pga_run.sh 0
```

Output goes to `baseline_runs/pga_map_elites/seed<SEED>_job<JOBID>/` and the job log to `slurm-pga-<JOBID>.out`. A 96M-step run took about 1.5 to 2 hours on 32 cores. There is no resume.

### LIMEN

LIMEN uses the same LLM server as LLM-GE, so start that first.

```bash
sbatch server.sh
sbatch sota/MujocoRL/Baseline/LIMEN-main/slurm/limen_run.sh [RUN_DIR]
```

`limen_run.sh` is the controller. It asks the LLM for a batch of 32 candidates, submits each training as its own GPU job (`slurm/train_job.sh`), waits for them, and repeats. It stops before its 8-hour limit and resubmits itself until the budget is spent.

Output goes to `baseline_runs/limen/walker2d/`: the archive, `best_interface.py`, `trained_models/` and `job_logs/`. The controller log is `slurm-limen-<JOBID>.out`.

Things to know:

- The LLM server ends after 8 hours and LIMEN does not restart it. Resubmit `server.sh` when it ends; LIMEN waits up to an hour for it.
- Do not run LIMEN and LLM-GE against the same server at the same time. The server applies one request's temperature to a whole batch.
- Running the same command again resumes the run in `RUN_DIR`. Pass a new `RUN_DIR` to start a separate run.
- One training takes about 30 minutes. A batch of 32 took about 4 hours when only 3 to 4 trainings ran at once.

### Comparing on the LLM-GE grid

```bash
python sota/MujocoRL/Baseline/compare_baselines.py \
    --limen baseline_runs/limen/walker2d \
    --pga baseline_runs/pga_map_elites/seed0_job<JOBID> \
    --timesteps 26000000
```

This cuts LIMEN at the given number of training steps, takes the PGA-MAP-Elites archive snapshot closest to it, and places every solution on the 32×32 foot-contact grid the way `run_improved.py` does. It prints best fitness, coverage and QD-score. QD-score is the sum of the best fitness in each filled cell, with no offset, which is the `QD-score(raw)` LLM-GE prints. If the PGA-MAP-Elites run has finished, it also prints a row for the final archive re-evaluated over 10 episodes.

## Results so far

One run per method, so these are preliminary. LLM-GE and LIMEN are compared at 26M steps because that is what the available LLM-GE run used: 52 successful trainings.

### At about 26M training steps

| | Steps | Trainings | Best fitness | Cells filled | QD-score | Mean per filled cell |
|---|---|---|---|---|---|---|
| LLM-GE | 26M | 52 | 4775.6 | 30 (2.9%) | 81,198 | 2707 |
| LIMEN | 26M | 52 | 4280.8 | 31 (3.0%) | 47,230 | 1524 |
| PGA-MAP-Elites, run 1 | 27.6M | n/a | 3071.9 | 763 (74.5%) | 992,771 | 1301 |
| PGA-MAP-Elites, run 2 | 25.9M | n/a | 3024.0 | 767 (74.9%) | 895,482 | 1168 |

The PGA-MAP-Elites rows score each policy on one episode, which overstates its coverage (see below). The other two rows average 10 episodes.

### PGA-MAP-Elites at 96M steps, re-evaluated over 10 episodes

| | Best fitness | Cells filled | QD-score | Mean per filled cell | Policies evaluated |
|---|---|---|---|---|---|
| Run 1 (job 6202104) | 4042.4 | 575 (56.2%) | 831,115 | 1445 | 229,100 |
| Run 2 (job 6219749) | 4539.0 | 566 (55.3%) | 827,924 | 1463 | 209,700 |

Both runs used seed 0; they differ because the parallel workers do not finish in the same order each time. Coverage and QD-score agree to within about 2%, while the best single policy varies by about 12%.

### The LLM-GE run in detail (job 5958942, 27 September)

| Phase | Genes | Failed | Trained | Cells filled after | QD-score after |
|---|---|---|---|---|---|
| Initial population | 32 | 25 | 7 | 7 | 20,128 |
| Generation 1 | 32 | 25 | 7 | 10 | 28,007 |
| Generation 2 | 32 | 23 | 9 | 14 | 38,552 |
| Generation 3 | 32 | 21 | 11 | 21 | 62,583 |
| Generation 4 | 32 | 14 | 18 | 30 | 81,198 |

The job was cut off by its 8-hour limit during generation 5.

### What the results show

- LLM-GE has the best single policy and the strongest elites at 26M steps.
- LIMEN's best policy is close to LLM-GE's, found with a learner it is not allowed to change. Its other elites are weaker.
- LLM-GE and LIMEN fill almost the same number of cells from the same number of trainings.
- PGA-MAP-Elites fills far more cells, with weaker individual policies at 26M. By 96M its best policy is in the same range as the other two.

### Caveats

- **One run each.** Only PGA-MAP-Elites has a repeat.
- **Single-episode scores are optimistic.** An archive keeps a policy in an unusual cell if one episode happened to put it there. When the same PGA-MAP-Elites archive is re-scored over 10 episodes, coverage falls from about 765 cells to about 570 and the best score drops. There is no 10-episode score for PGA-MAP-Elites at 26M yet.
- **LLM-GE's best gene came from the initial population.** Coverage and QD-score grew every generation, but the top score did not improve. Whether evolution adds anything beyond sampling has not been tested.
- **108 of LLM-GE's 160 genes failed** and were never trained. The cause has not been investigated.
- **The LLM-GE run's settings are not confirmed from its log.** It started shortly after the commit that switched to TD3; the stats files from that run record the algorithm and step count.
- **LLM-GE's second objective is the standard deviation of the reward,** not the parameter count. The fitness tuple takes the first two columns of the results file, and parameter count is the fourth. The archive uses mean reward only, so the numbers above are unaffected.
- **Some LIMEN candidates are duplicates.** Five trainings in its second batch returned an identical score, which indicates the LLM returned the same code. LIMEN has no duplicate check.

## Not done yet

- A PGA-MAP-Elites run stopped at 26M and re-evaluated over 10 episodes.
- LLM-GE and LIMEN run to the full 96M.
- Control runs for LLM-GE: the seed network trained repeatedly with no LLM, and LLM-generated genes with no evolution.
- Retraining each method's best candidate on several seeds.
