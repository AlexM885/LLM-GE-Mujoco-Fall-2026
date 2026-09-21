# XQC baseline: research and feasibility summary

Status: research and smoke test done. Scaffold into the fork (sota/MujocoXQC) is NOT built yet. Work was stopped early on purpose: this week's goal was to pick the baseline learner and show it runs. The heavy runs will happen on PACE ICE.

## 1. What I did today

- Chose XQC (ICLR 2026) as the replacement for SB3 PPO and read its code end to end: entry point, env selection, config keys, actor, critic, replay buffer, and how a policy is sampled.
- Got XQC installed and running on my laptop (Apple M5, 10 cores, 16 GB, no GPU). It needed two dependency fixes (section 3).
- Ran the XQC training loop on Walker2d (num_seeds=1, 2 updates per env step) and measured speed: about 86 ms per env step in the update phase.
- Pulled XQC's published per-seed results for Walker2d and Humanoid from its repo and put them in a table (section 3).
- Worked out the plan to plug XQC into MAP-Elites without touching MujocoRL, run_improved.py, the archive code, or templates/Mujoco/. Nothing in the fork changed except this file.

Not done: sota/MujocoXQC scaffold, constants_MujocoXQC.py, the standalone train_xqc.py test, the Humanoid smoke run, and the Humanoid foot-geom check. The Walker2d 10k run was stopped at about 4,000 steps, the Humanoid run never started.

## 2. Why XQC

XQC (Palenicek et al., ICLR 2026) is a soft actor-critic variant with a well-conditioned critic (batch norm + weight norm + a distributional cross-entropy loss). It reports state-of-the-art sample efficiency on 55 proprioceptive control tasks, including Gym MuJoCo, beating SimbaV2, BRO, BRC, CrossQ+WN and MR.Q, while using about 2M parameters versus SimbaV2's 9M. Sample efficiency is exactly what we need, because every individual in LLM-GE gets a small training budget. Runner-up and fallback: SimbaV2. Caveat: the "beats SimbaV2" claim is the paper's aggregate over many tasks. I have not verified it per environment (see Risks).

## 3. Smoke test results

Machine: Apple M5, 10 cores, 16 GB, CPU only. XQC's own entry point (train_parallel.py), env Walker2d-v4, num_seeds=1, start_training=1000.

| Env | Steps | Return | Wall-clock |
|---|---|---|---|
| Walker2d-v4, XQC repo | 3,000 (finished) | -1.50 at step 1, -2.52 at step 3,000 (2 eval episodes) | 190 s total, about 16 s of that is JIT compile |
| Walker2d-v4, XQC repo | 10,000 (stopped at about 4,060) | -1.55 at step 1, no later eval reached | 4 min 3 s when stopped |
| Walker2d-v4, XQC repo | 20,000 (stopped) | none | killed at 14 min, output was hidden by my own pipe |
| Humanoid-v4, XQC repo | not run | none | none |
| Current PPO baseline (PACE run) | | | |

Read this honestly: the returns above are at untrained level. 3,000 steps is far too few to learn anything. The runs prove the pipeline works end to end (env, replay buffer, 2 gradient updates per step, evaluation). They do not prove learning yet.

Speed (measured, Walker2d, steps 1,500 to 3,000): about 86 ms per env step, roughly 11.6 steps per second. Extrapolated linearly on this laptop CPU, one seed:

| Budget | Approx time on this laptop |
|---|---|
| 10k steps | 14 min |
| 300k steps | 7.2 hours |
| 1M steps | 24 hours |
| 32 x 30 = 960 individuals at 300k | about 6,900 hours |

These are laptop CPU numbers, extrapolated from a 1,500-step window. GPU speed on PACE and Humanoid speed are not measured.

Published XQC results (repo file results/xqc.csv, column avg_return, value at env step 1,000,000, all 10 seeds present):

| Env | Method | Final return (mean, std across seeds) | Seeds |
|---|---|---|---|
| Walker2d-v4 | XQC | 6256 (std 525, range 5736 to 7193) | 10 |
| Humanoid-v4 | XQC | 11461 (std 214, range 11103 to 11846) | 10 |
| Walker2d-v4 | SimbaV2 | not in the repo data | n/a |
| Humanoid-v4 | SimbaV2 | not in the repo data | n/a |
| Either env | any other baseline | not in the repo data | n/a |

For reference, XQC's published mean at 300k steps is 5669 on Walker2d-v4 and 9088 on Humanoid-v4. The results folder contains only two files, xqc.csv and xqc_visual.csv, and every row has exp_name = xqc. I confirmed my local clone matches the remote HEAD (9a6832b). Their results use the v4 environments, ours use v5, so these numbers are not directly comparable to our fitness.

Install fixes (local edits to ./xqc/pyproject.toml, not committed anywhere):

1. `jax[cuda12]` and `jaxlib[cuda12]` have no macOS wheels, so `uv sync` failed. Changed to plain `jax==0.4.30` and `jaxlib==0.4.30` (CPU). On PACE, keep the cuda12 extra.
2. XQC pins Gymnasium to a git commit (d92e030, Feb 2025) whose pyproject has a duplicate extra name, which uv 0.12.12 rejects. Changed to `gymnasium==1.0.0`, the nearest release that supports Python 3.9 and has v5 envs. This is close to the pin but not identical.
3. XQC requires Python 3.9 exactly. `uv sync` then takes about 30 s. mujoco-py installed without error but was never imported.
4. The logs show flax and JAX deprecation warnings about "flatten-up-to". They are harmless.

## 4. File structure (planned, not yet added)

```
sota/MujocoXQC/
  README.md
  INTEGRATION_NOTES.md
  network.py            seed genome, get_xqc_config(), get_shaping_config()
  xqc_adapter.py        build + train XQC, return a policy with .predict()
  train_xqc.py          same CLI and CSV contract as train_rl.py
  rollout_eval.py       imports sota/MujocoRL/eval.py by path (one source of truth)
  shaping.py            training-only shaping wrapper, OFF by default
  eval_env/pyproject.toml   Python 3.9 uv project
  third_party/xqc       git submodule pinned to 9a6832b
  models/ results/ stats/   with .gitkeep
src/cfg/constants_MujocoXQC.py
```

Design decisions already made from the code reading (none of this is tested yet):

- XQC's training loop lives inside the Hydra `main()` of train_parallel.py, not in a reusable function. The adapter has to re-implement that loop (about 60 lines) using XQC's real classes (XQCLearner, ParallelReplayBuffer, RewardNormalizer).
- Gym MuJoCo envs are hard-coded to v4 names in xqc/envs/mujoco.py, and `xqc.envs` imports dm_control, humanoid_bench and myosuite unconditionally. The adapter should skip `xqc.envs` and build its own v5 env from the fitness-pinned `make_env` in eval.py.
- The genome surface is the agent config keys in conf/agent/base_agent.yaml and xqc.yaml (learning rates, hidden dims for actor and critic, policy_delay, tau, init_temperature, min_v/max_v, critic_loss, n_critics, BN/WN/LN flags) plus top-level batch_size, updates_per_step, n_steps, start_training, replay_buffer_size.
- The number of value bins is hard-coded to 101 in xqc_learner.py (`{"categorical": 101, "mse": 1}`). It is not a config key, so it cannot be evolved without patching XQC.
- `XQCLearner` accepts `**kwargs`, so a misspelled key is silently ignored. The adapter must validate keys against the real set. Also, the categorical loss hard-codes two critics (`repeats=2`), so `n_critics` must stay 2 with that loss.
- XQC trains with actions rescaled to [-1, 1]. Humanoid's raw action range is +/-0.4. The policy handed to the evaluator must rescale back to the raw range, otherwise fitness is wrong on Humanoid.
- Deterministic actions come from XQC's own sampler with temperature 0.0. The actor uses batch norm, so its batch_stats must travel with the policy.
- MujocoRL's eval_env is Python 3.12 (torch 2.6, gymnasium >= 1.2.3, mujoco >= 3.6). XQC needs Python 3.9 (jax 0.4.30, numpy 1.24.1, mujoco 3.1.6, gymnasium 1.0.0). They cannot share an environment, so XQC gets its own eval_env. eval.py imports stable_baselines3 at module level but only uses it in its own CLI, so importing it by path with a stub for that one module should work. This is untested.
- run_improved.py reads results from `{SOTA_ROOT}/results/{gene_id}_results.csv`, and src/cfg/constants.py is a symlink to constants_Mujoco.py. Switching learners later is repointing that symlink to constants_MujocoXQC.py.

## 5. How MAP-Elites plugs in

XQC becomes the per-individual learner, replacing SB3 PPO. Everything else stays the same: the archive, the foot-contact descriptors (foot_contact_0, foot_contact_1 over 32 x 32 bins), the fitness (true unshaped Walker2d-v5 return over up to 1000 steps), and the results CSV columns that train_rl.py writes. Failed genes write the same -999999.0 sentinel row. The LLM mutates a small genome file (hyperparameters and architecture keys) instead of a PPO policy class. One gap: the prompts in templates/Mujoco/ are worded for SB3 PPO and name sota/MujocoRL/network.py, so an XQC prompt group is needed before evolution can run. I did not touch templates/Mujoco/.

## 6. Risks

- SimbaV2 comparison is unverified. XQC's public results contain no SimbaV2 or other baseline data, so I cannot say from this data whether SimbaV2 beats XQC on Walker2d or Humanoid. The "XQC beats SimbaV2" claim is the paper's aggregate over 55 tasks. Someone needs to pull the per-env numbers from the paper or the SimbaV2 repo before we lean on that claim for these two envs. If SimbaV2 wins on either, that changes the fallback story.
- Published numbers need a sanity check. XQC reports 11461 on Humanoid-v4, while its own code lists a TD3 reference of 5165 for the same env. That may be fine, but it should be cross-checked against the paper before quoting it.
- Compute per individual. At about 86 ms per step on a laptop CPU, 300k steps is about 7 hours. A 32 x 30 run is far out of reach on CPU. We need to measure real throughput on a PACE GPU node before choosing a per-individual budget (300k vs 1M steps) or shrinking population and generations.
- JAX vs SB3 codebase. Two separate Python environments (3.9 with JAX, 3.12 with torch), pinned old JAX (0.4.30), Gymnasium 1.0.0 instead of XQC's exact pin, and a submodule dependency. Expect setup friction for teammates. Also v4 vs v5: the fitness spec needs v5 because v4 grants the healthy bonus on the terminating step, so published v4 returns are not comparable to ours.
- Archive collapse with a strong learner. A good learner may drive most genomes to the same gait, so foot-contact descriptors cluster and few niches fill. The mitigation for Idea 2 is a training-only shaping wrapper (for example a bonus for hitting a target foot-contact ratio), OFF by default. It cannot leak into fitness because evaluation builds a separate raw Gymnasium env with no wrapper, and the evaluator should assert that no shaping wrapper is present. This is a design, not yet built or tested.
- No learning evidence yet. The only runs so far were at most 3,000 finished steps, at untrained-level returns.

## 7. Next steps

1. On PACE: run XQC's own entry point on Walker2d and Humanoid to a real budget (at least 300k steps) to confirm learning and get GPU throughput.
2. Build the sota/MujocoXQC scaffold and constants_MujocoXQC.py as planned, then run the standalone Walker2d test (20k steps, 2 eval episodes, seed 0) and check the CSV columns match train_rl.py exactly. Check that foot geoms are found on both Walker2d-v5 and Humanoid-v5.
3. Plug into run_improved.py on Walker2d with a small population and short budget.
4. Add the shaping wrapper and the Idea 2 allocator.
5. Scale to Humanoid on PACE.

## 8. Slide version

- Switching the learner from SB3 PPO to XQC (ICLR 2026): SAC-based, reports SOTA sample efficiency on 55 proprioceptive tasks with about 2M parameters. Fallback: SimbaV2.
- XQC installs and trains locally; smoke test ran end to end on Walker2d. Speed on a laptop CPU is about 86 ms per env step, so about 7 hours per 300k steps. PACE GPUs are required.
- Published XQC (v4, 10 seeds, 1M steps): Walker2d 6256, Humanoid 11461. SimbaV2 per-env numbers are not in XQC's data and still need to be checked.
- MAP-Elites stays unchanged: XQC is only the per-individual learner, with the same descriptors, fitness and results CSV contract. Adapter, separate Python 3.9 eval env and training-only shaping hook are designed but not yet built.
- Next: measure real throughput on PACE, build the scaffold and run it standalone, plug into run_improved.py on Walker2d, then shaping and Idea 2, then Humanoid.

## Appendix: reproducing the published-results table

```python
import pandas as pd
df = pd.read_csv("xqc/results/xqc.csv")   # columns: exp_name, env_name, seed, metric, env_step, value
for env in ["Walker2d-v4", "Humanoid-v4"]:
    d = df[df.env_name == env]
    f = d[d.env_step == d.env_step.max()]  # 1,000,000 steps, 10 seeds
    print(env, f.value.mean(), f.value.std(), f.seed.nunique())
```
