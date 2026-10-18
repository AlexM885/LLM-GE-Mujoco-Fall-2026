# Discovering Reinforcement Learning Interfaces with Large Language Models

<p align="center">
  <a href="https://arxiv.org/abs/2605.03408"><img src="https://img.shields.io/badge/arXiv-2605.03408-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://akshat-sj.github.io/limen/"><img src="https://img.shields.io/badge/Project_Page-akshat--sj.github.io/limen-1f7a2c?style=flat-square" alt="Project Page"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License"></a>
</p>

<p align="center">
  <b><a href="https://akshat-sj.github.io/">Akshat Singh Jaswal</a></b> &nbsp;·&nbsp;
  <b><a href="https://x.com/nevashish">Ashish Baghel</a></b> &nbsp;·&nbsp;
  <b><a href="https://paraschopra.com">Paras Chopra</a></b>
  <br>
  <a href="https://lossfunk.com">Lossfunk</a>
</p>

<p align="center">
  <video src="https://github.com/Lossfunk/LIMEN/raw/main/docs/teaser.mp4" controls width="720"></video>
</p>

---

📢 **Accepted at the Reinforcement Learning Conference (RLC) 2026**

**LIMEN** (**L**earning **I**nterfaces via **M**DP-guided **E**volutio**N**) is a framework for automatically discovering the *interface* between an RL agent and its environment, i.e. the observation function and reward function the agent learns on.

Manually designing observations and rewards is the bottleneck for applying RL to new tasks. LIMEN replaces that manual work with LLM-guided evolutionary search: an LLM proposes candidate `(observation, reward)` programs in Python, PPO trains a policy on each, and a MAP-Elites archive evolves better designs over time.

The framework is environment-agnostic. Use it for any task where you have a simulator and a binary success metric.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running on a New Environment](#running-on-a-new-environment)
- [Citation](#citation)

## Features

- **Joint search over observation and reward** as executable JAX-compatible Python programs.
- **MAP-Elites + island model** keeps a structurally diverse population (binned by observation dim and reward complexity).
- **Cascade evaluation** filters hopeless candidates with a 3-stage crash check (syntax, import, JIT) plus a short training run before paying for full evaluation.
- **Multi-environment**: XLand-MiniGrid (discrete), MuJoCo Playground (Panda, Go1), Brax (Ant, Humanoid).
- **Provider-agnostic LLM client** for any OpenAI-compatible endpoint (OpenRouter, OpenAI, Together, Groq, Fireworks, vLLM, Ollama, ...).
- **Model ensembling**: weighted random sampling across multiple LLMs in a single run.

## Installation

Requires Python 3.10+ and a CUDA-capable GPU. Tested on Ubuntu 20.04 / 22.04.

We recommend [`uv`](https://docs.astral.sh/uv/) for the install. Drop in `pip install` anywhere below if you prefer.

```bash
# Clone
git clone https://github.com/Lossfunk/LIMEN.git
cd LIMEN

# Environment + core install
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .

# JAX with CUDA (adjust for your toolkit version)
uv pip install -U "jax[cuda12]"
```

**For MuJoCo tasks** (Panda, Go1), also install MuJoCo Playground:

```bash
git clone https://github.com/google-deepmind/mujoco_playground.git
uv pip install -e mujoco_playground
```

**LLM API key.** Copy `.env.example` to `.env` and add your key. Any OpenAI-compatible endpoint works; default config points at OpenRouter.

```bash
cp .env.example .env
# edit .env: OPENAI_API_KEY=sk-...
```

Switch providers by setting `llm.api_base` in your YAML config (or `--api-base` on the CLI):

| Provider   | `api_base`                          |
|------------|-------------------------------------|
| OpenRouter | `https://openrouter.ai/api/v1`      |
| OpenAI     | `https://api.openai.com/v1`         |
| Together   | `https://api.together.xyz/v1`       |
| Groq       | `https://api.groq.com/openai/v1`    |
| vLLM/local | `http://localhost:8000/v1`          |

## Quick Start

```bash
python run.py --config configs/easy_pickup.yaml \
              --task "Pick up the blue pyramid."
```

Each run writes to a timestamped directory under `runs/`:

```
runs/<timestamp>/
├── best_interface.py        # best evolved (observation, reward) pair
├── evolution_trace.jsonl    # full evolution history
├── database/                # MAP-Elites checkpoint
└── evolution.log            # detailed logs
```

### Example commands

```bash
# XLand-MiniGrid
python run.py --config configs/easy_pickup.yaml          --task "Pick up the blue pyramid."
python run.py --config configs/medium_place_near.yaml    --task "Place the yellow pyramid adjacent to the green square."
python run.py --config configs/hard_rule_chain.yaml      --task "Pick up the blue pyramid (transforms to green ball), place near yellow hex."

# MuJoCo Playground
python run.py --config configs/panda_pick_and_track.yaml
python run.py --config configs/go1_push_recovery.yaml

# Brax locomotion
python run.py --config configs/brax_ant.yaml
```

### Common flags

| Flag              | Description                                                          |
|-------------------|----------------------------------------------------------------------|
| `--config`        | Path to YAML config                                                  |
| `--task`          | Natural language task description                                    |
| `--iterations`    | Number of evolution iterations                                       |
| `--mode`          | `full` (default) / `reward_only` / `obs_only` / `default` / `random` |
| `--model`         | LLM override (e.g. `anthropic/claude-sonnet-4`)                      |
| `--api-base`      | OpenAI-compatible endpoint                                           |
| `--api-key`       | Override `OPENAI_API_KEY`                                            |
| `--timesteps`     | Short (cascade) training timesteps                                   |
| `--timesteps-full`| Full evaluation timesteps                                            |
| `--num-seeds`     | Seeds for multi-seed fitness averaging                               |
| `--resume`        | Resume from a previous run directory                                 |

Full options: `configs/easy_pickup.yaml`.

## What LIMEN Produces

Each run produces a `best_interface.py` containing two functions the policy learns on. See `examples/evolved_interfaces/` for the best programs discovered in our experiments.

```python
import jax
import jax.numpy as jnp

def get_observation(state):
    # Multi-scale geometry, directional indicators, phase encoding, etc.
    return jnp.concatenate([gyro, gravity, joint_offsets, ...])

def compute_reward(state, action, next_state):
    # Potential-based shaping + milestone bonuses + smoothness penalties
    return position_reward + upright_bonus - action_penalty
```

The LLM discovers these designs automatically. No hand-engineered reward shaping required.

## Running on a New Environment

**1.** Subclass `EnvAdapter`:

```python
from limen.adapters.base import EnvAdapter

class MyAdapter(EnvAdapter):
    def get_dummy_state(self):           ...   # for crash-filter validation
    def get_default_obs_fn(self):        ...   # baseline (reward_only ablation)
    def get_default_reward_fn(self):     ...   # baseline (None = env built-in)
```

**2.** Register it:

```python
from limen.adapters import register_adapter
register_adapter("my_env", "my_module.MyAdapter")
```

**3.** Write `contexts/my_env.md` describing the state object, action space, and JAX constraints. This is the API reference the LLM uses. See `contexts/xminigrid.md` for an example.

**4.** Create `configs/my_env.yaml` (copy from an existing config).

**5.** Run:

```bash
python run.py --config configs/my_env.yaml --task "Description of the task."
```

## Citation

```bibtex
@misc{jaswal2026discoveringreinforcementlearninginterfaces,
      title={Discovering Reinforcement Learning Interfaces with Large Language Models},
      author={Akshat Singh Jaswal and Ashish Baghel and Paras Chopra},
      year={2026},
      eprint={2605.03408},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.03408},
}
```
