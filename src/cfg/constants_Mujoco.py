import os
import sys
import numpy as np
import torch
import platform
import yaml

#: Root directory of the repository (auto-detected from this file's location)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: DATA_PATH (not used for RL, kept for compatibility)
DATA_PATH = "./cifar10"
PACE_ICE = True
RUNLINE_AMP = ""
#: Location where the current seed repo resides
SOTA_ROOT = os.path.join(ROOT_DIR, 'sota/MujocoRL')
#: Location where the network architecture for the seed resides
SEED_NETWORK = os.path.join(SOTA_ROOT, "network.py")
#: Model basename prefix (used in file naming: network_{gene_id}.py)
MODEL = "network"
#: Path to local LLM model path used by server.py for LLM operations
MODEL_PATH = "/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct/"
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "1648"))
#: Directory where LLM-generated model variants are stored
VARIANT_DIR = os.path.join(SOTA_ROOT, "models")
#slurm output
SLURM_OUTPUT_PATH = "run_job_outputs/"
#: The training/evaluation script for RL (relative path for cluster compatibility)
TRAIN_FILE = "sota/MujocoRL/train_rl.py"
ISLAND_TEMP_SCRIPT = os.path.join("src", "island_temp_script_{ISLAND_NUM}.sh")
#: Dedicated uv project used only for Mujoco RL evaluation jobs (relative path for cluster)
MUJOCO_EVAL_PROJECT_DIR = "sota/MujocoRL/eval_env"
#: Keep eval runs configurable without changing the shared project environment.
MUJOCO_EVAL_TIMESTEPS = int(os.getenv("MUJOCO_EVAL_TIMESTEPS", "500000"))
MUJOCO_EVAL_EPISODES = int(os.getenv("MUJOCO_EVAL_EPISODES", "10"))
MUJOCO_EVAL_MAX_STEPS = int(os.getenv("MUJOCO_EVAL_MAX_STEPS", "1000"))

#: Output directory for intermediate generation data
OUTPUT_DIR = "mujoco_rl_output"
PORT = int(os.getenv("LLMGE_PORT", "8169"))

CLUSTER = "pace-ice"
LLM_MODEL = 'llama3.3'
ENVIRONMENT_DIR = os.path.join(ROOT_DIR, ".venv")
SLURM_CONFIG_DIR = os.path.join(ROOT_DIR, "slurm-config/")
_slurm_config_path = os.path.join(SLURM_CONFIG_DIR, 'slurm_config.yaml')
if os.path.exists(_slurm_config_path):
    with open(_slurm_config_path, 'r') as _f:
        _slurm_config = yaml.safe_load(_f)
    LLM_GPU = _slurm_config.get('gpu_selection', 'H200|H100')
    PYTHON_BASH_SCRIPT_TEMPLATE = _slurm_config.get('python_bash_script', '')
    LLM_BASH_SCRIPT_TEMPLATE = _slurm_config.get('llm_bash_script', '')
    ISLANDS_BASH_SCRIPT_TEMPLATE = _slurm_config.get('islands_bash_script', '')
else:
    # Fallback defaults if slurm_config.yaml hasn't been generated yet
    LLM_GPU = 'H200|H100'
    PYTHON_BASH_SCRIPT_TEMPLATE = ''
    LLM_BASH_SCRIPT_TEMPLATE = ''
    ISLANDS_BASH_SCRIPT_TEMPLATE = ''
LOCAL_LLM = True
HOSTNAME_DIR = os.path.join(ROOT_DIR, "hostname.log")

# Multi-island settings used by islands_wrapper.py.
GLOBAL_DATA_PATH = "global_data"
DEFAULT_PROMPT_GROUP = "Mujoco/Normal"
PROMPT_GROUP_TEMPLATE = "templates/{prompt_group}/*.txt"
PROMPTS = f"templates/{DEFAULT_PROMPT_GROUP}/*.txt"
CONSTANT_RULES_PATH = "templates/Mujoco/ConstantRules.txt"

SLURM_MIXT_INPUT_X = SEED_NETWORK
SLURM_MIXT_INPUT_Y = os.path.join(SOTA_ROOT, "models/Menghao/network_x.py")
SLURM_MIXT_OUTPUT = os.path.join(SOTA_ROOT, "models/Menghao/network_z.py")
SLURM_MIXT_TOP_P = 0.15
SLURM_MIXT_TEMPERATURE = 0.1
SLURM_MIXT_APPLY_QUALITY_CONTROL = True
SLURM_MIXT_BIT = 8

ISLAND_CONTROLLER_RUN_NAME = "mujoco_islands_run"
ISLAND_CONTROLLER_NUM_ISLANDS = 1
ISLAND_CONTROLLER_LLMS = "llama3"
ISLAND_CONTROLLER_PROMPT_GROUPS = "Mujoco/Normal"

QC_CHECK_BOOL = False
HUGGING_FACE_BOOL = False
INFERENCE_SUBMISSION = True
CUF_TIMEOUT = 3600 * 30  # 30 hours

#: Whether to run llm-ge locally (True) or distribute across a slurm cluster (False)
LOCAL = False
if LOCAL:
	RUN_COMMAND = 'bash'
	DELAYED_CHECK = False
else:
	RUN_COMMAND = 'sbatch'
	DELAYED_CHECK = True

#: Whether host uses macOS
MACOS = False
DEVICE = 'cuda' if getattr(torch, "cuda", None) is not None and torch.cuda.is_available() else 'cpu'

# Available LLM identifiers.
LLM_QWEN = 'qwen25'
LLM_MIXTRAL = 'mixtral'
LLM_LLAMA3 = 'llama3'
LLM_GEMMA2 = 'gemma2'
LLM_GEMMA3 = 'gemma3'
LLM_DEEPSEEK = 'deepseek'
LLM_GEMINI = 'gemini'
ISLAND_LLMS = [LLM_LLAMA3]
MAX_ISLANDS = len(ISLAND_LLMS)

FORBIDDEN_PATTERNS = [
        ("def forward(", "overrides ActorCriticPolicy.forward()"),
        ("def _predict(", "overrides ActorCriticPolicy._predict()"),
        ("def evaluate_actions(", "overrides ActorCriticPolicy.evaluate_actions()"),
        ("def get_distribution(", "overrides ActorCriticPolicy.get_distribution()"),
        ("def predict_values(", "overrides ActorCriticPolicy.predict_values()"),
        ("self.mlp_extractor =", "replaces SB3's mlp_extractor with an incompatible module"),
        ("self.mlp_extractor=", "replaces SB3's mlp_extractor with an incompatible module"),
        ("self.mlp_extractor.", "accesses unstable SB3 mlp_extractor internals"),
        ("shared_net", "uses removed/unstable SB3 MlpExtractor internals"),
    ]
#: Python run command (uses uv for dependency management)
UV_PYTHON = f"env -u VIRTUAL_ENV uv run --isolated --project {MUJOCO_EVAL_PROJECT_DIR} python"

# resolves to {MODEL}_{gene_id}; train_rl.py expects models.network_<gene_id>
RUNLINE_TMP = "{}_{}"
EVAL_RUNLINE = (
    f"{UV_PYTHON} {{}} "
    "-network models.{} "
    f"-timesteps {MUJOCO_EVAL_TIMESTEPS} "
    f"-eval_episodes {MUJOCO_EVAL_EPISODES} "
    f"-eval_max_steps {MUJOCO_EVAL_MAX_STEPS}"
)
EVAL_NO_PROGRESS_TIMEOUT_SECONDS = int(os.getenv("LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS", str(40 * 60)))

#: Trainer config the genes of this run use: "ppo" (sota/MujocoRL, the default
#: above) or "xqc" (sota/XQC/llmge). Set LLMGE_ALGO in the environment that starts
#: run_improved.py; sbatch passes it on to the LLM and evaluation jobs, so every
#: process of an island agrees on the seed gene, prompts, rules and trainer.
LLMGE_ALGO = os.getenv("LLMGE_ALGO", "ppo")
if LLMGE_ALGO == "xqc":
    SOTA_ROOT = os.path.join(ROOT_DIR, "sota/XQC/llmge")
    SEED_NETWORK = os.path.join(SOTA_ROOT, "network.py")
    VARIANT_DIR = os.path.join(SOTA_ROOT, "models")
    TRAIN_FILE = "sota/XQC/train_xqc_gene.py"
    DEFAULT_PROMPT_GROUP = "Mujoco/XQC"
    PROMPTS = f"templates/{DEFAULT_PROMPT_GROUP}/*.txt"
    CONSTANT_RULES_PATH = "templates/Mujoco/XQC_ConstantRules.txt"
    #: Per-individual budget: steps x seeds (seeds train batched in one process).
    XQC_EVAL_TIMESTEPS = int(os.getenv("XQC_EVAL_TIMESTEPS", "100000"))
    XQC_EVAL_SEEDS = int(os.getenv("XQC_EVAL_SEEDS", "3"))
    FORBIDDEN_PATTERNS = [
        ("import ", "XQC genes return plain dicts; no imports"),
        ("open(", "file access"),
        ("exec(", "dynamic code"),
        ("eval(", "dynamic code"),
        ("__", "dunder access"),
        ("class ", "XQC genes define no classes"),
    ]
    # train_xqc_gene.py runs in the XQC JAX venv (sota/XQC/pace/env.sh activates it).
    EVAL_RUNLINE = (
        "export XQC_REPO=$PWD && source sota/XQC/pace/env.sh && python {} "
        "-network models.{} "
        f"-timesteps {XQC_EVAL_TIMESTEPS} -num_seeds {XQC_EVAL_SEEDS} "
        f"-eval_episodes {MUJOCO_EVAL_EPISODES}"
    )
    # Counted from submission, so it includes queue time.
    EVAL_NO_PROGRESS_TIMEOUT_SECONDS = int(os.getenv("LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS", str(6 * 60 * 60)))
elif LLMGE_ALGO != "ppo":
    raise ValueError(f"LLMGE_ALGO={LLMGE_ALGO!r}: expected 'ppo' or 'xqc'")

#: LLM GPU constraint string for SLURM
LLM_GPU = 'nvidia-gpu'

#: Template script for submitting job for evaluation.
PYTHON_BASH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=evaluateGene
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "nvidia-gpu"
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
echo "Launching Python Evaluation"
hostname

module load cuda
module load uv
export CUDA_VISIBLE_DEVICES=0
unset VIRTUAL_ENV

export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/
export HF_TOKEN="${{HF_TOKEN}}"
export HUGGINGFACE_HUB_TOKEN="${{HF_TOKEN}}"

# Run Python script
{}
"""

#: Template script for submitting a prompt to the LLM
LLM_BASH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=llm_oper
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "{}"
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
echo "Launching AIsurBL"
hostname

module load cuda
module load uv
export CUDA_VISIBLE_DEVICES=0

export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/
export HF_TOKEN="${{HF_TOKEN}}"
export HUGGINGFACE_HUB_TOKEN="${{HF_TOKEN}}"

# Run Python script
{}
"""

"""
Evolution Constants/Params
"""
#: 2-objective setup: maximize reward, minimize parameter count.
FITNESS_WEIGHTS = (1.0, -1.0)
INVALID_FITNESS_MAX = tuple([float(x * np.inf * -1) for x in FITNESS_WEIGHTS])
#: A unique placeholder value used before fitness is evaluated
PLACEHOLDER_FITNESS = tuple([int(x * 9999999999 * -1) for x in FITNESS_WEIGHTS])
NUM_EOT_ELITES = 1
GENERATION = 0
PROB_QC = 0.0
PROB_EOT = 0.0  # Disable EoT for initial RL runs (needs prior elite genes)
num_generations = int(os.getenv("LLMGE_NUM_GENERATIONS", "30"))  # Number of generations
start_population_size = int(os.getenv("LLMGE_POP_SIZE", "32"))  # Starting population size
population_size = int(os.getenv("LLMGE_POP_SIZE", "32"))  # Population size each generation
crossover_probability = 0.35  # Probability of mating two individuals
mutation_probability = 0.8  # Probability of mutating an individual
num_elites = 8
hof_size = 100
max_gen_attempts = 5
migration_gen = 5
"""
MAP-Elites Constants
"""
#: Use MAP-Elites (illumination over a behaviour grid) instead of NSGA-II
#: selection. When False, run_improved.py keeps its original elitist loop.
USE_MAP_ELITES = True
#: Number of bins per behaviour dimension; the archive is MAP_BINS ** 2 cells.
MAP_BINS = 20
#: Behaviour descriptors, as (metric name, min, max). The metric names must
#: match columns written by sota/MujocoRL/train_rl.py. Edit this list to
#: re-dimension the archive - get_bin() adapts to however many entries it has.
MAP_ELITES_DESCRIPTORS = [
    ("mean_distance", -500.0, 2000.0),   # reasonable for HalfCheetah
    # Per-step HalfCheetah control cost is 0.1 * ||a||^2 <= 0.6 (6 actions in [-1, 1]);
    # the old 0-10 range put every individual in the bottom 2 of 20 bins.
    ("mean_control_cost", 0.0, 0.6),
]
#: Results column that ranks occupants competing for the same cell (maximised).
MAP_ELITES_OBJECTIVE = "mean_reward"
#: Sentinel reward that sota/MujocoRL/train_rl.py writes when a generated model
#: cannot be built, trained or evaluated. Genes at or below it never enter the
#: archive, so a broken model cannot occupy a niche or be drawn as a parent.
FAILED_EVAL_SENTINEL = -999999.0

"""
Misc. Non-sense
"""
DNA_TXT = """
⠀⠀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⢀⣠⣤⣶⣶⠶⣶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⣀⣹⣟⣛⣛⣻⣿⣿⣿⡾⠟⢉⣴⠟⢁⣴⠋⣹⣷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠈⠛⠛⣿⠉⢉⣩⠵⠚⠁⢀⡴⠛⠁⣠⠞⠁⣰⠏⠸⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⢻⣷⠋⠁⠀⢀⡴⠋⠀⢀⡴⠋⠀⣼⠃⠀⡼⢿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢻⣆⣠⡴⠋⠀⠀⣠⠟⠀⢀⡾⠁⠀⡼⠁⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠻⣯⡀⠀⢀⡼⠃⠀⢠⡟⠀⢀⡾⠁⢀⣾⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠙⠻⣶⣟⡀⠀⣰⠏⠀⢀⡾⠁⠀⣼⢹⣿⣀⣤⣤⣴⠶⢿⡿⠛⢛⣷⢶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠻⠿⠶⠶⠾⠷⠶⠿⠛⢻⣟⠉⣥⠟⠁⣠⠟⠀⢠⠞⠁⣄⡿⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⠞⠁⢀⡴⠋⠀⣴⠋⠀⣰⠟⠀⣤⡾⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡄⢠⠞⠁⢀⡾⠁⢀⡼⠃⢀⡴⠋⠀⢸⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣷⠋⠀⣰⠏⠀⣠⠟⠀⣰⠟⠁⢀⡴⠛⣿⠀⠀⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣧⡼⠃⢀⡼⠋⢠⡞⠁⣠⣞⣋⣤⣶⣿⡟⠛⣿⠛⠛⣻⠟⠷⢶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠻⣦⣾⣤⣴⣯⡶⠾⠟⠛⠉⠉⠉⣿⡇⢠⡏⠀⣰⠏⠀⢀⣼⠋⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⡾⠀⢰⠏⠀⢠⡞⠁⠀⣠⠞⢻⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣷⠇⢠⠏⠀⣰⠋⠀⣠⠞⠁⠀⢀⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡟⢠⠟⢀⡼⠁⣠⠞⠁⣀⣴⢾⣿⣤⣿⣦⣄⣀⡀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⡟⣠⠏⣠⠞⣁⣴⣾⣿⣿⣿⣿⣿⣿⡏⢹⡏⠛⠳⣦⣄⡀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⢷⣾⣷⠿⠿⠛⠉⠀⠀⠈⠳⣬⣿⡟⣾⠁⠀⣼⠃⠉⠻⠆
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣧⡏⠀⣼⠃⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠁⡼⠁⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣟⡼⠁⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡿⠁⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢙⣃⠀⠀
"""
