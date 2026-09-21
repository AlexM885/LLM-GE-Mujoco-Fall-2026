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
#: Gymnasium environment the genes are trained and evaluated on. Walker2d is
#: the MuJoCo counterpart of the QDWalker task in Nilsson & Cully (GECCO '21).
#: v5 is the current revision; it fixes v4's bug of granting healthy_reward on
#: the terminating unhealthy step. Do not pool v4 and v5 results: the reward
#: accounting differs and trajectories diverge numerically within ~10 steps.
MUJOCO_ENV_ID = os.getenv("MUJOCO_ENV_ID", "Walker2d-v5")
#: Keep eval runs configurable without changing the shared project environment.
MUJOCO_EVAL_TIMESTEPS = int(os.getenv("MUJOCO_EVAL_TIMESTEPS", "500000"))
MUJOCO_EVAL_EPISODES = int(os.getenv("MUJOCO_EVAL_EPISODES", "10"))
MUJOCO_EVAL_MAX_STEPS = int(os.getenv("MUJOCO_EVAL_MAX_STEPS", "1000"))

#: Output directory for intermediate generation data
OUTPUT_DIR = "mujoco_rl_output"
PORT = 8169

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
    f"-env {MUJOCO_ENV_ID} "
    f"-timesteps {MUJOCO_EVAL_TIMESTEPS} "
    f"-eval_episodes {MUJOCO_EVAL_EPISODES} "
    f"-eval_max_steps {MUJOCO_EVAL_MAX_STEPS}"
)
EVAL_NO_PROGRESS_TIMEOUT_SECONDS = int(os.getenv("LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS", str(40 * 60)))

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
num_generations = 30  # Number of generations
start_population_size = 32  # Starting population size
population_size = 32  # Population size each generation
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
#: Bins per behaviour dimension. The paper discretizes the QDWalker behaviour
#: space into 1024 niches, 32 bins per dimension, over a 2-D descriptor.
MAP_BINS = 32
#: Behaviour descriptors, as (metric name, min, max). The metric names must
#: match columns written by sota/MujocoRL/train_rl.py. Edit this list to
#: re-dimension the archive - get_bin() adapts to however many entries it has.
#:
#: Following Nilsson & Cully (GECCO '21), the descriptor is the proportion of
#: simulation steps each foot spends in contact with the ground. Walker2d has
#: two feet, so the archive is 32 x 32 = 1024 niches. Both components are
#: proportions and therefore bounded to [0, 1] by construction.
#: foot_contact_0 / _1 are ordered by MuJoCo geom id: for Walker2d-v5 that is
#: foot_geom (right) then foot_left_geom (left).
MAP_ELITES_DESCRIPTORS = [
    ("foot_contact_0", 0.0, 1.0),
    ("foot_contact_1", 0.0, 1.0),
]
#: Results column that ranks occupants competing for the same cell (maximised).
#:
#: This is the episode fitness F: the undiscounted return over a rollout of at
#: most 1000 steps, averaged over MUJOCO_EVAL_EPISODES rollouts.
#:
#:     F = sum_t ( r_forward(t) + r_healthy(t) - c_ctrl(t) )
#:
#:     r_forward = 1.0   * v_x                 (v_x = dx/dt)
#:     r_healthy = 1.0   if healthy else 0     (z in [0.8, 2.0], pitch in [-1, 1])
#:     c_ctrl    = 0.001 * ||a_t||^2           (sum of the 6 squared torques)
#:
#: Leaving either healthy range terminates the episode. The weights are pinned
#: explicitly in sota/MujocoRL/eval.py (WALKER_REWARD_SPEC) and validated at the
#: start of every evaluation job, so a gymnasium upgrade cannot silently change
#: what fitness means. The three terms are recorded per gene as
#: mean_return_forward / _healthy / _ctrl and sum to mean_reward.
#:
#: Indicative bands: <50 falls immediately, 300-800 shuffling,
#: 3000+ sustained locomotion.
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
