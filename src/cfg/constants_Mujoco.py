import os
import sys
import numpy as np
import torch
import platform
import yaml

#: Root directory of the repository (auto-detected from this file's location)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Run options (robot, seed, search method) live in run_options.py, shared with
#: run_mesb.py. Imported by path as well, for scripts that load this file
#: outside the src.cfg / cfg package.
try:
    from .run_options import *  # noqa: F401,F403
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_options import *  # noqa: F401,F403

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
#: Gymnasium environment the genes are trained and evaluated on, from ROBOT
#: (run_options.py). Walker2d is the MuJoCo counterpart of the QDWalker task in
#: Nilsson & Cully (GECCO '21); v5 fixes v4's bug of granting healthy_reward on
#: the terminating unhealthy step. Do not pool v4 and v5 results: the reward
#: accounting differs and trajectories diverge numerically within ~10 steps.
MUJOCO_ENV_ID = os.getenv("MUJOCO_ENV_ID", ENV_ID)
#: One-line description of the robot's task, filled into the mutation prompts
#: (<ENV_ID> and <ROBOT_TASK_NOTE> in templates/Mujoco).
ROBOT_TASK_NOTE = ROBOT_PROFILE["task_note"]
#: Keep eval runs configurable without changing the shared project environment.
MUJOCO_EVAL_TIMESTEPS = int(os.getenv("MUJOCO_EVAL_TIMESTEPS", "500000"))
MUJOCO_EVAL_EPISODES = int(os.getenv("MUJOCO_EVAL_EPISODES", "10"))
MUJOCO_EVAL_MAX_STEPS = int(os.getenv("MUJOCO_EVAL_MAX_STEPS", "1000"))
#: Environment-step budget of a whole run, matched by --max_timesteps of the
#: PGA-MAP-Elites baseline. Only genes that trained successfully are charged,
#: MUJOCO_EVAL_TIMESTEPS each, so the run ends after MAX_SUCCESSFUL_EVALS of
#: them rather than after num_generations. Set to 0 to disable the budget.
MUJOCO_TOTAL_TIMESTEPS = int(os.getenv("MUJOCO_TOTAL_TIMESTEPS", "96000000"))
MAX_SUCCESSFUL_EVALS = MUJOCO_TOTAL_TIMESTEPS // MUJOCO_EVAL_TIMESTEPS

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

# Generated by Claude
FORBIDDEN_PATTERNS = [
        ("def forward(", "overrides TD3Policy.forward()"),
        ("def _predict(", "overrides TD3Policy._predict()"),
        ("def make_actor(", "overrides TD3Policy.make_actor()"),
        ("def make_critic(", "overrides TD3Policy.make_critic()"),
        ("self.actor =", "replaces SB3's TD3 actor with an incompatible module"),
        ("self.actor=", "replaces SB3's TD3 actor with an incompatible module"),
        ("self.critic =", "replaces SB3's TD3 critics with an incompatible module"),
        ("self.critic=", "replaces SB3's TD3 critics with an incompatible module"),
        ("self.actor_target", "touches SB3's TD3 target networks"),
        ("self.critic_target", "touches SB3's TD3 target networks"),
        ("ActorCriticPolicy", "on-policy SB3 policy; the seed is a TD3Policy"),
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
# Generated by Claude
#: TD3 takes a gradient step per env step, so 500k timesteps runs far longer
#: than PPO did; 6h stays under the 8h SLURM limit in PYTHON_BASH_SCRIPT_TEMPLATE.
EVAL_NO_PROGRESS_TIMEOUT_SECONDS = int(os.getenv("LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS", str(6 * 60 * 60)))

#: Seed gene and learner (SEED in run_options.py): "td3" is the default above,
#: "xqc" switches the seed gene, prompts, rules and trainer to sota/XQC/llmge.
#: Every process of an island reads the same SEED, so they all agree.
if SEED == "xqc":
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
        f"-env {MUJOCO_ENV_ID} "
        f"-timesteps {XQC_EVAL_TIMESTEPS} -num_seeds {XQC_EVAL_SEEDS} "
        f"-eval_episodes {MUJOCO_EVAL_EPISODES}"
    )
    # Counted from submission, so it includes queue time.
    EVAL_NO_PROGRESS_TIMEOUT_SECONDS = int(os.getenv("LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS", str(6 * 60 * 60)))

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
#: Generations between island migrations. run_improved.py stops at the next multiple
#: of it (islands.py then migrates and relaunches); 0 = no migration, run straight to
#: num_generations (single island, chained controller jobs resume from checkpoints).
migration_gen = int(os.getenv("LLMGE_MIGRATION_GEN", "5"))
#: Slurm time limit of each gene's evaluation job, passed as sbatch -t (overrides the
#: "#SBATCH -t" line of slurm_config.yaml python_bash_script). Empty = keep the yaml's.
GENE_JOB_TIME = os.getenv("LLMGE_GENE_TIME", "")
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
#: foot_geom (right) then foot_left_geom (left); for HalfCheetah, bfoot (back)
#: then ffoot (front). Both learners (train_rl.py, train_xqc_gene.py) write them.
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
#: Pareto-front-per-cell mode (PARETO_FRONT_PER_CELL in run_options.py; Shazeb's
#: Multi-Objective MAP-Elites, Pierrot et al., 2022): each cell keeps a Pareto
#: front over MOME_OBJECTIVES, at most MOME_FRONT_SIZE genes, instead of one
#: elite. Parents are drawn by picking a filled cell uniformly, then a gene of
#: its front uniformly. Does not work with the LLM emitters.
PARETO_FRONT_PER_CELL_MODE = PARETO_FRONT_PER_CELL
#: Results columns, all maximised: speed (sum of the forward reward per episode) and
#: minus effort (sum of reward_ctrl, <= 0).
MOME_OBJECTIVES = ["mean_return_forward", "mean_return_ctrl"]
MOME_FRONT_SIZE = int(os.getenv("LLMGE_MOME_FRONT_SIZE", "4"))
#: Fixed hypervolume reference point (same for every run of a robot, so
#: hypervolume and MOQD score compare across runs): no forward progress, and the
#: largest possible episode effort, ctrl_cost_weight * 6 actuators * 1^2 * 1000
#: steps (6 on Walker2d, 600 on HalfCheetah).
MOME_REFERENCE_POINT = (0.0, -ROBOT_PROFILE["max_episode_effort"])
#: Sentinel reward that sota/MujocoRL/train_rl.py writes when a generated model
#: cannot be built, trained or evaluated. Genes at or below it never enter the
#: archive, so a broken model cannot occupy a niche or be drawn as a parent.
FAILED_EVAL_SENTINEL = -999999.0
#: QD-LLMs of Koh & Wong et al. (GECCO Companion '26), USE_LLM_EMITTERS in
#: run_options.py. When on, an LLM allocator splits every generation's
#: offspring between three LLM emitters, uniform, sparse and directional
#: (Eq. 8). When off, parents are drawn uniformly from the archive and no
#: curiosity or emitter history is kept.
#: The sparse emitter draws a cell with probability proportional to
#:
#:     w_c = (1 - alpha) * s_c + alpha * q_c
#:
#:     s_c = 1 / (1 + n_c)    n_c = occupied cells among the 8 surrounding c
#:     q_c = curiosity of c   recent success of c's elite as a parent
SPARSE_EMITTER_ALPHA = 0.5
#: Curiosity credited to the parent's cell per offspring, by what the offspring
#: did to the archive. Offspring that failed to train change nothing.
CURIOSITY_NEW_CELL = 1.0
CURIOSITY_REPLACED_ELITE = 0.5
CURIOSITY_REJECTED = -0.1
#: Every cell's curiosity is multiplied by this once per generation.
CURIOSITY_DECAY = 0.95
#: Rejections can push w_c below zero; it is clipped to this so that no cell
#: is ever excluded from selection outright.
SPARSE_EMITTER_MIN_WEIGHT = 0.01
#: The directional emitter (Sec. 3.2.3) aims each offspring at an empty target cell far
#: from every occupied one. The source parent is drawn with the sparse weights
#: above; the guide is the elite furthest along the source-to-target direction
#: (Eq. 6). One LLM crossover of the two, told the descriptor gap and the
#: target, produces the child. Within a generation the i-th directional
#: offspring aims at the i-th furthest empty cell, so they do not all chase
#: the same one.
#:
#: What each descriptor means, quoted to the LLM in the directional prompt.
MAP_ELITES_DESCRIPTOR_NOTES = {
    f"foot_contact_{i}": f"fraction of simulation steps the {foot} touches the ground"
    for i, foot in enumerate(ROBOT_PROFILE["foot_names"])
}
#: The allocator (Sec. 3.2, step 1) shows the LLM the archive size and total
#: fitness and each emitter's success rate (archive insertions / new
#: offspring) over the last EMITTER_ALLOCATOR_WINDOW generations, and asks how
#: many of the generation's offspring each emitter should produce.
EMITTER_ALLOCATOR_WINDOW = 10
EMITTER_ALLOCATOR_TEMPERATURE = 0.5
#: Seconds to wait for the allocator's answer before splitting the generation
#: evenly between the emitters instead.
EMITTER_ALLOCATOR_TIMEOUT = 900

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
