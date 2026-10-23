"""Default configuration for the MESB MuJoCo driver (``run_mesb.py``).

This is the single source of defaults; every value can be overridden on the
``run_mesb.py`` command line, and the resolved values are written to
``<run_dir>/config.json``. Evolution defaults mirror
``src/cfg/constants_Mujoco.py`` (this branch's base, 6b73ae167) so that MESB
and the legacy NSGA-II loop differ only in what the experiment varies.

Deliberately free of torch / gymnasium imports so the driver stays light.
"""

import sys
from pathlib import Path

# Robot and seed learner come from the shared run options (run_options.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_options  # noqa: E402

#: Repository root, derived from this file's location (never hard-coded).
ROOT_DIR = Path(__file__).resolve().parents[2]
#: Seed network / domain.
SOTA_ROOT = ROOT_DIR / "sota" / "MujocoRL"
SEED_NETWORK = SOTA_ROOT / "network.py"
TRAIN_SCRIPT = SOTA_ROOT / "train_gene.py"
LLM_OPERATOR_SCRIPT = ROOT_DIR / "src" / "mesb_llm_operator.py"
#: Prompt templates (the existing MuJoCo prompts used by run_improved.py).
MUTATION_PROMPT_GLOB = "templates/Mujoco/Normal/*.txt"
CONSTANT_RULES_PATH = "templates/Mujoco/ConstantRules.txt"
#: Centralised output location (same name as OUTPUT_DIR in constants_Mujoco.py).
OUTPUT_ROOT = "mujoco_rl_output"

# --- Learner ("algo") profiles -------------------------------------------------
#: ``--algo`` picks the learner each gene is trained with. Everything that
#: differs between learners lives here; the archive, selection and logging
#: are shared. The "xqc" profile uses Shazeb's XQC island from
#: shazeb/halfcheetah-xqc unchanged: its seed gene, prompts, rules and its
#: gene validator (sota/XQC/train_xqc_gene.py).
ALGO = run_options.SEED
ALGO_PROFILES = {
    # The sota/MujocoRL seed on this branch is a TD3 gene (get_td3_kwargs);
    # train_gene.py trains it with TD3, as train_rl.py does. TD3 takes a gradient
    # step per env step, so each gene gets its own GPU job (td3_gene.sbatch).
    "td3": {
        "seed_network": SEED_NETWORK,
        "prompt_glob": MUTATION_PROMPT_GLOB,
        "constant_rules": CONSTANT_RULES_PATH,
        "required_functions": ("get_policy_kwargs", "get_td3_kwargs"),
        "train_script": TRAIN_SCRIPT,
        # Same per-gene training as run_improved.py (run_options.GENE_TIMESTEPS).
        "eval_timesteps": run_options.GENE_TIMESTEPS["td3"],
        "eval_num_seeds": 1,
        "slurm_script": ROOT_DIR / "scripts" / "mesb" / "td3_gene.sbatch",
    },
    # PPO genes, for a PPO seed network passed with --seed-network.
    "ppo": {
        "seed_network": SEED_NETWORK,
        "prompt_glob": MUTATION_PROMPT_GLOB,
        "constant_rules": CONSTANT_RULES_PATH,
        "required_functions": ("get_policy_kwargs", "get_ppo_kwargs"),
        "train_script": TRAIN_SCRIPT,
        "eval_timesteps": 500_000,
        "eval_num_seeds": 1,
        "slurm_script": None,
    },
    "xqc": {
        "seed_network": ROOT_DIR / "sota" / "XQC" / "llmge" / "network.py",
        "prompt_glob": "templates/Mujoco/XQC/*.txt",
        "constant_rules": "templates/Mujoco/XQC_ConstantRules.txt",
        "required_functions": ("get_architecture_kwargs", "get_xqc_kwargs"),
        "train_script": SOTA_ROOT / "train_gene_xqc.py",
        #: Shazeb's island budget: 100k env steps x 3 seeds trained together
        #: (run_options.GENE_TIMESTEPS / GENE_TRAIN_SEEDS, as run_improved.py).
        "eval_timesteps": run_options.GENE_TIMESTEPS["xqc"],
        "eval_num_seeds": run_options.GENE_TRAIN_SEEDS["xqc"],
        #: One GPU job per gene (sbatch --wait); see scripts/mesb/xqc_gene.sbatch.
        "slurm_script": ROOT_DIR / "scripts" / "mesb" / "xqc_gene.sbatch",
    },
}
#: Shazeb's gene validator (stdlib only): checks keys and ranges of an XQC gene.
XQC_GENE_VALIDATOR = ROOT_DIR / "sota" / "XQC" / "train_xqc_gene.py"
XQC_TRAIN_SCRIPT = ROOT_DIR / "sota" / "XQC" / "train_xqc.py"

# --- Environment / evaluation ------------------------------------------------
ENV_ID = run_options.ENV_ID
EVAL_TIMESTEPS = 500_000
EVAL_EPISODES = 10
EVAL_MAX_STEPS = 1000
#: Standardised evaluation seeds: episode i of every gene uses EVAL_SEED + i.
EVAL_SEED = 1000
EVAL_DEVICE = "cpu"
#: Wall-clock limit per gene (train + eval); exceeded -> status "failed".
EVAL_TIMEOUT_SEC = 6 * 60 * 60

# --- Budget (shared with run_improved.py through run_options.py) --------------
#: Environment steps of the whole run, charged per successfully trained gene
#: (eval_timesteps x eval_num_seeds). The run ends when it is spent.
TOTAL_TIMESTEPS = run_options.TOTAL_TIMESTEPS
#: A row in milestones.csv (and the archive snapshot of that generation) at
#: every multiple of this many steps.
MILESTONE_TIMESTEPS = run_options.MILESTONE_TIMESTEPS

# --- Evolution (mirrors constants_Mujoco.py) -----------------------------------
SEED = 0
#: Only used when TOTAL_TIMESTEPS is 0; with a budget, --num-generations is an
#: optional extra cap.
NUM_GENERATIONS = 30
START_POPULATION_SIZE = 32
POPULATION_SIZE = 32
CROSSOVER_PROBABILITY = 0.35
MUTATION_PROBABILITY = 0.8
NUM_ELITES = 8
HOF_SIZE = 100
TOP_P = 0.1
#: Temperature ranges used by run_improved.py for each operator.
CREATE_TEMPERATURE = (0.05, 0.4)
CROSSOVER_TEMPERATURE = (0.05, 0.1)
MUTATION_TEMPERATURE = (0.02, 0.35)
#: Wall-clock limit per LLM operator call. Generous on purpose: a call waits
#: (up to LLM_SERVER_READY_TIMEOUT, 4 h by default) for a replacement server
#: when the 8-hour LLM server job ends mid-run.
LLM_TIMEOUT_SEC = 5 * 60 * 60
#: LLM operator calls run concurrently (the server batches requests).
LLM_WORKERS = 8
#: Model name handed to the existing operators, like run_improved.py's
#: --llm_model (ISLAND_LLMS[0] in constants_Mujoco.py; served by server.py).
LLM_MODEL = "llama3"

#: Legacy NSGA-II objectives, read by *name* from the result JSON:
#: maximise mean_reward, minimise param_count (the documented intent of
#: FITNESS_WEIGHTS in constants_Mujoco.py; see docs/MESB.md for the
#: positional-parsing bug that made it minimise std_reward instead).
NSGA2_OBJECTIVES = (("mean_reward", 1.0), ("param_count", -1.0))

# --- MESB archive ------------------------------------------------------------------
#: Cells per descriptor (foot_contact_0, foot_contact_1). 32 x 32 matches
#: run_improved.py's foot-contact grid (MAP_BINS) so the two can be compared.
MESB_DIMS = (32, 32)
#: Paper default: recompute boundaries every 100 successful evaluations.
MESB_REMAP_FREQUENCY = 100
#: Paper default xi = infinity (keep every successful evaluation).
MESB_BUFFER_CAPACITY = None
#: Project adaptation (off by default): remap once per generation instead.
MESB_REMAP_AT_GENERATION_END = False
#: Optional offset for the shifted QD score; raw QD score is always logged.
MESB_QD_SCORE_OFFSET = None
#: Fixed-grid ranges of run_improved.py's MAP-Elites (MAP_ELITES_DESCRIPTORS). Only used
#: with --archive-boundaries fixed. NOTE: that commit's control cost was a
#: per-step mean; ours is a per-episode total (see docs/MESB.md), so choose
#: ranges for the fixed baseline explicitly.
#: Foot contact is a proportion, so the fixed grid spans [0, 1] on both axes.
LEGACY_FIXED_RANGES = ((0.0, 1.0), (0.0, 1.0))
