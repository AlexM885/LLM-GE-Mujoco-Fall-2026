"""Default configuration for the MESB MuJoCo driver (``run_mesb.py``).

This is the single source of defaults; every value can be overridden on the
``run_mesb.py`` command line, and the resolved values are written to
``<run_dir>/config.json``. Evolution defaults mirror
``src/cfg/constants_Mujoco.py`` (this branch's base, 6b73ae167) so that MESB
and the legacy NSGA-II loop differ only in what the experiment varies.

Deliberately free of torch / gymnasium imports so the driver stays light.
"""

from pathlib import Path

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

# --- Environment / evaluation ------------------------------------------------
ENV_ID = "HalfCheetah-v4"
EVAL_TIMESTEPS = 500_000
EVAL_EPISODES = 10
EVAL_MAX_STEPS = 1000
#: Standardised evaluation seeds: episode i of every gene uses EVAL_SEED + i.
EVAL_SEED = 1000
EVAL_DEVICE = "cpu"
#: Wall-clock limit per gene (train + eval); exceeded -> status "failed".
EVAL_TIMEOUT_SEC = 6 * 60 * 60

# --- Evolution (mirrors constants_Mujoco.py) -----------------------------------
SEED = 0
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
#: Cells per descriptor (mean_distance, mean_control_cost). 20 matches the
#: legacy fixed-grid MAP_BINS so the two approaches can be compared directly.
MESB_DIMS = (20, 20)
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
LEGACY_FIXED_RANGES = ((-500.0, 2000.0), (0.0, 10.0))
