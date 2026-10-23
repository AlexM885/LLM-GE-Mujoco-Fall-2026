"""Run options: which robot, seed learner and search method a run uses.

The one place to choose them. Both drivers read these values:
run_improved.py (through constants_Mujoco.py) and run_mesb.py (through
constants_mesb.py). Every option can also be set by an environment variable,
which wins over the default here. sbatch passes the submitting environment on
to the LLM and evaluation jobs, so every process of a run agrees.

    ROBOT                  LLMGE_ROBOT                 "walker2d" | "halfcheetah"
    SEED                   LLMGE_SEED                  "td3" | "xqc"
    USE_LLM_EMITTERS       USE_LLM_EMITTERS            on | off
    SLIDING_BOUNDARIES     LLMGE_SLIDING_BOUNDARIES    on | off
    PARETO_FRONT_PER_CELL  LLMGE_PARETO_FRONT_PER_CELL on | off
    USE_VLM                LLMGE_USE_VLM               on | off
    TOTAL_TIMESTEPS        MUJOCO_TOTAL_TIMESTEPS      step budget of the whole run
    MILESTONE_TIMESTEPS    LLMGE_MILESTONE_TIMESTEPS   results snapshot every this many steps

On/off variables accept 1/0, true/false, yes/no, on/off.

Combinations that cannot work together (see check_run_options):

* SLIDING_BOUNDARIES runs a different driver, run_mesb.py, with its own
  archive whose cell edges move during the run. The LLM emitters and the
  Pareto-front-per-cell archive live in run_improved.py and assume a fixed
  grid, so neither works with it.
* USE_LLM_EMITTERS assumes one elite per cell; PARETO_FRONT_PER_CELL keeps up
  to MOME_FRONT_SIZE genes per cell. Pick one.

Deliberately free of heavy imports: run_mesb.py and the LLM jobs load it too.
"""

import os

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off", "")


def _flag(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name}={value!r}: expected on/off (1/0, true/false, yes/no)")


#: Robot the genes are trained and evaluated on.
ROBOT = os.getenv("LLMGE_ROBOT", "walker2d").strip().lower()

#: Seed gene and learner each gene is trained with:
#:   "td3"  Stable-Baselines3 TD3 (sota/MujocoRL/network.py, train_rl.py)
#:   "xqc"  Shazeb's XQC learner in JAX (sota/XQC/llmge/network.py, train_xqc_gene.py)
#: LLMGE_ALGO, the older name used by the XQC scripts, is still read; its
#: "ppo" meant the sota/MujocoRL seed, which is TD3 on this branch.
_legacy_algo = os.getenv("LLMGE_ALGO", "").strip().lower()
SEED = os.getenv("LLMGE_SEED", {"ppo": "td3"}.get(_legacy_algo, _legacy_algo) or "td3").strip().lower()

#: QD-LLMs (Koh & Wong et al., GECCO Companion '26): an LLM allocator splits
#: each generation between the uniform, sparse and directional LLM emitters.
#: Off: parents are drawn uniformly from the archive.
USE_LLM_EMITTERS = _flag("USE_LLM_EMITTERS", True)

#: MAP-Elites with Sliding Boundaries (Aayush): cell edges follow the
#: distribution of the genes found so far. Runs run_mesb.py instead of
#: run_improved.py; run.sh picks the driver from this option.
SLIDING_BOUNDARIES = _flag("LLMGE_SLIDING_BOUNDARIES", False)

#: Pareto-front-per-cell mode (Shazeb's Multi-Objective MAP-Elites): each cell
#: keeps a Pareto front over speed and effort instead of a single elite.
#: LLMGE_ARCHIVE=mome, the older switch, still turns it on.
PARETO_FRONT_PER_CELL = _flag("LLMGE_PARETO_FRONT_PER_CELL",
                              os.getenv("LLMGE_ARCHIVE", "").strip().lower() == "mome")

#: VLM-authored training reward (Neel's design), refreshed every generation.
#: After each generation's archive update the driver records rollouts of the
#: archive's best and worst genes, and a VLM (Gemma 3) watches them and writes
#: a Python aux_reward(obs, action, info) bonus, seeing its own previous one
#: too (Eureka-style reflection; src/vlm_loop.py, scripts/vlm/vlm_generation.sbatch).
#: The next generation trains with that bonus added to its reward; fitness is
#: still measured on the plain reward. Generation 0 trains without a bonus.
#: TD3 genes train with sota/MujocoRL/train_rl_vlm.py instead of train_rl.py;
#: XQC genes add the bonus in their training env. When off, no gene is shaped,
#: even if LLMGE_AUX_REWARD_CACHE is set in the environment.
USE_VLM = _flag("LLMGE_USE_VLM", False)
#: The VLM, and the Hugging Face cache that holds its weights. Neel's job runs
#: offline from his own cache by default (the shared vip-vvk cache lacks these
#: models); point LLMGE_VLM_HF_HOME at a copy you can read.
VLM_MODEL_ID = os.getenv("LLMGE_VLM_MODEL_ID", "google/gemma-3-12b-it")
VLM_HF_HOME = os.getenv("LLMGE_VLM_HF_HOME", "/storage/ice1/5/8/nsani3/hf_cache")
#: Genes the VLM watches each generation: the best and the worst of the
#: archive, for contrast (Neel found best-only gave no signal).
VLM_TOP_GENES = int(os.getenv("LLMGE_VLM_TOP_GENES", "2"))
VLM_BOTTOM_GENES = int(os.getenv("LLMGE_VLM_BOTTOM_GENES", "1"))
#: Weight of the bonus. The VLM always proposes 0.1, which Neel found has no
#: effect; his results used 10 (the wrapper caps it at 5, AuxRewardWrapper.MAX_LAM).
VLM_AUX_LAM = os.getenv("LLMGE_AUX_REWARD_LAM", "10")
#: Longest wait for one generation's VLM job (queue + captures + VLM).
VLM_TIMEOUT_SECONDS = int(os.getenv("LLMGE_VLM_TIMEOUT_SECONDS", str(3 * 60 * 60)))

#: Step budget: environment steps of training summed over every gene that
#: trained successfully (failed genes cost no budget). Each gene is charged
#: what it trains: TD3, MUJOCO_EVAL_TIMESTEPS; XQC, XQC_EVAL_TIMESTEPS steps for
#: each of its XQC_EVAL_SEEDS seeds, which train together. 0 = no budget.
TOTAL_TIMESTEPS = int(os.getenv("MUJOCO_TOTAL_TIMESTEPS", "1000000000"))
#: Every time the run passes a multiple of this many steps, the driver
#: appends a row to milestones.csv and saves the archive as it stands, so runs
#: can be compared at equal budgets (96M is the PGA-MAP-Elites budget).
MILESTONE_TIMESTEPS = int(os.getenv("LLMGE_MILESTONE_TIMESTEPS", "96000000"))
#: What one gene trains, by seed learner (see TOTAL_TIMESTEPS).
GENE_TIMESTEPS = {"td3": int(os.getenv("MUJOCO_EVAL_TIMESTEPS", "500000")),
                  "xqc": int(os.getenv("XQC_EVAL_TIMESTEPS", "100000"))}
GENE_TRAIN_SEEDS = {"td3": 1, "xqc": int(os.getenv("XQC_EVAL_SEEDS", "3"))}

#: Everything that differs between robots.
ROBOT_PROFILES = {
    "walker2d": {
        # v5, not v4: v4 grants healthy_reward on the terminating unhealthy
        # step (see sota/MujocoRL/eval.py).
        "env_id": "Walker2d-v5",
        "task_note": ("Walker2d-v5 is a bipedal walking task that terminates early when the torso "
                      "height or angle leaves its healthy range, so staying upright matters as much "
                      "as forward speed."),
        # foot_contact_<i> follows MuJoCo geom-id order.
        "foot_names": ("right foot", "left foot"),
        # Largest possible episode effort: ctrl_cost_weight 0.001 * 6 actuators * 1^2 * 1000 steps.
        "max_episode_effort": 6.0,
    },
    "halfcheetah": {
        # v4, as in all of the team's HalfCheetah runs so far.
        "env_id": "HalfCheetah-v4",
        "task_note": ("HalfCheetah-v4 is a planar running task with no early termination: the reward "
                      "is forward speed minus a torque cost, so speed is what matters."),
        "foot_names": ("back foot", "front foot"),
        # ctrl_cost_weight 0.1 * 6 actuators * 1^2 * 1000 steps.
        "max_episode_effort": 600.0,
    },
}
SEEDS = ("td3", "xqc")


def check_run_options():
    """Raise ValueError for an unknown value or a combination that cannot work."""
    if ROBOT not in ROBOT_PROFILES:
        raise ValueError(f"LLMGE_ROBOT={ROBOT!r}: expected one of {', '.join(ROBOT_PROFILES)}")
    if SEED not in SEEDS:
        raise ValueError(f"LLMGE_SEED={SEED!r}: expected one of {', '.join(SEEDS)}")
    if SLIDING_BOUNDARIES and USE_LLM_EMITTERS:
        raise ValueError("Sliding boundaries (run_mesb.py) and the LLM emitters (run_improved.py) "
                         "do not work together: set USE_LLM_EMITTERS=0 or LLMGE_SLIDING_BOUNDARIES=0")
    if SLIDING_BOUNDARIES and PARETO_FRONT_PER_CELL:
        raise ValueError("Sliding boundaries (run_mesb.py) and the Pareto-front-per-cell archive "
                         "(run_improved.py) do not work together: turn one of them off")
    if USE_VLM and VLM_TOP_GENES + VLM_BOTTOM_GENES < 1:
        raise ValueError("LLMGE_VLM_TOP_GENES + LLMGE_VLM_BOTTOM_GENES must be at least 1")
    if MILESTONE_TIMESTEPS <= 0:
        raise ValueError("LLMGE_MILESTONE_TIMESTEPS must be positive")
    if USE_LLM_EMITTERS and PARETO_FRONT_PER_CELL:
        raise ValueError("The LLM emitters need one elite per cell and do not work with the "
                         "Pareto-front-per-cell archive: set USE_LLM_EMITTERS=0 or "
                         "LLMGE_PARETO_FRONT_PER_CELL=0")


def describe_run_options():
    return (f"robot={ROBOT} seed={SEED} llm_emitters={'on' if USE_LLM_EMITTERS else 'off'} "
            f"sliding_boundaries={'on' if SLIDING_BOUNDARIES else 'off'} "
            f"pareto_front_per_cell={'on' if PARETO_FRONT_PER_CELL else 'off'} "
            f"vlm={'on (' + VLM_MODEL_ID + ', every generation)' if USE_VLM else 'off'} "
            f"budget={TOTAL_TIMESTEPS:,} steps (snapshot every {MILESTONE_TIMESTEPS:,})")


check_run_options()
ROBOT_PROFILE = ROBOT_PROFILES[ROBOT]
ENV_ID = ROBOT_PROFILE["env_id"]
#: Environment steps one gene of this run trains (budget charge per gene).
STEPS_PER_GENE = GENE_TIMESTEPS[SEED] * GENE_TRAIN_SEEDS[SEED]


def aux_reward_shell_prefix(aux_reward_file=None):
    """Shell code that sets the trainers' aux-reward variables for one job.

    The trainers read LLMGE_AUX_REWARD_CACHE (and LLMGE_AUX_REWARD_LAM). Set
    them to this generation's VLM file when the VLM is on and there is one;
    clear them otherwise, so a stray value never shapes a gene.
    """
    if USE_VLM and aux_reward_file:
        return (f"export LLMGE_AUX_REWARD_CACHE='{aux_reward_file}' "
                f"LLMGE_AUX_REWARD_LAM='{VLM_AUX_LAM}' && ")
    return "unset LLMGE_AUX_REWARD_CACHE LLMGE_AUX_REWARD_LAM && "


def apply_aux_reward_env(aux_reward_file=None, environ=os.environ):
    """Same as aux_reward_shell_prefix, for processes started from Python."""
    if USE_VLM and aux_reward_file:
        environ["LLMGE_AUX_REWARD_CACHE"] = aux_reward_file
        environ["LLMGE_AUX_REWARD_LAM"] = VLM_AUX_LAM
    else:
        environ.pop("LLMGE_AUX_REWARD_CACHE", None)
        environ.pop("LLMGE_AUX_REWARD_LAM", None)
