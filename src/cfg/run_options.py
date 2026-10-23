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
    if USE_LLM_EMITTERS and PARETO_FRONT_PER_CELL:
        raise ValueError("The LLM emitters need one elite per cell and do not work with the "
                         "Pareto-front-per-cell archive: set USE_LLM_EMITTERS=0 or "
                         "LLMGE_PARETO_FRONT_PER_CELL=0")


def describe_run_options():
    return (f"robot={ROBOT} seed={SEED} llm_emitters={'on' if USE_LLM_EMITTERS else 'off'} "
            f"sliding_boundaries={'on' if SLIDING_BOUNDARIES else 'off'} "
            f"pareto_front_per_cell={'on' if PARETO_FRONT_PER_CELL else 'off'}")


check_run_options()
ROBOT_PROFILE = ROBOT_PROFILES[ROBOT]
ENV_ID = ROBOT_PROFILE["env_id"]
