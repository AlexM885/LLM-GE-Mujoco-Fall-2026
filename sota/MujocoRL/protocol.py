"""Fitness specification and evaluation protocol for the MuJoCo genes.

Dependency-free so that both the evaluation jobs (eval.py, train_rl.py,
reeval.py) and the evolution driver (run_improved.py, src/uqd) can import it.
eval.py re-exports these names.
"""

import os

DEFAULT_ENV_ID = os.getenv("MUJOCO_ENV_ID", "Walker2d-v5")

# Fitness specification for Walker2d. The episode fitness is the undiscounted
# return over a rollout of at most 1000 steps:
#
#     F = sum_t ( r_forward(t) + r_healthy(t) - c_ctrl(t) )
#
#     r_forward = forward_reward_weight * v_x        (v_x = dx/dt)
#     r_healthy = healthy_reward if healthy else 0
#     c_ctrl    = ctrl_cost_weight * ||a_t||^2       (sum over the 6 torques)
#
# "Healthy" means torso z in healthy_z_range and torso pitch in
# healthy_angle_range; leaving either range terminates the episode.
#
# These are the Gymnasium defaults, passed explicitly so the fitness function
# is pinned in this repo rather than inherited from whatever gymnasium version
# happens to be installed. Passing them is behaviourally a no-op today.
#
# NOTE: Walker2d-v4 does NOT satisfy this spec. It grants the healthy bonus on
# the terminating unhealthy step (measured: +1.000000 where healthy is False),
# so its returns are inflated by one healthy_reward. v5 fixes this.
WALKER_REWARD_SPEC = {
    "forward_reward_weight": 1.0,
    "ctrl_cost_weight": 0.001,
    "healthy_reward": 1.0,
    "healthy_z_range": (0.8, 2.0),
    "healthy_angle_range": (-1.0, 1.0),
    "terminate_when_unhealthy": True,
}
#: Max rollout length T. The Gymnasium time limit is also 1000.
WALKER_MAX_EPISODE_STEPS = 1000


def evaluation_protocol(env_id, max_steps, policy_device="cpu"):
    """Everything that defines what one evaluation episode measures.

    Statistics are only pooled across episodes that share this protocol (see
    src/uqd/stats.py). The episode count is not part of it: it is the number
    of samples, not a property of each sample. The policy device is part of
    it because GPU and CPU forward passes can differ in the last bits, which
    MuJoCo's contact dynamics can amplify; engine runs evaluate on CPU.
    """
    env_id = env_id or DEFAULT_ENV_ID
    return {
        "env_id": env_id,
        "max_steps": int(max_steps),
        "deterministic_policy": True,
        "reward_spec": WALKER_REWARD_SPEC if env_id.startswith("Walker2d") else None,
        "descriptor": "per-foot ground-contact fraction from the MuJoCo contact buffer",
        "policy_device": str(policy_device),
        "protocol_version": 2,
    }
