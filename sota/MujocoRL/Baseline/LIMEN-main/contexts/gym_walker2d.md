# Gymnasium Walker2d-v5 Environment Context

You are writing JAX-compatible Python functions (`get_observation`, `compute_reward`)
that operate on the simulator state of the Gymnasium MuJoCo `Walker2d-v5` robot.

---

## 1. Task Description

The Walker2d is a planar two-legged robot (torso, two thighs, two legs, two feet).
The goal is to make it **walk forward (+x direction) as fast as possible without falling**
by applying torques on its 6 hinge joints.

**Fitness** (what candidates are ranked by): the environment's own undiscounted episode
return, averaged over evaluation episodes of the trained policy:

```
F = sum over steps of ( forward_reward + healthy_reward - ctrl_cost )
forward_reward = 1.0   * (x_after - x_before) / dt
healthy_reward = 1.0   for every step the robot is healthy
ctrl_cost      = 0.001 * sum(action ** 2)
```

Higher = better. There is no upper bound; about 3000+ is sustained walking.
Your `compute_reward` is only the signal the agent is **trained** on. The fitness is always
computed with the formula above, never with your reward.

**Episode termination**: the robot is unhealthy when the torso height `qpos[1]` leaves
(0.8, 2.0) or the torso angle `qpos[2]` leaves (-1.0, 1.0). Max 1000 steps.

**Learner**: TD3 (off-policy, deterministic actor), trained from scratch for a fixed number
of steps. Observations are NOT normalized by the learner, so keep features in similar ranges.

---

## 2. State Object

| Field | Type | Description |
|---|---|---|
| `state.qpos` | `jax.Array (9,)` float32 | Generalized positions (see layout below) |
| `state.qvel` | `jax.Array (9,)` float32 | Generalized velocities (see layout below) |
| `state.obs` | `jax.Array (17,)` float32 | Default observation (see below) |
| `state.step_num` | `jax.Array ()` int32 | Steps taken in the current episode (0 after reset) |

There are no other fields.

---

## 3. Joint Layout

**qpos (9 dimensions):**

| Index | Content |
|---|---|
| 0 | Torso x position (world frame, m) |
| 1 | Torso z position, i.e. height (m). Initial value about 1.25 |
| 2 | Torso pitch angle (rad) |
| 3 | thigh_joint angle (right thigh) |
| 4 | leg_joint angle (right leg) |
| 5 | foot_joint angle (right foot) |
| 6 | thigh_left_joint angle |
| 7 | leg_left_joint angle |
| 8 | foot_left_joint angle |

**qvel (9 dimensions):** the time derivatives of the same 9 coordinates, in the same order
(`qvel[0]` is the forward velocity of the torso, `qvel[1]` the vertical velocity,
`qvel[2]` the pitch angular velocity, `qvel[3:9]` the joint angular velocities).

---

## 4. Default Observation (17 dims)

`state.obs` is `qpos[1:]` followed by `clip(qvel, -10, 10)`:
- `obs[0]`: torso height
- `obs[1]`: torso pitch angle
- `obs[2:8]`: the 6 joint angles
- `obs[8:17]`: the 9 velocities, clipped to [-10, 10]

Note: the x position is **excluded** from the default observation.

---

## 5. Action Space (6 actuators, continuous)

Actions are continuous torques in [-1, 1]:

| Index | Joint |
|---|---|
| 0 | thigh_joint (right) |
| 1 | leg_joint (right) |
| 2 | foot_joint (right) |
| 3 | thigh_left_joint |
| 4 | leg_left_joint |
| 5 | foot_left_joint |

---

## 6. Physical Constants

- **dt**: 0.008s (4 sub-steps of 0.002s)
- **Torso initial height**: 1.25m
- **Healthy height range**: (0.8, 2.0); **healthy angle range**: (-1.0, 1.0)
- **Episode length**: 1000 steps (8 seconds)

---

## 7. JAX Constraints

Your code must be **JAX-traceable** (`jax.jit` compatible).

**Do:**
```python
import jax
import jax.numpy as jnp

forward_vel = state.qvel[0]
height = state.qpos[1]
obs = jnp.concatenate([state.qpos[1:], jnp.clip(state.qvel, -10.0, 10.0)])
```

**Do NOT:**
```python
if height > 0.8:                        # BAD — Python if on JAX array
    reward = 1.0
import numpy as np                      # BAD — must use jax.numpy
for i in range(len(qpos)):              # BAD — use vectorized jnp ops
```

Only `import jax` and `import jax.numpy as jnp` are allowed.

---

## 8. Function Signatures

```python
def get_observation(state) -> jnp.ndarray:
    """Return a 1D float array of observations."""
    # Must return shape (obs_dim,) where obs_dim <= 512, with finite values
    ...

def compute_reward(state, action, next_state) -> jnp.ndarray:
    """Return a scalar reward."""
    # state: pre-step state, action: (6,) continuous, next_state: post-step state
    # Must return a finite scalar float
    ...
```

Note: `action` is a continuous array of shape `(6,)`, not a discrete integer.
