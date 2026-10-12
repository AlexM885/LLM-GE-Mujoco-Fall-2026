# Brax Ant Environment Context

You are writing JAX-compatible Python functions (`get_observation`, `compute_reward`)
that operate on Brax environment states for an Ant robot.

---

## 1. Task Description

The Ant is a 3D quadruped robot. The goal is to make the ant **run forward (+x direction)
as fast as possible** by applying torques on 8 hinge joints connecting its 4 legs to its torso.

**Fitness**: Total forward displacement over the episode (cur_x - prev_x summed across steps).
Higher = better. There is no upper bound.

**Episode termination**: Torso z-coordinate leaves [0.2, 1.0] (ant falls). Max 1000 steps.

---

## 2. State Object

| Field | Type | Description |
|---|---|---|
| `state.pipeline_state` | `brax.base.State` | Full physics simulation state |
| `state.pipeline_state.q` | `jax.Array (15,)` | Joint positions (see layout below) |
| `state.pipeline_state.qd` | `jax.Array (14,)` | Joint velocities (see layout below) |
| `state.pipeline_state.x.pos[0]` | `jax.Array (3,)` | Torso (x, y, z) position in world frame |
| `state.pipeline_state.x.rot[0]` | `jax.Array (4,)` | Torso orientation quaternion (w, x, y, z) |
| `state.pipeline_state.xd.vel[0]` | `jax.Array (3,)` | Torso linear velocity (world frame) |
| `state.pipeline_state.xd.ang[0]` | `jax.Array (3,)` | Torso angular velocity (world frame) |
| `state.obs` | `jax.Array (27,)` | Default observation (see below) |
| `state.reward` | `jax.Array ()` | Current reward scalar |
| `state.done` | `jax.Array ()` | Termination flag |

---

## 3. Joint Layout

**q (15 dimensions) — joint positions:**

| Indices | Dim | Content |
|---------|-----|---------|
| 0–1 | 2 | Torso x, y position (world frame) |
| 2–6 | 5 | Torso orientation: z-pos + quaternion (w,x,y,z) |
| 7–14 | 8 | Hinge joint angles: hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3, hip_4, ankle_4 |

**qd (14 dimensions) — joint velocities:**

| Indices | Dim | Content |
|---------|-----|---------|
| 0–5 | 6 | Torso velocity: (vx, vy, vz, wx, wy, wz) |
| 6–13 | 8 | Hinge joint velocities (same order as q) |

---

## 4. Default Observation (27 dims)

The default obs is `qpos[2:] + qvel`:
- `obs[0]`: z-coordinate of the torso (height)
- `obs[1:5]`: torso orientation quaternion (w, x, y, z)
- `obs[5:13]`: 8 hinge joint angles
- `obs[13:15]`: torso x, y velocity
- `obs[15]`: torso z velocity
- `obs[16:19]`: torso angular velocity (x, y, z)
- `obs[19:27]`: 8 hinge joint velocities

Note: x, y positions are **excluded** from default obs (not observable).

---

## 5. Action Space (8 actuators, continuous)

Actions are continuous torques in [-1, 1] applied to 8 hinge joints:

| Index | Joint | Description |
|---|---|---|
| 0 | hip_1 | Torso ↔ front left hip |
| 1 | ankle_1 | Front left hip ↔ front left ankle |
| 2 | hip_2 | Torso ↔ front right hip |
| 3 | ankle_2 | Front right hip ↔ front right ankle |
| 4 | hip_3 | Torso ↔ back left hip |
| 5 | ankle_3 | Back left hip ↔ back left ankle |
| 6 | hip_4 | Torso ↔ back right hip |
| 7 | ankle_4 | Back right hip ↔ back right ankle |

---

## 6. Physical Constants

- **dt**: 0.05s (5 sub-steps of 0.01s)
- **Torso initial height**: 0.75m
- **Healthy z range**: [0.2, 1.0] (episode ends if torso z leaves this range)
- **Episode length**: 1000 steps (50 seconds)

---

## 7. JAX Constraints

Your code must be **JAX-traceable** (`jax.jit` / `jax.vmap` compatible).

**Do:**
```python
import jax
import jax.numpy as jnp

torso_vel = state.pipeline_state.xd.vel[0]
forward_vel = torso_vel[0]
height = state.pipeline_state.x.pos[0, 2]
obs = jnp.concatenate([qpos_features, qvel_features])
```

**Do NOT:**
```python
if height > 0.5:                        # BAD — Python if on JAX array
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
    # state is the Brax State object (has state.pipeline_state, state.obs, etc.)
    # Must return shape (obs_dim,) where obs_dim <= 512
    ...

def compute_reward(state, action, next_state) -> jnp.ndarray:
    """Return a scalar reward."""
    # state: pre-step state, action: (8,) continuous, next_state: post-step state
    # Must return a scalar float
    ...
```

Note: `action` is a continuous array of shape `(8,)`, not a discrete integer.
