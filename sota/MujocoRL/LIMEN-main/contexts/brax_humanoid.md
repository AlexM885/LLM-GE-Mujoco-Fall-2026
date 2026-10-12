# Brax Humanoid Environment Context

You are writing JAX-compatible Python functions (`get_observation`, `compute_reward`)
that operate on Brax environment states for a Humanoid robot.

---

## 1. Task Description

The Humanoid is a 3D bipedal robot. The goal is to make the humanoid **run forward
(+x direction) as fast as possible** by applying torques on 17 hinge joints.

**Fitness**: Total forward displacement over the episode (cur_x - prev_x summed across steps).
Higher = better. There is no upper bound.

**Episode termination**: Torso z-coordinate leaves [0.8, 2.1] (humanoid falls). Max 1000 steps.

---

## 2. State Object

| Field | Type | Description |
|---|---|---|
| `state.pipeline_state` | `brax.base.State` | Full physics simulation state |
| `state.pipeline_state.q` | `jax.Array (24,)` | Generalized positions |
| `state.pipeline_state.qd` | `jax.Array (23,)` | Generalized velocities |
| `state.pipeline_state.x.pos[0]` | `jax.Array (3,)` | Torso (x, y, z) position in world frame |
| `state.pipeline_state.x.rot[0]` | `jax.Array (4,)` | Torso orientation quaternion (w, x, y, z) |
| `state.pipeline_state.xd.vel[0]` | `jax.Array (3,)` | Torso linear velocity (world frame) |
| `state.pipeline_state.xd.ang[0]` | `jax.Array (3,)` | Torso angular velocity (world frame) |
| `state.pipeline_state.x.pos` | `jax.Array (N,3)` | Positions of all bodies (torso + limbs) |
| `state.pipeline_state.xd.vel` | `jax.Array (N,3)` | Linear velocities of all bodies |
| `state.obs` | `jax.Array (244,)` | Default observation |
| `state.reward` | `jax.Array ()` | Current reward scalar |
| `state.done` | `jax.Array ()` | Termination flag |

---

## 3. Joint Layout

**q (24 dimensions) — generalized positions:**

| Indices | Dim | Content |
|---------|-----|---------|
| 0–2 | 3 | Torso (x, y, z) position (world frame) |
| 3–6 | 4 | Torso orientation quaternion (w, x, y, z) |
| 7–23 | 17 | Hinge joint angles (abdomen, hips, knees, shoulders, elbows) |

**qd (23 dimensions) — generalized velocities:**

| Indices | Dim | Content |
|---------|-----|---------|
| 0–5 | 6 | Torso velocity: (vx, vy, vz, wx, wy, wz) |
| 6–22 | 17 | Hinge joint velocities (same order as q[7:]) |

The 17 hinge joints (in order):
abdomen_z, abdomen_y, abdomen_x, right_hip_x, right_hip_z, right_hip_y, right_knee,
left_hip_x, left_hip_z, left_hip_y, left_knee, right_shoulder1, right_shoulder2,
right_elbow, left_shoulder1, left_shoulder2, left_elbow.

---

## 4. Default Observation (244 dims)

The default obs is a large concatenation including: qpos[2:], qvel, center-of-mass
inertia for each body, com velocities, actuator forces, and external contact forces.
You usually do NOT need to mirror this — designing a smaller, well-targeted obs is fine.

---

## 5. Action Space (17 actuators, continuous)

Actions are continuous torques in [-1, 1] applied to the 17 hinge joints
(same order as q[7:23]).

---

## 6. Physical Constants

- **dt**: 0.015s (3 sub-steps of 0.005s)
- **Torso initial height**: ~1.4m
- **Healthy z range**: [0.8, 2.1] (episode ends if torso z leaves this range)
- **Episode length**: 1000 steps (~15 seconds)

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
    # state is the Brax State object
    # Must return shape (obs_dim,) where obs_dim <= 512
    ...

def compute_reward(state, action, next_state) -> jnp.ndarray:
    """Return a scalar reward."""
    # state: pre-step state, action: (17,) continuous, next_state: post-step state
    # Must return a scalar float
    ...
```

Note: `action` is a continuous array of shape `(17,)`.
