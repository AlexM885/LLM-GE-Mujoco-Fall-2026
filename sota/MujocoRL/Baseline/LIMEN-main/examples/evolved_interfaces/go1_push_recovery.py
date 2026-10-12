"""
Best MDP Interface — evolved by LIMEN

Task:         Stand at the origin and recover from random force impulses applied to the torso. Success = survived full episode AND average position error < 10cm from origin.

Success rate: 55%
Obs dim:      98
Generation:   3
Iteration:    30
Mode:         full
"""

import jax
import jax.numpy as jnp

DEFAULT_POSE = jnp.array([0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8])


def get_observation(state) -> jnp.ndarray:
    # --- Orientation ---
    gravity = state.info["gravity"]       # (3,) gravity in body frame
    upvector = state.info["upvector"]     # (3,) up vector in world frame

    # --- Angular velocity ---
    gyro = state.info["gyro"] / 5.0       # (3,) normalized

    # --- Linear velocity ---
    local_linvel = state.info["local_linvel"] / 3.0  # (3,) body frame
    world_linvel = state.data.qvel[:3] / 3.0          # (3,) world frame

    # --- Position ---
    pos_xy = state.info["pos_xy"]         # (2,) XY offset from origin
    pos_dist = jnp.linalg.norm(pos_xy)

    # Direction to origin in world frame
    dir_to_origin_world = -pos_xy / (pos_dist + 1e-6)  # (2,)

    # Heading
    heading = state.info["heading"]
    cos_h = jnp.cos(-heading)
    sin_h = jnp.sin(-heading)

    # Direction to origin in body frame
    dir_to_origin_body = jnp.array([
        cos_h * dir_to_origin_world[0] - sin_h * dir_to_origin_world[1],
        sin_h * dir_to_origin_world[0] + cos_h * dir_to_origin_world[1],
    ])  # (2,)

    # Position at multiple scales
    pos_xy_coarse = pos_xy / 1.0
    pos_xy_fine = jnp.clip(pos_xy / 0.1, -5.0, 5.0)
    pos_dist_feat = jnp.array([pos_dist])

    # Body-frame position
    pos_body_x = cos_h * pos_xy[0] + sin_h * pos_xy[1]
    pos_body_y = -sin_h * pos_xy[0] + cos_h * pos_xy[1]
    pos_body = jnp.array([pos_body_x, pos_body_y])
    pos_body_fine = jnp.clip(jnp.array([pos_body_x / 0.1, pos_body_y / 0.1]), -5.0, 5.0)

    # --- Height ---
    height = state.info["height"]
    height_feat = jnp.array([height / 0.4])

    # --- Heading ---
    heading_feat = jnp.array([jnp.sin(heading), jnp.cos(heading)])

    # --- Joint state ---
    joint_angles = state.data.qpos[7:]
    joint_dev = (joint_angles - DEFAULT_POSE) / jnp.pi
    joint_vels = state.data.qvel[6:] / 15.0

    # --- Previous action ---
    last_act = state.info["last_act"]

    # --- Push force ---
    push_force = state.info["push_force"] / 400.0
    push_mag = jnp.linalg.norm(state.info["push_force"])
    push_active = jnp.array([jnp.tanh(push_mag / 100.0)])

    # --- Velocity toward origin ---
    vel_xy = state.data.qvel[:2]
    vel_toward_world = jnp.dot(vel_xy, dir_to_origin_world)
    vel_toward_feat = jnp.array([jnp.tanh(vel_toward_world)])

    local_vel_xy = state.info["local_linvel"][:2]
    vel_toward_body = jnp.dot(local_vel_xy, dir_to_origin_body)
    vel_toward_body_feat = jnp.array([jnp.tanh(vel_toward_body)])

    # --- Velocity magnitude ---
    vel_mag_feat = jnp.array([jnp.linalg.norm(vel_xy) / 3.0])

    # --- Uprightness features ---
    upvector_z = upvector[-1]
    tilt_feat = jnp.array([upvector_z])
    danger_feat = jnp.array([jnp.maximum(0.0, 0.7 - upvector_z)])

    # --- Gyro magnitude ---
    gyro_mag = jnp.array([jnp.linalg.norm(state.info["gyro"]) / 5.0])

    # --- Quaternion ---
    quat = state.data.qpos[3:7]

    # --- Actuator forces ---
    actuator_force = state.data.actuator_force / 50.0

    # --- Projected future position (simple linear extrapolation) ---
    # Where will the robot be in ~10 steps if it keeps current velocity?
    dt_horizon = 0.2  # seconds
    future_pos_xy = pos_xy + vel_xy * dt_horizon
    future_dist = jnp.linalg.norm(future_pos_xy)
    future_pos_feat = jnp.clip(future_pos_xy / 1.0, -5.0, 5.0)
    future_dist_feat = jnp.array([future_dist])

    # --- Angular momentum direction (for predicting fall direction) ---
    # gyro in body frame - cross with upvector gives fall tendency
    gyro_raw = state.info["gyro"]
    # Project gyro onto horizontal plane to get tipping tendency
    gyro_horiz = jnp.array([gyro_raw[0] / 5.0, gyro_raw[1] / 5.0])

    obs = jnp.concatenate([
        gravity,              # 3
        upvector,             # 3
        gyro,                 # 3
        local_linvel,         # 3
        world_linvel,         # 3
        pos_xy_coarse,        # 2
        pos_xy_fine,          # 2
        pos_body,             # 2
        pos_body_fine,        # 2
        pos_dist_feat,        # 1
        dir_to_origin_world,  # 2
        dir_to_origin_body,   # 2
        height_feat,          # 1
        heading_feat,         # 2
        joint_dev,            # 12
        joint_vels,           # 12
        last_act,             # 12
        push_force,           # 3
        push_active,          # 1
        vel_toward_feat,      # 1
        vel_toward_body_feat, # 1
        vel_mag_feat,         # 1
        tilt_feat,            # 1
        danger_feat,          # 1
        gyro_mag,             # 1
        quat,                 # 4
        actuator_force,       # 12
        future_pos_feat,      # 2
        future_dist_feat,     # 1
        gyro_horiz,           # 2
    ]).astype(jnp.float32)
    # Total: 3+3+3+3+3+2+2+2+2+1+2+2+1+2+12+12+12+3+1+1+1+1+1+1+1+4+12+2+1+2 = 91

    return obs


def compute_reward(state, action, next_state) -> jnp.ndarray:
    # ---- Core state ----
    upvector_z = next_state.info["upvector"][-1]
    pos_xy = next_state.info["pos_xy"]
    pos_dist = jnp.linalg.norm(pos_xy)
    prev_pos_dist = jnp.linalg.norm(state.info["pos_xy"])
    heading = next_state.info["heading"]
    height = next_state.info["height"]

    # ---- 1. Uprightness reward ----
    # Primary survival signal - smooth and strong
    uprightness = upvector_z  # [0,1]
    
    # Extra reward for being very upright (quadratic bonus near top)
    uprightness_bonus = jnp.maximum(0.0, upvector_z - 0.8) ** 2 * 10.0

    # Penalty ramp for dangerous tilt
    tilt_danger = jnp.maximum(0.0, 0.65 - upvector_z)
    tilt_penalty = tilt_danger ** 2 * 20.0

    # ---- 2. Position reward (STRUCTURAL CHANGE: no uprightness gating) ----
    # Always reward being near origin - even when recovering from a push
    # This gives gradient even when tilted, helping the robot understand
    # it needs to return to origin regardless of orientation
    
    # Multi-scale exponential rewards
    pos_r_wide = jnp.exp(-1.0 * pos_dist)    # wide gradient (1/e at 1m)
    pos_r_medium = jnp.exp(-5.0 * pos_dist)  # medium gradient
    pos_r_tight = jnp.exp(-20.0 * pos_dist)  # tight near origin
    position_reward = 0.3 * pos_r_wide + 0.4 * pos_r_medium + 0.3 * pos_r_tight

    # ---- 3. Progress reward ----
    # Dense signal: reward any movement toward origin
    dist_improvement = prev_pos_dist - pos_dist
    progress_reward = jnp.clip(dist_improvement * 40.0, -1.0, 1.0)

    # ---- 4. Velocity management ----
    vel_xy = next_state.data.qvel[:2]
    vel_mag = jnp.linalg.norm(vel_xy)
    dir_to_origin = -pos_xy / (pos_dist + 1e-6)
    vel_toward_origin = jnp.dot(vel_xy, dir_to_origin)

    # Reward velocity toward origin when far, penalize velocity when near
    far_weight = jnp.tanh(pos_dist * 8.0)
    near_weight = 1.0 - far_weight

    vel_toward_reward = far_weight * jnp.tanh(vel_toward_origin * 3.0) * 0.6
    vel_still_penalty = near_weight * jnp.tanh(vel_mag * 4.0) * 0.4

    # ---- 5. Heading reward ----
    heading_reward = jnp.exp(-3.0 * jnp.abs(heading))

    # ---- 6. Height reward ----
    # Reward being tall - proxy for standing
    height_reward = jnp.tanh(height * 7.0)

    # ---- 7. Angular velocity penalty ----
    gyro_mag = jnp.linalg.norm(next_state.info["gyro"])
    gyro_penalty = jnp.tanh(gyro_mag * 0.2) * 0.15

    # ---- 8. Action smoothness ----
    last_act = state.info["last_act"]
    action_diff = jnp.sum(jnp.square(action - last_act))
    smoothness_penalty = jnp.tanh(action_diff * 0.04) * 0.05

    # ---- 9. Torque penalty ----
    actuator_force = next_state.data.actuator_force
    torque_penalty = jnp.tanh(jnp.sum(jnp.square(actuator_force)) * 0.00008) * 0.04

    # ---- 10. Fall penalty ----
    fallen = (upvector_z < 0.3).astype(jnp.float32)
    fall_penalty = fallen * 20.0

    # ---- 11. Success criterion bonus ----
    # Directly reward being within 10cm AND upright
    within_10cm_upright = jnp.exp(-20.0 * pos_dist) * jnp.clip(upvector_z, 0.0, 1.0)
    success_bonus = within_10cm_upright * 2.0

    # ---- 12. Survival bonus ----
    survival = 0.3

    # ---- Combine ----
    # KEY STRUCTURAL CHANGE: position reward is NOT gated by uprightness
    # This means the agent always gets gradient to return to origin
    # Uprightness is still primary through its own large weight
    reward = (
        4.0 * uprightness                    # Stay upright (primary)
        + uprightness_bonus                  # Extra for very upright
        - tilt_penalty                       # Danger zone penalty
        + 4.0 * position_reward              # Return to origin (ungated!)
        + 2.0 * progress_reward              # Progress toward origin
        + success_bonus                      # Bonus for success criterion
        + 0.5 * heading_reward               # Maintain heading
        + 0.3 * height_reward                # Maintain height
        + vel_toward_reward                  # Move toward origin when far
        - vel_still_penalty                  # Stay still when near
        - gyro_penalty                       # Don't spin
        - smoothness_penalty                 # Smooth actions
        - torque_penalty                     # Energy efficiency
        - fall_penalty                       # Hard fall penalty
        + survival                           # Survive
    )

    return reward.astype(jnp.float32)
