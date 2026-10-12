
# ========== Start: GeneCrossed

# ========== End:
# --PROMPT LOG--

ALGO = "xqc"

# --OPTION--
# -- NOTE --
# Note: XQC (ICLR 2026) is an off-policy actor-critic in JAX: a tanh-Gaussian actor
# and an ensemble of categorical critics, both MLPs with batch norm and weight norm.
# get_architecture_kwargs() must return a dict of network settings for the XQC agent.
# Only plain Python values (ints, floats, bools, strings, lists of ints) are allowed.
# Prefer mutating hidden_dims_critic, hidden_dims_actor and the
# normalization flags. n_critics must stay 2.# -- NOTE --
# ===============================
# === Combined XQC Hyperparameters ===
# ===============================

def get_combined_xqc_kwargs():
    return dict(
        # Moderate learning rates for stability and speed
        actor_lr=2e-4,
        critic_lr=2e-4,
        temp_lr=2e-4,
        lr_end=1e-5,
        
        # Reasonable tau for exploration without destabilization
        tau=0.0075,
        
        # Policy delay giving critic time to adapt
        policy_delay=2,
        
        # Initial temperature encouraging balanced exploration
        init_temperature=0.05,
        
        # Moderate weight decay for regularization
        weight_decay=0.005,
        
        # Batch size and updates per step optimized for resource usage and speed
        batch_size=192,
        updates_per_step=1,
    )

# Example usage
combined_kwargs = get_combined_xqc_kwargs()
print(combined_kwargs)
# --OPTION--

# ===============================
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    return dict(
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        lr_end=3e-5,
        tau=0.005,
        policy_delay=3,
        init_temperature=0.01,
        weight_decay=0.0,
        batch_size=256,
        updates_per_step=2,
    )
