
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
def get_combined_xqc_kwargs():
    return dict(
        # Use moderate learning rates for stability and reasonable convergence speed
        actor_lr=2e-4,
        critic_lr=2e-4,
        temp_lr=2e-4,
        lr_end=1e-5,  # Gradually decrease learning rate towards this end value
        
        # Choose a moderate tau for updating target networks
        tau=0.0075,
        
        # Apply a moderate policy delay to stabilize actor-critic updates
        policy_delay=2,
        
        # Start with a moderate initial temperature for balanced exploration-exploitation
        init_temperature=0.05,
        
        # Apply some regularization to prevent overfitting
        weight_decay=0.005,
        
        # Use a larger batch size for more stable gradients, assuming sufficient memory
        batch_size=384,
        
        # Update the critic (and possibly other components) twice per environment step
        updates_per_step=2,
    )
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
