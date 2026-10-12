
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
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    """
    Returns a dictionary of hyperparameters for the XQC model.
    
    These hyperparameters can significantly affect the model's performance.
    It's recommended to perform hyperparameter tuning for optimal results.
    """
    return dict(
        actor_lr=3e-4,         # Learning rate for the actor
        critic_lr=3e-4,        # Learning rate for the critic
        temp_lr=3e-4,          # Learning rate for the temperature
        lr_end=3e-5,           # Final learning rate
        tau=0.005,             # Target network update rate
        policy_delay=3,        # Delay for policy updates
        init_temperature=0.01, # Initial temperature value
        weight_decay=0.0,      # Weight decay for regularization
        batch_size=256,        # Batch size for training
        updates_per_step=2,    # Number of updates per step
    )

# Example usage:
if __name__ == "__main__":
    xqc_hyperparams = get_xqc_kwargs()
    print("XQC Hyperparameters:", xqc_hyperparams)
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
