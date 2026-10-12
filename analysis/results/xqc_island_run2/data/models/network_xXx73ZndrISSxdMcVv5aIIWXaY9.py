
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
    return dict(
        actor_lr=2e-4,  # A moderate learning rate
        critic_lr=2e-4,  # Consistent with actor_lr for balanced learning
        temp_lr=2e-4,   # Moderate temperature learning rate
        lr_end=2e-5,    # Ending learning rate for gradual decrease
        tau=0.0075,     # A compromise between exploration and exploitation
        policy_delay=2, # Less frequent policy updates for stability
        init_temperature=0.05, # Initial temperature for a good starting point
        weight_decay=0.005,   # Some regularization to prevent overfitting
        batch_size=192,       # A moderate batch size for efficient learning
        updates_per_step=1,   # Standard update frequency
    )
# --OPTION--
# ===============================
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    return dict(
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=5e-5,
        lr_end=1e-5,
        tau=0.005,
        policy_delay=2,
        init_temperature=0.1,
        weight_decay=0.001,
        batch_size=256,
        updates_per_step=2,
    )
