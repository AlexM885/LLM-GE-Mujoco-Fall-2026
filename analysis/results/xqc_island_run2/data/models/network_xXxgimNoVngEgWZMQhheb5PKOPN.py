
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
# === Combined XQC Hyperparameter Gene ===
# ===============================

def get_combined_xqc_kwargs():
    return dict(
        actor_lr=5e-4,  # Moderate learning rate
        critic_lr=5e-4,
        temp_lr=5e-4,
        lr_end=5e-5,  # Slightly higher end learning rate for smoother decay
        tau=0.01,  # Retained as it seems like a standard value for soft update
        policy_delay=2,  # More conservative update schedule for stability
        init_temperature=0.1,  # Initial temperature retained as per original
        weight_decay=0.001,  # Small regularization to prevent overfitting
        batch_size=256,  # Balanced batch size for efficiency and stability
        updates_per_step=2,  # More frequent updates than the first snippet for faster learning
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
        tau=0.01,
        policy_delay=1,
        init_temperature=0.1,
        weight_decay=0.001,
        batch_size=256,
        updates_per_step=2,
    )
