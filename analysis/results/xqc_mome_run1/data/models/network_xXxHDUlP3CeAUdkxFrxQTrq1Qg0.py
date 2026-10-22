
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:

# ========== Start: GeneCrossed

# ========== End:
# ========== End:

# ========== Start: GeneCrossed

# ========== End:

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
def get_architecture_kwargs():
    return dict(
        hidden_dims_critic=[768, 384, 192, 96], 
        hidden_dims_actor=[288, 144, 72, 36],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=False,
    )
# --OPTION--
# ===============================
# === Combined XQC Hyperparameter Gene ===
# ===============================

def get_combined_xqc_kwargs():
    return dict(
        actor_lr=2e-4,  # Moderate learning rate
        critic_lr=2e-4,  # Consistent with actor_lr for balance
        temp_lr=1e-4,   # Slightly lower to stabilize temperature learning
        lr_end=1e-6,    # Lower end learning rate for fine-tuning
        tau=0.0075,     # Balanced update rate for target networks
        policy_delay=2, # Reasonable delay to prevent overfitting
        init_temperature=0.05, # Initial temperature as in originals
        weight_decay=0.00075, # Moderate regularization
        batch_size=384,      # Compromise between memory usage and training speed
        updates_per_step=2,  # More frequent updates for faster convergence
    )
