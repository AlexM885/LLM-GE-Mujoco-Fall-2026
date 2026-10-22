
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
# ========== End:

# ========== Start: GeneCrossed

# ========== End:
# ========== End:

# ========== Start: GeneCrossed

# ========== End:

# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:

# ========== Start: GeneCrossed

# ========== End:
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
        hidden_dims_critic=[1536, 1280, 1024, 768],
        hidden_dims_actor=[768, 512, 256, 128],
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
# === Enhanced XQC Hyperparameters ===
# ===============================

def get_enhanced_xqc_kwargs():
    return dict(
        actor_lr=2e-4,  # Moderate learning rate
        critic_lr=2e-4,  # Moderate learning rate
        temp_lr=3e-5,   # Slightly reduced temperature learning rate
        lr_end=1e-5,    # Same end learning rate as both snippets
        tau=0.0075,     # Tau value between the two snippets for balance
        policy_delay=2, # Consistent with both snippets
        init_temperature=0.1, # Initial temperature remains the same
        weight_decay=0.005,  # Moderate weight decay
        batch_size=192,      # Batch size between the two values for a balance
        updates_per_step=1,  # Single update per step for simplicity and resource efficiency
    )
