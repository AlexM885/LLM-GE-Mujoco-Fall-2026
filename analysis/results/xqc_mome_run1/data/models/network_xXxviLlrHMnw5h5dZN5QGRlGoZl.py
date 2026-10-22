
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:

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
        hidden_dims_critic=[512, 512, 256],
        hidden_dims_actor=[512, 256, 128],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=True,
    )
# --OPTION--
def get_combined_xqc_kwargs():
    return dict(
        actor_lr=2e-4,  # A moderate learning rate between the two options
        critic_lr=2e-4,
        temp_lr=2e-4,
        lr_end=5e-6,  # Lower end learning rate for finer tuning
        tau=0.0075,  # A balance between stability and adaptability
        policy_delay=2,
        init_temperature=0.075,  # Balanced initial temperature
        weight_decay=0.005,  # Moderate regularization
        batch_size=256,  # Increased batch size for better gradient estimation
        updates_per_step=1,  # Conservative update strategy
    )
