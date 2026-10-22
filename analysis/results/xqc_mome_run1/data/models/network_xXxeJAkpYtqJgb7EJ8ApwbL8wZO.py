
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
        hidden_dims_critic=[2048, 2048, 2048, 2048],
        hidden_dims_actor=[512, 512, 512],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=False,
        use_layer_norm=True,
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=True,
    )
# --OPTION--
def get_combined_xqc_kwargs():
    return dict(
        actor_lr=1e-3,  # Moderate learning rate for stability and speed
        critic_lr=1e-3,  # Consistent with actor_lr for balanced training
        temp_lr=1e-4,   # Temperature learning rate remains consistent
        lr_end=1e-5,    # End learning rate for decay
        tau=0.01,       # Soft update parameter
        policy_delay=2, # Delay for policy updates
        init_temperature=0.1, # Initial temperature
        weight_decay=0.01,    # Weight decay for regularization
        batch_size=256,      # Moderate batch size for balance
        updates_per_step=2,  # Increased updates for faster adaptation
    )
