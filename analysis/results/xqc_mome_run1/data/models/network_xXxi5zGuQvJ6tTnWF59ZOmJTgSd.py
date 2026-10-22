
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:

# ========== Start: GeneCrossed

# ========== End:
# ========== End:

# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
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
        hidden_dims_critic=[2560, 2560, 1280, 1280],  # Slightly increased critic capacity
        hidden_dims_actor=[1600, 1600, 800, 800],     # Balanced actor capacity
        n_critics=2,
        use_batch_norm=True,                          # Enable batch normalization for stability
        use_weight_norm=True,
        use_layer_norm=False,                         # Disable layer normalization to avoid over-normalization
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=True,                        # Enable skip connections to help with deep network training
    )
# --OPTION--
def get_combined_xqc_kwargs():
    return dict(
        actor_lr=4e-4,  # Moderate learning rate
        critic_lr=4e-4,
        temp_lr=4e-4,
        lr_end=5e-6,  # Allows for a decent decrease
        tau=0.005,  # Retained from original configurations
        policy_delay=2,  # Also retained
        init_temperature=0.075,  # Balances exploration and exploitation
        weight_decay=0.005,  # Moderate regularization
        batch_size=384,  # Compromise between speed and generalization
        updates_per_step=2,  # Retained for consistency
    )
