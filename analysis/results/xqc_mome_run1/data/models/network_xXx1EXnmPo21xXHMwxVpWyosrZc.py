
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
        # Critic Network Settings
        hidden_dims_critic=[1536, 768, 384, 192],  # Balanced critic capacity
        # Actor Network Settings
        hidden_dims_actor=[768, 384, 192, 96],  # Reduced but balanced actor capacity
        # Number of Critics
        n_critics=2,
        # Normalization Flags
        use_batch_norm=True,  # Enable batch normalization for stability
        use_weight_norm=True,  # Weight normalization for better gradient flow
        use_layer_norm=False,  # Disable layer normalization based on snippet preferences
        pre_activation_bn=True,  # Apply batch normalization before activation
        normalize_last_layer=True,  # Normalize the last layer for improved outputs
        skip_connections=False,  # No skip connections as per the second snippet's preference
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=1e-4,
        lr_end=1e-5,
        tau=0.005,
        policy_delay=2,
        init_temperature=0.1,
        weight_decay=0.001,
        batch_size=256,
        updates_per_step=1,
    )
