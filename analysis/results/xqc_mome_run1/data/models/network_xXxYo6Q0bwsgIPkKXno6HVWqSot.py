
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
        hidden_dims_critic=[2048, 2048, 1024, 1024],  # Deeper critic
        hidden_dims_actor=[256, 256, 256, 256],  # Simpler yet effective actor
        n_critics=2,
        use_batch_norm=True,  # Batch normalization from the first snippet
        use_weight_norm=True,  # Weight normalization common in both
        use_layer_norm=False,  # Layer normalization not used in the first snippet
        pre_activation_bn=True,  # Pre-activation BN from the first snippet
        normalize_last_layer=True,  # Normalize last layer as in the first snippet
        skip_connections=False,  # No skip connections as in the first snippet
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
        batch_size=1024,
        updates_per_step=2,
    )
