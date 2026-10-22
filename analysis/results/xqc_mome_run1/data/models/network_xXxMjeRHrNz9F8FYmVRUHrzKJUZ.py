
# ========== Start: GeneCrossed

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
    """
    Returns a dictionary of network settings for the XQC agent, combining enhancements
    from both provided code snippets for potentially improved predictive metrics.
    """
    return dict(
        # Using larger dimensions as in Code Snippet 1 for increased capacity
        hidden_dims_critic=[2048, 1536, 1024, 768],
        # Adjusting actor dimensions for balance, considering the impact of larger critic dimensions
        hidden_dims_actor=[512, 512, 384, 384, 256],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=True,
        # Enabling skip connections as in Code Snippet 1 for deeper network training
        skip_connections=True,
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
        init_temperature=0.05,
        weight_decay=0.001,
        batch_size=256,
        updates_per_step=2,
    )
