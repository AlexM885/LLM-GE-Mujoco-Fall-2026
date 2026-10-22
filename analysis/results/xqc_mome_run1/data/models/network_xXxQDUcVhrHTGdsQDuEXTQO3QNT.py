
# ========== Start: GeneCrossed

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
    """
    Returns a dictionary of network settings for the XQC agent.
    
    The settings are chosen to potentially enhance the model's predictive metrics
    by combining a deeper critic network with a moderately sized actor network.
    """
    return dict(
        hidden_dims_critic=[1024, 1024, 1024, 1024],  # Deeper critic for complex state understanding
        hidden_dims_actor=[512, 512, 512, 512],  # Moderately sized actor for effective action selection
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=False,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=5e-3,
        critic_lr=5e-3,
        temp_lr=5e-5,
        lr_end=1e-5,
        tau=0.01,
        policy_delay=2,
        init_temperature=0.1,
        weight_decay=0.05,
        batch_size=1024,
        updates_per_step=4,
    )
