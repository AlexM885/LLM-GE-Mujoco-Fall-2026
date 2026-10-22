
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
        # Increased hidden dimensions for both critic and actor
        hidden_dims_critic=[2048, 2048, 2048, 2048],
        hidden_dims_actor=[1024, 1024, 1024, 1024],
        
        # Retaining the requirement of having 2 critics
        n_critics=2,
        
        # Combining normalization techniques
        use_batch_norm=True,  # From the first snippet
        use_weight_norm=True,  # From the second snippet
        use_layer_norm=False,  # Choosing one over the other based on performance
        
        # Other architectural choices
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=True,  # Enabling skip connections for potential performance boost
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=1e-3,
        critic_lr=1e-3,
        temp_lr=1e-6,
        lr_end=1e-5,
        tau=0.01,
        policy_delay=4,
        init_temperature=0.05,
        weight_decay=0.01,
        batch_size=512,
        updates_per_step=3,
    )
