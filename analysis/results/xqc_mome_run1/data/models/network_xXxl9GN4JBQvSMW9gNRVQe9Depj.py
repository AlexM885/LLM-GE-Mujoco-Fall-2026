
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
    Returns a dictionary of network settings for the XQC agent, combining elements from two alternatives.
    
    The approach aims to balance depth and width of both critic and actor networks, 
    considering the need for both expressive power and efficient computation.
    """
    return dict(
        # Combining depth with increased capacity for the critic
        hidden_dims_critic=[1280, 1152, 1024, 512],
        
        # Adopting a more complex actor configuration for better policy representation
        hidden_dims_actor=[512, 512, 512, 256],
        
        # Keeping other parameters consistent with the originals
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
        actor_lr=1e-3,
        critic_lr=1e-3,
        temp_lr=5e-5,
        lr_end=5e-6,
        tau=0.01,
        policy_delay=3,
        init_temperature=0.05,
        weight_decay=0.0005,
        batch_size=2048,
        updates_per_step=3,
    )
