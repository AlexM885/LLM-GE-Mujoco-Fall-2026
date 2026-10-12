
# ========== Start: GeneCrossed

# ========== End:

def get_architecture_kwargs():
    """
    Returns a dictionary containing the architecture settings for the XQC agent.
    
    This implementation combines elements from two alternative configurations to potentially
    elevate the model's performance or efficiency.
    """
    return dict(
        # Critic configuration with increased capacity
        hidden_dims_critic=[1024, 1024, 512],
        
        # Actor configuration with slightly reduced complexity for efficiency
        hidden_dims_actor=[256, 256, 256],
        
        # Number of critics; must remain 2 as per the problem statement
        n_critics=2,
        
        # Normalization flags
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=False,
    )
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
        hidden_dims_critic=[1024, 1024, 512, 512],
        hidden_dims_actor=[512, 512, 256, 256],
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
        updates_per_step=2,
    )
