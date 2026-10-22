
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
    # Exploring deeper and wider critic networks
    hidden_dims_critic = [1024, 1024, 1024, 1024, 1024]
    
    # Shallower actor networks for simplicity
    hidden_dims_actor = [256, 256, 256]
    
    # Keeping the number of critics constant as per requirements
    n_critics = 2
    
    # Enabling batch normalization but disabling weight normalization for critics
    use_batch_norm = True
    use_weight_norm = False
    
    # Disabling layer normalization
    use_layer_norm = False
    
    # Applying batch normalization before activation
    pre_activation_bn = True
    
    # Normalizing the last layer for stability
    normalize_last_layer = True
    
    # Enabling skip connections for better gradient flow
    skip_connections = True
    
    return dict(
        hidden_dims_critic=hidden_dims_critic,
        hidden_dims_actor=hidden_dims_actor,
        n_critics=n_critics,
        use_batch_norm=use_batch_norm,
        use_weight_norm=use_weight_norm,
        use_layer_norm=use_layer_norm,
        pre_activation_bn=pre_activation_bn,
        normalize_last_layer=normalize_last_layer,
        skip_connections=skip_connections,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=5e-5,
        lr_end=1e-5,
        tau=0.01,
        policy_delay=2,
        init_temperature=0.1,
        weight_decay=0.001,
        batch_size=128,
        updates_per_step=2,
    )
