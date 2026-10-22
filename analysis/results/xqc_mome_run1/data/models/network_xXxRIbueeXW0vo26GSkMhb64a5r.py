
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
        hidden_dims_critic=[512, 512, 256, 256],
        hidden_dims_actor=[256, 256, 128, 128],
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
    # Balancing learning rates for stability and convergence speed
    actor_lr = 2e-4  
    critic_lr = 2e-4  
    temp_lr = 1e-4
    
    # Choosing a moderate tau value for update stability
    tau = 0.0075  
    
    # Setting initial temperature for a balanced start
    init_temperature = 0.075  
    
    # Moderate weight decay to prevent overfitting without hindering learning
    weight_decay = 0.005  
    
    # Increasing batch size and updates per step for faster convergence
    batch_size = 192  
    updates_per_step = 1.5  
    
    # Other parameters
    lr_end = 1e-5
    policy_delay = 2
    
    return dict(
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        temp_lr=temp_lr,
        lr_end=lr_end,
        tau=tau,
        policy_delay=policy_delay,
        init_temperature=init_temperature,
        weight_decay=weight_decay,
        batch_size=batch_size,
        updates_per_step=updates_per_step,
    )
