
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
# ========== End:

# ========== Start: GeneCrossed

# ========== End:
# ========== End:

# ========== Start: GeneCrossed
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
        hidden_dims_critic=[512, 512, 256],
        hidden_dims_actor=[256, 256],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=False,
        use_layer_norm=False,
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=False,
    )
# --OPTION--
def get_xqc_kwargs():
    # Moderate learning rates with potential scheduling
    actor_lr = 1e-3  # Middle ground between the two snippets
    critic_lr = 1e-3
    temp_lr = 1e-4  # Lower temperature learning rate as in both snippets
    
    # Tau value balancing stability and adaptability
    tau = 0.0075  # Midpoint of the values from the two snippets
    
    # Policy delay allowing for exploration
    policy_delay = 1  # As in the second snippet for faster adaptation
    
    # Initial temperature
    init_temperature = 0.1  # Consistent across both snippets
    
    # Weight decay to prevent overfitting
    weight_decay = 0.005  # Between the values of the two snippets
    
    # Batch size and updates per step for efficient training
    batch_size = 512  # Compromise between the two snippet values
    updates_per_step = 2  # More than the first snippet but less than the second
    
    # Learning rate end for scheduling
    lr_end = 1e-5  # Consistent across both snippets for convergence
    
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
