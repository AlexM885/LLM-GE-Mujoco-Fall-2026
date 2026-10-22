
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
    # Increase critic depth and width for improved representation learning
    hidden_dims_critic = [1024, 2048, 4096, 2048, 1024]
    
    # Increase actor capacity for more effective exploration
    hidden_dims_actor = [256, 512, 256]
    
    # Retain the required number of critics
    n_critics = 2
    
    # Utilize batch normalization for enhanced stability
    use_batch_norm = True
    
    # Enable weight normalization to potentially improve stability and performance
    use_weight_norm = True
    
    # Disable layer normalization to prevent interference with batch normalization
    use_layer_norm = False
    
    # Apply batch normalization prior to activation for smoother gradients
    pre_activation_bn = True
    
    # Normalize the final layer to mitigate gradient explosion risks
    normalize_last_layer = True
    
    # Incorporate skip connections to facilitate efficient gradient propagation
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
        actor_lr=1e-3,
        critic_lr=1e-3,
        temp_lr=1e-4,
        lr_end=1e-5,
        tau=0.01,
        policy_delay=2,
        init_temperature=0.1,
        weight_decay=0.01,
        batch_size=512,
        updates_per_step=3,
    )
