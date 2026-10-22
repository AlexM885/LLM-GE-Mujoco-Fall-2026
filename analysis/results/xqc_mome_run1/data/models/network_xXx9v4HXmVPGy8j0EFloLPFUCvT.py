
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
    # Deeper critic network for improved representation learning
    hidden_dims_critic = [1024, 2048, 1024, 512]
    
    # Slightly deeper actor network for better exploration
    hidden_dims_actor = [256, 512, 256]
    
    # Required number of critics remains unchanged
    n_critics = 2
    
    # Enable batch normalization for stability
    use_batch_norm = True
    
    # Disable weight normalization to avoid potential redundancy
    use_weight_norm = False
    
    # Disable layer normalization as it may interfere with batch normalization
    use_layer_norm = False
    
    # Apply batch normalization before activation for smoother gradients
    pre_activation_bn = True
    
    # Normalize the last layer to prevent gradient explosion
    normalize_last_layer = True
    
    # Disable skip connections as they may not significantly impact performance
    skip_connections = False
    
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
        # Use a moderate learning rate for actors and critics, slightly favoring exploration
        actor_lr=2e-4,
        critic_lr=2e-4,
        # Keep temperature learning rate as in the second snippet for better entropy regularization
        temp_lr=1e-4,
        # Gradually decrease learning rate to ensure convergence
        lr_end=1e-5,
        # Choose a tau value that balances stability and adaptability
        tau=0.0075,
        # Maintain a moderate policy delay to avoid overfitting while allowing for timely adaptations
        policy_delay=2,
        # Initialize temperature similarly to encourage exploration
        init_temperature=0.1,
        # Apply a moderate weight decay to prevent overfitting
        weight_decay=0.005,
        # Use a larger batch size for more stable gradients, considering computational resources
        batch_size=512,
        # Perform multiple updates per step to speed up training without overwhelming the system
        updates_per_step=1.5,
    )
