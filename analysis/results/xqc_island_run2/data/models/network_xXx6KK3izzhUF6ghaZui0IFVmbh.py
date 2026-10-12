
# ========== Start: GeneCrossed

# ========== End:

def get_architecture_kwargs():
    # Define the architecture settings for the XQC agent.
    # Consider modifying these parameters to see their effect on performance.
    
    # Number of critics should remain at 2 as per the note.
    n_critics = 2
    
    # Hidden dimensions for the critic and actor networks.
    # These are key areas to experiment with for improving performance.
    hidden_dims_critic = [2048, 1024, 1024, 512]  # Example dimensions for the critic network
    hidden_dims_actor = [512, 512, 512, 256]      # Example dimensions for the actor network
    
    # Normalization techniques can significantly affect training stability and speed.
    use_batch_norm = True       # Enable batch normalization
    use_weight_norm = True      # Enable weight normalization
    use_layer_norm = False      # Disable layer normalization by default
    pre_activation_bn = True    # Apply batch normalization before activation
    normalize_last_layer = True # Normalize the last layer
    
    # Skip connections can aid in training deeper networks.
    skip_connections = False    # Disable skip connections by default
    
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
# -- NOTE --
# Note: XQC (ICLR 2026) is an off-policy actor-critic in JAX: a tanh-Gaussian actor
# and an ensemble of categorical critics, both MLPs with batch norm and weight norm.
# get_architecture_kwargs() must return a dict of network settings for the XQC agent.
# Only plain Python values (ints, floats, bools, strings, lists of ints) are allowed.
# Prefer mutating hidden_dims_critic, hidden_dims_actor and the
# normalization flags. n_critics must stay 2.# -- NOTE --
def get_architecture_kwargs():
    return dict(
        hidden_dims_critic=[2048, 1024, 1024, 512],
        hidden_dims_actor=[512, 512, 512, 256],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=False,
    )
# --OPTION--

# ===============================
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    return dict(
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        lr_end=3e-5,
        tau=0.005,
        policy_delay=3,
        init_temperature=0.01,
        weight_decay=0.0,
        batch_size=256,
        updates_per_step=2,
    )
