
# ========== Start: GeneCrossed

# ========== End:

def get_architecture_kwargs():
    return dict(
        # Combine deeper critic network from Snippet 1 with wider layers from Snippet 2
        hidden_dims_critic=[2048, 1024, 1024, 512, 256],
        
        # Use a moderately complex actor network
        hidden_dims_actor=[512, 512, 256],
        
        # Keep the number of critics as specified
        n_critics=2,
        
        # Enable batch and weight normalization for stability
        use_batch_norm=True,
        use_weight_norm=True,
        
        # Disable layer normalization to avoid over-normalization
        use_layer_norm=False,
        
        # Apply batch normalization before activation
        pre_activation_bn=True,
        
        # Normalize the last layer for better output scaling
        normalize_last_layer=True,
        
        # Do not use skip connections to keep the model simple
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
        hidden_dims_critic=[2048, 1024, 1024, 512, 256], 
        hidden_dims_actor=[512, 512, 512, 512], # Slightly increased depth of actor network
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
