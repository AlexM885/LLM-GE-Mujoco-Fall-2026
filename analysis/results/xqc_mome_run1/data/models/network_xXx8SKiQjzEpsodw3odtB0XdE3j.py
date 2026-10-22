
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
# ========== End:

# ========== Start: GeneCrossed

# ========== End:

# ========== Start: GeneCrossed

# ========== End:
# ========== End:

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
        # Slightly increased capacity for the critic network
        hidden_dims_critic=[1152, 1152, 1152, 1152],
        
        # Moderate increase in capacity for the actor network
        hidden_dims_actor=[576, 288, 144, 72],
        
        # Number of critics remains constant
        n_critics=2,
        
        # Enable batch normalization
        use_batch_norm=True,
        
        # Enable weight normalization for stability
        use_weight_norm=True,
        
        # Disable layer normalization
        use_layer_norm=False,
        
        # Apply batch normalization before activation
        pre_activation_bn=True,
        
        # Normalize the last layer for output stability
        normalize_last_layer=True,
        
        # No skip connections
        skip_connections=False,
    )
# --OPTION--
def get_enhanced_xqc_kwargs():
    return dict(
        actor_lr=1e-3,
        critic_lr=1e-3,
        temp_lr=1e-4,
        lr_end=1e-6,
        tau=0.01,
        weight_decay=0.0005,
        batch_size=512,
        updates_per_step=2,
        policy_delay=2,
        init_temperature=0.1,
    )
