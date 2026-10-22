
# ========== Start: GeneCrossed

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
        # Increased capacity for critic network, aiming for a balance
        hidden_dims_critic=[1536, 1536, 768, 768],
        
        # Moderate increase in capacity for actor network
        hidden_dims_actor=[768, 384, 192, 96],
        
        # Number of critics remains constant
        n_critics=2,
        
        # Enable batch normalization for stability
        use_batch_norm=True,
        
        # Enable weight normalization for further stability
        use_weight_norm=True,
        
        # Enable layer normalization for faster convergence
        use_layer_norm=True,
        
        # Apply batch normalization before activation
        pre_activation_bn=True,
        
        # Normalize the last layer for output stability
        normalize_last_layer=True,
        
        # Include skip connections for deeper networks
        skip_connections=True,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=1e-3,
        critic_lr=1e-3,
        temp_lr=1e-6,
        lr_end=1e-5,
        tau=0.01,
        policy_delay=4,
        init_temperature=0.05,
        weight_decay=0.01,
        batch_size=512,
        updates_per_step=3,
    )
