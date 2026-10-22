
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
        # Combine deeper critic network from Snippet 1 with wider layers from Snippet 2
        hidden_dims_critic=[3072, 2304, 2048, 1536, 1024],
        
        # Use a mix of wide and deep layers for the actor network
        hidden_dims_actor=[1024, 768, 512, 384, 256, 128],
        
        # Keep the number of critics as specified
        n_critics=2,
        
        # Enable batch normalization for stability
        use_batch_norm=True,
        
        # Disable weight normalization based on Snippet 2
        use_weight_norm=False,
        
        # Enable layer normalization for additional stability
        use_layer_norm=True,
        
        # Do not apply batch normalization before activation
        pre_activation_bn=False,
        
        # Normalize the last layer for better output scaling
        normalize_last_layer=False,
        
        # Include skip connections for better gradient flow
        skip_connections=True,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=2e-4,
        lr_end=1e-5,
        tau=0.005,
        policy_delay=2,
        init_temperature=0.1,
        weight_decay=0.001,
        batch_size=256,
        updates_per_step=1,
    )
