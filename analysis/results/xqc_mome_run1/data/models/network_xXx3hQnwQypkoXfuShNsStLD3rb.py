
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
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
        # Deeper critic network for learning complex representations
        hidden_dims_critic=[2560, 2304, 2048, 1024],
        
        # Larger actor network for better exploration and exploitation
        hidden_dims_actor=[1280, 1152, 1024, 768],
        
        # Number of critics remains constant at 2
        n_critics=2,
        
        # Enable batch normalization for stabilizing training
        use_batch_norm=True,
        
        # Enable weight normalization for further stabilization
        use_weight_norm=True,
        
        # Disable layer normalization based on the second snippet
        use_layer_norm=False,
        
        # Keep pre-activation batch normalization disabled
        pre_activation_bn=False,
        
        # Do not normalize the last layer
        normalize_last_layer=False,
        
        # Enable skip connections to facilitate training deeper networks
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
