
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
# ========== End:

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
        # Deeper and slightly adjusted dimensions for both critic and actor
        hidden_dims_critic=[2048, 1536, 1024, 768, 512],
        hidden_dims_actor=[768, 512, 384, 256, 128],
        
        # Keep the number of critics constant as per the problem statement
        n_critics=2,
        
        # Enable batch normalization for stability
        use_batch_norm=True,
        
        # Disable weight normalization based on the second snippet's approach
        use_weight_norm=False,
        
        # Enable layer normalization for additional stabilization across layers
        use_layer_norm=True,
        
        # Adjust other parameters based on the insights from both snippets
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=True,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=1e-4,
        lr_end=1e-5,
        tau=0.005,
        policy_delay=4,
        init_temperature=0.5,
        weight_decay=0.001,
        batch_size=1024,
        updates_per_step=3,
    )
