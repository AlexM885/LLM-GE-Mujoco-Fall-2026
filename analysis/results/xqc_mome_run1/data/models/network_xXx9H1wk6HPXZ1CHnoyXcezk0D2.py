
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
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
        # Combining layer dimensions to potentially capture more complex patterns
        hidden_dims_critic=[2048, 1024, 768, 512],
        hidden_dims_actor=[1024, 512, 384, 256],
        
        # Keeping n_critics as required
        n_critics=2,
        
        # Enabling batch normalization for stability, but disabling weight normalization
        use_batch_norm=True,
        use_weight_norm=False,
        
        # Enabling layer normalization for additional stability
        use_layer_norm=True,
        
        # Disabling pre-activation batch normalization and last layer normalization
        pre_activation_bn=False,
        normalize_last_layer=False,
        
        # Enabling skip connections to help with deeper networks
        skip_connections=True,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=1e-4,
        critic_lr=1e-4,
        temp_lr=5e-5,
        lr_end=5e-6,
        tau=0.01,
        policy_delay=4,
        init_temperature=0.5,
        weight_decay=0.0005,
        batch_size=512,
        updates_per_step=2,
    )
