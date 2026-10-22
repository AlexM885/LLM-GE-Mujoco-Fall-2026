
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
        # Balanced approach to hidden dimensions, mixing depth and width
        hidden_dims_critic=[2048, 1024, 512, 256],
        hidden_dims_actor=[1024, 512, 256, 128],
        
        # Selective application of normalization techniques
        n_critics=2,
        use_batch_norm=True,  # General stabilization
        use_weight_norm=True,  # Scaling
        use_layer_norm=False,  # Unless specific layers benefit
        pre_activation_bn=True,  # To reduce internal covariate shift
        normalize_last_layer=True,  # For output stability
        
        # Include skip connections for deep networks
        skip_connections=True,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=1e-3,
        critic_lr=1e-3,
        temp_lr=5e-4,
        lr_end=5e-5,
        tau=0.002,
        policy_delay=1,
        init_temperature=0.01,
        weight_decay=0.0005,
        batch_size=512,
        updates_per_step=3,
    )
