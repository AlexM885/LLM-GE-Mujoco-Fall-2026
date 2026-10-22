
# ========== Start: GeneCrossed

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
        # Increased depth and width for both critic and actor networks
        hidden_dims_critic=[1024, 768, 512, 256],
        hidden_dims_actor=[512, 384, 256, 128],
        
        # Retaining the ensemble size as per the problem statement
        n_critics=2,
        
        # Applying batch normalization and weight normalization
        use_batch_norm=True,
        use_weight_norm=True,
        
        # Disabling layer normalization but enabling pre-activation BN
        use_layer_norm=False,
        pre_activation_bn=True,
        
        # Normalizing the last layer
        normalize_last_layer=True,
        
        # Enabling skip connections for better gradient flow
        skip_connections=True,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        lr_end=1e-5,
        tau=0.005,
        policy_delay=2,
        init_temperature=0.1,
        weight_decay=0.01,
        batch_size=256,
        updates_per_step=1,
    )
