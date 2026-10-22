
# ========== Start: GeneCrossed
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
        # Balanced critic network depth and width
        hidden_dims_critic=[1024, 1024, 512, 512],
        
        # Wider actor network for diverse policy generation
        hidden_dims_actor=[512, 512, 256, 256],
        
        # Maintain the specified number of critics
        n_critics=2,
        
        # Combine batch and weight normalization for stability
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        
        # Other parameters for consistency and improved training
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=False,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=1e-2,
        critic_lr=1e-2,
        temp_lr=1e-3,
        lr_end=1e-4,
        tau=0.01,
        policy_delay=2,
        init_temperature=0.05,
        weight_decay=0.01,
        batch_size=512,
        updates_per_step=2,
    )
