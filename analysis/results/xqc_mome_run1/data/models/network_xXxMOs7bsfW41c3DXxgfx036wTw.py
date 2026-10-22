
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
        # Adopt a deeper critic network for more complex state-action value representation
        hidden_dims_critic=[512, 512, 512, 512, 256],
        
        # Use a wider actor network for generating diverse policies
        hidden_dims_actor=[512, 512, 256, 256],
        
        # Maintain the number of critics as specified
        n_critics=2,
        
        # Combine normalization techniques for stable training
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=True,
        
        # Other parameters remain consistent with the improvements
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
        init_temperature=1.0,
        weight_decay=0.01,
        batch_size=2048,
        updates_per_step=4,
    )
