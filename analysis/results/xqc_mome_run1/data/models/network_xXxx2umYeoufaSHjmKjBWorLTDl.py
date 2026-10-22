
# ========== Start: GeneCrossed
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
        hidden_dims_critic=[2560, 2304, 2048, 1024],
        hidden_dims_actor=[1152, 1024, 768, 512],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=False,
        normalize_last_layer=True,
        skip_connections=False,
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
        weight_decay=0.005,  # Moderate regularization
        batch_size=512,      # Increased for faster training
        updates_per_step=3,  # Increased for faster training
    )
