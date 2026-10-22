
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
        hidden_dims_critic=[512, 512, 512, 512, 128],  # Slightly reduce the last layer size for efficiency
        hidden_dims_actor=[512, 512, 256, 128],  # Reduce the last layer size for better generalization
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,  # Disable layer normalization for simplicity
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=False,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=3e-3,
        critic_lr=3e-3,
        temp_lr=1e-4,
        lr_end=1e-5,
        tau=0.005,
        policy_delay=1,
        init_temperature=0.1,
        weight_decay=0.001,
        batch_size=1024,
        updates_per_step=3,
    )
