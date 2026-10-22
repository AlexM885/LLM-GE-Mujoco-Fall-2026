
# ========== Start: GeneCrossed

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
# -- NOTE --
# Note: XQC (ICLR 2026) is an off-policy actor-critic in JAX: a tanh-Gaussian actor
# and an ensemble of categorical critics, both MLPs with batch norm and weight norm.
# get_architecture_kwargs() must return a dict of network settings for the XQC agent.
# Only plain Python values (ints, floats, bools, strings, lists of ints) are allowed.
# Prefer mutating hidden_dims_critic, hidden_dims_actor and the
# normalization flags. n_critics must stay 2.# -- NOTE --
def get_architecture_kwargs():
    return dict(
        hidden_dims_critic=[2048, 2048, 1280, 640],  # Combining increased capacity with reduced dimensions for better generalization
        hidden_dims_actor=[1024, 512, 256, 256],  # Gradually decreasing dimensions to capture complex features
        n_critics=2,
        use_batch_norm=True,  # Batch normalization can help stabilize training
        use_weight_norm=True,  # Weight normalization can improve optimization
        use_layer_norm=False,  # Layer normalization might not be necessary with batch and weight normalization
        pre_activation_bn=True,  # Applying BN before activation can improve representation learning
        normalize_last_layer=True,  # Normalizing the last layer can help with stability and generalization
        skip_connections=False,  # Skip connections may not be necessary with the chosen normalization strategies
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=1e-3,
        critic_lr=1e-3,
        temp_lr=1e-4,
        lr_end=1e-5,
        tau=0.01,
        policy_delay=2,
        init_temperature=0.1,
        weight_decay=0.01,
        batch_size=512,
        updates_per_step=3,
    )
