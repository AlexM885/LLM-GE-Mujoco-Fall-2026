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
        hidden_dims_critic=[1024, 512, 512, 256],  # Slightly increased critic capacity
        hidden_dims_actor=[256, 256, 256, 256],  # Keeping actor capacity consistent
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=False,  # Disabling weight norm to see if batch norm suffices
        use_layer_norm=True,  # Enabling layer norm for additional regularization
        pre_activation_bn=False,  # Changing BN application point
        normalize_last_layer=False,  # Not normalizing last layer for simplicity
        skip_connections=False,
    )
# --OPTION--
# ===============================
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    return dict(
        actor_lr=5e-4,
        critic_lr=5e-4,
        temp_lr=1e-4,
        lr_end=1e-5,
        tau=0.002,
        policy_delay=1,
        init_temperature=0.05,
        weight_decay=0.0005,
        batch_size=512,
        updates_per_step=3,
    )
