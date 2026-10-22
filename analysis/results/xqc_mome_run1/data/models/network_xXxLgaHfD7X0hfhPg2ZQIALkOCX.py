
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
    # Balancing depth and width, starting with larger dimensions and gradually reducing them
    return dict(
        hidden_dims_critic=[1024, 768, 512, 256],  # A mix, starting high and decreasing
        hidden_dims_actor=[512, 384, 256, 128],   # Similarly balanced for the actor
        n_critics=2,
        use_batch_norm=True,       # Maintaining batch normalization for stability
        use_weight_norm=True,      # Weight normalization for better generalization
        use_layer_norm=False,      # No layer normalization based on the original settings
        pre_activation_bn=True,    # Pre-activation batch normalization
        normalize_last_layer=True, # Normalizing the last layer
        skip_connections=False,    # No skip connections as per the originals
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=1e-4,
        lr_end=1e-5,
        tau=0.005,
        policy_delay=2,
        init_temperature=0.05,
        weight_decay=0.001,
        batch_size=256,
        updates_per_step=2,
    )
