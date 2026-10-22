
# ========== Start: GeneCrossed

# ========== End:

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
def get_architecture_kwargs():
    return dict(
        hidden_dims_critic=[512, 1024, 1024, 512],
        hidden_dims_actor=[128, 256, 256, 128],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=False,
        skip_connections=True,
    )
# --OPTION--
def get_combined_xqc_kwargs():
    return dict(
        # Moderate learning rates for actor and critic
        actor_lr=1e-3,
        critic_lr=1e-3,
        # Lower temperature learning rate for more stable entropy adjustment
        temp_lr=1e-5,
        # Gradual decrease to a low final learning rate
        lr_end=1e-5,
        # Small tau for stable target network updates
        tau=0.01,
        # Balanced policy delay for updating actor and critic
        policy_delay=2,
        # Initial temperature encouraging some exploration
        init_temperature=0.1,
        # Moderate weight decay for regularization
        weight_decay=0.001,
        # Efficient batch size balancing memory and gradient stability
        batch_size=256,
        # Updates per step for a good trade-off between speed and stability
        updates_per_step=1,
    )
