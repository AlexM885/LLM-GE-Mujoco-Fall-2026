
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
# ========== End:

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
        hidden_dims_critic=[2048, 1024, 1024, 512],
        hidden_dims_actor=[512, 512, 256, 256],
        n_critics=2,
        use_batch_norm=False,
        use_weight_norm=True,
        use_layer_norm=True,
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=True,
    )
# --OPTION--
def get_combined_xqc_kwargs():
    return dict(
        actor_lr=5e-3,  # Moderate starting learning rate
        critic_lr=5e-3,  # Consistent with actor_lr for balanced learning
        temp_lr=1e-2,   # Slightly higher for adaptive exploration
        lr_end=1e-5,    # Gentle learning rate decay
        tau=0.01,       # Reasonable target network update rate for stability
        policy_delay=2, # Delayed actor updates for improved stability
        init_temperature=0.5, # Encourages exploration
        weight_decay=0.01,    # Standard regularization strength
        batch_size=1024,      # Larger batch size for efficient training
        updates_per_step=2,   # Frequent updates for rapid adaptation
    )
