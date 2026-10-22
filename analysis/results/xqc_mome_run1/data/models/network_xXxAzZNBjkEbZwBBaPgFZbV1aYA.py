
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
        hidden_dims_critic=[2048, 1536, 1024],
        hidden_dims_actor=[512, 384, 256],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=False,
        use_layer_norm=False,
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=False,
    )
# --OPTION--
def get_combined_xqc_kwargs():
    return dict(
        actor_lr=1e-4,  # Stable convergence
        critic_lr=1e-4,  # Consistent with actor_lr for balanced learning
        temp_lr=5e-5,   # Faster temperature adaptation without being too aggressive
        lr_end=5e-6,    # Allows for finer tuning towards the end of training
        tau=0.01,       # Soft update parameter, consistent across both snippets
        policy_delay=2, # Delayed policy updates, as in both snippets
        init_temperature=0.1, # Initial temperature, same in both snippets
        weight_decay=0.0005,  # Moderate regularization to prevent overfitting
        batch_size=256,      # A moderate batch size balancing speed and variability
        updates_per_step=1,  # Standard single update per step for simplicity
    )
