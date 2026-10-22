
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
    return dict(
        hidden_dims_critic=[1408, 1280, 768, 640],
        hidden_dims_actor=[704, 640, 384],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=False,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=4e-5,         # Lower learning rate for more precise convergence
        critic_lr=4e-5,
        temp_lr=4e-5,
        lr_end=5e-6,           # Lower end learning rate for finer tuning
        tau=0.01,               # Higher tau for smoother target network update
        policy_delay=3,        # Slightly longer delay to allow for better exploration
        init_temperature=0.05, # Lower initial temperature for less exploration at start
        weight_decay=0.001,    # Lower weight decay to prevent over-regularization
        batch_size=512,        # Batch size remains consistent for stable gradient estimation
        updates_per_step=3,    # Number of updates per step remains the same for consistent learning pace
    )
