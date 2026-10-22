
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
        hidden_dims_critic=[2560, 2304, 2048, 1024], 
        hidden_dims_actor=[1280, 1152, 1024, 768],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=True,
    )
# --OPTION--
def get_combined_xqc_kwargs():
    """
    Returns a dictionary of combined hyperparameters from Snippet 1 and Snippet 2.
    This combination aims to leverage faster initial learning rates with a larger batch size and lower end learning rate for refined tuning.
    """
    return dict(
        actor_lr=3e-4,  # Faster initial learning rate
        critic_lr=3e-4,  # Faster initial learning rate
        temp_lr=1e-4,   # Original temperature learning rate for stability
        lr_end=5e-6,    # Lower end learning rate for finer tuning
        tau=0.005,      # Original tau value for soft update
        policy_delay=2, # Original policy delay
        init_temperature=0.05, # Original initial temperature
        weight_decay=0.001,    # Original weight decay
        batch_size=512,        # Larger batch size for better gradient estimation
        updates_per_step=2,    # Original updates per step
    )
