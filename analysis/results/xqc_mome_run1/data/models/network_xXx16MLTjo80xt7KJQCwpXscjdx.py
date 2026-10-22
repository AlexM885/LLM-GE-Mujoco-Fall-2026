
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
# ========== End:

# ========== Start: GeneCrossed

# ========== End:
# ========== End:

# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
# ========== End:

# ========== Start: GeneCrossed

# ========== End:
# ========== End:

# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:

# ========== Start: GeneCrossed

# ========== End:
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
        hidden_dims_critic=[3840, 2880, 1920],
        hidden_dims_actor=[1920, 960, 480],
        n_critics=2,
        use_batch_norm=False,
        use_weight_norm=True,
        use_layer_norm=True,
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=True,
    )
# --OPTION--
def get_xqc_kwargs(
    actor_lr=8e-4,
    critic_lr=8e-4,
    temp_lr=5e-5,
    lr_end=5e-5,
    tau=0.01,
    policy_delay=2,
    init_temperature=0.1,
    weight_decay=0.01,
    batch_size=512,
    updates_per_step=1,
):
    """
    Returns a dictionary of hyperparameters for an XQC model.
    
    Parameters:
    - actor_lr: Learning rate for the actor.
    - critic_lr: Learning rate for the critic.
    - temp_lr: Learning rate for the temperature.
    - lr_end: Final learning rate.
    - tau: Target network update rate.
    - policy_delay: Delay for policy updates.
    - init_temperature: Initial temperature value.
    - weight_decay: Weight decay rate.
    - batch_size: Batch size for training.
    - updates_per_step: Number of updates per step.
    
    Returns:
    A dictionary containing all specified hyperparameters.
    """
    return dict(
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        temp_lr=temp_lr,
        lr_end=lr_end,
        tau=tau,
        policy_delay=policy_delay,
        init_temperature=init_temperature,
        weight_decay=weight_decay,
        batch_size=batch_size,
        updates_per_step=updates_per_step,
    )
