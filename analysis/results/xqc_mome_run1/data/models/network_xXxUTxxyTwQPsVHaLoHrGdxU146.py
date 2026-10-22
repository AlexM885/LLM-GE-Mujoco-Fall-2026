
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
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
        hidden_dims_critic=[1024, 512, 256, 128],
        hidden_dims_actor=[256, 128, 64, 32],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=False,
    )
# --OPTION--
def get_xqc_kwargs(
    actor_lr: float = 3e-5,
    critic_lr: float = 3e-5,
    temp_lr: float = 3e-5,
    lr_end: float = 5e-6,
    tau: float = 0.005,
    policy_delay: int = 2,
    init_temperature: float = 0.05,
    weight_decay: float = 0.001,
    batch_size: int = 512,
    updates_per_step: int = 2,
):
    """
    Returns a dictionary of hyperparameters for an XQC model.

    Args:
    - actor_lr (float): Actor learning rate. Defaults to 3e-5.
    - critic_lr (float): Critic learning rate. Defaults to 3e-5.
    - temp_lr (float): Temperature learning rate. Defaults to 3e-5.
    - lr_end (float): Final learning rate. Defaults to 5e-6.
    - tau (float): Target network update rate. Defaults to 0.005.
    - policy_delay (int): Policy update delay. Defaults to 2.
    - init_temperature (float): Initial temperature value. Defaults to 0.05.
    - weight_decay (float): Weight decay rate. Defaults to 0.001.
    - batch_size (int): Mini-batch size. Defaults to 512.
    - updates_per_step (int): Number of updates per step. Defaults to 2.

    Returns:
    - dict: A dictionary containing the specified hyperparameters.
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
