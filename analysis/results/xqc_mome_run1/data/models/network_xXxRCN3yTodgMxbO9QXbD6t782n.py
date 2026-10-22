
# ========== Start: GeneCrossed
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
def get_xqc_kwargs(
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    temp_lr: float = 3e-4,
    lr_end: float = 1e-5,
    tau: float = 0.005,
    policy_delay: int = 2,
    init_temperature: float = 0.1,
    weight_decay: float = 0.001,
    batch_size: int = 256,
    updates_per_step: int = 1,
):
    """
    Returns a dictionary of hyperparameters for an XQC model.

    Args:
    - actor_lr (float): Actor learning rate. Defaults to 3e-4.
    - critic_lr (float): Critic learning rate. Defaults to 3e-4.
    - temp_lr (float): Temperature learning rate. Defaults to 3e-4.
    - lr_end (float): Final learning rate. Defaults to 1e-5.
    - tau (float): Target network update rate. Defaults to 0.005.
    - policy_delay (int): Policy update delay. Defaults to 2.
    - init_temperature (float): Initial temperature value. Defaults to 0.1.
    - weight_decay (float): Weight decay rate. Defaults to 0.001.
    - batch_size (int): Mini-batch size. Defaults to 256.
    - updates_per_step (int): Number of updates per step. Defaults to 1.

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
