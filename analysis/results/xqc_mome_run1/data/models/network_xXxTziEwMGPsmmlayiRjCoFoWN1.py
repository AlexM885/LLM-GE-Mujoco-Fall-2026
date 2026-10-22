
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
    # Increasing critic depth and width for improved representation learning
    hidden_dims_critic = [512, 1024, 2048, 1024, 512]
    
    # Maintaining a simpler actor architecture for efficient exploration
    hidden_dims_actor = [128, 256, 128]
    
    # Retaining the required number of critics
    n_critics = 2
    
    # Utilizing batch normalization for enhanced stability
    use_batch_norm = True
    
    # Disabling weight normalization due to potential redundancy with batch normalization
    use_weight_norm = False
    
    # Avoiding layer normalization to prevent interference with batch normalization
    use_layer_norm = False
    
    # Applying batch normalization prior to activation for smoother gradients
    pre_activation_bn = True
    
    # Normalizing the final layer to mitigate gradient explosion risks
    normalize_last_layer = True
    
    # Incorporating skip connections to facilitate efficient gradient propagation
    skip_connections = True
    
    return dict(
        hidden_dims_critic=hidden_dims_critic,
        hidden_dims_actor=hidden_dims_actor,
        n_critics=n_critics,
        use_batch_norm=use_batch_norm,
        use_weight_norm=use_weight_norm,
        use_layer_norm=use_layer_norm,
        pre_activation_bn=pre_activation_bn,
        normalize_last_layer=normalize_last_layer,
        skip_connections=skip_connections,
    )
# --OPTION--
def get_xqc_kwargs(
    actor_lr: float = 5e-4,
    critic_lr: float = 5e-4,
    temp_lr: float = 5e-4,
    lr_end: float = 1e-4,
    tau: float = 0.005,
    policy_delay: int = 1,
    init_temperature: float = 0.5,
    weight_decay: float = 0.001,
    batch_size: int = 256,
    updates_per_step: int = 2,
):
    """
    Returns a dictionary of hyperparameters for an XQC model.

    Args:
    - actor_lr (float): Actor learning rate. Defaults to 5e-4.
    - critic_lr (float): Critic learning rate. Defaults to 5e-4.
    - temp_lr (float): Temperature learning rate. Defaults to 5e-4.
    - lr_end (float): Final learning rate. Defaults to 1e-4.
    - tau (float): Target network update rate. Defaults to 0.005.
    - policy_delay (int): Policy update delay. Defaults to 1.
    - init_temperature (float): Initial temperature value. Defaults to 0.5.
    - weight_decay (float): Weight decay rate. Defaults to 0.001.
    - batch_size (int): Mini-batch size. Defaults to 256.
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
