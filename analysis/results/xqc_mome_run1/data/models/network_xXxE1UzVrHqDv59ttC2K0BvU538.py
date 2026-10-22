
# ========== Start: GeneCrossed

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
        hidden_dims_critic=[1280, 960, 640],
        hidden_dims_actor=[640, 480, 320],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=False,
        use_layer_norm=True,
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=False,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        # Balanced learning rates, slightly higher than the lowest to ensure progress without overshooting
        actor_lr=2e-4,
        critic_lr=2e-4,
        temp_lr=1e-4,
        
        # Gradual decrease to a lower end learning rate
        lr_end=1e-5,
        
        # Moderate tau for stable target network updates
        tau=0.0075,
        
        # Policy delay remains consistent with both originals
        policy_delay=2,
        
        # Initial temperature as in the originals
        init_temperature=0.1,
        
        # Weight decay adjusted to mitigate overfitting without hindering learning
        weight_decay=0.005,
        
        # Batch size increased for more stable gradients, assuming computational resources allow
        batch_size=192,
        
        # Updates per step remain at 1 for simplicity and consistency with originals
        updates_per_step=1,
    )
