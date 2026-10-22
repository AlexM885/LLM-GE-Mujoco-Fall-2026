
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:

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
    # Enhanced version considering deeper and wider networks
    # with adjusted normalization techniques
    return dict(
        hidden_dims_critic=[2048, 1536, 1024, 768],  # Increased dimensions
        hidden_dims_actor=[512, 512, 384, 384, 256],  # Adjusted dimensions
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=True,  # Enabled skip connections for potential improvement
    )
# --OPTION--
def get_xqc_kwargs():
    """
    Returns a dictionary of hyperparameters for the XQC model.
    
    This implementation combines elements from different approaches to potentially enhance predictive metrics.
    """
    return dict(
        actor_lr=1e-4,         # Increased learning rate for faster convergence
        critic_lr=1e-4,
        temp_lr=1e-4,
        lr_end=1e-5,           # Adjusted end learning rate for fine-tuning
        tau=0.002,             # Decreased tau for more aggressive updates
        policy_delay=1,        # Reduced policy delay for more frequent updates
        init_temperature=0.1,  # Increased initial temperature for more exploration
        weight_decay=0.01,     # Slightly increased regularization
        batch_size=1024,       # Increased batch size for improved training efficiency
        updates_per_step=3,    # More frequent updates for faster convergence
    )
