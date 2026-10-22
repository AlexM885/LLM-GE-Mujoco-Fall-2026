
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
        hidden_dims_critic=[2048, 1920, 1024, 896],
        hidden_dims_actor=[1024, 960, 512],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=False,
        use_layer_norm=True,
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=True,
    )
# --OPTION--
def get_combined_xqc_kwargs():
    """
    Returns a dictionary of combined hyperparameters for XQC algorithm.
    
    This combination aims to balance stability, exploration, and optimization speed.
    """
    return dict(
        actor_lr=3e-4,  # Stable learning rate
        critic_lr=3e-4,  # Consistent with actor LR
        temp_lr=1e-4,   # Standard temperature learning rate
        lr_end=1e-5,    # Final learning rate for convergence
        tau=0.01,       # Soft update rate for target networks
        policy_delay=2, # Update critic twice as often as actor
        init_temperature=0.1, # Initial temperature for exploration
        weight_decay=0.0005, # Conservative against overfitting
        batch_size=512,      # Larger batch for faster training
        updates_per_step=3,  # Aggressive optimization
    )

# Example usage
combined_kwargs = get_combined_xqc_kwargs()
print(combined_kwargs)
