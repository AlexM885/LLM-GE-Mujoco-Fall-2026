
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
        # Combining depth and width for critic and actor networks
        hidden_dims_critic=[1024, 1024, 512, 512, 256],  # Deeper and wider critic
        hidden_dims_actor=[512, 512, 256, 256],  # Wider actor
        
        n_critics=2,
        
        # Utilizing multiple normalization techniques
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=True,  # Enabling layer normalization
        
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=False,
    )
# --OPTION--
def get_combined_xqc_kwargs():
    """
    Returns a dictionary of combined hyperparameters for the XQC algorithm.
    
    The selection aims to balance convergence speed, stability, exploration, and regularization.
    """
    return dict(
        actor_lr=8e-4,  # Faster initial learning
        critic_lr=8e-4,  # Faster initial learning
        temp_lr=1e-4,   # Moderate learning rate for temperature
        lr_end=1e-5,    # Lower final learning rate for fine-tuning
        tau=0.01,       # Moderate target network update rate
        policy_delay=2, # Delayed actor updates for stability
        init_temperature=0.1, # Moderate initial exploration
        weight_decay=0.01,    # Regularization to prevent overfitting
        batch_size=256,      # Balance between update stability and memory usage
        updates_per_step=1,  # Standard number of updates per environment step
    )

# Example usage
combined_kwargs = get_combined_xqc_kwargs()
print(combined_kwargs)
