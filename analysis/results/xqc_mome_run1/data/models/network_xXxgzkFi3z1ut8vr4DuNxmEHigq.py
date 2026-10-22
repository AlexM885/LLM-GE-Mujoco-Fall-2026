
# ========== Start: GeneCrossed

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
        hidden_dims_critic=[1280, 896, 896, 896],
        hidden_dims_actor=[512, 512, 512, 512],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=False,
    )
# --OPTION--
# ===============================
# === Enhanced XQC Hyperparameter Gene ===
# ===============================

def get_enhanced_xqc_kwargs():
    """
    Returns a dictionary of hyperparameters for an enhanced XQC model.
    
    This version combines elements from different configurations to potentially improve predictive metrics.
    """
    return dict(
        # Moderate learning rates for stability and convergence
        actor_lr=1e-4,
        critic_lr=1e-4,
        temp_lr=1e-4,
        
        # Gradual learning rate end for smoother convergence
        lr_end=1e-5,
        
        # Small tau for soft updates
        tau=0.005,
        
        # Policy delay adjusted for potential improvement in certain scenarios
        policy_delay=2,
        
        # Initialization temperature encouraging exploration
        init_temperature=0.02,
        
        # Weight decay to prevent overfitting
        weight_decay=0.0005,
        
        # Batch size optimized for training efficiency
        batch_size=384,
        
        # Updates per step for balancing exploration and exploitation
        updates_per_step=2,
    )

# Example usage
enhanced_kwargs = get_enhanced_xqc_kwargs()
print(enhanced_kwargs)
