
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:

# ========== Start: GeneCrossed

# ========== End:
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
        hidden_dims_critic=[512, 512, 256, 256],
        hidden_dims_actor=[256, 256, 128, 128],
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
    
    This function combines elements from different approaches to potentially improve the model's predictive metrics.
    """
    return dict(
        # Moderate learning rates for stability and convergence
        actor_lr=2e-4,
        critic_lr=2e-4,
        temp_lr=2e-5,
        
        # Lower end learning rate for fine-tuning
        lr_end=1e-5,
        
        # Small update rate for target networks
        tau=0.01,
        
        # Delayed policy updates for stability
        policy_delay=2,
        
        # Initial temperature for exploration
        init_temperature=0.1,
        
        # Moderate weight decay to prevent overfitting
        weight_decay=0.005,
        
        # Larger batch size for more efficient training
        batch_size=192,
        
        # Increased updates per step for faster convergence
        updates_per_step=1.5,
    )

# Example usage
enhanced_kwargs = get_enhanced_xqc_kwargs()
print(enhanced_kwargs)
