
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
        hidden_dims_critic=[1024, 512, 512, 256],
        hidden_dims_actor=[512, 256, 256, 128],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=False,
        skip_connections=True,
    )
# --OPTION--
# ===============================
# === Enhanced XQC Hyperparameter Gene ===
# ===============================

def get_enhanced_xqc_kwargs():
    """
    Returns a dictionary of hyperparameters for an enhanced XQC model.
    This combines potentially beneficial elements from two alternative sets of hyperparameters.
    """
    return dict(
        # Higher learning rates for faster convergence
        actor_lr=5e-4,
        critic_lr=5e-4,
        temp_lr=5e-5,
        
        # Lower end learning rate to ensure convergence
        lr_end=1e-5,
        
        # Small tau for soft updates
        tau=0.01,
        
        # Policy delay to stabilize training
        policy_delay=2,
        
        # Initial temperature
        init_temperature=0.1,
        
        # Smaller weight decay to reduce overfitting
        weight_decay=0.001,
        
        # Larger batch size for better gradient estimation
        batch_size=256,
        
        # More updates per step for improved stability
        updates_per_step=2,
    )

# Example usage
if __name__ == "__main__":
    enhanced_kwargs = get_enhanced_xqc_kwargs()
    print("Enhanced XQC Hyperparameters:", enhanced_kwargs)
