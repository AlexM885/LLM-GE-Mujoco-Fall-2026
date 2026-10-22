
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
# ========== End:

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
        hidden_dims_critic=[1280, 640, 640, 640],
        hidden_dims_actor=[320, 320, 320, 320],
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
# === Combined XQC Hyperparameter Gene ===
# ===============================

def get_combined_xqc_kwargs():
    """
    Combines elements from different XQC hyperparameter configurations
    to potentially elevate model performance or efficiency.
    
    Returns:
        dict: A dictionary containing the combined hyperparameters.
    """
    return dict(
        # Balance learning rates for stability and convergence speed
        actor_lr=5e-4,  
        critic_lr=5e-4,
        temp_lr=3e-5,
        
        # Ensure learning rate ends at a value that prevents overshooting
        lr_end=1e-5,
        
        # Maintain a small tau for soft updates
        tau=0.01,
        
        # Policy delay remains consistent with original configurations
        policy_delay=2,
        
        # Initial temperature set for exploration
        init_temperature=0.1,
        
        # Weight decay to prevent overfitting
        weight_decay=0.01,
        
        # Increase batch size for better gradient estimation while considering resource constraints
        batch_size=256,
        
        # Updates per step remain as originally configured
        updates_per_step=1,
    )

# Example usage
if __name__ == "__main__":
    combined_kwargs = get_combined_xqc_kwargs()
    print("Combined XQC Hyperparameters:")
    for key, value in combined_kwargs.items():
        print(f"{key}: {value}")
