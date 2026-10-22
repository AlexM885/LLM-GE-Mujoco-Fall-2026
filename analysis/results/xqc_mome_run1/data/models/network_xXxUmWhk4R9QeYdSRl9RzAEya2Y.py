
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
        hidden_dims_critic=[2048, 2048, 1024, 512],
        hidden_dims_actor=[512, 512, 256],
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
# === Enhanced XQC Hyperparameters ===
# ===============================

def get_enhanced_xqc_kwargs():
    """
    Returns a dictionary of enhanced hyperparameters for the XQC model.
    
    This combination aims to leverage the strengths of both provided snippets,
    balancing exploration, exploitation, and stability during training.
    """
    return dict(
        # Increased learning rates for faster convergence
        actor_lr=4e-4,
        critic_lr=4e-4,
        
        # Higher temperature learning rate for more aggressive exploration
        temp_lr=6e-5,
        
        # Lower end learning rate for fine-tuning
        lr_end=5e-5,
        
        # Moderate tau for soft updates
        tau=0.01,
        
        # Maintain policy delay as in originals for consistency
        policy_delay=2,
        
        # Initial temperature remains consistent with originals
        init_temperature=0.1,
        
        # Moderate weight decay to prevent overfitting without hindering learning
        weight_decay=0.005,
        
        # Larger batch size for more stable gradient estimates
        batch_size=256,
        
        # Increase updates per step for faster convergence
        updates_per_step=2,
    )
