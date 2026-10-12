
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
# ===============================
# === Enhanced XQC Hyperparameter Gene ===
# ===============================

def get_enhanced_xqc_kwargs():
    """
    Returns a dictionary of combined hyperparameters for enhanced predictive metrics.
    """
    return dict(
        # Moderate learning rates
        actor_lr=5e-4,
        critic_lr=5e-4,
        temp_lr=5e-4,
        lr_end=1e-5,
        
        # Tau remains consistent across both snippets
        tau=0.01,
        
        # Adjust policy delay based on specific needs
        policy_delay=1,  # Default from the first snippet
        
        # Initial temperature as per the original snippets
        init_temperature=0.1,
        
        # Incorporate weight decay to prevent overfitting
        weight_decay=0.005,  # A moderate value between the two snippets
        
        # Batch size that balances training time and performance
        batch_size=256,  # A compromise between 512 and 128
        
        # Updates per step
        updates_per_step=2,  # Between the values of the two snippets
    )

# Example usage
enhanced_kwargs = get_enhanced_xqc_kwargs()
print(enhanced_kwargs)
# --OPTION--
# ===============================
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    return dict(
        actor_lr=1e-3,
        critic_lr=1e-3,
        temp_lr=1e-3,
        lr_end=1e-4,
        tau=0.01,
        policy_delay=1,
        init_temperature=0.1,
        weight_decay=0.0,
        batch_size=512,
        updates_per_step=3,
    )
