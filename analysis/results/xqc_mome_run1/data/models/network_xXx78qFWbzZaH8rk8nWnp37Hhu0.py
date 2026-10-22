
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
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
# -- NOTE --
# Note: XQC (ICLR 2026) is an off-policy actor-critic in JAX: a tanh-Gaussian actor
# and an ensemble of categorical critics, both MLPs with batch norm and weight norm.
# get_architecture_kwargs() must return a dict of network settings for the XQC agent.
# Only plain Python values (ints, floats, bools, strings, lists of ints) are allowed.
# Prefer mutating hidden_dims_critic, hidden_dims_actor and the
# normalization flags. n_critics must stay 2.# -- NOTE --
def get_architecture_kwargs():
    # Combining insights from both snippets to potentially enhance predictive metrics
    return dict(
        # Adjusting critic dimensions to explore a balanced approach
        hidden_dims_critic=[1152, 768, 768, 768],  
        # Adjusting actor dimensions based on observed patterns
        hidden_dims_actor=[416, 416, 416, 416],
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
    Returns a dictionary of combined hyperparameters from two XQC models.
    
    The idea is to leverage the strengths of each set of hyperparameters to potentially improve the model's performance.
    """
    return dict(
        # Using slightly lower learning rates for more stable convergence 
        actor_lr=3e-5,
        critic_lr=3e-5,
        temp_lr=3e-5,
        
        # A moderate end learning rate
        lr_end=2e-5,
        
        # Tau value that balances exploration and exploitation
        tau=0.0075,
        
        # Policy delay that allows for sufficient exploration before updating the policy
        policy_delay=3,
        
        # Initial temperature that encourages exploration without being too high or too low
        init_temperature=0.075,
        
        # Weight decay to prevent overfitting but not so high as to hinder learning
        weight_decay=0.005,
        
        # Batch size remains consistent with both snippets for simplicity
        batch_size=256,
        
        # Updates per step is set to ensure there's a good balance between exploration and policy updates
        updates_per_step=1.5,
    )
