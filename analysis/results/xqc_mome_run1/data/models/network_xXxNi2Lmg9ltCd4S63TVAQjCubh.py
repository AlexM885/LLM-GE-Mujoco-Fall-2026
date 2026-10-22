
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
# ========== End:
# ========== End:

# ========== Start: GeneCrossed

# ========== End:

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
        # Increase capacity for the critic network significantly
        hidden_dims_critic=[2048, 2048, 2048, 2048],
        
        # Increase capacity for the actor network moderately
        hidden_dims_actor=[1024, 512, 256, 128],
        
        # Number of critics remains constant
        n_critics=2,
        
        # Enable batch normalization for stability
        use_batch_norm=True,
        
        # Enable weight normalization for further stability
        use_weight_norm=True,
        
        # Disable layer normalization to avoid over-normalization
        use_layer_norm=False,
        
        # Apply batch normalization before activation for better flow
        pre_activation_bn=True,
        
        # Normalize the last layer for output stability
        normalize_last_layer=True,
        
        # No skip connections to maintain simplicity
        skip_connections=False,
    )
# --OPTION--
# ===============================
# === Enhanced XQC Hyperparameters ===
# ===============================

def get_enhanced_xqc_kwargs():
    return dict(
        # Use a moderate learning rate, slightly lower than the highest but higher than the lowest
        actor_lr=2e-4,
        critic_lr=2e-4,
        
        # Choose a temperature learning rate that balances exploration and exploitation
        temp_lr=1e-5,
        
        # Set a reasonable end learning rate
        lr_end=1e-6,
        
        # Select a tau value that allows for stable updates without being too slow or too fast
        tau=0.0075,
        
        # Policy delay remains consistent with both snippets
        policy_delay=2,
        
        # Initial temperature as per the original snippets
        init_temperature=0.1,
        
        # Weight decay to prevent overfitting
        weight_decay=0.001,
        
        # Batch size is set to a moderate level, considering computational resources and stability
        batch_size=384,
        
        # Updates per step are balanced to ensure frequent enough updates without overwhelming the system
        updates_per_step=1.5,  # Note: This may need adjustment based on the actual implementation details
    )
