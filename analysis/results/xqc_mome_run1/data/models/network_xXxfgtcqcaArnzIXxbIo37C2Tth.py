
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
        hidden_dims_critic=[1024, 1024, 512],
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
def get_combined_xqc_kwargs():
    """
    Combines hyperparameters from two different setups to potentially improve model performance or efficiency.
    
    Returns:
        dict: A dictionary containing the combined hyperparameters.
    """
    return dict(
        # Higher learning rates for faster initial convergence
        actor_lr=1e-3,
        critic_lr=1e-3,
        
        # Same temperature learning rate as both snippets
        temp_lr=1e-4,
        
        # Lower end learning rate for fine-tuning
        lr_end=1e-5,
        
        # Small tau for soft updates
        tau=0.01,
        
        # Policy delay remains consistent with both snippets
        policy_delay=2,
        
        # Initial temperature consistent across snippets
        init_temperature=0.1,
        
        # Weight decay for regularization
        weight_decay=0.01,
        
        # Smaller batch size for efficiency and stability
        batch_size=128,
        
        # Fewer updates per step for computational efficiency
        updates_per_step=1,
    )

# Example usage
combined_kwargs = get_combined_xqc_kwargs()
print(combined_kwargs)
