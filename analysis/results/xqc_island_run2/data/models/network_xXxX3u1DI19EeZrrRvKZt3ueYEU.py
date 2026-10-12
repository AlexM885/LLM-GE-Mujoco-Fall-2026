
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
def get_combined_xqc_kwargs():
    """
    Returns a dictionary of combined hyperparameters for the XQC model.
    
    This function combines elements from two alternative code snippets to enhance the model's predictive metrics.
    """
    return dict(
        # Use a moderate learning rate for the actor and critic
        actor_lr=1e-3,
        critic_lr=1e-3,
        
        # Use a lower learning rate for the temperature
        temp_lr=1e-4,
        
        # Set the end learning rate
        lr_end=1e-5,
        
        # Set the target update rate
        tau=0.01,
        
        # Delay policy updates
        policy_delay=2,
        
        # Initialize the temperature
        init_temperature=0.1,
        
        # Apply weight decay
        weight_decay=0.01,
        
        # Use a moderate batch size
        batch_size=256,
        
        # Perform multiple updates per step
        updates_per_step=2,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=5e-3,
        critic_lr=5e-3,
        temp_lr=5e-5,
        lr_end=1e-5,
        tau=0.01,
        policy_delay=2,
        init_temperature=0.1,
        weight_decay=0.05,
        batch_size=1024,
        updates_per_step=2,
    )
