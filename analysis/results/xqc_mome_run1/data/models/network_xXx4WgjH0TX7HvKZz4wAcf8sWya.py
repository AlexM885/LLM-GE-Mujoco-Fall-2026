
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:

# ========== Start: GeneCrossed

# ========== End:
# ========== End:

# ========== Start: GeneCrossed
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
    # Balancing the critic dimensions for potential improvement
    return dict(
        hidden_dims_critic=[1536, 1536, 1024, 1024],  # A balanced approach
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
def get_xqc_kwargs():
    """
    Returns a dictionary of hyperparameters for the XQC model, combining elements from different approaches.
    
    The idea is to balance exploration and exploitation by selecting moderate values for key hyperparameters.
    """
    return dict(
        # Moderate learning rates for actor and critic
        actor_lr=5e-4,
        critic_lr=5e-4,
        
        # Adjusted temperature learning rate for better exploration-exploitation trade-off
        temp_lr=5e-6,
        
        # Learning rate end point
        lr_end=5e-6,
        
        # Tau value for soft update
        tau=0.01,
        
        # Policy delay to ensure sufficient exploration before updating the target network
        policy_delay=2,
        
        # Initial temperature for entropy regularization
        init_temperature=0.1,
        
        # Weight decay to prevent overfitting
        weight_decay=0.005,
        
        # Batch size for training
        batch_size=512,
        
        # Updates per step to balance exploration and exploitation
        updates_per_step=2,
    )
