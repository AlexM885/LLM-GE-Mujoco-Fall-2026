
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
        hidden_dims_critic=[1400, 720, 720, 720],
        hidden_dims_actor=[360, 360, 360, 360],
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
    Returns a dictionary of combined hyperparameters for the XQC model.
    
    This implementation combines elements from different approaches to potentially
    enhance the model's predictive metrics.
    """
    return dict(
        # Moderate learning rates for stability and reasonable convergence speed
        actor_lr=5e-4,
        critic_lr=5e-4,
        temp_lr=5e-4,
        
        # Moderate end learning rate for fine-tuning towards the end of training
        lr_end=5e-5,
        
        # Moderate tau value for soft update of target networks
        tau=0.005,
        
        # Moderate policy delay to ensure sufficient exploration before updating the policy
        policy_delay=5,
        
        # Moderate initial temperature for entropy regularization
        init_temperature=1.0,
        
        # Moderate weight decay to prevent overfitting
        weight_decay=0.001,
        
        # Moderate batch size for mini-batch gradient descent
        batch_size=512,
        
        # Moderate number of updates per step for the actor and critic
        updates_per_step=2,
    )
