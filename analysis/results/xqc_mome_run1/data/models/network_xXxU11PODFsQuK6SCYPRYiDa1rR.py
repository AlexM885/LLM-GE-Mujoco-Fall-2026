
# ========== Start: GeneCrossed

# ========== End:

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
        hidden_dims_critic=[896, 640, 448],  # Slightly reduced from [1024, 768, 512]
        hidden_dims_actor=[448, 320, 224, 128],  # Adjusted from [512, 384, 256, 128]
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=True,
    )
# --OPTION--
# ===============================
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    """
    Returns a dictionary of hyperparameters for the XQC model.
    
    :return: A dictionary containing hyperparameters for the XQC model.
    """
    return dict(
        # Learning rate for the actor network
        actor_lr=1e-4,
        
        # Learning rate for the critic network
        critic_lr=1e-4,
        
        # Learning rate for the temperature parameter
        temp_lr=1e-4,
        
        # Final learning rate after decay
        lr_end=1e-5,
        
        # Target network update rate
        tau=0.01,
        
        # Delay between policy updates
        policy_delay=2,
        
        # Initial temperature value
        init_temperature=0.1,
        
        # Weight decay coefficient
        weight_decay=0.01,
        
        # Batch size for training
        batch_size=128,
        
        # Number of updates per step
        updates_per_step=1,
    )
