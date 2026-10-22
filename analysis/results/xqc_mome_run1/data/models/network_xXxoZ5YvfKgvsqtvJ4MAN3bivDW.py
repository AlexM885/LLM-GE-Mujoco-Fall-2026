
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
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
    """
    Returns a dictionary of network settings for the XQC agent, combining elements
    from different configurations to potentially enhance predictive metrics.
    """
    return dict(
        # Balanced critic dimensions for stability and performance
        hidden_dims_critic=[1536, 1536, 1024, 1024],
        
        # Actor dimensions adjusted for balance and reduced complexity
        hidden_dims_actor=[512, 512, 512, 512],
        
        # Number of critics remains constant as per the problem statement
        n_critics=2,
        
        # Enable batch normalization for stabilization
        use_batch_norm=True,
        
        # Weight normalization is used for better generalization
        use_weight_norm=True,
        
        # Layer normalization is disabled based on the original configurations
        use_layer_norm=False,
        
        # Pre-activation batch normalization is enabled for improved training dynamics
        pre_activation_bn=True,
        
        # Normalization of the last layer is enabled for better output scaling
        normalize_last_layer=True,
        
        # Skip connections are disabled to reduce complexity and promote deeper representations
        skip_connections=False,
    )
# --OPTION--
# ===============================
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    return dict(
        actor_lr=1e-4,
        critic_lr=1e-4,
        temp_lr=1e-4,
        lr_end=1e-5,
        tau=0.01,
        policy_delay=2,
        init_temperature=0.1,
        weight_decay=0.01,
        batch_size=128,
        updates_per_step=1,
    )
