
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
# ========== End:

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
def get_architecture_kwargs():
    """
    Returns a dictionary of network settings for the XQC agent, combining
    enhancements from both provided snippets for potentially improved predictive metrics.
    """
    return dict(
        # Deeper and wider critic network for better feature extraction
        hidden_dims_critic=[2048, 1536, 1024, 768],
        
        # Adjusted actor network dimensions for balanced learning
        hidden_dims_actor=[512, 512, 384, 384, 256],
        
        # Number of critics remains constant as per the problem statement
        n_critics=2,
        
        # Enable batch normalization for stability
        use_batch_norm=True,
        
        # Weight normalization enabled for improved generalization
        use_weight_norm=True,
        
        # Layer normalization disabled as per the second snippet's approach
        use_layer_norm=False,
        
        # Pre-activation batch normalization enabled
        pre_activation_bn=True,
        
        # Normalize the last layer for better output distribution
        normalize_last_layer=True,
        
        # Skip connections enabled to facilitate deeper network training
        skip_connections=True,
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
        lr_end=1e-6,
        tau=0.002,
        policy_delay=1,
        init_temperature=0.1,
        weight_decay=0.01,
        batch_size=1024,
        updates_per_step=3,
    )
