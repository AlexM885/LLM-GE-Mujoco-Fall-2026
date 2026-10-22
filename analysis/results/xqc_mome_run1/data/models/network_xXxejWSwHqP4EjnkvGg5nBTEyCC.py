
# ========== Start: GeneCrossed

# ========== End:

# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

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
    return dict(
        # Slightly deeper and wider critic network for improved value estimation
        hidden_dims_critic=[768, 768, 384, 192],
        
        # Slightly wider actor network for richer policy representation
        hidden_dims_actor=[384, 384, 192, 96],
        
        # Maintain the ensemble of critics as specified
        n_critics=2,
        
        # Apply batch normalization for stabilization
        use_batch_norm=True,
        
        # Weight normalization can help in certain scenarios but may not always be beneficial
        use_weight_norm=False,
        
        # Layer normalization can be useful but might not be necessary with batch norm
        use_layer_norm=False,
        
        # Pre-activation batch normalization can affect the learning dynamics
        pre_activation_bn=False,
        
        # Normalizing the last layer might help with stability but depends on the task
        normalize_last_layer=False,
        
        # Skip connections can aid in training deeper networks
        skip_connections=True,
    )
# --OPTION--
# ===============================
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    return dict(
        actor_lr=3e-5,
        critic_lr=3e-5,
        temp_lr=3e-5,
        lr_end=5e-6,
        tau=0.005,
        policy_delay=2,
        init_temperature=0.05,
        weight_decay=0.001,
        batch_size=512,
        updates_per_step=2,
    )
