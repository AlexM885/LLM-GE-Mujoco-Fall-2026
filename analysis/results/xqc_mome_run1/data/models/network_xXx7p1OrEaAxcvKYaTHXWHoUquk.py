
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:

# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:

# ========== Start: GeneCrossed

# ========== End:
# ========== End:
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
        # Increase critic capacity for better value estimation
        hidden_dims_critic=[1536, 1536, 768, 768],
        
        # Increase actor capacity for more complex policies
        hidden_dims_actor=[896, 448, 224, 112],
        
        # Keep the number of critics constant
        n_critics=2,
        
        # Enable batch normalization for improved stability
        use_batch_norm=True,
        
        # Enable weight normalization for better generalization
        use_weight_norm=True,
        
        # Disable layer normalization to reduce complexity
        use_layer_norm=False,
        
        # Apply batch normalization before activation for better conditioning
        pre_activation_bn=True,
        
        # Normalize the last layer for more stable outputs
        normalize_last_layer=True,
        
        # Include skip connections to help with deeper networks
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
        lr_end=1e-5,
        tau=0.01,
        policy_delay=2,
        init_temperature=0.1,
        weight_decay=0.01,
        batch_size=128,
        updates_per_step=1,
    )
