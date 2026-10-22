
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
    return dict(
        # Deeper and wider critic network inspired by the first snippet
        hidden_dims_critic=[1024, 1024, 1024, 1024],
        
        # Actor network dimensions inspired by the second snippet but adjusted
        hidden_dims_actor=[1536, 1536, 768, 768],
        
        # Keep the number of critics constant as per the note
        n_critics=2,
        
        # Enable batch normalization as it's commonly beneficial
        use_batch_norm=True,
        
        # Disable weight normalization based on the second snippet
        use_weight_norm=False,
        
        # Enable layer normalization as suggested by the second snippet
        use_layer_norm=True,
        
        # Use pre-activation batch normalization for potential improvement
        pre_activation_bn=False,
        
        # Normalize the last layer for consistency with common practices
        normalize_last_layer=True,
        
        # Include skip connections for potential improvement in deep networks
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
