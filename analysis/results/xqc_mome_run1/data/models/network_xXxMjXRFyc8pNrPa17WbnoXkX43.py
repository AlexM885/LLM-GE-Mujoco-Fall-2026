
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
        # Larger hidden dimensions for both critic and actor, similar to Code Snippet 2
        hidden_dims_critic=[3072, 3072, 1536, 1536],
        hidden_dims_actor=[1536, 1536, 768, 768],
        
        # Keep n_critics as required
        n_critics=2,
        
        # Enable batch normalization as it's widely used and effective
        use_batch_norm=True,
        
        # Weight normalization can help with generalization
        use_weight_norm=True,
        
        # Disable layer normalization to avoid over-normalization
        use_layer_norm=False,
        
        # Apply batch normalization before activation for stability
        pre_activation_bn=True,
        
        # Normalize the last layer for better output scaling
        normalize_last_layer=True,
        
        # Include skip connections to aid in training deeper networks
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
