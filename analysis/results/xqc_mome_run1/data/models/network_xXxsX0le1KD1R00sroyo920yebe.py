
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
        # Deeper and slightly adjusted dimensions inspired by the first snippet
        hidden_dims_critic=[1024, 896, 768],
        hidden_dims_actor=[512, 448, 384, 192],
        
        # Keep the number of critics constant as per the problem statement
        n_critics=2,
        
        # Use batch normalization as it's commonly effective
        use_batch_norm=True,
        
        # Weight normalization can help but may not always be necessary; let's include it
        use_weight_norm=True,
        
        # Layer normalization can be beneficial in certain contexts but let's exclude it for simplicity
        use_layer_norm=False,
        
        # Pre-activation batch normalization can affect performance; let's keep it simple
        pre_activation_bn=False,
        
        # Normalizing the last layer might not always be beneficial; let's avoid it
        normalize_last_layer=False,
        
        # Skip connections can help with deep networks but might not be necessary here
        skip_connections=False,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=1e-4,
        lr_end=1e-5,
        tau=0.005,
        policy_delay=2,
        init_temperature=0.1,
        weight_decay=0.001,
        batch_size=1024,
        updates_per_step=2,
    )
