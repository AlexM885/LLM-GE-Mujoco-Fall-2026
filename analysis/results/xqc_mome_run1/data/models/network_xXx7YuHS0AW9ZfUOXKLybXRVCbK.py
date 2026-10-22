
# ========== Start: GeneCrossed
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
# ========== End:

# ========== Start: GeneCrossed

# ========== End:

# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
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
    # Combining hidden dimensions to leverage strengths from both configurations
    hidden_dims_critic = [2048, 1152, 1152, 1024]  
    hidden_dims_actor = [256, 448, 448, 256]
    
    # Selectively applying normalization techniques based on their benefits
    use_batch_norm = True
    use_weight_norm = True
    use_layer_norm = False  # Based on the second snippet's configuration
    
    # Applying pre-activation batch normalization as per the first snippet
    pre_activation_bn = True
    
    # Normalizing the last layer as suggested by both snippets
    normalize_last_layer = True
    
    # Excluding skip connections based on the second snippet's approach
    skip_connections = False
    
    return dict(
        hidden_dims_critic=hidden_dims_critic,
        hidden_dims_actor=hidden_dims_actor,
        n_critics=2,  # Maintaining the number of critics as specified
        use_batch_norm=use_batch_norm,
        use_weight_norm=use_weight_norm,
        use_layer_norm=use_layer_norm,
        pre_activation_bn=pre_activation_bn,
        normalize_last_layer=normalize_last_layer,
        skip_connections=skip_connections,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=3e-4,  
        critic_lr=1e-5,
        temp_lr=1e-6,   
        lr_end=1e-6,    
        tau=0.01,      
        policy_delay=6, 
        init_temperature=0.1, 
        weight_decay=0.001,  
        batch_size=512,       
        updates_per_step=3,   
    )
