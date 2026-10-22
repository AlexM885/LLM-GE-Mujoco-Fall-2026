
# ========== Start: GeneCrossed
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
# ========== End:

# ========== Start: GeneCrossed

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
    return dict(
        # Using larger dimensions for the critic as in the second snippet
        # but keeping the actor dimensions moderate for balance
        hidden_dims_critic=[512, 512, 256],  
        hidden_dims_actor=[256, 256, 128],
        
        # Keeping the number of critics constant as per the requirement
        n_critics=2,
        
        # Enabling batch and weight normalization for stability
        # but disabling layer normalization to avoid over-normalization
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        
        # Disabling pre-activation BN and last layer normalization
        # to maintain simplicity and avoid potential overfitting
        pre_activation_bn=False,
        normalize_last_layer=False,
        
        # Disabling skip connections to keep the architecture straightforward
        skip_connections=False,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=5e-5,  
        critic_lr=5e-5,
        temp_lr=1e-6,   
        lr_end=1e-7,    
        tau=0.01,      
        policy_delay=6, 
        init_temperature=0.1, 
        weight_decay=0.001,  
        batch_size=128,       
        updates_per_step=1,   
    )
