
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
        # Combining deeper critic network with wider actor network
        hidden_dims_critic=[1024, 1024, 1024, 512],
        hidden_dims_actor=[1024, 512, 256],
        
        # Keeping the number of critics as specified
        n_critics=2,
        
        # Enabling batch normalization for stability
        use_batch_norm=True,
        
        # Weight normalization can also aid in stabilizing weights during training
        use_weight_norm=True,
        
        # Layer normalization is not preferred in this case based on the notes
        use_layer_norm=False,
        
        # Pre-activation batch normalization can help in certain architectures
        pre_activation_bn=True,
        
        # Normalizing the last layer may help in some cases but depends on the task
        normalize_last_layer=False,
        
        # Skip connections can help in deep networks but are not necessary here
        skip_connections=False,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=1e-4,
        critic_lr=1e-4,
        temp_lr=1e-4,
        lr_end=1e-6,
        tau=0.01,
        policy_delay=2,
        init_temperature=0.01,
        weight_decay=0.0001,
        batch_size=1024,
        updates_per_step=2,
    )
