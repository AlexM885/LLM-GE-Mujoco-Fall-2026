
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:

# ========== Start: GeneCrossed

# ========== End:
# ========== End:

# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

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
    Returns a dictionary of network settings for the XQC agent, combining insights from both code snippets.
    
    :return: A dictionary containing architecture settings.
    """
    return dict(
        # Adjusting critic dimensions to explore a more complex critic network
        hidden_dims_critic=[1280, 896, 896, 896],
        
        # Adjusting actor dimensions to maintain a simpler yet effective actor network
        hidden_dims_actor=[320, 320, 320, 320],
        
        # Keeping the number of critics as specified
        n_critics=2,
        
        # Enabling batch normalization for stability
        use_batch_norm=True,
        
        # Disabling weight normalization to reduce computational overhead
        use_weight_norm=False,
        
        # Disabling layer normalization to simplify the architecture
        use_layer_norm=False,
        
        # Applying normalization before activation for potential improvement in training dynamics
        pre_activation_bn=True,
        
        # Normalizing the last layer to potentially aid in generalization
        normalize_last_layer=True,
        
        # Including skip connections to facilitate gradient flow through deep networks
        skip_connections=True,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=1e-3,
        lr_end=1e-5,
        tau=0.005,
        policy_delay=1,
        init_temperature=0.5,
        weight_decay=0.0001,
        batch_size=512,
        updates_per_step=2,
    )
