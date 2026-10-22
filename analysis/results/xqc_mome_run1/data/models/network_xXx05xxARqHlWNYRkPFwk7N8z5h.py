
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
        # Combining the depth and width of critic networks for better representation
        hidden_dims_critic=[2048, 1152, 1024, 1024],
        
        # Actor network with a balance between depth and width for effective policy learning
        hidden_dims_actor=[1024, 576, 512, 256],
        
        # Keeping the number of critics as specified
        n_critics=2,
        
        # Applying batch normalization and weight normalization for stability
        use_batch_norm=True,
        use_weight_norm=True,
        
        # Using layer normalization to further stabilize the activations
        use_layer_norm=True,
        
        # Pre-activation batch normalization for improved gradient flow
        pre_activation_bn=True,
        
        # Normalizing the last layer to ensure consistent output scales
        normalize_last_layer=True,
        
        # Including skip connections to facilitate deeper network training
        skip_connections=True,
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
        init_temperature=0.05,
        weight_decay=0.001,
        batch_size=1024,
        updates_per_step=2,
    )
