
# ========== Start: GeneCrossed

# ========== End:

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
        hidden_dims_critic=[2560, 2304, 2048, 1024], 
        hidden_dims_actor=[1280, 1152, 1024, 768],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=True,
    )
# --OPTION--
def get_combined_xqc_kwargs():
    return dict(
        # Use a moderate learning rate for stability and decent convergence speed
        actor_lr=1.5e-4,
        critic_lr=1.5e-4,
        temp_lr=1e-4,
        
        # Keep the end learning rate low for fine-tuning
        lr_end=1e-5,
        
        # Choose a tau value that offers a balance between stability and adaptability
        tau=0.008,
        
        # Maintain a policy delay to ensure the critic has a chance to learn before updating the actor
        policy_delay=2,
        
        # Start with a moderate initial temperature for a balanced exploration-exploitation strategy
        init_temperature=0.0875,
        
        # Apply a moderate level of regularization
        weight_decay=0.0075,
        
        # Use a moderately sized batch for a balance between speed and gradient noise
        batch_size=192,
        
        # Update the model twice per step for quicker refinement without excessive computation
        updates_per_step=1.5
    )
