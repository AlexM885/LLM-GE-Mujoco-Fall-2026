
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
# ========== End:

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
    return dict(
        hidden_dims_critic=[1024, 1024, 512],
        hidden_dims_actor=[512, 512, 256],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=False,
        use_layer_norm=True,
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=True,
    )
# --OPTION--
def get_combined_xqc_kwargs():
    return dict(
        # Use a moderate learning rate for stability and decent convergence speed
        actor_lr=2e-4,
        critic_lr=2e-4,
        temp_lr=3e-5,
        
        # Lower end learning rate for fine-tuning towards the end of training
        lr_end=5e-6,
        
        # A moderate tau value for updating target networks
        tau=0.0075,
        
        # Maintain a policy delay to ensure critic updates are more frequent
        policy_delay=2,
        
        # Start with a moderate initial temperature for a balance between exploration and exploitation
        init_temperature=0.1,
        
        # Apply a moderate weight decay to prevent overfitting without hindering learning
        weight_decay=0.005,
        
        # Choose a batch size that balances memory usage and gradient stability
        batch_size=192,
        
        # Update the model a couple of times per step for efficient learning
        updates_per_step=1.5,
    )
