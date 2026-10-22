
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
        # Adopt a deeper critic network for more complex state-action value representation
        hidden_dims_critic=[512, 512, 512, 512, 256],
        
        # Use a wider actor network for generating diverse policies
        hidden_dims_actor=[512, 512, 256, 256],
        
        # Maintain the number of critics as specified
        n_critics=2,
        
        # Combine normalization techniques for stable training
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=True,
        
        # Other parameters remain consistent with the improvements
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=False,
    )
# --OPTION--
def get_combined_xqc_kwargs():
    return dict(
        actor_lr=1e-3,  # Moderate learning rate
        critic_lr=1e-3,  # Moderate learning rate
        temp_lr=1e-3,    # Moderate learning rate
        lr_end=1e-5,     # Low final learning rate for fine-tuning
        tau=0.005,       # Smaller tau for less frequent target updates
        policy_delay=2,  # Delayed actor updates for stability
        init_temperature=0.1,  # Balanced initial temperature
        weight_decay=0.001,   # Moderate regularization
        batch_size=1024,      # Moderate batch size for efficiency and stability
        updates_per_step=3,   # Balanced number of updates per step
    )
