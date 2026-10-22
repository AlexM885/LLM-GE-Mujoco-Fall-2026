
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
        hidden_dims_critic=[256, 256, 128],
        hidden_dims_actor=[256, 128, 64],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=False,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        # Using moderate learning rates for a balance between convergence speed and stability
        actor_lr=1e-4,  
        critic_lr=1e-4,
        temp_lr=5e-6,   # Lower temperature learning rate for more controlled exploration
        lr_end=5e-7,    # Ensuring learning rates decrease over time for fine-tuning
        tau=0.005,      # Smaller tau for more stable updates
        policy_delay=4, # Higher policy delay for more stable critic adaptation
        init_temperature=0.05, # Moderate initial temperature for balanced exploration and exploitation
        weight_decay=0.0005,  # Small weight decay to prevent overfitting without hindering learning
        batch_size=256,       # Moderate batch size for a balance between computational efficiency and gradient accuracy
        updates_per_step=2,   # More frequent updates for faster adaptation
    )
