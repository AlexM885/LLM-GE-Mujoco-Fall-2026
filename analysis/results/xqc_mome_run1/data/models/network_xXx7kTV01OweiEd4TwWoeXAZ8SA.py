
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
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
        hidden_dims_critic=[3072, 3072, 1536, 1536],
        hidden_dims_actor=[1536, 1536, 768, 768],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=False,
    )
# --OPTION--
def get_combined_xqc_kwargs():
    return dict(
        # Moderate learning rates for balance
        actor_lr=1e-4,
        critic_lr=1e-4,
        temp_lr=5e-6,  # Slightly lower for controlled exploration
        lr_end=5e-7,   # Decrease learning rates over time
        tau=0.005,     # Smaller for more stable updates
        policy_delay=4, # Higher for more stable critic adaptation
        init_temperature=0.05, # Moderate for balanced exploration and exploitation
        weight_decay=0.0005,  # Small to prevent overfitting
        batch_size=256,       # Moderate for balance between efficiency and accuracy
        updates_per_step=2,   # More frequent for faster adaptation
    )
