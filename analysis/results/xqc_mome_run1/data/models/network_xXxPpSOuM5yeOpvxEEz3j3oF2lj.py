
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:

# ========== Start: GeneCrossed

# ========== End:
# ========== End:

# ========== Start: GeneCrossed

# ========== End:

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
        hidden_dims_critic=[2048, 2048, 1024, 1024],  # A balance between depth and width
        hidden_dims_actor=[1024, 1024, 512, 512],     # Similarly balanced for the actor
        n_critics=2,
        use_batch_norm=True,                          # Batch normalization for stability
        use_weight_norm=True,                         # Weight normalization for better generalization
        use_layer_norm=False,                         # No layer normalization as per original notes
        pre_activation_bn=True,                       # Applying BN before activation for non-linearity
        normalize_last_layer=True,                    # Normalizing the last layer for better output
        skip_connections=False,                       # No skip connections as per the originals
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
        batch_size=256,
        updates_per_step=2,
    )
