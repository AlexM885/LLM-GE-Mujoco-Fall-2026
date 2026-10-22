
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
        hidden_dims_critic=[512, 512, 256],
        hidden_dims_actor=[256, 256, 128],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=False,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=False,
        skip_connections=False,
    )
# --OPTION--
def get_xqc_kwargs():
    return dict(
        actor_lr=3e-5,  # Lower learning rate for more precise updates
        critic_lr=3e-5,  # Consistent with actor learning rate
        temp_lr=3e-5,   # Same as actor and critic for balanced update
        lr_end=5e-6,    # Lower end learning rate for fine-tuning
        tau=0.005,      # Smaller tau for softer updates
        policy_delay=2, # Consistent policy delay
        init_temperature=0.05, # Lower initial temperature for more cautious start
        weight_decay=0.001,    # Lower weight decay to prevent over-regularization
        batch_size=512,        # Moderate batch size for a balance between speed and accuracy
        updates_per_step=2,    # Fewer updates per step to avoid overfitting
    )
