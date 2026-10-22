
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
    return dict(
        hidden_dims_critic=[1024, 1024, 1024, 1024],
        hidden_dims_actor=[512, 512, 512, 512],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=False,
    )
# --OPTION--
# ===============================
# === Enhanced XQC Hyperparameters ===
# ===============================

def get_enhanced_xqc_kwargs():
    return dict(
        actor_lr=5e-4,  # Increased learning rate for faster adaptation
        critic_lr=5e-4,
        temp_lr=5e-4,
        lr_end=5e-5,  # Adjusted end learning rate for smoother convergence
        tau=0.01,  # Slightly increased tau for more stable updates
        policy_delay=1,  # Reduced policy delay for more frequent updates
        init_temperature=0.1,  # Higher initial temperature for increased exploration
        weight_decay=0.001,  # Reduced weight decay to allow for more aggressive learning
        batch_size=256,  # Increased batch size for improved stability and accuracy
        updates_per_step=2,  # More frequent updates for faster learning
    )
