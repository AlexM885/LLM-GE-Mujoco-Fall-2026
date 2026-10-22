
# ========== Start: GeneCrossed

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
        hidden_dims_critic=[2048, 1024, 512], 
        hidden_dims_actor=[1536, 768, 384],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=True,
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=True,
    )
# --OPTION--
# ===============================
# === Enhanced XQC Hyperparameters ===
# ===============================

def get_enhanced_xqc_kwargs():
    return dict(
        actor_lr=2e-5,  # A moderate learning rate
        critic_lr=2e-5,
        temp_lr=2e-5,
        lr_end=1e-6,  # Lower end learning rate for decay
        tau=0.0075,  # A compromise between exploration and stability
        policy_delay=2,
        init_temperature=0.075,  # Balanced initial exploration
        weight_decay=0.005,  # Moderate regularization
        batch_size=256,  # A balance between efficiency and stability
        updates_per_step=1,  # Focused updates
    )
