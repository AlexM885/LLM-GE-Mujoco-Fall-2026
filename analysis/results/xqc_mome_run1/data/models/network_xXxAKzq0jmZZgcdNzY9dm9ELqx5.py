
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
        hidden_dims_critic=[1024, 512, 256],  # Reduced depth and width for simpler problems
        hidden_dims_actor=[512, 256, 128],  # Adjusted to balance with critic complexity
        n_critics=2,
        use_batch_norm=True,  # Batch normalization helps with stability
        use_weight_norm=False,  # Weight normalization can sometimes hinder performance
        use_layer_norm=False,  # Layer normalization is less commonly used in this context
        pre_activation_bn=True,  # Applying BN before activation can help learning
        normalize_last_layer=False,  # Normalizing the last layer can affect output scale
        skip_connections=False,  # Skip connections are useful in deeper networks but not necessary here
    )
# --OPTION--
# ===============================
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    return dict(
        actor_lr=3e-5,
        critic_lr=3e-5,
        temp_lr=3e-5,
        lr_end=5e-6,
        tau=0.005,
        policy_delay=2,
        init_temperature=0.05,
        weight_decay=0.001,
        batch_size=512,
        updates_per_step=2,
    )
