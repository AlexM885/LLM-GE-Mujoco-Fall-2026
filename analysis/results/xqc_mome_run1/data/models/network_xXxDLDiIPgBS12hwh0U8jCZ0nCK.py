
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
        hidden_dims_critic=[1024, 512, 256],  # Reduced depth and width for efficiency
        hidden_dims_actor=[512, 256, 128],  # Adjusted to balance with critic complexity
        n_critics=2,  # Must be 2 for XQC
        use_batch_norm=True,  # Helpful for stabilizing training
        use_weight_norm=False,  # Weight norm can sometimes hinder performance
        use_layer_norm=False,  # Layer norm is less commonly used in these architectures
        pre_activation_bn=True,  # Applying BN before activation can help
        normalize_last_layer=False,  # Often not necessary and can limit expressiveness
        skip_connections=False,  # Skip connections can aid in deeper networks but may not be needed here
    )
# --OPTION--
# ===============================
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    return dict(
        actor_lr=1e-4,
        critic_lr=1e-4,
        temp_lr=1e-4,
        lr_end=1e-6,
        tau=0.002,
        policy_delay=1,
        init_temperature=0.1,
        weight_decay=0.01,
        batch_size=1024,
        updates_per_step=3,
    )
