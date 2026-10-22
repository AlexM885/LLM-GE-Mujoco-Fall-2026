
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
        hidden_dims_critic=[1024, 512, 256, 128],
        hidden_dims_actor=[512, 256, 128, 64],
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
# === Combined XQC Hyperparameters ===
# ===============================

def get_combined_xqc_kwargs():
    return dict(
        actor_lr=2e-5,  # Moderate learning rate
        critic_lr=2e-5,
        temp_lr=2e-5,
        lr_end=1e-6,  # Lower end learning rate for gradual decrease
        tau=0.0075,  # Balanced update rate for target networks
        policy_delay=2,  # Consistent with both snippets for stability
        init_temperature=0.075,  # Moderate initial temperature
        weight_decay=0.005,  # Balanced regularization
        batch_size=256,  # Compromise between gradient quality and efficiency
        updates_per_step=1,  # Standard update frequency
    )
