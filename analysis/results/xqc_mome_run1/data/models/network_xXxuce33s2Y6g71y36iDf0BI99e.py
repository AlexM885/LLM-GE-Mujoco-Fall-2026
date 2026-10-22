
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
# ========== End:
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
    # Combining beneficial aspects from both snippets
    return dict(
        hidden_dims_critic=[1408, 1408, 1152, 896],  # Balanced dimensions
        hidden_dims_actor=[768, 768, 576, 448],     # Reduced but still capable
        n_critics=2,
        use_batch_norm=True,                        # For stabilization
        use_weight_norm=True,                       # Additional stabilization
        use_layer_norm=False,                       # Based on specific needs
        pre_activation_bn=True,                     # Beneficial for deep networks
        normalize_last_layer=True,                  # Can help with output scaling
        skip_connections=True,                      # Facilitates gradient flow
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
        lr_end=5e-6,
        tau=0.002,
        policy_delay=1,
        init_temperature=0.05,
        weight_decay=0.001,
        batch_size=512,
        updates_per_step=1,
    )
