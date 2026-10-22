
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

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
        hidden_dims_critic=[3072, 3072, 3072, 3072],
        hidden_dims_actor=[1280, 1280, 1280, 1280],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=True,
    )
# --OPTION--
# ===============================
# === Combined XQC Hyperparameter Gene ===
# ===============================

def get_combined_xqc_kwargs():
    return dict(
        actor_lr=3e-5,  # Lower learning rate from Snippet 2
        critic_lr=3e-5,  # Lower learning rate from Snippet 2
        temp_lr=3e-5,  # Lower learning rate from Snippet 2
        lr_end=1e-5,  # Lower lr_end from Snippet 1
        tau=0.005,  # Lower tau from Snippet 2
        policy_delay=2,  # Same policy delay from both snippets
        init_temperature=0.05,  # Lower initial temperature from Snippet 2
        weight_decay=0.001,  # Lower weight decay from Snippet 2
        batch_size=512,  # Larger batch size from Snippet 2
        updates_per_step=2,  # More updates per step from Snippet 2
    )
