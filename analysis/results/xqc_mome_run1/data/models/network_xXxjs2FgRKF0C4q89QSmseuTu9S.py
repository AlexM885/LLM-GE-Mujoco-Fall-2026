
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
def get_optimized_architecture_kwargs():
    """
    Returns optimized architecture kwargs for the XQC agent, balancing complexity and efficiency.
    
    :return: A dictionary containing optimized network settings.
    """
    return dict(
        hidden_dims_critic=[1024, 1024, 1024, 1024],  # Deeper critic for complex state understanding
        hidden_dims_actor=[256, 256, 256, 256],  # Simpler yet effective actor for action selection
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
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    return dict(
        actor_lr=1e-3,
        critic_lr=1e-3,
        temp_lr=5e-5,
        lr_end=5e-5,
        tau=0.005,
        policy_delay=2,
        init_temperature=0.05,
        weight_decay=0.01,
        batch_size=512,
        updates_per_step=2,
    )
