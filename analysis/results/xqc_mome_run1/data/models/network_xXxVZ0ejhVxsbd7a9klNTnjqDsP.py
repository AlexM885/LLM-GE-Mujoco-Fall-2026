
# ========== Start: GeneCrossed

# ========== End:

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
    # Combining insights from both snippets for a potentially better performance
    return dict(
        hidden_dims_critic=[2048, 2048, 1024, 1024],  # Larger dimensions for more capacity
        hidden_dims_actor=[1024, 1024, 512, 512],     # Balanced approach for the actor
        n_critics=2,
        use_batch_norm=True,                          # Enable batch normalization for stability
        use_weight_norm=True,                         # Weight normalization as in the original
        use_layer_norm=True,                          # Layer normalization for generalization
        pre_activation_bn=False,                      # Pre-activation BN might not be necessary
        normalize_last_layer=False,                   # Normalizing the last layer may not be needed
        skip_connections=True,                        # Skip connections for deeper networks
    )
# --OPTION--
# ===============================
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    return dict(
        actor_lr=1e-4,
        critic_lr=1e-4,
        temp_lr=1e-6,
        lr_end=1e-6,
        tau=0.01,
        policy_delay=2,
        init_temperature=0.1,
        weight_decay=0.001,
        batch_size=512,
        updates_per_step=1,
    )
