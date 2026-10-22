
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
# -- NOTE --
# Note: XQC (ICLR 2026) is an off-policy actor-critic in JAX: a tanh-Gaussian actor
# and an ensemble of categorical critics, both MLPs with batch norm and weight norm.
# get_architecture_kwargs() must return a dict of network settings for the XQC agent.
# Only plain Python values (ints, floats, bools, strings, lists of ints) are allowed.
# Prefer mutating hidden_dims_critic, hidden_dims_actor and the
# normalization flags. n_critics must stay 2.# -- NOTE --
def get_architecture_kwargs():
    # Combining insights from both snippets to potentially enhance predictive metrics
    return dict(
        # Adjusting critic dimensions to explore a balanced approach
        hidden_dims_critic=[1152, 768, 768, 768],  
        # Adjusting actor dimensions based on observed patterns
        hidden_dims_actor=[416, 416, 416, 416],
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
# === Combined XQC Hyperparameter Gene ===
# ===============================

def get_combined_xqc_kwargs():
    return dict(
        actor_lr=2e-4,  # Moderate learning rate
        critic_lr=2e-4,
        temp_lr=2e-4,
        lr_end=1e-5,  # Smaller end learning rate for fine-tuning
        tau=0.0075,  # Moderate soft update coefficient
        policy_delay=2,
        init_temperature=0.05,
        weight_decay=0.00075,  # Moderate regularization
        batch_size=512,  # Increased batch size for faster training
        updates_per_step=2,  # More updates per step for potentially faster convergence
    )
