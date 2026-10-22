
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
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
        hidden_dims_critic=[2048, 2048, 1024, 1024],
        hidden_dims_actor=[1024, 1024, 512, 512],
        n_critics=2,
        use_batch_norm=False,
        use_weight_norm=True,
        use_layer_norm=True,
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=True,
    )
# --OPTION--
# ===============================
# === Enhanced XQC Hyperparameter Gene ===
# ===============================

def get_enhanced_xqc_kwargs():
    """
    Returns a dictionary of hyperparameters for an enhanced XQC model.
    
    This function combines elements from different approaches to potentially improve the model's predictive metrics.
    """
    return dict(
        actor_lr=1e-4,
        critic_lr=1e-4,
        temp_lr=1e-6,  # Lower temperature learning rate for stable convergence
        lr_end=1e-6,   # Lower final learning rate for finer tuning
        tau=0.01,
        policy_delay=2,
        init_temperature=0.1,
        weight_decay=0.001,  # Smaller weight decay for less regularization
        batch_size=512,      # Larger batch size for faster training and potential better generalization
        updates_per_step=1,
    )

# Example usage
enhanced_kwargs = get_enhanced_xqc_kwargs()
print(enhanced_kwargs)
