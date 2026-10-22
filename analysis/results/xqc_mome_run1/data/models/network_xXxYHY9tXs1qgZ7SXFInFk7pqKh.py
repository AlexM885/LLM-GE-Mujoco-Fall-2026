
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:

# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:

# ========== Start: GeneCrossed

# ========== End:
# ========== End:
# ========== End:

# ========== Start: GeneCrossed
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
    return dict(
        hidden_dims_critic=[1536, 768, 384, 192],
        hidden_dims_actor=[768, 384, 192, 96],
        n_critics=2,
        use_batch_norm=False,
        use_weight_norm=True,
        use_layer_norm=True,
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=False,
    )
# --OPTION--
# ===============================
# === Combined XQC Hyperparameter Gene ===
# ===============================

def get_combined_xqc_kwargs():
    return dict(
        # Balanced learning rates for stability and convergence speed
        actor_lr=2e-4,
        critic_lr=2e-4,
        temp_lr=3e-5,
        
        # Gradual decrease to a moderate end learning rate
        lr_end=5e-6,
        
        # Moderate tau for balanced stability and adaptability
        tau=0.0075,
        
        # Policy delay remains consistent with originals for periodic updates
        policy_delay=2,
        
        # Initial temperature as in originals for exploration
        init_temperature=0.1,
        
        # Weight decay at a moderate level for regularization
        weight_decay=0.00075,
        
        # Batch size that balances memory usage and gradient stability
        batch_size=512,
        
        # Updates per step that allow for decent progress without overshooting
        updates_per_step=2.5,
    )
