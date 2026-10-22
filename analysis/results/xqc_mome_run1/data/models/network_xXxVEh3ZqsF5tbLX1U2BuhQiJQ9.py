
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
# === Enhanced XQC Hyperparameters ===
# ===============================

def get_enhanced_xqc_kwargs():
    return dict(
        # Using slightly higher learning rates for faster convergence
        actor_lr=2e-4,
        critic_lr=2e-4,
        temp_lr=1.5e-4,
        
        # Lower end learning rate to ensure stability towards the end of training
        lr_end=1e-5,
        
        # A moderate tau value for soft update
        tau=0.0075,
        
        # Policy delay remains consistent with both snippets
        policy_delay=2,
        
        # Initial temperature as per the original snippets
        init_temperature=0.1,
        
        # Weight decay set to a moderate level
        weight_decay=0.005,
        
        # Batch size is moderately increased for better gradient estimation
        batch_size=192,
        
        # Updates per step remain as in the originals
        updates_per_step=1,
    )
