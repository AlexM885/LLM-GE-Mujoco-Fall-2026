
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
        hidden_dims_critic=[1280, 1152, 1024, 896],
        hidden_dims_actor=[640, 576, 512, 448],
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
        # Moderate learning rates for stability and speed
        actor_lr=2e-4,
        critic_lr=2e-4,
        temp_lr=2e-4,
        lr_end=1e-5,
        
        # Tau for timely yet stable updates
        tau=0.0075,
        
        # Policy delay for stable actor-critic interaction
        policy_delay=2,
        
        # Initial temperature for exploration
        init_temperature=0.05,
        
        # Weight decay for regularization
        weight_decay=0.005,
        
        # Batch size and updates per step for efficient training
        batch_size=192,
        updates_per_step=1,
    )
