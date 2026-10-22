
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
        hidden_dims_critic=[1280, 1280, 640, 640],
        hidden_dims_actor=[640, 640, 320, 320],
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
        # Balanced learning rates for stability and progress
        actor_lr=4e-4,
        critic_lr=4e-4,
        
        # Moderate temperature learning rate for adaptation
        temp_lr=1e-4,
        
        # Gradual decrease in learning rate
        lr_end=1e-5,
        
        # Frequent enough updates for adaptation without causing instability
        tau=0.0035,
        
        # Update policy slightly less often than critic for balance
        policy_delay=1.5,
        
        # Initial temperature that promotes exploration without being too extreme
        init_temperature=0.075,
        
        # Weight decay for preventing overfitting
        weight_decay=0.00075,
        
        # Batch size and updates per step for efficient computation
        batch_size=384,
        updates_per_step=2,
    )
