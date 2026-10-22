
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
# normalization flags. n_critics must stay 2.
# -- NOTE --

# ===============================
# === Architecture Gene Space ===
# ===============================

# Seed = the XQC paper settings (12,598 +- 1,174 on HalfCheetah-v4 at 1M steps, 10 seeds).

def get_architecture_kwargs():
    return dict(
        hidden_dims_critic=[512, 512, 512, 512],
        hidden_dims_actor=[256, 256, 256, 256],
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
# === Combined XQC Hyperparameters ===
# ===============================

def get_combined_xqc_kwargs():
    return dict(
        # Balance between fast convergence and stability
        actor_lr=2e-4,
        critic_lr=2e-4,
        temp_lr=3e-5,
        
        # Gradual decrease to a small learning rate end value
        lr_end=1e-5,
        
        # Moderate update rate for target networks
        tau=0.005,
        
        # Delay policy updates to stabilize training
        policy_delay=2,
        
        # Encourage exploration initially
        init_temperature=0.05,
        
        # Apply moderate regularization
        weight_decay=0.0005,
        
        # Use a moderate batch size for a balance between speed and generalization
        batch_size=384,
        
        # Update twice per step for faster convergence
        updates_per_step=2,
    )

# Example usage
combined_kwargs = get_combined_xqc_kwargs()
print(combined_kwargs)
