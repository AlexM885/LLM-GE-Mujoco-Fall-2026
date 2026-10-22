
# ========== Start: GeneCrossed
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
        hidden_dims_critic=[1024, 1024, 512, 512],
        hidden_dims_actor=[512, 512, 256, 256],
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
    """
    Returns a dictionary of combined hyperparameters for the XQC model.
    
    These parameters are chosen based on balancing stability, convergence speed, 
    exploration, and computational efficiency.
    """
    return dict(
        actor_lr=3e-5,         # Moderate learning rate for stability and speed
        critic_lr=3e-5,        # Consistent with actor_lr for balanced updates
        temp_lr=3e-5,          # Same reasoning as above
        lr_end=5e-6,           # Lower end learning rate for fine-tuning
        tau=0.005,             # Smaller tau for more stable target network updates
        policy_delay=2,        # Balanced actor-critic update schedule
        init_temperature=0.05, # Moderate initial exploration
        weight_decay=0.001,    # Less regularization for larger models
        batch_size=512,        # Larger batch size for faster training
        updates_per_step=2,    # Faster convergence with more updates per step
    )

# Example usage
if __name__ == "__main__":
    combined_kwargs = get_combined_xqc
