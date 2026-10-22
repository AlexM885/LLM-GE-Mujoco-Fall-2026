
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
        hidden_dims_critic=[4096, 4096, 2048, 2048],
        hidden_dims_actor=[2048, 2048, 1024, 1024],
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
    """
    Returns a dictionary of combined hyperparameters for XQC.
    
    This combination aims to balance exploration, exploitation, and learning stability.
    """
    return dict(
        actor_lr=3e-4,         # Actor learning rate
        critic_lr=3e-4,        # Critic learning rate
        temp_lr=7.5e-5,        # Temperature learning rate, a compromise between the two snippets
        lr_end=1e-5,           # Final learning rate
        tau=0.005,             # Target network update rate
        policy_delay=2,        # Delayed policy updates
        init_temperature=0.075,# Initial temperature, balancing exploration and exploitation
        weight_decay=0.001,    # Weight decay for regularization
        batch_size=256,        # Batch size for training
        updates_per_step=2,    # Number of updates per step
    )

# Example usage
if __name__ == "__main__":
    combined_kwargs = get_combined_xqc_kwargs()
    print("Combined XQC Hyperparameters:")
    for key, value in combined_kwargs.items():
        print(f"{key}: {value}")
