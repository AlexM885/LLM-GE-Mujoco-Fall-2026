
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
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
def get_architecture_kwargs():
    return dict(
        # Combining deeper critic networks from Snippet 2 with wider actor networks inspired by Snippet 1
        hidden_dims_critic=[2048, 2048, 1024, 1024],
        hidden_dims_actor=[1024, 512, 256],  # Adjusted to balance complexity and avoid overfitting
        
        # Keeping the number of critics as specified
        n_critics=2,
        
        # Selecting normalization techniques based on their utility in stabilizing training
        use_batch_norm=True,  # Helpful in many cases for stabilizing activations
        use_weight_norm=True,  # Can help with optimization
        use_layer_norm=False,  # Less commonly used but can be beneficial in certain architectures
        
        # Pre-activation BN and normalizing the last layer can be useful but may require tuning
        pre_activation_bn=False,
        normalize_last_layer=False,
        
        # Skip connections can aid in training deep networks
        skip_connections=True,
    )
# --OPTION--
# ===============================
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    return dict(
        actor_lr=3e-5,
        critic_lr=3e-5,
        temp_lr=3e-5,
        lr_end=5e-6,
        tau=0.005,
        policy_delay=2,
        init_temperature=0.05,
        weight_decay=0.001,
        batch_size=512,
        updates_per_step=2,
    )
