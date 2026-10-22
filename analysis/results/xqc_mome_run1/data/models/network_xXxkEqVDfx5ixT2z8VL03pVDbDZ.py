
# ========== Start: GeneCrossed

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
        # Using deeper critic networks with larger dimensions
        hidden_dims_critic=[2048, 1024, 512, 256],
        
        # Deeper actor networks with moderate dimensions
        hidden_dims_actor=[1024, 512, 256, 128],
        
        # Keeping the number of critics constant as per the problem statement
        n_critics=2,
        
        # Enabling batch normalization for stability
        use_batch_norm=True,
        
        # Also enabling weight normalization for better generalization
        use_weight_norm=True,
        
        # Not using layer normalization in this example
        use_layer_norm=False,
        
        # Applying batch normalization before activation
        pre_activation_bn=True,
        
        # Normalizing the last layer for improved stability
        normalize_last_layer=True,
        
        # Not using skip connections in this combined version
        skip_connections=False,
    )
# --OPTION--
# ===============================
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    return dict(
        actor_lr=4e-5,
        critic_lr=4e-5,
        temp_lr=5e-5,
        lr_end=5e-6,
        tau=0.005,
        policy_delay=2,
        init_temperature=0.05,
        weight_decay=0.001,
        batch_size=512,
        updates_per_step=1,
    )
