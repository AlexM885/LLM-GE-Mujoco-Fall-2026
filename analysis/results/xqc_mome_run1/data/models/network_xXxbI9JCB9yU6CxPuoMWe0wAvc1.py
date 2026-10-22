
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
# ========== End:

# ========== Start: GeneCrossed

# ========== End:
# ========== End:
# ========== End:

# ========== Start: GeneCrossed

# ========== End:

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
        # Increase depth slightly while balancing with width for better feature extraction
        hidden_dims_critic=[2048, 1792, 1536, 1280, 1024],
        hidden_dims_actor=[1024, 896, 768, 640, 512],
        
        # Keep the required number of critics constant as per XQC requirements
        n_critics=2,
        
        # Enable batch normalization for improved stability across deeper networks
        use_batch_norm=True,
        
        # Disable weight normalization to avoid potential redundancy with batch normalization
        use_weight_norm=False,
        
        # Disable layer normalization due to potential interaction issues with batch normalization
        use_layer_norm=False,
        
        # Other parameters adjusted based on experimental needs and stability considerations
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=True,
    )
# --OPTION--
# ===============================
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    return dict(
        actor_lr=1e-5,
        critic_lr=1e-5,
        temp_lr=1e-5,
        lr_end=1e-5,
        tau=0.01,
        policy_delay=2,
        init_temperature=0.1,
        weight_decay=0.01,
        batch_size=256,
        updates_per_step=1,
    )
