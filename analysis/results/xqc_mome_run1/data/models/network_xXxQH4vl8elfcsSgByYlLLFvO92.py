
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
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
        # Increased dimensions for critic and actor, balancing between the two snippets
        hidden_dims_critic=[1024, 1024, 512, 512],
        hidden_dims_actor=[512, 512, 256, 256],
        
        # Keeping n_critics as required
        n_critics=2,
        
        # Enabling batch normalization as it's commonly beneficial
        use_batch_norm=True,
        
        # Weight normalization can also be helpful, so enable it
        use_weight_norm=True,
        
        # Layer normalization might not be necessary with batch norm, but let's keep it simple and disable for now
        use_layer_norm=False,
        
        # Pre-activation batch normalization can affect the learning dynamics; let's try without it
        pre_activation_bn=False,
        
        # Normalizing the last layer might not always be beneficial; let's disable it
        normalize_last_layer=False,
        
        # Skip connections can help with deeper networks; let's enable them
        skip_connections=True,
    )
# --OPTION--
# ===============================
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    return dict(
        actor_lr=1e-4,
        critic_lr=1e-4,
        temp_lr=1e-4,
        lr_end=1e-5,
        tau=0.01,
        policy_delay=2,
        init_temperature=0.1,
        weight_decay=0.01,
        batch_size=128,
        updates_per_step=1,
    )
