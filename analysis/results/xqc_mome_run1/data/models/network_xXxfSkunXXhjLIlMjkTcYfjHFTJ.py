
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
def get_xqc_kwargs():
    return dict(
        actor_lr=3e-3,         # Higher learning rate for faster convergence
        critic_lr=3e-3,        # Higher learning rate for faster convergence
        temp_lr=1e-4,          # Temperature learning rate remains unchanged
        lr_end=1e-5,           # Final learning rate remains unchanged
        tau=0.01,              # Lower tau for potential stability improvement
        policy_delay=2,        # Policy delay from Snippet 1 for consistency
        init_temperature=0.1,  # Initial temperature remains unchanged
        weight_decay=0.01,     # Weight decay from Snippet 1 for stronger regularization
        batch_size=1024,       # Larger batch size for more diverse experience sampling
        updates_per_step=3,    # Updates per step remain unchanged
    )
