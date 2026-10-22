
# ========== Start: GeneCrossed
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
        hidden_dims_critic=[2048, 1024, 512, 256], 
        hidden_dims_actor=[768, 384, 192, 96],
        n_critics=2,
        use_batch_norm=False,
        use_weight_norm=True,
        use_layer_norm=True,
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=True,
    )
# --OPTION--
def get_xqc_kwargs():
    # Balancing act between the two provided snippets
    return dict(
        actor_lr=4.5e-5,  # A compromise between 4e-5 and 5e-4
        critic_lr=4.5e-5,  # Similar compromise for critic LR
        temp_lr=2.5e-4,   # Middle ground for temperature LR
        lr_end=3e-6,      # Compromise between 5e-6 and 1e-5
        tau=0.0075,       # Average of 0.005 and 0.01
        policy_delay=2.5, # Intermediate policy delay
        init_temperature=0.075, # Balanced initial temperature
        weight_decay=0.005,     # Midpoint for weight decay
        batch_size=512,         # No change as both snippets agree
        updates_per_step=3,     # Also unchanged
    )
