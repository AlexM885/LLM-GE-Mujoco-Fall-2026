
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
def get_xqc_kwargs(
    actor_lr=1e-4,
    critic_lr=1e-4,
    temp_lr=1e-4,
    lr_end=1e-5,
    tau=0.01,
    policy_delay=2,
    init_temperature=0.1,
    weight_decay=0.01,
    batch_size=512,
    updates_per_step=2,
):
    return dict(
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        temp_lr=temp_lr,
        lr_end=lr_end,
        tau=tau,
        policy_delay=policy_delay,
        init_temperature=init_temperature,
        weight_decay=weight_decay,
        batch_size=batch_size,
        updates_per_step=updates_per_step,
    )
# --OPTION--

# ===============================
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    return dict(
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        lr_end=3e-5,
        tau=0.005,
        policy_delay=3,
        init_temperature=0.01,
        weight_decay=0.0,
        batch_size=256,
        updates_per_step=2,
    )
