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
ERROR
