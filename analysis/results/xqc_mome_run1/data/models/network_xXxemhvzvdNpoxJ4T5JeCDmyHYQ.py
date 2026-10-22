
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
        hidden_dims_critic=[1024, 512, 512, 256],
        hidden_dims_actor=[256, 256, 256, 128],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=False,
        use_layer_norm=True,
        pre_activation_bn=False,
        normalize_last_layer=False,
        skip_connections=True,
    )
# --OPTION--
import numpy as np

def get_xqc_kwargs(base_lr=1e-4, base_temp_lr=1e-4, lr_end=1e-5, tau=0.01, policy_delay=2, init_temperature=0.1, weight_decay=0.01, batch_size=128, updates_per_step=1):
    """
    Returns a dictionary of hyperparameters for an XQC model.
    
    Parameters:
    - base_lr: Base learning rate for actors and critics.
    - base_temp_lr: Learning rate for temperature.
    - lr_end: Ending learning rate.
    - tau: Tau value for soft update.
    - policy_delay: Delay for policy update.
    - init_temperature: Initial temperature.
    - weight_decay: Weight decay.
    - batch_size: Batch size for training.
    - updates_per_step: Number of updates per step.
    
    Returns:
    - A dictionary containing all hyperparameters.
    """
    return dict(
        actor_lr=base_lr,
        critic_lr=base_lr,
        temp_lr=base_temp_lr,
        lr_end=lr_end,
        tau=tau,
        policy_delay=policy_delay,
        init_temperature=init_temperature,
        weight_decay=weight_decay,
        batch_size=batch_size,
        updates_per_step=updates_per_step,
    )

# Example usage with default parameters
kwargs = get_xqc_kwargs()
print(kwargs)

# Example usage with custom parameters
custom_kwargs = get_xqc_kwargs(base_lr=1e-3, batch_size=256)
print(custom_kwargs)
