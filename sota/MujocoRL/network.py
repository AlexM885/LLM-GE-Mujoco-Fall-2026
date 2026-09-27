# --PROMPT LOG--

import torch.nn as nn
from stable_baselines3.td3.policies import TD3Policy

# --OPTION--

# -- NOTE --
# Note: The class GenePolicy inherits from TD3Policy (Stable-Baselines3).
# It must always accept *args and **kwargs and pass them to super().__init__().
# get_policy_kwargs() must return a dict with "policy_class" key.
# get_td3_kwargs() must return a dict of valid TD3 hyperparameters, plus the
# optional "exploration_noise" key (std of Gaussian action noise, handled by train_rl.py).
# Do not override forward(), _predict(), make_actor() or make_critic(); let SB3 handle those.
# Prefer mutating HIDDEN_PI, HIDDEN_QF, ACTIVATION, N_CRITICS and TD3 kwargs.
# -- NOTE --

# ===============================
# === Architecture Gene Space ===
# ===============================


# Seed follows Nilsson & Cully, "Policy Gradient Assisted MAP-Elites" (GECCO '21):

HIDDEN_PI = [128, 128]
HIDDEN_QF = [256, 256]
ACTIVATION = nn.ReLU
N_CRITICS = 2



class GenePolicy(TD3Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            activation_fn=ACTIVATION,
            net_arch=dict(
                pi=HIDDEN_PI,
                qf=HIDDEN_QF
            ),
            n_critics=N_CRITICS,
        )


def get_policy_kwargs():
    return {
        "policy_class": GenePolicy
    }

# --OPTION--

# ===============================
# === TD3 Hyperparameter Gene ===
# ===============================

def get_td3_kwargs():
    # Values from Table 1 of the PGA-MAP-Elites paper; learning_starts and
    # exploration_noise follow the reference TD3 implementation (Fujimoto et al.).
    return dict(
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=25_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        exploration_noise=0.1,
        verbose=0,
    )
