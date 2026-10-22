
# ========== Start: GeneCrossed
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
        hidden_dims_critic=[1152, 1152, 576, 576],  # Slightly increased critic capacity
        hidden_dims_actor=[384, 384, 256, 256, 128],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=True,  # Ensuring last layer normalization for stability
        skip_connections=False,
    )
# --OPTION--
import optuna

def objective(trial):
    # Define the hyperparameter space
    actor_lr = trial.suggest_loguniform('actor_lr', 1e-5, 1e-3)
    critic_lr = trial.suggest_loguniform('critic_lr', 1e-5, 1e-3)
    temp_lr = trial.suggest_loguniform('temp_lr', 1e-5, 1e-3)
    lr_end = trial.suggest_loguniform('lr_end', 1e-6, 1e-4)
    tau = trial.suggest_uniform('tau', 0.001, 0.01)
    policy_delay = trial.suggest_int('policy_delay', 1, 10)
    init_temperature = trial.suggest_uniform('init_temperature', 0.001, 0.1)
    weight_decay = trial.suggest_uniform('weight_decay', 0.0, 0.1)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
    updates_per_step = trial.suggest_int('updates_per_step', 1, 5)

    # Here you would train your model with these hyperparameters and return the loss or metric of interest
    # For demonstration purposes, let's assume we're minimizing a fictional loss function
    loss = actor_lr + critic_lr + temp_lr + lr_end + tau + policy_delay + init_temperature + weight_decay + batch_size / 100 + updates_per_step
    
    return loss

study = optuna.create_study(direction='minimize')
n_trials = 50
study.optimize(objective, n_trials=n_trials)

best_params = study.best_trial.params
print("Best Parameters:", best_params)
