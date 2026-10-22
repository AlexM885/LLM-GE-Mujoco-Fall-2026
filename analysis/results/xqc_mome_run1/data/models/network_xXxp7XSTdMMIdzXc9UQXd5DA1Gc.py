
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

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
        hidden_dims_critic=[1024, 1024, 1024, 1024],
        hidden_dims_actor=[512, 512, 512, 512],
        n_critics=2,
        use_batch_norm=True,
        use_weight_norm=True,
        use_layer_norm=False,
        pre_activation_bn=True,
        normalize_last_layer=True,
        skip_connections=False,
    )
# --OPTION--
# ===============================
# === XQC Hyperparameter Gene ===
# ===============================

def get_xqc_kwargs():
    return dict(
        # Balanced learning rates for stability and progress
        actor_lr=4e-4,
        critic_lr=4e-4,
        
        # Moderate temperature learning rate for adaptation
        temp_lr=2e-5,
        
        # Gradual decrease to a lower end learning rate for fine-tuning
        lr_end=5e-6,
        
        # Tau for soft update of target networks
        tau=0.01,
        
        # Policy delay for updating the policy network less frequently than the critic
        policy_delay=2,
        
        # Initial temperature for a balanced start in exploration-exploitation
        init_temperature=0.075,
        
        # Weight decay for regularization against overfitting
        weight_decay=0.00075,
        
        # Batch size for training
        batch_size=256,
        
        # Updates per step to control the frequency of target network updates
        updates_per_step=1.5,
    )
