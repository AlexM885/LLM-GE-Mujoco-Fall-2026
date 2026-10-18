"""
Retrain one network file with sota/MujocoRL/train_rl.py, exactly as an LLM-GE
gene is trained, but with all output kept in a separate folder.

Used to check whether a single good LLM-GE result holds up when the same gene
is trained again (TD3 is not seeded in train_rl.py, so every run differs).

    python sota/MujocoRL/Baseline/reproduction/retrain.py NETWORK_FILE OUT_DIR TAG
"""

import importlib.util
import os
import shutil
import sys
import types

network_file = os.path.abspath(sys.argv[1])
out_dir = os.path.abspath(sys.argv[2])
tag = sys.argv[3]

# sota/MujocoRL holds train_rl.py, eval.py and network.py
mujoco_rl_dir = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(mujoco_rl_dir, "train_rl.py")):
    mujoco_rl_dir = os.path.dirname(mujoco_rl_dir)
sys.path.insert(0, mujoco_rl_dir)
import train_rl  # noqa: E402

# train_rl.main imports "models.network_<gene_id>"; register the file under that
# name (with a "models" parent, which saving the trained model needs as well)
module_name = f"models.network_{tag}"
parent = types.ModuleType("models")
parent.__path__ = []
sys.modules["models"] = parent
spec = importlib.util.spec_from_file_location(module_name, network_file)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
setattr(parent, f"network_{tag}", module)
spec.loader.exec_module(module)

os.makedirs(out_dir, exist_ok=True)
train_rl.main(
    tag,
    env_id=train_rl.DEFAULT_ENV_ID,
    timesteps=int(os.getenv("MUJOCO_EVAL_TIMESTEPS", "500000")),
    eval_episodes=10,
    eval_max_steps=1000,
    model_dir=os.path.join(out_dir, "trained_models"),
    stats_dir=os.path.join(out_dir, "stats"),
)

# train_rl.py always writes the results csv next to itself; move it into OUT_DIR
results = os.path.join(mujoco_rl_dir, "results", f"{tag}_results.csv")
if os.path.exists(results):
    shutil.move(results, os.path.join(out_dir, f"{tag}_results.csv"))
