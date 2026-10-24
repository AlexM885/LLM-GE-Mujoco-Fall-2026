"""MuJoCo-specific helpers that need no simulator imports."""

import importlib.util
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROTOCOL_FILE = os.path.join(REPO_ROOT, "sota", "MujocoRL", "protocol.py")

#: Same as MAP_ELITES_DESCRIPTORS / MAP_BINS in src/cfg/constants_Mujoco.py.
DESCRIPTORS = [("foot_contact_0", 0.0, 1.0), ("foot_contact_1", 0.0, 1.0)]
BINS = 32


def _protocol_module():
    spec = importlib.util.spec_from_file_location("mujoco_eval_protocol", PROTOCOL_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluation_protocol(env_id, max_steps, policy_device="cpu"):
    """The protocol record the evaluation jobs write (sota/MujocoRL/protocol.py)."""
    return _protocol_module().evaluation_protocol(env_id, max_steps, policy_device)
