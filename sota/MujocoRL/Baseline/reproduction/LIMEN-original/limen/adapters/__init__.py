"""Environment adapter factory and exports.

To register a custom adapter:
    from limen.adapters import register_adapter
    register_adapter("my_env", "my_package.my_adapter.MyAdapter")
"""

from limen.adapters.base import EnvAdapter

# Registry: adapter_type -> class or dotted import path (lazy)
_ADAPTER_REGISTRY: dict[str, str | type] = {
    "xminigrid": "limen.adapters.xminigrid.XMinigridAdapter",
    "mujoco": "limen.adapters.mujoco.MujocoAdapter",
    "brax": "limen.adapters.brax.BraxAdapter",
}


def register_adapter(name: str, cls_or_path) -> None:
    """Register an adapter type.

    Args:
        name: Adapter type string (used in config's adapter_type field).
        cls_or_path: Either an EnvAdapter subclass, or a dotted import path
            string like "my_package.my_module.MyAdapter" (lazy import).
    """
    _ADAPTER_REGISTRY[name] = cls_or_path


def _resolve_adapter(cls_or_path) -> type:
    """Resolve a class or dotted path to an actual class."""
    if isinstance(cls_or_path, type):
        return cls_or_path
    module_path, class_name = cls_or_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_adapter(adapter_type: str, config) -> EnvAdapter:
    """Create an environment adapter by type name.

    Args:
        adapter_type: Adapter identifier (e.g., "xminigrid", "mujoco", "brax").
        config: Full system Config object.

    Returns:
        An EnvAdapter instance.
    """
    if adapter_type not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown adapter type: {adapter_type!r}. "
            f"Available: {list(_ADAPTER_REGISTRY.keys())}"
        )
    cls = _resolve_adapter(_ADAPTER_REGISTRY[adapter_type])
    return cls(config)
