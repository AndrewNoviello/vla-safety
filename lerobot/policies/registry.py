_CONFIG_REGISTRY: dict[str, type] = {}


def register_policy(name: str, config_cls: type) -> None:
    """Register a config class under the given policy name."""
    _CONFIG_REGISTRY[name] = config_cls


def get_config_class(name: str) -> type:
    if name not in _CONFIG_REGISTRY:
        raise ValueError(f"Unknown policy type '{name}'. Available: {list(_CONFIG_REGISTRY)}")
    return _CONFIG_REGISTRY[name]


def get_known_policies() -> list[str]:
    return list(_CONFIG_REGISTRY)


# ---------------------------------------------------------------------------
# Register all known policies by importing their config modules.
# Add new models here.
# ---------------------------------------------------------------------------

from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: F401, E402
