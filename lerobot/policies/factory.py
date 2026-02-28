from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from lerobot.types import FeatureType, PolicyFeature
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.registry import get_config_class, get_known_policies
from lerobot.policies.utils import validate_visual_features_consistency
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """Retrieve a policy class by its registered name."""
    try:
        return _get_policy_cls_from_policy_name(name=name)
    except Exception as e:
        raise ValueError(f"Policy type '{name}' is not available.") from e


def make_policy_config(policy_type: str, **kwargs):
    """Instantiate a policy configuration object by type string."""
    config_cls = get_config_class(policy_type)
    return config_cls(**kwargs)


def make_configuration(
    policy_type: str,
    input_features: dict[str, PolicyFeature],
    output_features: dict[str, PolicyFeature],
    pretrained_path: str | Path | None = None,
    use_peft: bool = False,
    device: str | None = None,
    **overrides,
):
    """Build a complete policy configuration from explicit input/output features.

    This is the first step of the config-first policy creation flow.
    After calling this, pass the returned config to make_policy().
    """
    cfg = make_policy_config(
        policy_type,
        input_features=input_features,
        output_features=output_features,
        pretrained_path=pretrained_path,
        use_peft=use_peft,
        device=device,
        **overrides,
    )
    _complete_features(cfg)
    validate_visual_features_consistency(cfg, {**input_features, **output_features})
    return cfg


def _complete_features(cfg) -> None:
    """Apply policy-specific feature completion to a config in place."""
    if cfg.type == "pi05":
        for i in range(cfg.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            cfg.input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *cfg.image_resolution),
            )
        if OBS_STATE not in cfg.input_features:
            cfg.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(cfg.max_state_dim,),
            )
        if ACTION not in cfg.output_features:
            cfg.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(cfg.max_action_dim,),
            )
    # PI0: no completion needed


def make_pre_post_processors(
    policy_type: str,
    policy: PreTrainedPolicy | None = None,
    policy_cfg=None,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Callable, Callable]:
    """Create pre-/post-processor callables. Requires policy_cfg for all policy types."""
    if policy_cfg is not None:
        try:
            return _make_processors_from_policy_config(config=policy_cfg, dataset_stats=dataset_stats)
        except Exception as e:
            raise ValueError(f"Processor for policy type '{policy_cfg.type}' is not implemented.") from e

    raise ValueError("policy_cfg is required to create pre/post processors.")


def _load_policy_with_peft(policy_cls, pretrained_path: str | Path, **kwargs) -> "PreTrainedPolicy":
    """Load a policy from a PEFT adapter checkpoint."""
    from peft import PeftConfig, PeftModel

    logging.info("Loading policy's PEFT adapter.")
    peft_config = PeftConfig.from_pretrained(pretrained_path)
    base_path = peft_config.base_model_name_or_path
    if not base_path:
        raise ValueError(
            "No pretrained model name found in adapter config. "
            "Can't instantiate the pre-trained policy on which the adapter was trained."
        )
    policy = policy_cls.from_pretrained(pretrained_name_or_path=base_path, **kwargs)
    return PeftModel.from_pretrained(policy, pretrained_path, config=peft_config)


def make_policy(
    config,
    *,
    dataset_stats: dict | None = None,
    dataset_meta=None,
) -> PreTrainedPolicy:
    """Instantiate a policy from a pre-built configuration.

    This is the second step of the config-first policy creation flow.
    Build the config first with make_configuration(), then pass it here.
    """
    policy_cls = get_policy_class(config.type)
    kwargs = {
        "config": config,
        "dataset_stats": dataset_stats,
        "dataset_meta": dataset_meta,
    }

    if config.pretrained_path and config.use_peft:
        policy = _load_policy_with_peft(policy_cls, config.pretrained_path, **kwargs)
    elif config.pretrained_path:
        policy = policy_cls.from_pretrained(pretrained_name_or_path=config.pretrained_path, **kwargs)
    else:
        policy = policy_cls(**kwargs)

    policy.to(config.device)
    return policy


def _get_policy_cls_from_policy_name(name: str):
    if name not in get_known_policies():
        raise ValueError(
            f"Unknown policy name '{name}'. Available policies: {get_known_policies()}"
        )

    config_cls = get_config_class(name)
    config_cls_name = config_cls.__name__

    model_name = config_cls_name.removesuffix("Config")
    if model_name == config_cls_name:
        raise ValueError(
            f"The config class name '{config_cls_name}' does not follow the expected naming convention. "
            "Make sure it ends with 'Config'!"
        )
    cls_name = model_name + "Policy"
    module_path = config_cls.__module__.replace("configuration_", "modeling_")

    module = importlib.import_module(module_path)
    policy_cls = getattr(module, cls_name)
    return policy_cls


def _make_processors_from_policy_config(
    config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Callable, Callable]:
    policy_type = config.type
    function_name = f"make_{policy_type}_pre_post_processors"
    module_path = config.__class__.__module__.replace("configuration_", "processor_")
    logging.debug(
        f"Instantiating pre/post processors using function '{function_name}' from module '{module_path}'"
    )
    module = importlib.import_module(module_path)
    function = getattr(module, function_name)
    return function(config, dataset_stats=dataset_stats)
