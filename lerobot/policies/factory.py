from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.registry import get_config_class, get_known_policies
from lerobot.policies.utils import validate_visual_features_consistency


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
    policy_type: str,
    ds_meta,
    pretrained_path: str | Path | None = None,
    use_peft: bool = False,
    device: str | None = None,
    rename_map: dict[str, str] | None = None,
    **overrides,
):
    """Instantiate a policy model from a dataset."""
    policy_cls = get_policy_class(policy_type)
    features = dataset_to_policy_features(ds_meta.features)

    input_features = {key: ft for key, ft in features.items() if ft.type is not FeatureType.ACTION}
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}

    cfg = make_policy_config(policy_type, pretrained_path=pretrained_path, use_peft=use_peft, device=device, **overrides)
    cfg.output_features = output_features
    if not cfg.input_features:
        cfg.input_features = input_features
    cfg.validate_features()

    kwargs = {
        "config": cfg,
        "dataset_stats": ds_meta.stats if hasattr(ds_meta, "stats") else None,
        "dataset_meta": ds_meta,
    }

    if cfg.pretrained_path and cfg.use_peft:
        policy = _load_policy_with_peft(policy_cls, cfg.pretrained_path, **kwargs)
    elif cfg.pretrained_path:
        policy = policy_cls.from_pretrained(pretrained_name_or_path=cfg.pretrained_path, **kwargs)
    else:
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    if not rename_map:
        validate_visual_features_consistency(cfg, features)
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
