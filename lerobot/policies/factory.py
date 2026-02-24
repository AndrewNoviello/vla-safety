from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import validate_visual_features_consistency


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """Retrieve a policy class by its registered name."""
    if name == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy
    elif name == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        return PI05Policy
    else:
        try:
            return _get_policy_cls_from_policy_name(name=name)
        except Exception as e:
            raise ValueError(f"Policy type '{name}' is not available.") from e


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    """Instantiate a policy configuration object by type string. Not used for PI0."""
    if policy_type == "pi05":
        return PI05Config(**kwargs)
    else:
        try:
            config_cls = PreTrainedConfig.get_choice_class(policy_type)
            return config_cls(**kwargs)
        except Exception as e:
            raise ValueError(f"Policy type '{policy_type}' is not available.") from e


def make_pre_post_processors(
    policy_type: str,
    policy: PreTrainedPolicy | None = None,
    policy_cfg: PreTrainedConfig | None = None,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Callable, Callable]:
    """Create pre-/post-processor callables. For PI0 use policy; for others use policy_cfg."""
    if policy_type == "pi0" and policy is not None:
        from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors

        return make_pi0_pre_post_processors(
            input_features=policy.input_features,
            output_features=policy.output_features,
            device=policy.config.device,
            tokenizer_max_length=48,
            dataset_stats=dataset_stats,
        )

    if policy_cfg is not None and isinstance(policy_cfg, PI05Config):
        from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

        return make_pi05_pre_post_processors(config=policy_cfg, dataset_stats=dataset_stats)

    if policy_cfg is not None:
        try:
            return _make_processors_from_policy_config(config=policy_cfg, dataset_stats=dataset_stats)
        except Exception as e:
            raise ValueError(f"Processor for policy type '{policy_cfg.type}' is not implemented.") from e

    raise ValueError("For PI0 pass policy; for other policies pass policy_cfg.")


def make_policy(
    policy_type: str,
    ds_meta,
    pretrained_path: str | Path | None = None,
    use_peft: bool = False,
    device: str | None = None,
    rename_map: dict[str, str] | None = None,
    **overrides,
) -> PreTrainedPolicy:
    """Instantiate a policy model from a dataset."""
    policy_cls = get_policy_class(policy_type)
    features = dataset_to_policy_features(ds_meta.features)

    input_features = {key: ft for key, ft in features.items() if ft.type is not FeatureType.ACTION}
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}

    if policy_type == "pi0":
        kwargs: dict[str, Any] = {
            "input_features": input_features,
            "output_features": output_features,
            "device": device,
            "dataset_stats": ds_meta.stats if hasattr(ds_meta, "stats") else None,
            "dataset_meta": ds_meta,
            **overrides,
        }

        if pretrained_path and not use_peft:
            policy = policy_cls.from_pretrained(
                pretrained_name_or_path=pretrained_path,
                input_features=input_features,
                output_features=output_features,
                dataset_stats=kwargs.get("dataset_stats"),
                dataset_meta=ds_meta,
                **overrides,
            )
        elif pretrained_path and use_peft:
            from peft import PeftConfig, PeftModel

            logging.info("Loading policy's PEFT adapter.")
            peft_config = PeftConfig.from_pretrained(pretrained_path)
            base_path = peft_config.base_model_name_or_path
            if not base_path:
                raise ValueError(
                    "No pretrained model name found in adapter config. "
                    "Can't instantiate the pre-trained policy on which the adapter was trained."
                )
            policy = policy_cls.from_pretrained(
                pretrained_name_or_path=base_path,
                input_features=input_features,
                output_features=output_features,
                dataset_stats=kwargs.get("dataset_stats"),
                dataset_meta=ds_meta,
                **overrides,
            )
            policy = PeftModel.from_pretrained(policy, pretrained_path, config=peft_config)
        else:
            if use_peft:
                raise ValueError(
                    "Instantiating a policy with use_peft=True without a checkpoint is not supported. "
                    "See lerobot_train.py for PEFT training."
                )
            policy = policy_cls(**kwargs)

        resolved_device = policy.config.device
        policy.to(resolved_device)
        if not rename_map:
            validate_visual_features_consistency_pi0(policy, features)
        return policy

    # PI05 / PI0Fast: use config-based path
    cfg = make_policy_config(policy_type, pretrained_path=pretrained_path, use_peft=use_peft, device=device, **overrides)
    cfg.output_features = output_features
    if not cfg.input_features:
        cfg.input_features = input_features

    kwargs = {
        "config": cfg,
        "dataset_stats": ds_meta.stats if hasattr(ds_meta, "stats") else None,
        "dataset_meta": ds_meta,
    }

    if cfg.pretrained_path and not cfg.use_peft:
        policy = policy_cls.from_pretrained(pretrained_name_or_path=cfg.pretrained_path, **kwargs)
    elif cfg.pretrained_path and cfg.use_peft:
        from peft import PeftConfig, PeftModel

        logging.info("Loading policy's PEFT adapter.")
        peft_config = PeftConfig.from_pretrained(cfg.pretrained_path)
        kwargs["pretrained_name_or_path"] = peft_config.base_model_name_or_path
        if not kwargs["pretrained_name_or_path"]:
            raise ValueError(
                "No pretrained model name found in adapter config. "
                "Can't instantiate the pre-trained policy on which the adapter was trained."
            )
        policy = policy_cls.from_pretrained(**kwargs)
        policy = PeftModel.from_pretrained(policy, cfg.pretrained_path, config=peft_config)
    else:
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    if not rename_map:
        validate_visual_features_consistency(cfg, features)
    return policy


def validate_visual_features_consistency_pi0(policy: PreTrainedPolicy, features: dict) -> None:
    """Validate visual features for PI0 (uses policy attributes instead of config)."""
    expected_visuals = {k for k, v in policy.input_features.items() if v.type == FeatureType.VISUAL}
    provided_visuals = {k for k, v in features.items() if v.type == FeatureType.VISUAL}
    policy_subset = expected_visuals.issubset(provided_visuals)
    dataset_subset = provided_visuals.issubset(expected_visuals)
    if not (policy_subset or dataset_subset):
        from lerobot.policies.utils import raise_feature_mismatch_error

        raise_feature_mismatch_error(provided_visuals, expected_visuals)


def _get_policy_cls_from_policy_name(name: str) -> type[PreTrainedPolicy]:
    if name not in PreTrainedConfig.get_known_choices():
        raise ValueError(
            f"Unknown policy name '{name}'. Available policies: {PreTrainedConfig.get_known_choices()}"
        )

    config_cls = PreTrainedConfig.get_choice_class(name)
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
    config: PreTrainedConfig,
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
