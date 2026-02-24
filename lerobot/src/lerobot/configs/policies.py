# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import builtins
import dataclasses
import json
import os
import types
from dataclasses import dataclass, field
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, TypeVar, Union, get_args, get_origin, get_type_hints

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import auto_select_torch_device, is_amp_available, is_torch_device_available

T = TypeVar("T", bound="PreTrainedConfig")
logger = getLogger(__name__)


def _is_optional(tp) -> tuple[bool, type | None]:
    """Return (True, inner_type) if *tp* is ``X | None`` or ``Optional[X]``."""
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return True, args[0]
    return False, None


def _serialize(val: Any) -> Any:
    """Recursively convert a value to a JSON-compatible representation."""
    if val is None:
        return None
    if isinstance(val, Enum):
        return val.value
    if isinstance(val, Path):
        return str(val)
    if dataclasses.is_dataclass(val) and not isinstance(val, type):
        return {f.name: _serialize(getattr(val, f.name)) for f in dataclasses.fields(val)}
    if isinstance(val, dict):
        return {k: _serialize(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize(v) for v in val]
    return val


def _deserialize(val: Any, target_type: Any) -> Any:
    """Recursively convert a JSON value to the expected Python type."""
    if val is None:
        return None

    is_opt, inner = _is_optional(target_type)
    if is_opt:
        return _deserialize(val, inner)

    if isinstance(target_type, type) and issubclass(target_type, Enum):
        return target_type(val)

    if isinstance(target_type, type) and issubclass(target_type, Path):
        return Path(val)

    if dataclasses.is_dataclass(target_type):
        hints = get_type_hints(target_type)
        kwargs = {}
        for f in dataclasses.fields(target_type):
            if f.name in val:
                kwargs[f.name] = _deserialize(val[f.name], hints[f.name])
        return target_type(**kwargs)

    origin = get_origin(target_type)
    args = get_args(target_type)

    if origin is dict and args:
        _, val_type = args
        return {k: _deserialize(v, val_type) for k, v in val.items()}

    if origin is tuple and args:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_deserialize(v, args[0]) for v in val)
        return tuple(_deserialize(v, t) for v, t in zip(val, args))

    if isinstance(val, list) and origin is not list:
        # Best-effort: convert JSON arrays back to tuples when the hint is bare `tuple`
        if target_type is tuple:
            return tuple(val)

    return val


@dataclass
class PreTrainedConfig(HubMixin, abc.ABC):
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
    """

    _REGISTRY: ClassVar[dict[str, type["PreTrainedConfig"]]] = {}

    n_obs_steps: int = 1

    input_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] | None = field(default_factory=dict)

    device: str | None = None
    use_amp: bool = False

    use_peft: bool = False

    push_to_hub: bool = True  # type: ignore[assignment]
    repo_id: str | None = None

    private: bool | None = None
    tags: list[str] | None = None
    license: str | None = None
    pretrained_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logger.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

        if self.use_amp and not is_amp_available(self.device):
            logger.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

    # ------------------------------------------------------------------
    # Registry helpers (replaces draccus.ChoiceRegistry)
    # ------------------------------------------------------------------

    @classmethod
    def register_subclass(cls, name: str):
        """Decorator that registers a config subclass under *name*."""
        def decorator(subclass):
            cls._REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get_choice_class(cls, name: str) -> type["PreTrainedConfig"]:
        if name not in cls._REGISTRY:
            raise ValueError(f"Unknown policy type '{name}'. Available: {list(cls._REGISTRY)}")
        return cls._REGISTRY[name]

    @classmethod
    def get_known_choices(cls) -> list[str]:
        return list(cls._REGISTRY)

    @classmethod
    def get_choice_name(cls, subclass: type) -> str | None:
        for name, registered_cls in cls._REGISTRY.items():
            if registered_cls is subclass:
                return name
        return None

    @property
    def type(self) -> str:
        choice_name = self.get_choice_name(self.__class__)
        if not isinstance(choice_name, str):
            raise TypeError(f"Expected string from get_choice_name, got {type(choice_name)}")
        return choice_name

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def observation_delta_indices(self) -> list | None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_delta_indices(self) -> list | None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reward_delta_indices(self) -> list | None:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        if not self.output_features:
            return None
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    # ------------------------------------------------------------------
    # Serialization / deserialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = _serialize(self)
        d["type"] = self.type
        return d

    def _save_pretrained(self, save_directory: Path) -> None:
        save_directory.mkdir(parents=True, exist_ok=True)
        with open(save_directory / CONFIG_NAME, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs: Any,
    ) -> T:
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                logger.error(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        if config_file is None:
            raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}")

        with open(config_file) as f:
            data = json.load(f)

        type_name = data.pop("type", None)

        if type_name is not None and type_name in cls._REGISTRY:
            target_cls = cls._REGISTRY[type_name]
        elif cls is not PreTrainedConfig and cls not in (PreTrainedConfig,):
            target_cls = cls
        else:
            raise ValueError(
                f"Cannot determine config subclass: 'type' field is '{type_name}' "
                f"and no matching registered subclass was found. Available: {list(cls._REGISTRY)}"
            )

        return _config_from_dict(target_cls, data)


def _config_from_dict(config_cls: type[T], data: dict[str, Any]) -> T:
    """Instantiate a config dataclass from a plain dict, converting types as needed."""
    hints = get_type_hints(config_cls)
    known_fields = {f.name for f in dataclasses.fields(config_cls)}
    kwargs: dict[str, Any] = {}
    for key, val in data.items():
        if key not in known_fields:
            continue
        target_type = hints.get(key)
        if target_type is not None:
            kwargs[key] = _deserialize(val, target_type)
        else:
            kwargs[key] = val
    return config_cls(**kwargs)
