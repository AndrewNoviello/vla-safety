"""
Sequential data processing pipeline for VLA policy pre- and post-processing.

This is the core module of the processor package. It provides:

- ProcessorStep: Abstract base class for a single transformation step.
- PolicyProcessorPipeline: Chains steps and handles Hub save/load.
- DeviceStep / AddBatchDimStep: Built-in concrete steps.
- Utility functions: to_tensor, from_tensor_to_numpy, images_to_chw_float, move_to_device.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import singledispatch
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import get_safe_torch_device


@singledispatch
def to_tensor(
    value: Any,
    *,
    dtype: torch.dtype | None = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert various data types to PyTorch tensors."""
    raise TypeError(f"Unsupported type for tensor conversion: {type(value)}")


@to_tensor.register(torch.Tensor)
def _(value: torch.Tensor, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    if dtype is not None:
        value = value.to(dtype=dtype)
    if device is not None:
        value = value.to(device=device)
    return value


@to_tensor.register(np.ndarray)
def _(value: np.ndarray, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    if value.ndim == 0:
        return torch.tensor(value.item(), dtype=dtype, device=device)
    tensor = torch.from_numpy(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


@to_tensor.register(int)
@to_tensor.register(float)
@to_tensor.register(np.integer)
@to_tensor.register(np.floating)
def _(value, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype, device=device)


@to_tensor.register(list)
@to_tensor.register(tuple)
def _(value: Sequence, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype, device=device)


@to_tensor.register(dict)
def _(value: dict, *, device=None, **kwargs) -> dict:
    if not value:
        return {}
    result = {}
    for key, sub_value in value.items():
        if sub_value is None:
            continue
        result[key] = to_tensor(sub_value, device=device, **kwargs)
    return result


def from_tensor_to_numpy(x: torch.Tensor | Any) -> np.ndarray | float | int | Any:
    """Convert a PyTorch tensor to a numpy array or scalar if applicable."""
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.detach().cpu().numpy()
    return x


def add_batch_dim(batch: dict[str, Any]) -> dict[str, Any]:
    """Unsqueeze dim=0 on state/image tensors and wrap a str task as a list."""
    result = dict(batch)

    for state_key in (OBS_STATE, OBS_ENV_STATE):
        if state_key in result:
            val = result[state_key]
            if isinstance(val, torch.Tensor) and val.dim() == 1:
                result[state_key] = val.unsqueeze(0)

    if OBS_IMAGE in result:
        val = result[OBS_IMAGE]
        if isinstance(val, torch.Tensor) and val.dim() == 3:
            result[OBS_IMAGE] = val.unsqueeze(0)

    for key, val in list(result.items()):
        if key.startswith(f"{OBS_IMAGES}.") and isinstance(val, torch.Tensor) and val.dim() == 3:
            result[key] = val.unsqueeze(0)

    if "action" in result:
        val = result["action"]
        if isinstance(val, torch.Tensor) and val.dim() == 1:
            result["action"] = val.unsqueeze(0)

    if "task" in result and isinstance(result["task"], str):
        result["task"] = [result["task"]]

    return result


def images_to_chw_float(batch: dict[str, Any]) -> dict[str, Any]:
    """Convert image observations from uint8 HWC to float32 CHW in [0, 1]."""
    result = dict(batch)

    def _convert(val: torch.Tensor) -> torch.Tensor:
        if val.dtype == torch.uint8:
            val = val.float() / 255.0
        if val.dim() == 3:
            val = val.permute(2, 0, 1)
        elif val.dim() == 4:
            val = val.permute(0, 3, 1, 2)
        return val

    if OBS_IMAGE in result and isinstance(result[OBS_IMAGE], torch.Tensor):
        result[OBS_IMAGE] = _convert(result[OBS_IMAGE])

    for key in list(result.keys()):
        if key.startswith(f"{OBS_IMAGES}.") and isinstance(result[key], torch.Tensor):
            result[key] = _convert(result[key])

    return result


def move_to_device(data: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    """Move all tensors in a flat dict to device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}


class ProcessorStep(ABC):
    """Abstract base class for a single step in a data processing pipeline."""

    @abstractmethod
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        ...

    def get_config(self) -> dict[str, Any]:
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        return None

    def reset(self) -> None:
        return None


class ProcessorMigrationError(Exception):
    """Raised when a model checkpoint needs migration to the processor format."""

    def __init__(self, model_path: str | Path, migration_command: str, original_error: str):
        self.model_path = model_path
        self.migration_command = migration_command
        self.original_error = original_error
        super().__init__(
            f"Model '{model_path}' requires migration to processor format. "
            f"Run: {migration_command}\n\nOriginal error: {original_error}"
        )


class IdentityProcessorStep(ProcessorStep):
    """No-op processor step that returns the batch unchanged.

    Accepts and ignores arbitrary kwargs so old checkpoint configs
    (e.g. ``rename_observations_processor`` with a ``rename_map``) can
    deserialise without errors.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        return batch


@dataclass
class AddBatchDimStep(ProcessorStep):
    """Add a batch dimension (size 1) to state/image/action tensors and wrap task strings."""

    _registry_name = "to_batch_processor"

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        return add_batch_dim(batch)


@dataclass
class DeviceStep(ProcessorStep):
    """Move all tensors in a batch dict to a target device and optionally cast dtype."""

    _registry_name = "device_processor"

    device: str = "cpu"
    float_dtype: str | None = None

    DTYPE_MAPPING = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "half": torch.float16,
        "float": torch.float32,
        "double": torch.float64,
    }

    def __post_init__(self):
        self.tensor_device: torch.device = get_safe_torch_device(self.device)
        self.device = self.tensor_device.type
        self.non_blocking = "cuda" in str(self.device)

        if self.float_dtype is not None:
            if self.float_dtype not in self.DTYPE_MAPPING:
                raise ValueError(
                    f"Invalid float_dtype '{self.float_dtype}'. "
                    f"Available options: {list(self.DTYPE_MAPPING.keys())}"
                )
            self._target_float_dtype = self.DTYPE_MAPPING[self.float_dtype]
        else:
            self._target_float_dtype = None

    def _process_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_cuda and self.tensor_device.type == "cuda":
            target_device = tensor.device
        else:
            target_device = self.tensor_device

        if target_device.type == "mps" and tensor.dtype == torch.float64:
            tensor = tensor.to(dtype=torch.float32)

        if tensor.device != target_device:
            tensor = tensor.to(target_device, non_blocking=self.non_blocking)

        if self._target_float_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=self._target_float_dtype)

        return tensor

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            k: self._process_tensor(v) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def get_config(self) -> dict[str, Any]:
        return {"device": self.device, "float_dtype": self.float_dtype}


@dataclass
class PolicyProcessorPipeline(HubMixin):
    """Sequential pipeline that chains ProcessorStep instances with Hub save/load."""

    steps: Sequence[ProcessorStep] = field(default_factory=list)
    name: str = "PolicyProcessorPipeline"

    def __call__(self, data: dict[str, Any] | torch.Tensor) -> dict[str, Any] | torch.Tensor:
        # Accept a bare tensor (e.g. action) for backward compat with
        # callers that do ``postprocessor(action_tensor)``.
        unwrap = False
        if isinstance(data, torch.Tensor):
            data = {"action": data}
            unwrap = True

        for step in self.steps:
            data = step(data)

        return data["action"] if unwrap else data

    def __post_init__(self):
        for i, step in enumerate(self.steps):
            if not isinstance(step, ProcessorStep):
                raise TypeError(f"Step {i} ({type(step).__name__}) must inherit from ProcessorStep")

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int | slice) -> ProcessorStep | PolicyProcessorPipeline:
        if isinstance(idx, slice):
            return PolicyProcessorPipeline(steps=self.steps[idx], name=self.name)
        return self.steps[idx]

    def reset(self):
        for step in self.steps:
            if hasattr(step, "reset"):
                step.reset()

    def __repr__(self) -> str:
        step_names = [step.__class__.__name__ for step in self.steps]
        if not step_names:
            steps_repr = "steps=0: []"
        elif len(step_names) <= 3:
            joiner = ", "
            steps_repr = f"steps={len(step_names)}: [{joiner.join(step_names)}]"
        else:
            displayed = f"{step_names[0]}, {step_names[1]}, ..., {step_names[-1]}"
            steps_repr = f"steps={len(step_names)}: [{displayed}]"
        joiner = ", "
        return f"PolicyProcessorPipeline({joiner.join([f'name={self.name!r}', steps_repr])})"

    def _save_pretrained(self, save_directory: Path, **kwargs):
        config_filename = kwargs.pop("config_filename", None)
        sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", self.name.lower())
        if config_filename is None:
            config_filename = f"{sanitized_name}.json"

        config: dict[str, Any] = {"name": self.name, "steps": []}

        for step_index, processor_step in enumerate(self.steps):
            registry_name = getattr(processor_step.__class__, "_registry_name", None)
            step_entry: dict[str, Any] = {}

            if registry_name:
                step_entry["registry_name"] = registry_name
            else:
                step_entry["class"] = (
                    f"{processor_step.__class__.__module__}.{processor_step.__class__.__name__}"
                )

            if hasattr(processor_step, "get_config"):
                step_entry["config"] = processor_step.get_config()

            if hasattr(processor_step, "state_dict"):
                state = processor_step.state_dict()
                if state:
                    cloned_state = {key: tensor.clone() for key, tensor in state.items()}
                    if registry_name:
                        state_filename = (
                            f"{sanitized_name}_step_{step_index}_{registry_name}.safetensors"
                        )
                    else:
                        state_filename = f"{sanitized_name}_step_{step_index}.safetensors"
                    save_file(cloned_state, os.path.join(str(save_directory), state_filename))
                    step_entry["state_file"] = state_filename

            config["steps"].append(step_entry)

        with open(os.path.join(str(save_directory), config_filename), "w") as fp:
            json.dump(config, fp, indent=2)

    def save_pretrained(
        self,
        save_directory: str | Path | None = None,
        *,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        card_kwargs: dict[str, Any] | None = None,
        config_filename: str | None = None,
        **push_to_hub_kwargs,
    ):
        if save_directory is None:
            from lerobot.utils.constants import HF_LEROBOT_HOME

            sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", self.name.lower())
            save_directory = HF_LEROBOT_HOME / "processors" / sanitized_name

        if not push_to_hub and config_filename is not None:
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
            self._save_pretrained(save_directory, config_filename=config_filename)
            return None

        if config_filename is not None:
            push_to_hub_kwargs["config_filename"] = config_filename

        return super().save_pretrained(
            save_directory=save_directory,
            repo_id=repo_id,
            push_to_hub=push_to_hub,
            card_kwargs=card_kwargs,
            **push_to_hub_kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        config_filename: str,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[str, str] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        overrides: dict[str, Any] | None = None,
        **kwargs,
    ) -> PolicyProcessorPipeline:
        model_id = str(pretrained_model_name_or_path)
        hub_kwargs = {
            "force_download": force_download,
            "resume_download": resume_download,
            "proxies": proxies,
            "token": token,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "revision": revision,
        }

        loaded_config, base_path = cls._load_config(model_id, config_filename, hub_kwargs)
        cls._validate_loaded_config(model_id, loaded_config, config_filename)
        steps, remaining_overrides = cls._build_steps_with_overrides(
            loaded_config, overrides or {}, model_id, base_path, hub_kwargs
        )
        cls._validate_overrides_used(remaining_overrides, loaded_config)

        return cls(
            steps=steps,
            name=loaded_config.get("name", "PolicyProcessorPipeline"),
        )

    @classmethod
    def _load_config(cls, model_id, config_filename, hub_kwargs):
        model_path = Path(model_id)

        if model_path.is_dir():
            config_path = model_path / config_filename
            if not config_path.exists():
                if cls._should_suggest_migration(model_path):
                    cls._suggest_processor_migration(model_id, f"Config file '{config_filename}' not found")
                raise FileNotFoundError(
                    f"Config file '{config_filename}' not found in directory '{model_id}'"
                )
            with open(config_path) as f:
                return json.load(f), model_path

        elif model_path.is_file():
            with open(model_path) as f:
                return json.load(f), model_path.parent

        else:
            try:
                config_path = hf_hub_download(
                    repo_id=model_id, filename=config_filename, repo_type="model", **hub_kwargs
                )
                with open(config_path) as f:
                    return json.load(f), Path(config_path).parent
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not find '{config_filename}' on the HuggingFace Hub at '{model_id}'"
                ) from e

    @classmethod
    def _validate_loaded_config(cls, model_id, loaded_config, config_filename):
        if not cls._is_processor_config(loaded_config):
            if Path(model_id).is_dir() and cls._should_suggest_migration(Path(model_id)):
                cls._suggest_processor_migration(
                    model_id,
                    f"Config file '{config_filename}' is not a valid processor configuration",
                )
            raise ValueError(
                f"Config file '{config_filename}' is not a valid processor configuration. "
                f"Expected a config with 'steps' field, but got: {list(loaded_config.keys())}"
            )

    @classmethod
    def _build_steps_with_overrides(cls, loaded_config, overrides, model_id, base_path, hub_kwargs):
        steps = []
        override_keys = set(overrides.keys())

        for step_entry in loaded_config["steps"]:
            step_class, step_key = cls._resolve_step_class(step_entry)
            step_instance = cls._instantiate_step(step_entry, step_class, step_key, overrides)
            cls._load_step_state(step_instance, step_entry, model_id, base_path, hub_kwargs)
            override_keys.discard(step_key)
            steps.append(step_instance)

        return steps, override_keys

    @classmethod
    def _resolve_step_class(cls, step_entry):
        if "registry_name" in step_entry:
            step_map = _build_step_class_map()
            name = step_entry["registry_name"]
            if name not in step_map:
                available = list(step_map.keys())
                raise ImportError(
                    f"Processor step '{name}' not found. Available steps: {available}"
                )
            return step_map[name], name
        else:
            full_class_path = step_entry["class"]
            module_path, class_name = full_class_path.rsplit(".", 1)
            try:
                module = importlib.import_module(module_path)
                step_class = getattr(module, class_name)
                return step_class, class_name
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Failed to load processor step '{full_class_path}': {e}"
                ) from e

    @classmethod
    def _instantiate_step(cls, step_entry, step_class, step_key, overrides):
        try:
            saved_cfg = step_entry.get("config", {})
            step_overrides = overrides.get(step_key, {})
            merged_cfg = {**saved_cfg, **step_overrides}
            return step_class(**merged_cfg)
        except Exception as e:
            step_name = step_entry.get("registry_name", step_entry.get("class", "Unknown"))
            cfg_dict = step_entry.get("config", {})
            raise ValueError(
                f"Failed to instantiate processor step '{step_name}' "
                f"with config {cfg_dict}: {e}"
            ) from e

    @classmethod
    def _load_step_state(cls, step_instance, step_entry, model_id, base_path, hub_kwargs):
        if "state_file" not in step_entry or not hasattr(step_instance, "load_state_dict"):
            return

        state_filename = step_entry["state_file"]
        if base_path and (base_path / state_filename).exists():
            state_path = str(base_path / state_filename)
        else:
            state_path = hf_hub_download(
                repo_id=model_id, filename=state_filename, repo_type="model", **hub_kwargs
            )
        step_instance.load_state_dict(load_file(state_path))

    @classmethod
    def _validate_overrides_used(cls, remaining_override_keys, loaded_config):
        if not remaining_override_keys:
            return
        available_keys = [
            step.get("registry_name") or step["class"].rsplit(".", 1)[1]
            for step in loaded_config["steps"]
        ]
        raise KeyError(
            f"Override keys {list(remaining_override_keys)} do not match any step. "
            f"Available step keys: {available_keys}"
        )

    @classmethod
    def _should_suggest_migration(cls, model_path):
        json_files = list(model_path.glob("*.json"))
        if not json_files:
            return False
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    config = json.load(f)
                if cls._is_processor_config(config):
                    return False
            except (json.JSONDecodeError, OSError):
                continue
        return True

    @classmethod
    def _is_processor_config(cls, config):
        if not isinstance(config.get("steps"), list):
            return False
        steps = config["steps"]
        if not steps:
            return True
        for step in steps:
            if not isinstance(step, dict):
                return False
            if not ("class" in step or "registry_name" in step):
                return False
        return True

    @classmethod
    def _suggest_processor_migration(cls, model_path, original_error):
        migration_command = (
            f"python src/lerobot/processor/migrate_policy_normalization.py "
            f"--pretrained-path {model_path}"
        )
        raise ProcessorMigrationError(model_path, migration_command, original_error)


def _build_step_class_map():
    """Map serialisation names to step classes for checkpoint loading."""
    from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
    from lerobot.processor.tokenizer_processor import ActionTokenizerProcessorStep, TokenizerProcessorStep
    from lerobot.policies.pi0.processor_pi0 import Pi0NewLineProcessor
    from lerobot.policies.pi0_fast.processor_pi0_fast import (
        Pi0FastPrepareStateAndLanguageTokenizerProcessorStep,
    )
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep

    return {
        "normalizer_processor": NormalizerProcessorStep,
        "unnormalizer_processor": UnnormalizerProcessorStep,
        "to_batch_processor": AddBatchDimStep,
        "device_processor": DeviceStep,
        "tokenizer_processor": TokenizerProcessorStep,
        "action_tokenizer_processor": ActionTokenizerProcessorStep,
        "rename_observations_processor": IdentityProcessorStep,
        "pi0_new_line_processor": Pi0NewLineProcessor,
        "pi0_fast_prepare_state_tokenizer_processor_step": Pi0FastPrepareStateAndLanguageTokenizerProcessorStep,
        "pi05_prepare_state_tokenizer_processor_step": Pi05PrepareStateTokenizerProcessorStep,
    }
