#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Sequential data processing pipeline for VLA policy pre- and post-processing.

Core components:
- ProcessorStep: Abstract base class for a single transformation step.
- PolicyProcessorPipeline: Chains steps together and handles Hub save/load.
- IdentityProcessorStep: No-op step, useful as a placeholder.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from lerobot.utils.hub import HubMixin

from .converters import batch_to_transition, transition_to_batch
from .core import EnvTransition


class ProcessorStep(ABC):
    """Abstract base class for a single step in a data processing pipeline.

    Subclasses implement `__call__` to transform an `EnvTransition`.
    Stateful steps can additionally implement `state_dict` and `load_state_dict`
    to support Hub serialization of learned parameters (e.g. normalization stats).
    """

    @abstractmethod
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Transform the transition.

        Args:
            transition: Input data transition to be processed.

        Returns:
            The processed transition.
        """
        return transition

    def get_config(self) -> dict[str, Any]:
        """Return JSON-serializable configuration for checkpoint saving."""
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the step's state tensors (e.g. normalization statistics)."""
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load state from a state dictionary."""
        return None

    def reset(self) -> None:
        """Reset any internal state."""
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
    """No-op processor step that returns the transition unchanged."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        return transition


@dataclass
class PolicyProcessorPipeline(HubMixin):
    """Sequential pipeline for VLA policy pre/post-processing with Hub save/load support.

    Chains `ProcessorStep` instances in order and converts between the caller's
    data format and the internal `EnvTransition` wire format via `to_transition`
    and `to_output` callables.

    Typical use:
        preprocessor, postprocessor = make_pi0_pre_post_processors(config, stats)
        batch = preprocessor(obs_batch)          # dict[str, Tensor] -> dict[str, Tensor]
        action_raw = policy.forward(batch)
        action = postprocessor(action_raw)       # Tensor -> Tensor

    Attributes:
        steps: The processing steps executed in order.
        name: Descriptive name (used as filename prefix when saving).
        to_transition: Converts raw input to EnvTransition (default: batch_to_transition).
        to_output: Converts final EnvTransition to desired output (default: transition_to_batch).
    """

    steps: Sequence[ProcessorStep] = field(default_factory=list)
    name: str = "PolicyProcessorPipeline"

    to_transition: Callable[[Any], EnvTransition] = field(
        default_factory=lambda: cast(Callable[[Any], EnvTransition], batch_to_transition), repr=False
    )
    to_output: Callable[[EnvTransition], Any] = field(
        default_factory=lambda: cast(Callable[[EnvTransition], Any], transition_to_batch), repr=False
    )

    def __call__(self, data: Any) -> Any:
        """Run data through the full pipeline."""
        transition = self.to_transition(data)
        for step in self.steps:
            transition = step(transition)
        return self.to_output(transition)

    def __post_init__(self):
        """Validate that all steps inherit from ProcessorStep."""
        for i, step in enumerate(self.steps):
            if not isinstance(step, ProcessorStep):
                raise TypeError(f"Step {i} ({type(step).__name__}) must inherit from ProcessorStep")

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int | slice) -> ProcessorStep | PolicyProcessorPipeline:
        if isinstance(idx, slice):
            return PolicyProcessorPipeline(
                steps=self.steps[idx],
                name=self.name,
                to_transition=self.to_transition,
                to_output=self.to_output,
            )
        return self.steps[idx]

    def reset(self):
        """Reset all stateful steps."""
        for step in self.steps:
            if hasattr(step, "reset"):
                step.reset()

    def __repr__(self) -> str:
        step_names = [step.__class__.__name__ for step in self.steps]
        if not step_names:
            steps_repr = "steps=0: []"
        elif len(step_names) <= 3:
            steps_repr = f"steps={len(step_names)}: [{', '.join(step_names)}]"
        else:
            displayed = f"{step_names[0]}, {step_names[1]}, ..., {step_names[-1]}"
            steps_repr = f"steps={len(step_names)}: [{displayed}]"
        return f"PolicyProcessorPipeline({', '.join([f'name={self.name!r}', steps_repr])})"

    # ---- Hub save / load ----

    def _save_pretrained(self, save_directory: Path, **kwargs):
        """Save pipeline config JSON and per-step state safetensors files."""
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
        """Save pipeline to a directory or push to the Hugging Face Hub.

        Args:
            save_directory: Directory to save into. Defaults to HF_LEROBOT_HOME/processors/{name}.
            repo_id: Hub repo ID (only used when push_to_hub=True).
            push_to_hub: Whether to push to the Hub after saving.
            config_filename: Override the default JSON filename.
        """
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
        to_transition: Callable[[Any], EnvTransition] | None = None,
        to_output: Callable[[EnvTransition], Any] | None = None,
        **kwargs,
    ) -> PolicyProcessorPipeline:
        """Load a pipeline from a local directory, file, or Hugging Face Hub.

        Args:
            pretrained_model_name_or_path: Hub repo ID, local directory, or local file.
            config_filename: Name of the pipeline JSON config file (always required).
            overrides: Per-step config overrides keyed by registry name or class name.
            to_transition: Custom input conversion (default: batch_to_transition).
            to_output: Custom output conversion (default: transition_to_batch).

        Returns:
            Loaded PolicyProcessorPipeline instance.

        Raises:
            FileNotFoundError: If the config file cannot be found.
            ValueError: If the config format is invalid.
            ImportError: If a step class cannot be resolved.
            KeyError: If an override key doesn't match any step.
            ProcessorMigrationError: If the checkpoint needs migration.
        """
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
            to_transition=to_transition or cast(Callable[[Any], EnvTransition], batch_to_transition),
            to_output=to_output or cast(Callable[[EnvTransition], Any], transition_to_batch),
        )

    @classmethod
    def _load_config(
        cls,
        model_id: str,
        config_filename: str,
        hub_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], Path]:
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
    def _validate_loaded_config(
        cls, model_id: str, loaded_config: dict[str, Any], config_filename: str
    ) -> None:
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
    def _build_steps_with_overrides(
        cls,
        loaded_config: dict[str, Any],
        overrides: dict[str, Any],
        model_id: str,
        base_path: Path | None,
        hub_kwargs: dict[str, Any],
    ) -> tuple[list[ProcessorStep], set[str]]:
        steps: list[ProcessorStep] = []
        override_keys = set(overrides.keys())

        for step_entry in loaded_config["steps"]:
            step_class, step_key = cls._resolve_step_class(step_entry)
            step_instance = cls._instantiate_step(step_entry, step_class, step_key, overrides)
            cls._load_step_state(step_instance, step_entry, model_id, base_path, hub_kwargs)
            override_keys.discard(step_key)
            steps.append(step_instance)

        return steps, override_keys

    @classmethod
    def _resolve_step_class(cls, step_entry: dict[str, Any]) -> tuple[type[ProcessorStep], str]:
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
    def _instantiate_step(
        cls,
        step_entry: dict[str, Any],
        step_class: type[ProcessorStep],
        step_key: str,
        overrides: dict[str, Any],
    ) -> ProcessorStep:
        try:
            saved_cfg = step_entry.get("config", {})
            step_overrides = overrides.get(step_key, {})
            merged_cfg = {**saved_cfg, **step_overrides}
            return step_class(**merged_cfg)
        except Exception as e:
            step_name = step_entry.get("registry_name", step_entry.get("class", "Unknown"))
            raise ValueError(
                f"Failed to instantiate processor step '{step_name}' "
                f"with config {step_entry.get('config', {})}: {e}"
            ) from e

    @classmethod
    def _load_step_state(
        cls,
        step_instance: ProcessorStep,
        step_entry: dict[str, Any],
        model_id: str,
        base_path: Path | None,
        hub_kwargs: dict[str, Any],
    ) -> None:
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
    def _validate_overrides_used(
        cls, remaining_override_keys: set[str], loaded_config: dict[str, Any]
    ) -> None:
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
    def _should_suggest_migration(cls, model_path: Path) -> bool:
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
    def _is_processor_config(cls, config: dict) -> bool:
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
    def _suggest_processor_migration(cls, model_path: str | Path, original_error: str) -> None:
        migration_command = (
            f"python src/lerobot/processor/migrate_policy_normalization.py "
            f"--pretrained-path {model_path}"
        )
        raise ProcessorMigrationError(model_path, migration_command, original_error)


def _build_step_class_map() -> dict[str, type]:
    """Explicit class map for checkpoint deserialization.

    This replaces the old ProcessorStepRegistry. All known step classes are listed
    here by their serialization name (the ``_registry_name`` class attribute).
    To add a new step, append its entry here.
    """
    from lerobot.processor.batch_processor import AddBatchDimensionProcessorStep
    from lerobot.processor.device_processor import DeviceProcessorStep
    from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
    from lerobot.processor.rename_processor import RenameObservationsProcessorStep
    from lerobot.processor.tokenizer_processor import ActionTokenizerProcessorStep, TokenizerProcessorStep
    from lerobot.policies.pi0.processor_pi0 import Pi0NewLineProcessor
    from lerobot.policies.pi0_fast.processor_pi0_fast import (
        Pi0FastPrepareStateAndLanguageTokenizerProcessorStep,
    )
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.policies.groot.processor_groot import (
        GrootActionUnpackUnnormalizeStep,
        GrootEagleCollateStep,
        GrootEagleEncodeStep,
        GrootPackInputsStep,
    )

    return {
        "normalizer_processor": NormalizerProcessorStep,
        "unnormalizer_processor": UnnormalizerProcessorStep,
        "to_batch_processor": AddBatchDimensionProcessorStep,
        "device_processor": DeviceProcessorStep,
        "tokenizer_processor": TokenizerProcessorStep,
        "action_tokenizer_processor": ActionTokenizerProcessorStep,
        "rename_observations_processor": RenameObservationsProcessorStep,
        "pi0_new_line_processor": Pi0NewLineProcessor,
        "pi0_fast_prepare_state_tokenizer_processor_step": Pi0FastPrepareStateAndLanguageTokenizerProcessorStep,
        "pi05_prepare_state_tokenizer_processor_step": Pi05PrepareStateTokenizerProcessorStep,
        "groot_pack_inputs_v3": GrootPackInputsStep,
        "groot_eagle_encode_v3": GrootEagleEncodeStep,
        "groot_eagle_collate_v3": GrootEagleCollateStep,
        "groot_action_unpack_unnormalize_v1": GrootActionUnpackUnnormalizeStep,
    }
