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

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.utils.constants import ACTION

from .converters import from_tensor_to_numpy, to_tensor
from .core import EnvTransition, PolicyAction, TransitionKey
from .pipeline import PolicyProcessorPipeline, ProcessorStep


@dataclass
class _NormalizationMixin:
    """Mixin providing core normalization/unnormalization logic for processor steps.

    Manages normalization statistics, converts them to tensors for efficient computation,
    and applies MEAN_STD or MIN_MAX normalization.

    **Stats Override Preservation:**
    When stats are explicitly provided during construction (e.g., via overrides in
    ``PolicyProcessorPipeline.from_pretrained()``), they are preserved even when
    ``load_state_dict()`` is called. This lets users override stats from a saved checkpoint.

    Attributes:
        features: Mapping of feature names to PolicyFeature objects.
        norm_map: Mapping from FeatureType to NormalizationMode.
        stats: Normalization statistics (mean, std, min, max) per feature key.
        device: PyTorch device for tensor operations.
        eps: Small value for numerical stability.
        normalize_observation_keys: If set, only normalize these observation keys.
    """

    features: dict[str, PolicyFeature]
    norm_map: dict[FeatureType, NormalizationMode]
    stats: dict[str, dict[str, Any]] | None = None
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    eps: float = 1e-8
    normalize_observation_keys: set[str] | None = None

    _tensor_stats: dict[str, dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)
    _stats_explicitly_provided: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        self._stats_explicitly_provided = self.stats is not None and bool(self.stats)

        # Robust JSON deserialization: enums may come back as strings.
        if self.features:
            first_val = next(iter(self.features.values()))
            if isinstance(first_val, dict):
                reconstructed = {}
                for key, ft_dict in self.features.items():
                    reconstructed[key] = PolicyFeature(
                        type=FeatureType(ft_dict["type"]), shape=tuple(ft_dict["shape"])
                    )
                self.features = reconstructed

        if self.norm_map and all(isinstance(k, str) for k in self.norm_map):
            self.norm_map = {
                FeatureType(k): NormalizationMode(v) for k, v in self.norm_map.items()
            }

        self.stats = self.stats or {}
        if self.dtype is None:
            self.dtype = torch.float32
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> _NormalizationMixin:
        """Move normalization stats to the specified device/dtype."""
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
        return self

    def state_dict(self) -> dict[str, Tensor]:
        """Return normalization statistics as a flat CPU state dictionary."""
        flat: dict[str, Tensor] = {}
        for key, sub in self._tensor_stats.items():
            for stat_name, tensor in sub.items():
                flat[f"{key}.{stat_name}"] = tensor.cpu()
        return flat

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Load normalization statistics from a state dictionary.

        If stats were explicitly provided during construction, they are preserved
        and this call is a no-op (supporting checkpoint-override workflows).
        """
        if self._stats_explicitly_provided and self.stats is not None:
            self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
            return

        self._tensor_stats.clear()
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.rsplit(".", 1)
            self._tensor_stats.setdefault(key, {})[stat_name] = tensor.to(
                dtype=torch.float32, device=self.device
            )

        self.stats = {}
        for key, tensor_dict in self._tensor_stats.items():
            self.stats[key] = {
                stat_name: from_tensor_to_numpy(tensor)
                for stat_name, tensor in tensor_dict.items()
            }

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""
        config = {
            "eps": self.eps,
            "features": {
                key: {"type": ft.type.value, "shape": ft.shape} for key, ft in self.features.items()
            },
            "norm_map": {ft_type.value: norm_mode.value for ft_type, norm_mode in self.norm_map.items()},
        }
        if self.normalize_observation_keys is not None:
            config["normalize_observation_keys"] = sorted(self.normalize_observation_keys)
        return config

    def _normalize_observation(self, observation: dict, inverse: bool) -> dict[str, Tensor]:
        """Apply (un)normalization to relevant features in an observation dict."""
        new_observation = dict(observation)
        for key, feature in self.features.items():
            if self.normalize_observation_keys is not None and key not in self.normalize_observation_keys:
                continue
            if feature.type != FeatureType.ACTION and key in new_observation:
                tensor = torch.as_tensor(new_observation[key])
                new_observation[key] = self._apply_transform(tensor, key, feature.type, inverse=inverse)
        return new_observation

    def _normalize_action(self, action: Tensor, inverse: bool) -> Tensor:
        """Apply (un)normalization to an action tensor."""
        return self._apply_transform(action, ACTION, FeatureType.ACTION, inverse=inverse)

    def _apply_transform(
        self, tensor: Tensor, key: str, feature_type: FeatureType, *, inverse: bool = False
    ) -> Tensor:
        """Apply normalization or unnormalization to a tensor.

        Supported modes:
          - IDENTITY: pass through unchanged.
          - MEAN_STD: center/scale with mean and std.
          - MIN_MAX: scale to [-1, 1] using min/max.

        Args:
            tensor: Input tensor.
            key: Feature key (for looking up stats).
            feature_type: FeatureType of the tensor.
            inverse: If True, apply inverse (unnormalization).

        Returns:
            Transformed tensor.

        Raises:
            ValueError: If normalization mode is unsupported or required stats are missing.
        """
        norm_mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
        if norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:
            return tensor

        if norm_mode not in (NormalizationMode.MEAN_STD, NormalizationMode.MIN_MAX):
            raise ValueError(
                f"Unsupported normalization mode: {norm_mode}. "
                f"Supported modes: IDENTITY, MEAN_STD, MIN_MAX."
            )

        # Ensure stats are on the same device and dtype as the input tensor.
        if self._tensor_stats and key in self._tensor_stats:
            first_stat = next(iter(self._tensor_stats[key].values()))
            if first_stat.device != tensor.device or first_stat.dtype != tensor.dtype:
                self.to(device=tensor.device, dtype=tensor.dtype)

        stats = self._tensor_stats[key]

        if norm_mode == NormalizationMode.MEAN_STD:
            mean = stats.get("mean")
            std = stats.get("std")
            if mean is None or std is None:
                raise ValueError(
                    "MEAN_STD normalization requires 'mean' and 'std' stats. "
                    "Ensure the dataset stats are correctly computed."
                )
            if inverse:
                return tensor * std + mean
            return (tensor - mean) / (std + self.eps)

        if norm_mode == NormalizationMode.MIN_MAX:
            min_val = stats.get("min")
            max_val = stats.get("max")
            if min_val is None or max_val is None:
                raise ValueError(
                    "MIN_MAX normalization requires 'min' and 'max' stats. "
                    "Ensure the dataset stats are correctly computed."
                )
            denom = max_val - min_val
            denom = torch.where(
                denom == 0,
                torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype),
                denom,
            )
            if inverse:
                return (tensor + 1) / 2 * denom + min_val
            return 2 * (tensor - min_val) / denom - 1

        return tensor


@dataclass
class NormalizerProcessorStep(_NormalizationMixin, ProcessorStep):
    """Processor step that normalizes observations and actions in a transition.

    Uses MEAN_STD or MIN_MAX normalization. Typically placed in the pre-processing
    pipeline before feeding data to a policy.
    """

    _registry_name = "normalizer_processor"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            new_transition[TransitionKey.OBSERVATION] = self._normalize_observation(
                observation, inverse=False
            )

        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")
        new_transition[TransitionKey.ACTION] = self._normalize_action(action, inverse=False)
        return new_transition


@dataclass
class UnnormalizerProcessorStep(_NormalizationMixin, ProcessorStep):
    """Processor step that unnormalizes observations and actions.

    Inverts the normalization applied by NormalizerProcessorStep. Typically placed in
    the post-processing pipeline to convert a policy's normalized output back to the
    original data scale.
    """

    _registry_name = "unnormalizer_processor"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            new_transition[TransitionKey.OBSERVATION] = self._normalize_observation(
                observation, inverse=True
            )

        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")
        new_transition[TransitionKey.ACTION] = self._normalize_action(action, inverse=True)
        return new_transition


def hotswap_stats(
    policy_processor: PolicyProcessorPipeline, stats: dict[str, dict[str, Any]]
) -> PolicyProcessorPipeline:
    """Replace normalization statistics in a PolicyProcessorPipeline.

    Creates a deep copy of the pipeline and updates stats on all normalization steps.
    Useful for adapting a pretrained policy to a different dataset without rebuilding
    the full pipeline.

    Args:
        policy_processor: The pipeline to modify.
        stats: New normalization statistics to apply.

    Returns:
        A new PolicyProcessorPipeline with updated statistics.
    """
    rp = deepcopy(policy_processor)
    for step in rp.steps:
        if isinstance(step, _NormalizationMixin):
            step.stats = stats
            step._tensor_stats = to_tensor(stats, device=step.device, dtype=step.dtype)
    return rp
