"""Normalization / unnormalization processor steps with statistics management."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.utils.constants import ACTION

from .pipeline import PolicyProcessorPipeline, ProcessorStep, from_tensor_to_numpy, to_tensor


@dataclass
class _NormalizationMixin:
    """Mixin providing core normalization/unnormalization logic.

    When stats are explicitly provided during construction they are preserved even
    when ``load_state_dict()`` is called (checkpoint-override workflow).
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
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
        return self

    def state_dict(self) -> dict[str, Tensor]:
        flat: dict[str, Tensor] = {}
        for key, sub in self._tensor_stats.items():
            for stat_name, tensor in sub.items():
                flat[f"{key}.{stat_name}"] = tensor.cpu()
        return flat

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
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
        config: dict[str, Any] = {
            "eps": self.eps,
            "features": {
                key: {"type": ft.type.value, "shape": ft.shape} for key, ft in self.features.items()
            },
            "norm_map": {ft_type.value: norm_mode.value for ft_type, norm_mode in self.norm_map.items()},
        }
        if self.normalize_observation_keys is not None:
            config["normalize_observation_keys"] = sorted(self.normalize_observation_keys)
        return config

    def _normalize_batch_observations(self, batch: dict[str, Any], inverse: bool) -> dict[str, Any]:
        """Apply (un)normalization to observation keys in a flat batch dict."""
        result = dict(batch)
        for key, feature in self.features.items():
            if self.normalize_observation_keys is not None and key not in self.normalize_observation_keys:
                continue
            if feature.type != FeatureType.ACTION and key in result:
                tensor = torch.as_tensor(result[key])
                result[key] = self._apply_transform(tensor, key, feature.type, inverse=inverse)
        return result

    def _normalize_action(self, action: Tensor, inverse: bool) -> Tensor:
        return self._apply_transform(action, ACTION, FeatureType.ACTION, inverse=inverse)

    def _apply_transform(
        self, tensor: Tensor, key: str, feature_type: FeatureType, *, inverse: bool = False
    ) -> Tensor:
        norm_mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
        if norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:
            return tensor

        if norm_mode not in (NormalizationMode.MEAN_STD, NormalizationMode.MIN_MAX):
            raise ValueError(
                f"Unsupported normalization mode: {norm_mode}. "
                f"Supported modes: IDENTITY, MEAN_STD, MIN_MAX."
            )

        if key in self._tensor_stats:
            first_stat = next(iter(self._tensor_stats[key].values()))
            if first_stat.device != tensor.device or first_stat.dtype != tensor.dtype:
                self.to(device=tensor.device, dtype=tensor.dtype)

        stats = self._tensor_stats[key]

        if norm_mode == NormalizationMode.MEAN_STD:
            mean = stats.get("mean")
            std = stats.get("std")
            if mean is None or std is None:
                raise ValueError("MEAN_STD normalization requires 'mean' and 'std' stats.")
            if inverse:
                return tensor * std + mean
            return (tensor - mean) / (std + self.eps)

        if norm_mode == NormalizationMode.MIN_MAX:
            min_val = stats.get("min")
            max_val = stats.get("max")
            if min_val is None or max_val is None:
                raise ValueError("MIN_MAX normalization requires 'min' and 'max' stats.")
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
    """Normalizes observation and action keys in a batch dict."""

    _registry_name = "normalizer_processor"

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = self._normalize_batch_observations(batch, inverse=False)

        action = batch.get(ACTION)
        if action is not None:
            if not isinstance(action, torch.Tensor):
                raise ValueError(f"Action should be a torch.Tensor, got {type(action)}")
            batch = dict(batch)
            batch[ACTION] = self._normalize_action(action, inverse=False)

        return batch


@dataclass
class UnnormalizerProcessorStep(_NormalizationMixin, ProcessorStep):
    """Unnormalizes observation and action keys in a batch dict (inverse of NormalizerProcessorStep)."""

    _registry_name = "unnormalizer_processor"

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = self._normalize_batch_observations(batch, inverse=True)

        action = batch.get(ACTION)
        if action is not None:
            if not isinstance(action, torch.Tensor):
                raise ValueError(f"Action should be a torch.Tensor, got {type(action)}")
            batch = dict(batch)
            batch[ACTION] = self._normalize_action(action, inverse=True)

        return batch


def hotswap_stats(
    policy_processor: PolicyProcessorPipeline, stats: dict[str, dict[str, Any]]
) -> PolicyProcessorPipeline:
    """Replace normalization statistics in a pipeline (returns a deep copy)."""
    rp = deepcopy(policy_processor)
    for step in rp.steps:
        if isinstance(step, _NormalizationMixin):
            step.stats = stats
            step._tensor_stats = to_tensor(stats, device=step.device, dtype=step.dtype)
    return rp
