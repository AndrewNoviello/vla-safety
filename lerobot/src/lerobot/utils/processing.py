"""Functional normalization utilities replacing the processor pipeline.

Provides normalize/unnormalize for training batches and model outputs,
plus a device-move helper. ImageNet stats are always applied to VISUAL features.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_EPS = 1e-8


def _ensure_tensor_on(t: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if t.device != device or t.dtype != dtype:
        return t.to(device=device, dtype=dtype)
    return t


def _normalize_mean_std(
    tensor: Tensor, mean: Tensor, std: Tensor, *, inverse: bool
) -> Tensor:
    mean = _ensure_tensor_on(mean, device=tensor.device, dtype=tensor.dtype)
    std = _ensure_tensor_on(std, device=tensor.device, dtype=tensor.dtype)
    if inverse:
        return tensor * std + mean
    return (tensor - mean) / (std + _EPS)


def _normalize_min_max(
    tensor: Tensor, min_val: Tensor, max_val: Tensor, *, inverse: bool
) -> Tensor:
    min_val = _ensure_tensor_on(min_val, device=tensor.device, dtype=tensor.dtype)
    max_val = _ensure_tensor_on(max_val, device=tensor.device, dtype=tensor.dtype)
    denom = max_val - min_val
    denom = torch.where(
        denom == 0,
        torch.tensor(_EPS, device=tensor.device, dtype=tensor.dtype),
        denom,
    )
    if inverse:
        return (tensor + 1) / 2 * denom + min_val
    return 2 * (tensor - min_val) / denom - 1


def _apply_norm(
    tensor: Tensor,
    key: str,
    stats: dict[str, dict[str, Tensor]],
    mode: NormalizationMode,
    *,
    inverse: bool = False,
) -> Tensor:
    if mode == NormalizationMode.IDENTITY or key not in stats:
        return tensor
    key_stats = stats[key]
    if mode == NormalizationMode.MEAN_STD:
        return _normalize_mean_std(
            tensor,
            torch.as_tensor(key_stats["mean"]),
            torch.as_tensor(key_stats["std"]),
            inverse=inverse,
        )
    if mode == NormalizationMode.MIN_MAX:
        return _normalize_min_max(
            tensor,
            torch.as_tensor(key_stats["min"]),
            torch.as_tensor(key_stats["max"]),
            inverse=inverse,
        )
    raise ValueError(f"Unsupported normalization mode: {mode}")


def normalize(
    batch: dict,
    stats: dict[str, dict[str, Tensor]],
    features: dict[str, PolicyFeature],
    norm_map: dict[FeatureType | str, NormalizationMode],
) -> dict:
    """Normalize observations and actions in a batch dict.

    VISUAL features always get ImageNet mean/std.
    STATE and ACTION features use the mode from *norm_map* with dataset *stats*.
    """
    resolved_map: dict[FeatureType, NormalizationMode] = {
        FeatureType(k) if isinstance(k, str) else k: NormalizationMode(v) if isinstance(v, str) else v
        for k, v in norm_map.items()
    }

    out = dict(batch)
    for key, feat in features.items():
        if key not in out:
            continue
        tensor = out[key]
        if not isinstance(tensor, Tensor):
            continue

        if feat.type == FeatureType.VISUAL:
            mean = _ensure_tensor_on(IMAGENET_MEAN, device=tensor.device, dtype=tensor.dtype)
            std = _ensure_tensor_on(IMAGENET_STD, device=tensor.device, dtype=tensor.dtype)
            out[key] = (tensor - mean) / (std + _EPS)
        else:
            mode = resolved_map.get(feat.type, NormalizationMode.IDENTITY)
            out[key] = _apply_norm(tensor, key, stats, mode, inverse=False)

    return out


def unnormalize(
    actions: Tensor,
    stats: dict[str, dict[str, Tensor]],
    norm_mode: NormalizationMode | str,
    action_key: str = "action",
) -> Tensor:
    """Reverse normalization on model-output actions."""
    mode = NormalizationMode(norm_mode) if isinstance(norm_mode, str) else norm_mode
    return _apply_norm(actions, action_key, stats, mode, inverse=True)


def to_device(batch: dict, device: torch.device | str) -> dict:
    """Recursively move all tensors in a (possibly nested) dict to *device*."""
    out: dict = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = to_device(v, device)
        else:
            out[k] = v
    return out
