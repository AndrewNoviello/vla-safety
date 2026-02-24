"""Processing utilities for VLA policy pre- and post-processing.

Pure functions for transforming batch dicts: tensor conversion, normalization,
device transfer, batch dimension handling, and image format conversion.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import singledispatch
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE


# ---------------------------------------------------------------------------
# Tensor conversion
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

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

    if ACTION in result:
        val = result[ACTION]
        if isinstance(val, torch.Tensor) and val.dim() == 1:
            result[ACTION] = val.unsqueeze(0)

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
    """Move all tensors in a flat dict to *device*."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def prepare_stats(
    raw_stats: dict[str, dict[str, Any]] | None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, dict[str, torch.Tensor]]:
    """Convert raw normalization stats (numpy / python) to tensors.

    Returns an empty dict when *raw_stats* is ``None``.
    """
    if not raw_stats:
        return {}
    return to_tensor(raw_stats, device=device, dtype=dtype)


def _ensure_stats_compat(
    stats: dict[str, dict[str, torch.Tensor]],
    key: str,
    tensor: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the sub-dict for *key*, moving tensors to match *tensor* if needed."""
    sub = stats[key]
    first = next(iter(sub.values()))
    if first.device != tensor.device or first.dtype != tensor.dtype:
        sub = {k: v.to(device=tensor.device, dtype=tensor.dtype) for k, v in sub.items()}
        stats[key] = sub
    return sub


def _apply_norm(
    tensor: torch.Tensor,
    key: str,
    feature_type: FeatureType,
    norm_map: dict[FeatureType, NormalizationMode],
    stats: dict[str, dict[str, torch.Tensor]],
    eps: float,
    *,
    inverse: bool,
) -> torch.Tensor:
    mode = norm_map.get(feature_type, NormalizationMode.IDENTITY)
    if mode == NormalizationMode.IDENTITY or key not in stats:
        return tensor

    sub = _ensure_stats_compat(stats, key, tensor)

    if mode == NormalizationMode.MEAN_STD:
        mean, std = sub["mean"], sub["std"]
        if inverse:
            return tensor * std + mean
        return (tensor - mean) / (std + eps)

    if mode == NormalizationMode.MIN_MAX:
        mn, mx = sub["min"], sub["max"]
        denom = mx - mn
        denom = torch.where(
            denom == 0,
            torch.tensor(eps, device=tensor.device, dtype=tensor.dtype),
            denom,
        )
        if inverse:
            return (tensor + 1) / 2 * denom + mn
        return 2 * (tensor - mn) / denom - 1

    raise ValueError(f"Unsupported normalization mode: {mode}")


def normalize(
    batch: dict[str, Any],
    stats: dict[str, dict[str, torch.Tensor]],
    features: dict[str, PolicyFeature],
    norm_map: dict[FeatureType, NormalizationMode],
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Normalize observation and action keys in *batch* (forward pass)."""
    if not stats:
        return batch
    result = dict(batch)
    for key, feat in features.items():
        if key in result and isinstance(result[key], torch.Tensor):
            result[key] = _apply_norm(result[key], key, feat.type, norm_map, stats, eps, inverse=False)
    return result


def unnormalize(
    batch: dict[str, Any],
    stats: dict[str, dict[str, torch.Tensor]],
    features: dict[str, PolicyFeature],
    norm_map: dict[FeatureType, NormalizationMode],
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Unnormalize observation and action keys in *batch* (inverse pass)."""
    if not stats:
        return batch
    result = dict(batch)
    for key, feat in features.items():
        if key in result and isinstance(result[key], torch.Tensor):
            result[key] = _apply_norm(result[key], key, feat.type, norm_map, stats, eps, inverse=True)
    return result
