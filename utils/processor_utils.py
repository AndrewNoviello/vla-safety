from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from utils.types import FeatureType, PolicyFeature
from utils.constants import (
    ACTION,
    OBS_ENV_STATE,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
    pad_value: float | None = None,
) -> torch.Tensor:
    """Resize image to target size without distortion by padding.

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)
        pad_value: Value for padded regions. If None, uses 0 for uint8, -1.0 for float32.
            For [0,1] float images, use 0 so that after *2-1 scaling padded regions become -1.

    Returns:
        Resized and padded tensor with same shape format as input
    """
    if images.shape[-1] <= 4:
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)

    _, _, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    if pad_value is not None:
        constant_value = pad_value
    else:
        constant_value = 0 if images.dtype == torch.uint8 else -1.0

    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=constant_value,
    )

    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)

    return padded_images


def resize_images_in_batch(
    batch: dict[str, Any],
    image_resolution: tuple[int, int],
    all_features: dict[str, Any],
) -> dict[str, Any]:
    """Resize visual features in batch to image_resolution (aspect-ratio preserving with pad)."""
    height, width = image_resolution
    result = dict(batch)
    for key, feat in all_features.items():
        if key not in result or getattr(feat, "type", None) != FeatureType.VISUAL:
            continue
        tensor = result[key]
        if not isinstance(tensor, Tensor) or tensor.ndim < 3:
            continue
        # Check spatial dims: [B, C, H, W] or [B, H, W, C]
        if tensor.shape[1] == 3:
            h, w = tensor.shape[2], tensor.shape[3]
        else:
            h, w = tensor.shape[1], tensor.shape[2]
        if (h, w) != (height, width):
            result[key] = resize_with_pad_torch(
                tensor, height, width, pad_value=0.0
            )
    return result


def prepare_observation_for_inference(
    observation: dict[str, Any],
    device: str | torch.device,
    task: str = "",
    robot_type: str = "",
) -> dict[str, Any]:
    """Convert numpy observation to tensors, add task, and move to device."""
    result = dict(observation)
    result["task"] = task or ""
    for key, val in result.items():
        if key == "task":
            continue
        if isinstance(val, np.ndarray):
            t = torch.from_numpy(val).float()
            if t.ndim == 3 and t.shape[-1] == 3:
                t = t.permute(2, 0, 1).unsqueeze(0)
                t = t / 255.0
            elif t.ndim == 1:
                t = t.unsqueeze(0)
            result[key] = t.to(device)
        elif isinstance(val, torch.Tensor):
            if val.ndim == 3 and val.shape[-1] == 3:
                val = val.permute(2, 0, 1).unsqueeze(0).float() / 255.0
            elif val.ndim == 1:
                val = val.unsqueeze(0)
            result[key] = val.to(device)
    return result


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


def tokenize_batch(
    batch: dict[str, Any],
    tokenizer: Any,
    *,
    task_key: str = "task",
    max_length: int = 512,
    padding_side: str = "right",
    padding: str = "max_length",
    truncation: bool = True,
) -> dict[str, Any]:
    batch = dict(batch)

    task = batch.get(task_key)
    if task is None:
        raise ValueError(f"Key '{task_key}' not found in batch.")
    if isinstance(task, str):
        task = [task]
    if not (isinstance(task, list) and all(isinstance(t, str) for t in task)):
        raise ValueError("Task must be a string or list of strings")

    tokenized = tokenizer(
        task,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        padding_side=padding_side,
        return_tensors="pt",
    )

    batch[OBS_LANGUAGE_TOKENS] = tokenized["input_ids"]
    batch[OBS_LANGUAGE_ATTENTION_MASK] = tokenized["attention_mask"].to(dtype=torch.bool)

    subtask = batch.get("subtask")
    if subtask is not None:
        if isinstance(subtask, str):
            subtask = [subtask]
        tokenized_sub = tokenizer(
            subtask,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            padding_side=padding_side,
            return_tensors="pt",
        )
        batch[OBS_LANGUAGE_SUBTASK_TOKENS] = tokenized_sub["input_ids"]
        batch[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK] = tokenized_sub["attention_mask"].to(
            dtype=torch.bool
        )

    return batch


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
    result: dict[str, dict[str, torch.Tensor]] = {}
    for key, sub in raw_stats.items():
        result[key] = {}
        for k, v in sub.items():
            if v is None:
                continue
            if isinstance(v, np.ndarray):
                result[key][k] = torch.from_numpy(v).to(dtype=dtype, device=device)
            elif isinstance(v, torch.Tensor):
                result[key][k] = v.to(dtype=dtype, device=device)
            else:
                result[key][k] = torch.tensor(v, dtype=dtype, device=device)
    return result


def _ensure_stats_compat(
    stats: dict[str, dict[str, torch.Tensor | np.ndarray]],
    key: str,
    tensor: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the sub-dict for *key*, converting to tensors and moving to match *tensor* if needed."""
    sub = stats[key]
    device, dtype = tensor.device, tensor.dtype
    # Convert numpy to tensor and ensure device/dtype match
    result = {}
    for k, v in sub.items():
        if isinstance(v, np.ndarray):
            result[k] = torch.from_numpy(v).to(device=device, dtype=dtype)
        elif isinstance(v, torch.Tensor):
            result[k] = v.to(device=device, dtype=dtype) if v.device != device or v.dtype != dtype else v
        else:
            result[k] = torch.tensor(v, device=device, dtype=dtype)
    stats[key] = result
    return result


def normalize(
    batch: dict[str, Any],
    stats: dict[str, dict[str, torch.Tensor]] | None,
    features: dict[str, PolicyFeature],
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Normalize observation and action keys in *batch* (forward pass).

    Visual features are identity by default. State and action features use dataset
    mean/std by default.
    """
    stats = stats or {}
    result = dict(batch)
    for key, feature in features.items():
        if key not in result:
            continue
        tensor = result[key]
        if not isinstance(tensor, Tensor):
            continue
        if feature.type is FeatureType.VISUAL:
            continue
        if key not in stats or "mean" not in stats[key] or "std" not in stats[key]:
            raise KeyError(f"Missing mean/std normalization stats for feature '{key}'.")
        sub = _ensure_stats_compat(stats, key, tensor)
        result[key] = (tensor - sub["mean"]) / (sub["std"] + eps)
    return result


def unnormalize(
    batch: dict[str, Any],
    stats: dict[str, dict[str, torch.Tensor]] | None,
    features: dict[str, PolicyFeature],
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Unnormalize observation and action keys in *batch* (inverse pass)."""
    stats = stats or {}
    result = dict(batch)
    for key, feature in features.items():
        if key not in result or not isinstance(result[key], torch.Tensor):
            continue
        if feature.type is FeatureType.VISUAL:
            continue
        tensor = result[key]
        if key not in stats or "mean" not in stats[key] or "std" not in stats[key]:
            raise KeyError(f"Missing mean/std normalization stats for feature '{key}'.")
        sub = _ensure_stats_compat(stats, key, tensor)
        result[key] = tensor * sub["std"] + sub["mean"]
    return result
