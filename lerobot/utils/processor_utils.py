from __future__ import annotations

from typing import Any
import numpy as np
import torch
from torch import Tensor
from lerobot.types import FeatureType, NormalizationMode, PolicyFeature

from lerobot.utils.constants import (
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


def move_to_device(data: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    """Move all tensors in a flat dict to *device*."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}


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
    stats: dict[str, dict[str, torch.Tensor]],
    features: dict[str, PolicyFeature],
    norm_map: dict[FeatureType | str, NormalizationMode | str],
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Normalize observation and action keys in *batch* (forward pass).

    Uses MEAN_STD from stats for each key. Stats file should include mean/std for all
    features (e.g. ImageNet values for images, dataset stats for state/actions).
    """
    result = dict(batch)
    for key in features:
        if key not in result:
            continue
        tensor = result[key]
        if not isinstance(tensor, Tensor):
            continue
        if key in stats:
            sub = _ensure_stats_compat(stats, key, tensor)
            mean, std = sub["mean"], sub["std"]
            result[key] = (tensor - mean) / (std + eps)
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
    for key in features:
        if key not in result or not isinstance(result[key], torch.Tensor):
            continue
        if key not in stats:
            continue
        tensor = result[key]
        sub = _ensure_stats_compat(stats, key, tensor)
        mean, std = sub["mean"], sub["std"]
        result[key] = tensor * std + mean
    return result
