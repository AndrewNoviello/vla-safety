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

"""Shared functional utilities for processor steps and policy processor files."""

from typing import Any

import torch

from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE


def add_batch_dim(obs: dict[str, Any]) -> dict[str, Any]:
    """Unsqueeze dim=0 on all state/image tensors; wrap a str 'task' value as a list.

    Args:
        obs: Observation dict. Keys follow the integer-indexed convention
             (e.g. ``"observation.images.0"``, ``"observation.state"``).

    Returns:
        A new dict with batch dimensions added.
    """
    result = dict(obs)

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

    if "task" in result and isinstance(result["task"], str):
        result["task"] = [result["task"]]

    return result


def images_to_chw_float(obs: dict[str, Any]) -> dict[str, Any]:
    """Convert image observations from uint8 HWC to float32 CHW in [0, 1].

    Operates on keys matching ``observation.image`` and ``observation.images.*``.

    Args:
        obs: Observation dict potentially containing uint8 HWC image tensors.

    Returns:
        A new dict with converted image tensors.
    """
    result = dict(obs)

    def _convert(val: torch.Tensor) -> torch.Tensor:
        if val.dtype == torch.uint8:
            val = val.float() / 255.0
        # HWC -> CHW (only for 3-D tensors; batched tensors are BHWC -> BCHW)
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
    """Move all tensors in a dict to the specified device.

    Args:
        data: Dict potentially containing torch.Tensor values (nested dicts are not traversed).
        device: Target device (e.g. ``"cuda"``, ``"cpu"``, or a ``torch.device``).

    Returns:
        A new dict with all tensors moved to ``device``.
    """
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
