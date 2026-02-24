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

"""Processor step for moving transition tensors to a target device and dtype."""

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.utils.utils import get_safe_torch_device

from .core import EnvTransition, PolicyAction, TransitionKey
from .pipeline import ProcessorStep


@dataclass
class DeviceProcessorStep(ProcessorStep):
    """Move all tensors in an EnvTransition to a specified device and optionally cast dtype.

    Handles multi-GPU scenarios: if both the tensor and target are CUDA devices, the
    tensor's existing GPU placement is preserved (Accelerate compatibility).

    Attributes:
        device: Target device string (e.g. "cpu", "cuda", "cuda:0").
        float_dtype: Target float dtype string (e.g. "float32", "bfloat16"). If None,
                     dtype is unchanged.
    """

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
        """Move a single tensor to the target device and cast its dtype."""
        if tensor.is_cuda and self.tensor_device.type == "cuda":
            target_device = tensor.device
        else:
            target_device = self.tensor_device

        # MPS workaround: float64 is unsupported on MPS.
        if target_device.type == "mps" and tensor.dtype == torch.float64:
            tensor = tensor.to(dtype=torch.float32)

        if tensor.device != target_device:
            tensor = tensor.to(target_device, non_blocking=self.non_blocking)

        if self._target_float_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=self._target_float_dtype)

        return tensor

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Move all tensors in the transition to the configured device."""
        new_transition = transition.copy()

        action = new_transition.get(TransitionKey.ACTION)
        if action is not None and not isinstance(action, PolicyAction):
            raise ValueError(f"If action is not None it should be a PolicyAction type, got {type(action)}")

        for key in (TransitionKey.ACTION, TransitionKey.REWARD, TransitionKey.DONE, TransitionKey.TRUNCATED):
            value = transition.get(key)
            if isinstance(value, torch.Tensor):
                new_transition[key] = self._process_tensor(value)

        for key in (TransitionKey.OBSERVATION, TransitionKey.COMPLEMENTARY_DATA):
            data_dict = transition.get(key)
            if data_dict is not None:
                new_transition[key] = {
                    k: self._process_tensor(v) if isinstance(v, torch.Tensor) else v
                    for k, v in data_dict.items()
                }

        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"device": self.device, "float_dtype": self.float_dtype}
