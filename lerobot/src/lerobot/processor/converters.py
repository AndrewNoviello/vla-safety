# !/usr/bin/env python

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

from collections.abc import Sequence
from functools import singledispatch
from typing import Any

import numpy as np
import torch

from lerobot.utils.constants import ACTION, DONE, INFO, OBS_PREFIX, REWARD, TRUNCATED

from .core import EnvTransition, PolicyAction, RobotObservation, TransitionKey


@singledispatch
def to_tensor(
    value: Any,
    *,
    dtype: torch.dtype | None = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert various data types to PyTorch tensors with configurable options.

    Args:
        value: Input value to convert (tensor, array, scalar, sequence, etc.).
        dtype: Target tensor dtype. If None, preserves original dtype.
        device: Target device for the tensor.

    Returns:
        A PyTorch tensor.

    Raises:
        TypeError: If the input type is not supported.
    """
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
        scalar_value = value.item()
        return torch.tensor(scalar_value, dtype=dtype, device=device)
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


def _extract_complementary_data(batch: dict[str, Any]) -> dict[str, Any]:
    """Extract complementary data (task, indices, pad flags) from a batch dictionary."""
    pad_keys = {k: v for k, v in batch.items() if "_is_pad" in k}
    task_key = {"task": batch["task"]} if "task" in batch else {}
    subtask_key = {"subtask": batch["subtask"]} if "subtask" in batch else {}
    index_key = {"index": batch["index"]} if "index" in batch else {}
    task_index_key = {"task_index": batch["task_index"]} if "task_index" in batch else {}
    episode_index_key = {"episode_index": batch["episode_index"]} if "episode_index" in batch else {}
    return {**pad_keys, **task_key, **subtask_key, **index_key, **task_index_key, **episode_index_key}


def create_transition(
    observation: RobotObservation | None = None,
    action: PolicyAction | None = None,
    reward: float = 0.0,
    done: bool = False,
    truncated: bool = False,
    info: dict[str, Any] | None = None,
    complementary_data: dict[str, Any] | None = None,
) -> EnvTransition:
    """Create an EnvTransition dictionary with sensible defaults."""
    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: reward,
        TransitionKey.DONE: done,
        TransitionKey.TRUNCATED: truncated,
        TransitionKey.INFO: info if info is not None else {},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data if complementary_data is not None else {},
    }


def transition_to_policy_action(transition: EnvTransition) -> PolicyAction:
    """Convert an EnvTransition to a PolicyAction tensor."""
    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")
    action = transition.get(TransitionKey.ACTION)
    if not isinstance(action, PolicyAction):
        raise ValueError(f"Action should be a PolicyAction type got {type(action)}")
    return action


def policy_action_to_transition(action: PolicyAction) -> EnvTransition:
    """Convert a PolicyAction tensor to an EnvTransition."""
    if not isinstance(action, PolicyAction):
        raise ValueError(f"Action should be a PolicyAction type got {type(action)}")
    return create_transition(action=action)


def batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    """Convert a batch dictionary from a dataset/dataloader into an EnvTransition.

    Args:
        batch: A batch dictionary with observation.*, action, task, etc. keys.

    Returns:
        An EnvTransition dictionary.
    """
    if not isinstance(batch, dict):
        raise ValueError(f"EnvTransition must be a dictionary. Got {type(batch).__name__}")

    action = batch.get(ACTION)
    if action is not None and not isinstance(action, PolicyAction):
        raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

    observation_keys = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}
    complementary_data = _extract_complementary_data(batch)

    return create_transition(
        observation=observation_keys if observation_keys else None,
        action=batch.get(ACTION),
        reward=batch.get(REWARD, 0.0),
        done=batch.get(DONE, False),
        truncated=batch.get(TRUNCATED, False),
        info=batch.get("info", {}),
        complementary_data=complementary_data if complementary_data else None,
    )


def transition_to_batch(transition: EnvTransition) -> dict[str, Any]:
    """Convert an EnvTransition back to the canonical batch format used in LeRobot.

    This is the inverse of batch_to_transition.
    """
    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    batch = {
        ACTION: transition.get(TransitionKey.ACTION),
        REWARD: transition.get(TransitionKey.REWARD, 0.0),
        DONE: transition.get(TransitionKey.DONE, False),
        TRUNCATED: transition.get(TransitionKey.TRUNCATED, False),
        INFO: transition.get(TransitionKey.INFO, {}),
    }

    comp_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
    if comp_data:
        batch.update(comp_data)

    observation = transition.get(TransitionKey.OBSERVATION)
    if isinstance(observation, dict):
        batch.update(observation)

    return batch


def identity_transition(transition: EnvTransition) -> EnvTransition:
    """Identity function for transitions; returns the input unchanged."""
    return transition
