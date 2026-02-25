#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.datasets.utils import flatten_dict, load_json, unflatten_dict, write_json
from lerobot.optim import load_optimizer_state, load_scheduler_state, save_optimizer_state, save_scheduler_state
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    RNG_STATE,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)


def _serialize_python_rng_state() -> dict[str, torch.Tensor]:
    py_state = random.getstate()
    return {
        "py_rng_version": torch.tensor([py_state[0]], dtype=torch.int64),
        "py_rng_state": torch.tensor(py_state[1], dtype=torch.int64),
    }


def _deserialize_python_rng_state(rng_state_dict: dict[str, torch.Tensor]) -> None:
    py_state = (rng_state_dict["py_rng_version"].item(), tuple(rng_state_dict["py_rng_state"].tolist()), None)
    random.setstate(py_state)


def _serialize_numpy_rng_state() -> dict[str, torch.Tensor]:
    np_state = np.random.get_state()
    assert np_state[0] == "MT19937"
    return {
        "np_rng_state_values": torch.tensor(np_state[1], dtype=torch.int64),
        "np_rng_state_index": torch.tensor([np_state[2]], dtype=torch.int64),
        "np_rng_has_gauss": torch.tensor([np_state[3]], dtype=torch.int64),
        "np_rng_cached_gaussian": torch.tensor([np_state[4]], dtype=torch.float32),
    }


def _deserialize_numpy_rng_state(rng_state_dict: dict[str, torch.Tensor]) -> None:
    np_state = (
        "MT19937",
        rng_state_dict["np_rng_state_values"].numpy(),
        rng_state_dict["np_rng_state_index"].item(),
        rng_state_dict["np_rng_has_gauss"].item(),
        rng_state_dict["np_rng_cached_gaussian"].item(),
    )
    np.random.set_state(np_state)


def _serialize_torch_rng_state() -> dict[str, torch.Tensor]:
    torch_rng_state_dict = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        torch_rng_state_dict["torch_cuda_rng_state"] = torch.cuda.get_rng_state()
    return torch_rng_state_dict


def _deserialize_torch_rng_state(rng_state_dict: dict[str, torch.Tensor]) -> None:
    torch.set_rng_state(rng_state_dict["torch_rng_state"])
    if torch.cuda.is_available() and "torch_cuda_rng_state" in rng_state_dict:
        torch.cuda.set_rng_state(rng_state_dict["torch_cuda_rng_state"])


def save_rng_state(save_dir: Path) -> None:
    rng_state_dict = {
        **_serialize_python_rng_state(),
        **_serialize_numpy_rng_state(),
        **_serialize_torch_rng_state(),
    }
    save_file(flatten_dict(rng_state_dict), save_dir / RNG_STATE)


def load_rng_state(save_dir: Path) -> None:
    flat_rng_state_dict = load_file(save_dir / RNG_STATE)
    rng_state_dict = unflatten_dict(flat_rng_state_dict)
    py_rng_state_dict = {k: v for k, v in rng_state_dict.items() if k.startswith("py")}
    np_rng_state_dict = {k: v for k, v in rng_state_dict.items() if k.startswith("np")}
    torch_rng_state_dict = {k: v for k, v in rng_state_dict.items() if k.startswith("torch")}
    _deserialize_python_rng_state(py_rng_state_dict)
    _deserialize_numpy_rng_state(np_rng_state_dict)
    _deserialize_torch_rng_state(torch_rng_state_dict)


def set_seed(seed, accelerator: Callable | None = None) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if accelerator:
        from accelerate.utils import set_seed as _accelerate_set_seed

        _accelerate_set_seed(seed)


def get_step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns the checkpoint sub-directory corresponding to the step number."""
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def save_training_step(step: int, save_dir: Path) -> None:
    write_json({"step": step}, save_dir / TRAINING_STEP)


def load_training_step(save_dir: Path) -> int:
    training_step = load_json(save_dir / TRAINING_STEP)
    return training_step["step"]


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink():
        last_checkpoint_dir.unlink()
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    is_peft: bool = False,
) -> None:
    """Save policy weights and optimizer/scheduler state."""
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir)
    save_training_state(checkpoint_dir, step, optimizer, scheduler)


def save_training_state(
    checkpoint_dir: Path,
    train_step: int,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> None:
    """
    Saves the training step, optimizer state, scheduler state, and rng state.

    Args:
        save_dir (Path): The directory to save artifacts to.
        train_step (int): Current training step.
        optimizer (Optimizer | None, optional): The optimizer from which to save the state_dict.
            Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler from which to save the state_dict.
            Defaults to None.
    """
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(train_step, save_dir)
    save_rng_state(save_dir)
    if optimizer is not None:
        save_optimizer_state(optimizer, save_dir)
    if scheduler is not None:
        save_scheduler_state(scheduler, save_dir)


def load_training_state(
    checkpoint_dir: Path, optimizer: Optimizer, scheduler: LRScheduler | None
) -> tuple[int, Optimizer, LRScheduler | None]:
    """
    Loads the training step, optimizer state, scheduler state, and rng state.
    This is used to resume a training run.

    Args:
        checkpoint_dir (Path): The checkpoint directory. Should contain a 'training_state' dir.
        optimizer (Optimizer): The optimizer to load the state_dict to.
        scheduler (LRScheduler | None): The scheduler to load the state_dict to (can be None).

    Raises:
        NotADirectoryError: If 'checkpoint_dir' doesn't contain a 'training_state' dir

    Returns:
        tuple[int, Optimizer, LRScheduler | None]: training step, optimizer and scheduler with their
            state_dict loaded.
    """
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)

    load_rng_state(training_state_dir)
    step = load_training_step(training_state_dir)
    optimizer = load_optimizer_state(optimizer, training_state_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, training_state_dir)

    return step, optimizer, scheduler
