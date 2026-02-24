"""
Optimizer and scheduler presets, plus checkpoint save/load utilities.

Each preset function takes (params, num_training_steps) and returns
(optimizer, scheduler, grad_clip_norm).
"""

import logging
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from lerobot.datasets.utils import flatten_dict, unflatten_dict, write_json
from lerobot.utils.constants import (
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
    SCHEDULER_STATE,
)
from lerobot.utils.io_utils import deserialize_json_into_object


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

OptimizerParams = (
    Iterable[torch.nn.Parameter]
    | Iterable[dict[str, Any]]
)

OptimizerResult = tuple[torch.optim.Optimizer, LRScheduler | None, float]


# ---------------------------------------------------------------------------
# Cosine-warmup scheduler (used by every preset)
# ---------------------------------------------------------------------------

def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
    decay_steps: int,
    peak_lr: float,
    decay_lr: float,
) -> LambdaLR:
    """Cosine decay with linear warmup, auto-scaled to fit num_training_steps."""
    actual_warmup = warmup_steps
    actual_decay = decay_steps

    if num_training_steps < decay_steps:
        scale = num_training_steps / decay_steps
        actual_warmup = int(warmup_steps * scale)
        actual_decay = num_training_steps
        logging.info(
            f"Auto-scaling LR scheduler: "
            f"num_training_steps ({num_training_steps}) < decay_steps ({decay_steps}). "
            f"Scaling warmup: {warmup_steps} -> {actual_warmup}, "
            f"decay: {decay_steps} -> {actual_decay} "
            f"(scale factor: {scale:.3f})"
        )

    alpha = decay_lr / peak_lr

    def lr_lambda(step: int) -> float:
        if step < actual_warmup:
            if step <= 0:
                return 1 / (actual_warmup + 1)
            frac = 1 - step / actual_warmup
            return (1 / (actual_warmup + 1) - 1) * frac + 1
        clamped = min(step, actual_decay)
        cosine = 0.5 * (1 + math.cos(math.pi * clamped / actual_decay))
        return (1 - alpha) * cosine + alpha

    return LambdaLR(optimizer, lr_lambda, -1)


# ---------------------------------------------------------------------------
# Per-policy presets
# ---------------------------------------------------------------------------

def make_pi0_optimizer(params: OptimizerParams, num_training_steps: int) -> OptimizerResult:
    optimizer = torch.optim.AdamW(params, lr=2.5e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    scheduler = cosine_warmup_scheduler(
        optimizer, num_training_steps,
        warmup_steps=1_000, decay_steps=30_000, peak_lr=2.5e-5, decay_lr=2.5e-6,
    )
    return optimizer, scheduler, 1.0


def make_groot_optimizer(params: OptimizerParams, num_training_steps: int) -> OptimizerResult:
    lr = 1e-4
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-5)
    scheduler = cosine_warmup_scheduler(
        optimizer, num_training_steps,
        warmup_steps=500, decay_steps=10_000, peak_lr=lr, decay_lr=lr * 0.1,
    )
    return optimizer, scheduler, 10.0


PRESETS: dict[str, Any] = {
    "pi0": make_pi0_optimizer,
    "pi0_fast": make_pi0_optimizer,
    "pi05": make_pi0_optimizer,
    "groot": make_groot_optimizer,
}


# ---------------------------------------------------------------------------
# Factory (called by the training script)
# ---------------------------------------------------------------------------

def make_optimizer_and_scheduler(
    policy_type: str, params: OptimizerParams, num_training_steps: int
) -> OptimizerResult:
    """Build optimizer, scheduler, and grad_clip_norm for the given policy type."""
    if policy_type not in PRESETS:
        raise ValueError(
            f"No optimizer preset for policy type '{policy_type}'. "
            f"Available: {list(PRESETS.keys())}"
        )
    return PRESETS[policy_type](params, num_training_steps)


# ---------------------------------------------------------------------------
# Optimizer state save / load  (used by train_utils for checkpoint resume)
# ---------------------------------------------------------------------------

def save_optimizer_state(
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer], save_dir: Path
) -> None:
    """Save optimizer state to disk."""
    if isinstance(optimizer, dict):
        for name, opt in optimizer.items():
            optimizer_dir = save_dir / name
            optimizer_dir.mkdir(exist_ok=True, parents=True)
            _save_single_optimizer_state(opt, optimizer_dir)
    else:
        _save_single_optimizer_state(optimizer, save_dir)


def _save_single_optimizer_state(optimizer: torch.optim.Optimizer, save_dir: Path) -> None:
    state = optimizer.state_dict()
    param_groups = state.pop("param_groups")
    flat_state = flatten_dict(state)
    save_file(flat_state, save_dir / OPTIMIZER_STATE)
    write_json(param_groups, save_dir / OPTIMIZER_PARAM_GROUPS)


def load_optimizer_state(
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer], save_dir: Path
) -> torch.optim.Optimizer | dict[str, torch.optim.Optimizer]:
    """Load optimizer state from disk."""
    if isinstance(optimizer, dict):
        loaded_optimizers = {}
        for name, opt in optimizer.items():
            optimizer_dir = save_dir / name
            if optimizer_dir.exists():
                loaded_optimizers[name] = _load_single_optimizer_state(opt, optimizer_dir)
            else:
                loaded_optimizers[name] = opt
        return loaded_optimizers
    else:
        return _load_single_optimizer_state(optimizer, save_dir)


def _load_single_optimizer_state(optimizer: torch.optim.Optimizer, save_dir: Path) -> torch.optim.Optimizer:
    current_state_dict = optimizer.state_dict()
    flat_state = load_file(save_dir / OPTIMIZER_STATE)
    state = unflatten_dict(flat_state)

    if "state" in state:
        loaded_state_dict = {"state": {int(k): v for k, v in state["state"].items()}}
    else:
        loaded_state_dict = {"state": {}}

    if "param_groups" in current_state_dict:
        param_groups = deserialize_json_into_object(
            save_dir / OPTIMIZER_PARAM_GROUPS, current_state_dict["param_groups"]
        )
        loaded_state_dict["param_groups"] = param_groups

    optimizer.load_state_dict(loaded_state_dict)
    return optimizer


# ---------------------------------------------------------------------------
# Scheduler state save / load  (used by train_utils for checkpoint resume)
# ---------------------------------------------------------------------------

def save_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> None:
    state_dict = scheduler.state_dict()
    write_json(state_dict, save_dir / SCHEDULER_STATE)


def load_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> LRScheduler:
    state_dict = deserialize_json_into_object(save_dir / SCHEDULER_STATE, scheduler.state_dict())
    scheduler.load_state_dict(state_dict)
    return scheduler
