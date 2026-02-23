"""
Optimizer and scheduler presets for each policy type.

Each preset function takes (params, num_training_steps) and returns
(optimizer, scheduler, grad_clip_norm).
"""

import logging
import math
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


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


OptimizerParams = (
    Iterable[torch.nn.Parameter]
    | Iterable[dict[str, Any]]
)

OptimizerResult = tuple[torch.optim.Optimizer, LRScheduler | None, float]


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


def make_optimizer_and_scheduler(
    policy_type: str, params: OptimizerParams, num_training_steps: int
) -> OptimizerResult:
    if policy_type not in PRESETS:
        raise ValueError(
            f"No optimizer preset for policy type '{policy_type}'. "
            f"Available: {list(PRESETS.keys())}"
        )
    return PRESETS[policy_type](params, num_training_steps)
