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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from lerobot.utils.utils import format_big_number


class AverageMeter:
    """
    Computes and stores the average and current value
    Adapted from https://github.com/pytorch/examples/blob/main/imagenet/main.py
    """

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:{avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


@dataclass
class MetricsTracker:
    """
    Track and log training metrics over time.

    Usage pattern:

    ```python
    metrics = {"loss": AverageMeter("loss", ":.3f")}
    train_tracker = MetricsTracker(batch_size, num_frames, num_episodes, metrics, steps=step)

    train_tracker.step()          # advance counters by one step
    train_tracker.loss = 0.42     # update a named AverageMeter
    logging.info(train_tracker)   # formatted string
    wandb.log(train_tracker.to_dict())
    train_tracker.reset_averages()
    ```
    """

    batch_size: int
    num_frames: int
    num_episodes: int
    metrics: dict[str, AverageMeter]
    steps: int = 0
    samples: int = 0
    episodes: float = 0.0
    epochs: float = 0.0
    accelerator: Callable | None = None

    def __post_init__(self) -> None:
        self._avg_samples_per_ep = self.num_frames / self.num_episodes
        # A sample is an (observation,action) pair; batch_size samples per step.
        self.samples = self.steps * self.batch_size
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self.num_frames

    def __setattr__(self, name: str, value: Any) -> None:
        # Route metric names to their AverageMeter; everything else is a normal attr.
        if name != "metrics" and hasattr(self, "metrics") and name in self.metrics:
            self.metrics[name].update(value)
        else:
            object.__setattr__(self, name, value)

    def step(self) -> None:
        """
        Updates metrics that depend on 'step' for one step.
        """
        self.steps += 1
        self.samples += self.batch_size
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self.num_frames

    def __str__(self) -> str:
        display_list = [
            f"step:{format_big_number(self.steps)}",
            # number of samples seen during training
            f"smpl:{format_big_number(self.samples)}",
            # number of episodes seen during training
            f"ep:{format_big_number(self.episodes)}",
            # number of time all unique samples are seen
            f"epch:{self.epochs:.2f}",
            *[str(m) for m in self.metrics.values()],
        ]
        return " ".join(display_list)

    def to_dict(self, use_avg: bool = True) -> dict[str, int | float]:
        """
        Returns the current metric values (or averages if `use_avg=True`) as a dict.
        """
        return {
            "steps": self.steps,
            "samples": self.samples,
            "episodes": self.episodes,
            "epochs": self.epochs,
            **{k: m.avg if use_avg else m.val for k, m in self.metrics.items()},
        }

    def reset_averages(self) -> None:
        """Resets average meters."""
        for m in self.metrics.values():
            m.reset()
