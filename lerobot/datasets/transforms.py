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
import collections
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform, functional as F  # noqa: N812


def _parse_sharpness_range(sharpness: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(sharpness, (int, float)):
        if sharpness < 0:
            raise ValueError("If sharpness is a single number, it must be non negative.")
        sharpness = [1.0 - sharpness, 1.0 + sharpness]
        sharpness[0] = max(sharpness[0], 0.0)
    elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
        sharpness = [float(v) for v in sharpness]
    else:
        raise TypeError(f"{sharpness=} should be a single number or a sequence with length 2.")

    if not 0.0 <= sharpness[0] <= sharpness[1]:
        raise ValueError(f"sharpness values should be between (0., inf), but got {sharpness}.")

    return float(sharpness[0]), float(sharpness[1])


def _sharpness_jitter_fn(img: torch.Tensor, sharpness_min: float, sharpness_max: float) -> torch.Tensor:
    """Apply random sharpness jitter. Expects tensor input [..., 1 or 3, H, W]."""
    sharpness_factor = torch.empty(1).uniform_(sharpness_min, sharpness_max).item()
    return F.adjust_sharpness(img, sharpness_factor=sharpness_factor)


def _apply_random_subset(
    img: Any,
    transforms: Sequence[Callable],
    weights: list[float],
    n_subset: int,
    random_order: bool,
) -> Any:
    """Apply a random subset of transforms to the input. Module-level for picklability."""
    selected_indices = torch.multinomial(torch.tensor(weights), n_subset)
    if not random_order:
        selected_indices = selected_indices.sort().values
    for i in selected_indices:
        img = transforms[i](img)
    return img


def image_transforms() -> Callable:
    """Return the standard image augmentation callable.

    Fixed augmentations: brightness, contrast, saturation, hue, sharpness, affine.
    Applies a random subset of 3 transforms per image. Picklable for DataLoader.
    """
    sharpness_min, sharpness_max = _parse_sharpness_range((0.5, 1.5))
    transforms_list = [
        v2.ColorJitter(brightness=(0.8, 1.2)),
        v2.ColorJitter(contrast=(0.8, 1.2)),
        v2.ColorJitter(saturation=(0.5, 1.5)),
        v2.ColorJitter(hue=(-0.05, 0.05)),
        partial(_sharpness_jitter_fn, sharpness_min=sharpness_min, sharpness_max=sharpness_max),
        v2.RandomAffine(degrees=(-5.0, 5.0), translate=(0.05, 0.05)),
    ]
    weights = [1.0] * len(transforms_list)
    n_subset = 3
    return partial(
        _apply_random_subset,
        transforms=transforms_list,
        weights=weights,
        n_subset=n_subset,
        random_order=False,
    )
