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


def default_transforms() -> dict[str, dict[str, Any]]:
    """Return the default transform configs.

    Each entry has: weight, type, kwargs.
    See https://pytorch.org/vision/0.18/auto_examples/transforms/plot_transforms_illustrations.html
    """
    return {
        "brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": (0.8, 1.2)}},
        "contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": (0.8, 1.2)}},
        "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": (0.5, 1.5)}},
        "hue": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": (-0.05, 0.05)}},
        "sharpness": {"weight": 1.0, "type": "SharpnessJitter", "kwargs": {"sharpness": (0.5, 1.5)}},
        "affine": {
            "weight": 1.0,
            "type": "RandomAffine",
            "kwargs": {"degrees": (-5.0, 5.0), "translate": (0.05, 0.05)},
        },
    }


def image_transforms_config(
    enable: bool = False,
    max_num_transforms: int = 3,
    random_order: bool = False,
    tfs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create image transforms config dict.

    Args:
        enable: Set to True to enable transforms during training.
        max_num_transforms: Max number of transforms (sampled) applied per frame. [1, len(tfs)].
        random_order: Apply transforms in random order when True.
        tfs: Transform configs. If None, uses default_transforms().
    """
    if tfs is None:
        tfs = default_transforms()
    return {
        "enable": enable,
        "max_num_transforms": max_num_transforms,
        "random_order": random_order,
        "tfs": tfs,
    }


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


def sharpness_jitter_fn(img: torch.Tensor, sharpness_min: float, sharpness_max: float) -> torch.Tensor:
    """Apply random sharpness jitter. Expects tensor input [..., 1 or 3, H, W]."""
    sharpness_factor = torch.empty(1).uniform_(sharpness_min, sharpness_max).item()
    return F.adjust_sharpness(img, sharpness_factor=sharpness_factor)


def sharpness_jitter(sharpness: float | Sequence[float]) -> Callable:
    """Return a callable that applies random sharpness jitter."""
    sharpness_min, sharpness_max = _parse_sharpness_range(sharpness)
    return partial(sharpness_jitter_fn, sharpness_min=sharpness_min, sharpness_max=sharpness_max)


def apply_random_subset(
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


def random_subset_apply(
    transforms: Sequence[Callable],
    p: list[float] | None = None,
    n_subset: int | None = None,
    random_order: bool = False,
) -> Callable:
    """Return a callable that applies a random subset of transforms."""
    if not isinstance(transforms, Sequence):
        raise TypeError("Argument transforms should be a sequence of callables")
    if p is None:
        p = [1] * len(transforms)
    elif len(p) != len(transforms):
        raise ValueError(
            f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
        )
    if n_subset is None:
        n_subset = len(transforms)
    elif not isinstance(n_subset, int):
        raise TypeError("n_subset should be an int or None")
    elif not (1 <= n_subset <= len(transforms)):
        raise ValueError(f"n_subset should be in the interval [1, {len(transforms)}]")

    total = sum(p)
    weights = [prob / total for prob in p]
    return partial(
        apply_random_subset,
        transforms=transforms,
        weights=weights,
        n_subset=n_subset,
        random_order=random_order,
    )


def make_transform_from_config(tf_cfg: dict[str, Any]) -> Callable:
    """Create a transform callable from a config dict with keys: weight, type, kwargs."""
    transform_type = tf_cfg.get("type", "Identity")
    kwargs = tf_cfg.get("kwargs", {})

    if transform_type == "SharpnessJitter":
        return sharpness_jitter(kwargs.get("sharpness", (0.5, 1.5)))

    transform_cls = getattr(v2, transform_type, None)
    if isinstance(transform_cls, type) and issubclass(transform_cls, Transform):
        return transform_cls(**kwargs)

    raise ValueError(
        f"Transform '{transform_type}' is not valid. It must be a class in "
        f"torchvision.transforms.v2 or 'SharpnessJitter'."
    )


def create_image_transforms(cfg: dict[str, Any]) -> Callable | None:
    """Create image transform callable from config dict, or None if disabled.

    Config dict has: enable, max_num_transforms, random_order, tfs.
    Returns None when enable is False or no transforms; otherwise returns a picklable callable.
    """
    if not cfg.get("enable", False):
        return None

    transforms_list = []
    weights_list = []
    for tf_cfg in cfg.get("tfs", {}).values():
        if tf_cfg.get("weight", 1.0) <= 0.0:
            continue
        transforms_list.append(make_transform_from_config(tf_cfg))
        weights_list.append(tf_cfg.get("weight", 1.0))

    n_subset = min(len(transforms_list), cfg.get("max_num_transforms", 3))
    if n_subset == 0:
        return None

    return random_subset_apply(
        transforms=transforms_list,
        p=weights_list,
        n_subset=n_subset,
        random_order=cfg.get("random_order", False),
    )
