from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


def _sharpness_jitter(img: torch.Tensor) -> torch.Tensor:
    """Randomly adjust sharpness. Samples a factor uniformly from [0.5, 1.5] (1.0 = no change).
    Input: tensor with shape [..., C, H, W], C in {1, 3}."""
    return F.adjust_sharpness(img, sharpness_factor=torch.empty(1).uniform_(0.5, 1.5).item())


def _apply_random_subset(
    img: Any,
    transforms: Sequence[Callable],
    weights: list[float],
    n_subset: int,
    random_order: bool,
) -> Any:
    """Apply a random subset of transforms to the image. Picks n_subset transforms (weighted),
    applies them in list order. Defined at module level so it can be pickled for DataLoader."""
    selected_indices = torch.multinomial(torch.tensor(weights), n_subset)
    if not random_order:
        selected_indices = selected_indices.sort().values
    for i in selected_indices:
        img = transforms[i](img)
    return img


def image_transforms() -> Callable:
    """Build the standard image augmentation pipeline for training.

    Returns a callable that applies 3 randomly chosen augmentations per image.
    Pool: brightness, contrast, saturation, hue, sharpness, affine. Each has equal weight.
    Picklable for DataLoader workers.
    """
    transforms_list = [
        v2.ColorJitter(brightness=(0.8, 1.2)),
        v2.ColorJitter(contrast=(0.8, 1.2)),
        v2.ColorJitter(saturation=(0.5, 1.5)),
        v2.ColorJitter(hue=(-0.05, 0.05)),
        _sharpness_jitter,
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
