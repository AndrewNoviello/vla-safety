"""Pre- and post-processing for the PI0 policy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from transformers import AutoTokenizer

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.utils.constants import ACTION
from lerobot.utils.processor_utils import (
    add_batch_dim,
    move_to_device,
    normalize,
    prepare_stats,
    tokenize_batch,
    unnormalize,
)

# Hardcoded normalization for PI0
PI0_NORMALIZATION_MAPPING: dict[FeatureType, NormalizationMode] = {
    FeatureType.VISUAL: NormalizationMode.IDENTITY,
    FeatureType.STATE: NormalizationMode.MEAN_STD,
    FeatureType.ACTION: NormalizationMode.MEAN_STD,
}


def _ensure_newline(batch: dict[str, Any]) -> dict[str, Any]:
    """Ensure ``batch["task"]`` strings end with a newline (PaliGemma compat)."""
    task = batch.get("task")
    if task is None:
        return batch
    batch = dict(batch)
    if isinstance(task, str):
        batch["task"] = task if task.endswith("\n") else f"{task}\n"
    elif isinstance(task, list):
        batch["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
    return batch


def make_pi0_pre_post_processors(
    input_features: dict[str, PolicyFeature],
    output_features: dict[str, PolicyFeature],
    device: str,
    tokenizer_max_length: int = 48,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[Callable[[dict[str, Any]], dict[str, Any]], Callable[[torch.Tensor], torch.Tensor]]:
    """Return ``(preprocess, postprocess)`` callables for PI0."""
    all_features = {**input_features, **output_features}
    output_features_dict = dict(output_features)
    norm_map = dict(PI0_NORMALIZATION_MAPPING)
    stats = prepare_stats(dataset_stats)

    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    def preprocess(batch: dict[str, Any]) -> dict[str, Any]:
        batch = add_batch_dim(batch)
        batch = _ensure_newline(batch)
        batch = tokenize_batch(
            batch,
            tokenizer,
            max_length=tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        )
        batch = move_to_device(batch, device)
        batch = normalize(batch, stats, all_features, norm_map)
        return batch

    def postprocess(action: torch.Tensor) -> torch.Tensor:
        result = unnormalize({ACTION: action}, stats, output_features_dict, norm_map)
        return result[ACTION].to("cpu")

    return preprocess, postprocess


def preprocess(
    obs: dict[str, Any],
    task: str | list[str],
    preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Preprocess a single observation for inference."""
    batch = dict(obs)
    batch["task"] = task
    return preprocessor(batch)


def postprocess(
    action: torch.Tensor,
    postprocessor: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Unnormalize and move action to CPU."""
    return postprocessor(action)
