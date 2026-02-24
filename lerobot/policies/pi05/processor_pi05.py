"""Pre- and post-processing for the PI05 policy."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import pad_vector
from lerobot.processor.pipeline import (
    add_batch_dim,
    move_to_device,
    normalize,
    prepare_stats,
    unnormalize,
)
from lerobot.processor.tokenizer_processor import TextTokenizer
from lerobot.utils.constants import ACTION, OBS_STATE


def _prepare_state_prompt(
    batch: dict[str, Any],
    max_state_dim: int = 32,
    task_key: str = "task",
) -> dict[str, Any]:
    """Discretize state and build the PaliGemma language prompt."""
    batch = dict(batch)
    state = batch.get(OBS_STATE)
    if state is None:
        raise ValueError("State is required for PI05")
    tasks = batch.get(task_key)
    if tasks is None:
        raise ValueError("No task found in batch")

    state = pad_vector(deepcopy(state), max_state_dim)
    state_np = state.cpu().numpy()
    discretized = np.digitize(state_np, bins=np.linspace(-1, 1, 257)[:-1]) - 1

    prompts = []
    for i, task in enumerate(tasks):
        cleaned = task.strip().replace("_", " ").replace("\n", " ")
        state_str = " ".join(map(str, discretized[i]))
        prompts.append(f"Task: {cleaned}, State: {state_str};\nAction: ")

    batch[task_key] = prompts
    return batch


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[Callable[[dict[str, Any]], dict[str, Any]], Callable[[torch.Tensor], torch.Tensor]]:
    """Return ``(preprocess, postprocess)`` callables for PI05."""
    all_features = {**config.input_features, **config.output_features}
    output_features = dict(config.output_features)
    norm_map = dict(config.normalization_mapping)
    stats = prepare_stats(dataset_stats)
    device = config.device
    max_state_dim = config.max_state_dim

    tokenizer = TextTokenizer(
        tokenizer_name="google/paligemma-3b-pt-224",
        max_length=config.tokenizer_max_length,
        padding_side="right",
        padding="max_length",
    )

    def preprocess(batch: dict[str, Any]) -> dict[str, Any]:
        batch = add_batch_dim(batch)
        batch = normalize(batch, stats, all_features, norm_map)
        batch = _prepare_state_prompt(batch, max_state_dim=max_state_dim)
        batch = tokenizer(batch)
        batch = move_to_device(batch, device)
        return batch

    def postprocess(action: torch.Tensor) -> torch.Tensor:
        result = unnormalize({ACTION: action}, stats, output_features, norm_map)
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
