"""Pre- and post-processing for the PI0-Fast policy."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
from lerobot.policies.pi0_fast.modeling_pi0_fast import pad_vector
from lerobot.processor.pipeline import (
    add_batch_dim,
    move_to_device,
    normalize,
    prepare_stats,
    unnormalize,
)
from lerobot.processor.tokenizer_processor import ActionTokenizer, TextTokenizer
from lerobot.utils.constants import ACTION, OBS_STATE


def _prepare_state_prompt(
    batch: dict[str, Any],
    max_state_dim: int = 32,
    task_key: str = "task",
) -> dict[str, Any]:
    """Discretize state and build the PaliGemma language prompt (pi0_fast variant)."""
    batch = dict(batch)
    state = batch.get(OBS_STATE)
    if state is None:
        raise ValueError("State is required for PI0Fast")
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
        prompts.append(f"Task: {cleaned}, State: {state_str};\n")

    batch[task_key] = prompts
    return batch


def make_pi0_fast_pre_post_processors(
    config: PI0FastConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[Callable[[dict[str, Any]], dict[str, Any]], Callable[[torch.Tensor], torch.Tensor]]:
    """Return ``(preprocess, postprocess)`` callables for PI0-Fast."""
    all_features = {**config.input_features, **config.output_features}
    output_features = dict(config.output_features)
    norm_map = dict(config.normalization_mapping)
    stats = prepare_stats(dataset_stats)
    device = config.device
    max_state_dim = config.max_state_dim

    text_tok = TextTokenizer(
        tokenizer_name=config.text_tokenizer_name,
        max_length=config.tokenizer_max_length,
        padding_side="right",
        padding="max_length",
    )
    action_tok = ActionTokenizer(
        action_tokenizer_name=config.action_tokenizer_name,
        max_action_tokens=config.max_action_tokens,
        fast_skip_tokens=config.fast_skip_tokens,
        paligemma_tokenizer_name=config.text_tokenizer_name,
    )

    def preprocess(batch: dict[str, Any]) -> dict[str, Any]:
        batch = add_batch_dim(batch)
        batch = normalize(batch, stats, all_features, norm_map)
        batch = _prepare_state_prompt(batch, max_state_dim=max_state_dim)
        batch = text_tok(batch)
        batch = action_tok(batch)
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
