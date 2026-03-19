from __future__ import annotations

from typing import Any

import torch

from transformers import AutoTokenizer

from utils.types import PolicyFeature
from utils.constants import ACTION
from utils.processor_utils import (
    add_batch_dim,
    normalize,
    to_device,
    tokenize_batch,
    unnormalize,
)


def _ensure_newline(batch: dict[str, Any]) -> dict[str, Any]:
    """Ensure batch['task'] strings end with a newline (PaliGemma compat)."""
    task = batch.get("task")
    if task is None:
        return batch
    batch = dict(batch)
    if isinstance(task, str):
        batch["task"] = task if task.endswith("\n") else f"{task}\n"
    elif isinstance(task, list):
        batch["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
    return batch


def preprocess_pi0(
    batch: dict[str, Any],
    stats: dict[str, dict[str, Any]] | None,
    all_features: dict[str, PolicyFeature],
    norm_map: dict[str, Any],
    tokenizer: AutoTokenizer,
    device: str | torch.device,
    *,
    max_length: int = 48,
    add_batch_dim: bool = False,
) -> dict[str, Any]:
    """Preprocess a batch for PI0 (training or inference)."""
    stats = stats or {}
    if add_batch_dim:
        batch = add_batch_dim(batch)
    batch = _ensure_newline(batch)
    batch = tokenize_batch(
        batch,
        tokenizer,
        max_length=max_length,
        padding_side="right",
        padding="max_length",
    )
    batch = to_device(batch, device)
    batch = normalize(batch, stats, all_features, norm_map)
    return batch


def postprocess_pi0(
    action: torch.Tensor,
    stats: dict[str, dict[str, Any]],
    output_features: dict[str, PolicyFeature],
    norm_map: dict[str, Any],
) -> torch.Tensor:
    """Postprocess PI0 action output (unnormalize and move to CPU)."""
    result = unnormalize({ACTION: action}, stats, output_features, norm_map)
    return result[ACTION].to("cpu")
