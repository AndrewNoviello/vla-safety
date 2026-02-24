"""Tokenizer callables for VLA policy text tokenization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoTokenizer
else:
    AutoTokenizer = None


class TextTokenizer:
    """Tokenize task text from ``batch["task"]`` and write tokens into the batch.

    Writes ``observation.language.tokens`` and ``observation.language.attention_mask``
    (and subtask variants if ``batch["subtask"]`` is present).
    """

    def __init__(
        self,
        *,
        tokenizer_name: str | None = None,
        tokenizer: Any | None = None,
        max_length: int = 512,
        task_key: str = "task",
        padding_side: str = "right",
        padding: str = "max_length",
        truncation: bool = True,
    ):
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Install it with `pip install 'lerobot[transformers-dep]'`."
            )
        if tokenizer is not None:
            self._tokenizer = tokenizer
        elif tokenizer_name is not None:
            if AutoTokenizer is None:
                raise ImportError("AutoTokenizer is not available")
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            raise ValueError("Either 'tokenizer' or 'tokenizer_name' must be provided.")

        self.max_length = max_length
        self.task_key = task_key
        self.padding_side = padding_side
        self.padding = padding
        self.truncation = truncation

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = dict(batch)

        task = batch.get(self.task_key)
        if task is None:
            raise ValueError(f"Key '{self.task_key}' not found in batch.")
        if isinstance(task, str):
            task = [task]
        if not (isinstance(task, list) and all(isinstance(t, str) for t in task)):
            raise ValueError("Task must be a string or list of strings")

        tokenized = self._tokenize(task)
        target_device = self._detect_device(batch)
        if target_device is not None:
            tokenized = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenized.items()
            }

        batch[OBS_LANGUAGE_TOKENS] = tokenized["input_ids"]
        batch[OBS_LANGUAGE_ATTENTION_MASK] = tokenized["attention_mask"].to(dtype=torch.bool)

        subtask = batch.get("subtask")
        if subtask is not None:
            if isinstance(subtask, str):
                subtask = [subtask]
            tokenized_sub = self._tokenize(subtask)
            if target_device is not None:
                tokenized_sub = {
                    k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                    for k, v in tokenized_sub.items()
                }
            batch[OBS_LANGUAGE_SUBTASK_TOKENS] = tokenized_sub["input_ids"]
            batch[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK] = tokenized_sub["attention_mask"].to(
                dtype=torch.bool
            )

        return batch

    def _detect_device(self, batch: dict[str, Any]) -> torch.device | None:
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value.device
        return None

    def _tokenize(self, text: list[str]) -> dict[str, torch.Tensor]:
        return self._tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            padding_side=self.padding_side,
            return_tensors="pt",
        )
