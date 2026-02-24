"""Tokenizer callables for VLA policy text and action tokenization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, AutoTokenizer
else:
    AutoProcessor = None
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


class ActionTokenizer:
    """Tokenize ``batch["action"]`` and write tokens / mask into the batch."""

    def __init__(
        self,
        *,
        action_tokenizer_name: str | None = None,
        action_tokenizer: Any | None = None,
        trust_remote_code: bool = True,
        max_action_tokens: int = 256,
        fast_skip_tokens: int = 128,
        paligemma_tokenizer_name: str = "google/paligemma-3b-pt-224",
    ):
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Install it with `pip install 'lerobot[transformers-dep]'`."
            )
        if action_tokenizer is not None:
            self._action_tokenizer = action_tokenizer
        elif action_tokenizer_name is not None:
            if AutoProcessor is None:
                raise ImportError("AutoProcessor is not available")
            self._action_tokenizer = AutoProcessor.from_pretrained(
                action_tokenizer_name, trust_remote_code=trust_remote_code
            )
        else:
            raise ValueError(
                "Either 'action_tokenizer' or 'action_tokenizer_name' must be provided."
            )

        self.max_action_tokens = max_action_tokens
        self.fast_skip_tokens = fast_skip_tokens
        self._paligemma_tokenizer = AutoTokenizer.from_pretrained(
            paligemma_tokenizer_name,
            trust_remote_code=trust_remote_code,
            add_eos_token=True,
            add_bos_token=False,
        )

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = dict(batch)
        action = batch.get(ACTION)
        if action is None:
            return batch

        tokens, mask = self._tokenize_action(action)
        batch[ACTION_TOKEN_MASK] = mask
        batch[ACTION_TOKENS] = tokens
        return batch

    def _act_tokens_to_paligemma_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._paligemma_tokenizer.vocab_size - 1 - self.fast_skip_tokens - tokens

    def _tokenize_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = action.device

        single_sample = action.dim() == 1
        if single_sample:
            action = action.unsqueeze(0)

        batch_size = action.shape[0]
        tokens_list = []
        masks_list = []

        for i in range(batch_size):
            action_cpu = action[i : i + 1].cpu()
            tokens = self._action_tokenizer(action_cpu)

            if isinstance(tokens, list) or not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            else:
                tokens = tokens.to(device=device)

            if tokens.dim() > 1:
                tokens = tokens.flatten()

            bos_id = self._paligemma_tokenizer.bos_token_id
            tokens = torch.cat(
                [
                    torch.tensor([bos_id], device=device),
                    torch.tensor(
                        self._paligemma_tokenizer.encode("Action: ", add_special_tokens=False),
                        device=device,
                    ),
                    self._act_tokens_to_paligemma_tokens(tokens),
                    torch.tensor(self._paligemma_tokenizer.encode("|"), device=device),
                ]
            )

            if len(tokens) > self.max_action_tokens:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max ({self.max_action_tokens}), truncating."
                )
                tokens = tokens[: self.max_action_tokens]
                mask = torch.ones(self.max_action_tokens, dtype=torch.bool, device=device)
            else:
                mask = torch.cat(
                    [
                        torch.ones(len(tokens), dtype=torch.bool, device=device),
                        torch.zeros(
                            self.max_action_tokens - len(tokens), dtype=torch.bool, device=device
                        ),
                    ]
                )
                tokens = torch.nn.functional.pad(
                    tokens, (0, self.max_action_tokens - len(tokens)), value=0
                )

            tokens_list.append(tokens)
            masks_list.append(mask)

        tokens_batch = torch.stack(tokens_list, dim=0)
        masks_batch = torch.stack(masks_list, dim=0)

        if single_sample:
            tokens_batch = tokens_batch.squeeze(0)
            masks_batch = masks_batch.squeeze(0)

        return tokens_batch.to(device), masks_batch.to(device)
