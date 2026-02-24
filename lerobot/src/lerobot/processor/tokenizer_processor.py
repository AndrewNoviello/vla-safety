"""Processor steps for tokenizing task descriptions and action sequences."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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

from .pipeline import ProcessorStep

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, AutoTokenizer
else:
    AutoProcessor = None
    AutoTokenizer = None


@dataclass
class TokenizerProcessorStep(ProcessorStep):
    """Tokenize task text from ``batch["task"]`` and write tokens into the batch.

    Writes ``observation.language.tokens`` and ``observation.language.attention_mask``
    (and subtask variants if ``batch["subtask"]`` is present).
    """

    _registry_name = "tokenizer_processor"

    tokenizer_name: str | None = None
    tokenizer: Any | None = None
    max_length: int = 512
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    input_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Install it with `pip install 'lerobot[transformers-dep]'`."
            )
        if self.tokenizer is not None:
            self.input_tokenizer = self.tokenizer
        elif self.tokenizer_name is not None:
            if AutoTokenizer is None:
                raise ImportError("AutoTokenizer is not available")
            self.input_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        else:
            raise ValueError("Either 'tokenizer' or 'tokenizer_name' must be provided.")

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = dict(batch)

        task = self._get_task(batch)
        if task is None:
            raise ValueError("Task cannot be None")

        tokenized_prompt = self._tokenize_text(task)
        target_device = self._detect_device(batch)

        if target_device is not None:
            tokenized_prompt = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenized_prompt.items()
            }

        batch[OBS_LANGUAGE_TOKENS] = tokenized_prompt["input_ids"]
        batch[OBS_LANGUAGE_ATTENTION_MASK] = tokenized_prompt["attention_mask"].to(dtype=torch.bool)

        subtask = self._get_subtask(batch)
        if subtask is not None:
            tokenized_subtask = self._tokenize_text(subtask)
            if target_device is not None:
                tokenized_subtask = {
                    k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                    for k, v in tokenized_subtask.items()
                }
            batch[OBS_LANGUAGE_SUBTASK_TOKENS] = tokenized_subtask["input_ids"]
            batch[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK] = tokenized_subtask["attention_mask"].to(
                dtype=torch.bool
            )

        return batch

    def _get_task(self, batch: dict[str, Any]) -> list[str] | None:
        task = batch.get(self.task_key)
        if task is None:
            raise ValueError(f"Key '{self.task_key}' not found in batch.")
        if isinstance(task, str):
            return [task]
        if isinstance(task, list) and all(isinstance(t, str) for t in task):
            return task
        return None

    def _get_subtask(self, batch: dict[str, Any]) -> list[str] | None:
        subtask = batch.get("subtask")
        if subtask is None:
            return None
        if isinstance(subtask, str):
            return [subtask]
        if isinstance(subtask, list) and all(isinstance(t, str) for t in subtask):
            return subtask
        return None

    def _detect_device(self, batch: dict[str, Any]) -> torch.device | None:
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value.device
        return None

    def _tokenize_text(self, text: str | list[str]) -> dict[str, torch.Tensor]:
        return self.input_tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            padding_side=self.padding_side,
            return_tensors="pt",
        )

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "max_length": self.max_length,
            "task_key": self.task_key,
            "padding_side": self.padding_side,
            "padding": self.padding,
            "truncation": self.truncation,
        }
        if self.tokenizer_name is not None and self.tokenizer is None:
            config["tokenizer_name"] = self.tokenizer_name
        return config


@dataclass
class ActionTokenizerProcessorStep(ProcessorStep):
    """Tokenize ``batch["action"]`` and write tokens / mask into the batch."""

    _registry_name = "action_tokenizer_processor"

    action_tokenizer_name: str | None = None
    action_tokenizer_input_object: Any | None = None
    trust_remote_code: bool = True
    max_action_tokens: int = 256
    fast_skip_tokens: int = 128
    paligemma_tokenizer_name: str = "google/paligemma-3b-pt-224"

    action_tokenizer: Any = field(default=None, init=False, repr=False)
    _paligemma_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Install it with `pip install 'lerobot[transformers-dep]'`."
            )
        if self.action_tokenizer_input_object is not None:
            self.action_tokenizer = self.action_tokenizer_input_object
        elif self.action_tokenizer_name is not None:
            if AutoProcessor is None:
                raise ImportError("AutoProcessor is not available")
            self.action_tokenizer = AutoProcessor.from_pretrained(
                self.action_tokenizer_name, trust_remote_code=self.trust_remote_code
            )
        else:
            raise ValueError(
                "Either 'action_tokenizer_input_object' or 'action_tokenizer_name' must be provided."
            )
        self._paligemma_tokenizer = AutoTokenizer.from_pretrained(
            self.paligemma_tokenizer_name,
            trust_remote_code=self.trust_remote_code,
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
        device = action.device if isinstance(action, torch.Tensor) else None

        single_sample = action.dim() == 1
        if single_sample:
            action = action.unsqueeze(0)

        batch_size = action.shape[0]
        tokens_list = []
        masks_list = []

        for i in range(batch_size):
            action_cpu = action[i : i + 1].cpu()
            tokens = self.action_tokenizer(action_cpu)

            if isinstance(tokens, list) or not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long, device=action.device)
            else:
                tokens = tokens.to(device=action.device)

            if tokens.dim() > 1:
                tokens = tokens.flatten()

            bos_id = self._paligemma_tokenizer.bos_token_id
            tokens = torch.cat(
                [
                    torch.tensor([bos_id], device=action.device),
                    torch.tensor(
                        self._paligemma_tokenizer.encode("Action: ", add_special_tokens=False),
                        device=action.device,
                    ),
                    self._act_tokens_to_paligemma_tokens(tokens),
                    torch.tensor(self._paligemma_tokenizer.encode("|"), device=action.device),
                ]
            )

            if len(tokens) > self.max_action_tokens:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self.max_action_tokens}), truncating."
                )
                tokens = tokens[: self.max_action_tokens]
                mask = torch.ones(self.max_action_tokens, dtype=torch.bool, device=action.device)
            else:
                mask = torch.cat(
                    [
                        torch.ones(len(tokens), dtype=torch.bool, device=action.device),
                        torch.zeros(
                            self.max_action_tokens - len(tokens), dtype=torch.bool, device=action.device
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

        if device is not None:
            tokens_batch = tokens_batch.to(device)
            masks_batch = masks_batch.to(device)

        return tokens_batch, masks_batch

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "max_action_tokens": self.max_action_tokens,
        }
        if self.action_tokenizer_name is not None and self.action_tokenizer_input_object is None:
            config["action_tokenizer_name"] = self.action_tokenizer_name
        return config
