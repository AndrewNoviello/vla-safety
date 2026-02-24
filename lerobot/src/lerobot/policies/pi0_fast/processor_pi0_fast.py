from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
from lerobot.policies.pi0_fast.modeling_pi0_fast import pad_vector
from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.processor.pipeline import (
    AddBatchDimStep,
    DeviceStep,
    PolicyProcessorPipeline,
    ProcessorStep,
)
from lerobot.processor.tokenizer_processor import ActionTokenizerProcessorStep, TokenizerProcessorStep
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


@dataclass
class Pi0FastPrepareStateAndLanguageTokenizerProcessorStep(ProcessorStep):
    """Discretize state and build the PaliGemma language prompt (pi0_fast variant)."""

    _registry_name = "pi0_fast_prepare_state_tokenizer_processor_step"

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = dict(batch)

        state = batch.get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI0Fast")
        tasks = batch.get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in batch")

        state = deepcopy(state)
        state = pad_vector(state, self.max_state_dim)

        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\n"
            full_prompts.append(full_prompt)

        batch[self.task_key] = full_prompts
        return batch


def make_pi0_fast_pre_post_processors(
    config: PI0FastConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    input_steps: list[ProcessorStep] = [
        AddBatchDimStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi0FastPrepareStateAndLanguageTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name=config.text_tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        ActionTokenizerProcessorStep(
            action_tokenizer_name=config.action_tokenizer_name,
            max_action_tokens=config.max_action_tokens,
            fast_skip_tokens=config.fast_skip_tokens,
            paligemma_tokenizer_name=config.text_tokenizer_name,
        ),
        DeviceStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline(steps=input_steps, name=POLICY_PREPROCESSOR_DEFAULT_NAME),
        PolicyProcessorPipeline(steps=output_steps, name=POLICY_POSTPROCESSOR_DEFAULT_NAME),
    )


def preprocess(
    obs: dict[str, Any],
    task: str | list[str],
    preprocessor: PolicyProcessorPipeline,
) -> dict[str, Any]:
    """Preprocess a single observation for inference."""
    batch = dict(obs)
    batch["task"] = task
    return preprocessor(batch)


def postprocess(
    action: torch.Tensor,
    postprocessor: PolicyProcessorPipeline,
) -> torch.Tensor:
    """Unnormalize and move action to CPU."""
    result = postprocessor({ACTION: action})
    return result[ACTION]
