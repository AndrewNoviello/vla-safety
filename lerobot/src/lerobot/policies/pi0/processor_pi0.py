#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from typing import Any

import torch

from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


class Pi0NewLineProcessor(ProcessorStep):
    """
    Ensures that the task description string ends with a newline character.

    This processing step is required for compatibility with the PaliGemma tokenizer,
    which expects a newline at the end of the text prompt. It handles both single
    strings and lists of strings for the 'task' key in complementary data.
    """

    _registry_name = "pi0_new_line_processor"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        comp = new_transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if comp is None:
            return new_transition

        comp = dict(comp)
        if "task" not in comp or comp["task"] is None:
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp
            return new_transition

        task = comp["task"]
        # Handle both string and list of strings
        if isinstance(task, str):
            if not task.endswith("\n"):
                comp["task"] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            comp["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
        # If task is neither string nor list of strings, leave unchanged

        new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return new_transition


def make_pi0_pre_post_processors(
    config: PI0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Adding a batch dimension.
    2. Appending a newline character to the task description for tokenizer compatibility.
    3. Tokenizing the text prompt using the PaliGemma tokenizer.
    4. Moving all data to the specified device.
    5. Normalizing input and output features based on dataset statistics.

    The post-processing pipeline handles the model's output by:
    1. Unnormalizing the output features to their original scale.
    2. Moving data to the CPU.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    input_steps: list[ProcessorStep] = [
        AddBatchDimensionProcessorStep(),
        Pi0NewLineProcessor(),  # Add newlines before tokenization for PaliGemma
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline(
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline(
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def preprocess(
    obs: dict[str, Any],
    task: str | list[str],
    preprocessor: PolicyProcessorPipeline,
) -> dict[str, Any]:
    """Preprocess a single observation for inference.

    Build the preprocessor once with make_pi0_pre_post_processors() and pass it here.

    Args:
        obs: Observation dict with integer-indexed camera keys, e.g.:
             {"observation.images.0": Tensor(H,W,C), "observation.state": Tensor(D)}
        task: Natural language instruction (string or batched list).
        preprocessor: Pipeline built by make_pi0_pre_post_processors().

    Returns:
        Batch dict ready for model forward pass.
    """
    batch = dict(obs)
    batch["task"] = task
    return preprocessor(batch)


def postprocess(
    action: torch.Tensor,
    postprocessor: PolicyProcessorPipeline,
) -> torch.Tensor:
    """Unnormalize and move model action output to CPU.

    Args:
        action: Raw action tensor from model forward pass.
        postprocessor: Pipeline built by make_pi0_pre_post_processors().

    Returns:
        Unnormalized action tensor on CPU.
    """
    return postprocessor(action)
