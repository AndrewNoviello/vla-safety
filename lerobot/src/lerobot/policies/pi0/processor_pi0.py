from typing import Any

import torch

from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.processor.pipeline import (
    AddBatchDimStep,
    DeviceStep,
    PolicyProcessorPipeline,
    ProcessorStep,
)
from lerobot.processor.tokenizer_processor import TokenizerProcessorStep
from lerobot.utils.constants import ACTION, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


class Pi0NewLineProcessor(ProcessorStep):
    """Ensures that ``batch["task"]`` ends with a newline (PaliGemma compat)."""

    _registry_name = "pi0_new_line_processor"

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = dict(batch)
        task = batch.get("task")
        if task is None:
            return batch

        if isinstance(task, str):
            if not task.endswith("\n"):
                batch["task"] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            batch["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]

        return batch


def make_pi0_pre_post_processors(
    config: PI0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    input_steps: list[ProcessorStep] = [
        AddBatchDimStep(),
        Pi0NewLineProcessor(),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
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
