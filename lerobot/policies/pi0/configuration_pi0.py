#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.policies.pi0.pi0_constants import (
    PI0_DEFAULT_ACTION_EXPERT,
    PI0_DEFAULT_CHUNK_SIZE,
    PI0_DEFAULT_COMPILE_MODE,
    PI0_DEFAULT_COMPILE_MODEL,
    PI0_DEFAULT_DTYPE,
    PI0_DEFAULT_FREEZE_VISION_ENCODER,
    PI0_DEFAULT_GRADIENT_CHECKPOINTING,
    PI0_DEFAULT_IMAGE_RESOLUTION,
    PI0_DEFAULT_MAX_ACTION_DIM,
    PI0_DEFAULT_MAX_PERIOD,
    PI0_DEFAULT_MAX_STATE_DIM,
    PI0_DEFAULT_MIN_PERIOD,
    PI0_DEFAULT_N_ACTION_STEPS,
    PI0_DEFAULT_NUM_INFERENCE_STEPS,
    PI0_DEFAULT_PALIGEMMA,
    PI0_DEFAULT_TIME_SAMPLING_BETA_ALPHA,
    PI0_DEFAULT_TIME_SAMPLING_BETA_BETA,
    PI0_DEFAULT_TIME_SAMPLING_OFFSET,
    PI0_DEFAULT_TIME_SAMPLING_SCALE,
    PI0_DEFAULT_TOKENIZER_MAX_LENGTH,
    PI0_DEFAULT_TRAIN_EXPERT_ONLY,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("pi0")
@dataclass
class PI0Config(PreTrainedConfig):
    paligemma_variant: str = PI0_DEFAULT_PALIGEMMA
    action_expert_variant: str = PI0_DEFAULT_ACTION_EXPERT
    dtype: str = PI0_DEFAULT_DTYPE

    n_obs_steps: int = 1
    chunk_size: int = PI0_DEFAULT_CHUNK_SIZE
    n_action_steps: int = PI0_DEFAULT_N_ACTION_STEPS

    max_state_dim: int = PI0_DEFAULT_MAX_STATE_DIM
    max_action_dim: int = PI0_DEFAULT_MAX_ACTION_DIM

    # Flow matching parameters: see openpi PI0Pytorch
    num_inference_steps: int = PI0_DEFAULT_NUM_INFERENCE_STEPS
    time_sampling_beta_alpha: float = PI0_DEFAULT_TIME_SAMPLING_BETA_ALPHA
    time_sampling_beta_beta: float = PI0_DEFAULT_TIME_SAMPLING_BETA_BETA
    time_sampling_scale: float = PI0_DEFAULT_TIME_SAMPLING_SCALE
    time_sampling_offset: float = PI0_DEFAULT_TIME_SAMPLING_OFFSET
    min_period: float = PI0_DEFAULT_MIN_PERIOD
    max_period: float = PI0_DEFAULT_MAX_PERIOD

    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = PI0_DEFAULT_IMAGE_RESOLUTION

    tokenizer_max_length: int = PI0_DEFAULT_TOKENIZER_MAX_LENGTH

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    gradient_checkpointing: bool = PI0_DEFAULT_GRADIENT_CHECKPOINTING
    compile_model: bool = PI0_DEFAULT_COMPILE_MODEL
    compile_mode: str = PI0_DEFAULT_COMPILE_MODE
    device: str | None = None

    freeze_vision_encoder: bool = PI0_DEFAULT_FREEZE_VISION_ENCODER
    train_expert_only: bool = PI0_DEFAULT_TRAIN_EXPERT_ONLY

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        """Pi0 does not add empty cameras or default features; validation is minimal."""
        pass

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
