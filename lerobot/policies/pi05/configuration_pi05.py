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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi05.pi05_constants import (
    PI05_DEFAULT_ACTION_EXPERT,
    PI05_DEFAULT_CHUNK_SIZE,
    PI05_DEFAULT_COMPILE_MODE,
    PI05_DEFAULT_COMPILE_MODEL,
    PI05_DEFAULT_DTYPE,
    PI05_DEFAULT_FREEZE_VISION_ENCODER,
    PI05_DEFAULT_GRADIENT_CHECKPOINTING,
    PI05_DEFAULT_IMAGE_RESOLUTION,
    PI05_DEFAULT_MAX_ACTION_DIM,
    PI05_DEFAULT_MAX_PERIOD,
    PI05_DEFAULT_MAX_STATE_DIM,
    PI05_DEFAULT_MIN_PERIOD,
    PI05_DEFAULT_N_ACTION_STEPS,
    PI05_DEFAULT_NUM_INFERENCE_STEPS,
    PI05_DEFAULT_OPTIMIZER_BETAS,
    PI05_DEFAULT_OPTIMIZER_EPS,
    PI05_DEFAULT_OPTIMIZER_GRAD_CLIP_NORM,
    PI05_DEFAULT_OPTIMIZER_LR,
    PI05_DEFAULT_OPTIMIZER_WEIGHT_DECAY,
    PI05_DEFAULT_PALIGEMMA,
    PI05_DEFAULT_SCHEDULER_DECAY_LR,
    PI05_DEFAULT_SCHEDULER_DECAY_STEPS,
    PI05_DEFAULT_SCHEDULER_WARMUP_STEPS,
    PI05_DEFAULT_TIME_SAMPLING_BETA_ALPHA,
    PI05_DEFAULT_TIME_SAMPLING_BETA_BETA,
    PI05_DEFAULT_TIME_SAMPLING_OFFSET,
    PI05_DEFAULT_TIME_SAMPLING_SCALE,
    PI05_DEFAULT_TOKENIZER_MAX_LENGTH,
    PI05_DEFAULT_TRAIN_EXPERT_ONLY,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = PI05_DEFAULT_PALIGEMMA
    action_expert_variant: str = PI05_DEFAULT_ACTION_EXPERT
    dtype: str = PI05_DEFAULT_DTYPE  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = PI05_DEFAULT_CHUNK_SIZE  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = PI05_DEFAULT_N_ACTION_STEPS  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = PI05_DEFAULT_MAX_STATE_DIM
    max_action_dim: int = PI05_DEFAULT_MAX_ACTION_DIM

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = PI05_DEFAULT_NUM_INFERENCE_STEPS
    time_sampling_beta_alpha: float = PI05_DEFAULT_TIME_SAMPLING_BETA_ALPHA
    time_sampling_beta_beta: float = PI05_DEFAULT_TIME_SAMPLING_BETA_BETA
    time_sampling_scale: float = PI05_DEFAULT_TIME_SAMPLING_SCALE
    time_sampling_offset: float = PI05_DEFAULT_TIME_SAMPLING_OFFSET
    min_period: float = PI05_DEFAULT_MIN_PERIOD
    max_period: float = PI05_DEFAULT_MAX_PERIOD

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = PI05_DEFAULT_IMAGE_RESOLUTION  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = PI05_DEFAULT_TOKENIZER_MAX_LENGTH  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = PI05_DEFAULT_GRADIENT_CHECKPOINTING  # Enable gradient checkpointing for memory optimization
    compile_model: bool = PI05_DEFAULT_COMPILE_MODEL  # Whether to use torch.compile for model optimization
    compile_mode: str = PI05_DEFAULT_COMPILE_MODE  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = PI05_DEFAULT_FREEZE_VISION_ENCODER  # Freeze only the vision encoder
    train_expert_only: bool = PI05_DEFAULT_TRAIN_EXPERT_ONLY  # Freeze entire VLM, train only action expert and projections

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = PI05_DEFAULT_OPTIMIZER_LR  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = PI05_DEFAULT_OPTIMIZER_BETAS
    optimizer_eps: float = PI05_DEFAULT_OPTIMIZER_EPS
    optimizer_weight_decay: float = PI05_DEFAULT_OPTIMIZER_WEIGHT_DECAY
    optimizer_grad_clip_norm: float = PI05_DEFAULT_OPTIMIZER_GRAD_CLIP_NORM

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = PI05_DEFAULT_SCHEDULER_WARMUP_STEPS
    scheduler_decay_steps: int = PI05_DEFAULT_SCHEDULER_DECAY_STEPS
    scheduler_decay_lr: float = PI05_DEFAULT_SCHEDULER_DECAY_LR

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
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
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
