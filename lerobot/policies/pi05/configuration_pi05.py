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

"""PI05 configuration and architecture constants."""

import builtins
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar

from lerobot.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.rtc.configuration_rtc import RTCConfig

# PI05 architecture and flow-matching constants
DEFAULT_IMAGE_SIZE = 224
PI05_DEFAULT_PALIGEMMA = "gemma_2b"
PI05_DEFAULT_ACTION_EXPERT = "gemma_300m"
PI05_DEFAULT_CHUNK_SIZE = 50
PI05_DEFAULT_N_ACTION_STEPS = 50
PI05_DEFAULT_MAX_STATE_DIM = 32
PI05_DEFAULT_MAX_ACTION_DIM = 32
PI05_DEFAULT_IMAGE_RESOLUTION = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
PI05_DEFAULT_DTYPE = "float32"
PI05_DEFAULT_NUM_INFERENCE_STEPS = 10
PI05_DEFAULT_TIME_SAMPLING_BETA_ALPHA = 1.5
PI05_DEFAULT_TIME_SAMPLING_BETA_BETA = 1.0
PI05_DEFAULT_TIME_SAMPLING_SCALE = 0.999
PI05_DEFAULT_TIME_SAMPLING_OFFSET = 0.001
PI05_DEFAULT_MIN_PERIOD = 4e-3
PI05_DEFAULT_MAX_PERIOD = 4.0
PI05_DEFAULT_GRADIENT_CHECKPOINTING = False
PI05_DEFAULT_COMPILE_MODEL = False
PI05_DEFAULT_COMPILE_MODE = "max-autotune"
PI05_DEFAULT_FREEZE_VISION_ENCODER = False
PI05_DEFAULT_TRAIN_EXPERT_ONLY = False
PI05_DEFAULT_OPTIMIZER_LR = 2.5e-5
PI05_DEFAULT_OPTIMIZER_BETAS = (0.9, 0.95)
PI05_DEFAULT_OPTIMIZER_EPS = 1e-8
PI05_DEFAULT_OPTIMIZER_WEIGHT_DECAY = 0.01
PI05_DEFAULT_OPTIMIZER_GRAD_CLIP_NORM = 1.0
PI05_DEFAULT_SCHEDULER_WARMUP_STEPS = 1_000
PI05_DEFAULT_SCHEDULER_DECAY_STEPS = 30_000
PI05_DEFAULT_SCHEDULER_DECAY_LR = 2.5e-6
PI05_DEFAULT_TOKENIZER_MAX_LENGTH = 200
from lerobot.utils.config_utils import load_config_from_checkpoint, save_config
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.utils import auto_select_torch_device, is_amp_available, is_torch_device_available

logger = getLogger(__name__)


@dataclass
class PI05Config:
    # The registered name for this policy type — used in config.json and the registry.
    type: ClassVar[str] = "pi05"

    # ------------------------------------------------------------------ #
    # Shared fields (common to all policy configs)                        #
    # ------------------------------------------------------------------ #
    n_obs_steps: int = 1

    input_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] | None = field(default_factory=dict)

    device: str | None = None
    use_amp: bool = False

    use_peft: bool = False
    pretrained_path: Path | None = None

    repo_id: str | None = None
    private: bool | None = None
    tags: list[str] | None = None
    license: str | None = None

    # ------------------------------------------------------------------ #
    # PI05-specific fields                                                #
    # ------------------------------------------------------------------ #
    paligemma_variant: str = PI05_DEFAULT_PALIGEMMA
    action_expert_variant: str = PI05_DEFAULT_ACTION_EXPERT
    dtype: str = PI05_DEFAULT_DTYPE

    chunk_size: int = PI05_DEFAULT_CHUNK_SIZE
    n_action_steps: int = PI05_DEFAULT_N_ACTION_STEPS

    max_state_dim: int = PI05_DEFAULT_MAX_STATE_DIM
    max_action_dim: int = PI05_DEFAULT_MAX_ACTION_DIM

    # Flow matching parameters
    num_inference_steps: int = PI05_DEFAULT_NUM_INFERENCE_STEPS
    time_sampling_beta_alpha: float = PI05_DEFAULT_TIME_SAMPLING_BETA_ALPHA
    time_sampling_beta_beta: float = PI05_DEFAULT_TIME_SAMPLING_BETA_BETA
    time_sampling_scale: float = PI05_DEFAULT_TIME_SAMPLING_SCALE
    time_sampling_offset: float = PI05_DEFAULT_TIME_SAMPLING_OFFSET
    min_period: float = PI05_DEFAULT_MIN_PERIOD
    max_period: float = PI05_DEFAULT_MAX_PERIOD

    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = PI05_DEFAULT_IMAGE_RESOLUTION

    # Add empty images when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = PI05_DEFAULT_TOKENIZER_MAX_LENGTH

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    gradient_checkpointing: bool = PI05_DEFAULT_GRADIENT_CHECKPOINTING
    compile_model: bool = PI05_DEFAULT_COMPILE_MODEL
    compile_mode: str = PI05_DEFAULT_COMPILE_MODE

    freeze_vision_encoder: bool = PI05_DEFAULT_FREEZE_VISION_ENCODER
    train_expert_only: bool = PI05_DEFAULT_TRAIN_EXPERT_ONLY

    # Optimizer settings
    optimizer_lr: float = PI05_DEFAULT_OPTIMIZER_LR
    optimizer_betas: tuple[float, float] = PI05_DEFAULT_OPTIMIZER_BETAS
    optimizer_eps: float = PI05_DEFAULT_OPTIMIZER_EPS
    optimizer_weight_decay: float = PI05_DEFAULT_OPTIMIZER_WEIGHT_DECAY
    optimizer_grad_clip_norm: float = PI05_DEFAULT_OPTIMIZER_GRAD_CLIP_NORM

    # Scheduler settings
    scheduler_warmup_steps: int = PI05_DEFAULT_SCHEDULER_WARMUP_STEPS
    scheduler_decay_steps: int = PI05_DEFAULT_SCHEDULER_DECAY_STEPS
    scheduler_decay_lr: float = PI05_DEFAULT_SCHEDULER_DECAY_LR

    def __post_init__(self):
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logger.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

        if self.use_amp and not is_amp_available(self.device):
            logger.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

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

    # ------------------------------------------------------------------ #
    # Feature helpers                                                     #
    # ------------------------------------------------------------------ #

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
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

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        if not self.output_features:
            return None
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    # ------------------------------------------------------------------ #
    # Serialization                                                       #
    # ------------------------------------------------------------------ #

    def _save_pretrained(self, save_directory: Path) -> None:
        save_config(self, save_directory, type_name=self.type)

    @classmethod
    def from_pretrained(
        cls: builtins.type["PI05Config"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> "PI05Config":
        return load_config_from_checkpoint(
            pretrained_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
        )


# Register with the policy registry.
from lerobot.policies.registry import register_policy  # noqa: E402

register_policy("pi05", PI05Config)
