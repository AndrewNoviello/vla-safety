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

import builtins
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
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
from lerobot.utils.config_utils import load_config_from_checkpoint, save_config
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.utils import auto_select_torch_device, is_amp_available, is_torch_device_available

logger = getLogger(__name__)

T = type["PI0Config"]


@dataclass
class PI0Config:
    # The registered name for this policy type — used in config.json and the registry.
    type: ClassVar[str] = "pi0"

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
    # PI0-specific fields                                                 #
    # ------------------------------------------------------------------ #
    paligemma_variant: str = PI0_DEFAULT_PALIGEMMA
    action_expert_variant: str = PI0_DEFAULT_ACTION_EXPERT
    dtype: str = PI0_DEFAULT_DTYPE

    chunk_size: int = PI0_DEFAULT_CHUNK_SIZE
    n_action_steps: int = PI0_DEFAULT_N_ACTION_STEPS

    max_state_dim: int = PI0_DEFAULT_MAX_STATE_DIM
    max_action_dim: int = PI0_DEFAULT_MAX_ACTION_DIM

    # Flow matching parameters
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

    freeze_vision_encoder: bool = PI0_DEFAULT_FREEZE_VISION_ENCODER
    train_expert_only: bool = PI0_DEFAULT_TRAIN_EXPERT_ONLY

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
        cls: builtins.type["PI0Config"],
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
    ) -> "PI0Config":
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

register_policy("pi0", PI0Config)
