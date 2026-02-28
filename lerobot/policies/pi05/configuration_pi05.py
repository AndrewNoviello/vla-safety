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
from pathlib import Path
from typing import Any, ClassVar

from lerobot.types import NormalizationMode, PolicyFeature
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.config_utils import load_config_from_checkpoint, save_config


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
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"

    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (224, 224)

    # Add empty images when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"

    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    # Optimizer settings
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
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
