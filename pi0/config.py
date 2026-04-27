import builtins
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from utils.types import NormalizationMode, PolicyFeature
from pi0.rtc_config import RTCConfig
from utils.config_utils import load_config_from_checkpoint, save_config

T = type["PI0Config"]


@dataclass
class PI0Config:
    type: ClassVar[str] = "pi0"

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

    dtype: str = "float32"

    # PaliGemma VLM backbone (gemma_2b)
    paligemma_width: int = 2048
    paligemma_depth: int = 18
    paligemma_mlp_dim: int = 16_384
    paligemma_num_heads: int = 8
    paligemma_num_kv_heads: int = 1
    paligemma_head_dim: int = 256

    # Action expert (gemma_300m)
    action_expert_width: int = 1024
    action_expert_depth: int = 18
    action_expert_mlp_dim: int = 4096
    action_expert_num_heads: int = 8
    action_expert_num_kv_heads: int = 1
    action_expert_head_dim: int = 256

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

    tokenizer_max_length: int = 48

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"

    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

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
