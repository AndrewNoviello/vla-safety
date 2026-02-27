"""Configuration dataclass for DINO-WM world model training.

Mirrors the style of PI0Config / PI05Config in lerobot, using plain Python
dataclasses (no Hydra) so it integrates naturally with the rest of the
lerobot configuration system.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DinoWMConfig:
    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset_repo_id: str = "lerobot/pusht_image"
    dataset_root: Optional[str] = None
    dataset_episodes: Optional[list] = None

    # ------------------------------------------------------------------
    # Temporal window
    # ------------------------------------------------------------------
    # Number of historical (context) frames fed to the predictor.
    num_hist: int = 3
    # Number of future frames to predict (only 1 is supported).
    num_pred: int = 1
    # Frame sub-sampling: use every Nth frame from the dataset trajectory.
    # Timestamps are spaced frameskip / fps seconds apart.
    frameskip: int = 1

    # ------------------------------------------------------------------
    # Image / encoder settings
    # ------------------------------------------------------------------
    img_size: int = 224          # Final spatial resolution fed to the decoder
    encoder_name: str = "dinov2_vits14"
    encoder_feature_key: str = "x_norm_patchtokens"  # or "x_norm_clstoken"
    train_encoder: bool = False  # Keep DINO frozen

    # ------------------------------------------------------------------
    # Action / proprio embedding
    # ------------------------------------------------------------------
    action_emb_dim: int = 10
    proprio_emb_dim: int = 10
    # How many times the action/proprio embedding is repeated along the
    # feature dimension when concat_dim == 1.
    num_action_repeat: int = 1
    num_proprio_repeat: int = 1
    # 0 → add proprio & action as extra tokens (extra patches per frame).
    #     Requires action_emb_dim == proprio_emb_dim == encoder.emb_dim (e.g. 384
    #     for dinov2_vits14).  Use 1 unless you need this mode.
    # 1 → concatenate them to the visual feature dimension (default, no constraint
    #     on action_emb_dim / proprio_emb_dim).
    concat_dim: int = 1

    # ------------------------------------------------------------------
    # Predictor (ViT)
    # ------------------------------------------------------------------
    has_predictor: bool = True
    train_predictor: bool = True
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_mlp_dim: int = 2048
    predictor_dropout: float = 0.1
    predictor_emb_dropout: float = 0.0

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------
    has_decoder: bool = True
    train_decoder: bool = True
    decoder_channel: int = 384
    decoder_n_res_block: int = 4
    decoder_n_res_channel: int = 128

    # ------------------------------------------------------------------
    # Training hyper-parameters
    # ------------------------------------------------------------------
    steps: int = 100_000
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    encoder_lr: float = 1e-6
    predictor_lr: float = 5e-4
    decoder_lr: float = 3e-4
    action_encoder_lr: float = 5e-4
    grad_clip_norm: float = 1.0

    # ------------------------------------------------------------------
    # Logging / checkpointing
    # ------------------------------------------------------------------
    output_dir: str = "outputs/dino_wm"
    log_freq: int = 100
    save_freq: int = 10_000
    save_checkpoint: bool = True
    wandb_enable: bool = False
    wandb_project: str = "dino_wm"
    wandb_entity: Optional[str] = None
    wandb_notes: Optional[str] = None
