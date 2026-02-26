#!/usr/bin/env python
"""DINO-WM world-model training using LeRobot datasets.

Trains VWorldModel (dino_wm) on any LeRobotDataset, using the same
dataset / dataloader infrastructure as the pi0 training script.

Edit the "Configuration" section below, then run:

    python -m lerobot.dino_wm_train

Or with accelerate for multi-GPU:

    accelerate launch -m lerobot.dino_wm_train

Notes
-----
* Only datasets whose image features are stored as dtype "image" (parquet)
  support multi-frame loading via delta_timestamps.  Datasets that store
  images as dtype "video" (mp4) will only provide a single frame per sample;
  for those you should set num_hist=1 and num_pred=1, or implement custom
  multi-frame video decoding.
* The dino_wm package must be importable.  This script prepends
  DINO_WM_PATH to sys.path before importing any dino_wm modules.
"""

import itertools
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from termcolor import colored

# ---------------------------------------------------------------------------
# lerobot imports come FIRST.  dino_wm has a local `datasets/` package that
# would shadow the HuggingFace `datasets` library if its path were on
# sys.path before the lerobot imports below.
# ---------------------------------------------------------------------------
from lerobot.configs.dino_wm_config import DinoWMConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.augmentation import image_transforms
from lerobot.datasets.utils import cycle, dataset_to_policy_features
from lerobot.types import FeatureType, NormalizationMode
from lerobot.utils.processor_utils import normalize, to_device
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.train_utils import set_seed
from lerobot.utils.utils import format_big_number, init_logging
from lerobot.utils.wandb_utils import WandBLogger

# ---------------------------------------------------------------------------
# Add dino_wm to sys.path AFTER lerobot imports to avoid the local
# `dino_wm/datasets/` package shadowing the HuggingFace `datasets` library.
# ---------------------------------------------------------------------------
DINO_WM_PATH = Path(__file__).resolve().parent.parent / "dino_wm"
if str(DINO_WM_PATH) not in sys.path:
    sys.path.insert(0, str(DINO_WM_PATH))

# dino_wm model imports
from models.visual_world_model import VWorldModel  # noqa: E402
from models.dino import DinoV2Encoder  # noqa: E402
from models.vit import ViTPredictor  # noqa: E402
from models.vqvae import VQVAE  # noqa: E402
from models.proprio import ProprioceptiveEmbedding  # noqa: E402

# =====================================================================
# Configuration -- edit these values for your experiment
# =====================================================================

CFG = DinoWMConfig(
    dataset_repo_id="lerobot/pusht_image",
    dataset_root=None,
    dataset_episodes=None,

    # Temporal window
    num_hist=3,
    num_pred=1,
    frameskip=1,

    # Image / encoder
    img_size=224,
    encoder_name="dinov2_vits14",
    encoder_feature_key="x_norm_patchtokens",
    train_encoder=False,

    # Action / proprio
    action_emb_dim=10,
    proprio_emb_dim=10,
    num_action_repeat=1,
    num_proprio_repeat=1,
    concat_dim=1,

    # Predictor
    has_predictor=True,
    train_predictor=True,
    predictor_depth=6,
    predictor_heads=16,
    predictor_mlp_dim=2048,
    predictor_dropout=0.1,

    # Decoder
    has_decoder=True,
    train_decoder=True,
    decoder_channel=384,
    decoder_n_embed=2048,
    decoder_n_res_block=4,
    decoder_n_res_channel=128,
    decoder_quantize=False,

    # Training
    steps=100_000,
    batch_size=32,
    num_workers=4,
    seed=42,
    encoder_lr=1e-6,
    predictor_lr=5e-4,
    decoder_lr=3e-4,
    action_encoder_lr=5e-4,
    grad_clip_norm=1.0,

    # Logging
    output_dir="outputs/dino_wm",
    log_freq=100,
    save_freq=10_000,
    save_checkpoint=True,
    wandb_enable=False,
    wandb_project="dino_wm",
    wandb_entity=None,
)

# =====================================================================
# Normalization map (analogous to PI0_NORM_MAP in lerobot_train.py)
# =====================================================================
# VISUAL: normalize() always applies ImageNet mean/std for VISUAL features
#         (hardcoded in normalize()), which is correct for DinoV2.
# STATE / ACTION: use MEAN_STD from dataset statistics.
DINO_WM_NORM_MAP = {
    FeatureType.VISUAL: NormalizationMode.IDENTITY,   # ImageNet applied anyway
    FeatureType.STATE:  NormalizationMode.MEAN_STD,
    FeatureType.ACTION: NormalizationMode.MEAN_STD,
}


# =====================================================================
# Helpers
# =====================================================================

def _detect_image_key(features: dict) -> str:
    """Return the first image-dtype observation key in the dataset."""
    # Prefer exact "observation.image", then any "observation.images.*"
    preferred = ("observation.image",)
    for k in preferred:
        if k in features and features[k]["dtype"] == "image":
            return k
    for k, v in features.items():
        if k.startswith("observation.") and v["dtype"] == "image":
            return k
    raise ValueError(
        "No image-dtype observation key found in dataset features. "
        "This script currently supports datasets with images stored as "
        "parquet (dtype='image'). For video-stored datasets (dtype='video') "
        "you need multi-frame video decoding support."
    )


def _make_delta_timestamps(
    cfg: DinoWMConfig, fps: float, features: dict
) -> dict[str, list[float]]:
    """Build delta_timestamps dict for LeRobotDataset.

    Each key in the returned dict requests `num_hist + num_pred` frames
    spaced by `frameskip / fps` seconds.
    """
    window = cfg.num_hist + cfg.num_pred
    step_s = cfg.frameskip / fps
    timestamps = [i * step_s for i in range(window)]

    delta_ts: dict[str, list[float]] = {}

    # Image features (only parquet-stored "image" dtype)
    for k, v in features.items():
        if k.startswith("observation.") and v["dtype"] == "image":
            delta_ts[k] = timestamps

    # State and action
    if "observation.state" in features:
        delta_ts["observation.state"] = timestamps
    if "action" in features:
        delta_ts["action"] = timestamps

    return delta_ts


def _build_model(
    cfg: DinoWMConfig,
    action_dim: int,
    proprio_dim: int,
) -> tuple[VWorldModel, dict]:
    """Instantiate VWorldModel and return (model, sub_modules_dict).

    sub_modules_dict contains references to individually-optimised
    sub-modules so that the training loop can call step() on them.
    """
    # --- Encoder ---
    encoder = DinoV2Encoder(
        name=cfg.encoder_name,
        feature_key=cfg.encoder_feature_key,
    )
    emb_dim: int = encoder.emb_dim  # 384 for dinov2_vits14

    if not cfg.train_encoder:
        for p in encoder.parameters():
            p.requires_grad = False

    # --- Proprio / Action encoders ---
    proprio_encoder = ProprioceptiveEmbedding(
        in_chans=proprio_dim,
        emb_dim=cfg.proprio_emb_dim,
    )
    action_encoder = ProprioceptiveEmbedding(
        in_chans=action_dim,
        emb_dim=cfg.action_emb_dim,
    )

    # --- Predictor ---
    predictor = None
    if cfg.has_predictor:
        # Number of visual patches after the encoder resize
        if encoder.latent_ndim == 1:
            num_vis_patches = 1
        else:
            decoder_scale = 16  # fixed by VQVAE stride=4 × stride=4
            num_side = cfg.img_size // decoder_scale
            num_vis_patches = num_side ** 2  # e.g. 14^2 = 196 for img_size=224

        # Extra tokens for proprio and action (concat_dim == 0)
        num_patches = num_vis_patches + (2 if cfg.concat_dim == 0 else 0)

        # Feature dimension seen by the ViT
        predictor_dim = emb_dim + (
            cfg.proprio_emb_dim * cfg.num_proprio_repeat
            + cfg.action_emb_dim * cfg.num_action_repeat
        ) * cfg.concat_dim

        predictor = ViTPredictor(
            num_patches=num_patches,
            num_frames=cfg.num_hist,
            dim=predictor_dim,
            depth=cfg.predictor_depth,
            heads=cfg.predictor_heads,
            mlp_dim=cfg.predictor_mlp_dim,
            dropout=cfg.predictor_dropout,
            emb_dropout=cfg.predictor_emb_dropout,
        )
        if not cfg.train_predictor:
            for p in predictor.parameters():
                p.requires_grad = False

    # --- Decoder ---
    decoder = None
    if cfg.has_decoder:
        decoder = VQVAE(
            channel=cfg.decoder_channel,
            n_res_block=cfg.decoder_n_res_block,
            n_res_channel=cfg.decoder_n_res_channel,
            emb_dim=emb_dim,
            n_embed=cfg.decoder_n_embed,
            quantize=cfg.decoder_quantize,
        )
        if not cfg.train_decoder:
            for p in decoder.parameters():
                p.requires_grad = False

    # --- World model ---
    model = VWorldModel(
        image_size=cfg.img_size,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        decoder=decoder,
        proprio_dim=cfg.proprio_emb_dim,
        action_dim=cfg.action_emb_dim,
        concat_dim=cfg.concat_dim,
        num_action_repeat=cfg.num_action_repeat,
        num_proprio_repeat=cfg.num_proprio_repeat,
        train_encoder=cfg.train_encoder,
        train_predictor=cfg.train_predictor,
        train_decoder=cfg.train_decoder,
    )

    sub = {
        "encoder": encoder,
        "proprio_encoder": proprio_encoder,
        "action_encoder": action_encoder,
        "predictor": predictor,
        "decoder": decoder,
    }
    return model, sub


def _make_optimizers(cfg: DinoWMConfig, sub: dict) -> list[torch.optim.Optimizer]:
    """Create one optimizer per trainable component group."""
    opts: list[torch.optim.Optimizer] = []

    if cfg.train_encoder:
        opts.append(
            torch.optim.Adam(sub["encoder"].parameters(), lr=cfg.encoder_lr)
        )

    if cfg.has_predictor and cfg.train_predictor:
        # Predictor + both embedding encoders share one optimizer
        params = itertools.chain(
            sub["predictor"].parameters(),
            sub["action_encoder"].parameters(),
            sub["proprio_encoder"].parameters(),
        )
        opts.append(
            torch.optim.AdamW(params, lr=cfg.predictor_lr)
        )

    if cfg.has_decoder and cfg.train_decoder:
        opts.append(
            torch.optim.Adam(sub["decoder"].parameters(), lr=cfg.decoder_lr)
        )

    return opts


def _reformat_batch(
    batch: dict,
    image_key: str,
    img_size: int,
) -> tuple[dict, torch.Tensor]:
    """Convert a LeRobot batch dict into VWorldModel inputs.

    LeRobot format (after normalise + to_device):
        batch[image_key]          : (B, T, C, H, W)  float, ImageNet-normalised
        batch["observation.state"]: (B, T, state_dim)
        batch["action"]           : (B, T, action_dim)

    VWorldModel expects:
        obs = {"visual": (B, T, C, H, W), "proprio": (B, T, proprio_dim)}
        act = (B, T, action_dim)
    """
    visual = batch[image_key].float()   # (B, T, C, H, W)

    # Resize spatial dims to img_size if the dataset resolution differs.
    if visual.shape[-1] != img_size or visual.shape[-2] != img_size:
        B, T, C, H, W = visual.shape
        visual = F.interpolate(
            visual.reshape(B * T, C, H, W),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, T, C, img_size, img_size)

    obs = {
        "visual": visual,
        "proprio": batch["observation.state"].float(),   # (B, T, state_dim)
    }
    act = batch["action"].float()   # (B, T, action_dim)
    return obs, act


def _save_checkpoint(
    output_dir: str,
    step: int,
    total_steps: int,
    model: VWorldModel,
    optimizers: list[torch.optim.Optimizer],
    accelerator: Accelerator,
) -> None:
    """Save model weights and optimizer states."""
    checkpoint_dir = (
        Path(output_dir) / "checkpoints" / f"{step:08d}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    torch.save(unwrapped.state_dict(), checkpoint_dir / "model.pt")

    for i, opt in enumerate(optimizers):
        torch.save(opt.state_dict(), checkpoint_dir / f"optimizer_{i}.pt")

    # Keep a symlink / copy to "latest"
    latest_link = Path(output_dir) / "checkpoints" / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(checkpoint_dir.name)

    logging.info(f"Saved checkpoint to {checkpoint_dir}")


# =====================================================================
# Main training function
# =====================================================================

def train(cfg: DinoWMConfig = CFG) -> None:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )

    init_logging(accelerator=accelerator)
    is_main = accelerator.is_main_process

    # --- W&B ---
    wandb_logger = None
    if cfg.wandb_enable and is_main:
        wandb_logger = WandBLogger(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            notes=cfg.wandb_notes,
            log_dir=cfg.output_dir,
            job_name="dino_wm",
            policy_type="dino_wm",
            seed=cfg.seed,
            dataset_repo_id=cfg.dataset_repo_id,
        )

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # --- Dataset ---
    if is_main:
        logging.info("Creating dataset (main process first)")

    image_transforms_fn = image_transforms()

    # We need to build delta_timestamps, which requires the dataset fps.
    # Load the dataset once without delta_timestamps to get fps / features.
    dataset_probe = LeRobotDataset(
        cfg.dataset_repo_id,
        episodes=cfg.dataset_episodes,
        root=cfg.dataset_root,
    )
    fps: float = dataset_probe.fps
    raw_features: dict = dataset_probe.features
    del dataset_probe   # free the probe; real dataset loads below

    image_key = _detect_image_key(raw_features)
    logging.info(f"Detected image key: {image_key!r}  (fps={fps})")

    # Check for video-dtype images and warn accordingly
    video_image_keys = [
        k for k, v in raw_features.items()
        if k.startswith("observation.") and v["dtype"] == "video"
    ]
    if video_image_keys:
        logging.warning(
            f"Dataset has video-stored images {video_image_keys}. "
            "Multi-frame loading for these is not implemented; they will "
            "provide only the current frame.  Consider using a dataset with "
            "parquet-stored images (dtype='image') for full DINO-WM training."
        )

    delta_timestamps = _make_delta_timestamps(cfg, fps, raw_features)

    # Infer action / proprio dims from dataset metadata
    action_dim: int = raw_features["action"]["shape"][-1]
    proprio_dim: int = raw_features["observation.state"]["shape"][-1]
    logging.info(f"action_dim={action_dim}, proprio_dim={proprio_dim}")

    accelerator.wait_for_everyone()

    dataset = LeRobotDataset(
        cfg.dataset_repo_id,
        episodes=cfg.dataset_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms_fn,
        root=cfg.dataset_root,
    )

    # Build PolicyFeature dict for use with normalize()
    policy_features = dataset_to_policy_features(raw_features)

    # --- Model ---
    if is_main:
        logging.info("Building DINO-WM model")

    model, sub_modules = _build_model(cfg, action_dim, proprio_dim)

    num_learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())

    if is_main:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"steps={cfg.steps} ({format_big_number(cfg.steps)})")
        logging.info(f"dataset.num_frames={dataset.num_frames} ({format_big_number(dataset.num_frames)})")
        logging.info(f"dataset.num_episodes={dataset.num_episodes}")
        effective_bs = cfg.batch_size * accelerator.num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} × {accelerator.num_processes} = {effective_bs}")
        logging.info(f"num_learnable_params={num_learnable} ({format_big_number(num_learnable)})")
        logging.info(f"num_total_params={num_total} ({format_big_number(num_total)})")

    # --- Optimizers ---
    optimizers = _make_optimizers(cfg, sub_modules)

    # --- Dataloader ---
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Prepare with accelerate
    accelerator.wait_for_everyone()
    prepare_args = [model, dataloader] + optimizers
    prepared = accelerator.prepare(*prepare_args)
    model = prepared[0]
    dataloader = prepared[1]
    optimizers = list(prepared[2:])

    dl_iter = cycle(dataloader)

    model.train()

    # --- Metrics ---
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Add per-component loss meters
    loss_keys = ["z_loss", "z_visual_loss", "z_proprio_loss"]
    if cfg.has_decoder:
        loss_keys += [
            "decoder_recon_loss_pred",
            "decoder_recon_loss_reconstructed",
            "decoder_vq_loss_pred",
            "decoder_vq_loss_reconstructed",
        ]
    for k in loss_keys:
        train_metrics[k] = AverageMeter(k, ":.4f")

    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        steps=0,
        accelerator=accelerator,
    )

    if is_main:
        logging.info("Starting DINO-WM training")

    for step in range(1, cfg.steps + 1):
        t0 = time.perf_counter()
        batch = next(dl_iter)
        # Normalize (ImageNet for visual; MEAN_STD for state/action)
        batch = normalize(batch, dataset.stats, policy_features, DINO_WM_NORM_MAP)
        batch = to_device(batch, device)
        train_tracker.dataloading_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        model.train()

        obs, act = _reformat_batch(batch, image_key, cfg.img_size)

        with accelerator.autocast():
            _z_pred, _vis_pred, _vis_recon, loss, loss_components = (
                accelerator.unwrap_model(model)(obs, act)
            )

        # Zero all optimizer gradients
        for opt in optimizers:
            opt.zero_grad()

        accelerator.backward(loss)

        # Clip & step each optimizer
        total_norm = 0.0
        for opt in optimizers:
            if cfg.grad_clip_norm > 0:
                norm = accelerator.clip_grad_norm_(
                    [p for pg in opt.param_groups for p in pg["params"]],
                    cfg.grad_clip_norm,
                )
                if isinstance(norm, torch.Tensor):
                    total_norm = max(total_norm, norm.item())
            opt.step()

        train_tracker.loss = loss.item()
        train_tracker.grad_norm = total_norm
        train_tracker.update_s = time.perf_counter() - t1

        # Update component loss meters
        gathered_components = accelerator.gather_for_metrics(loss_components)
        if is_main:
            for k, v in gathered_components.items():
                if k in train_metrics:
                    if isinstance(v, torch.Tensor):
                        v = v.mean().item()
                    train_metrics[k].update(v)

        train_tracker.step()

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main
        is_save_step = step % cfg.save_freq == 0 or step == cfg.steps

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                log_dict = train_tracker.to_dict()
                wandb_logger.log_dict(log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_save_step and is_main:
            logging.info(f"Saving checkpoint at step {step}")
            _save_checkpoint(
                cfg.output_dir,
                step,
                cfg.steps,
                model,
                optimizers,
                accelerator,
            )

        accelerator.wait_for_everyone()

    if is_main:
        logging.info("Training complete.")

    accelerator.end_training()


def main() -> None:
    train()


if __name__ == "__main__":
    main()
