import logging
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from termcolor import colored

from dino_wm.config import DinoWMConfig
from dino_wm.visual_world_model import VWorldModel
from dino_wm.encoder import DinoV2Encoder
from dino_wm.transition import TransitionModel
from dino_wm.decoder import Decoder

from data.lerobot_dataset import LeRobotDataset
from torchvision.transforms import v2 as T
from data.utils import POLICY_FEATURES, cycle, dataset_to_policy_features
from utils.types import FeatureType, NormalizationMode
from utils.processor_utils import normalize, to_device
from utils.train_utils import set_seed
from utils.utils import format_big_number, init_logging
from utils.wandb_utils import WandBLogger

CFG = DinoWMConfig(
    dataset_repo_id="AndrewNoviello/domino-world-v3",

    # Temporal window
    # frameskip=3 subsamples 30fps → effective 10fps so consecutive frames
    # show meaningful motion from the SO101 arm.
    num_hist=2,
    num_pred=1,
    frameskip=3,

    # Image / encoder
    img_size=224,
    encoder_name="dinov2_vits14",
    train_encoder=False,

    # Action / proprio
    action_emb_dim=10,
    proprio_emb_dim=10,
    num_action_repeat=1,
    num_proprio_repeat=1,
    concat_dim=1,

    # Predictor
    train_predictor=True,
    predictor_depth=6,
    predictor_heads=16,
    predictor_mlp_dim=2048,
    predictor_dropout=0.1,

    # Decoder
    train_decoder=True,
    decoder_channel=384,
    decoder_n_res_block=4,
    decoder_n_res_channel=128,

    # Training — larger batch/workers for better GPU utilization (A40 has headroom)
    steps=60_000,
    batch_size=24,
    num_workers=6,
    seed=42,
    encoder_lr=1e-6,
    predictor_lr=5e-4,
    decoder_lr=3e-4,
    action_encoder_lr=5e-4,
    grad_clip_norm=1.0,

    # Logging
    output_dir="outputs/dino_wm_v3",
    log_freq=100,
    save_freq=10_000,
    save_checkpoint=True,
    wandb_enable=False,
    wandb_project="dino_wm",
    wandb_entity=None,

    # Upload checkpoints to this HuggingFace model repo after training
    hf_model_repo_id="AndrewNoviello/domino-world-wm-v3",
)

# VISUAL: normalize() always applies ImageNet mean/std for VISUAL features
#         (hardcoded in normalize()), which is correct for DinoV2.
# STATE / ACTION: use MEAN_STD from dataset statistics.
DINO_WM_NORM_MAP = {
    FeatureType.VISUAL: NormalizationMode.IDENTITY,   # ImageNet applied anyway
    FeatureType.STATE:  NormalizationMode.MEAN_STD,
    FeatureType.ACTION: NormalizationMode.MEAN_STD,
}


def _detect_image_key(features: dict) -> str:
    """Return the first image-dtype key in the dataset."""
    preferred = ("image0", "observation.image", "observation.images.front")
    for k in preferred:
        if k in features and features[k]["dtype"] == "image":
            return k
    for k, v in features.items():
        if v["dtype"] == "image":
            return k
    raise ValueError(
        "No image observation key found in dataset features."
    )


def _make_delta_indices(cfg: DinoWMConfig, features: dict) -> dict[str, list[int]]:
    """Build delta_indices dict for LeRobotDataset.

    Each key requests `num_hist + num_pred` frames spaced by `frameskip` steps.
    """
    window = cfg.num_hist + cfg.num_pred
    indices = [i * cfg.frameskip for i in range(window)]

    delta_indices: dict[str, list[int]] = {}

    # Image features (image0, image1, ... or observation.images.*)
    for k, v in features.items():
        if v["dtype"] == "image":
            delta_indices[k] = indices

    # State and actions
    if "observation.state" in features:
        delta_indices["observation.state"] = indices
    if "action" in features:
        delta_indices["action"] = indices

    return delta_indices


def _build_model(
    cfg: DinoWMConfig,
    action_dim: int,
    proprio_dim: int,
) -> tuple[VWorldModel, dict]:
    """Instantiate VWorldModel and return (model, sub_modules_dict).

    sub_modules_dict contains references to individually-optimised
    sub-modules so that the training loop can call step() on them.
    """
    encoder = DinoV2Encoder(name=cfg.encoder_name)
    emb_dim: int = encoder.emb_dim  # 384 for dinov2_vits14

    if not cfg.train_encoder:
        for p in encoder.parameters():
            p.requires_grad = False

    # Number of visual patches after the encoder resize
    decoder_scale = 16  # fixed by decoder stride=4 × stride=4
    num_side = cfg.img_size // decoder_scale
    num_vis_patches = num_side**2  # e.g. 14^2 = 196 for img_size=224

    transition = TransitionModel(
        num_patches=num_vis_patches,
        num_frames=cfg.num_hist,
        emb_dim=emb_dim,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        proprio_emb_dim=cfg.proprio_emb_dim,
        action_emb_dim=cfg.action_emb_dim,
        concat_dim=cfg.concat_dim,
        num_proprio_repeat=cfg.num_proprio_repeat,
        num_action_repeat=cfg.num_action_repeat,
        depth=cfg.predictor_depth,
        heads=cfg.predictor_heads,
        mlp_dim=cfg.predictor_mlp_dim,
        dropout=cfg.predictor_dropout,
        emb_dropout=cfg.predictor_emb_dropout,
    )
    if not cfg.train_predictor:
        for p in transition.parameters():
            p.requires_grad = False

    decoder = Decoder(
        channel=cfg.decoder_channel,
        n_res_block=cfg.decoder_n_res_block,
        n_res_channel=cfg.decoder_n_res_channel,
        emb_dim=emb_dim,
    )
    if not cfg.train_decoder:
        for p in decoder.parameters():
            p.requires_grad = False

    model = VWorldModel(
        image_size=cfg.img_size,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        encoder=encoder,
        transition=transition,
        decoder=decoder,
        proprio_dim=cfg.proprio_emb_dim,
        action_dim=cfg.action_emb_dim,
        concat_dim=cfg.concat_dim,
        num_action_repeat=cfg.num_action_repeat,
        num_proprio_repeat=cfg.num_proprio_repeat,
        train_encoder=cfg.train_encoder,
        train_predictor=cfg.train_predictor,
        train_decoder=cfg.train_decoder,
        use_failure_head=cfg.use_failure_head,
        failure_head_hidden_dim=cfg.failure_head_hidden_dim,
    )

    sub = {
        "encoder": encoder,
        "transition": transition,
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

    if cfg.train_predictor:
        opts.append(
            torch.optim.AdamW(sub["transition"].parameters(), lr=cfg.predictor_lr)
        )

    if cfg.train_decoder:
        opts.append(
            torch.optim.Adam(sub["decoder"].parameters(), lr=cfg.decoder_lr)
        )

    return opts


def _reformat_batch(
    batch: dict,
    image_key: str,
) -> tuple[dict, torch.Tensor]:
    """Convert a LeRobot batch dict into VWorldModel inputs.

    LeRobot format (after normalise + to_device):
        batch[image_key]  : (B, T, C, H, W)  float, ImageNet-normalised
        batch["observation.state"]    : (B, T, state_dim)
        batch["action"]  : (B, T, action_dim)

    VWorldModel expects:
        obs = {"visual": (B, T, C, H, W), "proprio": (B, T, proprio_dim)}
        act = (B, T, action_dim)
    """
    visual = batch[image_key].float()   # (B, T, C, H, W)

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


def train(cfg: DinoWMConfig = CFG) -> None:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )

    init_logging(accelerator=accelerator)
    is_main = accelerator.is_main_process

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

    if is_main:
        logging.info("Creating dataset (main process first)")

    image_key = _detect_image_key(POLICY_FEATURES)
    logging.info(f"Detected image key: {image_key!r}")

    delta_indices = _make_delta_indices(cfg, POLICY_FEATURES)

    action_dim: int = POLICY_FEATURES["action"]["shape"][-1]
    proprio_dim: int = POLICY_FEATURES["observation.state"]["shape"][-1]
    logging.info(f"action_dim={action_dim}, proprio_dim={proprio_dim}")

    accelerator.wait_for_everyone()

    dataset = LeRobotDataset(
        cfg.dataset_repo_id,
        delta_indices=delta_indices,
        image_transforms=T.Resize((cfg.img_size, cfg.img_size), antialias=True),
    )

    # Optionally preload all video frames into RAM before spawning workers.
    if cfg.preload_frames and hasattr(dataset, "preload_video_frames"):
        if is_main:
            logging.info(
                "Preloading all video frames into RAM "
                "(cfg.preload_frames=True). This may take a few minutes ..."
            )
        dataset.preload_video_frames()
        if is_main:
            logging.info("Frame preload complete.")

    # Build PolicyFeature dict for use with normalize()
    policy_features = dataset_to_policy_features(POLICY_FEATURES)

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

    optimizers = _make_optimizers(cfg, sub_modules)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0,  # keep workers alive so LRU container cache survives across epochs
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

    effective_batch_size = cfg.batch_size * accelerator.num_processes
    avg_samples_per_ep = dataset.num_frames / dataset.num_episodes

    if is_main:
        logging.info("Starting DINO-WM training")
        if cfg.num_workers > 0:
            logging.info(
                f"Spawning {cfg.num_workers} DataLoader workers "
                "(persistent_workers=True). First batch may take 15-60 s "
                "while workers start up and warm their video container caches."
            )

    for step in range(1, cfg.steps + 1):
        t0 = time.perf_counter()
        batch = next(dl_iter)
        # Normalize (ImageNet for visual; MEAN_STD for state/action)
        batch = normalize(batch, dataset.stats, policy_features, DINO_WM_NORM_MAP)
        batch = to_device(batch, device)
        dataloading_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        model.train()

        obs, act = _reformat_batch(batch, image_key)

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

        update_s = time.perf_counter() - t1

        # Gather component losses (all processes must call for distributed sync)
        gathered = accelerator.gather_for_metrics(loss_components)
        gathered_components = {}
        if is_main:
            for k, v in gathered.items():
                if isinstance(v, torch.Tensor):
                    gathered_components[k] = v.mean().item()
                else:
                    gathered_components[k] = v

        is_save_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_log_step = step % cfg.log_freq == 0 or step == cfg.steps

        if is_main:
            samples = step * effective_batch_size
            episodes = samples / avg_samples_per_ep
            epochs = samples / dataset.num_frames
            if wandb_logger:
                log_dict = {
                    "steps": step,
                    "samples": samples,
                    "episodes": episodes,
                    "epochs": epochs,
                    "loss": loss.item(),
                    "grad_norm": total_norm,
                    "update_s": update_s,
                    "dataloading_s": dataloading_s,
                    **gathered_components,
                }
                wandb_logger.log_dict(log_dict, step)
            if is_log_step:
                logging.info(
                    f"[step {step}/{cfg.steps}] done — "
                    f"loss={loss.item():.4f} grdn={total_norm:.3f} "
                    f"data_s={dataloading_s:.3f} updt_s={update_s:.3f}"
                )

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

    if is_main and cfg.hf_model_repo_id:
        from huggingface_hub import HfApi
        logging.info(f"Uploading checkpoints to HuggingFace Hub: {cfg.hf_model_repo_id}")
        api = HfApi()
        api.create_repo(cfg.hf_model_repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=cfg.output_dir,
            repo_id=cfg.hf_model_repo_id,
            repo_type="model",
        )
        logging.info("Upload complete.")


def main() -> None:
    train()


if __name__ == "__main__":
    main()
