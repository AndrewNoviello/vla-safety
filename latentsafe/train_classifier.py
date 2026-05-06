"""Fine-tune the failure_head of a trained VWorldModel.

All parameters except failure_head are frozen. Training uses a binary margin
ranking loss:
  0 = safe   → score should be < -margin  (head outputs a negative value)
  1 = unsafe → score should be > +margin

The head is trained on **encoded** latents (output of `model.encode`), paired
with the frame's `failure_label`. `model.predict_failure(...)` strips the
action features before pooling, so the classifier scores visual+proprio state
rather than state-action. The world model's `z_loss` term enforces that
`model.predict(...)` outputs land in the same encoded-latent space, so the
head trained here works at deployment in both modes:
  - live monitor: `predict_failure(model.encode(obs, act))` (single frame).
  - filter / shielding: `predict_failure(model.predict(model.encode(...)))`.

Cached latents from `scripts/cache_latents.py` are already `model.encode`
output and need no recompute.

Usage
-----
The dataset must have a "failure_label" tensor of shape (T,) in {0, 1} per
sample (the flat-format LeRobotDataset emits this automatically by mapping
1 - label).

    python -m latentsafe.train_classifier \
        --wm_checkpoint runs/dino_wm_exp_merged/checkpoints/latest/model.pt \
        --steps 10000

Architecture note
-----------------
The failure_head was added to VWorldModel with use_failure_head=True.
It takes mean-pooled visual+proprio latents (B, T, encoder_dim + proprio_emb_dim)
→ (B, T, 1), excluding the action features from the full predictor latent.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import v2 as T

from dino_wm.config import DinoWMConfig
from dino_wm.decoder import Decoder
from dino_wm.encoder import DinoV2Encoder
from dino_wm.transition import TransitionModel
from dino_wm.visual_world_model import VWorldModel
from data.lerobot_dataset import LeRobotDataset
from data.utils import POLICY_FEATURES, cycle
from latentsafe.cached_latent_dataset import CachedLatentDataset
from utils.processor_utils import normalize, to_device
from utils.utils import init_logging

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Binary margin-based ranking loss
# ---------------------------------------------------------------------------

MARGIN = 1.0


def fail_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary margin ranking loss.

    Args:
        scores: (N,) predicted failure scores (higher = more unsafe)
        labels: (N,) integer labels in {0, 1}
    Returns:
        scalar loss
    """
    safe_mask = (labels == 0)
    unsafe_mask = (labels == 1)

    loss = torch.tensor(0.0, device=scores.device)
    n = 0

    # Safe: score < -MARGIN  → loss = max(0, score + MARGIN)
    if safe_mask.any():
        loss = loss + F.relu(scores[safe_mask] + MARGIN).mean()
        n += 1

    # Unsafe: score > +MARGIN  → loss = max(0, MARGIN - score)
    if unsafe_mask.any():
        loss = loss + F.relu(MARGIN - scores[unsafe_mask]).mean()
        n += 1

    return loss / max(n, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_image_key(features: dict) -> str:
    preferred = ("image0", "observation.image", "observation.images.front")
    for k in preferred:
        if k in features and features[k]["dtype"] == "image":
            return k
    for k, v in features.items():
        if v["dtype"] == "image":
            return k
    raise ValueError("No image observation key found in dataset features.")


def _build_model(cfg: DinoWMConfig, action_dim: int, proprio_dim: int, checkpoint_path: str) -> VWorldModel:
    """Load a trained VWorldModel and attach failure_head."""
    encoder = DinoV2Encoder(name=cfg.encoder_name)
    emb_dim = encoder.emb_dim

    decoder_scale = 16
    num_side = cfg.img_size // decoder_scale
    num_vis_patches = num_side ** 2

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

    decoder = Decoder(
        channel=cfg.decoder_channel,
        n_res_block=cfg.decoder_n_res_block,
        n_res_channel=cfg.decoder_n_res_channel,
        emb_dim=emb_dim,
    )

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
        train_encoder=False,
        train_predictor=False,
        train_decoder=False,
        use_failure_head=True,
        failure_head_hidden_dim=cfg.failure_head_hidden_dim,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # Allow missing failure_head keys (they're newly initialised)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        logging.warning(f"Unexpected keys in checkpoint: {unexpected}")
    if missing:
        non_fh = [k for k in missing if "failure_head" not in k]
        if non_fh:
            logging.warning(f"Missing keys (non-failure-head): {non_fh}")
        logging.info(f"Failure head keys will be randomly initialised: {[k for k in missing if 'failure_head' in k]}")

    return model


def _freeze_except_failure_head(model: VWorldModel) -> None:
    """Freeze everything except the failure_head."""
    for name, param in model.named_parameters():
        param.requires_grad = "failure_head" in name


def _extract_labels(batch: dict, has_labels: bool, batch_size: int, device) -> torch.Tensor:
    # Cached path returns failure_label as (B,); LeRobotDataset returns (B,) too
    # (failure_label isn't in delta_indices, so it's a scalar per sample). Guard
    # against the historical (B, T) shape just in case anyone re-windows it.
    if not (has_labels and "failure_label" in batch):
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    fl = batch["failure_label"]
    if fl.dim() == 2:
        fl = fl[:, -1]
    return fl.long()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    wm_checkpoint: str,
    dataset_repo_id: str,
    steps: int = 10_000,
    batch_size: int = 64,
    lr: float = 1e-4,
    val_frac: float = 0.1,
    output_dir: str = "outputs/classifier",
    device_str: str = "cuda",
    num_workers: int = 4,
    log_freq: int = 100,
    save_freq: int = 1_000,
    use_cached_latents: bool = False,
    cached_latents_dir: str = "runs/dino_wm_exp_merged/latents",
    # Config overrides — must match the world model that was trained
    num_hist: int = 2,
    num_pred: int = 1,
    frameskip: int = 3,
    img_size: int = 224,
    encoder_name: str = "dinov2_vits14",
    action_emb_dim: int = 10,
    proprio_emb_dim: int = 10,
    concat_dim: int = 1,
    predictor_depth: int = 6,
    predictor_heads: int = 16,
    predictor_mlp_dim: int = 2048,
    predictor_dropout: float = 0.1,
    failure_head_hidden_dim: int = 256,
):
    init_logging()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    cfg = DinoWMConfig(
        dataset_repo_id=dataset_repo_id,
        num_hist=num_hist,
        num_pred=num_pred,
        frameskip=frameskip,
        img_size=img_size,
        encoder_name=encoder_name,
        action_emb_dim=action_emb_dim,
        proprio_emb_dim=proprio_emb_dim,
        concat_dim=concat_dim,
        predictor_depth=predictor_depth,
        predictor_heads=predictor_heads,
        predictor_mlp_dim=predictor_mlp_dim,
        predictor_dropout=predictor_dropout,
        failure_head_hidden_dim=failure_head_hidden_dim,
        use_failure_head=True,
    )

    # Dataset
    action_dim  = POLICY_FEATURES["action"]["shape"][-1]
    proprio_dim = POLICY_FEATURES["observation.state"]["shape"][-1]

    if use_cached_latents:
        dataset = CachedLatentDataset(
            cached_latents_dir,
            num_hist=cfg.num_hist,
            num_pred=cfg.num_pred,
            frameskip=cfg.frameskip,
        )
        image_key = None
        policy_features = None
    else:
        image_key = _detect_image_key(POLICY_FEATURES)
        window = cfg.num_hist + cfg.num_pred
        indices = [i * cfg.frameskip for i in range(window)]
        delta_indices: dict = {}
        for k, v in POLICY_FEATURES.items():
            if v["dtype"] == "image":
                delta_indices[k] = indices
        if "observation.state" in POLICY_FEATURES:
            delta_indices["observation.state"] = indices
        if "action" in POLICY_FEATURES:
            delta_indices["action"] = indices

        dataset = LeRobotDataset(
            dataset_repo_id,
            delta_indices=delta_indices,
            image_transforms=T.Resize((cfg.img_size, cfg.img_size), antialias=True),
        )
        policy_features = dataset.policy_features

    has_labels = "failure_label" in (dataset[0] if hasattr(dataset, "__getitem__") else {})
    if not has_labels:
        logging.warning(
            "Dataset does not contain 'failure_label'. All samples will be treated as "
            "safe (label=0). The failure head will not learn to distinguish safe/unsafe."
        )

    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)

    # Model
    model = _build_model(cfg, action_dim, proprio_dim, wm_checkpoint)
    _freeze_except_failure_head(model)
    model = model.to(device)
    model.eval()
    model.failure_head.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable parameters (failure_head only): {trainable:,}")

    optimizer = torch.optim.Adam(model.failure_head.parameters(), lr=lr)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    train_iter = cycle(train_loader)

    for step in range(1, steps + 1):
        # --- Train step ---
        batch = next(train_iter)
        if use_cached_latents:
            batch = to_device(batch, device)
            z = batch["z"].float()                          # (B, window, P, D)
        else:
            batch = normalize(batch, dataset.stats, policy_features)
            batch = to_device(batch, device)
            visual = batch[image_key].float()
            obs = {"visual": visual, "proprio": batch["observation.state"].float()}
            act = batch["action"].float()
            with torch.no_grad():
                z = model.encode(obs, act)                  # (B, T, P, D)

        # Score the encoded latent of the last frame in the window. The model's
        # failure path strips action features before pooling, so this is a
        # state-only visual+proprio score even though z is action-conditioned.
        # The dataset aligns failure_label to win[-1], so this index pairs the
        # head's input with its target. No transition forward is needed —
        # predict() outputs are only used at deployment time.
        scores = model.predict_failure(z[:, -1:]).squeeze(-1)  # (B, 1)
        scores_last = scores[:, -1]                            # (B,)

        labels = _extract_labels(batch, has_labels, scores_last.shape[0], device)
        loss = fail_loss(scores_last, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Logging ---
        if step % log_freq == 0:
            logging.info(f"[step {step}/{steps}] train_loss={loss.item():.4f}")

        # --- Validation ---
        if step % save_freq == 0:
            model.failure_head.eval()
            val_losses = []
            with torch.no_grad():
                for vbatch in val_loader:
                    if use_cached_latents:
                        vbatch = to_device(vbatch, device)
                        vz = vbatch["z"].float()
                    else:
                        vbatch = normalize(vbatch, dataset.stats, policy_features)
                        vbatch = to_device(vbatch, device)
                        vvisual = vbatch[image_key].float()
                        vobs = {"visual": vvisual, "proprio": vbatch["observation.state"].float()}
                        vact = vbatch["action"].float()
                        vz = model.encode(vobs, vact)
                    vscores = model.predict_failure(vz[:, -1:]).squeeze(-1)[:, -1]
                    vlabels = _extract_labels(vbatch, has_labels, vscores.shape[0], device)
                    val_losses.append(fail_loss(vscores, vlabels).item())

            val_loss = float(np.mean(val_losses))
            logging.info(f"[step {step}] val_loss={val_loss:.4f}")

            ckpt = output_path / f"classifier_step{step:06d}.pt"
            torch.save(model.state_dict(), ckpt)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best = output_path / "classifier_best.pt"
                torch.save(model.state_dict(), best)
                logging.info(f"  → New best val_loss={best_val_loss:.4f}, saved to {best}")

            model.failure_head.train()

    logging.info("Classifier training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Fine-tune failure_head of VWorldModel")
    p.add_argument("--wm_checkpoint", required=True, help="Path to VWorldModel model.pt")
    repo_root = Path(__file__).resolve().parents[1]
    p.add_argument("--dataset_repo_id", default=str(repo_root / "data" / "exp_merged"))
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output_dir", default="outputs/classifier")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_freq", type=int, default=100)
    p.add_argument("--save_freq", type=int, default=1_000)
    p.add_argument(
        "--use_cached_latents",
        action="store_true",
        help="Load pre-computed DINO-WM latents from --cached_latents_dir instead "
             "of re-encoding images each step. Requires scripts/cache_latents.py "
             "to have been run first with --store_full_patches.",
    )
    p.add_argument(
        "--cached_latents_dir",
        default="runs/dino_wm_exp_merged/latents",
        help="Directory containing episode_NNN.pt + manifest.json from cache_latents.py.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        wm_checkpoint=args.wm_checkpoint,
        dataset_repo_id=args.dataset_repo_id,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        device_str=args.device,
        num_workers=args.num_workers,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        use_cached_latents=args.use_cached_latents,
        cached_latents_dir=args.cached_latents_dir,
    )
