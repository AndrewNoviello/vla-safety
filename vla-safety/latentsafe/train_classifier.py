"""Fine-tune the failure_head of a trained VWorldModel.

All parameters except failure_head are frozen. Training uses a margin-based
ranking loss over three label classes:
  0 = safe        → score should be < -margin  (head outputs a negative value)
  1 = unsafe      → score should be > +margin
  2 = weakly-unsafe → score should be > +gamma*margin  (softer constraint)

Usage
-----
Prepare a dataset that is the same format as the world-model training dataset,
but with an extra "failure_label" tensor of shape (T,) in {0, 1, 2} per sample.
Then call:

    python -m latentsafe.train_classifier \
        --wm_checkpoint outputs/dino_wm_v2/checkpoints/latest/model.pt \
        --dataset_repo_id AndrewNoviello/domino-world-v2 \
        --steps 10000

Architecture note
-----------------
The failure_head was added to VWorldModel with use_failure_head=True.
It takes the mean-pooled predictor latent (B, T, predictor_dim) → (B, T, 1).
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
from data.utils import POLICY_FEATURES, cycle, dataset_to_policy_features
from utils.types import FeatureType, NormalizationMode
from utils.processor_utils import normalize, to_device
from utils.utils import init_logging

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Margin-based ranking loss (mirrors reference train_dino_classifier.py)
# ---------------------------------------------------------------------------

MARGIN = 1.0
GAMMA = 0.75   # softer margin for weakly-unsafe


def fail_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Margin ranking loss for three-class failure labels.

    Args:
        scores: (N,) predicted failure scores (higher = more unsafe)
        labels: (N,) integer labels in {0, 1, 2}
    Returns:
        scalar loss
    """
    safe_mask   = (labels == 0)
    unsafe_mask = (labels == 1)
    weak_mask   = (labels == 2)

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

    # Weakly-unsafe: score > +GAMMA*MARGIN  → loss = max(0, GAMMA*MARGIN - score)
    if weak_mask.any():
        loss = loss + F.relu(GAMMA * MARGIN - scores[weak_mask]).mean()
        n += 1

    return loss / max(n, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NORM_MAP = {
    FeatureType.VISUAL: NormalizationMode.IDENTITY,
    FeatureType.STATE:  NormalizationMode.MEAN_STD,
    FeatureType.ACTION: NormalizationMode.MEAN_STD,
}


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

    action_dim  = POLICY_FEATURES["action"]["shape"][-1]
    proprio_dim = POLICY_FEATURES["observation.state"]["shape"][-1]

    dataset = LeRobotDataset(
        dataset_repo_id,
        delta_indices=delta_indices,
        image_transforms=T.Resize((cfg.img_size, cfg.img_size), antialias=True),
    )
    policy_features = dataset_to_policy_features(POLICY_FEATURES)

    # NOTE: The dataset must contain a "failure_label" key with integer labels {0,1,2}.
    # If it does not, all samples will be treated as safe (label=0) and the classifier
    # will not learn anything useful. Label your trajectories before running this script.
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
        batch = normalize(batch, dataset.stats, policy_features, NORM_MAP)
        batch = to_device(batch, device)

        visual = batch[image_key].float()
        obs = {"visual": visual, "proprio": batch["observation.state"].float()}
        act = batch["action"].float()

        with torch.no_grad():
            z = model.encode(obs, act)                      # (B, T, P, D)
            z_src = z[:, : cfg.num_hist]
            z_pred = model.predict(z_src)                   # (B, T, P, D)

        scores = model.predict_failure(z_pred).squeeze(-1)  # (B, T)
        # Use last predicted timestep for loss
        scores_last = scores[:, -1]                          # (B,)

        if has_labels and "failure_label" in batch:
            labels = batch["failure_label"][:, -1].long()   # (B,) — label at last frame
        else:
            labels = torch.zeros(scores_last.shape[0], dtype=torch.long, device=device)

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
                    vbatch = normalize(vbatch, dataset.stats, policy_features, NORM_MAP)
                    vbatch = to_device(vbatch, device)
                    vvisual = vbatch[image_key].float()
                    vobs = {"visual": vvisual, "proprio": vbatch["observation.state"].float()}
                    vact = vbatch["action"].float()
                    vz = model.encode(vobs, vact)
                    vz_src = vz[:, : cfg.num_hist]
                    vz_pred = model.predict(vz_src)
                    vscores = model.predict_failure(vz_pred).squeeze(-1)[:, -1]
                    if has_labels and "failure_label" in vbatch:
                        vlabels = vbatch["failure_label"][:, -1].long()
                    else:
                        vlabels = torch.zeros(vscores.shape[0], dtype=torch.long, device=device)
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
    p.add_argument("--dataset_repo_id", default="AndrewNoviello/domino-world-v2")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output_dir", default="outputs/classifier")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_freq", type=int, default=100)
    p.add_argument("--save_freq", type=int, default=1_000)
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
    )
