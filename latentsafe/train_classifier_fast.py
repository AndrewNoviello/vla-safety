"""Fast failure-head training using pre-computed pooled latents.

Reads from a latent cache produced by precompute_latents.py. Each training step
is just failure_head(pooled_pred) → loss → backward — no DINOv2, no predict().

Usage:
    python -m latentsafe.train_classifier_fast \
        --latent_cache outputs/latent_cache/v3_latents.pt \
        --wm_checkpoint outputs/dino_wm_v3/checkpoints/00060000/model.pt
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dino_wm.config import DinoWMConfig
from latentsafe.latent_dataset import LatentDataset
from latentsafe.train_classifier import (
    _build_model,
    _freeze_except_failure_head,
    fail_loss,
)
from data.utils import POLICY_FEATURES
from utils.utils import init_logging

logging.basicConfig(level=logging.INFO)


def train(
    latent_cache: str,
    wm_checkpoint: str,
    steps: int = 10_000,
    batch_size: int = 1024,
    lr: float = 1e-4,
    val_frac: float = 0.1,
    output_dir: str = "outputs/classifier_fast",
    device_str: str = "cuda",
    num_workers: int = 4,
    log_freq: int = 100,
    save_freq: int = 1_000,
):
    init_logging()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    dataset = LatentDataset(latent_cache)
    cfg_dict = dataset.config

    # Reconstruct model config from cache so architecture matches checkpoint
    num_hist                = cfg_dict.get("num_hist",                2)
    num_pred                = cfg_dict.get("num_pred",                1)
    img_size                = cfg_dict.get("img_size",                224)
    encoder_name            = cfg_dict.get("encoder_name",            "dinov2_vits14")
    action_emb_dim          = cfg_dict.get("action_emb_dim",          10)
    proprio_emb_dim         = cfg_dict.get("proprio_emb_dim",         10)
    concat_dim              = cfg_dict.get("concat_dim",              1)
    predictor_depth         = cfg_dict.get("predictor_depth",         6)
    predictor_heads         = cfg_dict.get("predictor_heads",         16)
    predictor_mlp_dim       = cfg_dict.get("predictor_mlp_dim",       2048)
    predictor_dropout       = cfg_dict.get("predictor_dropout",       0.1)
    failure_head_hidden_dim = cfg_dict.get("failure_head_hidden_dim", 256)

    cfg = DinoWMConfig(
        num_hist=num_hist,
        num_pred=num_pred,
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

    action_dim  = POLICY_FEATURES["action"]["shape"][-1]
    proprio_dim = POLICY_FEATURES["observation.state"]["shape"][-1]

    n_val   = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False,
                              pin_memory=(device.type == "cuda"))

    # Only the failure_head is needed at training time; load full model so
    # checkpoint weights and architecture are correct.
    model = _build_model(cfg, action_dim, proprio_dim, wm_checkpoint)
    _freeze_except_failure_head(model)
    model = model.to(device)
    model.failure_head.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable parameters (failure_head only): {trainable:,}")

    optimizer = torch.optim.Adam(model.failure_head.parameters(), lr=lr)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info("=== Fast Classifier Training (pooled latent cache) ===")
    logging.info(f"  Cache:   {latent_cache}  ({len(dataset)} samples)")
    logging.info(f"  Split:   train={n_train}  val={n_val}")
    logging.info(f"  Steps:   {steps}  batch={batch_size}  lr={lr}")
    logging.info(f"  Device:  {device}")
    logging.info(f"  Output:  {output_path}")

    best_val_loss = float("inf")

    def _iter_loader(loader):
        while True:
            yield from loader

    train_iter = _iter_loader(train_loader)

    for step in range(1, steps + 1):
        batch  = next(train_iter)
        pooled = batch["pooled_pred"].to(device, non_blocking=True)   # (B, D)
        labels = batch["failure_label"].to(device, non_blocking=True)  # (B,)

        scores = model.failure_head(pooled).squeeze(-1)   # (B,)
        loss   = fail_loss(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_freq == 0:
            safe_mask   = (labels == 0)
            unsafe_mask = (labels > 0)
            safe_score   = scores[safe_mask].mean().item()   if safe_mask.any()   else float("nan")
            unsafe_score = scores[unsafe_mask].mean().item() if unsafe_mask.any() else float("nan")
            logging.info(
                f"[step {step}/{steps}] loss={loss.item():.4f}"
                f"  safe_score={safe_score:+.3f} (n={safe_mask.sum().item()})"
                f"  unsafe_score={unsafe_score:+.3f} (n={unsafe_mask.sum().item()})"
            )

        if step % save_freq == 0:
            model.failure_head.eval()
            val_losses, all_vscores, all_vlabels = [], [], []
            with torch.no_grad():
                for vbatch in val_loader:
                    vpooled  = vbatch["pooled_pred"].to(device, non_blocking=True)
                    vlabels  = vbatch["failure_label"].to(device, non_blocking=True)
                    vscores  = model.failure_head(vpooled).squeeze(-1)
                    val_losses.append(fail_loss(vscores, vlabels).item())
                    all_vscores.append(vscores.cpu())
                    all_vlabels.append(vlabels.cpu())

            val_loss      = float(np.mean(val_losses))
            all_vscores_t = torch.cat(all_vscores)
            all_vlabels_t = torch.cat(all_vlabels)
            vsafe_mask    = (all_vlabels_t == 0)
            vunsafe_mask  = (all_vlabels_t > 0)
            val_accuracy     = ((all_vscores_t > 0).long() == (all_vlabels_t > 0).long()).float().mean().item()
            val_safe_score   = all_vscores_t[vsafe_mask].mean().item()   if vsafe_mask.any()   else float("nan")
            val_unsafe_score = all_vscores_t[vunsafe_mask].mean().item() if vunsafe_mask.any() else float("nan")

            logging.info(
                f"[step {step}] val_loss={val_loss:.4f}  accuracy={val_accuracy:.3f}"
                f"  safe_score={val_safe_score:+.3f} (n={vsafe_mask.sum().item()})"
                f"  unsafe_score={val_unsafe_score:+.3f} (n={vunsafe_mask.sum().item()})"
            )

            ckpt = output_path / f"classifier_step{step:06d}.pt"
            torch.save(model.state_dict(), ckpt)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best = output_path / "classifier_best.pt"
                torch.save(model.state_dict(), best)
                logging.info(f"  → New best val_loss={best_val_loss:.4f}, saved to {best}")

            model.failure_head.train()

    logging.info("Classifier training complete.")


def _parse_args():
    p = argparse.ArgumentParser(description="Fast failure-head training from pooled latent cache")
    p.add_argument("--latent_cache",  required=True, help="Path to .pt file from precompute_latents.py")
    p.add_argument("--wm_checkpoint", required=True, help="Path to VWorldModel model.pt")
    p.add_argument("--steps",       type=int,   default=10_000)
    p.add_argument("--batch_size",  type=int,   default=1024)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--output_dir",  default="outputs/classifier_fast")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--log_freq",    type=int,   default=100)
    p.add_argument("--save_freq",   type=int,   default=1_000)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        latent_cache=args.latent_cache,
        wm_checkpoint=args.wm_checkpoint,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        device_str=args.device,
        num_workers=args.num_workers,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
    )
