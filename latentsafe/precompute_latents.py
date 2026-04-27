"""Pre-compute and cache pooled predictor latents for faster failure-head training.

Runs encode() + predict() + mean-pool on every dataset sample once, writing
results to disk via numpy memmap (zero RAM accumulation). The output is a ~464 MB
.pt file instead of the ~261 GB that storing full patch tensors would require.

Usage:
    python -m latentsafe.precompute_latents \
        --wm_checkpoint outputs/dino_wm_v3/checkpoints/00060000/model.pt \
        --dataset_repo_id various-and-sundry/domino-world-v3-labeled \
        --output_path outputs/latent_cache/v3_latents.pt
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as Tv2

from data.lerobot_dataset import LeRobotDataset
from data.utils import POLICY_FEATURES, dataset_to_policy_features
from dino_wm.config import DinoWMConfig
from latentsafe.train_classifier import NORM_MAP, _build_model, _detect_image_key
from utils.processor_utils import normalize, to_device
from utils.utils import init_logging

logging.basicConfig(level=logging.INFO)


def precompute(
    wm_checkpoint: str,
    dataset_repo_id: str,
    output_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    device_str: str = "cuda",
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

    # Dataset setup — identical to train_classifier.py
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
        image_transforms=Tv2.Resize((cfg.img_size, cfg.img_size), antialias=True),
    )
    policy_features = dataset_to_policy_features(POLICY_FEATURES)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = _build_model(cfg, action_dim, proprio_dim, wm_checkpoint)
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)
    model.eval()

    N = len(dataset)
    D = model.emb_dim

    logging.info(f"Dataset: {N} frames, {dataset.num_episodes} episodes")
    logging.info(f"Window:  {window} frames (num_hist={num_hist}, num_pred={num_pred}, frameskip={frameskip})")
    logging.info(f"Device:  {device}")
    logging.info(f"Output:  pooled_pred ({N}, {D})  ≈ {N * D * 4 / 1e6:.0f} MB")

    has_labels = "safety" in dataset.hf_dataset.column_names
    if not has_labels:
        logging.warning("Dataset has no 'safety' column — all labels will be 0 (safe).")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tmp_z   = Path(str(out) + ".z.tmp")
    tmp_lbl = Path(str(out) + ".lbl.tmp")

    # Pre-allocate on-disk memmaps — each batch writes in-place, no RAM accumulation
    mmap_z     = np.memmap(tmp_z,   dtype="float32", mode="w+", shape=(N, D))
    mmap_label = np.memmap(tmp_lbl, dtype="int64",   mode="w+", shape=(N,))

    n_batches = len(loader)
    offset = 0
    t0 = time.monotonic()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = normalize(batch, dataset.stats, policy_features, NORM_MAP)
            batch = to_device(batch, device)

            visual = batch[image_key].float()
            obs    = {"visual": visual, "proprio": batch["observation.state"].float()}
            act    = batch["action"].float()

            z      = model.encode(obs, act)         # (B, T, P, D)
            z_src  = z[:, :num_hist]                # (B, num_hist, P, D)
            z_pred = model.predict(z_src)           # (B, num_hist, P, D)
            pooled = z_pred.mean(dim=2)[:, -1]      # (B, D) — pool patches, last timestep

            if has_labels and "failure_label" in batch:
                label = batch["failure_label"][:, -1].long()
            else:
                label = torch.zeros(pooled.shape[0], dtype=torch.long, device=device)

            b = pooled.shape[0]
            mmap_z[offset:offset + b]     = pooled.cpu().numpy()
            mmap_label[offset:offset + b] = label.cpu().numpy()
            offset += b

            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                elapsed = time.monotonic() - t0
                batches_done = i + 1
                eta = elapsed / batches_done * (n_batches - batches_done)
                logging.info(
                    f"  [{batches_done}/{n_batches}]"
                    f"  elapsed {elapsed:.0f}s"
                    f"  ETA {eta:.0f}s"
                    f"  ({batches_done / elapsed:.1f} batches/s)"
                )

    logging.info(f"Encoding done ({offset} samples). Converting mmap → .pt ...")

    cfg_dict = dict(
        num_hist=num_hist, num_pred=num_pred, frameskip=frameskip,
        img_size=img_size, encoder_name=encoder_name,
        action_emb_dim=action_emb_dim, proprio_emb_dim=proprio_emb_dim,
        concat_dim=concat_dim, predictor_depth=predictor_depth,
        predictor_heads=predictor_heads, predictor_mlp_dim=predictor_mlp_dim,
        predictor_dropout=predictor_dropout,
        failure_head_hidden_dim=failure_head_hidden_dim,
        dataset_repo_id=dataset_repo_id,
        wm_checkpoint=wm_checkpoint,
        emb_dim=D,
    )
    torch.save(
        {
            "pooled_pred":   torch.from_numpy(np.array(mmap_z[:offset])),
            "failure_label": torch.from_numpy(np.array(mmap_label[:offset])),
            "config": cfg_dict,
        },
        out,
    )

    # Clean up temp files
    del mmap_z, mmap_label
    tmp_z.unlink()
    tmp_lbl.unlink()

    size_mb = out.stat().st_size / 1e6
    logging.info(f"Saved → {out}  ({size_mb:.1f} MB)")
    logging.info(f"pooled_pred shape: ({offset}, {D})")


def _parse_args():
    p = argparse.ArgumentParser(description="Pre-compute pooled predictor latents")
    p.add_argument("--wm_checkpoint", required=True)
    p.add_argument("--dataset_repo_id", default="various-and-sundry/domino-world-v3-labeled")
    p.add_argument("--output_path", default="outputs/latent_cache/v3_latents.pt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    precompute(
        wm_checkpoint=args.wm_checkpoint,
        dataset_repo_id=args.dataset_repo_id,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_str=args.device,
    )
