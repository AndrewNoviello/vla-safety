"""Cache DINO-WM latents for the entire training dataset.

For each frame in the dataset, computes z = model.encode(obs, act) with T=1
and writes per-episode .pt files containing both full patch latents and
mean-pooled latents, plus failure_label and frame_index metadata. Streams
writes per-episode so peak memory is one episode worth of latents.

Coverage guarantee: every frame in the dataset is encoded. The script
fails loud if any episode is short or any episode is missing.

Usage:
    python scripts/cache_latents.py \\
        --wm_checkpoint runs/dino_wm_exp_merged/checkpoints/latest/model.pt \\
        --output_dir   runs/dino_wm_exp_merged/latents
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from data.lerobot_dataset import LeRobotDataset
from data.utils import POLICY_FEATURES
from dino_wm.train import CFG, _build_model
from utils.processor_utils import normalize, to_device
from utils.utils import init_logging

logger = logging.getLogger(__name__)


def _detect_image_key(features: dict) -> str:
    preferred = ("image0", "observation.image", "observation.images.front")
    for k in preferred:
        if k in features and features[k]["dtype"] == "image":
            return k
    for k, v in features.items():
        if v["dtype"] == "image":
            return k
    raise ValueError("No image observation key in features.")


def _flush_episode(
    out_dir: Path,
    ep_idx: int,
    buf: dict,
    store_full_patches: bool,
) -> int:
    """Sort buffered frames by frame_index, write episode .pt, return frame count."""
    order = sorted(range(len(buf["frame_index"])), key=lambda i: buf["frame_index"][i])
    z_mean = torch.stack([buf["z_mean"][i] for i in order])
    out: dict[str, torch.Tensor] = {
        "z_mean": z_mean,
        "failure_label": torch.tensor(
            [buf["failure_label"][i] for i in order], dtype=torch.int8
        ),
        "frame_index": torch.tensor(
            [buf["frame_index"][i] for i in order], dtype=torch.int64
        ),
        "index": torch.tensor(
            [buf["index"][i] for i in order], dtype=torch.int64
        ),
        "episode_index": torch.tensor(ep_idx, dtype=torch.int64),
    }
    if store_full_patches:
        out["z"] = torch.stack([buf["z"][i] for i in order])
    torch.save(out, out_dir / f"episode_{ep_idx:03d}.pt")
    return z_mean.shape[0]


def cache_latents(
    wm_checkpoint: str,
    dataset_repo_id: str,
    output_dir: str,
    *,
    store_full_patches: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
    max_episodes: int | None = None,
):
    init_logging()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    image_key = _detect_image_key(POLICY_FEATURES)
    delta_indices = {
        "observation.state": [0],
        "action": [0],
        image_key: [0],
    }
    dataset = LeRobotDataset(
        dataset_repo_id,
        delta_indices=delta_indices,
        image_transforms=T.Resize((CFG.img_size, CFG.img_size), antialias=True),
    )
    policy_features = dataset.policy_features
    action_dim = POLICY_FEATURES["action"]["shape"][-1]
    proprio_dim = POLICY_FEATURES["observation.state"]["shape"][-1]

    logger.info(
        f"Dataset: {dataset.num_frames} frames across {dataset.num_episodes} episodes"
    )

    model, _ = _build_model(CFG, action_dim, proprio_dim)
    state_dict = torch.load(wm_checkpoint, map_location=dev, weights_only=True)
    # failure_head may be missing from a vanilla WM checkpoint — load non-strict.
    model.load_state_dict(state_dict, strict=False)
    model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Expected per-episode lengths from the dataset's episode boundaries.
    # _episode_boundaries: {ep_idx: (start, end)} in absolute frame index.
    expected_len = {ep: end - start for ep, (start, end) in dataset._episode_boundaries.items()}

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=dev.type == "cuda",
    )

    # Streaming buffers. Episodes are visited contiguously (shuffle=False), but
    # a batch can straddle an episode boundary, so we keep all currently-active
    # episode buffers and flush when their length matches expected_len[ep].
    buffers: dict[int, dict] = {}
    flushed: set[int] = set()
    total_cached = 0
    P_dim: int | None = None
    D_dim: int | None = None

    for batch in loader:
        batch = normalize(batch, dataset.stats, policy_features)
        batch = to_device(batch, dev)

        visual = batch[image_key].float()              # (B, 1, 3, H, W)
        proprio = batch["observation.state"].float()   # (B, 1, 6)
        act = batch["action"].float()                  # (B, 1, 6)
        obs = {"visual": visual, "proprio": proprio}

        with torch.no_grad():
            z = model.encode(obs, act)  # (B, 1, P, D)

        z = z.squeeze(1)             # (B, P, D)
        z_mean = z.mean(dim=-2)      # (B, D)
        if P_dim is None:
            P_dim, D_dim = int(z.shape[1]), int(z.shape[2])

        z_h = z.to(torch.float16).cpu() if store_full_patches else None
        zm_h = z_mean.to(torch.float16).cpu()

        ep_ids = batch["episode_index"].view(-1).cpu().tolist()
        frm_ids = batch["frame_index"].view(-1).cpu().tolist()
        idx_ids = batch["index"].view(-1).cpu().tolist()
        fl = batch["failure_label"].view(-1).to(torch.int8).cpu().tolist()

        for i in range(len(ep_ids)):
            ep = int(ep_ids[i])
            if ep in flushed:
                raise AssertionError(
                    f"Frame for episode {ep} arrived after that episode was flushed — "
                    f"dataset is not in episode-contiguous order."
                )
            buf = buffers.setdefault(ep, {
                "z_mean": [],
                "failure_label": [],
                "frame_index": [],
                "index": [],
                **({"z": []} if store_full_patches else {}),
            })
            buf["z_mean"].append(zm_h[i])
            buf["failure_label"].append(int(fl[i]))
            buf["frame_index"].append(int(frm_ids[i]))
            buf["index"].append(int(idx_ids[i]))
            if store_full_patches:
                buf["z"].append(z_h[i])

            if len(buf["z_mean"]) == expected_len.get(ep, -1):
                n = _flush_episode(out_dir, ep, buf, store_full_patches)
                total_cached += n
                flushed.add(ep)
                del buffers[ep]
                if max_episodes is not None and len(flushed) >= max_episodes:
                    break

        if max_episodes is not None and len(flushed) >= max_episodes:
            break

    # Coverage / integrity assertions (skipped under --max_episodes)
    if max_episodes is None:
        if buffers:
            raise AssertionError(
                f"Some episodes never reached their expected length: "
                f"{sorted(buffers.keys())}"
            )
        if total_cached != dataset.num_frames:
            raise AssertionError(
                f"Cached {total_cached} frames but dataset has {dataset.num_frames}."
            )
        if len(flushed) != dataset.num_episodes:
            raise AssertionError(
                f"Cached {len(flushed)} episodes but dataset has {dataset.num_episodes}."
            )
        # Spot-check NaN/inf and verify frame_index is dense [0..T-1] in episode 0.
        ep0 = sorted(flushed)[0]
        d0 = torch.load(out_dir / f"episode_{ep0:03d}.pt", weights_only=True)
        if not torch.isfinite(d0["z_mean"]).all():
            raise AssertionError(f"Episode {ep0}: z_mean contains NaN/Inf")
        fi = d0["frame_index"].tolist()
        if fi != list(range(len(fi))):
            raise AssertionError(
                f"Episode {ep0}: frame_index gaps. Got {fi[:5]}...{fi[-5:]}"
            )

    manifest = {
        "wm_checkpoint": str(Path(wm_checkpoint).resolve()),
        "dataset_repo_id": dataset_repo_id,
        "num_frames": total_cached,
        "num_episodes": len(flushed),
        "predictor_dim": D_dim,
        "num_patches": P_dim,
        "store_full_patches": store_full_patches,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if store_full_patches and P_dim and D_dim:
        size_gb = total_cached * P_dim * D_dim * 2 / 1e9
    else:
        size_gb = total_cached * (D_dim or 0) * 2 / 1e9
    logger.info(
        f"Cached {total_cached} frames across {len(flushed)} episodes → {out_dir} "
        f"({size_gb:.2f} GB)."
    )


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--wm_checkpoint",
        default="runs/dino_wm_exp_merged/checkpoints/latest/model.pt",
    )
    p.add_argument("--dataset_repo_id", default=CFG.dataset_repo_id)
    p.add_argument("--output_dir", default="runs/dino_wm_exp_merged/latents")
    p.add_argument(
        "--store_full_patches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store full per-patch z (default True). Pass --no-store_full_patches "
             "to skip the ~13 GB tensor and only cache pooled z_mean.",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--max_episodes", type=int, default=None,
        help="Cache only the first N episodes (skips coverage assertions).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cache_latents(
        wm_checkpoint=args.wm_checkpoint,
        dataset_repo_id=args.dataset_repo_id,
        output_dir=args.output_dir,
        store_full_patches=args.store_full_patches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_episodes=args.max_episodes,
    )
