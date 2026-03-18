"""dino_wm_test.py — Compare real trajectory vs. DINO world-model rollout.

Picks one episode from the training dataset, runs the world model
autoregressively using only the real actions (never feeding future real frames
back in), and saves a side-by-side MP4:

    left  = original frames from the dataset
    right = frames decoded from world-model predictions

The first cfg.num_hist frames on the right are *reconstructions* of the seed
frames (encoder → predictor → decoder path); all subsequent frames are pure
world-model predictions.

Usage:
    python dino_wm_test.py [options]

Options:
    --hf-repo       HuggingFace model repo (default: from training config)
    --checkpoint    HF filename or local .pt path (default: checkpoints/latest/model.pt)
    --episode       Episode index to evaluate (default: 0)
    --output        Output MP4 path (default: wm_comparison_ep<N>.mp4)
    --device        cuda or cpu (default: auto)
    --save-frames   Also save individual PNG frames alongside the video
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2 as TVT  # avoid shadowing frame-count var T

from lerobot.configs.dino_wm_config import DinoWMConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import POLICY_FEATURES, dataset_to_policy_features
from lerobot.types import FeatureType, NormalizationMode
from lerobot.utils.processor_utils import normalize

# dino_wm_train sets up sys.path for dino_wm models and re-exports CFG etc.
from dino_wm_train import CFG, DINO_WM_NORM_MAP

# Inference wrapper (also ensures dino_wm models are importable)
from dino_wm_inference import DinoWMInference, _ACTION_DIM, _PROPRIO_DIM


def load_episode(
    dataset: LeRobotDataset,
    episode_idx: int,
    cfg: DinoWMConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load one episode at frameskip intervals, matching training preprocessing.

    Images are resized to cfg.img_size × cfg.img_size and kept in [0, 1].
    States and actions are returned *unnormalized* (raw dataset values).

    Returns:
        images:  (T, 3, img_size, img_size)  float [0, 1]
        states:  (T, proprio_dim)            float  (unnormalized)
        actions: (T, action_dim)             float  (unnormalized)
    """
    if episode_idx not in dataset._episode_boundaries:
        raise ValueError(
            f"Episode {episode_idx} not found. "
            f"Available indices: 0 .. {dataset.num_episodes - 1}"
        )

    ep_start, ep_end = dataset._episode_boundaries[episode_idx]
    ep_len = ep_end - ep_start

    # Absolute dataset indices at frameskip intervals
    abs_indices = [ep_start + i for i in range(0, ep_len, cfg.frameskip)]
    num_frames = len(abs_indices)

    # frame_index values inside the episode's MP4 (may differ from abs_indices)
    frame_indices_in_video = [
        int(dataset.hf_dataset[i]["frame_index"]) for i in abs_indices
    ]

    # Load raw video frames → resize to training resolution
    images_raw = dataset._load_video_frames(episode_idx, frame_indices_in_video)
    # images_raw: (T, 3, orig_H, orig_W)  float [0, 1]
    resize = TVT.Resize((cfg.img_size, cfg.img_size), antialias=True)
    images = resize(images_raw)  # (T, 3, img_size, img_size)

    # Load state and action from the parquet (via HF dataset)
    batch = dataset.hf_dataset[abs_indices]
    states = torch.stack(
        [torch.tensor(x).float() for x in batch["observation.state"]]
    )   # (T, 6)
    actions = torch.stack(
        [torch.tensor(x).float() for x in batch["action"]]
    )   # (T, 6)

    return images, states, actions


def _to_uint8_hwc(frames: torch.Tensor) -> np.ndarray:
    """Convert (T, 3, H, W) float tensor → (T, H, W, 3) uint8 numpy."""
    frames = frames.detach().cpu().clamp(0.0, 1.0)
    return (frames.permute(0, 2, 3, 1) * 255).byte().numpy()


def save_comparison_video(
    real_frames: torch.Tensor,   # (T, 3, H, W)  float [0, 1]
    wm_frames: torch.Tensor,     # (T, 3, H, W)  float (may be outside [0, 1])
    output_path: Path,
    fps: float = 10.0,
    save_frames: bool = False,
) -> None:
    """Write a side-by-side comparison MP4 (real | world-model)."""
    from torchvision.io import write_video

    real_np = _to_uint8_hwc(real_frames)   # (T, H,   W, 3)
    wm_np   = _to_uint8_hwc(wm_frames)    # (T, H,   W, 3)

    # Concatenate horizontally: (T, H, 2*W, 3)
    combined = np.concatenate([real_np, wm_np], axis=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(
        str(output_path),
        torch.from_numpy(combined),
        fps=fps,
        video_codec="libx264",
        options={"crf": "18"},
    )
    print(f"Saved comparison video ({len(real_np)} frames @ {fps:.1f} fps): {output_path}")

    if save_frames:
        from PIL import Image as PILImage
        frames_dir = output_path.with_suffix("")
        frames_dir.mkdir(parents=True, exist_ok=True)
        for t, frame in enumerate(combined):
            PILImage.fromarray(frame).save(frames_dir / f"frame_{t:04d}.png")
        print(f"  Saved {len(combined)} PNG frames to {frames_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Compare real trajectory vs. DINO world-model on one episode."
    )
    parser.add_argument(
        "--hf-repo", default=CFG.hf_model_repo_id,
        help="HuggingFace model repo ID (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/latest/model.pt",
        help="HF filename or local .pt path to model weights (default: %(default)s)",
    )
    parser.add_argument(
        "--episode", type=int, default=0,
        help="Episode index to evaluate (default: %(default)s)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output MP4 path (default: wm_comparison_ep<N>.mp4)",
    )
    parser.add_argument(
        "--device", default=None,
        help="'cuda' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--save-frames", action="store_true",
        help="Also save individual PNG frames alongside the video",
    )
    args = parser.parse_args()

    output_path = Path(args.output or f"wm_comparison_ep{args.episode:03d}.mp4")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading dataset: {CFG.dataset_repo_id}")
    dataset = LeRobotDataset(
        CFG.dataset_repo_id,
        delta_indices=None,
        image_transforms=None,   # we resize manually in load_episode()
    )
    policy_features = dataset_to_policy_features(POLICY_FEATURES)
    print(f"  {dataset.num_episodes} episodes, {dataset.num_frames} total frames")

    print(f"\nLoading episode {args.episode} (frameskip={CFG.frameskip}) ...")
    images, states, actions = load_episode(dataset, args.episode, CFG)
    num_frames = images.shape[0]
    print(f"  Episode length: {num_frames} frames (at frameskip={CFG.frameskip})")

    if num_frames < CFG.num_hist + 1:
        raise RuntimeError(
            f"Episode {args.episode} is too short ({num_frames} frames at "
            f"frameskip={CFG.frameskip}). Need at least "
            f"num_hist+1 = {CFG.num_hist + 1} frames."
        )

    norm_batch = normalize(
        {"observation.state": states, "action": actions},
        dataset.stats,
        policy_features,
        DINO_WM_NORM_MAP,
    )
    states_norm  = norm_batch["observation.state"]   # (T, 6)
    actions_norm = norm_batch["action"]              # (T, 6)

    obs_0 = {
        # Seed frames: first num_hist real observations
        "visual":  images[:CFG.num_hist].unsqueeze(0).to(device),   # (1, num_hist, 3, H, W)
        "proprio": states_norm[:CFG.num_hist].unsqueeze(0).to(device),  # (1, num_hist, 6)
    }
    # Actions for ALL frames (seed + future); rollout() splits internally
    all_actions = actions_norm.unsqueeze(0).to(device)   # (1, T, 6)

    local_pt = args.checkpoint if Path(args.checkpoint).exists() else None
    hf_filename = args.checkpoint if local_pt is None else None

    print("\nLoading world model ...")
    inference = DinoWMInference(
        hf_repo_id=args.hf_repo,
        checkpoint_filename=hf_filename or "checkpoints/latest/model.pt",
        local_model_path=local_pt,
        device=device,
    )

    print("Running autoregressive rollout ...")
    _z_obses, wm_visual = inference.rollout(obs_0, all_actions)
    # wm_visual: (1, num_frames + 1, 3, H, W)  — rollout yields one extra step
    wm_frames = wm_visual[0, :num_frames]   # (T, 3, H, W)

    effective_fps = 30.0 / CFG.frameskip
    save_comparison_video(
        real_frames=images,
        wm_frames=wm_frames,
        output_path=output_path,
        fps=effective_fps,
        save_frames=args.save_frames,
    )

    print(
        f"\nSummary:"
        f"\n  Episode:          {args.episode}"
        f"\n  Frames compared:  {num_frames} @ frameskip={CFG.frameskip} "
        f"(~{effective_fps:.1f} fps effective)"
        f"\n  Seed frames:      {CFG.num_hist}  (right panel = reconstruction)"
        f"\n  Predicted frames: {num_frames - CFG.num_hist}  (right panel = WM prediction)"
        f"\n  Output video:     {output_path}"
    )


if __name__ == "__main__":
    main()
