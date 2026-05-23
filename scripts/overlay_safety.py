"""Overlay safe/unsafe classifier scores on an episode video.

For each frame of the requested episode, runs the trained failure_head on
the cached DINO-WM latent and burns a SAFE/UNSAFE badge + score onto the
source video. Reuses `runs/dino_wm_exp_merged/latents/` so DINOv2 isn't
re-evaluated.

Usage:
    python scripts/overlay_safety.py \\
        --episode 0 \\
        --classifier_ckpt runs/classifier_exp_merged/classifier_best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from utils.utils import init_logging

logger = logging.getLogger(__name__)


def load_failure_head(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[nn.Sequential, int, int]:
    """Reconstruct the failure_head MLP from a classifier checkpoint.

    Reads the head's input dim and hidden dim from the checkpoint shapes so we
    can adapt to checkpoints trained against either the full pooled z (no
    action-stripping) or the stripped visual+proprio z.

    Returns (head, input_dim, hidden_dim).
    """
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    fh_sd = {
        k.removeprefix("failure_head."): v
        for k, v in sd.items()
        if k.startswith("failure_head.")
    }
    if not fh_sd:
        raise KeyError(f"No 'failure_head.*' tensors found in {checkpoint_path}")

    input_dim = int(fh_sd["0.weight"].shape[0])      # LayerNorm(input_dim)
    hidden_dim = int(fh_sd["1.weight"].shape[0])     # Linear(input_dim → hidden_dim)

    head = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    head.load_state_dict(fh_sd)
    head.to(device).eval()
    return head, input_dim, hidden_dim


def score_episode(
    z_full: torch.Tensor,
    head: nn.Sequential,
    strip_action_dim: int,
    device: torch.device,
    cls_token_index: int | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Score every frame.

    CLS-enabled caches score the CLS+proprio state token. Older patch-only
    caches fall back to action-stripped mean pooling.
    """
    scores: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, z_full.shape[0], batch_size):
            chunk = z_full[i : i + batch_size].to(device).float()  # (B, P, D)
            z_state = chunk[..., :-strip_action_dim] if strip_action_dim > 0 else chunk
            if cls_token_index is None:
                features = z_state.mean(dim=1)                      # (B, D')
            else:
                features = z_state[:, cls_token_index, :]           # (B, D')
            s = head(features).squeeze(-1)                          # (B,)
            scores.append(s.cpu())
    return torch.cat(scores).numpy()


def overlay_video(
    src_path: Path,
    dst_path: Path,
    scores: np.ndarray,
    gt_labels: np.ndarray | None,
    threshold: float,
) -> None:
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {src_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_frames, len(scores))
    if n_frames != len(scores):
        logger.warning(
            f"video has {n_frames} frames but {len(scores)} scores — "
            f"writing the first {n}"
        )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (w, h))

    # Score bar uses the score range observed in this episode.
    s_min, s_max = float(np.min(scores)), float(np.max(scores))
    s_range = max(s_max - s_min, 1e-6)

    bar_h = max(36, h // 12)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7 * (h / 240)

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break

        s = float(scores[i])
        unsafe = s > threshold
        color = (40, 40, 220) if unsafe else (60, 180, 60)  # BGR
        label = "UNSAFE" if unsafe else "SAFE"

        cv2.rectangle(frame, (0, 0), (w, bar_h), color, thickness=-1)
        cv2.putText(
            frame,
            f"{label}  score={s:+.2f}",
            (10, int(bar_h * 0.7)),
            font, font_scale, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Bottom score bar normalized to this episode's [min, max].
        norm = (s - s_min) / s_range
        bar_w = int(w * norm)
        cv2.rectangle(frame, (0, h - 8), (bar_w, h), color, thickness=-1)

        if gt_labels is not None:
            gt = int(gt_labels[i])
            gt_color = (40, 40, 220) if gt == 1 else (60, 180, 60)
            gt_text = f"GT={'UNSAFE' if gt else 'SAFE'}"
            (tw, _), _ = cv2.getTextSize(gt_text, font, font_scale, 2)
            cv2.putText(
                frame,
                gt_text,
                (w - tw - 10, int(bar_h * 0.7)),
                font, font_scale, gt_color, 2, cv2.LINE_AA,
            )

        writer.write(frame)

    cap.release()
    writer.release()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root = Path(__file__).resolve().parents[1]
    p.add_argument("--episode", type=int, required=True)
    p.add_argument(
        "--classifier_ckpt",
        default=str(repo_root / "runs" / "classifier_exp_merged" / "classifier_best.pt"),
    )
    p.add_argument(
        "--latents_dir",
        default=str(repo_root / "runs" / "dino_wm_exp_merged" / "latents"),
    )
    p.add_argument(
        "--video_dir",
        default=str(repo_root / "data" / "exp_merged" / "videos"),
    )
    p.add_argument("--output", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--threshold", type=float, default=0.0,
        help="Score > threshold → UNSAFE. Classifier was trained with margin ±1.0 so 0.0 is the natural cut.",
    )
    p.add_argument(
        "--show_gt", action="store_true",
        help="Also overlay the cached failure_label as ground truth.",
    )
    p.add_argument(
        "--action_dim", type=int, default=10,
        help="Action features at the tail of z (action_emb_dim * num_action_repeat). "
             "Used only if the checkpoint's failure_head expects the action-stripped dim.",
    )
    args = p.parse_args()

    init_logging()

    latents_path = Path(args.latents_dir) / f"episode_{args.episode:03d}.pt"
    video_path = Path(args.video_dir) / f"episode_{args.episode:03d}.mp4"
    if not latents_path.exists():
        raise FileNotFoundError(f"Cached latents not found: {latents_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Source video not found: {video_path}")

    output_path = (
        Path(args.output)
        if args.output
        else repo_root / "runs" / "safety_overlay" / f"episode_{args.episode:03d}.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data = torch.load(latents_path, weights_only=True, map_location="cpu")
    manifest_path = Path(args.latents_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    z = data["z"]  # (T, P, D) fp16
    D = int(z.shape[-1])
    logger.info(f"Episode {args.episode}: T={z.shape[0]} P={z.shape[1]} D={D}")

    head, head_in, head_hidden = load_failure_head(args.classifier_ckpt, device=device)
    if head_in == D:
        strip = 0
    elif head_in == D - args.action_dim:
        strip = args.action_dim
    else:
        raise ValueError(
            f"Classifier head expects input dim {head_in} but cached z has dim {D} "
            f"(action_dim={args.action_dim}). Pass --action_dim to match the head."
        )
    logger.info(
        f"failure_head: input_dim={head_in} hidden_dim={head_hidden} "
        f"strip_action_dim={strip}"
    )

    cls_token_index = manifest.get("cls_token_index") if manifest.get("include_cls_token", False) else None
    scores = score_episode(z, head, strip, device, cls_token_index=cls_token_index)
    logger.info(
        f"Scores: min={scores.min():+.3f} max={scores.max():+.3f} mean={scores.mean():+.3f}  "
        f"frac>{args.threshold:g}={float((scores > args.threshold).mean()):.3f}"
    )

    gt_labels = data["failure_label"].numpy() if args.show_gt else None
    overlay_video(video_path, output_path, scores, gt_labels=gt_labels, threshold=args.threshold)
    logger.info(f"Wrote overlaid video to {output_path}")


if __name__ == "__main__":
    main()
