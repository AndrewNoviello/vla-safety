"""Play a local episode as video with per-frame safety classifier annotations.

Loads every frame from a single episode, slides a num_hist-wide window over
them, runs the safety classifier at each step, and writes an annotated MP4.

Usage:
    python -m latentsafe.play_episode_classifier \
        --episode_dir /workspace/vla-safety/success-1 \
        --episode 0 \
        --checkpoint outputs/classifier/classifier_best.pt \
        --output out.mp4

The score shown on each frame is the classifier's prediction of whether the
NEXT step is safe (positive = safe, negative = unsafe).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2 as TVT

from dino_wm.config import DinoWMConfig
from dino_wm.decoder import Decoder
from dino_wm.encoder import DinoV2Encoder
from dino_wm.transition import TransitionModel
from dino_wm.visual_world_model import VWorldModel
from dino_wm.train import CFG, DINO_WM_NORM_MAP
from data.utils import POLICY_FEATURES, dataset_to_policy_features, load_stats
from utils.processor_utils import normalize
from utils.utils import cast_stats_to_numpy, load_json


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------

def load_local_episode(
    episode_dir: Path,
    episode_idx: int,
    frameskip: int,
    img_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load frames, states, actions from a local LeRobot-style episode folder.

    Returns:
        images:  (T, 3, img_size, img_size)  float32 [0, 1]
        states:  (T, 6)                       float32 (unnormalized)
        actions: (T, 6)                       float32 (unnormalized)
    """
    from torchcodec.decoders import VideoDecoder

    data_dir = episode_dir / "data"
    parquet_paths = sorted(data_dir.glob("episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No episode_*.parquet files in {data_dir}")

    df = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)
    ep_df = df[df["episode_index"] == episode_idx].reset_index(drop=True)
    if len(ep_df) == 0:
        available = sorted(df["episode_index"].unique().tolist())
        raise ValueError(f"Episode {episode_idx} not found. Available: {available}")

    idx_in_ep = list(range(0, len(ep_df), frameskip))
    ep_df = ep_df.iloc[idx_in_ep].reset_index(drop=True)

    video_path = episode_dir / "videos" / f"episode_{episode_idx:03d}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frame_indices = ep_df["frame_index"].tolist()
    decoder = VideoDecoder(str(video_path))
    batch = decoder.get_frames_at(indices=frame_indices)
    images_raw = (batch.data / 255.0).float()   # (T, 3, H, W)

    resize = TVT.Resize((img_size, img_size), antialias=True)
    images = resize(images_raw)

    states  = torch.tensor(ep_df["observation.state"].tolist(), dtype=torch.float32)
    actions = torch.tensor(ep_df["action"].tolist(), dtype=torch.float32)

    return images, states, actions


def load_local_stats(episode_dir: Path) -> dict | None:
    stats_path = episode_dir / "meta" / "stats.json"
    if not stats_path.exists():
        return None
    return cast_stats_to_numpy(load_json(stats_path))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_classifier(checkpoint: str, cfg: DinoWMConfig) -> VWorldModel:
    action_dim  = POLICY_FEATURES["action"]["shape"][-1]
    proprio_dim = POLICY_FEATURES["observation.state"]["shape"][-1]

    encoder = DinoV2Encoder(name=cfg.encoder_name)
    emb_dim = encoder.emb_dim
    num_vis_patches = (cfg.img_size // 16) ** 2

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
        dropout=0.0,
        emb_dropout=0.0,
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
    sd = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-frame classification
# ---------------------------------------------------------------------------

@torch.no_grad()
def classify_episode(
    model: VWorldModel,
    images: torch.Tensor,    # (T, 3, H, W) float [0, 1]
    states: torch.Tensor,    # (T, 6)  normalized
    actions: torch.Tensor,   # (T, 6)  normalized
    num_hist: int,
    device: torch.device,
) -> list[float]:
    """Return one safety score per frame.

    The score at frame t is the classifier's safety prediction given the
    history ending at frame t (predicts frame t+1).
    """
    T = images.shape[0]
    scores: list[float] = []
    model = model.to(device)

    for t in range(T):
        # Build a window of num_hist frames ending at t (pad with first frame)
        idxs = [max(0, t - num_hist + 1 + i) for i in range(num_hist)]
        visual  = images[idxs].unsqueeze(0).to(device)   # (1, num_hist, 3, H, W)
        proprio = states[idxs].unsqueeze(0).to(device)   # (1, num_hist, 6)
        act     = actions[idxs].unsqueeze(0).to(device)  # (1, num_hist, 6)

        obs = {"visual": visual, "proprio": proprio}
        z = model.encode(obs, act)
        z_pred = model.predict(z)
        score = model.predict_failure(z_pred)[:, -1, 0]  # (1,)
        scores.append(float(score.item()))

    return scores


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _annotate_frame(frame_np: np.ndarray, score: float, frame_idx: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    is_safe = score > 0.0
    label   = "SAFE" if is_safe else "UNSAFE"
    accent  = (40, 180, 99) if is_safe else (220, 68, 55)

    lines = [label, f"score: {score:+.3f}", f"frame: {frame_idx}"]

    image = Image.fromarray(frame_np)
    draw  = ImageDraw.Draw(image, "RGBA")
    line_h, pad = 18, 8
    box_h  = pad * 2 + line_h * len(lines)
    box_w  = 200
    draw.rounded_rectangle((8, 8, 8 + box_w, 8 + box_h), radius=10, fill=(0, 0, 0, 170))
    draw.rounded_rectangle((8, 8, 16, 8 + box_h), radius=4, fill=accent + (255,))
    y = 8 + pad
    for line in lines:
        draw.text((24, y), line, fill=(255, 255, 255, 255))
        y += line_h
    return np.asarray(image)


def save_annotated_video(
    images: torch.Tensor,   # (T, 3, H, W) float [0, 1]
    scores: list[float],
    output_path: Path,
    fps: float,
) -> None:
    from torchvision.io import write_video

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = []
    for t, (frame, score) in enumerate(zip(images, scores)):
        frame_np = (frame.detach().cpu().clamp(0, 1).permute(1, 2, 0) * 255).byte().numpy()
        annotated.append(_annotate_frame(frame_np, score, t))

    write_video(
        str(output_path),
        torch.from_numpy(np.stack(annotated)),
        fps=fps,
        video_codec="libx264",
        options={"crf": "18"},
    )
    print(f"Saved annotated video: {output_path} ({len(annotated)} frames @ {fps:.1f} fps)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Play a local episode with per-frame safety classifier overlay."
    )
    parser.add_argument("--episode_dir", required=True,
                        help="Path to local episode folder (e.g. success-1)")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index within the folder (default: 0)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to classifier checkpoint .pt")
    parser.add_argument("--output", default=None,
                        help="Output MP4 path (default: classifier_ep<N>.mp4)")
    parser.add_argument("--device", default=None,
                        help="'cuda' or 'cpu' (default: auto)")
    parser.add_argument("--frameskip", type=int, default=CFG.frameskip)
    parser.add_argument("--img_size",  type=int, default=CFG.img_size)
    parser.add_argument("--num_hist",  type=int, default=CFG.num_hist)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    episode_dir = Path(args.episode_dir)
    output_path = Path(args.output or f"classifier_ep{args.episode:03d}.mp4")

    cfg = CFG

    print(f"Loading episode {args.episode} from {episode_dir} ...")
    images, states, actions = load_local_episode(
        episode_dir, args.episode, args.frameskip, args.img_size
    )
    print(f"  {images.shape[0]} frames (at frameskip={args.frameskip})")

    stats = load_local_stats(episode_dir)
    if stats is not None:
        policy_features = dataset_to_policy_features(POLICY_FEATURES)
        norm = normalize(
            {"observation.state": states, "action": actions},
            stats, policy_features, DINO_WM_NORM_MAP,
        )
        states  = norm["observation.state"]
        actions = norm["action"]
        print("  Normalized states and actions.")
    else:
        print("  Warning: no stats.json found, skipping normalization.")

    print(f"Loading classifier from {args.checkpoint} ...")
    model = build_classifier(args.checkpoint, cfg)

    print("Classifying frames ...")
    scores = classify_episode(model, images, states, actions, args.num_hist, device)
    n_safe   = sum(1 for s in scores if s > 0)
    n_unsafe = len(scores) - n_safe
    print(f"  safe={n_safe}  unsafe={n_unsafe}  "
          f"mean_score={sum(scores)/len(scores):.3f}")

    effective_fps = 30.0 / args.frameskip
    save_annotated_video(images, scores, output_path, fps=effective_fps)


if __name__ == "__main__":
    main()
