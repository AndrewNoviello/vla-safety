"""World model rollout with per-frame safety classifier annotations.

Rolls out the world model autoregressively from a local episode (using real
actions but never feeding back real future frames), and produces a side-by-side
MP4 where both the real frames and the WM-predicted frames are annotated with
the safety classifier score.

    left  = real episode frames  + safety annotation
    right = WM-decoded frames    + safety annotation

Usage:
    python -m latentsafe.play_wm_safety \
        --episode_dir /workspace/vla-safety/success-1 \
        --episode 0 \
        --classifier_checkpoint outputs/classifier_fast_v3/classifier_best.pt \
        --output out.mp4
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2 as TVT

from einops import rearrange

from dino_wm.config import DinoWMConfig
from dino_wm.decoder import Decoder
from dino_wm.encoder import DinoV2Encoder
from dino_wm.train import CFG, DINO_WM_NORM_MAP
from dino_wm.transition import TransitionModel
from dino_wm.visual_world_model import VWorldModel
from data.utils import POLICY_FEATURES, dataset_to_policy_features
from utils.processor_utils import normalize
from utils.utils import cast_stats_to_numpy, load_json


# ---------------------------------------------------------------------------
# Episode loading  (same as play_episode_classifier.py)
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
    images_raw = (batch.data / 255.0).float()

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
# Classifier model
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
# Classification helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def classify_real_frames(
    model: VWorldModel,
    images: torch.Tensor,    # (T, 3, H, W) float [0, 1]
    states: torch.Tensor,    # (T, 6)  normalized
    actions: torch.Tensor,   # (T, 6)  normalized
    num_hist: int,
    device: torch.device,
) -> list[float]:
    """Safety score per real frame via sliding window."""
    T = images.shape[0]
    scores: list[float] = []

    for t in range(T):
        idxs = [max(0, t - num_hist + 1 + i) for i in range(num_hist)]
        visual  = images[idxs].unsqueeze(0).to(device)
        proprio = states[idxs].unsqueeze(0).to(device)
        act     = actions[idxs].unsqueeze(0).to(device)

        obs = {"visual": visual, "proprio": proprio}
        z = model.encode(obs, act)
        z_pred = model.predict(z)
        score = model.predict_failure(z_pred)[:, -1, 0]
        scores.append(float(score.item()))

    return scores


@torch.no_grad()
def rollout_and_classify(
    model: VWorldModel,
    obs_0: dict,             # {"visual": (1,num_hist,3,H,W), "proprio": (1,num_hist,6)}
    all_actions: torch.Tensor,  # (1, T, 6) normalized
    num_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    """WM open-loop rollout using the classifier model.

    Returns:
        wm_frames: (num_frames, 3, H, W) decoded WM images
        wm_scores: safety score per frame, derived directly from WM latents
                   (no re-encoding of decoded frames — score is based purely on
                   the WM's internal predicted latents, making it independent of
                   the real observation stream)
    """
    obs_0 = {k: v.to(device) for k, v in obs_0.items()}
    all_actions = all_actions.to(device)

    # z_full: (1, T+1, num_patches, emb_dim) — seed=encoder out, rest=predictor out
    z_obses, z_full = model.rollout(obs_0, all_actions)

    # Decode WM visual frames
    vis = z_obses["visual"][:, :num_frames]   # (1, T, num_patches, emb_dim)
    b, t_len, np_, ed = vis.shape
    decoded = model.decoder(vis)              # (1*T, 3, H, W)
    wm_frames = rearrange(decoded, "(b t) c h w -> b t c h w", b=b, t=t_len)[0]
    # wm_frames: (T, 3, H, W)

    # Safety scores directly from WM latents — no re-encoding of decoded images
    z_for_safety = z_full[:, :num_frames]     # (1, T, num_patches, emb_dim)
    raw_scores = model.predict_failure(z_for_safety)  # (1, T, 1)
    wm_scores = raw_scores[0, :, 0].cpu().tolist()    # list of T floats

    return wm_frames, wm_scores


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _annotate_frame(
    frame_np: np.ndarray,
    score: float,
    frame_idx: int,
    label_prefix: str = "",
) -> np.ndarray:
    from PIL import Image, ImageDraw

    is_safe = score > 0.0
    label   = ("SAFE" if is_safe else "UNSAFE")
    accent  = (40, 180, 99) if is_safe else (220, 68, 55)

    lines = [f"{label_prefix}{label}", f"score: {score:+.3f}", f"frame: {frame_idx}"]

    image = Image.fromarray(frame_np)
    draw  = ImageDraw.Draw(image, "RGBA")
    line_h, pad = 18, 8
    box_h  = pad * 2 + line_h * len(lines)
    box_w  = 220
    draw.rounded_rectangle((8, 8, 8 + box_w, 8 + box_h), radius=10, fill=(0, 0, 0, 170))
    draw.rounded_rectangle((8, 8, 16, 8 + box_h), radius=4, fill=accent + (255,))
    y = 8 + pad
    for line in lines:
        draw.text((24, y), line, fill=(255, 255, 255, 255))
        y += line_h
    return np.asarray(image)


def _to_uint8_hwc(frame: torch.Tensor) -> np.ndarray:
    return (frame.detach().cpu().clamp(0, 1).permute(1, 2, 0) * 255).byte().numpy()


def save_side_by_side_video(
    real_frames: torch.Tensor,   # (T, 3, H, W)
    wm_frames: torch.Tensor,     # (T, 3, H, W)
    real_scores: list[float],
    wm_scores: list[float],
    output_path: Path,
    fps: float,
    num_hist: int,
) -> None:
    from torchvision.io import write_video

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = []
    for t, (rf, wf, rs, ws) in enumerate(
        zip(real_frames, wm_frames, real_scores, wm_scores)
    ):
        real_np = _to_uint8_hwc(rf)
        wm_np   = _to_uint8_hwc(wf)

        # Mark seed frames (reconstructions) vs predictions on WM side
        wm_prefix = "[seed] " if t < num_hist else "[pred] "
        real_ann = _annotate_frame(real_np, rs, t, label_prefix="real: ")
        wm_ann   = _annotate_frame(wm_np,   ws, t, label_prefix=wm_prefix)

        # Side by side: (H, 2*W, 3)
        combined.append(np.concatenate([real_ann, wm_ann], axis=1))

    write_video(
        str(output_path),
        torch.from_numpy(np.stack(combined)),
        fps=fps,
        video_codec="libx264",
        options={"crf": "18"},
    )
    print(f"Saved side-by-side video: {output_path} ({len(combined)} frames @ {fps:.1f} fps)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="WM rollout + safety classifier overlay, side-by-side."
    )
    parser.add_argument("--episode_dir", required=True,
                        help="Local episode folder (e.g. success-1)")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index (default: 0)")
    parser.add_argument("--classifier_checkpoint", required=True,
                        help="Path to classifier checkpoint .pt")
    parser.add_argument("--output", default=None,
                        help="Output MP4 path (default: wm_safety_ep<N>.mp4)")
    parser.add_argument("--device", default=None,
                        help="'cuda' or 'cpu' (default: auto)")
    parser.add_argument("--frameskip", type=int, default=CFG.frameskip)
    parser.add_argument("--img_size",  type=int, default=CFG.img_size)
    parser.add_argument("--num_hist",  type=int, default=CFG.num_hist)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    episode_dir = Path(args.episode_dir)
    output_path = Path(args.output or f"wm_safety_ep{args.episode:03d}.mp4")
    cfg = CFG

    # ---- Load episode ----
    print(f"Loading episode {args.episode} from {episode_dir} ...")
    images, states, actions = load_local_episode(
        episode_dir, args.episode, args.frameskip, args.img_size
    )
    num_frames = images.shape[0]
    print(f"  {num_frames} frames (at frameskip={args.frameskip})")

    if num_frames < cfg.num_hist + 1:
        raise RuntimeError(
            f"Episode too short ({num_frames} frames). Need at least {cfg.num_hist + 1}."
        )

    stats = load_local_stats(episode_dir)
    policy_features = dataset_to_policy_features(POLICY_FEATURES)
    if stats is not None:
        norm = normalize(
            {"observation.state": states, "action": actions},
            stats, policy_features, DINO_WM_NORM_MAP,
        )
        states_norm  = norm["observation.state"]
        actions_norm = norm["action"]
        print("  Normalized states and actions.")
    else:
        print("  Warning: no stats.json found, skipping normalization.")
        states_norm  = states
        actions_norm = actions

    # ---- Load classifier (contains the WM — used for both rollout and safety) ----
    print("\nLoading classifier (WM + failure head) ...")
    classifier = build_classifier(args.classifier_checkpoint, cfg).to(device)

    obs_0 = {
        "visual":  images[:cfg.num_hist].unsqueeze(0).to(device),
        "proprio": states_norm[:cfg.num_hist].unsqueeze(0).to(device),
    }
    all_actions = actions_norm.unsqueeze(0).to(device)

    # ---- WM rollout + safety from latents ----
    print("Running autoregressive WM rollout ...")
    wm_frames, wm_scores = rollout_and_classify(
        classifier, obs_0, all_actions, num_frames, device
    )

    # ---- Real-frame safety (sliding window over real observations) ----
    print("Classifying real frames ...")
    real_scores = classify_real_frames(
        classifier, images, states_norm, actions_norm, cfg.num_hist, device
    )

    # ---- Save video ----
    effective_fps = 30.0 / args.frameskip
    save_side_by_side_video(
        real_frames=images,
        wm_frames=wm_frames,
        real_scores=real_scores,
        wm_scores=wm_scores,
        output_path=output_path,
        fps=effective_fps,
        num_hist=cfg.num_hist,
    )

    def _summarize(scores: list[float], label: str) -> None:
        n_safe = sum(1 for s in scores if s > 0)
        print(f"  {label}: safe={n_safe}/{len(scores)}  "
              f"mean={sum(scores)/len(scores):.3f}")

    print("\nSummary:")
    _summarize(real_scores, "real")
    _summarize(wm_scores,   "WM  ")
    print(f"  Seed frames: {cfg.num_hist}  (WM right panel = reconstruction)")
    print(f"  Predicted:   {num_frames - cfg.num_hist}")
    print(f"  Output:      {output_path}")


if __name__ == "__main__":
    main()
