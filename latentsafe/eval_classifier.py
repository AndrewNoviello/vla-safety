"""Evaluate the failure classifier (failure_head of VWorldModel).

Runs the trained model on held-out dataset samples and reports:
- Confusion matrix
- Per-class precision, recall, F1
- AUC-ROC (safe vs. unsafe, treating weakly-unsafe as unsafe)

Usage
-----
    python -m latentsafe.eval_classifier \
        --checkpoint outputs/classifier/classifier_best.pt \
        --dataset_repo_id AndrewNoviello/domino-world-v2 \
        --num_samples 2000
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from dino_wm.config import DinoWMConfig
from dino_wm.decoder import Decoder
from dino_wm.encoder import DinoV2Encoder
from dino_wm.transition import TransitionModel
from dino_wm.visual_world_model import VWorldModel
from data.lerobot_dataset import LeRobotDataset
from data.utils import POLICY_FEATURES, dataset_to_policy_features
from utils.types import FeatureType, NormalizationMode
from utils.processor_utils import normalize, to_device
from utils.utils import init_logging

logging.basicConfig(level=logging.INFO)

_NORM_MAP = {
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
    raise ValueError("No image observation key found.")


def _to_uint8_hwc(frame: torch.Tensor) -> np.ndarray:
    """Convert (3, H, W) float tensor in [0, 1] to uint8 HWC."""
    frame = frame.detach().cpu().clamp(0.0, 1.0)
    return (frame.permute(1, 2, 0) * 255).byte().numpy()


def _label_text(label: int) -> str:
    if label == 0:
        return "safe"
    if label == 1:
        return "unsafe"
    if label == 2:
        return "weak"
    return f"label={label}"


def _annotate_frame(
    frame: np.ndarray,
    pred_label: int,
    score: float,
    true_label: int | None,
) -> np.ndarray:
    """Overlay predicted / true safety labels onto an RGB frame."""
    from PIL import Image, ImageDraw

    pred_text = f"pred: {'unsafe' if pred_label > 0 else 'safe'}"
    score_text = f"score: {score:+.3f}"
    lines = [pred_text, score_text]
    if true_label is not None:
        lines.append(f"true: {_label_text(true_label)}")

    # Green for safe prediction, red for unsafe prediction.
    accent = (40, 180, 99) if pred_label == 0 else (220, 68, 55)

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    line_height = 18
    pad = 8
    box_h = pad * 2 + line_height * len(lines)
    box_w = 220
    draw.rounded_rectangle((8, 8, 8 + box_w, 8 + box_h), radius=10, fill=(0, 0, 0, 170))
    draw.rounded_rectangle((8, 8, 16, 8 + box_h), radius=4, fill=accent + (255,))
    y = 8 + pad
    for line in lines:
        draw.text((24, y), line, fill=(255, 255, 255, 255))
        y += line_height
    return np.asarray(image)


def _save_eval_video(
    frames: list[torch.Tensor],
    pred_labels: list[int],
    scores: list[float],
    true_labels: list[int | None],
    output_path: str | Path,
    fps: float,
) -> None:
    """Write an annotated MP4 for a subset of evaluated samples."""
    from torchvision.io import write_video

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotated = []
    for frame, pred_label, score, true_label in zip(frames, pred_labels, scores, true_labels):
        frame_np = _to_uint8_hwc(frame)
        annotated.append(_annotate_frame(frame_np, pred_label, score, true_label))

    write_video(
        str(output_path),
        torch.from_numpy(np.stack(annotated, axis=0)),
        fps=fps,
        video_codec="libx264",
        options={"crf": "18"},
    )
    logging.info(f"Saved annotated classifier video: {output_path} ({len(annotated)} frames @ {fps:.1f} fps)")


def _build_and_load(
    checkpoint: str,
    action_dim: int,
    proprio_dim: int,
    cfg: DinoWMConfig,
) -> VWorldModel:
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


def evaluate(
    checkpoint: str,
    dataset_repo_id: str,
    num_samples: int = 2000,
    batch_size: int = 64,
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
    failure_head_hidden_dim: int = 256,
    video_output: str | None = None,
    video_fps: float = 4.0,
    video_max_frames: int = 200,
) -> dict:
    init_logging()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    cfg = DinoWMConfig(
        dataset_repo_id=dataset_repo_id,
        num_hist=num_hist, num_pred=num_pred, frameskip=frameskip,
        img_size=img_size, encoder_name=encoder_name,
        action_emb_dim=action_emb_dim, proprio_emb_dim=proprio_emb_dim,
        concat_dim=concat_dim,
        predictor_depth=predictor_depth, predictor_heads=predictor_heads,
        predictor_mlp_dim=predictor_mlp_dim,
        use_failure_head=True, failure_head_hidden_dim=failure_head_hidden_dim,
    )

    image_key   = _detect_image_key(POLICY_FEATURES)
    action_dim  = POLICY_FEATURES["action"]["shape"][-1]
    proprio_dim = POLICY_FEATURES["observation.state"]["shape"][-1]

    window  = num_hist + num_pred
    indices = [i * frameskip for i in range(window)]
    delta_indices = {}
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
        image_transforms=T.Resize((img_size, img_size), antialias=True),
    )
    policy_features = dataset_to_policy_features(POLICY_FEATURES)

    # Cap to num_samples
    indices_to_eval = list(range(min(num_samples, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices_to_eval)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = _build_and_load(checkpoint, action_dim, proprio_dim, cfg).to(device)

    all_scores = []
    all_labels = []
    has_labels = False
    video_frames: list[torch.Tensor] = []
    video_preds: list[int] = []
    video_scores: list[float] = []
    video_labels: list[int | None] = []

    with torch.no_grad():
        for batch in loader:
            batch = normalize(batch, dataset.stats, policy_features, _NORM_MAP)
            batch = to_device(batch, device)

            visual = batch[image_key].float()
            obs    = {"visual": visual, "proprio": batch["observation.state"].float()}
            act    = batch["action"].float()

            z      = model.encode(obs, act)
            z_src  = z[:, :num_hist]
            z_pred = model.predict(z_src)

            scores = model.predict_failure(z_pred)[:, -1, 0]   # (B,) last timestep
            all_scores.append(scores.cpu())

            if video_output is not None and len(video_frames) < video_max_frames:
                frames_to_take = min(video_max_frames - len(video_frames), visual.shape[0])
                pred_batch = (scores > 0.0).long().cpu()
                label_batch = None
                if "failure_label" in batch:
                    label_batch = batch["failure_label"][:, -1].long().cpu()

                for idx in range(frames_to_take):
                    video_frames.append(visual[idx, -1].cpu())
                    video_preds.append(int(pred_batch[idx].item()))
                    video_scores.append(float(scores[idx].item()))
                    video_labels.append(int(label_batch[idx].item()) if label_batch is not None else None)

            if "failure_label" in batch:
                all_labels.append(batch["failure_label"][:, -1].long().cpu())
                has_labels = True

    all_scores = torch.cat(all_scores).numpy()

    if video_output is not None and video_frames:
        _save_eval_video(
            frames=video_frames,
            pred_labels=video_preds,
            scores=video_scores,
            true_labels=video_labels,
            output_path=video_output,
            fps=video_fps,
        )

    if not has_labels:
        logging.warning("No 'failure_label' found in dataset. Cannot compute metrics.")
        logging.info(f"Score stats: mean={all_scores.mean():.3f} std={all_scores.std():.3f}")
        return {"scores_mean": float(all_scores.mean()), "scores_std": float(all_scores.std())}

    all_labels = torch.cat(all_labels).numpy()

    # Predicted class: score > 0 → unsafe (1), score <= 0 → safe (0)
    preds = (all_scores > 0.0).astype(int)
    # Treat weakly-unsafe (2) as unsafe (1) for binary metrics
    binary_labels = (all_labels > 0).astype(int)

    # Confusion matrix
    tp = int(((preds == 1) & (binary_labels == 1)).sum())
    tn = int(((preds == 0) & (binary_labels == 0)).sum())
    fp = int(((preds == 1) & (binary_labels == 0)).sum())
    fn = int(((preds == 0) & (binary_labels == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy  = (tp + tn) / max(len(preds), 1)

    results = {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_safe":    int((all_labels == 0).sum()),
        "n_unsafe":  int((all_labels == 1).sum()),
        "n_weak":    int((all_labels == 2).sum()),
    }

    logging.info("=== Failure Classifier Evaluation ===")
    logging.info(f"  Samples:   {len(all_labels)}")
    logging.info(f"  Labels:    safe={results['n_safe']}  unsafe={results['n_unsafe']}  weak={results['n_weak']}")
    logging.info(f"  Accuracy:  {accuracy:.3f}")
    logging.info(f"  Precision: {precision:.3f}")
    logging.info(f"  Recall:    {recall:.3f}")
    logging.info(f"  F1:        {f1:.3f}")
    logging.info(f"  Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")

    return results


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--dataset_repo_id",  default="various-and-sundry/domino-world-v3-labeled")
    p.add_argument("--num_samples",      type=int, default=2000)
    p.add_argument("--batch_size",       type=int, default=64)
    p.add_argument("--device",           default="cuda")
    p.add_argument("--video_output",     default=None,
                   help="Optional MP4 path for annotated sample video")
    p.add_argument("--video_fps",        type=float, default=4.0)
    p.add_argument("--video_max_frames", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        checkpoint=args.checkpoint,
        dataset_repo_id=args.dataset_repo_id,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device_str=args.device,
        video_output=args.video_output,
        video_fps=args.video_fps,
        video_max_frames=args.video_max_frames,
    )
