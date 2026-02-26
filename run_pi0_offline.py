#!/usr/bin/env python
"""
Run π₀ (PI0) base policy offline: give it images, a text prompt, and proprio state,
and get predicted actions without a robot.

This script loads the base policy from the Hub, builds an observation from your
inputs, runs one forward pass, and prints (or optionally saves) the predicted
action chunk.

Usage:
    # With image paths (one image replicated to all 3 cameras, or 3 paths in order)
    python run_pi0_offline.py --prompt "Pick up the red block" --images /path/to/image.jpg

    # Or 3 images for base + left_wrist + right_wrist
    python run_pi0_offline.py --prompt "Pick up the red block" --images img1.jpg img2.jpg img3.jpg

    # Save action to .npy
    python run_pi0_offline.py --output action.npy

Requirements:
    - lerobot and deps installed (or run from workspace with lerobot in ./lerobot)
    - HuggingFace token if needed for lerobot/pi0_base
    - Images will be resized to 224x224 (PI0 default)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from workspace root; lerobot package is at ./lerobot/
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch

from lerobot.types import FeatureType
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0 import PI0Policy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import OBS_STATE


def load_image_as_numpy(path: str | Path, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load an image from path and return (H, W, 3) uint8, resized to size."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        from PIL import Image
    except ImportError:
        try:
            import cv2
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Could not read image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except ImportError:
            raise ImportError("Install Pillow (pip install Pillow) or opencv-python to load images.")
    else:
        img = np.array(Image.open(path).convert("RGB"))

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape {img.shape}")

    if (img.shape[0], img.shape[1]) != size:
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img)
            pil_img = pil_img.resize((size[1], size[0]), PILImage.BILINEAR)
            img = np.array(pil_img)
        except ImportError:
            import cv2
            img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)

    return img.astype(np.uint8)


def build_observation(
    policy,
    *,
    image_paths: list[str] | None = None,
    proprio: np.ndarray | list[float] | None = None,
    task: str = "",
    image_size: tuple[int, int] = (224, 224),
) -> dict[str, np.ndarray]:
    """
    Build an observation dict with keys matching policy.input_features.
    - image_paths: one path (replicated to all camera keys) or N paths in same order as
      policy input image keys.
    - proprio: 1D array of state; will be padded/trimmed to policy's state dim.
    - task: text instruction for the policy.
    """
    observation: dict[str, np.ndarray] = {}
    input_features = policy.input_features

    state_key = OBS_STATE
    if state_key in input_features:
        state_dim = input_features[state_key].shape[0]
        if proprio is not None:
            arr = np.asarray(proprio, dtype=np.float32).flatten()
            if len(arr) >= state_dim:
                observation[state_key] = arr[:state_dim].astype(np.float32)
            else:
                padded = np.zeros(state_dim, dtype=np.float32)
                padded[: len(arr)] = arr
                observation[state_key] = padded
        else:
            observation[state_key] = np.zeros(state_dim, dtype=np.float32)

    image_keys = [k for k, v in input_features.items() if v.type == FeatureType.VISUAL]
    if not image_keys:
        raise ValueError("PI0 base policy expects at least one image input.")

    if not image_paths:
        h, w = image_size[0], image_size[1]
        dummy = np.full((h, w, 3), 128, dtype=np.uint8)
        for k in image_keys:
            observation[k] = dummy.copy()
    else:
        if len(image_paths) == 1:
            single = load_image_as_numpy(image_paths[0], image_size)
            for k in image_keys:
                observation[k] = single.copy()
        else:
            if len(image_paths) != len(image_keys):
                raise ValueError(
                    f"Got {len(image_paths)} images but policy has {len(image_keys)} "
                    f"camera keys: {image_keys}. Pass 1 image (replicated) or {len(image_keys)} paths."
                )
            for path, k in zip(image_paths, image_keys):
                observation[k] = load_image_as_numpy(path, image_size)

    return observation


def run_inference(
    policy,
    preprocessor,
    postprocessor,
    observation: dict[str, np.ndarray],
    task: str,
    device: torch.device,
    use_amp: bool = False,
):
    """Run one inference step: observation (numpy) -> preprocess -> policy -> postprocess -> action."""
    observation = dict(observation)
    print('observation: ' + str(observation))
    print('input features: ' + str(policy.input_features))
    observation = prepare_observation_for_inference(
        observation,
        device=device,
        task=task,
        robot_type="",
    )
    observation = preprocessor(observation)
    ctx = (
        torch.autocast(device_type="cuda")
        if device.type == "cuda" and use_amp
        else torch.amp.autocast("cpu", enabled=False)
    )
    with torch.inference_mode(), ctx:
        action = policy.select_action(observation)
    action = postprocessor(action)
    return action


def main():
    parser = argparse.ArgumentParser(
        description="Run PI0 base policy offline with images, text prompt, and proprio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lerobot/pi0_base",
        help="Pretrained model id or path (default: lerobot/pi0_base)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Pick up the red block.",
        help="Text instruction for the policy",
    )
    parser.add_argument(
        "--proprio",
        type=float,
        nargs="+",
        default=None,
        help="Proprio/state vector (floats). If omitted, zeros are used.",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        default=None,
        help="Image path(s): 1 path (replicated to all cameras) or 3 paths for base, left_wrist, right_wrist",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, cpu, or mps (default: auto)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save action tensor as .npy",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision on CUDA",
    )
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Loading policy: {args.model}")
    policy = PI0Policy.from_pretrained(args.model)
    policy.to(device)
    policy.eval()
    policy.config.device = str(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_type="pi0", policy_cfg=policy.config, dataset_stats=None
    )

    observation = build_observation(
        policy,
        image_paths=args.images,
        proprio=args.proprio,
        task=args.prompt,
        image_size=tuple(policy.image_resolution),
    )
    print(f"Observation keys: {list(observation.keys())}")

    use_amp = not args.no_amp and device.type == "cuda"
    action = run_inference(
        policy,
        preprocessor,
        postprocessor,
        observation,
        task=args.prompt,
        device=device,
        use_amp=use_amp,
    )

    action_np = action.numpy() if isinstance(action, torch.Tensor) else action
    print(f"Action shape: {action_np.shape}")
    if action_np.ndim == 3:
        print(f"Action (first step): {action_np[0, 0, :]}")
    else:
        print(f"Action (first step): {action_np[0, :]}")

    if args.output:
        np.save(args.output, action_np)
        print(f"Saved action to {args.output}")

    return action_np


if __name__ == "__main__":
    main()
