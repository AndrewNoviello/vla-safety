"""Live safety monitor — runs the failure_head over the current robot state.

Subscribes to so101real/joint_state, joint_command, camera/image (same as
recorder.py) and prints SAFE / UNSAFE at 20 Hz. Each tick encodes the current
observation through DINO-WM and feeds the encoded latent directly to
predict_failure — no rollout, no history buffer.

Decision rule (matches eval_classifier.py):
    score >  0  →  UNSAFE
    score <= 0  →  SAFE

Press 'q' to quit.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import termios
import threading
import tty
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2 as T

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

from data.utils import POLICY_FEATURES, dataset_to_policy_features, load_stats
from dino_wm.train import CFG, _build_model
from latentsafe.ddpg_safety import SafetyActor, SafetyCritic
from utils.processor_utils import normalize, to_device

ROBOT_NS = "so101real"
TICK_HZ = 20

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _detect_image_key(features: dict) -> str:
    preferred = ("image0", "observation.image", "observation.images.front", "image")
    for k in preferred:
        if k in features and features[k]["dtype"] == "image":
            return k
    for k, v in features.items():
        if v["dtype"] == "image":
            return k
    raise ValueError("No image observation key found in POLICY_FEATURES.")


def _get_keypress() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class SafetyMonitor(Node):
    def __init__(self, checkpoint: str, dataset_root: str, device: str, policy_dir: str | None = None):
        super().__init__("safety_monitor")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        action_dim  = POLICY_FEATURES["action"]["shape"][-1]
        proprio_dim = POLICY_FEATURES["observation.state"]["shape"][-1]
        cfg = dataclasses.replace(CFG, use_failure_head=True)
        model, _ = _build_model(cfg, action_dim, proprio_dim)
        sd = torch.load(checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if any("failure_head" in k for k in missing):
            raise RuntimeError(
                f"Checkpoint {checkpoint} is missing failure_head weights — "
                f"was the classifier trained with use_failure_head=True?"
            )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model.to(self.device)

        self.actor = self.critic = None
        if policy_dir is not None:
            obs_dim    = self.model.emb_dim
            action_dim = POLICY_FEATURES["action"]["shape"][-1]
            self.actor  = SafetyActor(obs_dim, action_dim).to(self.device).eval()
            self.critic = SafetyCritic(obs_dim, action_dim).to(self.device).eval()
            self.actor.load_state_dict(torch.load(Path(policy_dir) / "actor.pt",  map_location=self.device))
            self.critic.load_state_dict(torch.load(Path(policy_dir) / "critic.pt", map_location=self.device))
            for p in (*self.actor.parameters(), *self.critic.parameters()):
                p.requires_grad = False

        self.stats = load_stats(Path(dataset_root))
        if self.stats is None:
            raise RuntimeError(f"No stats at {dataset_root}/meta/stats.json")
        self.policy_features = dataset_to_policy_features(POLICY_FEATURES)
        self.image_key = _detect_image_key(POLICY_FEATURES)
        self.image_resize = T.Resize((CFG.img_size, CFG.img_size), antialias=True)

        self.latest_positions: np.ndarray | None = None
        self.latest_commands:  np.ndarray | None = None
        self.latest_frame:     np.ndarray | None = None

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(JointState,        f"{ROBOT_NS}/joint_state",   self._cb_joint_state,   qos)
        self.create_subscription(Float64MultiArray, f"{ROBOT_NS}/joint_command", self._cb_joint_command, qos)
        self.create_subscription(Image,             f"{ROBOT_NS}/camera/image",  self._cb_image,         qos)

        self.create_timer(1.0 / TICK_HZ, self._tick)
        print(
            f"{BOLD}Safety monitor{RESET} on {self.device}  "
            f"{DIM}checkpoint: {checkpoint}{RESET}\n"
            f"{DIM}Waiting for sensor data… press 'q' to quit.{RESET}",
            flush=True,
        )

    def _cb_joint_state(self, msg: JointState):
        self.latest_positions = np.array(msg.position[:6], dtype=np.float32)

    def _cb_joint_command(self, msg: Float64MultiArray):
        self.latest_commands = np.array(msg.data[:6], dtype=np.float32)

    def _cb_image(self, msg: Image):
        # Camera publishes bgr8. Training mp4s decode to RGB (cv2 BGR→YUV at
        # write, torchcodec YUV→RGB at read), so flip channels here to match.
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.latest_frame = frame[..., ::-1].copy()

    @torch.no_grad()
    def _tick(self):
        if self.latest_positions is None or self.latest_frame is None:
            return

        img = torch.from_numpy(self.latest_frame).permute(2, 0, 1).float() / 255.0
        img = self.image_resize(img).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 224, 224)

        proprio = torch.from_numpy(self.latest_positions).unsqueeze(0).unsqueeze(0)  # (1, 1, 6)
        cmd = self.latest_commands if self.latest_commands is not None else np.zeros(6, dtype=np.float32)
        action = torch.from_numpy(cmd).unsqueeze(0).unsqueeze(0)  # (1, 1, 6)

        batch = {self.image_key: img, "observation.state": proprio, "action": action}
        batch = normalize(batch, self.stats, self.policy_features)
        batch = to_device(batch, self.device)

        obs = {
            "visual":  batch[self.image_key].float(),
            "proprio": batch["observation.state"].float(),
        }
        z = self.model.encode(obs, batch["action"].float())   # (1, 1, P, D)
        score = self.model.predict_failure(z)[0, -1, 0].item()

        value = None
        if self.actor is not None:
            pooled = z[:, -1].mean(dim=1)                # (1, predictor_dim)
            action_t = self.actor(pooled)                # (1, action_dim)
            value = self.critic(pooled, action_t).item()

        if score > 0:
            tag = f"{RED}{BOLD}UNSAFE{RESET}"
        else:
            tag = f"{GREEN}{BOLD}SAFE  {RESET}"
        # |score| < 1 = inside training margin; classifier is uncertain.
        margin = "" if abs(score) >= 1.0 else f" {YELLOW}(margin){RESET}"
        line = f"\r{tag}  {DIM}score={score:+.2f}{RESET}{margin}"
        if value is not None:
            v_color = GREEN if value > 0 else RED
            line += f"   {v_color}{BOLD}V={value:+.2f}{RESET}"
        print(line + "    ", end="", flush=True)


def _keyboard_thread():
    while rclpy.ok():
        try:
            if _get_keypress() == "q":
                break
        except (OSError, ValueError):
            break
    rclpy.shutdown()


def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(repo_root / "runs" / "classifier_exp_merged" / "classifier_best.pt"),
    )
    parser.add_argument(
        "--dataset_root",
        default=str(repo_root / "data" / "exp_merged"),
        help="Used for normalization stats only (meta/stats.json).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--policy_dir",
        default=None,
        help="Directory containing actor.pt and critic.pt from train_safety_ddpg. "
             "If set, also display V(s) = critic(obs, actor(obs)) per tick.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = SafetyMonitor(args.checkpoint, args.dataset_root, args.device, args.policy_dir)
    kb = threading.Thread(target=_keyboard_thread, daemon=True)
    kb.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        print()


if __name__ == "__main__":
    main()
