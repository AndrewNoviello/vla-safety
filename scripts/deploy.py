"""Safe Deployment — PI0 + Latent Safety Filter on a real SO101 arm.

A single ROS2 node that connects to two hardware-bridge nodes already running:

  hardware (this repo, already running)
  ├── robot_node_launcher  → publishes so101real/joint_state
  │                        → subscribes to so101real/joint_command
  └── webcam.py            → publishes so101real/camera/image

  scripts/deploy_ros.py (this script)
  └── SafeDeploymentNode   → subscribes to so101real/joint_state     ← from hardware
                           → subscribes to so101real/camera/image    ← from hardware
                           → publishes  so101real/joint_command      → to hardware

Data flow:

  hardware/webcam ──→ so101real/camera/image ──┐
                                               ├──→ PI0 (base policy) → proposed action ──┐
  hardware/robot  ──→ so101real/joint_state  ──┘                                          │
                                                                                          ▼
                                                                          ┌──────────────────────────┐
                                                                          │ LatentSafetyFilter       │
                                                                          │ V(ẑ_{t+1}) > ε ?        │
                                                                          │ YES → pass through       │
                                                                          │ NO  → override with      │
                                                                          │       safety policy      │
                                                                          └──────────────────────────┘
                                                                                          │
                                                                                          ▼
  hardware/robot  ←── so101real/joint_command ←──────────────────────────────────────────┘

Prerequisites (run from repo root first):
    python -m hardware.ros.scripts.robot_node_launcher robot=so101real &
    python hardware/ros/examples/webcam.py &

Then run this script:
    python scripts/deploy_ros.py \
        --wm-checkpoint outputs/dino_wm_v2/checkpoints/latest/model.pt \
        --actor-checkpoint outputs/safety_ddpg/checkpoints/epoch_0015/actor.pt \
        --critic-checkpoint outputs/safety_ddpg/checkpoints/epoch_0015/critic.pt \
        --prompt "pick up the domino"

    # Without safety filter (for comparison / debugging):
    python scripts/deploy_ros.py --disable-safety
"""

import argparse
import logging
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

from utils.paths import ASSETS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Defaults (override via CLI)
# ═══════════════════════════════════════════════════════════════════════════

ROBOT_NS = "so101real"
CONTROL_HZ = 15.0
SAFETY_EPSILON = 0.3
IMG_SIZE = 224
ACTION_DIM = 6
PROPRIO_DIM = 6

PI0_MODEL_ID = "AndrewNoviello/vla-safety-task-4"
DEFAULT_PROMPT = (
    "pick up the middle domino from the three domino row and place it flat "
    "on top of the other two dominos to form an arch"
)


# ═══════════════════════════════════════════════════════════════════════════
#  PI0 wrapper
# ═══════════════════════════════════════════════════════════════════════════

class PI0Wrapper:
    """Wraps PI0Policy for action-chunk inference."""

    def __init__(self, model_id: str, prompt: str, stats_path: str, device: str):
        from transformers import AutoTokenizer
        from pi0.config import PI0Config
        from pi0.policy import PI0Policy
        from pi0.processor import preprocess_pi0, postprocess_pi0
        from utils.utils import cast_stats_to_numpy, load_json
        from utils.processor_utils import prepare_observation_for_inference, prepare_stats
        from utils.constants import OBS_STATE

        self.device = torch.device(device)
        self.prompt = prompt

        logger.info(f"Loading PI0 from {model_id} ...")
        config = PI0Config.from_pretrained(model_id)
        config.compile_model = False
        self.policy = PI0Policy.from_pretrained(model_id, config=config)
        self.policy.to(self.device)
        self.policy.eval()
        self.policy.config.device = str(self.device)

        dataset_stats = None
        if Path(stats_path).exists():
            dataset_stats = cast_stats_to_numpy(load_json(stats_path))
        self._stats = prepare_stats(dataset_stats)
        self._all_features = {
            **self.policy.config.input_features,
            **self.policy.config.output_features,
        }
        self._output_features = dict(self.policy.config.output_features)
        self._tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        self._image_resolution = tuple(self.policy.image_resolution)
        self._preprocess_pi0 = preprocess_pi0
        self._postprocess_pi0 = postprocess_pi0
        self._prepare_obs = prepare_observation_for_inference
        self._OBS_STATE = OBS_STATE
        logger.info("PI0 loaded and ready.")

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def predict_action_chunk(self, image_rgb: np.ndarray, proprio_raw: np.ndarray) -> np.ndarray:
        """(H,W,3) uint8 RGB + (6,) float → (chunk_size, action_dim) float array."""
        from utils.types import FeatureType

        observation = {}
        input_features = self.policy.config.input_features

        state_key = self._OBS_STATE
        if state_key in input_features:
            state_dim = input_features[state_key].shape[0]
            arr = np.asarray(proprio_raw, dtype=np.float32).flatten()
            if len(arr) >= state_dim:
                observation[state_key] = arr[:state_dim]
            else:
                padded = np.zeros(state_dim, dtype=np.float32)
                padded[: len(arr)] = arr
                observation[state_key] = padded

        image_keys = [
            k for k, v in input_features.items() if v.type == FeatureType.VISUAL
        ]
        for k in image_keys:
            observation[k] = image_rgb.copy()

        observation = self._prepare_obs(
            observation, device=self.device, task=self.prompt, robot_type=""
        )
        observation = self._preprocess_pi0(
            observation,
            stats=self._stats,
            all_features=self._all_features,
            tokenizer=self._tokenizer,
            device=self.device,
            max_length=self.policy.config.tokenizer_max_length,
            add_batch_dim=True,
            image_resolution=self._image_resolution,
        )
        action_chunk = self.policy.predict_action_chunk(observation)
        action_chunk = self._postprocess_pi0(
            action_chunk,
            stats=self._stats,
            output_features=self._output_features,
        )
        return action_chunk[0].numpy()  # (chunk_size, action_dim)


# ═══════════════════════════════════════════════════════════════════════════
#  SafeDeploymentNode
# ═══════════════════════════════════════════════════════════════════════════

class SafeDeploymentNode(Node):
    """Single ROS2 node connecting to the hardware-bridge topics.

    Subscribes to (from the hardware bridge):
        so101real/joint_state    — JointState from robot_node_launcher
        so101real/camera/image   — Image from webcam.py

    Publishes to (consumed by the hardware bridge):
        so101real/joint_command  — Float64MultiArray to robot_node_launcher

    At each tick of the control timer:
      1. Grab the latest camera image and joint state from the hardware bridge.
      2. If the action queue is empty, run PI0 to get a new action chunk (all
         chunk_size actions are enqueued and executed one-per-tick).
      3. Pass the proposed action through the safety filter.
      4. Publish the (possibly overridden) action back to the hardware bridge.
    """

    def __init__(
        self,
        wm_checkpoint: str,
        actor_checkpoint: str,
        critic_checkpoint: str,
        stats_path: str,
        pi0_model_id: str = PI0_MODEL_ID,
        prompt: str = DEFAULT_PROMPT,
        control_hz: float = CONTROL_HZ,
        epsilon: float = SAFETY_EPSILON,
        device: str = "cuda",
        disable_safety: bool = False,
    ):
        super().__init__("safe_deployment_node")

        self._disable_safety = disable_safety

        # Latest sensor data (written by ROS callbacks, read by control timer)
        self._latest_image: np.ndarray | None = None
        self._latest_proprio: np.ndarray | None = None
        self._lock = threading.Lock()

        self._action_queue: deque[np.ndarray] = deque()

        # --- Load PI0 ---
        self.get_logger().info("Loading PI0 base policy ...")
        self._pi0 = PI0Wrapper(
            model_id=pi0_model_id,
            prompt=prompt,
            stats_path=stats_path,
            device=device,
        )

        # --- Load safety filter ---
        if not disable_safety:
            from latentsafe.safety_filter import LatentSafetyFilter
            self.get_logger().info("Loading Latent Safety Filter ...")
            self._safety = LatentSafetyFilter(
                wm_checkpoint=wm_checkpoint,
                ddpg_actor_checkpoint=actor_checkpoint,
                ddpg_critic_checkpoint=critic_checkpoint,
                stats_path=stats_path,
                device=device,
                epsilon=epsilon,
            )
            self._safety.reset()
        else:
            self._safety = None
            self.get_logger().warn("Safety filter DISABLED — running PI0 unfiltered!")

        # --- ROS subscriptions and publisher ---
        # Sensor topics: BEST_EFFORT (high-rate, drop-ok)
        sensor_qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        # Command topic: RELIABLE (don't drop joint commands)
        cmd_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            JointState, f"{ROBOT_NS}/joint_state", self._cb_joint_state, sensor_qos
        )
        self.create_subscription(
            Image, f"{ROBOT_NS}/camera/image", self._cb_camera, sensor_qos
        )
        self._cmd_pub = self.create_publisher(
            Float64MultiArray, f"{ROBOT_NS}/joint_command", cmd_qos
        )

        self.create_timer(1.0 / control_hz, self._control_tick)

        self.get_logger().info(
            f"SafeDeploymentNode: {control_hz} Hz | "
            f"safety={'ON ε=' + str(epsilon) if not disable_safety else 'OFF'} | "
            f"Waiting for hardware bridge topics ..."
        )

    # ------------------------------------------------------------------
    # ROS callbacks (receive data from the hardware bridge)
    # ------------------------------------------------------------------

    def _cb_joint_state(self, msg: JointState):
        """Receive joint positions from hardware/ros/scripts/robot_node_launcher."""
        positions = np.array(msg.position[:PROPRIO_DIM], dtype=np.float32)
        with self._lock:
            self._latest_proprio = positions

    def _cb_camera(self, msg: Image):
        """Receive camera frames from hardware/ros/examples/webcam.py."""
        frame_bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            self._latest_image = frame_rgb

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_tick(self):
        with self._lock:
            image = self._latest_image
            proprio = self._latest_proprio

        if image is None or proprio is None:
            return  # Hardware bridge not publishing yet

        # --- Refill action queue from PI0 when exhausted ---
        if len(self._action_queue) == 0:
            t0 = time.perf_counter()
            try:
                action_chunk = self._pi0.predict_action_chunk(image, proprio)
                self._action_queue.extend(action_chunk)  # enqueue all chunk_size actions
            except Exception as e:
                self.get_logger().error(f"PI0 inference failed: {e}")
                return
            dt = time.perf_counter() - t0
            self.get_logger().info(
                f"PI0 inference: {dt * 1000:.0f} ms — {len(self._action_queue)} actions queued"
            )

        proposed_action = self._action_queue.popleft()

        # --- Safety filter ---
        if self._safety is not None:
            t0 = time.perf_counter()
            image_224 = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            action_raw, info = self._safety.step(image_224, proprio, proposed_action)
            dt = time.perf_counter() - t0

            if info.get("overridden", False):
                self._action_queue.clear()
                self.get_logger().warn(
                    f"SAFETY OVERRIDE: V={info['safety_value']:.3f} "
                    f"(ε={self._safety.epsilon}). Queue flushed."
                )
            elif not info.get("warmup", False):
                self.get_logger().debug(
                    f"Safe: V={info['safety_value']:.3f}, "
                    f"fail={info['failure_score']:.3f}, "
                    f"dt={dt * 1000:.0f}ms"
                )
        else:
            action_raw = proposed_action

        # --- Publish to the hardware bridge (robot_node_launcher) ---
        msg = Float64MultiArray()
        msg.data = action_raw.tolist()
        self._cmd_pub.publish(msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset_episode(self):
        """Call when starting a new task episode."""
        self._action_queue.clear()
        self._pi0.reset()
        if self._safety is not None:
            self._safety.reset()
        self.get_logger().info("Episode reset.")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Deploy PI0 + Latent Safety Filter on a real SO101 arm.\n\n"
            "Requires the hardware bridge nodes to be running first (from repo root):\n"
            "  python -m hardware.ros.scripts.robot_node_launcher robot=so101real &\n"
            "  python hardware/ros/examples/webcam.py &\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--wm-checkpoint",
        default="outputs/dino_wm_v2/checkpoints/latest/model.pt",
        help="Path to VWorldModel weights.",
    )
    p.add_argument(
        "--actor-checkpoint",
        default="outputs/safety_ddpg/checkpoints/epoch_0015/actor.pt",
        help="Path to SafetyDDPG actor weights.",
    )
    p.add_argument(
        "--critic-checkpoint",
        default="outputs/safety_ddpg/checkpoints/epoch_0015/critic.pt",
        help="Path to SafetyDDPG critic weights.",
    )
    p.add_argument(
        "--stats-path",
        default=str(ASSETS / "new_stats_format.json"),
        help="Path to dataset stats JSON (state/action normalization).",
    )
    p.add_argument("--pi0-model", default=PI0_MODEL_ID)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--epsilon", type=float, default=SAFETY_EPSILON)
    p.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    p.add_argument("--device", default="cuda")
    p.add_argument("--disable-safety", action="store_true",
                   help="Run PI0 without the safety filter (for comparison).")
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = SafeDeploymentNode(
        wm_checkpoint=args.wm_checkpoint,
        actor_checkpoint=args.actor_checkpoint,
        critic_checkpoint=args.critic_checkpoint,
        stats_path=args.stats_path,
        pi0_model_id=args.pi0_model,
        prompt=args.prompt,
        control_hz=args.control_hz,
        epsilon=args.epsilon,
        device=args.device,
        disable_safety=args.disable_safety,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
