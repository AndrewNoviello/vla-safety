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

Real-time chunking (RTC):

  The control timer runs at `--control-hz` and consumes one action per tick
  from an ActionQueue. A dedicated PI0 worker thread runs *concurrently* with
  the control loop; it re-runs `predict_action_chunk` whenever the queue
  drops to the execution-horizon watermark, passing the leftover tail of the
  current chunk as `prev_chunk_left_over` so the new chunk's prefix is
  inpainting-guided toward the old plan. The first `real_delay` actions of
  every new chunk are discarded so the chunk slots in at the controller's
  current position. The ROS executor is multi-threaded so that camera and
  joint-state callbacks keep firing during inference.

  Reference: https://www.physicalintelligence.company/download/real_time_chunking.pdf

Data flow:

  hardware/webcam ──→ so101real/camera/image ──┐
                                               ├──→ PI0 (base policy) → action chunk ──┐
  hardware/robot  ──→ so101real/joint_state  ──┘                                        │
                                                                                        ▼
                                                                         ┌──────────────────────────┐
                                                                         │ LatentSafetyFilter       │
                                                                         │ V(ẑ_{t+1}) > ε ?         │
                                                                         │ YES → pass through       │
                                                                         │ NO  → override + flush   │
                                                                         └──────────────────────────┘
                                                                                        │
                                                                                        ▼
  hardware/robot  ←── so101real/joint_command ←────────────────────────────────────────┘

Prerequisites (run from repo root first):
    python -m hardware.ros.scripts.robot_node_launcher robot=so101real &
    python hardware/ros/examples/webcam.py &

Then run this script:
    python scripts/deploy.py \\
        --wm-checkpoint outputs/dino_wm_v2/checkpoints/latest/model.pt \\
        --actor-checkpoint outputs/safety_ddpg/checkpoints/epoch_0015/actor.pt \\
        --critic-checkpoint outputs/safety_ddpg/checkpoints/epoch_0015/critic.pt \\
        --prompt "pick up the domino"

    # Without safety filter (for comparison / debugging):
    python scripts/deploy.py --disable-safety

    # Async inference without RTC guidance (still overlapped, but seams may jump):
    python scripts/deploy.py --no-rtc
"""

import argparse
import logging
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

from utils.paths import REPO_ROOT

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

EXECUTION_HORIZON_DEFAULT = 10
NUM_INFERENCE_STEPS_DEFAULT = 10
DELAY_BUFFER_SIZE = 8
WARMUP_ITERS = 3

PI0_MODEL_ID = "AndrewNoviello/vla-safety-task-4"
DEFAULT_PROMPT = (
    "pick up the middle domino from the three domino row and place it flat "
    "on top of the other two dominos to form an arch"
)


# ═══════════════════════════════════════════════════════════════════════════
#  PI0 wrapper
# ═══════════════════════════════════════════════════════════════════════════

class PI0Wrapper:
    """Wraps PI0Policy for RTC-aware action-chunk inference.

    `predict_action_chunk` returns *both* the normalized chunk (for use as the
    next call's `prev_chunk_left_over`) and the post-processed chunk (in raw
    joint-command space, for execution on the robot).
    """

    def __init__(
        self,
        model_id: str,
        prompt: str,
        stats_path: str,
        device: str,
        *,
        rtc_enabled: bool = True,
        execution_horizon: int = EXECUTION_HORIZON_DEFAULT,
        num_inference_steps: int = NUM_INFERENCE_STEPS_DEFAULT,
        use_amp: bool = True,
    ):
        from transformers import AutoTokenizer
        from pi0.config import PI0Config
        from pi0.policy import PI0Policy
        from pi0.processor import preprocess_pi0, postprocess_pi0
        from pi0.rtc_config import RTCConfig
        from utils.types import RTCAttentionSchedule
        from utils.utils import cast_stats_to_numpy, load_json
        from utils.processor_utils import (
            add_batch_dim,
            normalize,
            prepare_observation_for_inference,
            prepare_stats,
            resize_images_in_batch,
            to_device,
        )
        from utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
            OBS_STATE,
        )

        self.device = torch.device(device)
        self.prompt = prompt
        self.rtc_enabled = rtc_enabled
        self.execution_horizon = int(execution_horizon)
        self._use_amp = use_amp and self.device.type == "cuda"

        logger.info(f"Loading PI0 from {model_id} ...")
        config = PI0Config.from_pretrained(model_id)
        config.compile_model = False
        config.device = str(self.device)
        config.num_inference_steps = int(num_inference_steps)
        config.use_amp = self._use_amp

        if rtc_enabled:
            config.rtc_config = RTCConfig(
                enabled=True,
                execution_horizon=self.execution_horizon,
                prefix_attention_schedule=RTCAttentionSchedule.EXP,
                max_guidance_weight=10.0,
            )
            logger.info(
                f"RTC ON: execution_horizon={self.execution_horizon} "
                f"schedule=EXP num_inference_steps={num_inference_steps} "
                f"use_amp={self._use_amp}"
            )
        else:
            config.rtc_config = None
            logger.info(
                f"RTC OFF (async inference only). "
                f"num_inference_steps={num_inference_steps} use_amp={self._use_amp}"
            )

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
        self.chunk_size = int(self.policy.config.chunk_size)

        self._tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        self._image_resolution = tuple(self.policy.image_resolution)

        # Hold references to the helpers we need at inference time.
        self._preprocess_pi0 = preprocess_pi0
        self._postprocess_pi0 = postprocess_pi0
        self._prepare_obs = prepare_observation_for_inference
        self._add_batch_dim = add_batch_dim
        self._normalize = normalize
        self._resize_images = resize_images_in_batch
        self._to_device = to_device
        self._OBS_STATE = OBS_STATE
        self._OBS_LANG_TOK = OBS_LANGUAGE_TOKENS
        self._OBS_LANG_MASK = OBS_LANGUAGE_ATTENTION_MASK

        # The prompt is immutable during a deployment — tokenize once.
        task = prompt if prompt.endswith("\n") else f"{prompt}\n"
        tokenized = self._tokenizer(
            [task],
            max_length=self.policy.config.tokenizer_max_length,
            truncation=True,
            padding="max_length",
            padding_side="right",
            return_tensors="pt",
        )
        self._cached_tokens = tokenized["input_ids"].to(self.device)
        self._cached_mask = tokenized["attention_mask"].to(
            dtype=torch.bool, device=self.device
        )

        logger.info("PI0 loaded and ready.")

    def reset(self):
        self.policy.reset()

    def _build_observation(self, image_rgb: np.ndarray, proprio_raw: np.ndarray) -> dict:
        from utils.types import FeatureType

        observation: dict = {}
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

        for k, v in input_features.items():
            if v.type == FeatureType.VISUAL:
                observation[k] = image_rgb.copy()
        return observation

    def _preprocess_cached(self, observation: dict) -> dict:
        """Run the same pipeline as `preprocess_pi0` but skip tokenization.

        Tokenization is replaced by injecting the pre-tokenized prompt that
        was computed once in `__init__`.
        """
        batch = self._add_batch_dim(observation)
        batch[self._OBS_LANG_TOK] = self._cached_tokens
        batch[self._OBS_LANG_MASK] = self._cached_mask
        batch = self._to_device(batch, self.device)
        batch = self._resize_images(batch, self._image_resolution, self._all_features)
        batch = self._normalize(batch, self._stats, self._all_features)
        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self,
        image_rgb: np.ndarray,
        proprio_raw: np.ndarray,
        *,
        prev_chunk_left_over: torch.Tensor | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one PI0 inference and return the chunk in two forms.

        Returns:
            original:  (chunk_size, action_dim) tensor on the policy device,
                       still in the model's normalized output space. Pass this
                       (as a slice) into the next call's `prev_chunk_left_over`.
            processed: (chunk_size, action_dim) tensor on CPU, post-processed
                       (un-normalized) into raw joint-command space. Feed this
                       to the robot.
        """
        observation = self._build_observation(image_rgb, proprio_raw)
        observation = self._preprocess_cached(observation)

        rtc_kwargs: dict = {}
        if self.rtc_enabled:
            if prev_chunk_left_over is not None:
                rtc_kwargs["prev_chunk_left_over"] = prev_chunk_left_over.to(self.device)
            if inference_delay is not None:
                rtc_kwargs["inference_delay"] = int(inference_delay)
            if execution_horizon is not None:
                rtc_kwargs["execution_horizon"] = int(execution_horizon)

        if self._use_amp:
            ctx = torch.autocast(device_type="cuda")
        else:
            ctx = torch.amp.autocast("cpu", enabled=False)

        with ctx:
            action_norm = self.policy.predict_action_chunk(observation, **rtc_kwargs)

        action_proc = self._postprocess_pi0(
            action_norm,
            stats=self._stats,
            output_features=self._output_features,
        )

        return action_norm[0].detach(), action_proc[0].detach()

    def warmup(self, n: int = WARMUP_ITERS) -> None:
        """Run a few dummy inferences so the first real one is fast."""
        logger.info(f"Warming up PI0 ({n} inferences) ...")
        dummy_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        dummy_proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        for i in range(n):
            t0 = time.perf_counter()
            self.predict_action_chunk(dummy_img, dummy_proprio)
            dt = time.perf_counter() - t0
            logger.info(f"  warmup {i + 1}/{n}: {dt * 1000:.0f} ms")
        self.reset()


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

    Threading model:
      - Sensor callbacks fire on a ReentrantCallbackGroup (parallel-safe).
      - The control timer fires on a MutuallyExclusiveCallbackGroup so two
        ticks never overlap.
      - A dedicated worker thread (not a ROS callback) runs PI0 inference
        whenever the ActionQueue drops to the execution-horizon watermark.
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
        *,
        rtc_enabled: bool = True,
        execution_horizon: int = EXECUTION_HORIZON_DEFAULT,
        num_inference_steps: int = NUM_INFERENCE_STEPS_DEFAULT,
        use_amp: bool = True,
    ):
        super().__init__("safe_deployment_node")

        from pi0.rtc_action_queue import ActionQueue
        from pi0.rtc_config import RTCConfig
        from utils.types import RTCAttentionSchedule

        self._disable_safety = disable_safety
        self._control_hz = float(control_hz)
        self._exec_horizon = int(execution_horizon)

        self._latest_image: np.ndarray | None = None
        self._latest_proprio: np.ndarray | None = None
        self._sensor_lock = threading.Lock()

        # --- Load PI0 ---
        self.get_logger().info("Loading PI0 base policy ...")
        self._pi0 = PI0Wrapper(
            model_id=pi0_model_id,
            prompt=prompt,
            stats_path=stats_path,
            device=device,
            rtc_enabled=rtc_enabled,
            execution_horizon=execution_horizon,
            num_inference_steps=num_inference_steps,
            use_amp=use_amp,
        )

        # --- Action queue ---
        # We always run the queue in "replace" mode (cfg.enabled=True) so that
        # async inference correctly drops the first `real_delay` actions of
        # every new chunk, regardless of whether RTC *guidance* is enabled.
        queue_cfg = RTCConfig(
            enabled=True,
            execution_horizon=self._exec_horizon,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
            max_guidance_weight=10.0,
        )
        self._action_queue = ActionQueue(queue_cfg)

        # Rolling buffer of past real_delay measurements; we pass max(buffer)
        # as a conservative `inference_delay` estimate to the next call, per
        # Algorithm 1 of the RTC paper.
        self._delay_buffer: deque[int] = deque(maxlen=DELAY_BUFFER_SIZE)
        self._first_inference = True

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

        # Warm up PI0 *before* we start serving the control timer so the first
        # real inference doesn't pay JIT / cudnn-autotune cost.
        self._pi0.warmup(n=WARMUP_ITERS)

        # --- ROS QoS profiles ---
        sensor_qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        cmd_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # --- Callback groups for the multi-threaded executor ---
        self._sensor_cbg = ReentrantCallbackGroup()
        self._control_cbg = MutuallyExclusiveCallbackGroup()

        self.create_subscription(
            JointState,
            f"{ROBOT_NS}/joint_state",
            self._cb_joint_state,
            sensor_qos,
            callback_group=self._sensor_cbg,
        )
        self.create_subscription(
            Image,
            f"{ROBOT_NS}/camera/image",
            self._cb_camera,
            sensor_qos,
            callback_group=self._sensor_cbg,
        )
        self._cmd_pub = self.create_publisher(
            Float64MultiArray, f"{ROBOT_NS}/joint_command", cmd_qos
        )

        self.create_timer(
            1.0 / self._control_hz,
            self._control_tick,
            callback_group=self._control_cbg,
        )

        # --- Inference worker thread ---
        self._refill_event = threading.Event()
        self._stop_event = threading.Event()
        self._inference_in_flight = threading.Event()
        self._worker = threading.Thread(
            target=self._inference_loop,
            name="pi0_inference_worker",
            daemon=True,
        )
        self._worker.start()
        # Kick off the first inference once sensors come online.
        self._refill_event.set()

        self.get_logger().info(
            f"SafeDeploymentNode: {self._control_hz:.1f} Hz | "
            f"safety={'ON ε=' + str(epsilon) if not disable_safety else 'OFF'} | "
            f"RTC={'ON' if rtc_enabled else 'OFF'} | "
            f"execution_horizon={self._exec_horizon} | "
            f"chunk_size={self._pi0.chunk_size} | "
            f"Waiting for hardware bridge topics ..."
        )

    # ------------------------------------------------------------------
    # ROS callbacks (receive data from the hardware bridge)
    # ------------------------------------------------------------------

    def _cb_joint_state(self, msg: JointState):
        positions = np.array(msg.position[:PROPRIO_DIM], dtype=np.float32)
        with self._sensor_lock:
            self._latest_proprio = positions

    def _cb_camera(self, msg: Image):
        frame_bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with self._sensor_lock:
            self._latest_image = frame_rgb

    # ------------------------------------------------------------------
    # Control loop (runs on the timer thread; never blocks on inference)
    # ------------------------------------------------------------------

    def _control_tick(self):
        action_t = self._action_queue.get()
        if action_t is None:
            # Starved: the inference worker hasn't landed a chunk yet (or the
            # queue was just flushed). Robot holds last setpoint until refill.
            self._refill_event.set()
            return
        proposed = action_t.cpu().numpy()

        action_raw = proposed
        if self._safety is not None:
            with self._sensor_lock:
                image, proprio = self._latest_image, self._latest_proprio
            if image is not None and proprio is not None:
                image_224 = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                t0 = time.perf_counter()
                action_raw, info = self._safety.step(image_224, proprio, proposed)
                safety_dt = time.perf_counter() - t0

                if info.get("overridden", False):
                    # Current plan is unsafe — flush and force a fresh chunk.
                    self._action_queue.clear()
                    self._refill_event.set()
                    self.get_logger().warn(
                        f"SAFETY OVERRIDE: V={info['safety_value']:.3f} "
                        f"(ε={self._safety.epsilon}). Queue flushed."
                    )
                elif not info.get("warmup", False):
                    self.get_logger().debug(
                        f"Safe: V={info['safety_value']:.3f}, "
                        f"fail={info['failure_score']:.3f}, "
                        f"safety_dt={safety_dt * 1000:.0f}ms"
                    )

        msg = Float64MultiArray()
        if isinstance(action_raw, np.ndarray):
            msg.data = action_raw.astype(float).tolist()
        else:
            msg.data = list(map(float, action_raw))
        self._cmd_pub.publish(msg)

        # Refill watermark: trigger the next inference while the tail of the
        # current chunk is still executing, so a new chunk arrives before
        # this one drains.
        if (
            self._action_queue.qsize() <= self._exec_horizon
            and not self._inference_in_flight.is_set()
        ):
            self._refill_event.set()

    # ------------------------------------------------------------------
    # Inference worker (runs on its own thread, overlapped with execution)
    # ------------------------------------------------------------------

    def _inference_loop(self):
        while not self._stop_event.is_set():
            self._refill_event.wait()
            self._refill_event.clear()
            if self._stop_event.is_set():
                return

            with self._sensor_lock:
                image, proprio = self._latest_image, self._latest_proprio
            if image is None or proprio is None:
                # Sensors not online yet. Sleep briefly and retry.
                time.sleep(0.05)
                self._refill_event.set()
                continue

            if self._first_inference:
                idx_before = None
                leftover = None
                delay_estimate = 0
            else:
                idx_before = self._action_queue.get_action_index()
                leftover = self._action_queue.get_left_over()
                delay_estimate = (
                    max(self._delay_buffer) if self._delay_buffer else self._exec_horizon
                )

            self._inference_in_flight.set()
            try:
                t0 = time.perf_counter()
                try:
                    original, processed = self._pi0.predict_action_chunk(
                        image,
                        proprio,
                        prev_chunk_left_over=leftover,
                        inference_delay=delay_estimate,
                        execution_horizon=self._exec_horizon,
                    )
                except Exception as e:
                    self.get_logger().error(f"PI0 inference failed: {e}")
                    continue
                dt = time.perf_counter() - t0

                real_delay = int(round(dt * self._control_hz))
                # Clamp so a transient slow inference doesn't discard the
                # whole chunk (leaving the queue immediately empty again).
                max_safe_delay = self._pi0.chunk_size - 1
                real_delay = max(0, min(real_delay, max_safe_delay))

                self._action_queue.merge(
                    original_actions=original,
                    processed_actions=processed,
                    real_delay=real_delay,
                    action_index_before_inference=idx_before,
                )

                if self._first_inference:
                    self._first_inference = False
                else:
                    self._delay_buffer.append(real_delay)

                self.get_logger().info(
                    f"PI0 inference: {dt * 1000:.0f} ms | "
                    f"real_delay={real_delay} d_est={delay_estimate} | "
                    f"queue={self._action_queue.qsize()}"
                )
            finally:
                self._inference_in_flight.clear()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset_episode(self):
        """Call when starting a new task episode."""
        self._action_queue.clear()
        self._delay_buffer.clear()
        self._first_inference = True
        self._pi0.reset()
        if self._safety is not None:
            self._safety.reset()
        self._refill_event.set()
        self.get_logger().info("Episode reset.")

    def shutdown(self):
        self._stop_event.set()
        self._refill_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)


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
        default=str(REPO_ROOT / "data" / "exp_success_v3_clean" / "meta" / "stats.json"),
        help="Path to dataset stats JSON (state/action normalization).",
    )
    p.add_argument("--pi0-model", default=PI0_MODEL_ID)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--epsilon", type=float, default=SAFETY_EPSILON)
    p.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--disable-safety",
        action="store_true",
        help="Run PI0 without the safety filter (for comparison).",
    )

    # --- RTC options ---
    rtc = p.add_mutually_exclusive_group()
    rtc.add_argument(
        "--rtc",
        dest="rtc",
        action="store_true",
        help="Enable RTC prefix-guided denoising (default).",
    )
    rtc.add_argument(
        "--no-rtc",
        dest="rtc",
        action="store_false",
        help="Disable RTC guidance (keep async inference but no inpainting).",
    )
    p.set_defaults(rtc=True)

    p.add_argument(
        "--execution-horizon",
        type=int,
        default=EXECUTION_HORIZON_DEFAULT,
        help=(
            "Refill watermark in controller ticks. Set ≳ expected inference "
            "delay so a new chunk lands before the queue drains."
        ),
    )
    p.add_argument(
        "--num-inference-steps",
        type=int,
        default=NUM_INFERENCE_STEPS_DEFAULT,
        help="Flow-matching denoising steps per chunk.",
    )

    amp = p.add_mutually_exclusive_group()
    amp.add_argument(
        "--use-amp",
        dest="use_amp",
        action="store_true",
        help="Use CUDA autocast (default on cuda).",
    )
    amp.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="Disable CUDA autocast.",
    )
    p.set_defaults(use_amp=True)

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
        rtc_enabled=args.rtc,
        execution_horizon=args.execution_horizon,
        num_inference_steps=args.num_inference_steps,
        use_amp=args.use_amp,
    )

    # 3 threads is plenty: sensor group (reentrant) + control timer + 1 spare.
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
