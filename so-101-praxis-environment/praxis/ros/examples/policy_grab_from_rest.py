"""
Policy Inference Node

Calls the remote policy REST API with the latest camera frame and joint
positions, receives a 50-step action chunk, and publishes each action
to so101real/joint_command at 20hz.

Observations (proprio + images) are captured only AFTER all actions in
the current chunk have been executed AND the robot has settled to its
final position, ensuring the model sees the robot's true post-action
state.

Usage:
  python praxis/ros/examples/policy_node.py

Environment variables (optional overrides):
  POLICY_URL    — default: https://wl8s9yjg56qyvf-8000.proxy.runpod.net/
  POLICY_PROMPT — natural language task prompt
"""

import os
import sys
import base64
import threading
import time
import queue
import tty
import termios
import cv2
import numpy as np
import requests
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Image

# ── config ─────────────────────────────────────────────────────────────────
POLICY_URL       = os.environ.get(
    "POLICY_URL", "https://wl8s9yjg56qyvf-8000.proxy.runpod.net/"
)
PREDICT_ENDPOINT = f"{POLICY_URL}/predict_json"
HEALTH_ENDPOINT  = f"{POLICY_URL}/health"

PROMPT           = os.environ.get(
    "POLICY_PROMPT", "pick up the middle domino from the three domino row and place it flat on top of the other two dominos to form an arch"
)
ROBOT_NS         = "so101real"
ACTION_HZ        = 30    # how fast to publish actions from the chunk
CHUNK_SIZE       = 50    # expected actions per chunk
OBS_WAIT_TIMEOUT = 10.0  # seconds to wait for first camera + joint data

# ── settle config ──────────────────────────────────────────────────────────
SETTLE_TOLERANCE = 1.5   # degrees — max per-joint error to consider "settled"
SETTLE_TIMEOUT   = 2.0   # seconds — max time to wait for settling
SETTLE_POLL_HZ   = 50    # how often to check joint positions while settling

# ── ANSI ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def cprint(msg):
    print(msg, flush=True)

def get_keypress():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class PolicyNode(Node):

    def __init__(self, prompt: str):
        super().__init__("policy_node")
        self.prompt = prompt

        # latest observations
        self.latest_positions = None
        self.latest_frame     = None

        # chunk execution state
        self.current_chunk = []
        self.chunk_index   = 0
        self.running       = False
        self.estopped      = False

        # the last action published — used to detect when the robot settles
        self.last_commanded_action = None

        # prefetch queue — background thread puts next chunk here
        self.next_chunk_queue = queue.Queue(maxsize=1)
        self.fetch_lock       = threading.Lock()
        self.fetching         = False

        # event signalling that the current chunk is fully executed
        # AND the robot has settled
        self.chunk_done_event = threading.Event()

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            JointState, f"{ROBOT_NS}/joint_state",  self._cb_joint_state, qos)
        self.create_subscription(
            Image,      f"{ROBOT_NS}/camera/image", self._cb_image,       qos)

        self._pub = self.create_publisher(
            Float64MultiArray, f"{ROBOT_NS}/joint_command", 10)

        # execution timer
        self.create_timer(1.0 / ACTION_HZ, self._tick)

        cprint(f"\n{BOLD}Policy node ready{RESET}  "
               f"{DIM}|  's' start   'e' estop   'q' quit{RESET}")
        cprint(f"{DIM}Prompt : \"{self.prompt}\"{RESET}")
        cprint(f"{DIM}Server : {POLICY_URL}{RESET}\n")

    # ── ROS callbacks ──────────────────────────────────────────────────────

    def _cb_joint_state(self, msg: JointState):
        self.latest_positions = np.array(msg.position, dtype=np.float32)

    def _cb_image(self, msg: Image):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
        self.latest_frame = frame.copy()

    # ── observation helpers ────────────────────────────────────────────────

    def _wait_for_observations(self) -> bool:
        """
        Poll until both camera and joint data have arrived, or timeout.
        Called from keyboard thread — spin() runs in main thread so
        callbacks fire normally during this wait.
        """
        cprint(f"{CYAN}  Waiting for camera + joint state …{RESET}")
        t0 = time.monotonic()
        while time.monotonic() - t0 < OBS_WAIT_TIMEOUT:
            if self.latest_frame is not None and self.latest_positions is not None:
                cprint(f"{GREEN}  ✓ Observations ready{RESET}")
                return True
            time.sleep(0.1)
        cprint(f"{RED}✗  Timed out waiting for observations after "
               f"{OBS_WAIT_TIMEOUT:.0f}s{RESET}")
        cprint(f"{RED}   Is the robot node running? Is the camera node running?{RESET}")
        return False

    def _wait_for_settle(self) -> bool:
        """
        Block until the reported joint positions are within
        SETTLE_TOLERANCE of the last commanded action, or until
        SETTLE_TIMEOUT expires.

        Called from the prefetch background thread.  ROS callbacks
        continue updating self.latest_positions from the main spin
        thread, so we just poll.

        Returns True if settled, False if timed out.
        """
        target = self.last_commanded_action
        if target is None:
            return True  # nothing commanded yet, nothing to wait for

        poll_interval = 1.0 / SETTLE_POLL_HZ
        t0 = time.monotonic()

        while time.monotonic() - t0 < SETTLE_TIMEOUT:
            pos = self.latest_positions
            if pos is not None and len(pos) == len(target):
                error = np.abs(pos - target)
                max_err = float(np.max(error))
                if max_err <= SETTLE_TOLERANCE:
                    dt = time.monotonic() - t0
                    cprint(
                        f"\r{GREEN}✓  Settled{RESET}  "
                        f"max error: {max_err:.2f}°  "
                        f"{CYAN}({dt*1000:.0f}ms){RESET}          "
                    )
                    return True
            time.sleep(poll_interval)

        # timed out — log the residual error for debugging
        pos = self.latest_positions
        if pos is not None and target is not None and len(pos) == len(target):
            error = np.abs(pos - target)
            cprint(
                f"\r{YELLOW}⚠  Settle timeout{RESET}  "
                f"max error: {float(np.max(error)):.2f}°  "
                f"per-joint: [{' '.join(f'{e:.1f}' for e in error)}]          "
            )
        else:
            cprint(f"\r{YELLOW}⚠  Settle timeout (no position data){RESET}")
        return False

    def _snapshot_observations(self):
        """
        Capture a consistent snapshot of the current observations.
        Must be called when the robot has settled after executing
        all actions in a chunk.
        """
        frame = self.latest_frame
        positions = self.latest_positions
        if frame is None or positions is None:
            return None, None
        return frame.copy(), positions.copy()

    # ── policy fetch ───────────────────────────────────────────────────────

    def _frame_to_b64(self, frame: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Failed to encode frame as JPEG")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _fetch_chunk_with_obs(
        self, frame: np.ndarray, positions: np.ndarray
    ) -> list | None:
        """
        Blocking HTTP call using the provided observation snapshot.
        Returns list of np arrays or None on error.
        """
        payload = {
            "prompt": self.prompt,
            "images": [self._frame_to_b64(frame)],
            "proprio": positions.tolist(),
        }
        try:
            t0 = time.monotonic()
            r  = requests.post(PREDICT_ENDPOINT, json=payload, timeout=60.0)
            dt = time.monotonic() - t0
            r.raise_for_status()
            data = r.json()
            raw  = data["action"]

            # Unwrap extra nesting if server returns [[[j1,j2,...], ...]]
            # instead of [[j1,j2,...], ...]
            if (len(raw) == 1
                    and isinstance(raw[0], list)
                    and isinstance(raw[0][0], list)):
                raw = raw[0]

            actions = [np.array(a, dtype=np.float32) for a in raw]
            cprint(
                f"\r{GREEN}↓  chunk received{RESET}  "
                f"{len(actions)} actions  "
                f"{CYAN}({dt*1000:.0f}ms){RESET}          "
            )
            return actions

        except requests.exceptions.Timeout:
            cprint(f"\n{RED}✗  Policy server timed out{RESET}")
        except requests.exceptions.ConnectionError as e:
            cprint(f"\n{RED}✗  Connection error: {e}{RESET}")
        except requests.exceptions.HTTPError:
            cprint(f"\n{RED}✗  HTTP {r.status_code}: {r.text[:300]}{RESET}")
        except (KeyError, ValueError) as e:
            cprint(f"\n{RED}✗  Bad response from server: {e}{RESET}")
        return None

    def _prefetch_worker(self):
        """
        Background thread:
        1. Waits for chunk_done_event (all actions published)
        2. Waits for the robot to physically settle at the last target
        3. Snapshots observations
        4. Fetches the next chunk from the policy server
        """
        # Block until the tick loop signals that all actions are published
        self.chunk_done_event.wait()
        self.chunk_done_event.clear()

        # Bail if we were woken by estop/stop
        if not self.running or self.estopped:
            with self.fetch_lock:
                self.fetching = False
            return

        # Wait for the robot to reach the last commanded position
        self._wait_for_settle()

        # Now snapshot — the robot has physically arrived
        frame, positions = self._snapshot_observations()
        if frame is None or positions is None:
            cprint(f"{YELLOW}⚠  No observations available for fetch{RESET}")
            with self.fetch_lock:
                self.fetching = False
            return

        cprint(
            f"\r{DIM}  Obs captured  "
            f"proprio: [{' '.join(f'{v:.1f}' for v in positions)}]{RESET}          "
        )

        chunk = self._fetch_chunk_with_obs(frame, positions)
        if chunk is not None:
            try:
                self.next_chunk_queue.put_nowait(chunk)
            except queue.Full:
                pass
        with self.fetch_lock:
            self.fetching = False

    def _trigger_prefetch(self):
        with self.fetch_lock:
            if self.fetching:
                return
            self.fetching = True
        t = threading.Thread(target=self._prefetch_worker, daemon=True)
        t.start()

    # ── execution tick ─────────────────────────────────────────────────────

    def _tick(self):
        if not self.running or self.estopped:
            return

        steps_remaining = len(self.current_chunk) - self.chunk_index

        # Start the prefetch thread early so it's ready and waiting.
        # It blocks on chunk_done_event until all actions are published,
        # then blocks again on _wait_for_settle().
        if steps_remaining <= len(self.current_chunk) // 2 and steps_remaining > 0:
            self._trigger_prefetch()

        # current chunk exhausted — signal prefetch, then wait for next chunk
        if steps_remaining <= 0:
            # Signal the background worker: "all actions are published"
            self.chunk_done_event.set()

            try:
                # Wait for settle + fetch to complete in the background thread
                self.current_chunk = self.next_chunk_queue.get(timeout=15.0)
                self.chunk_index   = 0
                steps_remaining    = len(self.current_chunk)
            except queue.Empty:
                cprint(f"\n{YELLOW}⚠  Waiting for next chunk …{RESET}")
                return

        # publish current action
        action   = self.current_chunk[self.chunk_index]
        msg      = Float64MultiArray()
        msg.data = action.tolist()
        self._pub.publish(msg)

        # track the last action we commanded for settle detection
        self.last_commanded_action = action.copy()

        print(
            f"\r{GREEN}▶ step {self.chunk_index+1:>2}/{len(self.current_chunk)}{RESET}  "
            f"remaining: {steps_remaining-1:>2}  "
            f"{CYAN}joints: [{' '.join(f'{v:6.1f}' for v in action)}]{RESET}   ",
            end="", flush=True,
        )
        self.chunk_index += 1

    # ── start / estop / stop ───────────────────────────────────────────────

    def start(self):
        """Called from keyboard thread — waits for obs then fetches first chunk."""
        if self.running:
            return
        if self.estopped:
            cprint(f"{RED}✗  E-stopped — restart the node to resume.{RESET}")
            return

        if not self._wait_for_observations():
            return

        # For the very first chunk, the robot is stationary — no settle needed
        frame, positions = self._snapshot_observations()
        if frame is None or positions is None:
            cprint(f"{RED}✗  Failed to snapshot observations.{RESET}")
            return

        cprint(f"{CYAN}  Fetching first chunk …{RESET}")
        chunk = self._fetch_chunk_with_obs(frame, positions)
        if chunk is None:
            cprint(f"{RED}✗  Failed to get first chunk — not starting.{RESET}")
            return

        self.current_chunk = chunk
        self.chunk_index   = 0
        self.running       = True
        cprint(f"{GREEN}{BOLD}▶  Executing!{RESET}  {DIM}('e' to estop){RESET}")

    def estop(self):
        self.running  = False
        self.estopped = True
        # Unblock any waiting prefetch thread
        self.chunk_done_event.set()
        print()
        cprint(f"{RED}{BOLD}⛔  E-STOP — policy execution halted.{RESET}")
        cprint(f"{DIM}Restart the node to resume.{RESET}")

    def stop(self):
        self.running = False
        # Unblock any waiting prefetch thread
        self.chunk_done_event.set()
        print()
        cprint(f"{YELLOW}■  Stopped.{RESET}")


# ── keyboard thread ────────────────────────────────────────────────────────

def keyboard_thread(node: PolicyNode):
    while rclpy.ok():
        key = get_keypress()
        if key == "s":
            node.start()
        elif key == "e":
            node.estop()
        elif key == "q":
            node.stop()
            rclpy.shutdown()
            break


def check_health() -> bool:
    cprint(f"{CYAN}Checking policy server health …{RESET}")
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=5.0)
        r.raise_for_status()
        cprint(f"{GREEN}✓  Server healthy: {r.json()}{RESET}")
        return True
    except Exception as e:
        cprint(f"{RED}✗  Server unreachable: {e}{RESET}")
        return False


def main():
    prompt = input(f"{BOLD}Task prompt [{PROMPT}]: {RESET}").strip() or PROMPT

    if not check_health():
        ans = input(
            f"{YELLOW}Server unhealthy — continue anyway? [y/N]: {RESET}"
        ).strip().lower()
        if ans != "y":
            return

    rclpy.init()
    node = PolicyNode(prompt=prompt)

    kb = threading.Thread(target=keyboard_thread, args=(node,), daemon=True)
    kb.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

