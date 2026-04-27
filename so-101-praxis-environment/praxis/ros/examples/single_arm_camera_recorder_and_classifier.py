"""
Single-Arm Recording Script — Flat Dataset Format Output

Subscribes to:
  so101real/joint_state     (20 Hz — master clock)
  so101real/joint_command   (20 Hz)
  so101real/ee_pose         (20 Hz)
  so101real/camera/image    (30 Hz — latest frame grabbed at each tick)

Keyboard:
  's' — start recording in GOOD state (when idle)
  'b' — start recording in BAD state (when idle)
  'g' — end trajectory as SUCCESS (when recording)
  'f' — mark failure, record 2 more seconds, then save (when recording)
  'd' — stop recording immediately and save (when recording in bad state)
  'q' — quit

Output layout  (matches the old flat format exactly, + label column):
  <DATA_ROOT>/<experiment_id>/
    data/
      episode_000.parquet     ← one file per episode
      episode_001.parquet
      …
    videos/
      episode_000.mp4         ← one file per episode
      episode_001.mp4
      …
    meta/
      stats.json              ← running min/max/mean/std over all episodes
    README.md

  Parquet columns per timestep:
    timestamp                 float64  — seconds since episode start
    observation.state_j{1‥6} float32  — joint positions
    action_j{1‥6}            float32  — joint commands
    ee_x/y/z/qx/qy/qz/qw    float32  — end-effector pose
    label                     float32  — 1.0 = GOOD, 0.0 = BAD  ← NEW
    frame_index               int64    — 0-based index within episode
"""

import sys
import json
import tty
import termios
import threading
import time
from pathlib import Path

import numpy as np
import cv2
import pyarrow as pa
import pyarrow.parquet as pq

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped

# ── Config ─────────────────────────────────────────────────────────────────
DATA_ROOT     = Path("/workspace/praxis-core/data")
ROBOT_NS      = "so101real"
FAILURE_EXTRA = 2.0    # seconds to keep recording after failure keypress
RECORD_FPS    = 20     # master clock / dataset fps
TASK_NAME     = "Single-arm manipulation"

CAM_HEIGHT = 480
CAM_WIDTH  = 640

# ── ANSI colors ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ── Helpers ────────────────────────────────────────────────────────────────

def cprint(msg: str):
    print(msg, flush=True)


def print_live(total_frames, n_good, n_bad, positions, failure_countdown=None, bad_mode=False):
    if failure_countdown is not None:
        state = f"{RED}{BOLD}✗ FAILURE — saving in {failure_countdown:.1f}s …{RESET}"
        color = RED
        label = f"{RED}{n_bad} bad{RESET}"
    elif bad_mode:
        state = f"{RED}{BOLD}✗ BAD STATE{RESET}"
        color = RED
        label = f"{RED}{n_bad} bad{RESET}"
    else:
        state = f"{GREEN}{BOLD}✓ GOOD{RESET}"
        color = GREEN
        label = f"{GREEN}{n_good} good{RESET}"

    joints = " ".join(f"{v:6.1f}" for v in positions) if positions is not None else "---"
    print(
        f"\r{state}  "
        f"{color}frame {total_frames:>5}{RESET}  "
        f"({label})  "
        f"{CYAN}joints: [{joints}]{RESET}   ",
        end="",
        flush=True,
    )


def get_keypress() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── Stats accumulator ──────────────────────────────────────────────────────

class RunningStats:
    """Incremental min / max / mean / std for scalar columns (excludes video)."""

    # Columns we track stats for (matches parquet scalar columns)
    COLS = (
        [f"observation.state_j{i+1}" for i in range(6)]
        + [f"action_j{i+1}" for i in range(6)]
        + ["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw"]
        + ["label"]
    )

    def __init__(self, path: Path):
        self.path = path
        self._n    = {c: 0     for c in self.COLS}
        self._mean = {c: 0.0   for c in self.COLS}
        self._M2   = {c: 0.0   for c in self.COLS}
        self._min  = {c: float("inf")  for c in self.COLS}
        self._max  = {c: float("-inf") for c in self.COLS}

        if path.exists():
            self._load()

    def update(self, col: str, value: float):
        self._n[col] += 1
        delta = value - self._mean[col]
        self._mean[col] += delta / self._n[col]
        self._M2[col] += delta * (value - self._mean[col])
        if value < self._min[col]:
            self._min[col] = value
        if value > self._max[col]:
            self._max[col] = value

    def update_frame(self, row: dict):
        for col in self.COLS:
            if col in row:
                self.update(col, float(row[col]))

    def save(self):
        stats = {}
        for col in self.COLS:
            n = self._n[col]
            std = (self._M2[col] / n) ** 0.5 if n > 1 else 0.0
            stats[col] = {
                "min":  self._min[col] if n > 0 else None,
                "max":  self._max[col] if n > 0 else None,
                "mean": self._mean[col] if n > 0 else None,
                "std":  std if n > 0 else None,
                "count": n,
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(stats, indent=2))

    def _load(self):
        try:
            stats = json.loads(self.path.read_text())
            for col in self.COLS:
                if col not in stats:
                    continue
                s = stats[col]
                n = s.get("count", 0)
                if n == 0:
                    continue
                self._n[col]    = n
                self._mean[col] = s["mean"]
                # Reconstruct M2 from stored std so we can continue accumulating
                self._M2[col]   = (s["std"] ** 2) * n if s["std"] is not None else 0.0
                self._min[col]  = s["min"]
                self._max[col]  = s["max"]
        except Exception as e:
            cprint(f"{YELLOW}⚠  Could not load existing stats ({e}) — starting fresh.{RESET}")


# ── Main recorder node ─────────────────────────────────────────────────────

class SingleArmRecorder(Node):

    def __init__(self, experiment_id: str, task_name: str = TASK_NAME):
        super().__init__("single_arm_recorder")
        self.experiment_id = experiment_id
        self.task_name     = task_name
        self.recording     = False
        self.bad_mode      = False   # True when started with 'b'
        self.failure_time  = None

        # Per-episode accumulator
        self._buf: list[dict] = []
        self._ep_start: float = 0.0

        # Latest sensor data
        self.latest_positions = None
        self.latest_commands  = None
        self.latest_ee_pose   = None
        self.latest_frame     = None

        # ── Dataset directories ────────────────────────────────────────────
        self.root       = DATA_ROOT / experiment_id
        self.data_dir   = self.root / "data"
        self.video_dir  = self.root / "videos"
        self.meta_dir   = self.root / "meta"
        for d in (self.data_dir, self.video_dir, self.meta_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Write README once
        readme = self.root / "README.md"
        if not readme.exists():
            readme.write_text(
                f"# {experiment_id}\n\n"
                f"Task: {task_name}\n\n"
                "Generated by single_arm_camera_recorder_and_classifier.py\n\n"
                "## Layout\n"
                "- `data/episode_NNN.parquet` — one file per episode\n"
                "- `videos/episode_NNN.mp4`   — one video per episode\n"
                "- `meta/stats.json`           — running scalar statistics\n\n"
                "## Parquet columns\n"
                "| Column | Type | Description |\n"
                "|--------|------|-------------|\n"
                "| timestamp | float64 | seconds since episode start |\n"
                "| frame_index | int64 | 0-based frame counter |\n"
                "| observation.state_j{1‒6} | float32 | joint positions |\n"
                "| action_j{1‒6} | float32 | joint commands |\n"
                "| ee_x/y/z/qx/qy/qz/qw | float32 | EE pose |\n"
                "| label | float32 | 1.0 = GOOD, 0.0 = BAD |\n"
            )

        # ── Stats ──────────────────────────────────────────────────────────
        self.stats = RunningStats(self.meta_dir / "stats.json")

        # ── Determine next episode number ──────────────────────────────────
        existing = sorted(self.data_dir.glob("episode_*.parquet"))
        self.traj_num = len(existing)  # next index to write

        cprint(
            f"{GREEN}Dataset root:{RESET} {self.root}\n"
            f"{GREEN}Episodes so far:{RESET} {BOLD}{self.traj_num}{RESET}"
        )

        # ── ROS subscriptions ──────────────────────────────────────────────
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(JointState,        f"{ROBOT_NS}/joint_state",   self._cb_joint_state,   qos)
        self.create_subscription(Float64MultiArray, f"{ROBOT_NS}/joint_command", self._cb_joint_command, qos)
        self.create_subscription(PoseStamped,       f"{ROBOT_NS}/ee_pose",       self._cb_ee_pose,       qos)
        self.create_subscription(Image,             f"{ROBOT_NS}/camera/image",  self._cb_image,         qos)

        self.create_timer(1.0 / RECORD_FPS, self._tick)
        self._print_status()

    # ── ROS callbacks ──────────────────────────────────────────────────────

    def _cb_joint_state(self, msg: JointState):
        self.latest_positions = np.array(msg.position[:6], dtype=np.float32)

    def _cb_joint_command(self, msg: Float64MultiArray):
        self.latest_commands = np.array(msg.data[:6], dtype=np.float32)

    def _cb_ee_pose(self, msg: PoseStamped):
        p, o = msg.pose.position, msg.pose.orientation
        self.latest_ee_pose = np.array(
            [p.x, p.y, p.z, o.x, o.y, o.z, o.w], dtype=np.float32
        )

    def _cb_image(self, msg: Image):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.latest_frame = frame.copy()

    # ── Master tick (20 Hz) ────────────────────────────────────────────────

    def _tick(self):
        if not self.recording or self.latest_positions is None:
            return

        n_frames = len(self._buf)

        if self.failure_time is not None:
            elapsed   = time.monotonic() - self.failure_time
            remaining = FAILURE_EXTRA - elapsed
            if remaining <= 0:
                self._save_trajectory(outcome="failure")
                return
            label = 0.0
            n_bad  = sum(1 for r in self._buf if r["label"] == 0.0)
            print_live(n_frames, n_frames - n_bad, n_bad,
                       self.latest_positions, failure_countdown=remaining)
        elif self.bad_mode:
            label = 0.0
            n_bad = n_frames  # all frames so far are bad
            print_live(n_frames, 0, n_bad, self.latest_positions, bad_mode=True)
        else:
            label = 1.0
            print_live(n_frames, n_frames, 0, self.latest_positions)

        timestamp = time.monotonic() - self._ep_start

        pos = self.latest_positions
        cmd = self.latest_commands  if self.latest_commands  is not None else np.zeros(6,  dtype=np.float32)
        ee  = self.latest_ee_pose   if self.latest_ee_pose   is not None else np.zeros(7,  dtype=np.float32)
        img = self.latest_frame     if self.latest_frame     is not None else np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)

        row = {
            "timestamp":   timestamp,
            "frame_index": n_frames,
            # joint positions
            **{f"observation.state_j{i+1}": float(pos[i]) for i in range(6)},
            # joint commands
            **{f"action_j{i+1}": float(cmd[i]) for i in range(6)},
            # EE pose
            "ee_x":  float(ee[0]), "ee_y":  float(ee[1]), "ee_z":  float(ee[2]),
            "ee_qx": float(ee[3]), "ee_qy": float(ee[4]), "ee_qz": float(ee[5]), "ee_qw": float(ee[6]),
            # label
            "label": label,
            # raw image (stored separately as video; kept here for stats only)
            "_frame": img,
        }

        self._buf.append(row)

    # ── Recording controls ─────────────────────────────────────────────────

    def start_recording(self):
        if self.recording:
            return
        self._buf         = []
        self.bad_mode     = False
        self.failure_time = None
        self._ep_start    = time.monotonic()
        self.recording    = True
        cprint(
            f"\n{GREEN}{BOLD}▶  Recording episode {self.traj_num}{RESET}  "
            f"{DIM}'g' = success   'f' = failure{RESET}"
        )

    def start_recording_bad(self):
        if self.recording:
            return
        self._buf         = []
        self.bad_mode     = True
        self.failure_time = None
        self._ep_start    = time.monotonic()
        self.recording    = True
        cprint(
            f"\n{RED}{BOLD}▶  Recording episode {self.traj_num} [BAD STATE]{RESET}  "
            f"{DIM}'d' = stop and save{RESET}"
        )

    def stop_bad_recording(self):
        if not self.recording or not self.bad_mode:
            return
        self._save_trajectory(outcome="failure")

    def flag_failure(self):
        if not self.recording or self.failure_time is not None:
            return
        self.failure_time = time.monotonic()
        cprint(
            f"\n{RED}{BOLD}✗  Failure flagged!{RESET}"
            f"{RED}  Recording {FAILURE_EXTRA:.0f}s more …{RESET}"
        )

    def flag_success(self):
        if not self.recording or self.failure_time is not None:
            return
        self._save_trajectory(outcome="success")

    def _save_trajectory(self, outcome: str):
        self.recording = False
        print()  # newline after live display

        if not self._buf:
            cprint(f"{YELLOW}⚠  Nothing recorded — skipping save.{RESET}")
            self._next_traj()
            return

        ep_id   = self.traj_num
        ep_name = f"episode_{ep_id:03d}"
        n_frames = len(self._buf)

        cprint(f"{CYAN}  Saving {ep_name} ({n_frames} frames) …{RESET}")

        # ── 1. Write video ─────────────────────────────────────────────────
        video_path = self.video_dir / f"{ep_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(video_path), fourcc, RECORD_FPS, (CAM_WIDTH, CAM_HEIGHT))
        for row in self._buf:
            frame_rgb = row["_frame"]
            vw.write(frame_rgb)
        vw.release()

        # ── 2. Build parquet (scalar columns only; no raw image bytes) ─────
        scalar_rows = []
        for row in self._buf:
            scalar = {k: v for k, v in row.items() if k != "_frame"}
            scalar_rows.append(scalar)
            self.stats.update_frame(scalar)

        table = pa.Table.from_pylist(scalar_rows)
        parquet_path = self.data_dir / f"{ep_name}.parquet"
        pq.write_table(table, parquet_path, compression="snappy")

        # ── 3. Update stats on disk ────────────────────────────────────────
        self.stats.save()

        # ── 4. Summary ────────────────────────────────────────────────────
        labels    = np.array([r["label"] for r in self._buf], dtype=np.float32)
        n_good    = int(np.sum(labels == 1.0))
        n_bad     = int(np.sum(labels == 0.0))
        duration  = n_frames / RECORD_FPS

        if outcome == "success":
            icon = f"{GREEN}{BOLD}✓  SUCCESS{RESET}"
        elif outcome == "failure":
            icon = f"{RED}{BOLD}✗  FAILURE{RESET}"
        else:
            icon = f"{YELLOW}{BOLD}⚠  {outcome.upper()}{RESET}"

        cprint(
            f"{icon}  {ep_name} saved  —  "
            f"{GREEN}{n_good} good{RESET} / {RED}{n_bad} bad{RESET}  "
            f"({duration:.1f}s)  →  {DIM}{self.root}{RESET}"
        )

        self._buf = []
        self._next_traj()

    def _next_traj(self):
        self.traj_num    += 1
        self.bad_mode     = False
        self.failure_time = None
        self._print_status()

    def _print_status(self):
        cprint(
            f"\n{BOLD}Ready for episode_{self.traj_num:03d}{RESET}  "
            f"{DIM}|  's' start (good)   'b' start (bad)   'g' success   'f' failure   'd' stop bad   'q' quit{RESET}"
        )

    def shutdown(self):
        if self.recording and self._buf:
            self._save_trajectory(outcome="interrupted")
        cprint(f"{GREEN}Recorder shut down.  Dataset at {self.root}{RESET}")


# ── Keyboard thread ────────────────────────────────────────────────────────

def keyboard_thread(recorder: SingleArmRecorder):
    while rclpy.ok():
        key = get_keypress()
        if key == "s":
            recorder.start_recording()
        elif key == "b":
            recorder.start_recording_bad()
        elif key == "g":
            recorder.flag_success()
        elif key == "f":
            recorder.flag_failure()
        elif key == "d":
            recorder.stop_bad_recording()
        elif key == "q":
            recorder.shutdown()
            rclpy.shutdown()
            break


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    experiment_id = input(f"{BOLD}Enter experiment ID (e.g. exp_01): {RESET}").strip()
    task_name = (
        input(f"{BOLD}Task description [{TASK_NAME}]: {RESET}").strip() or TASK_NAME
    )

    rclpy.init()
    recorder = SingleArmRecorder(experiment_id, task_name=task_name)
    kb = threading.Thread(target=keyboard_thread, args=(recorder,), daemon=True)
    kb.start()
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        recorder.shutdown()
    finally:
        recorder.destroy_node()


if __name__ == "__main__":
    main()

