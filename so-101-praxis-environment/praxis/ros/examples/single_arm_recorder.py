"""
Single-Arm Recording Script

Subscribes to joint states and joint commands from the so101real robot node.
Uses the robot node's 20hz update rate as the clock.

Save Path Structure:
  /workspace/praxis-core/data/exp_XX/traj_Y/
  - timestamps.npy:        [num_frames]
  - joint_positions.npy:   [num_frames, 6]
  - joint_commands.npy:    [num_frames, 6]
  - ee_poses.npy:          [num_frames, 7]  (x, y, z, qx, qy, qz, qw)
  - metadata.json

Usage:
  python praxis/ros/examples/single_arm_recorder.py

Press 's' to start/stop recording.
Press 'q' to quit.
"""

import os
import sys
import json
import time
import tty
import termios
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from pathlib import Path
from datetime import datetime

DATA_ROOT = Path("/workspace/praxis-core/data")
ROBOT_NS  = "so101real"


def get_keypress():
    """Non-blocking single keypress read."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class SingleArmRecorder(Node):

    def __init__(self, experiment_id: str):
        super().__init__("single_arm_recorder")
        self.experiment_id = experiment_id
        self.traj_num      = 0
        self.recording     = False

        # Latest data buffers
        self.latest_positions = None   # [6]
        self.latest_commands  = None   # [6]
        self.latest_ee_pose   = None   # [7]

        # Recording buffers
        self.buf_timestamps  = []
        self.buf_positions   = []
        self.buf_commands    = []
        self.buf_ee_poses    = []

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            JointState,
            f"{ROBOT_NS}/joint_state",
            self._cb_joint_state,
            qos,
        )
        self.create_subscription(
            Float64MultiArray,
            f"{ROBOT_NS}/joint_command",
            self._cb_joint_command,
            qos,
        )
        self.create_subscription(
            PoseStamped,
            f"{ROBOT_NS}/ee_pose",
            self._cb_ee_pose,
            qos,
        )

        # 20 hz timer — same rate as robot node
        self.create_timer(0.05, self._tick)
        self.get_logger().info(
            f"Recorder ready. Experiment: {experiment_id}. Press 's' to start/stop, 'q' to quit."
        )

    # ── callbacks ──────────────────────────────────────────────────────────

    def _cb_joint_state(self, msg: JointState):
        self.latest_positions = np.array(msg.position, dtype=np.float32)

    def _cb_joint_command(self, msg: Float64MultiArray):
        self.latest_commands = np.array(msg.data, dtype=np.float32)

    def _cb_ee_pose(self, msg: PoseStamped):
        p = msg.pose.position
        o = msg.pose.orientation
        self.latest_ee_pose = np.array(
            [p.x, p.y, p.z, o.x, o.y, o.z, o.w], dtype=np.float32
        )

    # ── timer tick ─────────────────────────────────────────────────────────

    def _tick(self):
        if not self.recording:
            return
        if self.latest_positions is None:
            return

        ts = self.get_clock().now().nanoseconds * 1e-9
        self.buf_timestamps.append(ts)
        self.buf_positions.append(self.latest_positions.copy())
        self.buf_commands.append(
            self.latest_commands.copy()
            if self.latest_commands is not None
            else np.zeros(6, dtype=np.float32)
        )
        self.buf_ee_poses.append(
            self.latest_ee_pose.copy()
            if self.latest_ee_pose is not None
            else np.zeros(7, dtype=np.float32)
        )

    # ── start / stop ───────────────────────────────────────────────────────

    def start_recording(self):
        self.buf_timestamps.clear()
        self.buf_positions.clear()
        self.buf_commands.clear()
        self.buf_ee_poses.clear()
        self.recording = True
        self.get_logger().info(
            f"▶ Recording traj_{self.traj_num} …  (press 's' to stop)"
        )

    def stop_and_save(self):
        self.recording = False
        n = len(self.buf_timestamps)
        if n == 0:
            self.get_logger().warn("Nothing recorded — skipping save.")
            return

        traj_dir = DATA_ROOT / f"exp_{self.experiment_id}" / f"traj_{self.traj_num}"
        traj_dir.mkdir(parents=True, exist_ok=True)

        np.save(traj_dir / "timestamps.npy",      np.array(self.buf_timestamps))
        np.save(traj_dir / "joint_positions.npy",  np.array(self.buf_positions))
        np.save(traj_dir / "joint_commands.npy",   np.array(self.buf_commands))
        np.save(traj_dir / "ee_poses.npy",         np.array(self.buf_ee_poses))

        duration = self.buf_timestamps[-1] - self.buf_timestamps[0] if n > 1 else 0.0
        metadata = {
            "experiment_id": self.experiment_id,
            "traj_num":      self.traj_num,
            "num_frames":    n,
            "duration_s":    round(duration, 3),
            "fps":           round(n / duration, 2) if duration > 0 else 0,
            "saved_at":      datetime.now().isoformat(),
            "joints":        [f"so101real_joint_{i+1}" for i in range(6)],
        }
        with open(traj_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.get_logger().info(
            f"■ Saved {n} frames ({duration:.1f}s) → {traj_dir}"
        )
        self.traj_num += 1


def keyboard_thread(recorder: SingleArmRecorder):
    while rclpy.ok():
        key = get_keypress()
        if key == "s":
            if recorder.recording:
                recorder.stop_and_save()
            else:
                recorder.start_recording()
        elif key == "q":
            if recorder.recording:
                recorder.stop_and_save()
            rclpy.shutdown()
            break


def main():
    experiment_id = input("Enter experiment ID (e.g. 01): ").strip()
    rclpy.init()
    recorder = SingleArmRecorder(experiment_id)
    kb = threading.Thread(target=keyboard_thread, args=(recorder,), daemon=True)
    kb.start()
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        if recorder.recording:
            recorder.stop_and_save()
    finally:
        recorder.destroy_node()


if __name__ == "__main__":
    main()
