"""
UR5 X-axis oscillation script.

Makes the UR5 real robot's end-effector oscillate between +0.05 and -0.05 meters
in the x-axis forever.

Usage:
  python -m praxis.ros.examples.ur5_oscillate_x
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from dataclasses import dataclass


@dataclass
class UR5OscillateConfig:
    namespace: str = 'ur5real'
    rate_hz: float = 50.0
    x_amplitude: float = 0.1  # oscillate between +0.05 and -0.05
    oscillation_period: float = 4.0  # seconds for one complete cycle
    ee_pose_sub_topic: str = 'ee_pose'
    ee_command_pub_topic: str = 'ee_command'


class UR5Oscillate(Node):
    """UR5 X-axis oscillation node."""

    def __init__(self, config: UR5OscillateConfig):
        super().__init__('ur5_oscillate_x')
        self.config: UR5OscillateConfig = config
        self.initial_pose: np.ndarray | None = None
        self.current_pose: np.ndarray | None = None
        self.time_elapsed: float = 0.0
        
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)

        # Subscribe to EE pose published by robot
        self.ee_pose_sub = self.create_subscription(
            PoseStamped,
            f'/{self.config.namespace}/{self.config.ee_pose_sub_topic}',
            self._ee_pose_callback,
            qos
        )

        # Publish target EE pose
        self.ee_command_pub = self.create_publisher(
            PoseStamped, 
            f'/{self.config.namespace}/{self.config.ee_command_pub_topic}', 
            qos
        )

        # Timer
        self.dt = 1.0 / self.config.rate_hz
        self.create_timer(self.dt, self._tick)

        self.get_logger().info(
            f"Starting UR5 X-axis oscillation: ±{self.config.x_amplitude}m, "
            f"period={self.config.oscillation_period}s"
        )
        self.get_logger().info(
            f"Publishing to '{self.ee_command_pub.topic}' at {self.config.rate_hz:.1f} Hz"
        )

    def _ee_pose_callback(self, msg: PoseStamped) -> None:
        """Sync current EE pose from robot."""
        pose = np.array([
            msg.pose.position.x, 
            msg.pose.position.y, 
            msg.pose.position.z,
            msg.pose.orientation.w, 
            msg.pose.orientation.x, 
            msg.pose.orientation.y, 
            msg.pose.orientation.z
        ])
        
        # Store initial pose on first callback
        if self.initial_pose is None:
            self.initial_pose = pose.copy()
            self.get_logger().info(
                f"Initial pose captured: x={pose[0]:.4f}, y={pose[1]:.4f}, z={pose[2]:.4f}"
            )
        
        self.current_pose = pose

    def _tick(self) -> None:
        """Tick function to compute and publish oscillating EE pose."""
        
        # Skip if initial pose is not captured yet
        if self.initial_pose is None:
            self.get_logger().warn("Waiting for initial EE pose...", throttle_duration_sec=1.0)
            return
        
        # Compute oscillation using sine wave
        # x_offset = A * sin(2π * t / T)
        # where A = amplitude, t = time, T = period
        angular_freq = 2.0 * np.pi / self.config.oscillation_period
        x_offset = self.config.x_amplitude * np.sin(angular_freq * self.time_elapsed)
        
        # Create target pose with oscillating x position
        target_pose = self.initial_pose.copy()
        target_pose[0] = self.initial_pose[0] + x_offset
        
        # Publish EE Pose as PoseStamped
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"{self.config.namespace}_base"
        msg.pose.position.x = float(target_pose[0])
        msg.pose.position.y = float(target_pose[1])
        msg.pose.position.z = float(target_pose[2])
        msg.pose.orientation.w = float(target_pose[3])
        msg.pose.orientation.x = float(target_pose[4])
        msg.pose.orientation.y = float(target_pose[5])
        msg.pose.orientation.z = float(target_pose[6])
        self.ee_command_pub.publish(msg)
        
        # Update time
        self.time_elapsed += self.dt
        
        # Log progress periodically
        if int(self.time_elapsed * 10) % 10 == 0:  # Every second
            self.get_logger().info(
                f"Oscillating: x_offset={x_offset:.4f}m, "
                f"target_x={target_pose[0]:.4f}m"
            )


def main():
    rclpy.init()
    try:
        node = UR5Oscillate(UR5OscillateConfig())
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down oscillation...")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

