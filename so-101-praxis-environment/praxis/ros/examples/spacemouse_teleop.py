"""
SpaceMouse end-effector teleoperation node.

Reads 6-DoF input from a 3Dconnexion SpaceMouse and publishes absolute EE pose
commands as geometry_msgs/PoseStamped on the robot's ee_command topic.

Usage:
  python -m praxis.ros.examples.spacemouse_teleop namespace=ur5real
  python -m praxis.ros.examples.spacemouse_teleop namespace=ur5sim
"""

import numpy as np

from praxis.devices.spacemouse import SpaceMouse
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped

from dataclasses import dataclass
import hydra
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R


@dataclass
class SpaceMouseTeleopConfig:
    robot: str = 'ur5sim'
    rate_hz: float = 20.0
    pos_scale: float = 0.05
    rot_scale: float = 0.200
    ee_pose_sub_topic: str = 'ee_pose'
    ee_command_pub_topic: str = 'ee_command'


class SpaceMouseTeleop(Node):
    """SpaceMouse teleoperation node."""

    def __init__(self, config: SpaceMouseTeleopConfig):
        super().__init__('spacemouse_teleop')
        self.config: SpaceMouseTeleopConfig = config
        self.curr_pose: np.ndarray | None = None
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)

        # Spacemouse device
        self.space_mouse = SpaceMouse()
        
        # Subscribe to EE pose published by robot
        self.ee_pose_sub = self.create_subscription(
            PoseStamped,
            f'/{self.config.robot}/{self.config.ee_pose_sub_topic}',
            self._ee_pose_callback,
            qos
        )

        # Publish target EE pose
        self.ee_command_pub = self.create_publisher(
            PoseStamped, 
            f'/{self.config.robot}/{self.config.ee_command_pub_topic}', 
            qos
        )

        # Timer
        self.create_timer(1.0 / self.config.rate_hz, self._tick)

        self.get_logger().info(
            f"Publishing target EE pose to '{self.ee_command_pub.topic}' at {self.config.rate_hz:.1f} Hz"
        )

    def _ee_pose_callback(self, msg: PoseStamped) -> None:
        """Sync current EE pose from robot."""
        self.curr_pose = np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
            msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z
        ])

    def _tick(self) -> None:
        """Tick function to read SpaceMouse state and publish EE pose."""
            
        try:
            data = self.space_mouse.get_state()
        except Exception as e:
            self.get_logger().error(f"SpaceMouse read error: {e}")
            return
        
        # Skip if current pose is not synced
        if self.curr_pose is None:
            return

        delta_pos = np.array([data.x, data.y, data.z]) * self.config.pos_scale
        delta_rot = np.array([data.roll, data.pitch, data.yaw]) * self.config.rot_scale
        if np.allclose(delta_rot, np.zeros(3), atol=1e-6):
            target_quat = self.curr_pose[3:]
        else:
            delta_quat = R.from_euler('xyz', delta_rot)
            original_quat = R.from_quat(self.curr_pose[3:], scalar_first=True)
            target_quat = delta_quat*original_quat
            target_quat = target_quat.as_quat(scalar_first=True)


        target_pos = self.curr_pose[:3] + delta_pos

        # Publish EE Pose as PoseStamped
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "ee_pose_target" # TODO
        msg.pose.position.x = float(target_pos[0])
        msg.pose.position.y = float(target_pos[1])
        msg.pose.position.z = float(target_pos[2])
        msg.pose.orientation.w = float(target_quat[0])
        msg.pose.orientation.x = float(target_quat[1])
        msg.pose.orientation.y = float(target_quat[2])
        msg.pose.orientation.z = float(target_quat[3])
        self.ee_command_pub.publish(msg)


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig = None):
    # Convert DictConfig to SpaceMouseTeleopConfig
    if cfg is None:
        config = SpaceMouseTeleopConfig()
    else:
        config = SpaceMouseTeleopConfig(**OmegaConf.to_container(cfg))
    
    rclpy.init()
    try:
        node = SpaceMouseTeleop(config)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    # Register the config with Hydra
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore.instance()
    cs.store(name="config", node=SpaceMouseTeleopConfig)
    
    main()
