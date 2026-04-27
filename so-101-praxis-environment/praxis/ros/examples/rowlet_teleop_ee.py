import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from pathlib import Path
import mujoco
from mujoco import MjData, MjSpec
from praxis import MODELS_DIR
import os

# SO101 MuJoCo model configuration for forward kinematics
SO101_XML_PATH = os.path.join(MODELS_DIR, "so101/mjcf/so101_new_calib.xml")
SO101_EE_SITE_NAME = "gripperframe"
SO101_DOF = 6


class Leader(Node):
    def __init__(self):
        super().__init__(node_name="rowlet_leader_ee")
        
        # Publisher for end-effector commands (8D: [x, y, z, qw, qx, qy, qz, gripper])
        self.pub_ee = self.create_publisher(
            msg_type=Float64MultiArray, 
            topic="rowlet/ee_pose_command", 
            qos_profile=10
        )
        
        # Subscriber to current EE pose
        self.ee_pose_sub = self.create_subscription(
            PoseStamped,
            "rowlet/ee_pose",
            self._ee_pose_callback,
            10
        )
        
        self.clock = Clock()
        
        # Initialize leader device
        leader_config = SO101LeaderConfig(
            id="rowlet_leader", 
            port="/dev/ttyACM0", 
            calibration_dir=Path("/home/Shared/calibration/teleoperators/so101_leader")
        )
        self.leader = SO101Leader(config=leader_config)
        self.leader.connect(calibrate=False)
        
        # Initialize MuJoCo for forward kinematics
        self._init_mujoco_fk()
        
        # Current follower EE pose
        self.current_ee_pose = None
        
        # Timer for publishing commands
        timer_period = 0.025  # 40 Hz
        self.timer = self.create_timer(timer_period, self.pub_callback)
        
        self.get_logger().info("Rowlet EE teleop node initialized")
        
    def _init_mujoco_fk(self):
        """Initialize MuJoCo model for forward kinematics."""
        self._spec = MjSpec.from_file(SO101_XML_PATH)
        self._model = self._spec.compile()
        self._data = MjData(self._model)
        self._ee_site_id = mujoco.mj_name2id(
            self._model, 
            mujoco.mjtObj.mjOBJ_SITE, 
            SO101_EE_SITE_NAME
        )
        assert self._ee_site_id >= 0, f"End-effector site '{SO101_EE_SITE_NAME}' not found"
        
        # Get joint limits from model for percentage conversion
        self._lower_bound = self._model.jnt_range[:SO101_DOF, 0]
        self._upper_bound = self._model.jnt_range[:SO101_DOF, 1]
        
    def _ee_pose_callback(self, msg: PoseStamped):
        """Callback to store current follower EE pose."""
        self.current_ee_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ])
    
    def _percentage_to_radians(self, q_percentage: np.ndarray) -> np.ndarray:
        """Convert joint positions from percentage format to radians."""
        q_rad = np.zeros(SO101_DOF)
        gripper_id = SO101_DOF - 1
        
        for i in range(SO101_DOF - 1):
            q_rad[i] = self._lower_bound[i] + \
                      ((q_percentage[i] + 100) / 200) * \
                      (self._upper_bound[i] - self._lower_bound[i])
        
        q_rad[gripper_id] = self._lower_bound[gripper_id] + \
                           (q_percentage[gripper_id] / 100) * \
                           (self._upper_bound[gripper_id] - self._lower_bound[gripper_id])
        
        return q_rad
    
    def _joints_to_ee_pose(self, q_percentage: np.ndarray) -> np.ndarray:
        """
        Convert leader joint positions (percentage) to EE pose using FK.
        
        Returns:
            8D array: [x, y, z, qw, qx, qy, qz, gripper_percentage]
        """
        # Convert percentage to radians
        q_rad = self._percentage_to_radians(q_percentage)
        
        # Update MuJoCo model
        self._data.qpos[:SO101_DOF] = q_rad
        mujoco.mj_forward(self._model, self._data)
        
        # Get EE pose (8D: position + orientation + gripper)
        pose = np.zeros(8, dtype=np.float64)
        pose[:3] = self._data.site(self._ee_site_id).xpos
        mat = self._data.site(self._ee_site_id).xmat
        mujoco.mju_mat2Quat(pose[3:7], mat)
        
        # Add gripper value (last joint, in percentage format)
        pose[7] = q_percentage[SO101_DOF - 1]

        
        return pose
        
    def pub_callback(self):
        """Publish end-effector command (8D: pose + gripper) based on leader position."""
        try:
            # Get leader action (joint positions in percentage)
            action = self.leader.get_action().values()
            q_percentage = np.array(list(action))
            
            # Convert leader joint positions to EE pose using FK (returns 8D: [x, y, z, qw, qx, qy, qz, gripper])
            leader_ee_pose = self._joints_to_ee_pose(q_percentage)
            
            # Create and publish Float64MultiArray message with 8D EE pose
            msg = Float64MultiArray()
            msg.data = [float(x) for x in leader_ee_pose]
            
            self.pub_ee.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error in pub_callback: {e}")


def main():
    rclpy.init()
    leader = Leader()
    try:
        rclpy.spin(leader)
    except KeyboardInterrupt:
        pass
    finally:
        leader.leader.disconnect()
        leader.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

