import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import numpy as np
from dataclasses import dataclass

from std_msgs.msg import Float64MultiArray, Bool, String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty, SetBool

from praxis.robots.base import BaseRobot, ControlMode


@dataclass
class RobotNodeConfig:
    """Configuration for RobotNode."""
    update_rate: float = 20.0  # Hz
    control_mode: str = "ee_pose"
    joint_state_topic: str = "joint_state"
    joint_command_topic: str = "joint_command"
    ee_pose_topic: str = "ee_pose"
    ee_command_topic: str = "ee_command"
    ee_pose_command_topic: str = "ee_pose_command"
    estop_topic: str = "estop"
    status_topic: str = "robot_status"


class RobotNode(Node):
    """
    ROS robot node that binds a BaseRobot to the ROS graph.
    """
    
    def __init__(
        self,
        robot: BaseRobot,
        config: RobotNodeConfig,
    ):
        """
        Initialize the RobotNode.
        
        Args:
            robot: The robot instance to bind to ROS
            config: Configuration for the ROS node
        """
        self.robot = robot
        self.config = config
        self.node_name = f"{robot.name}_node"
        
        # Initialize ROS node
        super().__init__(self.node_name, namespace=robot.name)
        
        # Initialize robot state
        self._is_connected = False
        self._is_enabled = False
        self._control_mode: ControlMode | None = None
        self._is_estopped = False
        
        # Setup QoS profiles
        self._qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
                
        # Initialize ROS interfaces
        self._setup_publishers()
        self._setup_subscribers()
        self._setup_services()
        self._setup_timers()
        
        # Start up robot
        self._startup_robot(self.config.control_mode)
        
        self.get_logger().info(f"RobotNode {self.node_name} initialized for {robot.name}")
    
    def _setup_publishers(self) -> None:
        """Setup ROS publishers."""
        self._joint_state_pub = self.create_publisher(
            JointState,
            self.config.joint_state_topic,
            self._qos_profile
        )
        
        self._ee_pose_pub = self.create_publisher(
            PoseStamped,
            self.config.ee_pose_topic,
            self._qos_profile
        )
        
        self._status_pub = self.create_publisher(
            String,
            self.config.status_topic,
            self._qos_profile
        )

    def _setup_subscribers(self) -> None:
        """Setup ROS subscribers."""
        # Joint command subscriber
        self._joint_cmd_sub = self.create_subscription(
            Float64MultiArray,
            self.config.joint_command_topic,
            self._joint_command_callback,
            self._qos_profile
        )
        
        # End-effector command subscriber
        self._ee_cmd_sub = self.create_subscription(
            PoseStamped,
            self.config.ee_command_topic,
            self._ee_command_callback,
            self._qos_profile
        )

        # End-effector pose command subscriber
        self._ee_pose_cmd_sub = self.create_subscription(
            Float64MultiArray,
            self.config.ee_pose_command_topic,
            self._ee_pose_command_callback,
            self._qos_profile
        )
        
        # Emergency stop subscriber
        self._estop_sub = self.create_subscription(
            Bool,
            self.config.estop_topic,
            self._estop_callback,
            self._qos_profile
        )

    def _setup_services(self) -> None:
        """Setup ROS services."""
        # Connect service
        self._connect_srv = self.create_service(
            Empty,
            "connect_robot",
            self._connect_service_callback
        )
        
        # Enable service
        self._enable_srv = self.create_service(
            Empty,
            "enable_robot",
            self._enable_service_callback
        )
        
        # Set control mode service
        self._control_mode_srv = self.create_service(
            SetBool,
            "set_control_mode",
            self._control_mode_service_callback
        )
        
        # Clear emergency stop service
        self._clear_estop_srv = self.create_service(
            Empty,
            "clear_estop",
            self._clear_estop_service_callback
        )
        
        # Shutdown robot service
        self._shutdown_srv = self.create_service(
            Empty,
            "shutdown_robot",
            self._shutdown_service_callback
        )
    
    def _setup_timers(self) -> None:
        """Setup ROS timers."""        
        # Main update timer
        self._update_timer = self.create_timer(
            1.0 / self.config.update_rate,
            self._update_callback
        )

        # If in sim, add simulation timer - runs at robot's simulation rate
        if hasattr(self.robot, 'simulation_step'):
            self._sim_timer = self.create_timer(
                self.robot.sim_dt,
                self._simulation_callback
            )
    
    def _startup_robot(self, control_mode: ControlMode) -> None:
        """Connect to, enable, and set control mode of the robot."""
        try:
            self._is_connected = self.robot.connect()
            if self._is_connected:
                self.get_logger().info(f"Connected to {self.robot.name}")
                
                # Enable the robot after connection
                self._is_enabled = self.robot.enable()
                if self._is_enabled:
                    self.get_logger().info(f"Robot enabled automatically")
                    
                    # Set control mode
                    success = self.robot.set_control_mode(control_mode)
                    if success:
                        self._control_mode = control_mode
                        self.get_logger().info(f"Control mode set to {control_mode}")
                    else:
                        self.get_logger().warn(f"Failed to set control mode")
                else:
                    self.get_logger().error("Failed to enable robot")
            else:
                self.get_logger().error(f"Failed to connect to {self.robot.name}")

        except Exception as e:
            self.get_logger().error(f"Error connecting to robot: {e}")
            self._is_connected = False
    
    def _publish_joint_state(self) -> None:
        """Publish joint state."""
        try:
            joint_positions = self.robot.get_joint_positions()
            joint_velocities = self.robot.get_joint_velocities() \
                if hasattr(self.robot, 'get_joint_velocities') else np.zeros(self.robot.dof)
            joint_efforts = self.robot.get_joint_efforts() \
                if hasattr(self.robot, 'get_joint_efforts') else np.zeros(self.robot.dof)
            
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = f"{self.robot.name}_base"
            msg.name = [f"{self.robot.name}_joint_{i+1}" for i in range(self.robot.dof)]
            msg.position = joint_positions.tolist()
            msg.velocity = joint_velocities.tolist()
            msg.effort = joint_efforts.tolist()
            
            self._joint_state_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing joint state: {e}")
    
    def _publish_ee_pose(self) -> None:
        """Publish end-effector pose."""
        try:
            ee_pose = self.robot.get_ee_pose()
            
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = f"{self.robot.name}_base"
            
            # Assuming pose is [x, y, z, qw, qx, qy, qz]
            msg.pose.position.x = float(ee_pose[0])
            msg.pose.position.y = float(ee_pose[1])
            msg.pose.position.z = float(ee_pose[2])
            msg.pose.orientation.w = float(ee_pose[3])
            msg.pose.orientation.x = float(ee_pose[4])
            msg.pose.orientation.y = float(ee_pose[5])
            msg.pose.orientation.z = float(ee_pose[6])
            
            self._ee_pose_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing end-effector pose: {e}")
    
    def _publish_status(self) -> None:
        """Publish robot status."""
        # TODO(Yunhai): use a custom status message instead of string
        try:
            status = {
                "connected": self._is_connected,
                "enabled": self._is_enabled,
                "control_mode": self._control_mode,
                "estop": self._is_estopped,
            }
            
            msg = String()
            msg.data = str(status)
            self._status_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing status: {e}")

    def _update_callback(self) -> None:
        """Main update callback to publish joint states, end-effector pose, and status."""
        if not self._is_connected:
            return
        
        try:
            self._publish_joint_state()
            self._publish_ee_pose()
            self._publish_status()
                
        except Exception as e:
            self.get_logger().error(f"Error in update callback: {e}")
    
    def _simulation_callback(self) -> None:
        """Simulation step callback."""
        if hasattr(self.robot, 'simulation_step'):
            self.robot.simulation_step()
    
    def _joint_command_callback(self, msg: Float64MultiArray) -> None:
        """Handle joint command messages."""
        if not self._is_connected or not self._is_enabled or self._is_estopped:
            self.get_logger().warn(f"Joint command ignored")
            return
        
        try:
            joint_targets = np.array(msg.data)
            self.robot.set_joint_target(joint_targets)

        except Exception as e:
            self.get_logger().error(f"Error handling joint command: {e}")

    def _ee_command_callback(self, msg: PoseStamped) -> None:
        """Handle end-effector command messages."""
        if not self._is_connected or not self._is_enabled or self._is_estopped:
            return
        
        try:
            # Convert PoseStamped to numpy array [x, y, z, qw, qx, qy, qz]
            pose = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ])
            self.robot.set_ee_target(pose)
                
        except Exception as e:
            self.get_logger().error(f"Error handling EE command: {e}")
    
    def _ee_pose_command_callback(self, msg: Float64MultiArray) -> None:
        """Handle end-effector pose command messages. Could be 7D or 8D."""
        if not self._is_connected or not self._is_enabled or self._is_estopped:
            return
        
        try:
            pose = np.array(msg.data)
            if len(pose) == 7:
                # 7D: [x, y, z, qw, qx, qy, qz]
                gripper = self.robot.get_joint_positions()[-1]
                pose = np.concatenate([pose, [gripper]])
                self.robot.set_ee_target(pose)
            elif len(pose) == 8:
                # 8D: [x, y, z, qw, qx, qy, qz, gripper]
                self.robot.set_ee_target(pose)
            else:
                self.get_logger().error(f"Invalid EE pose command dimension: {len(pose)}, expected 7 or 8")
            self.robot.set_ee_target(pose)
                
        except Exception as e:
            self.get_logger().error(f"Error handling EE pose command: {e}")
    
    def _estop_callback(self, msg: Bool) -> None:
        """Handle emergency stop messages."""
        try:
            if msg.data and not self._is_estopped:
                success = self.robot.estop()
                if success:
                    self._is_estopped = True
                    self.get_logger().warn("Emergency stop activated!")
                else:
                    self.get_logger().error("Failed to activate emergency stop")

        except Exception as e:
            self.get_logger().error(f"Error handling estop: {e}")
    
    def _connect_service_callback(self, request, response):
        """Handle connect robot service."""
        try:
            self._is_connected = self.robot.connect()
            if self._is_connected:
                self.get_logger().info("Robot connected via service")
            else:
                self.get_logger().error("Failed to connect to robot")

        except Exception as e:
            self.get_logger().error(f"Error in connect service: {e}")
            self._is_connected = False
        
        return response
    
    def _enable_service_callback(self, request, response):
        """Handle enable robot service."""
        if not self._is_connected:
            self.get_logger().error("Cannot enable robot: not connected")
            return response
        
        try:
            self._is_enabled = self.robot.enable()
            if self._is_enabled:
                self.get_logger().info("Robot enabled via service")
            else:
                self.get_logger().error("Failed to enable robot")

        except Exception as e:
            self.get_logger().error(f"Error in enable service: {e}")
            self._is_enabled = False
        
        return response
    
    def _control_mode_service_callback(self, request, response):
        """Handle set control mode service."""
        if not self._is_connected:
            self.get_logger().error("Cannot set control mode: not connected")
            response.success = False
            response.message = "Robot not connected"
            return response
        
        try:
            # Map boolean to control mode
            mode = "ee_pose" if request.data else "joint_pos"
            
            # TODO(Yunhai): define custom service to handle string control mode in the future
            # mode = request.data  # Direct string from request
            # if mode not in get_args(ControlMode):
            #     error_msg = f"Invalid control mode: {mode}. Valid modes: {get_args(ControlMode)}"
            #     self.get_logger().error(error_msg)
            #     response.success = False
            #     response.message = error_msg
            #     return response
                
            success = self.robot.set_control_mode(mode)
            if success:
                self._control_mode = mode
                self.get_logger().info(f"Control mode set to: {mode}")
                response.success = True
                response.message = f"Control mode set to {mode}"
            else:
                error_msg = f"Failed to set control mode to {mode}. Current control mode: {self._control_mode}"
                self.get_logger().error(error_msg)
                response.success = False
                response.message = error_msg

        except Exception as e:
            self.get_logger().error(f"Error in control mode service: {e}")
            response.success = False
            response.message = f"Error: {e}"
        
        return response
    
    def _clear_estop_service_callback(self, request, response):
        """Handle clear emergency stop service."""
        try:
            success = self.robot.clear_estop()
            if success:
                self._is_estopped = False
                self.get_logger().info("Emergency stop cleared")
            else:
                self.get_logger().error("Failed to clear emergency stop")

        except Exception as e:
            self.get_logger().error(f"Error in clear estop service: {e}")
        
        return response
    
    def _shutdown_service_callback(self, request, response):
        """Handle shutdown robot service."""
        try:
            self.robot.shutdown()
            # Update robot state after shutdown
            self._is_connected = False
            self._is_enabled = False
            self.get_logger().info("Robot shutdown completed")

        except Exception as e:
            self.get_logger().error(f"Error in shutdown service: {e}")
        
        return response


def run_robot_node(robot: BaseRobot, config: RobotNodeConfig) -> None:
    """
    Run a RobotNode with ROS2.
    """

    rclpy.init()
    
    try:
        node = RobotNode(robot, config)
        
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
    
    finally:
        rclpy.shutdown()
