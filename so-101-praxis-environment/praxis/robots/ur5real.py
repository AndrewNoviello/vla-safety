from dataclasses import dataclass, field
from typing import Optional, get_args
import numpy as np
import time
from praxis.robots.base import BaseRobot, RobotConfig, ControlMode
from scipy.spatial.transform import Rotation as R


# UR5e constants
UR5_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

""" Note on Servo vs Move commands:
Move commands are slower but have better trajectory planning, suitable for moving to a distinct pose.
Servo commands are faster but do not have trajectory planning, suitable for moving to a close pose.
"""

@dataclass
class UR5RealConfig(RobotConfig):
    name: str = "ur5real"
    dof: int = 6
    init_joint_positions: np.ndarray = field(default_factory=lambda: np.array([
        1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0
    ]))
    
    # Robot connection parameters
    robot_ip: str = "192.10.0.11"
    
    # Control parameters
    velocity: float = 0.5  # rad/s for joint control
    acceleration: float = 0.5  # rad/s^2 for joint control
    ee_velocity: float = 0.1  # m/s for end-effector control
    ee_acceleration: float = 0.3  # m/s^2 for end-effector control
    
    # Safety parameters
    blend_radius: float = 0.0  # blend radius for waypoint blending
    
    # Joint limits (UR5e standard limits in radians)
    joint_limits_lower: np.ndarray = field(default_factory=lambda: np.array(
        [-2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi]
    ))
    joint_limits_upper: np.ndarray = field(default_factory=lambda: np.array(
        [2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi]
    ))


class UR5Real(BaseRobot[UR5RealConfig]):
    """UR5e real robot implementation using UR RTDE interface."""
    
    def __init__(self, config: UR5RealConfig):
        super().__init__(config)
        
        # Import ur_rtde here to make it an optional dependency
        try:
            import rtde_control
            import rtde_receive
            self._rtde_control_module = rtde_control
            self._rtde_receive_module = rtde_receive
        except ImportError:
            raise ImportError(
                "ur_rtde is required for UR5Real. Install it with: pip install ur-rtde"
            )
        
        # Initialize connection objects
        self._rtde_c = self._rtde_control_module.RTDEControlInterface(self.config.robot_ip)
        self._rtde_r = self._rtde_receive_module.RTDEReceiveInterface(self.config.robot_ip)
        self._connected = self._rtde_r.isConnected();
        if self._connected:
            print(f"UR5Real: Successfully connected to {self.config.robot_ip}")
        else:
            print(f"UR5Real: Failed to establish connection")
        
        # Initialize robot state
        self._connected = False
        self._enabled = False
        self._estopped = False
        self._control_mode: Optional[ControlMode] = None
        
    def connect(self) -> bool:
        """Establish connection to the real UR5e robot."""
        self._connected = self._rtde_r.isConnected();
        return self._connected;
    
    def enable(self) -> bool:
        """Enable the robot for control."""
        
        if self._estopped:
            print("UR5Real: Cannot enable - robot is emergency stopped")
            return False
        
        if self._enabled:
            if self._rtde_r.getRobotMode() != 7 or self._rtde_r.getSafetyMode() != 0:
                self._enabled = False
                return False
            else:
                self._enabled = True
            return True
        
           
        robot_mode = self._rtde_r.getRobotMode()
        safety_status = self._rtde_r.getSafetyMode()
        
        print(f"UR5Real: Robot mode: {robot_mode}, Safety status: {safety_status}")

        if robot_mode != 7 or safety_status != 1: # check if robot is in a safe state
            print("UR5Real: Robot is not in a safe state")
            return False # return false if robot is not in a safe state
        
        self._enabled = True # set robot to enabled state
        return True
            
    
    def set_control_mode(self, mode: ControlMode) -> bool:
        """Set the control mode of the robot."""
        if mode not in get_args(ControlMode):
            print(f"UR5Real: Invalid control mode: {mode}")
            return False
        
        self._control_mode = mode
        print(f"UR5Real: Control mode set to {mode}")
        return True
    
    def estop(self) -> bool:
        """Emergency stop the robot."""
        if not self._connected:
            print("UR5Real: Cannot emergency stop - robot not connected")
            return False
        
        self._rtde_c.stopScript()
        self._estopped = True
        self._enabled = False
        return True
    
    def clear_estop(self) -> bool:
        """Clear the emergency stop of the robot."""
        # Check safety status
        safety_mode = self._rtde_r.getSafetyMode()
        
        if safety_mode in [1, 2, 3]:  # Normal, reduced, protective stop
            self._estopped = False
            return True
        else:
            print(f"UR5Real: Cannot clear emergency stop - safety mode: {safety_mode}")
            print("UR5Real: Please clear the error on the teach pendant")
            return False
       
    
    def shutdown(self) -> None:
        """Shutdown the robot."""
        print("UR5Real: Shutting down...")
        
        self._rtde_c.stopScript()
        self._rtde_c.disconnect()
        self._rtde_r.disconnect()
        self._enabled = False
        self._connected = False
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""

        q = self._rtde_r.getActualQ()
        return np.array(q)
    
    def set_joint_target(self, q_desired: np.ndarray) -> None:
        """Set joint target positions."""
        
        if self._control_mode != "joint_pos":
            print("UR5Real: Warning - not in joint_pos control mode")
        
        # Clip to joint limits
        q_clipped = np.clip(
            q_desired,
            self.config.joint_limits_lower,
            self.config.joint_limits_upper
        )
        
        # Send joint position command
        # servoJ signature: servoJ(q, velocity, acceleration, dt, lookahead_time, gain)
        self._rtde_c.servoJ(
            q_clipped.tolist(),
            self.config.velocity,
            self.config.acceleration,
            self.config.dt,
            self.config.lookahead_time,
            self.config.gain
        )
            
    
    def go_to_joint_positions(self, q_desired: np.ndarray) -> None:
        """Go to joint positions. 
        Please use this function when going to a distinct joint position.
        """

        
        self._rtde_c.moveJ(q_desired.tolist(), self.config.velocity, self.config.acceleration)
        #sleep for 0.1 seconds
        time.sleep(0.1)
    
    def get_ee_pose(self) -> np.ndarray:
        """Get end-effector pose (position + quaternion)."""    

        # Get TCP pose (x, y, z, rx, ry, rz)
        tcp_pose = self._rtde_r.getActualTCPPose()
        
        #convert to quaternion using scipy
        rotvec = tcp_pose[3:] 
        quat = R.from_rotvec(rotvec).as_quat(scalar_first=True)
        return np.concatenate([tcp_pose[:3], quat])

    
    def set_ee_target(self, pose: np.ndarray) -> None:
        """Set end-effector target pose (position + quaternion)."""
        
        if self._control_mode != "ee_pose":
            print("UR5Real: Warning - not in ee_pose control mode")
        
       
        assert pose.shape == (7,), "EE target pose should be 7D (pos + quat)"
        
        # Extract position and quaternion [x, y, z, qw, qx, qy, qz]
        pos = pose[:3]
        qw = pose[3]
        qx = pose[4]
        qy = pose[5]
        qz = pose[6]
        
        # Convert quaternion to axis-angle (rotation vector)
        angle = 2 * np.arccos(np.clip(qw, -1, 1))
        if angle < 1e-10:
            rotvec = np.zeros(3)
        else:
            s = np.sqrt(1 - qw * qw)
            axis = np.array([qx, qy, qz]) / s
            rotvec = axis * angle
        
        # Construct TCP pose for UR [x, y, z, rx, ry, rz]
        tcp_pose = np.concatenate([pos, rotvec])
        
        # Send end-effector pose command
        # servoL signature: servoL(pose, velocity, acceleration, dt, lookahead_time, gain)
        joints = self._rtde_c.getInverseKinematics(tcp_pose)
        self._rtde_c.servoJ(
            joints,
            self.config.velocity,
            self.config.acceleration,
            self.config.dt,
            self.config.lookahead_time,
            self.config.gain
        )
        

    def go_to_ee_pose(self, pose: np.ndarray) -> None:
        """Go to end-effector pose.
        Please use this function when going to a distinct end-effector pose.
        """
        
        self._rtde_c.moveL(pose.tolist(), self.config.ee_velocity, self.config.ee_acceleration)
        #sleep for 0.1 seconds
        time.sleep(0.1)