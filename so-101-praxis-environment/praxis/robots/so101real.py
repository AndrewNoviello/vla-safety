from dataclasses import dataclass, field
from typing import Optional, get_args
from pathlib import Path
import os
import numpy as np
import mujoco
from mujoco import MjData, MjSpec
from mujoco import viewer as mj_viewer
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower as SO101Follower
from praxis.robots.base import BaseRobot, ControlMode
from praxis import MODELS_DIR

FOLLOWER_JOINT_NAMES = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos']

# SO101 MuJoCo model configuration
SO101_XML_PATH = os.path.join(MODELS_DIR, "so101/mjcf/so101_new_calib.xml")
SO101_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SO101_EE_SITE_NAME = "gripperframe"
SO101_DOF = 6

@dataclass
class SO101RealConfig:
    """Base robot configuration dataclass."""
    name: str = "so101follower"
    port: str = "/dev/ttyACM1"
    id: int = "follower_v1"
    init_joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(7))
    joint_limits_upper: np.ndarray = field(default_factory=lambda: np.inf * np.ones(7))
    joint_limits_lower: np.ndarray = field(default_factory=lambda: -np.inf * np.ones(7))
    calibration: str = "/home/Shared/calibration/robots/so101_follower/"
    enable_viewer: bool = False
    viewer_fps: float = 30.0

class SO101Real(BaseRobot[SO101RealConfig]):
    """Base robot class."""

    def __init__(self, config: SO101RealConfig):
        """Initialize the robot."""
        super().__init__(config)
        config = SO101FollowerConfig(port=self.config.port, id=self.config.id)

        self.follower = SO101Follower(config=config)
        self._connected = False
        self._estopped = False
        self._enabled = True
        self._control_mode: Optional[ControlMode] = None
        self._viewer = None
        self._viewer_sync_counter = 0
        
        # Initialize MuJoCo simulation for FK/IK
        self._init_mujoco()
    
    def connect(self) -> bool:
        """Establish connection to the robot."""
        self.follower.connect(calibrate=False)
        self._connected = self.follower.is_connected
        return self._connected

    def enable(self) -> bool:
        """Enable / arm the robot for control."""
        if not self._connected:
            print("SO101Follower: Cannot enable - robot not connected")
            return False
        
        if self._estopped:
            print("SO101Follower: Cannot enable - robot is emergency stopped")
            return False
        
        if self._enabled:
            print("SO101Follower: Robot already enabled")
            return True
        
        self._enabled = True
        print("SO101Follower: Robot enabled")
        return True

    def set_control_mode(self, mode: ControlMode) -> bool:
        """Set the control mode of the robot."""
        if mode not in get_args(ControlMode):
            print(f"SO101Follower: Invalid control mode: {mode}")
            return False
        
        self._control_mode = mode
        print(f"SO101Follower: Control mode set to {mode}")
        return True

    def estop(self) -> bool:
        """Emergency stop the robot."""
        if not self._connected:
            print("SO101Follower: Cannot emergency stop - robot not connected")
            return False
        
        try:
            self.follower.disconnect()
            self._estopped = True
            self._enabled = False
            print("SO101Follower: Emergency stop activated")
            return True
        except Exception as e:
            print(f"SO101Follower: Emergency stop error: {e}")
            return False

    def clear_estop(self) -> bool:
        """Clear the emergency stop of the robot."""
        if not self._connected:
            print("SO101Follower: Cannot clear emergency stop - robot not connected")
            return False
        
        self._estopped = False
        print("SO101Follower: Emergency stop cleared")
        return True
    
    def shutdown(self) -> None:
        """Shutdown the robot."""
        self.follower.disconnect()
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        observation = self.follower.get_observation()
        q_percentage = np.array(observation.values())
        q_percentage_list = list(observation.values())
        # print(f"SO101Real: Joint positions: {q_percentage_list}")
        # print(f"SO101Real: Joint positions: {q_percentage}")
        # Update MuJoCo simulation with real robot joint positions
        self._update_mujoco_from_real(q_percentage_list)
        
        return q_percentage

    def set_joint_target(self, q_desired: np.ndarray) -> None:
        """Set joint target positions."""
        action = {}
        for name, act in zip(FOLLOWER_JOINT_NAMES, q_desired):
            action[name] = act.item()
        _ = self.follower.send_action(action)

    def get_ee_pose(self) -> np.ndarray:
        '''Get end-effector pose from MuJoCo simulation.'''
        # Ensure MuJoCo is up to date with real robot
        gripper_value = 0.0
        if self._connected:
            observation = self.follower.get_observation()
            q_percentage = np.array(observation.values())
            q_percentage_list = list(observation.values())
            # print(f"SO101Real: Joint positions: {q_percentage_list}")
            self._update_mujoco_from_real(q_percentage_list)
            gripper_value = q_percentage_list[-1]
        # print(f"SO101Real: EE pose: {self._data.site(self._ee_site_id).xpos}")
        # Get pose from MuJoCo
        pose = np.zeros(8, dtype=np.float64)
        pose[:3] = self._data.site(self._ee_site_id).xpos
        mat = self._data.site(self._ee_site_id).xmat
        # Use pose[3:7] to get exactly 4 elements for quaternion (not pose[3:] which would give 5)
        mujoco.mju_mat2Quat(pose[3:7], mat)
        # Add gripper value (last joint, in percentage format)
        pose[7] = gripper_value
        # print(f"SO101Real: EE pose: {pose}")
        return pose

    def set_ee_target(self, pose: np.ndarray) -> None:
        '''Set end-effector target pose using IK.'''
        assert pose.shape == (8,), "EE target pose should be 7D [x, y, z, qw, qx, qy, qz, gripper]"
        
        # Get current joint positions from real robot
        if self._connected:
            observation = self.follower.get_observation()
            q_init_percentage = np.array(observation.values())
            q_init_percentage_list = list(observation.values())
            self._update_mujoco_from_real(q_init_percentage_list)
        else:
            q_init_percentage = None
        # Solve IK using MuJoCo
        q_target_percentage, success = self._solve_ik(pose[:-1], q_init_percentage_list)
        q_target_percentage[-1] = pose[-1]
        if success:
            # Send joint command to real robot
            self.set_joint_target(q_target_percentage)
        else:
            print("SO101Real: IK did not converge, using best guess")
            self.set_joint_target(q_target_percentage)
    
    def _init_mujoco(self) -> None:
        """Initialize MuJoCo simulation for FK/IK."""
        # Load MuJoCo model
        self._spec = MjSpec.from_file(SO101_XML_PATH)
        self._model = self._spec.compile()
        self._data = MjData(self._model)
        
        # Get end-effector site ID
        self._ee_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, SO101_EE_SITE_NAME)
        assert self._ee_site_id >= 0, f"End-effector site '{SO101_EE_SITE_NAME}' not found"
        
        # Get joint limits for percentage conversion
        self._lower_bound = self._model.jnt_range[:SO101_DOF, 0]
        self._upper_bound = self._model.jnt_range[:SO101_DOF, 1]
        
        # Pre-allocate arrays for IK
        self._jac = np.zeros((6, self._model.nv))
        self._damping = 1e-4
        self._diag = self._damping * np.eye(6)
        self._error = np.zeros(6)
        self._error_pos = self._error[:3]
        self._error_ori = self._error[3:]
        self._site_quat = np.zeros(4)
        self._site_quat_conj = np.zeros(4)
        self._error_quat = np.zeros(4)
        
        # Initialize with default joint positions
        mujoco.mj_forward(self._model, self._data)
        
        # Launch viewer if enabled (use getattr for backward compatibility)
        enable_viewer = getattr(self.config, 'enable_viewer', False)
        if enable_viewer:
            self._launch_viewer()
    
    def _percentage_to_radians(self, q_percentage: np.ndarray) -> np.ndarray:
        """Convert joint positions from percentage format to radians.
        
        Args:
            q_percentage: Joint positions in percentage format
            
        Returns:
            Joint positions in radians
        """
        q_rad = np.zeros(SO101_DOF)
        gripper_id = SO101_DOF - 1
        
        for i in range(SO101_DOF - 1):
            q_rad[i] = self._lower_bound[i] + ((q_percentage[i] + 100) / 200) * (self._upper_bound[i] - self._lower_bound[i])
        
        q_rad[gripper_id] = self._lower_bound[gripper_id] + (q_percentage[gripper_id] / 100) * (self._upper_bound[gripper_id] - self._lower_bound[gripper_id])
        
        return q_rad
    
    def _radians_to_percentage(self, q_rad: np.ndarray) -> np.ndarray:
        """Convert joint positions from radians to percentage format.
        
        Args:
            q_rad: Joint positions in radians
            
        Returns:
            Joint positions in percentage format
        """
        q_percentage = np.zeros(SO101_DOF)
        gripper_id = SO101_DOF - 1
        
        for i in range(SO101_DOF - 1):
            q_percentage[i] = ((q_rad[i] - self._lower_bound[i]) * 200) / (self._upper_bound[i] - self._lower_bound[i]) - 100
        
        q_percentage[gripper_id] = ((q_rad[gripper_id] - self._lower_bound[gripper_id]) * 100) / (self._upper_bound[gripper_id] - self._lower_bound[gripper_id])
        
        return q_percentage
    
    def _update_mujoco_from_real(self, q_percentage) -> None:
        """Update MuJoCo simulation with real robot joint positions.
        
        Args:
            q_percentage: Joint positions in percentage format from real robot (list or array)
        """
        # Ensure q_percentage is a numpy array
        if not isinstance(q_percentage, np.ndarray):
            q_percentage = np.array(q_percentage, dtype=np.float64)
        
        # Convert to radians and update MuJoCo
        q_rad = self._percentage_to_radians(q_percentage)
        self._data.qpos[:SO101_DOF] = q_rad
        mujoco.mj_forward(self._model, self._data)
        
        # Sync viewer if running (throttle to viewer_fps)
        if self._viewer is not None:
            self._viewer_sync_counter += 1
            # Assuming update rate is around 20-50 Hz, sync at viewer_fps
            viewer_fps = getattr(self.config, 'viewer_fps', 30.0)
            sync_interval = max(1, int(50.0 / viewer_fps))
            if self._viewer_sync_counter % sync_interval == 0:
                try:
                    self._viewer.sync()
                except Exception:
                    # Viewer might have been closed
                    self._viewer = None
    
    def _solve_ik(
        self,
        target_pose: np.ndarray,
        q_init_percentage: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-4
    ) -> tuple[np.ndarray, bool]:
        """Solve inverse kinematics using MuJoCo.
        
        Args:
            target_pose: Target end-effector pose as [x, y, z, qw, qx, qy, qz]
            q_init_percentage: Initial joint positions in percentage format. If None, uses current.
            max_iterations: Maximum number of IK iterations
            tolerance: Convergence tolerance for position error (meters)
            
        Returns:
            Tuple of (joint_positions_percentage, success)
        """
        # Save current state
        qpos_backup = self._data.qpos.copy()
        
        # Initialize joint positions
        if q_init_percentage is not None:
            q_rad = self._percentage_to_radians(q_init_percentage)
            self._data.qpos[:SO101_DOF] = q_rad
            mujoco.mj_forward(self._model, self._data)
        
        # IK iteration loop
        for iteration in range(max_iterations):
            # Position error
            self._error_pos[:] = target_pose[:3] - self._data.site(self._ee_site_id).xpos
            
            # Orientation error
            mujoco.mju_mat2Quat(self._site_quat, self._data.site(self._ee_site_id).xmat)
            mujoco.mju_negQuat(self._site_quat_conj, self._site_quat)
            mujoco.mju_mulQuat(self._error_quat, target_pose[3:], self._site_quat_conj)
            mujoco.mju_quat2Vel(self._error_ori, self._error_quat, 1.0)
            
            # Check convergence
            pos_error_norm = np.linalg.norm(self._error_pos)
            ori_error_norm = np.linalg.norm(self._error_ori)
            
            if pos_error_norm < tolerance and ori_error_norm < 0.1:  # 0.1 rad ~ 5.7 degrees
                # Save result before restoring state
                q_result_rad = self._data.qpos[:SO101_DOF].copy()
                q_result_percentage = self._radians_to_percentage(q_result_rad)
                
                # Restore state
                self._data.qpos[:] = qpos_backup
                mujoco.mj_forward(self._model, self._data)
                
                return q_result_percentage, True
            
            # Get Jacobian
            mujoco.mj_jacSite(self._model, self._data, self._jac[:3], self._jac[3:], self._ee_site_id)
            
            # Solve: J @ dq = error using damped least squares
            # dq = J^T @ (J @ J^T + lambda*I)^(-1) @ error
            dq = self._jac.T @ np.linalg.solve(self._jac @ self._jac.T + self._diag, self._error)
            
            # Integrate joint velocities
            q = self._data.qpos.copy()
            mujoco.mj_integratePos(self._model, q, dq, 1.0)
            
            # Clip to joint limits
            np.clip(q[:SO101_DOF], self._lower_bound, self._upper_bound, out=q[:SO101_DOF])
            
            # Update joint positions
            self._data.qpos[:SO101_DOF] = q[:SO101_DOF]
            
            # Forward kinematics for next iteration
            mujoco.mj_forward(self._model, self._data)
        
        # IK did not converge - save best guess before restoring
        q_result_rad = self._data.qpos[:SO101_DOF].copy()
        q_result_percentage = self._radians_to_percentage(q_result_rad)
        
        # Restore state
        self._data.qpos[:] = qpos_backup
        mujoco.mj_forward(self._model, self._data)
        
        return q_result_percentage, False
    
    def _launch_viewer(self) -> None:
        """Launch MuJoCo viewer."""
        try:
            self._viewer = mj_viewer.launch_passive(self._model, self._data)
            print(f"SO101Real: MuJoCo viewer launched")
        except Exception as e:
            print(f"SO101Real: Viewer launch error: {e}")
            self._viewer = None
