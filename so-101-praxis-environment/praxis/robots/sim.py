from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar, get_args
import numpy as np
from dataclasses import dataclass, field
import numpy as np
import time
import mujoco
import os
from mujoco import viewer as mj_viewer
from mujoco import MjData, MjModel, MjSpec
from praxis.robots.base import ControlMode
from praxis import MODELS_DIR

@dataclass
class RobotSimConfig:
    xml_path: str
    actuator_names: np.ndarray 
    joint_names: np.ndarray
    body_names: np.ndarray
    ee_site_name: str 

    name: str = "base_sim_robot"
    dof: int = 7

    init_joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(7))
    joint_limits_upper: np.ndarray = field(default_factory=lambda: np.inf * np.ones(7))
    joint_limits_lower: np.ndarray = field(default_factory=lambda: -np.inf * np.ones(7))

    sim_dt: float = 0.002
    gravity_compensation: bool = True
    enable_viewer: bool = True
    viewer_fps: float = 30.0

    # IK parameters
    integration_dt: float = 1.0
    damping: float = 1e-4
    max_angvel: float = 0.0

T_RobotConfig = TypeVar("T_RobotConfig", bound=RobotSimConfig)

class RobotSim(ABC, Generic[T_RobotConfig]):
    '''Simulation robot class.'''

    def __init__(self, config: T_RobotConfig):
        self.config = config
        self._spec = MjSpec.from_file(os.path.join(MODELS_DIR, self.xml_path))

        if self.config.gravity_compensation:
            for name in self.config.body_names:
                self._spec.body(name).gravcomp = 1.0
            print(f"Applied gravity compensation to bodies: {self.body_names}")

        self._model = self._spec.compile()
        self._model.opt.timestep = self.config.sim_dt

        self._ee_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, self.config.ee_site_name)
        assert self._ee_site_id >= 0, "End-effector site not found"
        self._dof_ids = np.array([self._model.joint(name).id for name in self.joint_names])
        self._actuator_ids = np.array([self._model.actuator(name).id for name in self.actuator_names])

        self._data = MjData(self._model)

        # Initialize joint positions and targets
        self._data.qpos[:self.dof] = np.array(self.init_joint_positions)
        self._data.ctrl[:] = np.array(self.init_joint_positions)
        mujoco.mj_forward(self._model, self._data)
        self._step_count = 0
        self._connected = False
        self._enabled = False
        self._estopped = False
        self._control_mode: ControlMode | None = None
        self._ee_pose_target = self.get_ee_pose()  # use the initial pose as the target
    
        # Launch viewer if enabled
        if self.config.enable_viewer:
            self._launch_viewer()

        # Pre-allocate numpy arrays for IK
        self._jac = np.zeros((6, self._model.nv))
        self._diag = self.config.damping * np.eye(6)
        self._error = np.zeros(6)
        self._error_pos = self._error[:3]
        self._error_ori = self._error[3:]
        self._site_quat = np.zeros(4)
        self._site_quat_conj = np.zeros(4)
        self._error_quat = np.zeros(4)

    def simulation_step(self) -> None:
        """Perform one simulation step. Call this from a ROS timer."""
        # Run IK if in EE control
        # if self._control_mode == "ee_pose":
        #     self._ik_step()
        
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        # Sync viewer if running
        if self._viewer is not None and self._step_count % int(1 / self.config.viewer_fps / self.sim_dt) == 0:
            try:
                self._viewer.sync()
            except Exception:
                # Viewer might have been closed
                self._viewer = None

    def connect(self) -> bool:
        """Simulate connecting to the robot."""
        if self._connected:
            return True
        time.sleep(1.0)
        self._connected = True
        return True

    def enable(self) -> bool:
        """Simulate enabling the robot."""
        if self._estopped:
            return False
        time.sleep(1.0)
        self._enabled = True
        return True

    def set_control_mode(self, mode: ControlMode) -> bool:
        if mode not in get_args(ControlMode):
            return False
        self._control_mode = mode
        return True

    def estop(self) -> bool:
        """Simulate emergency stopping the robot."""
        self._estopped = True
        self._enabled = False
        return True
    
    def clear_estop(self) -> bool:
        """Simulate clearing the emergency stop."""
        time.sleep(1.0)
        self._estopped = False
        return True
    
    def _launch_viewer(self) -> None:
        """Launch MuJoCo viewer."""
        try:
            self._viewer = mj_viewer.launch_passive(self._model, self._data)
            print(f"{self.config.name}: Viewer launched")
        except Exception as e:
            print(f"Viewer launch error: {e}")
            self._viewer = None
        
    def shutdown(self) -> None:
        """Simulate shutting down the robot."""
        self._enabled = False
        self._connected = False
        self._viewer.close()
        self._viewer = None
    
    def get_joint_positions(self) -> np.ndarray:
        return np.array(self._data.qpos[: self.dof])
    
    def set_joint_target(self, q_desired: np.ndarray) -> None:
        self._data.ctrl[:self.config.dof] = q_desired 
        # Simulation step will handle stepping
    
    def get_ee_pose(self) -> np.ndarray:
        pose = np.zeros(7, dtype=np.float64)
        pose[:3] = self._data.site(self._ee_site_id).xpos
        mat = self._data.site(self._ee_site_id).xmat
        mujoco.mju_mat2Quat(pose[3:], mat)

        return pose
    
    def set_ee_target(self, pose: np.ndarray) -> None:
        assert pose.shape == (7,), "EE target pose should be 7D"
        self._ee_pose_target = pose.copy()

    def _ik_step(self) -> None:
        # Position error.
        self._error_pos[:] = self._ee_pose_target[:3] - self._data.site(self._ee_site_id).xpos

        # Orientation error.
        mujoco.mju_mat2Quat(self._site_quat, self._data.site(self._ee_site_id).xmat)
        mujoco.mju_negQuat(self._site_quat_conj, self._site_quat)
        mujoco.mju_mulQuat(self._error_quat, self._ee_pose_target[3:], self._site_quat_conj)
        mujoco.mju_quat2Vel(self._error_ori, self._error_quat, 1.0)

        jac = self._jac
        diag = self._diag

        # Get the Jacobian with respect to the end-effector site.
        mujoco.mj_jacSite(self._model, self._data, jac[:3], jac[3:], self._ee_site_id)

        # Solve system of equations: J @ dq = error.
        dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, self._error)

        # Scale down joint velocities if they exceed maximum.
        if self.config.max_angvel > 0:
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > self.config.max_angvel:
                dq *= self.config.max_angvel / dq_abs_max

        # Integrate joint velocities to obtain joint positions.
        q = self._data.qpos.copy()
        mujoco.mj_integratePos(self._model, q, dq, self.config.integration_dt)

        # Set the control signal.
        np.clip(q, self.config.joint_limits_lower, self.config.joint_limits_upper, out=q)
        self._data.ctrl[self._actuator_ids] = q[self._dof_ids]
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def dof(self) -> int:
        return self.config.dof
    
    @property
    def xml_path(self) -> str:
        return self.config.xml_path
    
    @property
    def actuator_names(self) -> list:
        return self.config.actuator_names
    
    @property
    def body_names(self) -> list:
        return self.config.body_names
    
    @property
    def init_joint_positions(self) -> list:
        return self.config.init_joint_positions
    
    @property
    def joint_names(self) -> list:
        return self.config.joint_names
    
    @property
    def sim_dt(self) -> float:
        return self.config.sim_dt
