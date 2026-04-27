from dataclasses import dataclass, field
from typing import Optional, get_args
import numpy as np
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower as SO101Follower
from hardware.robots.base import BaseRobot, ControlMode
from utils.paths import CALIBRATION

FOLLOWER_JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
SO101_DOF = len(FOLLOWER_JOINT_NAMES)

@dataclass
class SO101RealConfig:
    """Base robot configuration dataclass."""
    name: str = "so101follower"
    dof: int = SO101_DOF
    port: str = "/dev/ttyACM1"
    id: str = "follower_v1"
    init_joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(SO101_DOF))
    joint_limits_upper: np.ndarray = field(default_factory=lambda: np.inf * np.ones(SO101_DOF))
    joint_limits_lower: np.ndarray = field(default_factory=lambda: -np.inf * np.ones(SO101_DOF))
    calibration: str = field(
        default_factory=lambda: str(CALIBRATION / "robots" / "so_follower") + "/"
    )

class SO101Real(BaseRobot[SO101RealConfig]):
    """SO101 follower arm backed by LeRobot's hardware driver."""

    def __init__(self, config: SO101RealConfig):
        """Initialize the robot."""
        super().__init__(config)
        follower_config = SO101FollowerConfig(port=self.config.port, id=self.config.id)

        self.follower = SO101Follower(config=follower_config)
        self._connected = False
        self._estopped = False
        self._enabled = True
        self._control_mode: Optional[ControlMode] = None
    
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

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        observation = self.follower.get_observation()
        return np.array(list(observation.values()), dtype=np.float64)

    def set_joint_target(self, q_desired: np.ndarray) -> None:
        """Set joint target positions."""
        action = {}
        for name, act in zip(FOLLOWER_JOINT_NAMES, q_desired):
            action[name] = float(act)
        _ = self.follower.send_action(action)
