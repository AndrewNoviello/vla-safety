from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Literal
import numpy as np


ControlMode = Literal["joint_pos", "ee_pose"]


@dataclass
class RobotConfig:
    """Base robot configuration dataclass."""
    name: str = "base_robot"
    dof: int = 7
    init_joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(7))
    joint_limits_upper: np.ndarray = field(default_factory=lambda: np.inf * np.ones(7))
    joint_limits_lower: np.ndarray = field(default_factory=lambda: -np.inf * np.ones(7))

T_RobotConfig = TypeVar("T_RobotConfig", bound=RobotConfig)


class BaseRobot(ABC, Generic[T_RobotConfig]):
    """Base robot class."""

    def __init__(self, config: T_RobotConfig):
        """Initialize the robot."""
        self.config = config
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the robot."""
        pass

    @abstractmethod
    def enable(self) -> bool:
        """Enable / arm the robot for control."""
        pass

    @abstractmethod
    def set_control_mode(self, mode: ControlMode) -> bool:
        """Set the control mode of the robot."""
        pass

    @abstractmethod
    def estop(self) -> bool:
        """Emergency stop the robot."""
        pass

    @abstractmethod
    def clear_estop(self) -> bool:
        """Clear the emergency stop of the robot."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the robot."""
        pass

    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        pass

    @abstractmethod
    def set_joint_target(self, q_desired: np.ndarray) -> None:
        """Set joint target positions."""
        pass

    @abstractmethod
    def get_ee_pose(self) -> np.ndarray:
        """Get end-effector pose."""
        pass

    @abstractmethod
    def set_ee_target(self, pose: np.ndarray) -> None:
        """Set end-effector target pose."""
        pass

    @property
    def name(self) -> str:
        """Get the name of the robot."""
        return self.config.name
    
    @property
    def dof(self) -> int:
        """Get the number of degrees of freedom of the robot."""
        return self.config.dof
    
    @property
    def init_joint_positions(self) -> np.ndarray:
        """Get the initial joint positions of the robot."""
        return self.config.init_joint_positions
    
    @property
    def joint_limits_upper(self) -> np.ndarray:
        """Get the upper joint limits of the robot."""
        return self.config.joint_limits_upper
    
    @property
    def joint_limits_lower(self) -> np.ndarray:
        """Get the lower joint limits of the robot."""
        return self.config.joint_limits_lower

    @classmethod
    def from_config(cls, config: RobotConfig) -> "BaseRobot":
        """Create a robot instance from configuration."""
        return cls(config)
