from dataclasses import dataclass, field
import numpy as np
from praxis.robots.sim import RobotSim, RobotSimConfig

@dataclass
class SO101SimConfig(RobotSimConfig):
    xml_path: str = "so101/mjcf/so101_new_calib.xml"
    actuator_names: np.ndarray = field(default_factory=lambda: np.array(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper" ]))
    joint_names: np.ndarray = field(default_factory=lambda: np.array(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper" ]))
    body_names: np.ndarray = field(default_factory=lambda: np.array(["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"]))
    ee_site_name: str = "gripperframe"

    name: str = "so101sim"
    dof: int = 6

    init_joint_positions: np.ndarray = field(default_factory=lambda: np.array([
        -5.9985, -91.6443, 98.6511, -100.0, -100.0, 2.6418
    ]))
    joint_limits_upper: np.ndarray = field(default_factory=lambda: np.inf * np.ones(7))
    joint_limits_lower: np.ndarray = field(default_factory=lambda: -np.inf * np.ones(7))
    
    sim_dt: float = 0.002
    enable_viewer: bool = True
    viewer_fps: float = 30.0

    # IK parameters
    integration_dt: float = 1.0
    damping: float = 1e-4
    max_angvel: float = 0.0


class SO101Sim(RobotSim[SO101SimConfig]):
    '''Simulation robot class.'''

    def __init__(self, config: SO101SimConfig):
        super().__init__(config)
        self.lower_bound = self._model.jnt_range[:,0]
        self.upper_bound = self._model.jnt_range[:,1]

    def get_joint_positions(self) -> np.ndarray:
        action = np.array(self._data.qpos[: self.dof])

        # Converting from radians to percentage
        q_desired = np.array([((action[i] - self.lower_bound[i])*200)/(self.upper_bound[i] - self.lower_bound[i]) - 100 for i in range(self.dof)])
        gripper_id = self.dof-1
        q_desired[gripper_id] = ((action[[gripper_id]] - self.lower_bound[gripper_id])*100)/(self.upper_bound[gripper_id] - self.lower_bound[gripper_id])
        
        return q_desired
    
    def set_joint_target(self, q_desired: np.ndarray) -> None:
        # Converting from percentage to radians
        action = np.array([self.lower_bound[i] + ((q_desired[i]+100)/200)*(self.upper_bound[i] - self.lower_bound[i]) for i in range(self.dof)])
        gripper_id = self.dof-1
        action[gripper_id] = self.lower_bound[gripper_id] + (q_desired[gripper_id]/100)*(self.upper_bound[gripper_id] - self.lower_bound[gripper_id])
        self._data.ctrl[:self.config.dof] = action
        # Simulation step will handle stepping
    

    