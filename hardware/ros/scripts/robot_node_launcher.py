import sys
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

from hardware.ros.robot_node import run_robot_node


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    print(f"Configuration: {OmegaConf.to_yaml(cfg, resolve=True)}")
    
    try:
        # Create robot from config
        print(f"Creating robot: {cfg.robot._target_}")
        robot = instantiate(cfg.robot)
        
        # Create ROS node configuration
        print(f"Creating ROS node configuration: {cfg.ros_node._target_}")
        ros_config = instantiate(cfg.ros_node)
        
        # Run the RobotNode
        run_robot_node(robot, ros_config)
        
    except KeyboardInterrupt:
        print("\nShutting down...")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
