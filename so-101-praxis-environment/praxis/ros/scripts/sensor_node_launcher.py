"""
Sensor Node Launcher

Launch a ROS2 sensor node with Hydra configuration.

Usage:
  python -m praxis.ros.scripts.sensor_node_launcher
  python -m praxis.ros.scripts.sensor_node_launcher sensor=gsmini
  python -m praxis.ros.scripts.sensor_node_launcher sensor.config.sample_rate=30.0
  python -m praxis.ros.scripts.sensor_node_launcher sensor_node.update_rate=50.0
"""

import sys
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

from praxis.ros.sensor_node import run_sensor_node


@hydra.main(version_base=None, config_path="../config", config_name="sensor_config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    print("=" * 60)
    print("Sensor Node Launcher")
    print("=" * 60)
    print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    
    try:
        # Create sensor from config
        print(f"\nCreating sensor: {cfg.sensor._target_}")
        sensor = instantiate(cfg.sensor)
        
        # Create sensor node configuration
        print(f"Creating sensor node configuration: {cfg.sensor_node._target_}")
        sensor_node_config = instantiate(cfg.sensor_node)
        
        print(f"\nSensor: {sensor.config.name}")
        print(f"Update Rate: {sensor_node_config.update_rate} Hz")
        print(f"Topics:")
        print(f"  - /{sensor.config.name}/{sensor_node_config.sensor_reading_topic}")
        print(f"  - /{sensor.config.name}/{sensor_node_config.status_topic}")
        print("=" * 60)
        
        # Run the SensorNode
        run_sensor_node(sensor, sensor_node_config)
        
    except KeyboardInterrupt:
        print("\nShutting down...")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

